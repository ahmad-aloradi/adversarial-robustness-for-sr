"""
BregmanCallback: A callback for orchestrating sparsity in Bregman-based training.

Note: Despite the legacy name "pruner", Bregman learning starts with a sparse model
and allows it to become denser during training. The lambda scheduler adjusts
regularization strength to drive sparsity toward a target level.
"""

from typing import Any, List, Optional, Literal, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src import utils
from src.callbacks.pruning.shared_prune_utils import (
    ValidationSuppressor,
    compute_sparsity,
)
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from .lambda_scheduler import LambdaScheduler, _normalize_target_schedule

log = utils.get_pylogger(__name__)

# This determines how to steer lambda: based on all parameters sparsity or just pruned groups.
# Overall is more intuitive, but pruned is more prinicpled (feedback loop)
WHICH_SPARSITY_PERCENTAGE: Literal['overall', 'pruned'] = 'overall'


class BregmanPruner(Callback):
    """Orchestrates sparsity-related operations during Bregman-based training.

    This callback:
    - Applies initial sparsity to the model (via PruningManager)
    - Optionally updates regularization strength (lambda) per batch via LambdaScheduler
    - Logs sparsity metrics during training
    - Handles checkpointing of scheduler state
    """

    def __init__(
        self,
        sparsity_threshold: float = 1e-12,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        target_sparsity: Optional[Union[float, List[float]]] = None,
        tolerance: float = 0.01,
        rescale_mode: str = "none",
        lr_reduction_factor: float = 0.25,
    ):
        """
        Args:
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
            target_sparsity: Target sparsity for validation suppression. A list
                specifies a per-epoch schedule; the last value is held for all
                epochs beyond ``len(list) - 1``.
            rescale_mode: How to handle λ changes in the proximal step.
                "none": no rescaling (default).
                "subgradient_correction": adjust subgradient v to remain in ∂φ_new(θ).
                "nestrovs_adaptive_update": use ∇(λφ)*(v) = (1/λ)·prox_{λψ}(δv).
            lr_reduction_factor: Multiplier applied to optimizer LR when the
                scheduler detects uncontrolled sparsity oscillation. Must be
                in (0.0, 1.0).
        """
        super().__init__()
        if not (0.0 < lr_reduction_factor < 1.0):
            raise ValueError(
                f"lr_reduction_factor must be in (0.0, 1.0), got {lr_reduction_factor}"
            )
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        if target_sparsity is None:
            self._target_schedule: Optional[List[float]] = None
        else:
            self._target_schedule = _normalize_target_schedule(target_sparsity)
        # Public attribute kept for logging/back-compat: scalar when single
        # target, list when a schedule was provided.
        self.target_sparsity = target_sparsity
        self.rescale_mode = rescale_mode
        self.lr_reduction_factor = float(lr_reduction_factor)

        self.manager: Optional[PruningManager] = None
        self._initialized = False
        self._warmup_resolved = False
        self._ckpt_scheduler_state: Optional[dict] = None
        self._ckpt_last_sparsity: Optional[float] = None
        self._suppressor = ValidationSuppressor(tolerance=tolerance)

    def _current_target(self, epoch: int) -> Optional[float]:
        """Return the suppressor target for a given epoch, or None.

        For a per-epoch schedule, returns ``schedule[min(epoch, len-1)]``.
        For the legacy scalar, returns that scalar.
        """
        if self._target_schedule is None:
            return None
        idx = min(max(epoch, 0), len(self._target_schedule) - 1)
        return self._target_schedule[idx]

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Initialize the callback at the start of training."""
        self._validate_module(pl_module)
        self.manager = pl_module.pruning_manager

        if self._initialized:
            return

        if not trainer.optimizers:
            raise ValueError("BregmanPruner: No optimizers found.")
        if len(trainer.optimizers) > 1:
            raise ValueError("BregmanPruner supports only a single optimizer.")

        optimizer = trainer.optimizers[0]
        is_resuming = trainer.ckpt_path is not None

        if is_resuming:
            log.info("BregmanPruner: Resuming from checkpoint.")
        else:
            log.info("BregmanPruner: Applying initial sparsity...")
            self.manager.apply_initial_sparsity()

        self._setup_lambda_scheduler(optimizer, trainer, is_resuming)

        self._apply_lambda_to_groups(trainer)
        if is_resuming and self._ckpt_scheduler_state:
            log.info("Restored lambda values to optimizer parameter groups.")

        if self.lambda_scheduler is not None and self.rescale_mode != "none":
            for group in optimizer.param_groups:
                if "reg" in group:
                    group["reg"].rescale_mode = self.rescale_mode
            log.info(
                f"BregmanPruner: rescale_mode='{self.rescale_mode}' enabled."
            )

        self._initialized = True
        self._log_configuration(optimizer)
        self._log_group_assignments(pl_module)

        # One-time setup: skip sanity check and make val-monitoring callbacks
        # tolerant of epochs where we gate validation off.
        ValidationSuppressor.prepare(trainer)
        trainer.limit_val_batches = 0  # start suppressed; gate() flips it

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Resolve warmup steps and gate validation based on current sparsity.

        Sets ``limit_val_batches`` before Lightning decides whether to run the
        end-of-epoch validation loop. Runs on epoch 0 too — validation begins
        suppressed (see on_fit_start) and only opens once sparsity hits target.
        """
        if not self._initialized:
            return

        if self.lambda_scheduler is not None and not self._warmup_resolved:
            if hasattr(self.lambda_scheduler, "resolve_warmup_steps"):
                self.lambda_scheduler.resolve_warmup_steps(
                    trainer.num_training_batches
                )
            self._warmup_resolved = True

        # Track the per-epoch ramp target rather than the final-epoch value so
        # early-epoch validation isn't wrongly suppressed while sparsity sits
        # far above the ramp's first target.
        active_target = self._current_target(trainer.current_epoch)
        if active_target is not None:
            current_sparsity = self._overall_sparsity() if WHICH_SPARSITY_PERCENTAGE == 'overall' else self._pruned_sparsity()
            self._suppressor.gate(trainer, current_sparsity, active_target)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update lambda scheduler and log metrics after each batch."""
        if not self._initialized:
            return

        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer)

        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module, trainer)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log epoch-level sparsity and inject into callback_metrics."""
        if not self._initialized:
            return

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()

        # Inject end-of-epoch sparsity directly into callback_metrics so that
        # ModelCheckpoint filenames and train_log.txt get the true final value
        # (not a mean over all steps).
        trainer.callback_metrics["sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/pruned_sparsity"] = torch.tensor(
            pruned_sparsity
        )

        if self.verbose > 0:
            log.info(
                f"Epoch {trainer.current_epoch}: "
                f"Sparsity = {sparsity:.3%} (pruned = {pruned_sparsity:.3%})"
            )

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Re-gate right before validation runs, in case sparsity drifted
        between ``on_train_epoch_start`` and the end of the training epoch.
        """
        if not self._initialized:
            return
        active_target = self._current_target(trainer.current_epoch)
        if active_target is None:
            return
        current_sparsity = self._overall_sparsity() if WHICH_SPARSITY_PERCENTAGE == 'overall' else self._pruned_sparsity()
        self._suppressor.gate(trainer, current_sparsity, active_target)

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Save scheduler state to checkpoint."""
        if self.lambda_scheduler is not None:
            checkpoint[
                "lambda_scheduler_state"
            ] = self.lambda_scheduler.get_state()
            checkpoint["bregman_last_sparsity"
                       ] = self._overall_sparsity() if WHICH_SPARSITY_PERCENTAGE == 'overall' else self._pruned_sparsity()

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Load scheduler state from checkpoint."""
        self._ckpt_scheduler_state = checkpoint.get("lambda_scheduler_state")
        self._ckpt_last_sparsity = checkpoint.get("bregman_last_sparsity")

    # -------------------------------------------------------------------------
    # Scheduler management
    # -------------------------------------------------------------------------

    def _setup_lambda_scheduler(
        self, optimizer, trainer: Trainer, is_resuming: bool
    ) -> None:
        """Instantiate and configure the lambda scheduler."""
        if self.lambda_scheduler is None:
            return

        # Handle Hydra partial instantiation
        if not hasattr(self.lambda_scheduler, "step"):
            if not callable(self.lambda_scheduler):
                raise TypeError(
                    f"lambda_scheduler must have a 'step' method or be callable, "
                    f"got {type(self.lambda_scheduler)}"
                )
            self.lambda_scheduler = self.lambda_scheduler()

        # Restore state from checkpoint
        if is_resuming and self._ckpt_scheduler_state:
            self.lambda_scheduler.load_state(self._ckpt_scheduler_state)

        log.info(
            f"Lambda scheduler active: target_sparsity={self.lambda_scheduler.target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(self, trainer: Trainer) -> None:
        """Step the scheduler and update regularizer lambdas.

        w_t+1 = max(w_t + δ(λ_old − λ_new) − δ·lr·grad_step, 0)
        """
        current_sparsity = self._overall_sparsity() if WHICH_SPARSITY_PERCENTAGE == 'overall' else self._pruned_sparsity()

        # On first step after resume, pass the cached sparsity
        last_sparsity = self._ckpt_last_sparsity
        if last_sparsity is not None:
            self._ckpt_last_sparsity = None  # Use only once

        new_lambda = self.lambda_scheduler.step(
            current_sparsity, last_sparsity, trainer.global_step
        )

        # for group in trainer.optimizers[0].param_groups:
        #     if self._group_has_regularizer(group):
        #         scale = group.get("lambda_scale", 1.0)
        #         group["reg"].lamda = new_lambda * scale

        # Sparsity-oscillation detection is meaningless during lambda warmup
        # (lambda is frozen, so any drift isn't the scheduler's doing).
        warmup_steps = getattr(self.lambda_scheduler, "warmup_steps", 0)
        in_warmup = warmup_steps > 0 and trainer.global_step <= warmup_steps
        detected_oscillation = (
            not in_warmup
            and self.lambda_scheduler.detect_uncontrolled_oscillation(
                current_sparsity,
                tolerance=0.01,
                window_steps=1000,
                min_crossings=50,
            )
        )

        for group in trainer.optimizers[0].param_groups:
            if self._group_has_regularizer(group):
                scale = group.get("lambda_scale", 1.0)
                group["reg"].lamda = new_lambda * scale

            if detected_oscillation:
                old_lr = group["lr"]
                group["lr"] = old_lr * self.lr_reduction_factor
                log.warning(
                    f"LR: {old_lr:.2e} -> {group['lr']:.2e} in group "
                    f"{group.get('name', 'Unknown')} due to sparsity oscillation."
                )

    def _apply_lambda_to_groups(self, trainer: Trainer) -> None:
        """Apply current scheduler lambda to all regularized groups."""
        if self.lambda_scheduler is None:
            return
        current_lambda = self.lambda_scheduler.get_lambda()
        for group in trainer.optimizers[0].param_groups:
            if self._group_has_regularizer(group):
                scale = group.get("lambda_scale", 1.0)
                group["reg"].lamda = current_lambda * scale

    # -------------------------------------------------------------------------
    # Sparsity
    # -------------------------------------------------------------------------

    def _overall_sparsity(self) -> float:
        """Sparsity over all model parameters (true whole-model sparsity)."""
        params = list(self.manager.pl_module.parameters())
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _pruned_sparsity(self) -> float:
        """Sparsity over pruned parameter groups only."""
        params = self.manager.get_pruned_parameters()
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_metrics(
        self, pl_module: LightningModule, trainer: Trainer
    ) -> None:
        """Log sparsity and lambda metrics via Lightning's logging system.

        Uses pl_module.logging_params if available for consistent logging
        behavior.
        """
        # Use module's logging_params if available, otherwise use sensible defaults
        default_logging_params = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "prog_bar": False,
        }
        logging_params = getattr(
            pl_module, "logging_params", default_logging_params
        )

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()
        # Log per-step only for TensorBoard/WandB tracking;
        # epoch-level "sparsity" is injected in on_train_epoch_end.
        step_params = {**logging_params, "on_step": True, "on_epoch": False}
        pl_module.log("bregman/sparsity", sparsity, **step_params)
        pl_module.log(
            "bregman/pruned_sparsity", pruned_sparsity, **step_params
        )

        if self.lambda_scheduler:
            # Lambda changes per step, so always log on_step; override on_epoch to avoid noise
            lambda_params = {
                **logging_params,
                "on_epoch": False,
                "on_step": True,
            }
            pl_module.log(
                "bregman/global_lambda",
                self.lambda_scheduler.get_lambda(),
                **lambda_params,
            )

    @rank_zero_only
    def _log_configuration(self, optimizer) -> None:
        """Log the configuration of all parameter groups."""
        if self.verbose == 0:
            return

        log.info("=== Bregman Configuration ===")

        log.info(f"Optimizer: {type(optimizer).__name__}")

        if self.lambda_scheduler:
            sched_info = (
                f"Lambda Scheduler: target_sparsity={self.lambda_scheduler.target_sparsity}, "
                f"lambda={self.lambda_scheduler.get_lambda():.4f}, "
                f"update_frequency={self.lambda_scheduler.update_frequency}"
            )
            log.info(sched_info)
        else:
            log.info("Lambda Scheduler: None (static lambda mode)")

        for group in optimizer.param_groups:
            name = group.get("name", "unnamed")
            scale = group.get("lambda_scale", 1.0)
            lamda = group["reg"].lamda
            reg_type = type(group["reg"]).__name__
            log.info(
                f"  Group '{name}': {reg_type}, lambda={lamda:.4f}, scale={scale}"
            )

            # Safety check for non-uniform scaling
            if self._group_has_regularizer(group) and scale != 1.0:
                log.warning(
                    f"Group '{name}' has lambda_scale={scale} != 1.0. "
                    "Non-uniform regularization is generally not recommended."
                )

        log.info("Current sparsity by group:")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            sparsity = compute_sparsity(
                group["params"], threshold=self.sparsity_threshold
            )
            str_extras = (
                "(not pruned)" if group["applier"].sparsity_rate == 0.0 else ""
            )
            log.info(f"  {name}: {sparsity:.3%} {str_extras}")

        log.info(
            f"Overall sparsity: {self._overall_sparsity():.3%} "
            f"(pruned only: {self._pruned_sparsity():.3%})"
        )
        log.info("=== End Configuration ===")

    @rank_zero_only
    def _log_group_assignments(self, pl_module: LightningModule) -> None:
        """Log detailed group assignments (for debugging)."""
        if self.verbose < 2:
            return

        param_to_module = {
            id(p): ".".join(name.split(".")[:-1])
            for name, p in pl_module.named_parameters()
        }

        total_params = sum(
            p.numel()
            for group in self.manager.processed_groups
            for p in group["params"]
        )

        log.info("--- Parameter Group Assignments ---")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            # is_fallback = group["config"].get("is_fallback", False)
            modules = {param_to_module.get(id(p)) for p in group["params"]}
            modules.discard(None)
            group_params = sum(p.numel() for p in group["params"])
            pct = group_params / total_params * 100 if total_params else 0
            log.info(40 * "-")
            log.info(
                f"  {name}: {len(modules)} modules, "
                f"{group_params:,} params ({pct:.1f}%)"
            )
            if modules:
                for m in sorted(modules):
                    log.info(f"    {m}")
        log.info(40 * "-")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _group_has_regularizer(group: dict) -> bool:
        """Check if a param group has an active regularizer."""
        return (
            "reg" in group
            and hasattr(group["reg"], "lamda")
            and group.get("lambda_scale", 0.0) > 0.0
        )

    @staticmethod
    def _validate_module(pl_module: LightningModule) -> None:
        """Validate that the module has a pruning_manager."""
        if not hasattr(pl_module, "pruning_manager"):
            raise AttributeError(
                "LightningModule must have a 'pruning_manager' attribute. "
                "Please instantiate it in configure_optimizers()."
            )
        if not isinstance(pl_module.pruning_manager, PruningManager):
            raise TypeError(
                f"pruning_manager must be a PruningManager, got {type(pl_module.pruning_manager)}"
            )
