"""
BregmanCallback: A callback for orchestrating sparsity in Bregman-based training.

Note: Despite the legacy name "pruner", Bregman learning starts with a sparse model
and allows it to become denser during training. The lambda scheduler adjusts
regularization strength to drive sparsity toward a target level.
"""

from typing import Any, List, Literal, Optional, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src import utils
from src.callbacks.pruning.shared_prune_utils import (
    ValidationSuppressor,
    compute_sparsity,
)
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from .bregman_regularizers import RegL1L2, RegL1L2Conv
from .lambda_scheduler import LambdaScheduler
from .wp_scheduler import WpAnnealer

log = utils.get_pylogger(__name__)

# rescale_mode values that correct for λ *changes* between steps; these are
# meaningless without a scheduler actively moving λ. "weight_masking" is a
# primal-readout change that is independent of λ, so it is excluded here and
# activates with or without a lambda_scheduler.
_LAMBDA_CHANGE_MODES = frozenset(
    {
        "subgradient_correction",
        "nestrovs_adaptive_update",
    }
)

# How to steer lambda: over all model parameters, or only the regularized
# ("pruned") groups. Overall is more intuitive; pruned is more principled
# (the feedback loop sees exactly the params the regularizer acts on).
WHICH_SPARSITY_PERCENTAGE: Literal["overall", "pruned"] = "overall"


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
        wp_annealer: Optional[WpAnnealer] = None,
        wp_mode: str = "blend",
    ):
        """
        Args:
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
            target_sparsity: Final target sparsity for validation suppression.
                Validation stays suppressed until the model reaches it. A list
                collapses to its last entry.
            rescale_mode: How to handle the proximal step.
                "none": no rescaling (default).
                "subgradient_correction": adjust subgradient v to remain in ∂φ_new(θ).
                "nestrovs_adaptive_update": use ∇(λφ)*(v) = (1/λ)·prox_{λψ}(δv).
                "weight_masking": geometric-mean readout that gates the
                    candidate δv toward the current weight magnitude via w_p
                    (see WpAnnealer); independent of the lambda_scheduler.
            wp_annealer: Optional explore->commit annealer for w_p, used only
                when rescale_mode == "weight_masking".
            wp_mode: How w_p drives the weight_masking gate. "blend" is the
                deterministic geometric blend; "probabilistic" is a per-element
                Bernoulli(w_p) gate, so w_p is the fraction of the support still
                taking its exact reversible step. See weight_masked_prox_arg.
        """
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        # Validation gates on the FINAL target only (suppressed until the model
        # reaches it). A list collapses to its last entry for back-compat.
        if target_sparsity is None:
            self._target_final: Optional[float] = None
        elif isinstance(target_sparsity, (list, tuple)):
            if len(target_sparsity) == 0:
                raise ValueError("target_sparsity list must not be empty")
            self._target_final = float(target_sparsity[-1])
        else:
            self._target_final = float(target_sparsity)
        self.rescale_mode = rescale_mode
        self.wp_annealer = wp_annealer
        if wp_mode not in ("blend", "probabilistic"):
            raise ValueError(
                f"wp_mode must be 'blend' or 'probabilistic', got {wp_mode!r}"
            )
        self.wp_mode = wp_mode

        self.manager: Optional[PruningManager] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._initialized = False
        self._warmup_resolved = False
        self._ckpt_scheduler_state: Optional[dict] = None
        self._suppressor = ValidationSuppressor(tolerance=tolerance)
        # Resolved lazily from the trainer; w_p anneals over the full horizon.
        self._total_steps: Optional[int] = None
        self._last_wp: float = 1.0

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
        self._optimizer = optimizer
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

        needs_scheduler = self.rescale_mode in _LAMBDA_CHANGE_MODES
        if self.rescale_mode != "none" and (
            not needs_scheduler or self.lambda_scheduler is not None
        ):
            group_structured: List[str] = []
            for group in optimizer.param_groups:
                if self._group_has_regularizer(group):
                    group["reg"].rescale_mode = self.rescale_mode
                    if self.rescale_mode == "weight_masking":
                        group["reg"].wp_mode = self.wp_mode
                        # weight_masking gates per element; a group-lasso prox thresholds
                        # by row norm, so an element-wise latch can zero entries inside
                        # an otherwise-live row and break the group structure.
                        if isinstance(group["reg"], (RegL1L2Conv, RegL1L2)):
                            group_structured.append(group["name"])
            log.info(
                f"BregmanPruner: rescale_mode='{self.rescale_mode}' enabled."
            )
            if group_structured:
                log.warning(
                    f"BregmanPruner: weight_masking is a per-element readout "
                    f"but group(s) {group_structured} use a group-structured regularizer "
                    "(RegL1L2Conv) whose prox thresholds by row norm. "
                    "The element-wise latch can zero entries inside a live "
                    "row, breaking the group structure. Use an element-wise "
                    "regularizer (e.g. RegL1) for these groups, or a "
                    "non-masking rescale_mode.")
        elif needs_scheduler and self.lambda_scheduler is None:
            log.warning(
                f"BregmanPruner: rescale_mode='{self.rescale_mode}' ignored "
                "because it corrects for lambda changes but no "
                "lambda_scheduler is configured."
            )

        # w_p anneal is independent of the lambda scheduler.
        if self._wp_anneal_active():
            if not hasattr(self.wp_annealer, "value_at"):
                self.wp_annealer = self.wp_annealer()  # Hydra partial
            self._apply_wp_to_groups(
                optimizer, self.wp_annealer.value_at(0.0)
            )
            log.info(
                f"BregmanPruner: w_p anneal enabled "
                f"({self.wp_annealer.w_p_init} -> {self.wp_annealer.w_p_final}, "
                f"{self.wp_annealer.schedule}, mode={self.wp_mode})."
            )
        elif self.wp_annealer is not None:
            log.warning(
                "BregmanPruner: wp_annealer provided but rescale_mode != "
                "'weight_masking'; w_p has no effect and is ignored."
            )

        self._initialized = True
        self._log_configuration(optimizer)
        self._log_group_assignments(pl_module)

        # Fixed-lambda experiments run validation unconditionally.
        if (
            self.lambda_scheduler is not None
            and self._target_final is not None
        ):
            ValidationSuppressor.prepare(trainer)
            # Start suppressed; _gate_validation reopens it at each epoch end.
            trainer.limit_val_batches = 0
            log.info(
                "Validation suppression ENABLED for adaptive lambda scheduling."
            )
        else:
            log.info("Validation suppression DISABLED (no target provided).")

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Fail loud if a ramp can't finish; enforce min_epochs; log
        schedule."""
        if self.lambda_scheduler is None:
            return
        max_epochs = trainer.max_epochs
        if not (isinstance(max_epochs, int) and max_epochs > 0):
            return
        self.lambda_scheduler.verify_ramp_feasibility(max_epochs)

        # Fixed-target runs have nothing to ramp; the schedule is the constant.
        if self.lambda_scheduler._target_initial is None:
            return

        # Ensure min_epochs leaves room for the ramp to actually reach progress
        # 1.0 (one epoch past warmup+epochs_to_ramp, see verify_ramp_feasibility).
        ramp_end = (
            self.lambda_scheduler.warmup_epochs
            + self.lambda_scheduler._epochs_to_ramp
        )
        min_needed = ramp_end + 1
        current_min = trainer.fit_loop.min_epochs
        if current_min is None or current_min < min_needed:
            trainer.fit_loop.min_epochs = min_needed
            log.info(
                f"BregmanPruner: trainer.min_epochs {current_min} -> "
                f"{min_needed} to ensure the sparsity ramp completes."
            )

        n_batches = trainer.num_training_batches
        if isinstance(n_batches, int) and n_batches > 0:
            targets = self.lambda_scheduler.get_epoch_targets(
                max_epochs, n_batches
            )
            lines = [
                f"  epoch {epoch:3d}: {target:.4f}"
                for epoch, target in targets
            ]
            log.info(
                f"Bregman sparsity target schedule ({max_epochs} epochs):\n"
                + "\n".join(lines)
            )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Resolve warmup steps and re-seed the annealed w_p (resume-safe)."""
        if not self._initialized:
            return

        if self.lambda_scheduler is not None and not self._warmup_resolved:
            if hasattr(self.lambda_scheduler, "resolve_warmup_steps"):
                self.lambda_scheduler.resolve_warmup_steps(
                    trainer.num_training_batches
                )
            self._warmup_resolved = True

         # start with the right w_p value
        if self._wp_anneal_active():
            self._step_wp_annealer(trainer)

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

        if self._wp_anneal_active():
            self._step_wp_annealer(trainer)

        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module, trainer)

        # On the last batch, measured sparsity is the end-of-epoch value
        # Lightning is about to validate; this is the latest hook before it
        # reads limit_val_batches (via Trainer.enable_validation).
        if trainer.is_last_batch:
            self._gate_validation(trainer)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log epoch-level sparsity and inject into callback_metrics."""
        if not self._initialized:
            return

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()
        target = (
            self.lambda_scheduler.target_sparsity
            if self.lambda_scheduler is not None
            else self._target_final
        )

        # Inject end-of-epoch sparsity directly into callback_metrics so that
        # ModelCheckpoint filenames and train_log.txt get the true final value
        # (not a mean over all steps).
        trainer.callback_metrics["sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/pruned_sparsity"] = torch.tensor(
            pruned_sparsity
        )
        if target is not None:
            trainer.callback_metrics[
                "bregman/target_sparsity"
            ] = torch.tensor(target)

        if self.verbose > 0:
            target_str = (
                f", target = {target:.3%}" if target is not None else ""
            )
            log.info(
                f"Epoch {trainer.current_epoch}: "
                f"Sparsity = {sparsity:.3%} (pruned = {pruned_sparsity:.3%})"
                f"{target_str}"
            )

    def _gate_validation(self, trainer: Trainer) -> None:
        """Open or suppress the upcoming validation epoch.

        Gates on the FINAL target with the current (end-of-epoch) sparsity, so
        validation stays suppressed across the whole ramp and only opens once
        the model sits within tolerance of the final target — matching the
        gradual magnitude pruner's contract.
        """
        if self.lambda_scheduler is None or self._target_final is None:
            return
        current_sparsity = (
            self._overall_sparsity()
            if WHICH_SPARSITY_PERCENTAGE == "overall"
            else self._pruned_sparsity()
        )
        self._suppressor.gate(trainer, current_sparsity, self._target_final)

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Save scheduler state to checkpoint."""
        if self.lambda_scheduler is not None:
            checkpoint[
                "bregman_lambda_scheduler_state"
            ] = self.lambda_scheduler.get_state()

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Load scheduler state from checkpoint."""
        self._ckpt_scheduler_state = checkpoint.get(
            "bregman_lambda_scheduler_state",
            checkpoint.get("lambda_scheduler_state"),  # pre-rename compat
        )

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
            # _steps_per_epoch is restored with the state; skip re-resolution in
            # on_train_epoch_start so ramp_steps stays stable across the resume.
            self._warmup_resolved = True

        log.info(
            f"Lambda scheduler active: target_sparsity={self.lambda_scheduler.target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(self, trainer: Trainer) -> None:
        """Step the scheduler and update regularizer lambdas.

        w_t+1 = max(w_t + δ(λ_old − λ_new) − δ·lr·grad_step, 0)
        """
        current_sparsity = (
            self._overall_sparsity()
            if WHICH_SPARSITY_PERCENTAGE == "overall"
            else self._pruned_sparsity()
        )

        new_lambda = self.lambda_scheduler.step(
            current_sparsity, trainer.global_step
        )

        for group in trainer.optimizers[0].param_groups:
            if self._group_has_regularizer(group):
                scale = group.get("lambda_scale", 1.0)
                group["reg"].lamda = new_lambda * scale

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
    # Weight-activation probability (w_p) anneal
    # -------------------------------------------------------------------------

    def _wp_anneal_active(self) -> bool:
        """w_p is annealed only for the weight_masking readout; no scheduler
        dependency."""
        return (
            self.wp_annealer is not None
            and self.rescale_mode == "weight_masking"
        )

    def _step_wp_annealer(self, trainer: Trainer) -> None:
        """Push the annealed w_p onto every regularized group once per batch.

        Progress is global_step / total optimizer steps, so the anneal spans
        the whole run and survives resume (global_step is restored by
        Lightning; the horizon is recomputed).
        """
        if self._total_steps is None:
            total = trainer.estimated_stepping_batches
            assert isinstance(total, int) and total > 0, (
                "w_p anneal needs a finite training horizon; "
                f"trainer.estimated_stepping_batches={total}"
            )
            self._total_steps = total
        progress = min(1.0, trainer.global_step / self._total_steps)
        self._last_wp = self.wp_annealer.value_at(progress)
        self._apply_wp_to_groups(trainer.optimizers[0], self._last_wp)

    def _apply_wp_to_groups(self, optimizer, w_p: float) -> None:
        """Set w_p on every group whose regularizer is active."""
        for group in optimizer.param_groups:
            if self._group_has_regularizer(group):
                group["reg"].w_p = w_p

    # -------------------------------------------------------------------------
    # Sparsity
    # -------------------------------------------------------------------------

    def _overall_sparsity(self) -> float:
        """Sparsity over all model parameters (true whole-model sparsity)."""
        params = list(self.manager.pl_module.parameters())
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _regularized_parameters(self) -> List[torch.Tensor]:
        """Parameters in optimizer groups that carry an active regularizer."""
        assert self._optimizer is not None, "Optimizer must be set to compute pruned sparsity."
        params: List[torch.Tensor] = []
        for group in self._optimizer.param_groups:
            if self._group_has_regularizer(group):
                params.extend(group["params"])
        return params

    def _pruned_sparsity(self) -> float:
        """Sparsity over the regularized (Bregman-pruned) groups only."""
        params = self._regularized_parameters()
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

        if self._wp_anneal_active():
            wp_params = {**logging_params, "on_epoch": False, "on_step": True}
            pl_module.log("bregman/w_p", float(self._last_wp), **wp_params)

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
