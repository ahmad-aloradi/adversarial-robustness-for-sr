"""
BregmanPruner: A callback for orchestrating sparsity in Bregman-based training.

Note: Despite the legacy name "pruner", Bregman learning starts with a sparse
model and allows it to become denser during training. The lambda scheduler
adjusts regularization strength to drive sparsity toward a target level.
"""

from typing import Any, List, Literal, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src import utils
from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from .bregman_regularizers import RegNone
from .lambda_scheduler import LambdaScheduler, TargetScheduler

log = utils.get_pylogger(__name__)

# How to steer lambda: over all model parameters, or only the regularized
# ("pruned") groups. Overall is more intuitive; pruned is more principled
# (the feedback loop sees exactly the params the regularizer acts on).
WHICH_SPARSITY_PERCENTAGE: Literal["overall", "pruned"] = "overall"


class BregmanPruner(Callback):
    """Orchestrates sparsity-related operations during Bregman-based training.

    This callback:
    - Applies initial sparsity to the model (via PruningManager)
    - Optionally updates regularization strength (lambda) per batch via
      LambdaScheduler (fixed-target feedback)
    - Logs sparsity metrics and checkpoints the scheduler state
    """

    def __init__(
        self,
        sparsity_threshold: float = 1e-12,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        target_sparsity: Optional[float] = None,
        target_scheduler: Optional[TargetScheduler] = None,  # ramps the target
    ):
        """
        Args:
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
            target_sparsity: Final sparsity setpoint: the value the feasibility
                check guards and the gates key off. Also the fixed controller
                target when target_scheduler is None. Required when a
                lambda_scheduler is set.
            target_scheduler: Optional per-epoch ramp of the controller target
                (progressive mode). When set, the controller chases its moving
                target each step while the gates still key off target_sparsity.
        """
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        self.target_scheduler = target_scheduler
        if target_sparsity is None:
            self._target_sparsity: Optional[float] = None
        else:
            self._target_sparsity = float(target_sparsity)
            if not 0.0 <= self._target_sparsity <= 1.0:
                raise ValueError(
                    "target_sparsity must be in [0.0, 1.0], got "
                    f"{self._target_sparsity}"
                )

        self.manager: Optional[PruningManager] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._initialized = False
        self._ckpt_scheduler_state: Optional[dict] = None

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Initialize the callback at the start of training."""
        self.manager = pl_module.pruning_manager

        if not trainer.optimizers:
            raise ValueError("BregmanPruner: No optimizers found.")
        if len(trainer.optimizers) > 1:
            raise ValueError("BregmanPruner supports only a single optimizer.")

        # Lightning rebuilds the optimizer on every fit(); refresh the reference
        # even when already initialized so sparsity reads live param tensors.
        optimizer = trainer.optimizers[0]
        self._optimizer = optimizer

        if self._initialized:
            return

        is_resuming = trainer.ckpt_path is not None

        self._verify_gate_target_reachable(pl_module)

        if is_resuming:
            log.info("BregmanPruner: Resuming from checkpoint.")
        else:
            log.info("BregmanPruner: Applying initial sparsity...")
            self.manager.apply_initial_sparsity()

        self._setup_lambda_scheduler(optimizer, trainer, is_resuming)
        self._apply_lambda_to_groups(trainer)
        if is_resuming and self._ckpt_scheduler_state:
            log.info("Restored lambda values to optimizer parameter groups.")

        self._initialized = True
        self._log_configuration(optimizer)
        self._log_group_assignments(pl_module)

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

        # Whole-model sparsity is the dominant per-batch cost; compute once and
        # share it with the scheduler step, the metric logging, and the gate.
        overall_sparsity = self._overall_sparsity()

        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer, overall_sparsity)

        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module, overall_sparsity)

        # Last batch: publish overall sparsity before the validation gate reads it.
        if trainer.is_last_batch:
            trainer.callback_metrics["sparsity"] = torch.tensor(
                overall_sparsity
            )

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log epoch-level sparsity and inject into callback_metrics."""
        if not self._initialized:
            return

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()
        target = self._current_target(trainer)

        # Inject end-of-epoch sparsity directly into callback_metrics so that
        # ModelCheckpoint filenames and train_log.txt get the true final value
        # (not a mean over all steps).
        trainer.callback_metrics["sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/pruned_sparsity"] = torch.tensor(
            pruned_sparsity
        )
        pl_module.log(
            "bregman/pruned_sparsity",
            pruned_sparsity,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )
        if target is not None:
            trainer.callback_metrics["bregman/target_sparsity"] = torch.tensor(
                target
            )

        if self.verbose > 0:
            target_str = (
                f", target = {target:.3%}" if target is not None else ""
            )
            log.info(
                f"Epoch {trainer.current_epoch}: "
                f"Sparsity = {sparsity:.3%} (pruned = {pruned_sparsity:.3%})"
                f"{target_str}"
            )

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Save scheduler state to checkpoint."""
        if self.lambda_scheduler is not None:
            checkpoint["bregman_lambda_scheduler_state"] = (
                self.lambda_scheduler.get_state()
            )

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
        # Progressive mode feeds a moving target to the feedback controller, so
        # it needs one; fail loud rather than ignoring the ramp.
        if self.target_scheduler is not None and self.lambda_scheduler is None:
            raise ValueError(
                "BregmanPruner.target_scheduler requires a lambda_scheduler; "
                "the controller is what tracks the ramped target."
            )
        if self.lambda_scheduler is None:
            return

        # Configs pass the scheduler as a Hydra partial (_partial_: true).
        if not hasattr(self.lambda_scheduler, "step"):
            self.lambda_scheduler = self.lambda_scheduler()

        if self._target_sparsity is None:
            raise ValueError(
                "BregmanPruner.target_sparsity is required when a "
                "lambda_scheduler is set; it is the controller setpoint."
            )

        if self.target_scheduler is not None:
            if not hasattr(self.target_scheduler, "target_at"):
                self.target_scheduler = self.target_scheduler()
            assert (
                self.target_scheduler.final_sparsity == self._target_sparsity
            ), (
                "target_scheduler.final_sparsity "
                f"({self.target_scheduler.final_sparsity}) must equal "
                f"target_sparsity ({self._target_sparsity}); the gates key off "
                "the latter."
            )
            self.target_scheduler.verify_schedule_feasibility(
                trainer.max_epochs
            )

        # Restore state from checkpoint
        if is_resuming and self._ckpt_scheduler_state:
            self.lambda_scheduler.load_state(self._ckpt_scheduler_state)

        log.info(
            f"Lambda scheduler active: target_sparsity={self._target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(
        self, trainer: Trainer, overall_sparsity: float
    ) -> None:
        """Step the scheduler and update regularizer lambdas."""
        current_sparsity = (
            overall_sparsity
            if WHICH_SPARSITY_PERCENTAGE == "overall"
            else self._pruned_sparsity()
        )

        new_lambda = self.lambda_scheduler.step(
            current_sparsity,
            self._current_target(trainer),
            trainer.global_step,
        )
        self._broadcast_lambda(trainer.optimizers[0], new_lambda)

    def _current_target(self, trainer: Trainer) -> Optional[float]:
        """Controller target now: the epoch's ramp value, else the fixed one."""
        if self.target_scheduler is None:
            return self._target_sparsity
        return self.target_scheduler.target_at(trainer.current_epoch)

    def _apply_lambda_to_groups(self, trainer: Trainer) -> None:
        """Apply current scheduler lambda to all regularized groups."""
        if self.lambda_scheduler is None:
            return
        current_lambda = self.lambda_scheduler.get_lambda()
        self._broadcast_lambda(trainer.optimizers[0], current_lambda)

    def _broadcast_lambda(self, optimizer, lambda_value: float) -> None:
        """Set reg.lamda = λ · lambda_scale on every regularized group."""
        for group in optimizer.param_groups:
            if self._group_regularized(group):
                group["reg"].lamda = lambda_value * self._lambda_scale(group)

    # -------------------------------------------------------------------------
    # Sparsity
    # -------------------------------------------------------------------------

    def _overall_sparsity(self) -> float:
        """Sparsity over all model parameters (true whole-model sparsity)."""
        params = list(self.manager.pl_module.parameters())
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _regularized_parameters(self) -> List[torch.Tensor]:
        """Parameters in optimizer groups that carry an active regularizer."""
        assert (
            self._optimizer is not None
        ), "Optimizer must be set to compute pruned sparsity."
        params: List[torch.Tensor] = []
        for group in self._optimizer.param_groups:
            if self._group_has_regularizer(group):
                params.extend(group["params"])
        return params

    def _pruned_sparsity(self) -> float:
        """Sparsity over the regularized (Bregman-pruned) groups only."""
        params = self._regularized_parameters()
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _verify_gate_target_reachable(
        self, pl_module: LightningModule
    ) -> None:
        """Fail loud if the gate target can never reopen validation.

        Validation, checkpointing, and early stopping reopen only once overall
        sparsity reaches the target band. Non-regularized groups (norm/bias)
        stay dense, so overall sparsity is capped at prunable/total; a target
        above that ceiling leaves the band unreachable for the whole run.
        """
        if self._target_sparsity is None:
            return
        total = sum(p.numel() for p in pl_module.parameters())
        prunable = sum(p.numel() for p in self._regularized_parameters())
        ceiling = prunable / total if total else 0.0
        assert self._target_sparsity <= ceiling, (
            f"target_sparsity={self._target_sparsity} exceeds the achievable "
            f"overall-sparsity ceiling={ceiling:.4f} "
            f"({prunable}/{total} params prunable); the validation gate could "
            f"never reopen. Lower target_sparsity or regularize more groups."
        )
        if WHICH_SPARSITY_PERCENTAGE != "overall":
            log.warning(
                f"Lambda steers on '{WHICH_SPARSITY_PERCENTAGE}' sparsity "
                f"while the gates measure overall sparsity; the setpoint "
                f"may never enter the validation band (gate target "
                f"{self._target_sparsity})."
            )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_metrics(
        self, pl_module: LightningModule, overall_sparsity: float
    ) -> None:
        """Log sparsity and lambda metrics via Lightning's logging system."""
        default_logging_params = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "prog_bar": False,
        }
        logging_params = getattr(
            pl_module, "logging_params", default_logging_params
        )
        # Per-step only for TensorBoard/WandB tracking; epoch-level values are
        # injected in on_train_epoch_end (pruned_sparsity there avoids a full
        # scan every batch).
        per_step = {**logging_params, "on_step": True, "on_epoch": False}

        pl_module.log("bregman/sparsity", overall_sparsity, **per_step)

        if self.lambda_scheduler:
            pl_module.log(
                "bregman/global_lambda",
                self.lambda_scheduler.get_lambda(),
                **per_step,
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
                f"Lambda Scheduler: target_sparsity={self._target_sparsity}, "
                f"lambda={self.lambda_scheduler.get_lambda():.4f}, "
                f"update_frequency={self.lambda_scheduler.update_frequency}"
            )
            log.info(sched_info)
        else:
            log.info("Lambda Scheduler: None (static lambda mode)")

        for group in optimizer.param_groups:
            name = group.get("name", "unnamed")
            lamda = group["reg"].lamda
            reg_type = type(group["reg"]).__name__
            if not self._group_has_regularizer(group):
                log.info(f"  Group '{name}': {reg_type} (inactive)")
                continue
            scale = self._lambda_scale(group)
            log.info(
                f"  Group '{name}': {reg_type}, lambda={lamda:.4f}, scale={scale}"
            )

            if scale != 1.0:
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
    def _lambda_scale(group: dict) -> float:
        """lambda_scale of a param group; a missing key is a config bug."""
        if "lambda_scale" not in group:
            raise KeyError(
                f"Group '{group.get('name')}' has no lambda_scale; every "
                "Bregman group must set optimizer_settings.lambda_scale."
            )
        return group["lambda_scale"]

    @staticmethod
    def _group_regularized(group: dict) -> bool:
        """A group carrying a thresholding regularizer (non-RegNone with a
        lamda); its reg.lamda tracks λ_global · lambda_scale."""
        return (
            "reg" in group
            and hasattr(group["reg"], "lamda")
            and not isinstance(group["reg"], RegNone)
        )

    @staticmethod
    def _group_has_regularizer(group: dict) -> bool:
        """An actively pruning group: a thresholding regularizer with
        lambda_scale > 0."""
        return (
            BregmanPruner._group_regularized(group)
            and BregmanPruner._lambda_scale(group) > 0.0
        )
