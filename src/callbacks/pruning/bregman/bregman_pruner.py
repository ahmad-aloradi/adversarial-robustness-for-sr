"""BregmanPruner: orchestrates sparsity during Bregman training.

Despite the legacy name "pruner", Bregman learning starts from a sparse model
and lets it densify; the lambda scheduler steers the regularization strength so
sparsity settles at the target. Reporting lives in :mod:`bregman_report`.
"""

from typing import Any, List, Literal, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from src import utils
from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from .bregman_regularizers import (
    is_regularized,
    lambda_scale,
    thresholds_weights,
)
from .bregman_report import (
    log_configuration,
    log_group_assignments,
    log_step_metrics,
)
from .lambda_scheduler import LambdaScheduler

log = utils.get_pylogger(__name__)

# Which sparsity the controller steers on: over all model parameters, or only
# the regularized ("pruned") groups. Pruned is what the regularizer acts on and
# matches the magnitude pruner's `pruning/sparsity`, so a target means the same
# thing in both stacks. The gates must band the same quantity: "pruned" pairs
# with sparsity_metric: bregman/pruned_sparsity, "overall" with sparsity.
WHICH_SPARSITY_PERCENTAGE: Literal["overall", "pruned"] = "pruned"


class BregmanPruner(Callback):
    """Orchestrates sparsity-related operations during Bregman-based training.

    This callback:
    - Applies initial sparsity to the model (via PruningManager)
    - Optionally steers the regularization strength (lambda) per batch via
      LambdaScheduler
    - Logs sparsity metrics and checkpoints the scheduler state
    """

    def __init__(
        self,
        target_sparsity: float,
        sparsity_threshold: float = 1e-12,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
    ):
        """
        Args:
            target_sparsity: Sparsity setpoint: the value the gates key off and
                the controller drives toward.
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
        """
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
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

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Initialize the callback at the start of training."""
        self.manager = pl_module.pruning_manager

        if not trainer.optimizers:
            raise ValueError("BregmanPruner: No optimizers found.")
        if len(trainer.optimizers) > 1:
            raise ValueError("BregmanPruner supports only a single optimizer.")

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

        self._setup_lambda_scheduler(is_resuming)
        if self.lambda_scheduler is not None:
            self._broadcast_lambda(self.lambda_scheduler.get_lambda())
        if is_resuming and self._ckpt_scheduler_state:
            log.info("Restored lambda values to optimizer parameter groups.")

        self._initialized = True
        if self.verbose > 0:
            log_configuration(
                optimizer,
                self.manager,
                self.lambda_scheduler,
                self._target_sparsity,
                self.sparsity_threshold,
                self._overall_sparsity(),
                self._pruned_sparsity(),
            )
        if self.verbose > 1:
            log_group_assignments(pl_module, self.manager)

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

        # Both scans are the dominant per-batch cost; compute once and share
        # them with the scheduler step, the metric logging, and the gates.
        overall_sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()
        steering_sparsity = (
            overall_sparsity
            if WHICH_SPARSITY_PERCENTAGE == "overall"
            else pruned_sparsity
        )

        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer, steering_sparsity)

        log_step_metrics(
            pl_module, self.lambda_scheduler, overall_sparsity, pruned_sparsity
        )

        # Last batch: publish before the validation gate reads its metric.
        if trainer.is_last_batch:
            self._publish_sparsity(trainer, overall_sparsity, pruned_sparsity)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log epoch-level sparsity and inject into callback_metrics."""
        if not self._initialized:
            return

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()

        self._publish_sparsity(trainer, sparsity, pruned_sparsity)
        trainer.callback_metrics["bregman/target_sparsity"] = torch.tensor(
            self._target_sparsity
        )

        if self.verbose > 0:
            log.info(
                f"Epoch {trainer.current_epoch}: "
                f"Sparsity = {sparsity:.3%} (pruned = {pruned_sparsity:.3%})"
                f", target = {self._target_sparsity:.3%}"
            )

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

    def _setup_lambda_scheduler(self, is_resuming: bool) -> None:
        """Instantiate and configure the lambda scheduler."""
        if self.lambda_scheduler is None:
            return

        # Configs pass the scheduler as a Hydra partial (_partial_: true).
        if not hasattr(self.lambda_scheduler, "step"):
            self.lambda_scheduler = self.lambda_scheduler()

        if is_resuming and self._ckpt_scheduler_state:
            self.lambda_scheduler.load_state(self._ckpt_scheduler_state)

        log.info(
            "Lambda scheduler active: "
            f"target_sparsity={self.lambda_scheduler.target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(
        self, trainer: Trainer, current_sparsity: float
    ) -> None:
        """Step the controller on the steering metric, then broadcast
        lambda."""
        new_lambda = self.lambda_scheduler.step(
            current_sparsity, trainer.global_step
        )
        self._broadcast_lambda(new_lambda)

    def _broadcast_lambda(self, lambda_value: float) -> None:
        """Set reg.lamda = λ · lambda_scale on every thresholding group."""
        for group in self._optimizer.param_groups:
            if thresholds_weights(group):
                group["reg"].lamda = lambda_value * lambda_scale(group)

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
            if is_regularized(group):
                params.extend(group["params"])
        return params

    def _pruned_sparsity(self) -> float:
        """Sparsity over the regularized (Bregman-pruned) groups only."""
        params = self._regularized_parameters()
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    @staticmethod
    def _publish_sparsity(
        trainer: Trainer, overall: float, pruned: float
    ) -> None:
        """Put sparsity where the gates and checkpoint filenames read it.

        These are the true end-of-window values, not a mean over the steps.
        """
        trainer.callback_metrics["sparsity"] = torch.tensor(overall)
        trainer.callback_metrics["bregman/sparsity"] = torch.tensor(overall)
        trainer.callback_metrics["bregman/pruned_sparsity"] = torch.tensor(
            pruned
        )

    def _verify_gate_target_reachable(
        self, pl_module: LightningModule
    ) -> None:
        """Fail loud if an overall-sparsity target can never be reached.

        Norm/bias groups stay dense, so overall sparsity cannot exceed
        prunable/total; a target above that ceiling leaves every sparsity gate
        shut for the whole run. The pruned groups can go all the way to 1.0, so
        they have no such ceiling.
        """
        if WHICH_SPARSITY_PERCENTAGE != "overall":
            return
        total = sum(p.numel() for p in pl_module.parameters())
        prunable = sum(p.numel() for p in self._regularized_parameters())
        ceiling = prunable / total if total else 0.0
        assert self._target_sparsity <= ceiling, (
            f"target_sparsity={self._target_sparsity} exceeds the achievable "
            f"overall-sparsity ceiling={ceiling:.4f} "
            f"({prunable}/{total} params prunable); the gates could never "
            f"reopen. Lower target_sparsity or regularize more groups."
        )
