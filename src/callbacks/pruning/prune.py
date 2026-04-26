from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import Callback, LightningModule, Trainer

from src.callbacks.pruning.checkpoint_handler import PrunedCheckpointHandler
from src.callbacks.pruning.parameter_manager import ParameterManager
from src.callbacks.pruning.scheduler import PruningScheduler
from src.callbacks.pruning.shared_prune_utils import (
    ValidationSuppressor,
    compute_sparsity,
)
from src.utils import get_pylogger

logger = get_pylogger(__name__)


class MagnitudePruner(Callback):
    """Orchestrates pruning schedule and execution using a Feedback Loop.

    Logs and manages trackers at the end of the epoch to ensure correct
    ordering.
    """

    def __init__(
        self,
        pruning_fn: str = "l1_unstructured",
        amount: float = 0.5,
        initial_amount: float = 0.0,
        epochs_to_ramp: int = 10,
        scheduled_pruning: bool = False,
        schedule_type: str = "linear",
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        use_global_unstructured: bool = True,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = 1,
        prune_bias: bool = False,
        make_pruning_permanent: bool = True,
        min_param_elements: int = 100,
        verbose: int = 1,
        tolerance: float = 0.01,
        **kwargs,
    ):
        self.pruning_fn_name = pruning_fn
        self.scheduled = scheduled_pruning
        self.make_permanent = make_pruning_permanent
        self.verbose = verbose
        self.global_unstructured = use_global_unstructured
        self.dim = pruning_dim
        self.norm = pruning_norm
        self.manual_params = parameters_to_prune
        self.tolerance = tolerance

        # Scheduler Config
        self.final_amount = amount
        self.initial_amount = initial_amount if scheduled_pruning else amount

        if scheduled_pruning:
            self.epochs_to_ramp = max(1, epochs_to_ramp)
        else:
            assert (
                not epochs_to_ramp
            ), "epochs_to_ramp should be None when scheduled_pruning is False."

        if self.scheduled:
            self.scheduler = PruningScheduler(
                schedule_type=schedule_type,
                final_sparsity=self.final_amount,
                epochs_to_ramp=self.epochs_to_ramp,
                initial_sparsity=self.initial_amount,
                verbose=(verbose > 0),
            )
        else:
            self.scheduler = None

        # Config & Manager
        class Config:
            pass

        self.cfg = Config()
        self.cfg.prune_bias = prune_bias
        self.cfg.min_param_elements = min_param_elements
        self.cfg.pruning_dim = pruning_dim
        self.manager = ParameterManager(self.cfg)

        # State
        self._target_params = []
        self._logged_overview = False
        self._suppressor = ValidationSuppressor()

        # Temp state for logging
        self._current_epoch_target = 0.0
        self._last_status = "Init"

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self._target_params = self.manager.collect_parameters(
            pl_module, self.manual_params
        )
        if self.verbose and not self._logged_overview:
            self.manager.log_overview()
            self._logged_overview = True

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """One-time setup: skip sanity check and start suppressed."""
        if self.scheduled:
            ValidationSuppressor.prepare(trainer)
            trainer.limit_val_batches = 0

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.scheduled and self.scheduler:
            if isinstance(trainer.max_epochs, int) and trainer.max_epochs > 0:
                self.scheduler.verify_schedule_feasibility(trainer.max_epochs)

            if self.verbose:
                logger.info("Pruning Schedule Mapping:")
                for ep in sorted(self.scheduler.schedule_map.keys()):
                    logger.info(
                        f"  Epoch {ep}: {self.scheduler.schedule_map[ep]:.4f}"
                    )

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        # 1. Structural preparation (via Handler)
        state_dict = checkpoint.get("state_dict", {})
        PrunedCheckpointHandler.reconstruct_pruning_structure(
            pl_module, state_dict, verbose=self.verbose
        )

        # 2. Restore Scheduler State
        if "magnitude_pruner_state" in checkpoint:
            state = checkpoint["magnitude_pruner_state"]
            self.scheduled = state.get("scheduled", self.scheduled)

            if self.scheduled:
                if "scheduler_state" not in state:
                    raise RuntimeError(
                        "Pruning Error: Scheduler state missing from checkpoint."
                    )

                s_state = state["scheduler_state"]

                if self.scheduler is None:
                    self.scheduler = PruningScheduler(
                        schedule_type=s_state["schedule_type"],
                        final_sparsity=s_state["final_sparsity"],
                        epochs_to_ramp=s_state["epochs_to_ramp"],
                        initial_sparsity=s_state["initial_sparsity"],
                        verbose=(self.verbose > 0),
                    )

                self.scheduler.load_state_dict(s_state)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        state = {
            "scheduled": self.scheduled,
        }
        if self.scheduled and self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()

        checkpoint["magnitude_pruner_state"] = state

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Executes Pruning logic BEFORE training step."""
        if not self._target_params:
            self._target_params = self.manager.collect_parameters(
                pl_module, self.manual_params
            )

        current_epoch = trainer.current_epoch

        # 1. Determine Target
        if self.scheduled:
            # Always returns a float (never None), holding final value if ramp is done.
            target_amount = self.scheduler.get_target_sparsity(current_epoch)
        else:
            if current_epoch == 0:
                target_amount = self.final_amount
            else:
                target_amount = self.final_amount  # Maintain target

        self._current_epoch_target = target_amount

        # 2. Measure Current State
        current_sparsity = compute_sparsity(self._target_params)

        # 3. Apply Pruning
        # We enforce the target if we are below it OR if we need to clean up Identity artifacts (resumption).
        # We check monotonicity to prevent accidental un-pruning.
        if current_sparsity > (target_amount + self.tolerance):
            raise RuntimeError(
                f"Pruning Error: Current sparsity ({current_sparsity:.4f}) > "
                f"target ({target_amount:.4f}). Cannot un-prune weights."
            )

        # Apply Logic:
        # If target > current -> Must Prune.
        # If target == current -> We SHOULD re-apply to ensure we have a clean L1 mask
        # (getting rid of Identity from checkpoint). This is safe because L1 re-selects the zeros.

        self._remove_pruning_masks()
        self._apply_pruning(target_amount)

        # Verify
        new_sparsity = compute_sparsity(self._target_params)
        self._verify_sparsity_jump(
            current_sparsity, new_sparsity, target_amount
        )

        # Determine status for logging
        if abs(new_sparsity - current_sparsity) > 1e-4:
            self._last_status = "Pruned"
        else:
            self._last_status = "Maintained"

        # 4. Gate validation based on current sparsity vs final target.
        if self.scheduled:
            self._suppressor.gate(trainer, new_sparsity, self.final_amount)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Logs metrics and manages trackers AFTER training step."""
        pruned_sparsity = compute_sparsity(self._target_params)
        overall_sparsity = compute_sparsity(
            list(pl_module.parameters()), threshold=1e-12
        )

        # 1. Log Metrics (Recorder)
        if hasattr(pl_module, "log"):
            # prog_bar=False avoids stale display at start of next epoch
            # "sparsity" = true whole-model sparsity (consistent with Bregman)
            pl_module.log(
                "sparsity",
                overall_sparsity,
                prog_bar=False,
                on_epoch=True,
            )
            # "pruning/sparsity" = pruned params only
            pl_module.log(
                "pruning/sparsity",
                pruned_sparsity,
                prog_bar=False,
                on_epoch=True,
            )

        # 2. Console Monitor
        if self.verbose:
            logger.info(
                f"[Pruning Monitor] Epoch {trainer.current_epoch}: "
                f"Target={self._current_epoch_target:.2%} | "
                f"Result={pruned_sparsity:.2%} | "
                f"Overall={overall_sparsity:.2%} | Status: {self._last_status}"
            )

    def on_train_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.make_permanent:
            if self.verbose:
                logger.info("Finalizing pruning (merging masks)...")
            self._remove_pruning_masks()
            if self.verbose:
                logger.info(
                    f"Permanently pruned {len(self._target_params)} parameters."
                )

    def _remove_pruning_masks(self):
        for module, name in self._target_params:
            if pytorch_prune.is_pruned(module):
                pytorch_prune.remove(module, name)

    def _verify_sparsity_jump(
        self, old_sparsity: float, new_sparsity: float, applied_amount: float
    ):
        if old_sparsity > 0.1 and new_sparsity < old_sparsity - 0.05:
            raise RuntimeError(
                f"Pruning Safety Error: Sparsity dropped from {old_sparsity:.2%} to {new_sparsity:.2%}. "
                f"Target was {applied_amount:.2%}. Check for mask conflicts."
            )

    def _apply_pruning(self, amount: float):
        if amount <= 1e-7:
            return

        kwargs = {"amount": amount}
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if "ln_structured" in self.pruning_fn_name:
            if self.norm is None:
                raise ValueError("pruning_norm required for ln_structured")
            kwargs["n"] = self.norm

        try:
            if (
                self.global_unstructured
                and "unstructured" in self.pruning_fn_name
            ):
                class_name = (
                    self.pruning_fn_name.replace("_", " ")
                    .title()
                    .replace(" ", "")
                )
                method_class = getattr(pytorch_prune, class_name, None)

                if method_class is None:
                    mapping = {
                        "l1_unstructured": pytorch_prune.L1Unstructured,
                        "random_unstructured": pytorch_prune.RandomUnstructured,
                    }
                    method_class = mapping.get(self.pruning_fn_name)

                if method_class is None:
                    raise ValueError(
                        f"Could not resolve Pruning Class for '{self.pruning_fn_name}'."
                    )

                pytorch_prune.global_unstructured(
                    self._target_params, pruning_method=method_class, **kwargs
                )
            else:
                fn = getattr(
                    pytorch_prune,
                    self.pruning_fn_name,
                    pytorch_prune.l1_unstructured,
                )
                for module, name in self._target_params:
                    fn(module, name=name, **kwargs)

        except Exception as e:
            logger.error(f"Pruning application failed: {e}")
            raise e
