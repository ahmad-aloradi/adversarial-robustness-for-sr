import copy
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.pruning import ModelPruning

from src.utils import get_pylogger
from src.callbacks.pruning.scheduler import PruningScheduler
from src.callbacks.pruning.parameter_manager import ParameterManager, ParameterSnapshotter

logger = get_pylogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    pruning_fn: Union[str, Callable] = "l1_unstructured"
    amount: Union[int, float] = 0.5
    use_global_unstructured: bool = True
    apply_pruning: bool = True
    make_pruning_permanent: bool = True
    use_lottery_ticket_hypothesis: bool = False
    resample_parameters: bool = False
    parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None
    pruning_dim: Optional[int] = None
    pruning_norm: Optional[int] = None
    verbose: int = 0
    pruning_trigger: str = "epoch_start"
    scheduled_pruning: bool = False
    final_amount: Optional[float] = None
    epochs_to_ramp: int = 10
    save_when_sparser_than: Optional[float] = None
    prune_bias: bool = False
    min_param_elements: int = 100
    schedule_type: str = "constant"


@dataclass
class CheckpointState:
    """Manages checkpoint-related state."""
    callbacks: List[ModelCheckpoint] = field(default_factory=list)
    original_settings: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    metric_snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    disabled_for_validation: bool = False
    is_active: Optional[bool] = None
    warmup_announced: bool = False


@dataclass
class EarlyStoppingState:
    """Manages early stopping state."""
    callbacks: List[EarlyStopping] = field(default_factory=list)
    original_settings: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    is_disabled: bool = False
    best_reset: bool = False


@dataclass
class PruningState:
    """Manages pruning state."""
    completed: bool = False
    latest_sparsity: float = 0.0
    parameter_summary_logged: bool = False
    is_resuming: bool = False
    loaded_from_checkpoint: bool = False
    saved_state: Optional[Dict[str, Any]] = None
    current_epoch: int = 0


class MagnitudePruner(ModelPruning):
    """ This callback extends PyTorch Lightning's ModelPruning with additional features:
    - Scheduled pruning with gradual sparsity increase
    - Checkpoint management based on sparsity thresholds
    - State persistence for training resumption
    - Comprehensive parameter logging and validation
    - Support for various pruning strategies (L1, structured, etc.)
    
    Parameters
    ----------
    pruning_fn : Union[str, Callable], default="l1_unstructured"
        Pruning function name from torch.nn.utils.prune or custom callable
    amount : Union[int, float], default=0.5
        Fraction (0-1) or absolute number of parameters to prune
    use_global_unstructured : bool, default=True
        If True, prunes globally across all parameters; if False, per-parameter
    apply_pruning : bool, default=True
        Whether to actually apply pruning (can be callable for dynamic control)
    make_pruning_permanent : bool, default=True
        If True, removes pruning masks at end of training
    use_lottery_ticket_hypothesis : bool, default=False
        If True, resets remaining weights to initial values after pruning
    resample_parameters : bool, default=False
        If True, resamples pruned weights (used with lottery ticket hypothesis)
    parameters_to_prune : Optional[List[Tuple[nn.Module, str]]], default=None
        Specific (module, param_name) tuples to prune; if None, auto-collects
    pruning_dim : Optional[int], default=None
        Dimension for structured pruning (required for structured methods)
    pruning_norm : Optional[int], default=None
        Norm order for structured pruning
    verbose : int, default=0
        Verbosity level: 0=silent, 1=basic info, 2=detailed debug
    pruning_trigger : str, default="epoch_start"
        When to apply pruning: "pre_training", "epoch_start", or "epoch_end"
    scheduled_pruning : bool, default=False
        If True, gradually increases sparsity from 0 to final_amount
    final_amount : Optional[float], default=None
        Target sparsity for scheduled pruning (uses amount if None)
    epochs_to_ramp : int, default=10
        Number of epochs to reach final_amount in scheduled pruning
    schedule_type : str, default="linear"
        Controls how sparsity increases during scheduled pruning. "linear" matches
        the previous behavior (linearly increasing target sparsity). "constant"
        keeps the per-epoch prune ratio (percentage of remaining weights) nearly
        constant so that logs report a stable pruning percentage each epoch.
    save_when_sparser_than : Optional[float], default=None
        Only save checkpoints when sparsity exceeds this threshold
    prune_bias : bool, default=False
        Whether to include bias parameters in pruning
    min_param_elements : int, default=100
        Minimum number of elements for a parameter to be prunable
    **kwargs
        Additional arguments passed to parent ModelPruning
    """

    VALID_PRUNING_TRIGGERS = ("pre_training", "epoch_start", "epoch_end")
    SCHEDULE_TYPES = ("linear", "constant")

    def __init__(
        self,
        pruning_fn: Union[str, Callable] = "l1_unstructured",
        amount: Union[int, float] = 0.5,
        use_global_unstructured: bool = True,
        apply_pruning: bool = True,
        make_pruning_permanent: bool = True,
        use_lottery_ticket_hypothesis: bool = False,
        resample_parameters: bool = False,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
        verbose: int = 0,
        pruning_trigger: str = "epoch_start",
        scheduled_pruning: bool = False,
        final_amount: Optional[float] = None,
        epochs_to_ramp: int = 10,
        save_when_sparser_than: Optional[float] = None,
        prune_bias: bool = False,
        min_param_elements: int = 10,
        schedule_type: str = "constant",
        **kwargs
    ):
        # Store configuration first
        self.config = PruningConfig(
            pruning_fn=pruning_fn,
            amount=amount,
            use_global_unstructured=use_global_unstructured,
            apply_pruning=apply_pruning,
            make_pruning_permanent=make_pruning_permanent,
            use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
            resample_parameters=resample_parameters,
            parameters_to_prune=parameters_to_prune,
            pruning_dim=pruning_dim,
            pruning_norm=pruning_norm,
            verbose=verbose,
            pruning_trigger=pruning_trigger,
            scheduled_pruning=scheduled_pruning,
            final_amount=final_amount,
            epochs_to_ramp=epochs_to_ramp,
            save_when_sparser_than=save_when_sparser_than,
            prune_bias=prune_bias,
            min_param_elements=min_param_elements,
            schedule_type=schedule_type,
        )
        
        # Calculate effective amount for parent class
        if scheduled_pruning:
            self.config.final_amount = final_amount if final_amount is not None else amount
            self.config.epochs_to_ramp = max(1, epochs_to_ramp)
            effective_amount = 0.0  # Start with no pruning
        else:
            effective_amount = amount

        # Initialize parent classes through MRO
        super().__init__(
            pruning_fn=pruning_fn,
            amount=effective_amount,
            use_global_unstructured=use_global_unstructured,
            apply_pruning=apply_pruning,
            make_pruning_permanent=make_pruning_permanent,
            use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
            resample_parameters=resample_parameters,
            parameters_to_prune=parameters_to_prune,
            pruning_dim=pruning_dim,
            pruning_norm=pruning_norm,
            verbose=verbose,
            prune_on_train_epoch_end=False,
            **kwargs
        )
        
        # Initialize state management
        self.checkpoint_state = CheckpointState()
        self.early_stopping_state = EarlyStoppingState()
        self.pruning_state = PruningState()
        self.parameter_manager = ParameterManager(self.config)

        self.scheduler: Optional[PruningScheduler] = None
        if self.config.scheduled_pruning:
            self.scheduler = PruningScheduler(
                self.config.schedule_type,
                self.config.final_amount,
                self.config.epochs_to_ramp
            )
        
        # Additional initialization
        self._parameters_to_prune = parameters_to_prune
        self._checkpoint_metric_snapshots: Dict[int, Dict[str, Any]] = {}
        self._best_score_has_been_reset = False
        self._early_stopping_best_reset = False
        self._checkpointing_active: Optional[bool] = None
        self._warmup_announced = False
        
        # Validate pruning trigger
        if pruning_trigger not in self.VALID_PRUNING_TRIGGERS:
            raise ValueError(f"Unexpected pruning_trigger '{pruning_trigger}'")

        if schedule_type not in self.SCHEDULE_TYPES:
            raise ValueError(f"schedule_type must be one of {self.SCHEDULE_TYPES}, got '{schedule_type}'")
        if not self.config.scheduled_pruning and schedule_type != "linear":
            raise ValueError("schedule_type='constant' requires scheduled_pruning=True")
        
        # Validate structured pruning requirements
        if "ln_structured" in str(pruning_fn) and pruning_dim is None:
            raise ValueError("`pruning_dim` must be specified for structured pruning.")

        # Handle scheduled pruning validation
        if pruning_trigger == "pre_training" and scheduled_pruning:
            raise ValueError("Scheduled pruning requires epoch-based triggers, not pre_training.")

        # Fix the _parameter_names issue - this is required by the base class
        if parameters_to_prune is not None:
            self._parameter_names = [param_name for _, param_name in parameters_to_prune]
        else:
            self._parameter_names = []

    def _log(self, message: str, level: int = 1) -> None:
        """Log messages with appropriate log level."""
        if self.config.verbose >= level:
            if level >= 2:
                logger.debug(message)
            else:
                logger.info(message)

    def _get_sparsity_target(self, current_epoch: int) -> float:
        """Calculate target sparsity for current epoch."""
        if not self.config.scheduled_pruning:
            return self.config.amount
        
        if self.scheduler is None:
            # Fallback if scheduler wasn't initialized (shouldn't happen if scheduled_pruning is True)
            return 0.0

        return self.scheduler.get_target_sparsity(current_epoch)
 
    def _warn_if_schedule_cannot_reach_target(self, trainer: Trainer) -> None:
        """Emit warning when requested sparsity is unreachable."""
        if not self.config.scheduled_pruning or self.scheduler is None:
            return

        max_epochs = getattr(trainer, "max_epochs", None)
        if not isinstance(max_epochs, (int, float)) or max_epochs < 1:
            # Cannot forecast without valid max_epochs
            return

        if not self.scheduler.is_target_reachable(int(max_epochs)):
            forecast = self.scheduler.get_target_sparsity(int(max_epochs) - 1)
            logger.warning(
                "Scheduled pruning target %.2f%% is unreachable with max_epochs=%s and epochs_to_ramp=%s; "
                "training would top out at %.2f%% sparsity.",
                self.config.final_amount * 100,
                max_epochs,
                self.config.epochs_to_ramp,
                forecast * 100,
            )

    def _determine_pruning_amount(self, current_epoch: int, current_sparsity: float) -> Optional[Tuple[float, float]]:
        """Calculate pruning amount needed to reach target sparsity."""
        target_sparsity = self._get_sparsity_target(current_epoch)
        
        # Early return if target reached
        if self._is_target_reached(current_sparsity, target_sparsity):
            return None
        
        # Calculate pruning amount
        remaining_weights = 1.0 - current_sparsity
        if remaining_weights <= 0:
            raise RuntimeError(
                f"No remaining weights to prune (current sparsity: {current_sparsity:.2%})"
            )
        
        prune_amt = (target_sparsity - current_sparsity) / remaining_weights
        prune_amt = min(max(0.0, prune_amt), 1.0)
        
        return prune_amt, target_sparsity

    def _is_target_reached(self, current_sparsity: float, target_sparsity: float) -> bool:
        """Check if target sparsity has been reached within tolerance."""
        tolerance = 1e-2
        return current_sparsity >= target_sparsity - tolerance

    def _maybe_restore_parameters_from_state(self, pl_module: LightningModule) -> None:
        """Restore parameters_to_prune from checkpoint metadata when available."""
        if self._parameters_to_prune is not None:
            return

        state = self.pruning_state.saved_state
        if not state:
            return

        if 'parameters_to_prune_info' not in state:
            raise KeyError("Checkpoint state missing 'parameters_to_prune_info'")

        self._log("Restoring parameters_to_prune from checkpoint...")
        restored_params = ParameterSnapshotter.restore(
            state['parameters_to_prune_info'],
            pl_module,
        )
        self._parameters_to_prune = restored_params
        self.parameter_manager.prunable_parameters = restored_params
        self._log(f"Restored {len(restored_params)} parameters for pruning")
        
        # Also restore the _parameter_names for the base class
        self._parameter_names = [param_name for _, param_name in restored_params]

    def _collect_parameters_if_needed(self, pl_module: LightningModule) -> None:
        """
        Collects and categorizes all parameters in the model into prunable and non-prunable sets.
        Ensures mutual exclusivity and completeness (Union == Total Parameters).
        """
        # If manual parameters were provided, still collect to allow logging and
        # validation; ParameterManager will filter out non-prunable or invalid
        # selections and produce the final prunable list.
        if self._parameters_to_prune is not None:
            self.parameter_manager.collect_parameters(pl_module, self._parameters_to_prune)
            self._parameters_to_prune = self.parameter_manager.prunable_parameters
            self._parameter_names = [n for m, n in self._parameters_to_prune]
            return

        self.parameter_manager.collect_parameters(pl_module, None)
        
        # Sync back to MagnitudePruner (auto-collection case)
        self._parameters_to_prune = self.parameter_manager.prunable_parameters
        self._parameter_names = [n for m, n in self._parameters_to_prune]

    def _log_parameter_overview(self) -> None:
        """Log overview of prunable and non-prunable parameters."""
        if self.pruning_state.parameter_summary_logged:
            return

        self.parameter_manager.log_overview()
        self.pruning_state.parameter_summary_logged = True

    def _run_pruning(self, current_epoch: int, pl_module: LightningModule = None) -> bool:
        """Execute pruning with improved error handling and logging."""
        if self.pruning_state.completed:
            if current_epoch <= 0:
                self._log(f"[Epoch {current_epoch}] Pruning already completed, skipping.")
            return False
        
        # Normalize epoch for display
        display_epoch = current_epoch if current_epoch >= 0 else 0
        
        # Check if pruning is enabled
        should_prune = self.config.apply_pruning
        if callable(should_prune):
            should_prune = should_prune(current_epoch)
        
        if not should_prune:
            self._log(f"Epoch {display_epoch}: skip (prune={should_prune})", level=2)
            return False

        # Get valid parameters
        try:
            valid_params = self.parameter_manager.get_valid_parameters()
        except RuntimeError as e:
            logger.error(f"Failed to get valid parameters: {e}")
            return False

        # Compute current sparsity
        current_sparsity = self.parameter_manager.compute_sparsity(valid_params)
        self.pruning_state.latest_sparsity = current_sparsity

        # Determine pruning amount
        schedule_epoch = -1 if (current_epoch <= 0 and self.config.scheduled_pruning) else current_epoch
        plan = self._determine_pruning_amount(schedule_epoch, current_sparsity)
        
        if plan is None:
            return False

        prune_amt, target_sparsity = plan

        # Apply pruning
        try:
            self.apply_pruning(prune_amt, pl_module, prevalidated_params=valid_params)
        except Exception as e:
            logger.error(f"Failed to apply pruning: {e}")
            return False

        # Handle lottery ticket hypothesis
        use_lth = self.config.use_lottery_ticket_hypothesis
        if callable(use_lth):
            use_lth = use_lth(current_epoch)
        
        if use_lth:
            self.apply_lottery_ticket_hypothesis()

        # Update sparsity after pruning
        self.pruning_state.latest_sparsity = self.parameter_manager.compute_sparsity(valid_params)

        # Log results
        label = "[Scheduled Pruning]" if self.config.scheduled_pruning else "[Fixed Pruning]"
        target_msg = (
            f"epoch target {target_sparsity:.2%}, final target {self.config.final_amount:.2%}"
            if self.config.scheduled_pruning
            else f"target {target_sparsity:.2%}"
        )
        self._log(
            f"{label} Epoch {display_epoch:>2}: pruned {prune_amt:.2%} of remaining params "
            f"({target_msg}, achieved {self.pruning_state.latest_sparsity:.2%})"
        )

        return True

    def apply_pruning(
        self,
        amount: Union[int, float],
        pl_module: LightningModule = None,
        prevalidated_params: Optional[List[Tuple[nn.Module, str]]] = None,
    ) -> None:
        """Apply pruning using validated parameters."""
        self.parameter_manager.invalidate_cache()

        valid_params = prevalidated_params or self.parameter_manager.get_valid_parameters()
        if not valid_params:
            raise RuntimeError("apply_pruning called without any valid parameters to prune.")

        # Temporarily override parameters_to_prune for parent class
        original_params = self._parameters_to_prune
        self._parameters_to_prune = valid_params

        try:
            super().apply_pruning(amount)
            self._log(f"Applied pruning amount {amount:.2%} to {len(valid_params)} parameters", level=2)
        except Exception as exc:
            logger.error(f"Pruning failed: {exc}")
            raise
        finally:
            self._parameters_to_prune = original_params
            self.parameter_manager.invalidate_cache()

    def make_pruning_permanent(self) -> None:
        """Remove pruning masks from all pruned parameters.
        
        This method permanently removes the pruning masks, converting masked
        parameters back to regular parameters with zeros in pruned locations.
        """
        if not self.config.make_pruning_permanent:
            return

        self._log("Finalizing pruning: removing masks and reparametrizations...")

        pruned_params = self.parameter_manager.get_actually_pruned_parameters()
        if not pruned_params:
            self._log("No pruned parameters found (nothing to finalize).")
            return

        successful_removals = 0
        for module, param_name in pruned_params:
            if self._remove_pruning_mask(module, param_name):
                successful_removals += 1

        self._log(f"Successfully removed {successful_removals}/{len(pruned_params)} pruning masks")

    def _remove_pruning_mask(self, module: nn.Module, param_name: str) -> bool:
        """Remove pruning mask from a single parameter."""
        if not (pytorch_prune.is_pruned(module) and hasattr(module, f"{param_name}_mask")):
            return False

        try:
            pytorch_prune.remove(module, param_name)
            self._log(f"Mask removed: {module.__class__.__name__}.{param_name}", level=2)
            return True
        except Exception as e:
            raise RuntimeError(
                f"Could not remove mask for {module.__class__.__name__}.{param_name}"
            ) from e

    # Lightning Callback Methods
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle pruning state restoration after checkpoint loading."""
        self.pruning_state.is_resuming = bool(trainer.ckpt_path or self.pruning_state.loaded_from_checkpoint)
        self._maybe_restore_parameters_from_state(pl_module)
        self._collect_parameters_if_needed(pl_module)
        self._log_parameter_overview()

        state = self.pruning_state.saved_state
        if state is not None:
            self._log(f"Restoring pruning state from checkpoint: {state}")
            
            # Restore state
            self.config.scheduled_pruning = state["scheduled_pruning"]
            
            if self.config.scheduled_pruning:
                self.config.final_amount = state["final_amount"]
                self.config.epochs_to_ramp = state["epochs_to_ramp"]
                # Re-initialize scheduler with restored config
                self.scheduler = PruningScheduler(
                    self.config.schedule_type,
                    self.config.final_amount,
                    self.config.epochs_to_ramp
                )
                self.pruning_state.current_epoch = state["current_epoch"]
                
                self._log(
                    f"Resuming scheduled pruning at epoch {state['current_epoch']} "
                    f"(target: {self.config.final_amount:.3f}, ramp: {self.config.epochs_to_ramp} epochs)"
                )
            else:
                self.config.amount = state["amount"]
            
            self.pruning_state.latest_sparsity = state["checkpoint_sparsity"]
            self.pruning_state.completed = state.get("pruning_completed", False)
            self.pruning_state.saved_state = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle pruning before training starts."""
        self.pruning_state.is_resuming = bool(trainer.ckpt_path or self.pruning_state.loaded_from_checkpoint)
        self.pruning_state.loaded_from_checkpoint = False
        
        trigger_pre_training = self.config.pruning_trigger == "pre_training"
        
        if self.config.save_when_sparser_than is not None:
            self._setup_checkpoint_callbacks(trainer)
            self._setup_early_stopping_callbacks(trainer)

        if self.config.scheduled_pruning:
            self._warn_if_schedule_cannot_reach_target(trainer)

        if not trigger_pre_training:
            return

        # Apply pre-training pruning if needed
        if self.pruning_state.completed:
            self._log("Pruning already completed. Skipping pre-training pruning.")
            return

        current_sparsity = self.pruning_state.latest_sparsity
        target_sparsity = self.config.final_amount if self.config.scheduled_pruning else self.config.amount
        tolerance = 1e-4

        if (self.pruning_state.is_resuming and 
            isinstance(target_sparsity, float) and 
            current_sparsity >= target_sparsity - tolerance):
            self._log("Target sparsity already reached before checkpoint. Skipping pre-training pruning.")
            return

        self._log("Applying pre-training pruning")
        self._run_pruning(0, pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run pruning at the beginning of the training epoch."""
        if self.config.pruning_trigger == "epoch_start":
            self.pruning_state.current_epoch = trainer.current_epoch
            self._run_pruning(trainer.current_epoch, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run pruning at the end of the training epoch when requested."""
        if self.config.pruning_trigger == "epoch_end":
            self.pruning_state.current_epoch = trainer.current_epoch
            self._run_pruning(trainer.current_epoch, pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Conditionally disable checkpointing before validation runs."""
        if self.config.save_when_sparser_than is None:
            return

        self.checkpoint_state.disabled_for_validation = False

        # Check current sparsity
        try:
            valid_params = self.parameter_manager.get_valid_parameters()
            current_sparsity = self.parameter_manager.compute_sparsity(valid_params)
        except RuntimeError:
            current_sparsity = self.pruning_state.latest_sparsity
        
        self.pruning_state.latest_sparsity = current_sparsity
        
        # Determine if we should save based on sparsity threshold
        should_save = current_sparsity >= self.config.save_when_sparser_than - 0.01
        
        if not should_save:
            reason = (
                f"Epoch {trainer.current_epoch}: Sparsity {current_sparsity:.2%} < "
                f"{self.config.save_when_sparser_than:.2%}. Disabling checkpoint saving."
            )
            self.checkpoint_state.disabled_for_validation = True
            self._disable_best_checkpointing(trainer, reason, capture_metric_state=True)
        else:
            reason = (
                f"Epoch {trainer.current_epoch}: Sparsity {current_sparsity:.2%} reached threshold "
                f"{self.config.save_when_sparser_than:.2%}. Re-enabling checkpoint saving."
            )
            self._enable_best_checkpointing(
                trainer,
                reason=reason,
                should_reset_best_score=not self._best_score_has_been_reset,
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore checkpoint metric state after guarded validation runs."""
        if self.config.save_when_sparser_than is None:
            return

        if self.checkpoint_state.disabled_for_validation:
            self._restore_checkpoint_metrics()
            self.checkpoint_state.disabled_for_validation = False

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle end of training with pruning finalization."""
        if self.config.save_when_sparser_than is not None:
            self._handle_training_end_checkpointing(trainer)

        # Finalize pruning if requested
        if self.config.make_pruning_permanent:
            try:
                self.make_pruning_permanent()
                self.pruning_state.completed = True
                self._log("Pruning finalized and marked as completed.")
            except Exception as e:
                logger.error(f"Error during pruning finalization: {e}")

    def _handle_training_end_checkpointing(self, trainer: Trainer) -> None:
        """Handle checkpointing at end of training."""
        try:
            valid_params = self.parameter_manager.get_valid_parameters()
        except RuntimeError:
            valid_params = []

        current_sparsity = (
            self.parameter_manager.compute_sparsity(valid_params)
            if valid_params
            else self.pruning_state.latest_sparsity
        )
        self.pruning_state.latest_sparsity = current_sparsity

        tolerance = 1e-4
        if current_sparsity < self.config.save_when_sparser_than - tolerance:
            reason = (
                f"Training ended with sparsity {current_sparsity:.2%} < "
                f"{self.config.save_when_sparser_than:.2%}; disabling best checkpoint exports."
            )
            self._disable_best_checkpointing(trainer, reason, capture_metric_state=True)
        else:
            self._restore_checkpoint_settings(include_monitor=True)

    def _setup_checkpoint_callbacks(self, trainer: Trainer) -> None:
        """Setup checkpoint callbacks for sparsity-based saving."""
        self.checkpoint_state.callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        
        if not self.checkpoint_state.callbacks:
            raise RuntimeError(
                "No ModelCheckpoint callback found while save_when_sparser_than is set."
            )

        for i, cb in enumerate(self.checkpoint_state.callbacks):
            self.checkpoint_state.original_settings[i] = {
                "monitor": getattr(cb, 'monitor', None),
                "save_top_k": getattr(cb, 'save_top_k', 1),
                "save_last": getattr(cb, 'save_last', None),
            }
        
        self._checkpointing_active = True

    def _setup_early_stopping_callbacks(self, trainer: Trainer) -> None:
        """Setup early stopping callbacks."""
        self.early_stopping_state.callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
        ]
        
        for i, cb in enumerate(self.early_stopping_state.callbacks):
            self.early_stopping_state.original_settings[i] = {
                "patience": getattr(cb, "patience", None),
                "wait_count": getattr(cb, "wait_count", 0),
            }

    def _disable_early_stopping(self) -> None:
        """Disable early stopping callbacks during warmup."""
        if not self.early_stopping_state.callbacks or self.early_stopping_state.is_disabled:
            return

        for idx, callback in enumerate(self.early_stopping_state.callbacks):
            # Save current state
            if idx not in self.early_stopping_state.original_settings:
                self.early_stopping_state.original_settings[idx] = {
                    "patience": getattr(callback, "patience", None),
                    "wait_count": getattr(callback, "wait_count", 0),
                }
            
            # Set patience to infinity to prevent triggering
            callback.patience = float('inf')
            callback.wait_count = 0
            if hasattr(callback, "cooldown_counter"):
                callback.cooldown_counter = 0
            if hasattr(callback, "stopped_epoch"):
                callback.stopped_epoch = 0

        self.early_stopping_state.is_disabled = True
        logger.debug("Early stopping disabled during warmup")

    def _restore_early_stopping(self) -> None:
        """Re-enable early stopping callbacks after warmup."""
        if not self.early_stopping_state.callbacks or not self.early_stopping_state.is_disabled:
            return

        for idx, callback in enumerate(self.early_stopping_state.callbacks):
            original = self.early_stopping_state.original_settings.get(idx, {})
            patience = original.get("patience")
            if patience is not None and patience != float('inf'):
                callback.patience = patience
            
            # Reset wait counter to start fresh after warmup
            callback.wait_count = 0
            if hasattr(callback, "cooldown_counter"):
                callback.cooldown_counter = 0
            if hasattr(callback, "stopped_epoch"):
                callback.stopped_epoch = 0

        self.early_stopping_state.is_disabled = False
        logger.debug("Early stopping re-enabled after warmup")

    def _reset_early_stopping_metrics(self) -> None:
        """Reset early stopping best score after warmup."""
        if not self.early_stopping_state.callbacks:
            return

        for callback in self.early_stopping_state.callbacks:
            # Reset best score to worst possible value based on mode
            mode = getattr(callback, "mode", "min")
            extreme = float("inf") if mode == "min" else float("-inf")
            
            # Handle both tensor and float best_score
            best_score = getattr(callback, "best_score", None)
            if isinstance(best_score, torch.Tensor):
                callback.best_score = torch.tensor(extreme, device=best_score.device)
            else:
                callback.best_score = torch.tensor(extreme)
            
            callback.wait_count = 0
            if hasattr(callback, "cooldown_counter"):
                callback.cooldown_counter = 0
            if hasattr(callback, "stopped_epoch"):
                callback.stopped_epoch = 0
        
        logger.debug("Early stopping metrics reset")

    def _disable_best_checkpointing(self, trainer: Trainer, reason: str, *, capture_metric_state: bool = False) -> None:
        """Temporarily disable best checkpoint saving."""
        if not self.checkpoint_state.callbacks:
            return

        state_changed = self._checkpointing_active is not False
        if trainer.is_global_zero and reason and state_changed:
            logger.info(reason)

        self._checkpointing_active = False

        for idx, callback in enumerate(self.checkpoint_state.callbacks):
            if capture_metric_state:
                self._snapshot_checkpoint_metrics(idx, callback)
            callback.save_top_k = 0

        if state_changed and trainer.is_global_zero and not self._warmup_announced:
            logger.info(
                "Checkpoint/EarlyStopping warmup active until sparsity >= %.2f%%.",
                self.config.save_when_sparser_than * 100,
            )
            self._warmup_announced = True
            # Actually disable early stopping during warmup
            self._disable_early_stopping()

    def _enable_best_checkpointing(self, trainer: Trainer, *, reason: str = None, should_reset_best_score: bool = False) -> None:
        """Restore checkpointing behavior."""
        if not self.checkpoint_state.callbacks:
            return

        state_changed = self._checkpointing_active is not True
        if trainer.is_global_zero and reason and state_changed:
            logger.info(reason)

        self._checkpointing_active = True
        self._restore_checkpoint_settings(include_monitor=False)

        if state_changed and trainer.is_global_zero and self._warmup_announced:
            logger.info("Warmup complete; checkpointing and EarlyStopping re-enabled.")
            self._warmup_announced = False
            # Actually restore early stopping after warmup
            self._restore_early_stopping()
            
        # Reset early stopping best score if requested
        if should_reset_best_score and not self.early_stopping_state.best_reset:
            self._reset_early_stopping_metrics()
            self.early_stopping_state.best_reset = True

    def _restore_checkpoint_settings(self, include_monitor: bool = False) -> None:
        """Restore cached checkpoint settings."""
        for i, callback in enumerate(self.checkpoint_state.callbacks):
            if i not in self.checkpoint_state.original_settings:
                logger.warning(f"No cached settings for checkpoint callback {i}")
                continue
                
            original = self.checkpoint_state.original_settings[i]
            
            if include_monitor:
                callback.monitor = original["monitor"]
            callback.save_top_k = original["save_top_k"]
            callback.save_last = original["save_last"]

    def _snapshot_checkpoint_metrics(self, index: int, callback: ModelCheckpoint) -> None:
        """Capture checkpoint metric state."""
        attrs = (
            "best_model_score",
            "best_model_path",
            "current_score",
            "kth_value",
            "kth_best_model_path",
        )
        snapshot: Dict[str, Any] = {}

        for attr in attrs:
            if hasattr(callback, attr):
                snapshot[attr] = copy.deepcopy(getattr(callback, attr))

        best_k_models = getattr(callback, "best_k_models", None)
        if isinstance(best_k_models, dict):
            snapshot["best_k_models"] = copy.deepcopy(best_k_models)

        if snapshot:
            self._checkpoint_metric_snapshots[index] = snapshot

    def _restore_checkpoint_metrics(self) -> None:
        """Restore checkpoint metric state."""
        if not self._checkpoint_metric_snapshots:
            return

        for index, snapshot in self._checkpoint_metric_snapshots.items():
            if index >= len(self.checkpoint_state.callbacks):
                continue

            callback = self.checkpoint_state.callbacks[index]
            best_k_models = snapshot.get("best_k_models")
            if best_k_models is not None:
                callback.best_k_models = copy.deepcopy(best_k_models)

            for attr, value in snapshot.items():
                if attr == "best_k_models":
                    continue
                setattr(callback, attr, copy.deepcopy(value))

        self._checkpoint_metric_snapshots.clear()

    # State management for checkpoints
    def get_state(self) -> dict:
        """Returns information about the current sparsity."""
        valid_params = self.parameter_manager.get_valid_parameters()
        current_sparsity = self.parameter_manager.compute_sparsity(valid_params)
        return {
            "current_sparsity": current_sparsity,
            "target_sparsity": self.config.final_amount if self.config.scheduled_pruning else self.config.amount,
            "scheduled_pruning": self.config.scheduled_pruning,
            "schedule_type": self.config.schedule_type,
        }

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Save the pruning callback state to checkpoint."""
        try:
            valid_params = self.parameter_manager.get_valid_parameters()
            if valid_params:
                self.pruning_state.latest_sparsity = self.parameter_manager.compute_sparsity(valid_params)
        except RuntimeError:
            pass  # No valid parameters yet

        # Save essential state
        pruning_state = {
            "scheduled_pruning": self.config.scheduled_pruning,
            "current_epoch": trainer.current_epoch,
            "pruning_completed": self.pruning_state.completed,
            "checkpoint_sparsity": self.pruning_state.latest_sparsity,
        }
        
        if self.config.scheduled_pruning:
            pruning_state.update({
                "final_amount": self.config.final_amount,
                "epochs_to_ramp": self.config.epochs_to_ramp,
                "current_amount": self.scheduler.get_target_sparsity(trainer.current_epoch),
                "constant_schedule_fraction": self.scheduler.constant_fraction if self.scheduler else None,
            })
        else:
            pruning_state["amount"] = self.config.amount
            
        # Save parameters_to_prune information
        if self.parameter_manager.prunable_parameters:
            pruning_state["parameters_to_prune_info"] = ParameterSnapshotter.serialize(
                self.parameter_manager.prunable_parameters,
                pl_module,
            )
            
        checkpoint["magnitude_pruner_state"] = pruning_state

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Load pruning state for training resumption."""
        self._log("Loading pruning state from checkpoint")
        
        if "magnitude_pruner_state" in checkpoint:
            pruner_state = checkpoint["magnitude_pruner_state"]
            
            if pruner_state is None:
                raise RuntimeError("Expected non-null magnitude_pruner_state in checkpoint")

            self.pruning_state.saved_state = pruner_state
            self.pruning_state.completed = pruner_state.get("pruning_completed", False)
            self.pruning_state.latest_sparsity = pruner_state["checkpoint_sparsity"]
            self.pruning_state.loaded_from_checkpoint = True
            
            self._log(f"Loaded pruning state: {self.pruning_state.saved_state}")
