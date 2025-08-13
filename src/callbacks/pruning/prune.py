from pytorch_lightning.callbacks.pruning import ModelPruning
import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import List, Tuple, Callable, Optional, Union, Dict, Any
import warnings
from collections import defaultdict
from pytorch_lightning import LightningModule, Trainer

from src import utils

log = utils.get_pylogger(__name__)


class MagnitudePruner(ModelPruning):
    """
    MagnitudePruner - Enhanced version of PyTorch Lightning's ModelPruning
    
    Adds safety features, scheduled pruning, and metrics collection while maintaining
    all original functionalities in a more compact implementation.
    
    Parameters
    ----------
    pruning_fn : Union[Callable, str], default="l1_unstructured"
        The pruning function or name from torch.nn.utils.prune
    amount : Union[int, float], default=0.5
        Amount of parameters to prune
    use_global_unstructured : bool, default=True
        Whether to apply pruning globally across all parameters
    apply_pruning : bool, default=True
        Whether to actually apply pruning
    make_pruning_permanent : bool, default=True
        Whether to permanently remove pruned weights after training
    use_lottery_ticket_hypothesis : bool, default=False
        Whether to reset remaining weights to original values
    resample_parameters : bool, default=False
        Used with lottery ticket hypothesis for parameter resampling
    parameters_to_prune : Optional[List[Tuple[nn.Module, str]]], default=None
        Specific parameters to prune
    pruning_dim : Optional[int], default=None
        Dimension for structured pruning
    pruning_norm : Optional[int], default=None
        Norm for structured pruning
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed)
    prune_on_train_epoch_start : bool, default=True
        Whether to prune at the start of each training epoch
    prune_on_train_epoch_end : bool, default=False
        Whether to prune at the end of each training epoch
    scheduled_pruning : bool, default=False
        Whether to gradually increase pruning over epochs
    initial_amount : float, default=0.0
        Starting pruning rate for scheduled pruning
    final_amount : Optional[float], default=None
        Final pruning amount (uses amount if None)
    epochs_to_ramp : int, default=10
        Epochs to ramp from initial to final amount
    collect_metrics : bool, default=False
        Whether to collect detailed sparsity metrics
    **kwargs : Additional arguments passed to parent ModelPruning
    """
    
    def __init__(
        self,
        pruning_fn: Union[Callable, str] = "l1_unstructured",
        amount: Union[int, float] = 0.8,
        use_global_unstructured: bool = True,
        apply_pruning: bool = True,
        make_pruning_permanent: bool = True,
        use_lottery_ticket_hypothesis: bool = False,
        resample_parameters: bool = False,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
        verbose: int = 0,
        prune_on_train_epoch_start: bool = True,
        prune_on_train_epoch_end: bool = False,
        scheduled_pruning: bool = False,
        initial_amount: float = 0.0,
        final_amount: Optional[float] = None,
        epochs_to_ramp: int = 10,
        collect_metrics: bool = False,
        save_when_sparser_than: Optional[float] = None,
        **kwargs
    ):
        # Store all key parameters first since they're used throughout the class
        self.verbose = verbose
        self._pruning_fn_name = pruning_fn if isinstance(pruning_fn, str) else getattr(pruning_fn, "__name__", "unknown_callable")
        self.use_global_unstructured = use_global_unstructured
        self._apply_pruning = apply_pruning  # Renamed to avoid conflict with method
        self.should_make_pruning_permanent = make_pruning_permanent
        self._use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis  # Renamed to avoid conflict
        self.resample_parameters = resample_parameters
        self._parameters_to_prune = parameters_to_prune
        self.pruning_dim = pruning_dim
        self.pruning_norm = pruning_norm
        self.prune_on_train_epoch_start = prune_on_train_epoch_start
        
        # Add support for pruning before training starts
        self.prune_on_train_start = kwargs.get('prune_on_train_start', False)
        
        if prune_on_train_epoch_start and prune_on_train_epoch_end:
            raise ValueError("`prune_on_train_epoch_start` and `prune_on_train_epoch_end` cannot both be True.")

        # Validate structured pruning requirements
        if "ln_structured" in self._pruning_fn_name and pruning_dim is None:
            raise ValueError("`pruning_dim` must be specified for structured pruning.")

        self.scheduled_pruning = scheduled_pruning

        if scheduled_pruning:
            # ignore user‐passed amount, use schedule
            self.initial_amount = initial_amount
            self.final_amount = final_amount if final_amount is not None else amount  # Changed: use amount if final_amount is None
            self.epochs_to_ramp = max(1, epochs_to_ramp)
            self._validate_params(None, True, self.initial_amount, self.final_amount)
            effective_amount = self.initial_amount
        else:
            # fixed mode
            self.amount = amount
            self.initial_amount = self.final_amount = None
            self.epochs_to_ramp = None
            self._validate_params(self.amount, False, None, None)
            effective_amount = self.amount

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
            prune_on_train_epoch_end=prune_on_train_epoch_end,
        )
        
        # Initialize metrics and caching
        self.collect_metrics = collect_metrics
        self.metrics = defaultdict(list) if collect_metrics else None
        self._validated_params_cache = None
        self.skipped_params = {}
        self.save_when_sparser_than = save_when_sparser_than
        self._checkpoint_callbacks = []
        self._removed_checkpoint_callbacks = [] # To store temporarily removed callbacks
        self._checkpoint_original_save_on_train_epoch_end = {}
        self._original_update_best_and_save = {}
        self._has_reset_checkpoint_best_score = False

        
    def _validate_params(self, amount, scheduled_pruning, initial_amount, final_amount):
        """Validate initialization parameters."""
        if amount is not None and isinstance(amount, float) and not (0 <= amount <= 1):
            raise ValueError(f"Pruning amount {amount} must be between 0 and 1")
            
        if scheduled_pruning:
            if not (0 <= initial_amount <= 1):
                raise ValueError(f"Initial amount {initial_amount} must be between 0 and 1")
            if final_amount is not None and not (0 <= final_amount <= 1):
                raise ValueError(f"Final amount {final_amount} must be between 0 and 1")
            if final_amount is not None and initial_amount >= final_amount:
                raise ValueError("Initial amount must be less than final amount")

    def _get_current_amount(self, current_epoch: int) -> float:
        """Calculate current pruning amount for scheduled pruning."""
        if not self.scheduled_pruning:
            return self.amount

        # A ramp of N epochs should complete at the *end* of epoch N-1.
        if current_epoch >= self.epochs_to_ramp - 1:
            return self.final_amount
        
        # Linear interpolation: use (epoch + 1) to start ramp from epoch 0
        progress = (current_epoch + 1) / self.epochs_to_ramp
        progress = min(1.0, max(0.0, progress))  # Ensure progress is between [0, 1]
        
        current_amount = self.initial_amount + (self.final_amount - self.initial_amount) * progress
        
        # Safety check: never exceed final_amount regardless of calculation
        current_amount = min(current_amount, self.final_amount)
                
        return current_amount

    def _get_valid_parameters(self) -> List[Tuple[nn.Module, str]]:
        """Get validated parameters with caching."""
        if self._validated_params_cache is not None:
            return self._validated_params_cache

        valid_params = []
        self.skipped_params.clear()
        
        # Ensure we have parameters to check
        if not hasattr(self, '_parameters_to_prune') or not self._parameters_to_prune:
            log.warning("No parameters specified for pruning")
            return []
        
        is_structured = "ln_structured" in self._pruning_fn_name
        
        if self.verbose > 1:
            log.debug(f"Validating {len(self._parameters_to_prune)} parameters for {'structured' if is_structured else 'unstructured'} pruning")
        
        for module, param_name in self._parameters_to_prune:
            try:
                # After pruning, the parameter is a property. We need the underlying tensor.
                # The `_orig` attribute holds the original tensor.
                param_to_validate = getattr(module, f"{param_name}_orig", None)
                # If not pruned yet, get the parameter directly.
                if param_to_validate is None:
                    param_to_validate = getattr(module, param_name, None)

                # Comprehensive validation
                if param_to_validate is None:
                    self._log_skip(module, param_name, "Parameter doesn't exist")
                elif not isinstance(param_to_validate, torch.Tensor):
                    self._log_skip(module, param_name, f"Not a tensor (type: {type(param_to_validate)})")
                elif param_to_validate.dim() == 0:
                    self._log_skip(module, param_name, "Scalar tensor")
                elif param_to_validate.numel() == 0:
                    self._log_skip(module, param_name, "Empty tensor")
                elif param_to_validate.numel() < 2:
                    self._log_skip(module, param_name, f"Too few parameters ({param_to_validate.numel()}) for meaningful pruning")
                elif is_structured:
                    # Structured pruning validation
                    if param_to_validate.dim() < 2:
                        self._log_skip(module, param_name, f"Skipping 1D tensor (shape: {param_to_validate.shape}) for structured pruning")
                        continue
                    
                    if self.pruning_dim is None:
                        self._log_skip(module, param_name, "pruning_dim is not set for structured pruning")
                        continue

                    if param_to_validate.dim() <= self.pruning_dim:
                        self._log_skip(
                            module, param_name,
                            f"Parameter has {param_to_validate.dim()} dimensions, "
                            f"but structured pruning requires more than {self.pruning_dim} dimensions"
                        )
                        continue

                    # Only add if all checks pass for structured pruning
                    if self.verbose > 1:
                        log.debug(f"VALID for structured pruning: {module.__class__.__name__}.{param_name} (shape: {param_to_validate.shape})")
                    valid_params.append((module, param_name))
                else:
                    # Unstructured pruning - less restrictive
                    if self.verbose > 1:
                        log.debug(f"VALID for unstructured pruning: {module.__class__.__name__}.{param_name} (shape: {param_to_validate.shape})")
                    valid_params.append((module, param_name))
                    
            except Exception as e:
                self._log_skip(module, param_name, f"Error accessing: {str(e)}")
        
        # Report results
        if self.verbose > 0:
            log.info(f"Parameter validation complete: {len(valid_params)} valid, {len(self.skipped_params)} skipped")
            if self.skipped_params and self.verbose > 1:
                skipped_list = list(self.skipped_params.keys())
                log.debug(f"Skipped parameters: {skipped_list}")
        
        self._validated_params_cache = valid_params
        return valid_params

    def _get_default_parameters_to_prune(self) -> List[Tuple[nn.Module, str]]:
        """Get default parameters to prune with better filtering."""
        # This method should only be called from within _get_valid_parameters or on_fit_start
        # We need access to pl_module, so we'll defer this logic
        return []

    def make_pruning_permanent(self, pl_module):
        """
        Permanently remove pruning masks from all pruned parameters.
        """
        if not self.should_make_pruning_permanent:
            return

        if self.verbose > 0:
            log.info("Finalizing pruning: removing masks and reparametrizations...")

        # Get only parameters that are actually pruned
        pruned_params = self._get_actually_pruned_parameters()
        
        if not pruned_params:
            if self.verbose > 0:
                log.info("No pruned parameters found (nothing to finalize).")
            return

        successful_removals = 0
        for module, param_name in pruned_params:
            try:
                # Double-check the parameter is actually pruned before trying to remove
                if pytorch_prune.is_pruned(module) and hasattr(module, f"{param_name}_mask"):
                    # remove mask/orig and restore parameter
                    pytorch_prune.remove(module, param_name)
                    successful_removals += 1
                    if self.verbose > 1:
                        log.debug(f"Mask removed: {module.__class__.__name__}.{param_name}")
                elif self.verbose > 1:
                    log.debug(f"Skipping {module.__class__.__name__}.{param_name} - not pruned or no mask")
            except Exception as e:
                if self.verbose > 0:
                    log.warning(
                        f"Could not remove mask for "
                        f"{module.__class__.__name__}.{param_name}: {e}"
                    )

        if self.verbose > 0:
            log.info(f"Successfully removed {successful_removals}/{len(pruned_params)} pruning masks")

    def _get_actually_pruned_parameters(self) -> List[Tuple[nn.Module, str]]:
        """Get only parameters that are actually pruned (have masks)."""
        if not hasattr(self, '_parameters_to_prune') or not self._parameters_to_prune:
            return []
        
        pruned_params = []
        for module, param_name in self._parameters_to_prune:
            try:
                # Check if this specific parameter is pruned
                if (pytorch_prune.is_pruned(module) and 
                    hasattr(module, f"{param_name}_mask") and
                    hasattr(module, f"{param_name}_orig")):
                    pruned_params.append((module, param_name))
            except Exception as e:
                if self.verbose > 1:
                    log.debug(f"Error checking if {module.__class__.__name__}.{param_name} is pruned: {e}")
                continue
        
        return pruned_params

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        Handle pruning state restoration after checkpoint loading is complete.
        """
        # Restore parameters_to_prune if not set and we have saved info
        if (self._parameters_to_prune is None and 
            hasattr(self, '_saved_state') and 
            'parameters_to_prune_info' in self._saved_state):
            
            if self.verbose > 0:
                log.info("Restoring parameters_to_prune from checkpoint...")
            
            # Rebuild parameters_to_prune by matching module names
            restored_params = []
            saved_info = self._saved_state['parameters_to_prune_info']
            
            for module_class_name, param_name, _ in saved_info:
                # Find matching modules in the current model
                for module in pl_module.modules():
                    if (module.__class__.__name__ == module_class_name and 
                        hasattr(module, param_name)):
                        restored_params.append((module, param_name))
                        if self.verbose > 1:
                            log.debug(f"Restored parameter: {module_class_name}.{param_name}")
                        break
            
            if restored_params:
                self._parameters_to_prune = restored_params
                if self.verbose > 0:
                    log.info(f"Restored {len(restored_params)} parameters for pruning")
            else:
                log.warning("Could not restore any parameters_to_prune from checkpoint")
        
        # If still None, fall back to default parameter collection
        if self._parameters_to_prune is None:
            if self.verbose > 0:
                log.info("parameters_to_prune is None, using default parameter collection...")
            
            self._parameters_to_prune = self._collect_default_prunable_parameters(pl_module)
            
            if self._parameters_to_prune:
                if self.verbose > 0:
                    log.info(f"Using {len(self._parameters_to_prune)} default parameters for pruning")
            else:
                log.error("No parameters found for pruning!")
                
        if hasattr(self, '_saved_state'):
            state = self._saved_state
            
            if self.verbose > 0:
                log.info(f"Restoring pruning state from checkpoint: {state}")
            
            # Restore the scheduled pruning configuration
            self.scheduled_pruning = state.get("scheduled_pruning", self.scheduled_pruning)
            
            if self.scheduled_pruning:
                # For scheduled pruning, adjust the initial_amount to continue from checkpoint
                checkpoint_epoch = state.get("current_epoch", 0)
                checkpoint_amount = state.get("current_amount", self.initial_amount)
                
                # Update initial_amount to the checkpoint amount so ramping continues correctly
                self.initial_amount = checkpoint_amount
                self.final_amount = state.get("final_amount", self.final_amount)
                
                # Adjust epochs_to_ramp based on remaining epochs
                original_epochs_to_ramp = state.get("epochs_to_ramp", self.epochs_to_ramp)
                epochs_completed = checkpoint_epoch + 1
                self.epochs_to_ramp = max(1, original_epochs_to_ramp - epochs_completed)
                
                if self.verbose > 0:
                    log.info(f"Resuming scheduled pruning: from {checkpoint_amount:.3f} to {self.final_amount:.3f} over {self.epochs_to_ramp} remaining epochs")
            else:
                # Fixed pruning
                self.amount = state.get("amount", self.amount)
                
            # Clean up
            delattr(self, '_saved_state')

    def _collect_default_prunable_parameters(self, pl_module) -> List[Tuple[nn.Module, str]]:
        """Collect default parameters for pruning with intelligent filtering."""
        default_params = []
        
        # Focus on weight parameters from major layer types
        for name, module in pl_module.named_modules():
            # Primary targets: weight parameters from major layer types
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    # Only include weight parameters that are substantial enough for pruning
                    if module.weight.numel() >= 10:  # Minimum threshold
                        default_params.append((module, 'weight'))
                        
                # For bias, only include if it's from a linear layer and substantial
                if (hasattr(module, 'bias') and module.bias is not None and 
                    isinstance(module, nn.Linear) and module.bias.numel() >= 10):
                    default_params.append((module, 'bias'))
            
            # Additional targets: BatchNorm and LayerNorm weight parameters (but not bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                if (hasattr(module, 'weight') and module.weight is not None and 
                    isinstance(module.weight, torch.Tensor) and module.weight.numel() >= 10):
                    default_params.append((module, 'weight'))
        
        if self.verbose > 1:
            log.debug(f"Collected {len(default_params)} default parameters for pruning")
            
        return default_params

    def setup(self, trainer, pl_module, stage):
        """Setup for training stage - collect default parameters if needed."""
        if self.verbose > 0:
            log.info(f"MagnitudePruner: setup called with stage={stage}")

        # For training stage, ensure we have default parameters if needed
        if str(stage) in ['TrainerFn.FITTING', 'fit'] and self._parameters_to_prune is None:
            if self.verbose > 0:
                log.info("MagnitudePruner: Setting up default parameters for training")
            self._parameters_to_prune = self._collect_default_prunable_parameters(pl_module)
            
            if self._parameters_to_prune and self.verbose > 0:
                log.info(f"MagnitudePruner: Using {len(self._parameters_to_prune)} default parameters for pruning")

    def _run_pruning(self, current_epoch: int, pl_module: LightningModule = None) -> None:
        """Override to handle scheduled pruning and safety."""
        # If this is called from on_train_epoch_end, current_epoch will be >= 0.
        # If called from on_train_start, we use -1.
        # This check prevents epoch-end pruning if the user has disabled it.
        if current_epoch >= 0 and not self.prune_on_train_epoch_start:
            return

        # Check pruning toggle
        should_prune = self._apply_pruning(current_epoch) if callable(self._apply_pruning) else bool(self._apply_pruning)
        if not should_prune:
            if self.verbose > 1:
                log.debug(f"Epoch {current_epoch}: skip (prune={should_prune})")
            return

        prune_amt = 0.0
        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params) if valid_params else 0.0
        
        if self.scheduled_pruning:
            # --- SCHEDULED PRUNING LOGIC ---
            target_sparsity = self._get_current_amount(current_epoch)
            
            if current_sparsity >= target_sparsity:
                if self.verbose > 0:
                    log.info(
                        f"[Epoch {current_epoch}] | Target sparsity {target_sparsity:.2%} reached (current {current_sparsity:.2%}), skipping."
                    )
                return
            
            remaining_weights = 1.0 - current_sparsity
            if remaining_weights > 0:
                prune_amt = (target_sparsity - current_sparsity) / remaining_weights
                prune_amt = min(max(0.0, prune_amt), 1.0)
            else:
                # Nothing left to prune
                log.warning(
                    f"[Epoch {current_epoch}] | No remaining weights to prune (current sparsity: {current_sparsity:.2%})."
                )
                return
        else:
            # --- FIXED PRUNING LOGIC ---
            target_sparsity = self.amount
            
            # Check if target sparsity already reached (important for checkpoint resumption)
            if current_sparsity >= target_sparsity - 1e-3: # 1e-3 tolerance for floating point precision
                if self.verbose > 0:
                    log.info(
                        f"[Epoch {current_epoch}] | Pruning target {target_sparsity:.2%} already reached (current {current_sparsity:.2%}), skipping."
                    )
                return
            
            # Calculate how much additional pruning is needed
            remaining_weights = 1.0 - current_sparsity
            if remaining_weights > 0:
                prune_amt = (target_sparsity - current_sparsity) / remaining_weights
                prune_amt = min(max(0.0, prune_amt), 1.0)
            else:
                # Nothing left to prune
                log.warning(
                    f"[Epoch {current_epoch}] | No remaining weights to prune (current sparsity: {current_sparsity:.2%})."
                )
                return

        if self.verbose > 0:
            label = "[Scheduled Pruning]" if self.scheduled_pruning else "[Fixed Pruning]"
            log.info(
                f"{label} Epoch {current_epoch:>2}: "
                f"prune {prune_amt:.2%} of remaining params"
                + (f" (aiming for total sparsity {target_sparsity:.2%})" if self.scheduled_pruning else "")
            )

        try:
            self.apply_pruning(prune_amt, pl_module)
        except Exception as e:
            log.error(f"Pruning failed at epoch {current_epoch}: {e}")
            if self.verbose > 0:
                raise

        # lottery ticket hypothesis
        use_lth = self._use_lottery_ticket_hypothesis(current_epoch) if callable(self._use_lottery_ticket_hypothesis) else bool(self._use_lottery_ticket_hypothesis)
        if use_lth:
            self.apply_lottery_ticket_hypothesis()

    def apply_pruning(self, amount: Union[int, float], pl_module: LightningModule = None) -> None:
        """Override with comprehensive safety checks and robust device compatibility."""
        # Invalidate cache since we're about to change the model
        self._validated_params_cache = None
        
        # Get valid parameters
        valid_params = self._get_valid_parameters()
        if not valid_params:
            log.warning("No valid parameters found for pruning")
            return
        
        # Store sparsity before pruning for comparison (only if collecting metrics)
        sparsity_before = 0.0
        if self.collect_metrics:
            sparsity_before = self._compute_current_sparsity(valid_params)
        
        # Temporarily replace parameters
        original_params = self._parameters_to_prune
        self._parameters_to_prune = valid_params
        
        is_structured = "ln_structured" in self._pruning_fn_name

        try:
            # Ensure device consistency - move any existing masks to correct device
            self._ensure_device_consistency(pl_module)
            
            # Check device consistency after fixing
            devices = set()
            for module, param_name in self._parameters_to_prune:
                param = getattr(module, param_name, None)
                if isinstance(param, torch.Tensor):
                    devices.add(param.device)
            
            has_device_issues = len(devices) > 1
            if has_device_issues and self.verbose > 0:
                log.warning(f"Parameters are still on different devices after fix attempt: {devices}. Will use local pruning as fallback.")

            # Try global pruning first, fallback to local if device issues occur
            if self.use_global_unstructured and not has_device_issues:
                try:
                    # Get pruning method class for global pruning
                    pruning_method_class = self._get_pruning_method(for_global=True)
                    pytorch_prune.global_unstructured(
                        self._parameters_to_prune, 
                        pruning_method=pruning_method_class, 
                        amount=amount
                    )
                    if self.verbose > 1:
                        log.debug("Global pruning completed successfully")
                except RuntimeError as e:
                    if "device" in str(e).lower():
                        if self.verbose > 0:
                            log.warning(f"Global pruning failed due to device issues: {e}. Falling back to local pruning.")
                        has_device_issues = True
                    else:
                        raise
            
            # Use local pruning if global failed or if we detected device issues upfront
            if not self.use_global_unstructured or has_device_issues:
                if self.verbose > 0:
                    pruning_type = "local" if not self.use_global_unstructured else "fallback local"
                    log.info(f"Applying {pruning_type} {self._pruning_fn_name} pruning to {len(self._parameters_to_prune)} parameters")
                
                # Get pruning method function for local pruning
                pruning_method_func = self._get_pruning_method(for_global=False)
                
                successful_prunes = 0
                for module, name in self._parameters_to_prune:
                    try:
                        # Ensure this specific parameter and any existing masks are on the same device
                        self._ensure_parameter_device_consistency(pl_module, module, name)
                        
                        if is_structured:
                            # For structured pruning, pass additional kwargs
                            kwargs = {}
                            if self.pruning_dim is not None:
                                kwargs["dim"] = self.pruning_dim
                            if self.pruning_norm is not None:
                                kwargs["n"] = self.pruning_norm
                            pruning_method_func(module, name, amount, **kwargs)
                        else:
                            # For unstructured pruning, simple call
                            pruning_method_func(module, name, amount)
                        successful_prunes += 1
                    except Exception as e:
                        param = getattr(module, name, "N/A")
                        shape = getattr(param, "shape", "N/A")
                        device = getattr(param, "device", "N/A")
                        log.error(f"Failed to prune {module.__class__.__name__}.{name} (shape: {shape}, device: {device}): {e}")
                        if self.verbose > 1:
                            log.debug(f"Available device: {device}, param type: {type(param)}")
                
                if self.verbose > 0:
                    log.info(f"Successfully pruned {successful_prunes}/{len(self._parameters_to_prune)} parameters")
                
            # Update metrics AFTER pruning is applied
            if self.collect_metrics:
                # Recompute sparsity after pruning
                sparsity_after = self._compute_current_sparsity(valid_params)
                delta = sparsity_after - sparsity_before
                self.metrics["magnitude_pruner/overall_sparsity"].append(sparsity_after)
                
                if self.verbose > 0:
                    log.info(
                        f"Pruning applied: sparsity {sparsity_before:.2%} → "
                        f"{sparsity_after:.2%} (Δ{delta:.2%})"
                    )
                
        finally:
            self._parameters_to_prune = original_params
            # Invalidate cache again since model structure changed
            self._validated_params_cache = None

    def _get_pruning_method(self, for_global: bool = True) -> Callable:
        """Resolve pruning method from string or callable."""
        # Map string names to pruning methods
        if for_global:
            # For global pruning, we need the method classes
            from torch.nn.utils.prune import L1Unstructured, RandomUnstructured, LnStructured, RandomStructured
            pruning_methods = {
                'l1_unstructured': L1Unstructured,
                'random_unstructured': RandomUnstructured, 
                'ln_structured': LnStructured,
                'random_structured': RandomStructured,
            }
        else:
            # For local pruning, we need the method functions
            pruning_methods = {
                'l1_unstructured': pytorch_prune.l1_unstructured,
                'random_unstructured': pytorch_prune.random_unstructured,
                'ln_structured': pytorch_prune.ln_structured,
                'random_structured': pytorch_prune.random_structured,
            }
        
        if isinstance(self._pruning_fn_name, str):
            if self._pruning_fn_name not in pruning_methods:
                raise ValueError(f"Unknown pruning function: {self._pruning_fn_name}. Available: {list(pruning_methods.keys())}")
            return pruning_methods[self._pruning_fn_name]
        elif callable(self.pruning_fn):
            return self.pruning_fn
        else:
            raise TypeError(f"Invalid pruning_fn type: {type(self.pruning_fn)}")

    def _ensure_device_consistency(self, pl_module: LightningModule = None) -> None:
        """Ensure all parameters, their masks, and originals live on a consistent device."""
        if not self._parameters_to_prune:
            return

        # Get target device from PyTorch Lightning module if available
        target_device = None
        if pl_module is not None:
            target_device = pl_module.device
        
        # Fallback: get device from first parameter
        if target_device is None:
            for module, param_name in self._parameters_to_prune:
                param = getattr(module, param_name, None)
                if isinstance(param, torch.Tensor):
                    target_device = param.device
                    break

        if target_device is None:
            return

        for module, param_name in self._parameters_to_prune:
            self._ensure_parameter_device_consistency(pl_module, module, param_name, target_device)

    def _ensure_parameter_device_consistency(
        self,
        pl_module: LightningModule = None,
        module: nn.Module = None,
        param_name: str = None,
        target_device: torch.device = None
    ) -> None:
        """Ensure a specific parameter, its mask, and *_orig live on target_device."""
        if module is None or param_name is None:
            return
            
        param = getattr(module, param_name, None)
        if not isinstance(param, torch.Tensor):
            return

        # Determine target device from PyTorch Lightning module if available
        if target_device is None and pl_module is not None:
            target_device = pl_module.device
        
        # Fallback to parameter's current device
        if target_device is None:
            target_device = param.device

        # Helper to safely move tensors respecting PyTorch Lightning's device management
        def _move_tensor_safely(owner: nn.Module, name: str, tensor: torch.Tensor):
            if tensor.device == target_device:
                return
                
            if self.verbose > 1:
                log.debug(f"Moving {owner.__class__.__name__}.{name} from {tensor.device} to {target_device}")
            
            try:
                # Create the moved tensor first
                moved_tensor = tensor.to(target_device, non_blocking=False)
                
                # For registered parameters - use proper PyTorch Lightning compatible approach
                if hasattr(owner, "_parameters") and name in owner._parameters:
                    if owner._parameters[name] is not None:
                        # Replace the parameter data in-place to maintain PyTorch Lightning's tracking
                        with torch.no_grad():
                            owner._parameters[name].data = moved_tensor.data
                            # Ensure the parameter itself is on the correct device
                            if hasattr(owner._parameters[name], 'device') and owner._parameters[name].device != target_device:
                                owner._parameters[name] = owner._parameters[name].to(target_device)
                # For registered buffers
                elif hasattr(owner, "_buffers") and name in owner._buffers:
                    if owner._buffers[name] is not None:
                        owner._buffers[name] = moved_tensor
                # For plain attributes
                else:
                    setattr(owner, name, moved_tensor)
                    
            except Exception as e:
                if self.verbose > 0:
                    log.warning(f"Failed to move {owner.__class__.__name__}.{name} to {target_device}: {e}")

        # Move the main parameter
        if param.device != target_device:
            _move_tensor_safely(module, param_name, param)

        # Move mask and original parameters if they exist
        mask_name = f"{param_name}_mask"
        orig_name = f"{param_name}_orig"

        # Move mask buffer (registered buffer during pruning)
        if hasattr(module, mask_name):
            mask = getattr(module, mask_name)
            if isinstance(mask, torch.Tensor) and mask.device != target_device:
                _move_tensor_safely(module, mask_name, mask)

        # Move original parameter
        if hasattr(module, orig_name):
            orig = getattr(module, orig_name)
            if isinstance(orig, torch.Tensor) and orig.device != target_device:
                _move_tensor_safely(module, orig_name, orig)

    def _log_skip(self, module: nn.Module, param_name: str, reason: str) -> None:
        """Log skipped parameter."""
        param_key = f"{module.__class__.__name__}.{param_name}"
        self.skipped_params[param_key] = reason

    def _compute_current_sparsity(self, parameters: List[Tuple[nn.Module, str]]) -> float:
        """Compute current sparsity using PyTorch's actual built-in utilities."""
        try:
            total_params = 0
            pruned_params = 0
            
            for module, param_name in parameters:
                param = getattr(module, param_name, None)
                if not isinstance(param, torch.Tensor):
                    continue
                
                # Check if module is pruned using PyTorch's built-in utility
                if pytorch_prune.is_pruned(module):
                    # Module has pruning applied, check for specific parameter mask
                    mask_name = f"{param_name}_mask"
                    if hasattr(module, mask_name):
                        mask = getattr(module, mask_name)
                        if isinstance(mask, torch.Tensor):
                            total_params += mask.numel()
                            pruned_params += (mask == 0).sum().item()
                            continue  # Move to next parameter
                
                # For unpruned parameters or if specific mask doesn't exist
                # Count the parameter but assume no pruning
                total_params += param.numel()
                # Count natural zeros (if any)
                pruned_params += (param == 0).sum().item()
            
            return pruned_params / max(1, total_params)
            
        except Exception as e:
            if self.verbose > 1:
                log.debug(f"Error in sparsity computation, using fallback: {e}")
            return self._compute_sparsity_fallback(parameters)
    
    def _compute_sparsity_fallback(self, parameters: List[Tuple[nn.Module, str]]) -> float:
        """Fallback sparsity computation using direct parameter inspection."""
        total_params = 0
        zero_params = 0
        
        for module, param_name in parameters:
            try:
                # Check for pruned parameters first (PyTorch pruning pattern)
                orig_param_name = f"{param_name}_orig"
                mask_param_name = f"{param_name}_mask"
                
                if hasattr(module, orig_param_name) and hasattr(module, mask_param_name):
                    # Parameter is pruned, use mask to compute sparsity
                    param_orig = getattr(module, orig_param_name)
                    mask = getattr(module, mask_param_name)
                    
                    if isinstance(param_orig, torch.Tensor) and isinstance(mask, torch.Tensor):
                        total_params += param_orig.numel()
                        # In PyTorch pruning: mask == 0 means pruned
                        zero_params += (mask == 0).sum().item()
                elif hasattr(module, param_name):
                    # Parameter is not pruned, check the parameter directly
                    param = getattr(module, param_name)
                    if isinstance(param, torch.Tensor):
                        total_params += param.numel()
                        zero_params += (param == 0).sum().item()
                        
            except Exception as e:
                if self.verbose > 1:
                    log.debug(f"Error in fallback sparsity for {module.__class__.__name__}.{param_name}: {e}")
                continue
        
        return zero_params / max(1, total_params)

    def _update_metrics(self) -> None:
        """Update sparsity metrics using built-in computation."""
        if not self.collect_metrics:
            return
            
        # Use our verified sparsity computation method
        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params)
        
        self.metrics["magnitude_pruner/overall_sparsity"].append(current_sparsity)
        
        if self.verbose > 0:
            log.info(f"Current sparsity: {current_sparsity:.4f}")

    def on_train_start(self, trainer, pl_module) -> None:
        """Handle pruning before training starts if enabled."""
        if self.save_when_sparser_than is not None:
            from pytorch_lightning.callbacks import ModelCheckpoint
            
            # Handle checkpoint callbacks
            self._checkpoint_callbacks = [
                cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
            ]
            if not self._checkpoint_callbacks:
                warnings.warn(
                    "No ModelCheckpoint callback found, but `save_when_sparser_than` is set."
                )
            else:
                for i, cb in enumerate(self._checkpoint_callbacks):
                    if hasattr(cb, '_save_on_train_epoch_end'):
                        self._checkpoint_original_save_on_train_epoch_end[i] = cb._save_on_train_epoch_end
            
        if self.prune_on_train_start:
            if self.verbose > 0:
                log.info("[PreTraining Pruning] Applying pruning before training starts")
            
            # For pre-training pruning, use epoch -1 to indicate before training
            self._run_pruning(-1, pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run pruning at the beginning of the training epoch."""
        if self.prune_on_train_epoch_start:
            self._run_pruning(trainer.current_epoch, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Override to add metrics logging."""
        # Ensure metrics are up-to-date before logging
        if self.collect_metrics:
            self._update_metrics()

        # Log metrics if available
        if self.collect_metrics and self.metrics and trainer.logger:
            try:
                if "magnitude_pruner/overall_sparsity" in self.metrics and self.metrics["magnitude_pruner/overall_sparsity"]:
                    latest_sparsity = self.metrics["magnitude_pruner/overall_sparsity"][-1]
                    
                    # Determine the current pruning amount to log
                    current_pruning_target_amount = self.amount
                    if self.scheduled_pruning:
                        current_pruning_target_amount = self._get_current_amount(trainer.current_epoch)

                    # Log to all available loggers
                    loggers = trainer.logger.loggers if hasattr(trainer.logger, 'loggers') else [trainer.logger]
                    for logger in loggers:
                        try:
                            if hasattr(logger, 'log_metrics'):
                                logger.log_metrics(
                                    {
                                        "magnitude_pruning/sparsity": latest_sparsity,
                                        "magnitude_pruning/target_amount": current_pruning_target_amount
                                    }, 
                                    step=trainer.global_step
                                )
                                break
                        except Exception as logger_error:
                            if self.verbose > 1:
                                log.debug(f"Logger {type(logger).__name__} failed: {logger_error}")
                            continue
                            
            except Exception as e:
                if self.verbose > 1:
                    log.debug(f"Failed to log metrics: {e}")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Conditionally disable checkpointing before validation runs."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params)
        should_save = current_sparsity >= self.save_when_sparser_than - 0.01 # 1% threshold for saving

        if not should_save:
            # Sparsity is below threshold. Disable saving for this validation run.
            for i, cb in enumerate(self._checkpoint_callbacks):
                if hasattr(cb, '_save_on_train_epoch_end'):
                    cb._save_on_train_epoch_end = False
                
                # Also prevent the callback from updating its best score.
                if hasattr(cb, '_update_best_and_save'):
                    if i not in self._original_update_best_and_save:
                        self._original_update_best_and_save[i] = cb._update_best_and_save
                    cb._update_best_and_save = lambda *args, **kwargs: None # No-op
            
            if trainer.is_global_zero:
                log.info(
                    f"Epoch {trainer.current_epoch}: Sparsity {current_sparsity:.2%} < {self.save_when_sparser_than:.2%}. "
                    "Disabling checkpoint saving for this validation run."
                )
        elif not self._has_reset_checkpoint_best_score:
            # This is the first epoch where sparsity is >= threshold.
            # Reset the best_model_score to start tracking from a clean slate.
            for cb in self._checkpoint_callbacks:
                if hasattr(cb, 'best_model_score') and hasattr(cb, 'mode'):
                    if cb.mode == "min":
                        cb.best_model_score = torch.tensor(float("inf"))
                    else:
                        cb.best_model_score = torch.tensor(float("-inf"))
                    
                    if trainer.is_global_zero:
                        log.info(f"Epoch {trainer.current_epoch}: Sparsity threshold reached. Resetting best_model_score for ModelCheckpoint to start tracking.")
            self._has_reset_checkpoint_best_score = True

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore checkpointing callbacks after validation."""
        # Restore the original behavior of the checkpoint callbacks, but only if saving should be enabled.
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params)
        should_save = current_sparsity >= self.save_when_sparser_than

        # Only re-enable the checkpoint's internal save flag if the sparsity threshold has been met.
        if should_save:
            for i, cb in enumerate(self._checkpoint_callbacks):
                # Restore the original value. If not found, default to True.
                original_state = self._checkpoint_original_save_on_train_epoch_end.get(i, True)
                if hasattr(cb, '_save_on_train_epoch_end'):
                    cb._save_on_train_epoch_end = original_state
                
                # Restore the original update method if it was replaced
                if i in self._original_update_best_and_save:
                    cb._update_best_and_save = self._original_update_best_and_save[i]
            
            if trainer.is_global_zero:
                 log.info(f"Epoch {trainer.current_epoch}: Re-enabled ModelCheckpoint saving for subsequent runs.")

    def on_train_end(self, trainer, pl_module) -> None:
        """Override to safely handle the end of training with pruning."""
        try:
            # Only make pruning permanent if explicitly asked for
            if self.should_make_pruning_permanent:
                self.make_pruning_permanent(pl_module)
        except Exception as e:
            log.error(f"Error during pruning finalization: {e}")
                
        # Log final metrics
        if self.collect_metrics and self.metrics:
            self._update_metrics()

        # Restore original checkpointing behavior
        if self._checkpoint_callbacks:
            for i, cb in enumerate(self._checkpoint_callbacks):
                cb._save_on_train_epoch_end = self._checkpoint_original_save_on_train_epoch_end.get(i, True)

    def get_state(self) -> dict:
        """Returns information about the current sparsity of the pruned modules."""
        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params)
        return {
            "current_sparsity": current_sparsity,
            "target_sparsity": self.final_amount if self.scheduled_pruning else self.amount,
            "scheduled_pruning": self.scheduled_pruning
        }

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save the pruning callback state to checkpoint."""
        # Save the essential state for resuming scheduled pruning
        pruning_state = {
            "scheduled_pruning": self.scheduled_pruning,
            "current_epoch": trainer.current_epoch,
        }
        
        if self.scheduled_pruning:
            pruning_state.update({
                "initial_amount": self.initial_amount,
                "final_amount": self.final_amount,
                "epochs_to_ramp": self.epochs_to_ramp,
                "current_amount": self._get_current_amount(trainer.current_epoch),
            })
        else:
            pruning_state["amount"] = self.amount
            
        # Save parameters_to_prune information for restoration
        if self._parameters_to_prune:
            pruning_state["parameters_to_prune_info"] = [
                (module.__class__.__name__, param_name, id(module))
                for module, param_name in self._parameters_to_prune
            ]
            
        checkpoint["magnitude_pruner_state"] = pruning_state

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Load pruning state for training resumption.
        
        Note: State dict conversion for testing is handled by PrunedCheckpointHandler
        to avoid duplication and ensure proper coordination between callbacks.
        """
        if self.verbose > 0:
            log.info("MagnitudePruner: on_load_checkpoint called")

        # Load the internal state of the pruner
        if "magnitude_pruner_state" in checkpoint:
            self._saved_state = checkpoint["magnitude_pruner_state"]
            if self.verbose > 0:
                log.info(f"MagnitudePruner: Loaded pruning state: {self._saved_state}")
