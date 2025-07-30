from pytorch_lightning.callbacks.pruning import ModelPruning
import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import List, Tuple, Callable, Optional, Union, Dict, Any
import warnings
from collections import defaultdict

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
    prune_on_train_epoch_end : bool, default=True
        Whether to prune at epoch end
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
        prune_on_train_epoch_end: bool = True,
        scheduled_pruning: bool = False,
        initial_amount: float = 0.0,
        final_amount: Optional[float] = None,
        epochs_to_ramp: int = 10,
        collect_metrics: bool = False,
        **kwargs
    ):
        # Store all key parameters first since they're used throughout the class
        self.verbose = verbose
        self.pruning_fn = pruning_fn
        self.use_global_unstructured = use_global_unstructured
        self._apply_pruning = apply_pruning  # Renamed to avoid conflict with method
        self.should_make_pruning_permanent = make_pruning_permanent
        self._use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis  # Renamed to avoid conflict
        self.resample_parameters = resample_parameters
        self._parameters_to_prune = parameters_to_prune
        self.pruning_dim = pruning_dim
        self.pruning_norm = pruning_norm
        self.prune_on_train_epoch_end = prune_on_train_epoch_end
        
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
            
        # PyTorch Lightning epochs are 0-based, so we adjust accordingly
        # The first epoch (epoch 0) should use initial_amount
        if current_epoch >= self.epochs_to_ramp:
            return self.final_amount
        
        # Linear interpolation: directly use proportion of completed epochs
        progress = current_epoch / self.epochs_to_ramp
        progress = min(1.0, max(0.0, progress))  # Ensure progress is between [0, 1]
        
        current_amount = self.initial_amount + (self.final_amount - self.initial_amount) * progress
        
        # Safety check: never exceed final_amount regardless of calculation
        current_amount = min(current_amount, self.final_amount)
                
        return current_amount

    def _run_pruning(self, current_epoch: int) -> None:
        """Override to handle scheduled pruning and safety."""
        # Check pruning toggle
        should_prune = self._apply_pruning(current_epoch) if callable(self._apply_pruning) else bool(self._apply_pruning)
        if not should_prune:
            if self.verbose > 1:
                log.debug(f"Epoch {current_epoch}: skip (prune={should_prune})")
            return

        if self.scheduled_pruning:
            # Calculate target sparsity for this epoch
            target_sparsity = self._get_current_amount(current_epoch)
            
            # Get current sparsity
            valid_params = self._get_valid_parameters()
            current_sparsity = self._compute_current_sparsity(valid_params) if valid_params else 0.0
            
            # Don't prune if we're already at or above the target sparsity
            if current_sparsity >= target_sparsity:
                if self.verbose > 0:
                    log.info(
                        f"[Epoch {current_epoch}] | "
                        f"Target sparsity {target_sparsity:.2%} | reached "
                        f"(current {current_sparsity:.2%}), skipping."
                    )
                return
            
            # Calculate the amount to prune from remaining weights to reach target sparsity
            # Formula: amount = (target_sparsity - current_sparsity) / (1 - current_sparsity)
            remaining_weights = 1.0 - current_sparsity
            if remaining_weights > 0:
                prune_amt = (target_sparsity - current_sparsity) / remaining_weights
                # Safety clamp
                prune_amt = min(max(0.0, prune_amt), 0.99)  # Never prune 100% of remaining weights
            else:
                if self.verbose > 0:
                    log.warning(f"Epoch {current_epoch}: no weights left to prune (sparsity={current_sparsity:.4f})")
                return
        else:
            # Fixed pruning
            prune_amt = self.amount

        if self.verbose > 0:
            label = "[Scheduled]" if self.scheduled_pruning else "[Fixed]"
            log.info(
                f"{label} Epoch {current_epoch:>2}: "
                f"prune {prune_amt:.2%} of remaining params"
                + (f"(aiming for total sparsity {target_sparsity:.2%})" if self.scheduled_pruning else "")
            )

        try:
            self.apply_pruning(prune_amt)
        except Exception as e:
            log.error(f"Pruning failed at epoch {current_epoch}: {e}")
            if self.verbose > 0:
                raise

        # lottery ticket hypothesis
        use_lth = self._use_lottery_ticket_hypothesis(current_epoch) if callable(self._use_lottery_ticket_hypothesis) else bool(self._use_lottery_ticket_hypothesis)
        if use_lth:
            self.apply_lottery_ticket_hypothesis()

    def apply_pruning(self, amount: Union[int, float]) -> None:
        """Override with comprehensive safety checks."""
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
        
        try:
            if self.use_global_unstructured:
                pruning_method = self._get_pruning_method()
                pytorch_prune.global_unstructured(
                    self._parameters_to_prune, 
                    pruning_method=pruning_method, 
                    amount=amount
                )
            else:
                super().apply_pruning(amount)
                
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
        
        for module, param_name in self._parameters_to_prune:
            try:
                param = getattr(module, param_name, None)
                
                # Comprehensive validation
                if param is None:
                    self._log_skip(module, param_name, "Parameter doesn't exist")
                elif not isinstance(param, torch.Tensor):
                    self._log_skip(module, param_name, f"Not a tensor (type: {type(param)})")
                elif param.dim() == 0:
                    self._log_skip(module, param_name, "Scalar tensor")
                elif param.numel() == 0:
                    self._log_skip(module, param_name, "Empty tensor")
                else:
                    valid_params.append((module, param_name))
                    
            except Exception as e:
                self._log_skip(module, param_name, f"Error accessing: {str(e)}")
        
        # Report skipped parameters
        if self.skipped_params and self.verbose > 0:
            keys = list(self.skipped_params.keys())
            log.info(
                f"Skipped {len(keys)} invalid parameters for pruning "
                f"(showing up to 5): {keys[:5]}"
            )
        
        self._validated_params_cache = valid_params
        return valid_params

    def _get_pruning_method(self) -> Callable:
        """Resolve pruning method from string or callable."""
        if isinstance(self.pruning_fn, str):
            if not hasattr(pytorch_prune, self.pruning_fn):
                raise ValueError(f"Unknown pruning function: {self.pruning_fn}")
            return getattr(pytorch_prune, self.pruning_fn)
        elif callable(self.pruning_fn):
            return self.pruning_fn
        else:
            raise TypeError(f"Invalid pruning_fn type: {type(self.pruning_fn)}")

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

    def make_pruning_permanent(self, pl_module):
        """
        Permanently remove pruning masks from all validated parameters.
        """
        if not self.should_make_pruning_permanent:
            return

        if self.verbose > 0:
            log.info("Finalizing pruning: removing masks and reparametrizations...")

        valid_params = self._get_valid_parameters()
        if not valid_params:
            log.warning("No pruned parameters found (nothing to finalize).")
            return

        for module, param_name in valid_params:
            try:
                # remove mask/orig and restore parameter
                pytorch_prune.remove(module, param_name)
                if self.verbose > 1:
                    log.debug(f"Mask removed: {module.__class__.__name__}.{param_name}")
            except Exception as e:
                log.warning(
                    f"Could not remove mask for "
                    f"{module.__class__.__name__}.{param_name}: {e}"
                )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Override to add metrics logging."""
        super().on_train_epoch_end(trainer, pl_module)
        
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
                                        "magnitude_pruning/amount": current_pruning_target_amount  # Changed: log current target amount
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

    def get_sparsity_info(self) -> dict:
        """Returns information about the current sparsity of the pruned modules."""
        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params) if valid_params else 0.0
        
        return {
            "current_sparsity": current_sparsity,
            "target_sparsity": self.final_amount if self.scheduled_pruning else self.amount,
            "scheduled_pruning": self.scheduled_pruning
        }
