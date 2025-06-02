from pytorch_lightning.callbacks.pruning import ModelPruning
import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import List, Tuple, Callable, Optional, Union, Dict, Any, Set
import warnings
import logging
from collections import defaultdict
import numpy as np

from src import utils

log = utils.get_pylogger(__name__)


class SafeModelPruning(ModelPruning):
    """
    SafeModelPruning - Enhanced version of PyTorch Lightning's ModelPruning callback
    This callback extends PyTorch Lightning's ModelPruning with safety features
    to prevent common errors like the 'float object has no attribute device' error
    that occurs when non-tensor parameters are included in pruning. It also adds:
    - Parameter validation to identify and skip problematic parameters
    - Scheduled pruning with customizable ramping
    - Detailed metrics collection and reporting
    - Better error handling and debug information

    Parameters
    ----------
    pruning_fn : Union[Callable, str], default="l1_unstructured"
        The pruning function or name of function from torch.nn.utils.prune.
        Common options include 'l1_unstructured', 'random_unstructured'.
    amount : Union[int, float, List[Union[int, float]]], default=0.5
        Amount of parameters to prune. If float between 0 and 1, interpreted as
        fraction of parameters to prune. If int, interpreted as absolute number.
    use_global_unstructured : bool, default=True
        Whether to apply pruning globally across all parameters (True) or
        individually per parameter (False).
    apply_pruning : bool, default=True
        Whether to actually apply pruning or just create pruning method.
        This controls the initial pruning state.
    make_pruning_permanent : bool, default=True
        Whether to permanently remove pruned weights after training.
        This functionality is implemented in the parent ModelPruning class
        and is applied at the end of training.
    use_lottery_ticket_hypothesis : bool, default=False
        If True, reset remaining weights to their original values as per the
        lottery ticket hypothesis.
    resample_parameters : bool, default=False
        If True, new weights are resampled at each pruning step.
    parameters_to_prune : Optional[List[Tuple[nn.Module, str]]], default=None
        List of (module, parameter_name) tuples specifying which parameters to prune.
        If None, all model parameters will be examined.
    pruning_dim : Optional[int], default=None
        Dimension along which to prune. If None, pruning is applied across all dims.
    pruning_norm : Optional[int], default=None
        Norm to use for structured pruning methods.
    verbose : int, default=0
        Verbosity level controlling logging detail:
          0: Silent mode - only critical errors are shown
          1: Basic mode - show pruning progress and overall sparsity
          2: Detailed mode - show pruning statistics, parameter counts, and warnings
    prune_on_train_epoch_end : bool, default=True
        Whether to apply pruning at the end of each training epoch.
    scheduled_pruning : bool, default=False
        Whether to use scheduled pruning with increasing sparsity over time.
    initial_amount : float, default=0.0
        Starting pruning rate when using scheduled_pruning.
    final_amount : Optional[float], default=None
        Final pruning amount when using scheduled_pruning. If None, uses `amount`.
    epochs_to_ramp : int, default=10
        Number of epochs over which to linearly increase pruning from initial_amount
        to final_amount when using scheduled_pruning.
    collect_metrics : bool, default=False
        Whether to collect and log detailed sparsity metrics during training.

    Examples
    --------
    >>> # Basic usage
    >>> pruning_callback = SafeModelPruning(amount=0.5)
    >>>
    >>> # With scheduled pruning
    >>> pruning_callback = SafeModelPruning(
    >>>     scheduled_pruning=True,
    >>>     initial_amount=0.1,
    >>>     final_amount=0.8,
    >>>     epochs_to_ramp=5,
    >>>     collect_metrics=True
    >>> )
    >>>
    >>> # Add to PyTorch Lightning trainer
    >>> trainer = Trainer(callbacks=[pruning_callback])
    
    Notes
    -----
    - When using scheduled_pruning, the amount parameter will be overridden by the
      calculated amount based on the current epoch.
    - For best performance, specify exact parameters_to_prune rather than allowing
      the callback to discover them dynamically.
    - Set verbose > 0 to get detailed information about skipped parameters.
    """
    
    def __init__(
        self,
        pruning_fn: Union[Callable, str] = "l1_unstructured",
        amount: Union[int, float, List[Union[int, float]]] = 0.5,
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
        # Validate inputs first
        self._validate_init_params(amount, use_lottery_ticket_hypothesis, make_pruning_permanent,
                                 scheduled_pruning, initial_amount, final_amount)
        
        # Configure scheduled pruning BEFORE calling parent constructor
        if scheduled_pruning:
            self.scheduled_pruning = True
            self.initial_amount = initial_amount
            self.final_amount = final_amount if final_amount is not None else amount
            self.epochs_to_ramp = max(1, epochs_to_ramp)
            # Start with initial amount for the parent constructor
            amount = self.initial_amount
            log.info(f"Using scheduled pruning: {self.initial_amount:.4f} → {self.final_amount:.4f} "
                     f"over {self.epochs_to_ramp} epochs")
        else:
            self.scheduled_pruning = False
            self.initial_amount = None
            self.final_amount = None
            self.epochs_to_ramp = None
        
        # Call parent constructor with potentially modified amount
        super().__init__(
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
            prune_on_train_epoch_end=prune_on_train_epoch_end,
            **kwargs
        )
        
        # Initialize metrics collection
        self.collect_metrics = collect_metrics
        self.metrics = defaultdict(list)
        
        # Cache for parameter validation
        self.skipped_params = {}
        self._validated_params_cache = None
        self._current_sparsity = 0.0
        self._current_metrics = {}

    def _validate_init_params(self, amount, use_lottery_ticket_hypothesis, 
                            make_pruning_permanent, scheduled_pruning,
                            initial_amount, final_amount):
        """Validate initialization parameters to catch configuration issues early."""
        # Check for conflicting options
        if use_lottery_ticket_hypothesis and make_pruning_permanent:
            warnings.warn(
                "Using lottery ticket hypothesis with permanent pruning may not work as expected. "
                "The lottery ticket hypothesis typically requires reverting weights to their original values."
            )
            
        # Validate amount is in proper range for percentage pruning
        if isinstance(amount, float) and not (0 <= amount <= 1):
            raise ValueError(f"Pruning amount {amount} must be between 0 and 1 when specified as a float.")
            
        # Validate scheduled pruning parameters
        if scheduled_pruning:
            if not (0 <= initial_amount <= 1):
                raise ValueError(f"Initial pruning amount {initial_amount} must be between 0 and 1.")
            
            if final_amount is not None and not (0 <= final_amount <= 1):
                raise ValueError(f"Final pruning amount {final_amount} must be between 0 and 1.")
                
            if final_amount is not None and initial_amount >= final_amount:
                raise ValueError(
                    f"Initial pruning amount ({initial_amount}) must be less than "
                    f"final pruning amount ({final_amount}) for scheduled pruning."
                )

    def get_pruning_amount(self, current_epoch: int) -> float:
        """
        Calculate the current pruning amount based on scheduling.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            float: Current pruning amount
        """
        if not self.scheduled_pruning:
            return self.amount  # Static amount when not using scheduled pruning
        
        # Linear ramp from initial to final amount over epochs_to_ramp
        if current_epoch >= self.epochs_to_ramp:
            return self.final_amount
        else:
            # Linear interpolation
            progress = current_epoch / self.epochs_to_ramp
            return self.initial_amount + (self.final_amount - self.initial_amount) * progress

    def _run_pruning(self, current_epoch: int) -> None:
        """Override parent's _run_pruning to handle scheduled pruning."""
        # Get the pruning decision function/value from parent class
        # This can be either:
        # - A boolean: True/False for always/never prune
        # - A callable: function(epoch) -> bool for dynamic decisions
        apply_pruning_attr = getattr(self, '_apply_pruning', True)
        
        # Determine if we should prune this epoch
        if callable(apply_pruning_attr):
            # Dynamic: call the function with current epoch
            should_prune = apply_pruning_attr(current_epoch)
        else:
            # Static: use the boolean value directly
            should_prune = bool(apply_pruning_attr)
        
        # Calculate current pruning amount (handles scheduled pruning)
        if self.scheduled_pruning:
            amount = self.get_pruning_amount(current_epoch)
            # Update the parent's amount attribute for consistency
            self.amount = amount
            verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
            if verbose_level > 0:
                log.info(f"Epoch {current_epoch}: Applying pruning with amount={amount:.4f}")
        else:
            # Handle both callable and non-callable amount (edge case)
            if callable(self.amount):
                amount = self.amount(current_epoch)
            else:
                amount = self.amount
        
        # Early exit if we shouldn't prune or amount is zero
        if not should_prune or not amount:
            verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
            if verbose_level > 1:
                log.debug(f"Epoch {current_epoch}: Skipping pruning (should_prune={should_prune}, amount={amount})")
            return
            
        # Apply the actual pruning
        self.apply_pruning(amount)

        # Handle lottery ticket hypothesis if enabled
        use_lth = getattr(self, '_use_lottery_ticket_hypothesis', False)
        if callable(use_lth):
            should_use_lth = use_lth(current_epoch)
        else:
            should_use_lth = bool(use_lth)
            
        if should_use_lth:
            self.apply_lottery_ticket_hypothesis()

    def apply_pruning(self, amount: Union[int, float]) -> None:
        """Override parent's apply_pruning to add safety checks."""
        # Invalidate cache since model structure might have changed
        self._validated_params_cache = None
        
        # Get valid parameters for pruning before calling parent
        valid_parameters = self._get_valid_parameters()
        
        if not valid_parameters:
            log.warning("No valid parameters found for pruning after filtering.")
            return
            
        # Temporarily replace parameters list with filtered version
        original_params = self._parameters_to_prune
        self._parameters_to_prune = valid_parameters
        
        try:
            # Fix: Call the correct parent method for applying pruning
            if self.use_global_unstructured:
                self._apply_global_pruning(amount)
            else:
                # Apply pruning individually to each parameter
                for module, param_name in valid_parameters:
                    pruning_method = self._resolve_pruning_method()
                    pruning_method(module, param_name, amount=amount)
        finally:
            # Always restore original parameters list
            self._parameters_to_prune = original_params
            
        # Update our custom metrics
        if self.collect_metrics:
            self._update_sparsity_metrics(valid_parameters)
            
        # Debug sparsity calculation discrepancy
        verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
        if verbose_level > 1:
            self._debug_sparsity_calculation()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Override to add sparsity debugging at epoch end."""
        # Fix: Call parent method correctly - it expects trainer and pl_module
        super().on_train_epoch_end(trainer, pl_module)
        
        # Store trainer reference for debugging
        self._trainer = trainer
        
        # Log comprehensive sparsity info if collect_metrics is enabled
        if self.collect_metrics:
            verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
            if verbose_level > 0 and hasattr(pl_module, 'get_model_sparsity_info'):
                sparsity_info = pl_module.get_model_sparsity_info()
                
                # Fix: Use proper logger access pattern
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    # Handle both single logger and logger collection
                    loggers = trainer.logger.loggers if hasattr(trainer.logger, 'loggers') else [trainer.logger]
                    
                    for logger in loggers:
                        try:
                            if hasattr(logger, 'log_metrics'):
                                logger.log_metrics({
                                    'pruning/model_sparsity': sparsity_info['overall_sparsity'],
                                    'pruning/total_parameters': sparsity_info['total_parameters'],
                                    'pruning/pruned_parameters': sparsity_info['pruned_parameters'],
                                    'pruning/modules_with_masks': len(sparsity_info['modules_with_masks']),
                                    'pruning/modules_without_masks': len(sparsity_info['modules_without_masks'])
                                }, step=trainer.global_step)
                                break  # Successfully logged, exit loop
                        except Exception as e:
                            log.debug(f"Failed to log to {type(logger).__name__}: {e}")
                            continue

    def _apply_global_pruning(self, amount: float) -> None:
        """Apply global pruning using PyTorch's global_unstructured method."""
        try:
            # Resolve pruning method to ensure it's callable
            pruning_method = self._resolve_pruning_method()
            pytorch_prune.global_unstructured(
                self._parameters_to_prune, 
                pruning_method=pruning_method, 
                amount=amount
            )
        except Exception as e:
            log.error(f"Failed to apply global pruning: {e}")
            raise

    def _update_sparsity_metrics(self, parameters: List[Tuple[nn.Module, str]]) -> None:
        """
        Calculate and store current sparsity metrics using PyTorch's pruning methodology.
        
        Args:
            parameters: List of (module, parameter_name) tuples to check for sparsity
        """
        if not self.collect_metrics:
            return
            
        total_params = 0
        zero_params = 0
        
        # Current epoch metrics for logging
        current_metrics = {}
        
        # Track parameters with and without masks for debugging
        params_with_masks = 0
        params_without_masks = 0
        
        for module, param_name in parameters:
            try:
                # Get the original parameter
                param = getattr(module, param_name, None)
                if param is None or not isinstance(param, torch.Tensor):
                    continue
                    
                # Check if the module has a pruning mask for this parameter
                mask_name = f"{param_name}_mask"
                
                if hasattr(module, mask_name):
                    # Use the mask to calculate sparsity (this is the correct approach)
                    mask = getattr(module, mask_name)
                    if isinstance(mask, torch.Tensor):
                        total_elements = mask.numel()
                        # In PyTorch pruning, 0 in mask means pruned, 1 means kept
                        kept_elements = mask.sum().item()
                        zero_elements = total_elements - kept_elements
                        
                        params_with_masks += 1
                        
                        total_params += total_elements
                        zero_params += zero_elements
                        
                        # Store individual parameter sparsity
                        param_id = f"{module.__class__.__name__}.{param_name}"
                        sparsity = float(zero_elements) / max(1, total_elements)
                        self.metrics[f"sparsity/{param_id}"].append(sparsity)
                        current_metrics[f"sparsity/{param_id}"] = sparsity
                else:
                    # For parameters without pruning masks, they shouldn't contribute to sparsity
                    params_without_masks += 1
                    
                    # Log natural sparsity for debugging if verbose
                    tensor_value = param.data
                    total_elements = tensor_value.numel()
                    natural_zeros = (tensor_value == 0).sum().item()
                    
                    if natural_zeros > 0:
                        param_id = f"{module.__class__.__name__}.{param_name}"
                        natural_sparsity = float(natural_zeros) / max(1, total_elements)
                        verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
                        if verbose_level > 1:
                            log.debug(f"Parameter {param_id} has {natural_sparsity:.4f} natural sparsity (no pruning mask)")
                            
            except Exception as e:
                log.warning(f"Error processing parameter {module.__class__.__name__}.{param_name}: {e}")
                continue
        
        # Calculate global sparsity (only from parameters with pruning masks)
        overall_sparsity = float(zero_params) / max(1, total_params) if total_params > 0 else 0
        self._current_sparsity = overall_sparsity
        self.metrics["sparsity/overall"].append(overall_sparsity)
        current_metrics["sparsity/overall"] = overall_sparsity
        
        # Store current metrics for logging
        self._current_metrics = current_metrics
        
        verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
        if verbose_level > 0:
            log.info(f"Current model sparsity: {overall_sparsity:.4f} "
                    f"({zero_params}/{total_params} parameters)")
            if verbose_level > 1:
                log.debug(f"Parameters with masks: {params_with_masks}, "
                         f"without masks: {params_without_masks}")

    def _get_valid_parameters(self) -> List[Tuple[nn.Module, str]]:
        """
        Get a filtered list of valid parameters for pruning, using caching for performance.
        
        Returns:
            List of valid (module, parameter_name) tuples
        """
        # Use cached result if available and parameters haven't changed
        if self._validated_params_cache is not None:
            return self._validated_params_cache
            
        # Reset skipped params for a fresh validation
        self.skipped_params = {}
        valid_parameters_to_prune = []
        
        # Fix: Ensure _parameters_to_prune exists
        if not hasattr(self, '_parameters_to_prune') or self._parameters_to_prune is None:
            log.warning("No parameters specified for pruning. Using all model parameters.")
            # This would need to be set elsewhere, but we can't auto-discover here
            # without the model reference
            return []
        
        for module, param_name in self._parameters_to_prune:
            try:
                param = getattr(module, param_name, None)
                
                # Check if parameter exists
                if param is None:
                    self._log_skipped_param(module, param_name, "Parameter doesn't exist")
                    continue
                    
                # Check if parameter is a tensor
                if not isinstance(param, torch.Tensor):
                    self._log_skipped_param(module, param_name, f"Not a tensor (type: {type(param)})")
                    continue
                    
                # Basic checks for prunable tensor
                if param.dim() == 0:  # Skip scalar tensors
                    self._log_skipped_param(module, param_name, "Scalar tensor cannot be pruned")
                    continue
                    
                # Check if parameter requires gradient (optional - frozen params might still be prunable)
                if not param.requires_grad:
                    verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
                    if verbose_level > 1:
                        log.debug(f"Parameter {module.__class__.__name__}.{param_name} doesn't require grad but will be pruned")
                
                # Parameter is valid for pruning
                valid_parameters_to_prune.append((module, param_name))
                
            except Exception as e:
                self._log_skipped_param(module, param_name, f"Error accessing: {str(e)}")
        
        # Report on skipped parameters if any
        verbose_level = getattr(self, '_verbose', getattr(self, 'verbose', 0))
        if self.skipped_params and verbose_level > 0:
            self._report_skipped_params()
        
        # Cache the result
        self._validated_params_cache = valid_parameters_to_prune
        return valid_parameters_to_prune

    def _resolve_pruning_method(self) -> Callable:
        """
        Resolve the pruning method from the pruning_fn parameter.
        
        Returns:
            Callable: The resolved pruning method
        """
        if isinstance(self.pruning_fn, str):
            # Check if the string is a built-in PyTorch pruning method
            if hasattr(pytorch_prune, self.pruning_fn):
                return getattr(pytorch_prune, self.pruning_fn)
            else:
                raise ValueError(f"Pruning function '{self.pruning_fn}' not found in torch.nn.utils.prune")
        elif callable(self.pruning_fn):
            # If it's already a callable, return it directly
            return self.pruning_fn
        else:
            raise TypeError(f"Expected pruning_fn to be a string or callable, got {type(self.pruning_fn)}")
            
    def _log_skipped_param(self, module: nn.Module, param_name: str, reason: str) -> None:
        """
        Log information about parameters skipped during pruning.
        
        Args:
            module: The module containing the parameter
            param_name: Name of the parameter
            reason: Reason for skipping
        """
        module_name = module.__class__.__name__
        param_key = f"{module_name}.{param_name}"
        
        if param_key not in self.skipped_params:
            self.skipped_params[param_key] = reason
            
    def _report_skipped_params(self) -> None:
        """Report summary of skipped parameters."""
        if not self.skipped_params:
            return
            
        log.info(f"Skipped {len(self.skipped_params)} parameters during pruning:")
        for param_key, reason in self.skipped_params.items():
            log.info(f"  {param_key}: {reason}")

    def _debug_sparsity_calculation(self) -> None:
        """Debug sparsity calculations by comparing different methods."""
        try:
            # Get the model from trainer if available
            if hasattr(self, '_trainer') and self._trainer is not None:
                model = self._trainer.lightning_module
                
                # Check if model has the sparsity info method
                if hasattr(model, 'get_model_sparsity_info'):
                    sparsity_info = model.get_model_sparsity_info()
                    
                    log.debug("=== SPARSITY DEBUG COMPARISON ===")
                    log.debug(f"Custom calculation: {self._current_sparsity:.4f}")
                    log.debug(f"Model sparsity info: {sparsity_info['overall_sparsity']:.4f}")
                    log.debug(f"Pruned parameters: {sparsity_info['pruned_parameters']}/{sparsity_info['total_parameters']}")
                    log.debug(f"Modules with masks: {len(sparsity_info['modules_with_masks'])}")
                    log.debug(f"Modules without masks: {len(sparsity_info['modules_without_masks'])}")
                    
                    # Show top 5 sparsest modules
                    if sparsity_info['modules_with_masks']:
                        sorted_modules = sorted(sparsity_info['modules_with_masks'], 
                                              key=lambda x: x['sparsity'], reverse=True)
                        log.debug("Top 5 sparsest modules:")
                        for i, module_info in enumerate(sorted_modules[:5]):
                            log.debug(f"  {i+1}. {module_info['module']}: {module_info['sparsity']:.4f}")
                    
                    log.debug("=== END SPARSITY DEBUG ===")
                    
        except Exception as e:
            log.warning(f"Error in sparsity debug comparison: {str(e)}")
