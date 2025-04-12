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
        
        # Store all configuration parameters locally to avoid undefined attributes
        self.pruning_fn = pruning_fn
        
        # For scheduled_pruning, amount is dynamic - store original amount value for reference
        self.original_amount = amount 
        self.amount = amount  # May be overridden by scheduled_pruning
        self.use_global_unstructured = use_global_unstructured
        self.use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis
        self.resample_parameters = resample_parameters
        self.parameters_to_prune = parameters_to_prune
        self.pruning_dim = pruning_dim
        self.pruning_norm = pruning_norm
        self.verbose = verbose
        self.prune_on_train_epoch_end = prune_on_train_epoch_end
        
        # Configure scheduled pruning
        if scheduled_pruning:
            # When using scheduled pruning:
            # - initial_amount is the starting pruning rate
            # - final_amount (if provided) is the ending pruning rate, otherwise use original_amount
            # - amount will be dynamically updated during training
            self.scheduled_pruning = True
            self.initial_amount = initial_amount
            self.final_amount = final_amount if final_amount is not None else self.original_amount
            self.epochs_to_ramp = max(1, epochs_to_ramp)
            
            # Start with initial amount for pruning
            self.amount = self.initial_amount
            log.info(f"Using scheduled pruning: {self.initial_amount:.4f} → {self.final_amount:.4f} "
                     f"over {self.epochs_to_ramp} epochs")
        else:
            # Not using scheduled pruning - amount remains static
            self.scheduled_pruning = False
            self.initial_amount = None
            self.final_amount = None
            self.epochs_to_ramp = None
        
        super().__init__(
            pruning_fn=pruning_fn,
            amount=self.amount,  # Use potentially modified amount
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
        
        # Initialize metrics collection if enabled
        self.collect_metrics = collect_metrics
        self.metrics = defaultdict(list)
        
        # Cache for parameter validation to avoid redundant checks
        self.skipped_params = {}
        self._validated_params_cache = None
        self._current_sparsity = 0.0
        self._last_param_count = 0
        self._last_amount = None  # Initialize to track amount changes
        
        # Add storage for current epoch metrics
        self._current_metrics = {}
        
        # Initialize protection against undefined attributes
        self._prune_kwargs = getattr(self, '_prune_kwargs', {})
        self._parameters_to_prune = getattr(self, '_parameters_to_prune', [])
        
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
                
            if final_amount is not None and initial_amount > final_amount:
                raise ValueError(
                    f"Initial pruning amount ({initial_amount}) cannot be greater than "
                    f"final pruning amount ({final_amount})."
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

    def _apply_global_pruning(self, amount: Union[int, float]) -> None:
        """
        Apply global pruning while safely filtering out non-tensor parameters.
        
        Args:
            amount: Amount of parameters to prune
        """
        # Ensure we have _parameters_to_prune from parent class
        if not hasattr(self, '_parameters_to_prune'):
            self._parameters_to_prune = []
            log.warning("_parameters_to_prune not found, initializing to empty list")
            
        # Ensure we have _prune_kwargs from parent class
        if not hasattr(self, '_prune_kwargs'):
            self._prune_kwargs = {}
            log.warning("_prune_kwargs not found, initializing to empty dict")
        
        # Only invalidate cache if necessary
        should_invalidate_cache = False
        
        # Check if the parameter list has changed
        if hasattr(self, '_last_param_count') and len(self._parameters_to_prune) != self._last_param_count:
            should_invalidate_cache = True
        
        # Check if the amount has changed significantly (for methods sensitive to amount)
        if hasattr(self, '_last_amount') and self._last_amount is not None and abs(amount - self._last_amount) > 1e-5:
            should_invalidate_cache = True
        
        # Reset cache if needed
        if should_invalidate_cache:
            self._validated_params_cache = None
        
        # Update tracking variables
        self._last_param_count = len(self._parameters_to_prune) if self._parameters_to_prune else 0
        self._last_amount = amount
        
        if not self._parameters_to_prune:
            log.warning("No parameters to prune. Skipping pruning step.")
            return
        
        # Use cached validated parameters if available, otherwise build the list
        valid_parameters_to_prune = self._get_valid_parameters()
            
        if not valid_parameters_to_prune:
            log.warning("No valid parameters found for pruning after filtering.")
            return
            
        # Now apply pruning only to valid parameters
        log.info(f"Applying pruning with amount={amount:.4f} to {len(valid_parameters_to_prune)} parameters")
        
        # Resolve the pruning method to use
        try:
            pruning_method = self._resolve_pruning_method()
        except (ValueError, TypeError) as e:
            log.error(f"Error resolving pruning method: {str(e)}")
            raise
        
        try:
            if self.use_global_unstructured:
                # When using global unstructured pruning
                log.info("Using global unstructured pruning")
                pytorch_prune.global_unstructured(
                    valid_parameters_to_prune,
                    pruning_method,
                    amount=amount,
                    **self._prune_kwargs
                )
            else:
                # When not using global unstructured, apply pruning to each parameter individually
                log.info("Applying individual parameter pruning")
                for module, param_name in valid_parameters_to_prune:
                    if self.pruning_dim is not None:
                        pruning_method(module, param_name, amount=amount, dim=self.pruning_dim, **self._prune_kwargs)
                    else:
                        pruning_method(module, param_name, amount=amount, **self._prune_kwargs)
                        
            # After pruning, update the current sparsity estimate
            self._update_sparsity_metrics(valid_parameters_to_prune)
            
        except Exception as e:
            log.error(f"Error during pruning application: {str(e)}")
            raise
    
    def _update_sparsity_metrics(self, parameters: List[Tuple[nn.Module, str]]) -> None:
        """
        Calculate and store current sparsity metrics.
        
        Args:
            parameters: List of (module, parameter_name) tuples to check for sparsity
        """
        if not self.collect_metrics:
            return
            
        total_params = 0
        zero_params = 0
        
        # Current epoch metrics for logging
        current_metrics = {}
        
        for module, param_name in parameters:
            # Check if the module has a mask for this parameter
            mask_name = f"{param_name}_mask"
            
            if hasattr(module, mask_name):
                # This is the correct way to access masks in PyTorch's pruning
                mask = getattr(module, mask_name)
                total_elements = mask.numel()
                zero_elements = total_elements - mask.sum().item()
                
                total_params += total_elements
                zero_params += zero_elements
                
                # Store individual parameter sparsity
                param_id = f"{module.__class__.__name__}.{param_name}"
                sparsity = float(zero_elements) / max(1, total_elements)
                self.metrics[f"sparsity/{param_id}"].append(sparsity)
                current_metrics[f"sparsity/{param_id}"] = sparsity
            else:
                # For parameters without explicit pruning masks, check tensor directly
                param = getattr(module, param_name)
                if isinstance(param, torch.Tensor):
                    tensor_value = param.data
                    total_elements = tensor_value.numel()
                    zero_elements = (tensor_value == 0).sum().item()
                    
                    total_params += total_elements
                    zero_params += zero_elements
                    
                    # Only log if significant zeros exist
                    if zero_elements > 0:
                        param_id = f"{module.__class__.__name__}.{param_name}"
                        sparsity = float(zero_elements) / max(1, total_elements)
                        self.metrics[f"sparsity/{param_id}"].append(sparsity)
                        current_metrics[f"sparsity/{param_id}"] = sparsity
        
        # Calculate global sparsity
        overall_sparsity = float(zero_params) / max(1, total_params) if total_params > 0 else 0
        self._current_sparsity = overall_sparsity
        self.metrics["sparsity/overall"].append(overall_sparsity)
        current_metrics["sparsity/overall"] = overall_sparsity
        
        # Store current metrics for logging
        self._current_metrics = current_metrics
        
        if self.verbose > 0:
            log.info(f"Current model sparsity: {overall_sparsity:.4f}")
    
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
                
                # Parameter is valid for pruning
                valid_parameters_to_prune.append((module, param_name))
                
            except Exception as e:
                self._log_skipped_param(module, param_name, f"Error accessing: {str(e)}")
        
        # Report on skipped parameters if any
        if self.skipped_params and self.verbose > 0:
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
        full_param_name = f"{module_name}.{param_name}"
        
        if full_param_name not in self.skipped_params:
            self.skipped_params[full_param_name] = reason
        
    def _report_skipped_params(self) -> None:
        """Report all parameters that were skipped during pruning in a well-formatted layout."""
        total_skipped = len(self.skipped_params)
        
        # Create a separator line for visual clarity
        separator = "-" * 60
        
        log.warning(f"{separator}")
        log.warning(f"PRUNING SKIPPED PARAMETERS: {total_skipped} parameters excluded")
        log.warning(f"{separator}")
        
        # Group parameters by reason to reduce repetition
        by_reason = defaultdict(list)
        for param_name, reason in self.skipped_params.items():
            by_reason[reason].append(param_name)
        
        # Display parameters grouped by reason
        for i, (reason, params) in enumerate(by_reason.items()):
            if i > 0:
                log.warning("")  # Add separation between reason groups
                
            log.warning(f"• Reason: {reason}")
            # If there are many parameters with the same reason, summarize
            if len(params) > 10:
                for param in params[:5]:
                    log.warning(f"  - {param}")
                log.warning(f"  - ... and {len(params) - 5} more similar parameters")
            else:
                for param in params:
                    log.warning(f"  - {param}")
        
        log.warning(f"{separator}")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> Dict[str, Any]:
        """
        Add information about skipped parameters and metrics to the checkpoint.
        """
        # Call parent's on_save_checkpoint method
        state_dict = super(SafeModelPruning, self).on_save_checkpoint(trainer, pl_module, checkpoint) or {}
        
        # Add our custom state
        state_dict["skipped_params"] = self.skipped_params
        state_dict["metrics"] = dict(self.metrics)
        state_dict["current_sparsity"] = self._current_sparsity
        
        # For scheduled pruning, save the last amount used
        if self.scheduled_pruning:
            state_dict["last_pruning_amount"] = self.get_pruning_amount(trainer.current_epoch)
            
        return state_dict
        
    def on_load_checkpoint(self, trainer, pl_module, callback_state) -> None:
        """
        Load information about skipped parameters from the checkpoint.
        """
        # Call parent's on_load_checkpoint method
        super(SafeModelPruning, self).on_load_checkpoint(trainer, pl_module, callback_state)
        
        if "skipped_params" in callback_state:
            self.skipped_params = callback_state["skipped_params"]
            
        if "metrics" in callback_state:
            self.metrics.update(callback_state["metrics"])
            
        if "current_sparsity" in callback_state:
            self._current_sparsity = callback_state["current_sparsity"]
        
    def setup(self, trainer, pl_module, stage=None):
        """
        Additional setup step to validate parameters before starting training.
        """
        # Call parent's setup method
        super(SafeModelPruning, self).setup(trainer, pl_module, stage)
        
        # Early validation of parameters to provide warnings before training starts
        if self.verbose > 0:
            self.analyze_model_parameters(pl_module)
            
    def analyze_model_parameters(self, pl_module):
        """
        Analyze the model's parameter structure and provide detailed diagnostic information.
        
        Args:
            pl_module: The PyTorch Lightning module
        """
        total_params = 0
        total_elements = 0
        prunable_params = 0
        prunable_elements = 0
        
        # Categorize parameters
        param_stats = {
            'requires_grad': 0,
            'no_grad': 0,
            'scalar': 0,
            'tensor': 0,
            'non_tensor': 0,
            'problematic': 0,
        }
        
        # Track parameter shapes
        shape_counts = defaultdict(int)
        
        log.info("Analyzing model parameters for pruning compatibility...")
        
        for name, param in pl_module.named_parameters():
            total_params += 1
            
            # Count by parameter attributes
            if not isinstance(param, torch.Tensor):
                param_stats['non_tensor'] += 1
                param_stats['problematic'] += 1
                continue
                
            param_stats['tensor'] += 1
            total_elements += param.numel()
            
            if param.requires_grad:
                param_stats['requires_grad'] += 1
            else:
                param_stats['no_grad'] += 1
            
            if param.dim() == 0:
                param_stats['scalar'] += 1
                param_stats['problematic'] += 1
                continue
            
            # Count shape distributions
            shape_str = f"{list(param.shape)}"
            shape_counts[shape_str] += 1
            
            # Count prunable parameters
            prunable_params += 1
            prunable_elements += param.numel()
        
        # Parameters identified by pruning
        identified_for_pruning = len(self._parameters_to_prune) if hasattr(self, '_parameters_to_prune') else 0
        valid_for_pruning = len(self._get_valid_parameters())
        
        # Print detailed analysis
        log.info(f"Parameter Analysis Summary:")
        log.info(f"  - Total parameters: {total_params} ({total_elements:,} elements)")
        log.info(f"  - Prunable parameters: {prunable_params} ({prunable_elements:,} elements)")
        log.info(f"  - Identified for pruning: {identified_for_pruning}")
        log.info(f"  - Valid for pruning after filtering: {valid_for_pruning}")
        log.info(f"  - Skipped parameters: {len(self.skipped_params)}")
        
        if valid_for_pruning < prunable_params:
            log.warning(f"Only {valid_for_pruning}/{prunable_params} potential parameters will be pruned!")
            log.warning(f"This may result in lower than expected sparsity.")
        
        log.info(f"Parameter categories:")
        log.info(f"  - Tensor parameters: {param_stats['tensor']}")
        log.info(f"  - Non-tensor parameters: {param_stats['non_tensor']}")
        log.info(f"  - With gradients: {param_stats['requires_grad']}")
        log.info(f"  - Without gradients: {param_stats['no_grad']}")
        log.info(f"  - Scalar parameters: {param_stats['scalar']}")
        log.info(f"  - Problematic parameters: {param_stats['problematic']}")
        
        # Show most common shapes
        if self.verbose > 1:
            log.info("Most common parameter shapes:")
            for shape, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                log.info(f"  - {shape}: {count} parameters")
                
        return {
            'total_params': total_params,
            'prunable_params': prunable_params,
            'valid_for_pruning': valid_for_pruning,
            'coverage': valid_for_pruning / max(1, prunable_params)
        }
        
    def on_train_start(self, trainer, pl_module):
        """
        Hook before training starts to ensure all parameter information is up to date.
        """
        super().on_train_start(trainer, pl_module)
        
        # If first time, re-analyze to make sure we have all parameters
        if not hasattr(self, '_analysis_complete'):
            stats = self.analyze_model_parameters(pl_module)
            self._analysis_complete = True
            
            # If low coverage, provide clear warning
            if stats['coverage'] < 0.8:  # Less than 80% coverage
                log.warning(f"LOW PRUNING COVERAGE: Only {stats['valid_for_pruning']} out of {stats['prunable_params']} "
                           f"parameters ({stats['coverage']:.1%}) will be pruned.")
                log.warning("This will result in lower than expected overall sparsity!")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Only call parent implementation without resetting cache every epoch.
        """
        # Directly call the parent method using proper syntax
        super(SafeModelPruning, self).on_train_epoch_start(trainer, pl_module)
        
    def get_model_sparsity(self) -> float:
        """
        Get the current model sparsity level.
        
        Returns:
            float: The current overall model sparsity (0.0 to 1.0)
        """
        return self._current_sparsity
