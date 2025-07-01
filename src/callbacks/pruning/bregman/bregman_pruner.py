"""Bregman Pruner implementation with consistent sparsity calculation."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import List, Tuple, Optional
from pytorch_lightning.callbacks import Callback

from src import utils

log = utils.get_pylogger(__name__)


class BregmanPruner(Callback):
    """
    Bregman Pruner callback for neural network pruning.
    
    This implementation uses the same sparsity calculation logic as SafeModelPruning
    to ensure consistent sparsity readings with PyTorch pruning masks.
    """
    
    def __init__(
        self,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize BregmanPruner.
        
        Args:
            parameters_to_prune: List of (module, parameter_name) tuples to prune
            verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.parameters_to_prune = parameters_to_prune or []
        self.verbose = verbose
        
    def _compute_sparsity(self, parameters: List[Tuple[nn.Module, str]]) -> float:
        """
        Compute current sparsity using PyTorch's actual built-in utilities.
        
        This method follows the same logic as SafeModelPruning._compute_current_sparsity:
        1. For each target module and parameter name, check if the module is pruned 
           via torch.nn.utils.prune.is_pruned(module)
        2. If pruned, use the associated <param_name>_mask tensor to count pruned 
           entries (mask == 0)
        3. If not pruned or no mask found, fall back to counting natural zeros 
           in the parameter tensor
           
        Args:
            parameters: List of (module, parameter_name) tuples
            
        Returns:
            float: Sparsity ratio (pruned_params / total_params)
        """
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
        """
        Fallback sparsity computation using direct parameter inspection.
        
        Args:
            parameters: List of (module, parameter_name) tuples
            
        Returns:
            float: Sparsity ratio (pruned_params / total_params)
        """
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
    
    def get_current_sparsity(self) -> float:
        """
        Get current sparsity for the configured parameters.
        
        Returns:
            float: Current sparsity ratio
        """
        return self._compute_sparsity(self.parameters_to_prune)
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Called at the end of each training epoch."""
        if self.verbose > 0:
            current_sparsity = self.get_current_sparsity()
            log.info(f"Epoch {trainer.current_epoch}: Current sparsity: {current_sparsity:.4f}")