"""
BregmanPruner - Callback for Bregman divergence based sparse regularization.

This callback implements structured sparsity using Bregman regularization that encourages
specific sparsity targets through adaptive lambda scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import warnings
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import logging

log = logging.getLogger(__name__)


class BregmanPruner(Callback):
    """
    Callback for Bregman divergence based sparse regularization.
    
    This callback applies Bregman regularization to encourage sparsity in model parameters.
    It dynamically adjusts the regularization strength (lambda) based on the current
    sparsity level compared to a target sparsity.
    
    Parameters
    ----------
    target_sparsity : float
        Target sparsity ratio (between 0 and 1)
    initial_lambda : float, default=1.0
        Initial regularization strength
    lambda_update_rate : float, default=0.001
        Rate at which lambda is updated
    update_frequency : int, default=100
        Number of steps between lambda updates
    parameters_to_prune : Optional[List[Tuple[nn.Module, str]]], default=None
        Specific parameters to apply regularization to
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed)
    """
    
    def __init__(
        self,
        target_sparsity: float = 0.8,
        initial_lambda: float = 1.0,
        lambda_update_rate: float = 0.001,
        update_frequency: int = 100,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        verbose: int = 0,
    ):
        super().__init__()
        
        if not 0 <= target_sparsity <= 1:
            raise ValueError(f"target_sparsity must be between 0 and 1, got {target_sparsity}")
        
        self.target_sparsity = target_sparsity
        self.current_lambda = initial_lambda
        self.lambda_update_rate = lambda_update_rate
        self.update_frequency = update_frequency
        self.parameters_to_prune = parameters_to_prune
        self.verbose = verbose
        
        # Internal state
        self._step_count = 0
        self._validated_params_cache = None
        self._last_computed_sparsity = 0.0
        
    def setup(self, trainer, pl_module, stage: str) -> None:
        """Setup callback by validating parameters."""
        if stage == "fit":
            self._validate_and_cache_parameters(pl_module)
    
    def _validate_and_cache_parameters(self, pl_module) -> List[Tuple[nn.Module, str]]:
        """Validate and cache parameters to regularize."""
        if self._validated_params_cache is not None:
            return self._validated_params_cache
            
        valid_params = []
        
        if self.parameters_to_prune is None:
            # Auto-discover parameters (typically weight parameters in Conv2d, Linear, etc.)
            for name, module in pl_module.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                    if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                        valid_params.append((module, 'weight'))
        else:
            # Use specified parameters
            for module, param_name in self.parameters_to_prune:
                param = getattr(module, param_name, None)
                if param is not None and isinstance(param, torch.Tensor):
                    valid_params.append((module, param_name))
                elif self.verbose > 0:
                    log.warning(f"Parameter {param_name} not found in module {module}")
        
        self._validated_params_cache = valid_params
        
        if self.verbose > 0:
            log.info(f"BregmanPruner initialized with {len(valid_params)} parameters")
            
        return valid_params
    
    def _compute_current_sparsity(self, pl_module) -> float:
        """
        Compute current sparsity across all tracked parameters.
        
        This method provides a consistent sparsity calculation that accounts for:
        1. PyTorch pruning masks (if present)
        2. Natural zeros in parameters
        3. Proper parameter counting
        """
        valid_params = self._validate_and_cache_parameters(pl_module)
        
        if not valid_params:
            return 0.0
            
        total_params = 0
        zero_params = 0
        
        for module, param_name in valid_params:
            try:
                # Check if module has pruning masks (from PyTorch pruning)
                if pytorch_prune.is_pruned(module):
                    mask_name = f"{param_name}_mask"
                    if hasattr(module, mask_name):
                        mask = getattr(module, mask_name)
                        if isinstance(mask, torch.Tensor):
                            total_params += mask.numel()
                            # In PyTorch pruning: mask == 0 means pruned
                            zero_params += (mask == 0).sum().item()
                            continue
                
                # For non-pruned parameters, check the parameter directly
                param = getattr(module, param_name, None)
                if isinstance(param, torch.Tensor):
                    total_params += param.numel()
                    # Count actual zeros in the parameter
                    zero_params += (param == 0).sum().item()
                    
            except Exception as e:
                if self.verbose > 1:
                    log.debug(f"Error computing sparsity for {module.__class__.__name__}.{param_name}: {e}")
                continue
        
        if total_params == 0:
            return 0.0
            
        sparsity = zero_params / total_params
        return sparsity
    
    def _update_lambda(self, current_sparsity: float) -> None:
        """Update lambda based on current vs target sparsity."""
        sparsity_diff = self.target_sparsity - current_sparsity
        
        # Increase lambda if below target, decrease if above target
        lambda_delta = sparsity_diff * self.lambda_update_rate
        new_lambda = self.current_lambda + lambda_delta
        
        # Keep lambda positive and reasonable
        new_lambda = max(0.0001, min(new_lambda, 100.0))
        
        if self.verbose > 0 and abs(new_lambda - self.current_lambda) > 1e-6:
            direction = "↗" if new_lambda > self.current_lambda else "↘"
            log.info(
                f"Sparsity {current_sparsity:.3f}% vs target {self.target_sparsity:.1f}% → "
                f"Lambda {direction} {self.current_lambda:.8f} → {new_lambda:.8f}"
            )
        
        self.current_lambda = new_lambda
    
    def _apply_bregman_regularization(self, pl_module) -> torch.Tensor:
        """Apply Bregman regularization to encourage sparsity."""
        valid_params = self._validate_and_cache_parameters(pl_module)
        
        if not valid_params:
            return torch.tensor(0.0, device=next(pl_module.parameters()).device)
        
        total_reg_loss = torch.tensor(0.0, device=next(pl_module.parameters()).device)
        
        for module, param_name in valid_params:
            try:
                # Get the actual parameter (accounting for pruning)
                if pytorch_prune.is_pruned(module):
                    # For pruned modules, get the original parameter
                    orig_param_name = f"{param_name}_orig"
                    if hasattr(module, orig_param_name):
                        param = getattr(module, orig_param_name)
                    else:
                        param = getattr(module, param_name)
                else:
                    param = getattr(module, param_name)
                
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    # Apply L1 regularization (encourages sparsity)
                    reg_loss = torch.abs(param).sum()
                    total_reg_loss += reg_loss
                    
            except Exception as e:
                if self.verbose > 1:
                    log.debug(f"Error applying regularization to {module.__class__.__name__}.{param_name}: {e}")
                continue
        
        return self.current_lambda * total_reg_loss
    
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update lambda and log metrics every update_frequency steps."""
        self._step_count += 1
        
        if self._step_count % self.update_frequency == 0:
            # Compute current sparsity
            current_sparsity = self._compute_current_sparsity(pl_module)
            self._last_computed_sparsity = current_sparsity
            
            # Update lambda based on sparsity
            self._update_lambda(current_sparsity)
            
            # Log detailed metrics
            if self.verbose > 0:
                log.info(f"Step {self._step_count}: Sparsity={current_sparsity:.3f}%, lambda={self.current_lambda:.4f}")
            
            # Log to trainer if available
            if trainer.logger is not None:
                trainer.logger.log_metrics({
                    "bregman/sparsity": current_sparsity,
                    "bregman/lambda": self.current_lambda,
                    "bregman/target_sparsity": self.target_sparsity,
                }, step=trainer.global_step)
    
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Log sparsity at epoch start."""
        current_sparsity = self._compute_current_sparsity(pl_module)
        self._last_computed_sparsity = current_sparsity
        
        if self.verbose > 0:
            log.info(f"Epoch {trainer.current_epoch}: Sparsity of pruned modules = {current_sparsity:.3f}%")
    
    def get_regularization_loss(self, pl_module) -> torch.Tensor:
        """
        Get the current Bregman regularization loss.
        
        This method should be called from the model's training_step to add
        the regularization loss to the total loss.
        """
        return self._apply_bregman_regularization(pl_module)
    
    def get_current_sparsity(self) -> float:
        """Get the last computed sparsity value."""
        return self._last_computed_sparsity
    
    def get_current_lambda(self) -> float:
        """Get the current lambda value."""
        return self.current_lambda