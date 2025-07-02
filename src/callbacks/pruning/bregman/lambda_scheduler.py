"""
LambdaScheduler for Bregman pruning with sparsity smoothing.

This module provides a lambda scheduler that implements a smoothing mechanism
to handle spurious zero sparsity readings during training.
"""

from typing import Optional, Union, Callable
import logging

log = logging.getLogger(__name__)


class LambdaScheduler:
    """
    Lambda scheduler with sparsity smoothing mechanism.
    
    This scheduler implements a simple smoothing mechanism to handle spurious
    zero sparsity readings that can occur during training when sparsity is
    computed before applying masks.
    
    Parameters
    ----------
    initial_lambda : float
        Initial lambda value for regularization
    target_sparsity : float
        Target sparsity level to achieve
    adjustment_factor : float, default=1.1
        Factor by which to adjust lambda when sparsity is below target
    min_lambda : float, default=1e-6
        Minimum lambda value
    max_lambda : float, default=1e3
        Maximum lambda value
    """
    
    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: float = 0.9,
        adjustment_factor: float = 1.1,
        min_lambda: float = 1e-6,
        max_lambda: float = 1e3
    ):
        self.lambda_value = initial_lambda
        self.target_sparsity = target_sparsity
        self.adjustment_factor = adjustment_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # Sparsity smoothing mechanism
        self._last_sparsity: Optional[float] = None
        
    def step(self, current_sparsity: float) -> float:
        """
        Update lambda based on current sparsity with smoothing.
        
        If current_sparsity is exactly 0.0 but we have a valid last sparsity
        reading, use the last sparsity instead to avoid spurious zero readings.
        
        Parameters
        ----------
        current_sparsity : float
            Current model sparsity (may contain spurious zeros)
            
        Returns
        -------
        float
            Updated lambda value
        """
        # Apply smoothing mechanism for spurious zero readings
        effective_sparsity = self._get_effective_sparsity(current_sparsity)
        
        # Update lambda based on effective sparsity
        if effective_sparsity < self.target_sparsity:
            # Increase lambda to encourage more sparsity
            self.lambda_value *= self.adjustment_factor
        elif effective_sparsity > self.target_sparsity:
            # Decrease lambda since we're above target
            self.lambda_value /= self.adjustment_factor
        
        # Clamp lambda to valid range
        self.lambda_value = max(self.min_lambda, min(self.max_lambda, self.lambda_value))
        
        # Store this sparsity reading if it's valid (not a spurious zero)
        if current_sparsity > 0.0:
            self._last_sparsity = current_sparsity
            
        return self.lambda_value
    
    def _get_effective_sparsity(self, current_sparsity: float) -> float:
        """
        Get effective sparsity using smoothing mechanism.
        
        Parameters
        ----------
        current_sparsity : float
            Raw sparsity reading
            
        Returns
        -------
        float
            Effective sparsity after smoothing
        """
        # If current reading is exactly 0.0 and we have a valid last reading,
        # use the last reading to avoid spurious zeros
        if current_sparsity == 0.0 and self._last_sparsity is not None:
            log.debug(
                f"Spurious zero sparsity detected, using last valid reading: "
                f"{self._last_sparsity:.4f}"
            )
            return self._last_sparsity
        
        return current_sparsity
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value
    
    def reset(self) -> None:
        """Reset the scheduler state."""
        self._last_sparsity = None
        
    def get_state(self) -> dict:
        """Get scheduler state for debugging."""
        return {
            'lambda_value': self.lambda_value,
            'target_sparsity': self.target_sparsity,
            'last_sparsity': self._last_sparsity,
            'adjustment_factor': self.adjustment_factor,
            'min_lambda': self.min_lambda,
            'max_lambda': self.max_lambda
        }