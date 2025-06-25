"""
Lambda scheduler for Bregman regularization parameter adjustment.
"""
import torch
from typing import Optional
from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """
    Scheduler for the regularization parameter 'lamda' in the optimizer's regularizer.
    """
    def __init__(self, optimizer, warmup=0, increment=0.05, cooldown=0, target_sparsity=0.9, reg_param="lamda", log_frequency=100):
        self.optimizer = optimizer
        self.warmup = warmup
        self.increment = increment
        self.cooldown = cooldown
        self.cooldown_val = cooldown
        self.target_sparse = target_sparsity
        self.reg_param = reg_param
        self.log_frequency = log_frequency
        self._log_counter = 0
        
        # Validate optimizer on initialization
        self._validate_optimizer()

    def _validate_optimizer(self):
        """Validate that the optimizer is compatible with this scheduler."""
        if self.optimizer is None:
            raise ValueError("No optimizer provided to LambdaScheduler")
            
        if not hasattr(self.optimizer, 'param_groups') or not self.optimizer.param_groups:
            raise ValueError("Optimizer must have param_groups")
            
        # Check that all parameter groups have regularizers
        for i, group in enumerate(self.optimizer.param_groups):
            if 'reg' not in group or group['reg'] is None:
                raise ValueError(f"Parameter group {i} has no regularizer ('reg' field)")
                
            reg = group['reg']
            if not hasattr(reg, self.reg_param):
                raise ValueError(
                    f"Regularizer in parameter group {i} does not have parameter '{self.reg_param}'"
                )

    def step(self, current_sparsity):
        """
        Perform a scheduling step.
        
        Args:
            current_sparsity: Current sparsity level
            
        Returns:
            Updated regularization parameter value
        """
        # Warmup phase: do nothing but decrement the counter
        if self.warmup > 0:
            self.warmup -= 1
            return self._get_current_param()

        elif self.warmup == 0:
            self.warmup = -1
            return self._get_current_param()

        else:
            # Cooldown phase: wait before next update
            if self.cooldown_val > 0:
                self.cooldown_val -= 1
                return self._get_current_param()
            else:
                self.cooldown_val = self.cooldown
                new_param = None
                self._log_counter += 1
                should_log = self._log_counter % self.log_frequency == 0
                
                for group in self.optimizer.param_groups:
                    reg = group['reg']
                    old_param = getattr(reg, self.reg_param)

                    # Update the regularization parameter according to target sparsity
                    if current_sparsity < self.target_sparse:
                        # Not sparse enough - increase regularization strength
                        new_param = old_param + self.increment
                        setattr(reg, self.reg_param, new_param)
                        if should_log:
                            log.info(f"Sparsity {current_sparsity:.3%} < target {self.target_sparse:.1%} → Lambda {old_param:.8f} → {new_param:.8f}")
                    else:
                        # Too sparse - decrease regularization strength
                        new_param = max(old_param - self.increment, 0.0)
                        setattr(reg, self.reg_param, new_param)
                        if should_log:
                            log.info(f"Sparsity {current_sparsity:.3%} ≥ target {self.target_sparse:.1%} → Lambda {old_param:.8f} → {new_param:.8f}")
                        
                    # For Bregman optimizers, reinitialize subgradients when regularization changes
                    if hasattr(self.optimizer, 'initialize_sub_grad'):
                        for p in group['params']:
                            if p in self.optimizer.state:
                                state = self.optimizer.state[p]
                                state['sub_grad'] = self.optimizer.initialize_sub_grad(p, reg, group['delta'])
                
                return new_param
                
    def _get_current_param(self):
        """Get current regularization parameter value from first parameter group."""
        if self.optimizer and self.optimizer.param_groups:
            reg = self.optimizer.param_groups[0]['reg']
            return getattr(reg, self.reg_param, 1.0)
        return 1.0

    def get_state(self):
        """Get scheduler state for debugging/monitoring."""
        return {
            'warmup': self.warmup,
            'cooldown_val': self.cooldown_val,
            'target_sparsity': self.target_sparse,
            'current_param': self._get_current_param(),
        }