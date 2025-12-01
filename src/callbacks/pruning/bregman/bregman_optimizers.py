"""
Bregman optimizers adapted from BregmanLearning repository.
These implement linearized Bregman iterations for sparse neural network training.
"""
import torch
import math
from typing import Optional, Union
from .bregman_regularizers import BregmanRegularizer, RegNone


class LinBreg(torch.optim.Optimizer):
    """Linearized Bregman optimizer.
    
    Implementation of the baseline algorithm from "A Bregman Learning Framework 
    for Sparse Neural Networks" by Bungert et al.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        momentum: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        if reg is None:
            reg = RegNone()
            
        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
        super(LinBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            delta = group['delta']
            reg = group['reg'] 
            step_size = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['sub_grad'] = self.initialize_sub_grad(p, reg, delta)
                    state['momentum_buffer'] = None

                # Update step
                state['step'] += 1

                # Get current subgradient
                sub_grad = state['sub_grad']
                
                # Update subgradient
                if momentum > 0.0:
                    mom_buff = state['momentum_buffer']
                    if state['momentum_buffer'] is None:
                        mom_buff = torch.zeros_like(grad)
 
                    mom_buff.mul_(momentum)
                    mom_buff.add_((1 - momentum) * step_size * grad) 
                    state['momentum_buffer'] = mom_buff
                    sub_grad.add_(-mom_buff)
                else:
                    sub_grad.add_(-step_size * grad)
                
                # Update parameters using proximal operator
                p.data = reg.prox(delta * sub_grad, delta)
        
        return loss
        
    def initialize_sub_grad(self, p: torch.Tensor, reg: BregmanRegularizer, delta: float):
        """Initialize subgradient for Bregman iterations."""
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group['reg']
            
            for p in group['params']:
                group_reg_val += reg(p)
                
            reg_vals.append(group_reg_val)
            
        return reg_vals


class AdaBreg(torch.optim.Optimizer):
    """Adaptive Bregman optimizer (Adam-style acceleration).
    
    Combines adaptive moment estimation with Bregman iterations.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        if reg is None:
            reg = RegNone()
            
        defaults = dict(lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)
        super(AdaBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            delta = group['delta']
            reg = group['reg']
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['sub_grad'] = self.initialize_sub_grad(p, reg, delta)
                    state['exp_avg'] = torch.zeros_like(state['sub_grad'])
                    state['exp_avg_sq'] = torch.zeros_like(state['sub_grad'])
                
                # Update step
                state['step'] += 1
                step = state['step']
                
                # Get state variables
                sub_grad = state['sub_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # Compute step size
                step_size = lr / bias_correction1
                
                # Update subgradient
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Update parameters using proximal operator
                p.data = reg.prox(delta * sub_grad, delta)
        
        return loss
        
    def initialize_sub_grad(self, p: torch.Tensor, reg: BregmanRegularizer, delta: float):
        """Initialize subgradient for Bregman iterations."""
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group['reg']
            
            for p in group['params']:
                group_reg_val += reg(p)
                
            reg_vals.append(group_reg_val)
            
        return reg_vals


class ProxSGD(torch.optim.Optimizer):
    """Proximal SGD optimizer.
    
    Standard proximal gradient method for comparison.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        if reg is None:
            reg = RegNone()
            
        defaults = dict(lr=lr, reg=reg)
        super(ProxSGD, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            reg = group['reg'] 
            step_size = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    
                # Gradient step
                p.data.add_(-step_size * grad)
                # Proximal step
                p.data = reg.prox(p.data, step_size)
        
        return loss
                
    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group['reg']
            
            for p in group['params']:
                group_reg_val += reg(p)
                
            reg_vals.append(group_reg_val)
            
        return reg_vals


# Registry for easy instantiation
OPTIMIZER_REGISTRY = {
    "linbreg": LinBreg,
    "adabreg": AdaBreg,
    "proxsgd": ProxSGD,
}


def get_bregman_optimizer(name: str):
    """Factory function to get Bregman optimizer class."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    return OPTIMIZER_REGISTRY[name]