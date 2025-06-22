"""
Bregman Regularizers for sparse neural network training.

This module provides various regularizers compatible with the BregmanPruner,
replicating the implementations from the `TimRoith/BregmanLearning` repository.

Each regularizer is initialized with a base strength `lamda` and an internal
dynamic multiplier `mu` (defaulting to 1.0), which is adjusted by the LamdaScheduler.
The effective regularization strength used in calculations is the product `mu * lamda`.
"""
import torch
import torch.nn as nn
import math


class BregmanRegularizer:
    """Base class for Bregman regularizers."""
    def __init__(self, lamda: float = 1.0, delta: float = 1.0):
        self.lamda = lamda
        self.mu = delta  # Dynamic multiplier, adjusted by scheduler

    def __call__(self, x: torch.Tensor) -> float:
        raise NotImplementedError

    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        raise NotImplementedError

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegNone(BregmanRegularizer):
    """Null regularizer (equivalent to standard training)."""
    def __call__(self, x: torch.Tensor) -> float:
        return 0.0
    
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return x
    
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(v)


class RegL1(BregmanRegularizer):
    """L1 norm regularizer."""
    def __call__(self, x: torch.Tensor) -> float:
        effective_lamda = self.mu * self.lamda
        return effective_lamda * torch.norm(x, p=1).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * effective_lamda), min=0)
        
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        return effective_lamda * torch.sign(v)


class RegL1Pos(BregmanRegularizer):
    """L1 norm regularizer with positivity constraint."""
    def __call__(self, x: torch.Tensor) -> float:
        effective_lamda = self.mu * self.lamda
        return effective_lamda * torch.norm(x, p=1).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        # Apply soft thresholding first, then clamp to ensure positivity
        soft_thresholded = torch.sign(x) * torch.clamp(torch.abs(x) - (delta * effective_lamda), min=0)
        return torch.clamp(soft_thresholded, min=0)
        
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        return effective_lamda * torch.sign(v)


class RegL1L2(BregmanRegularizer):
    """L1-L2 group sparsity regularizer (group lasso)."""
    def __call__(self, x: torch.Tensor) -> float:
        effective_lamda = self.mu * self.lamda
        if x.dim() < 2: return 0.0 # Not applicable for vectors
        return effective_lamda * math.sqrt(x.shape[-1]) * torch.norm(torch.norm(x, p=2, dim=1), p=1).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        if x.dim() < 2: return x # Not applicable for vectors
        effective_lamda = self.mu * self.lamda
        thresh = delta * effective_lamda * math.sqrt(x.shape[-1])
        
        nx = torch.norm(x, p=2, dim=1, keepdim=True)
        # Avoid division by zero by adding a small epsilon where the norm is zero
        nx_safe = nx + (nx == 0).float() * 1e-8
        
        scale = torch.clamp(1 - thresh / nx_safe, min=0)
        return x * scale
    
    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2: return torch.zeros_like(x) # Not applicable for vectors
        effective_lamda = self.mu * self.lamda
        thresh = effective_lamda * math.sqrt(x.shape[-1])
        
        nx = torch.norm(x, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        nx_safe = nx + (nx == 0).float() * 1e-8
        
        return thresh * (x / nx_safe)

class RegL1L2Conv(RegL1L2):
    """L1-L2 regularizer for convolutional layers."""
    def __call__(self, x: torch.Tensor) -> float:
        if x.dim() < 2: return 0.0
        return super().__call__(x.view(x.shape[0] * x.shape[1], -1))
    
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        if x.dim() < 2: return x
        original_shape = x.shape
        ret = super().prox(x.view(original_shape[0] * original_shape[1], -1), delta)
        return ret.view(original_shape)
    
    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2: return torch.zeros_like(x)
        original_shape = x.shape
        ret = super().sub_grad(x.view(original_shape[0] * original_shape[1], -1))
        return ret.view(original_shape)

class RegSoftBernoulli(BregmanRegularizer):
    """Soft Bernoulli regularizer for encouraging sparsity with noise."""
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        soft_threshold = torch.clamp(torch.abs(x) - (delta * effective_lamda), min=0)
        # In the original repo, the probability is hardcoded. Making it small.
        noise = torch.bernoulli(0.01 * torch.ones_like(x))
        return torch.sign(x) * torch.max(soft_threshold, noise)
    
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        effective_lamda = self.mu * self.lamda
        return effective_lamda * torch.sign(v)

# Dictionary to easily access regularizers by name
_REGULARIZERS = {
    "none": RegNone,
    "l1": RegL1,
    "l1_pos": RegL1Pos,
    "l1_l2": RegL1L2,
    "l1_l2_conv": RegL1L2Conv,
    "soft_bernoulli": RegSoftBernoulli,
}

def get_regularizer(name: str, **kwargs) -> BregmanRegularizer:
    """
    Factory function to get a regularizer instance by name.
    
    Parameters
    ----------
    name : str
        Name of the regularizer (e.g., "l1", "l1_l2").
    **kwargs
        Keyword arguments to pass to the regularizer's constructor (e.g., lamda).
        
    Returns
    -------
    BregmanRegularizer
        An instance of the specified regularizer.
    """
    name = name.lower()
    if name not in _REGULARIZERS:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(_REGULARIZERS.keys())}")
    
    return _REGULARIZERS[name](**kwargs)