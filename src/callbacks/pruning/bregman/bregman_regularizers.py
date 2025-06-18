"""
Bregman regularizers adapted from BregmanLearning repository.
These implement various sparsity-inducing regularizers for neural network training.
"""
import torch
import math
from typing import Protocol


class BregmanRegularizer(Protocol):
    """Protocol defining the interface for Bregman regularizers."""
    
    def __call__(self, x: torch.Tensor) -> float:
        """Evaluate the regularizer at x."""
        ...
    
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Proximal operator of the regularizer."""
        ...
    
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        """Subgradient of the regularizer at v."""
        ...


class RegNone:
    """No regularization - identity regularizer."""
    
    def __call__(self, x: torch.Tensor) -> float:
        return 0.0
    
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return x
    
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(v)


class RegL1:
    """L1 regularization (LASSO)."""
    
    def __init__(self, lamda: float = 1.0):
        self.lamda = lamda
        
    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * torch.norm(x, p=1).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda), min=0)
        
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


class RegL1Positive:
    """L1 regularization with positivity constraint."""
    
    def __init__(self, lamda: float = 1.0):
        self.lamda = lamda
        
    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * torch.norm(x, p=1).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return torch.clamp(
            torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda), min=0),
            min=0
        )
        
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


class RegL1L2:
    """Group LASSO (L1 norm of L2 norms)."""
    
    def __init__(self, lamda: float = 1.0):
        self.lamda = lamda
        
    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * math.sqrt(x.shape[-1]) * torch.norm(
            torch.norm(x, p=2, dim=1), p=1
        ).item()
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        thresh = delta * self.lamda * math.sqrt(x.shape[-1])
        
        ret = torch.clone(x)
        nx = torch.norm(x, p=2, dim=1).view(x.shape[0], 1)       
        
        ind = torch.where((nx != 0))[0]
        
        ret[ind] = x[ind] * torch.clamp(
            1 - torch.clamp(thresh / nx[ind], max=1), min=0
        )
        return ret
    
    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        thresh = self.lamda * math.sqrt(x.shape[-1])
        nx = torch.norm(x, p=2, dim=1).view(x.shape[0], 1)      
        ind = torch.where((nx != 0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind] / nx[ind]
        return thresh * ret


class RegL1L2Conv(RegL1L2):
    """Group LASSO adapted for convolutional kernels."""
    
    def __init__(self, lamda: float = 1.0):
        super().__init__(lamda=lamda)
        
    def __call__(self, x: torch.Tensor) -> float:
        return super().__call__(x.view(x.shape[0] * x.shape[1], -1))
    
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        ret = super().prox(x.view(x.shape[0] * x.shape[1], -1), delta)
        return ret.view(x.shape)
    
    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        ret = super().sub_grad(x.view(x.shape[0] * x.shape[1], -1))
        return ret.view(x.shape)


class RegSoftBernoulli:
    """Soft Bernoulli regularizer for sparse initialization."""
    
    def __init__(self, lamda: float = 1.0):
        self.lamda = lamda
        
    def prox(self, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return torch.sign(x) * torch.max(
            torch.clamp(torch.abs(x) - (delta * self.lamda), min=0),
            torch.bernoulli(0.01 * torch.ones_like(x))
        )
    
    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


# Registry for easy instantiation
REGULARIZER_REGISTRY = {
    "none": RegNone,
    "l1": RegL1,
    "l1_positive": RegL1Positive,
    "l1_l2": RegL1L2,
    "l1_l2_conv": RegL1L2Conv,
    "soft_bernoulli": RegSoftBernoulli,
}


def get_regularizer(name: str, **kwargs) -> BregmanRegularizer:
    """Factory function to create regularizers."""
    if name not in REGULARIZER_REGISTRY:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(REGULARIZER_REGISTRY.keys())}")
    return REGULARIZER_REGISTRY[name](**kwargs)