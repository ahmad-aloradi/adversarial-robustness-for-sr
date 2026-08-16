"""Bregman Regularizers for sparse neural network training.

This module provides various regularizers compatible with the BregmanPruner,
replicating the implementations from the `TimRoith/BregmanLearning` repository.

Each regularizer is initialized with a base strength `lamda`. The `delta` parameter
is passed to the `prox` method during optimization steps.
"""

import math

import torch


class BregmanRegularizer:
    """Base class for Bregman regularizers."""

    def __init__(self, lamda: float = 1.0, delta: float = 1.0):
        self.lamda = lamda

    def __call__(self, x: torch.Tensor) -> float:
        raise NotImplementedError

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        """Proximal operator.

        When lamda is None, uses self.lamda (set by the scheduler).
        """
        raise NotImplementedError

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegNone(BregmanRegularizer):
    """Null regularizer (equivalent to standard training)."""

    def __call__(self, x: torch.Tensor) -> float:
        return 0.0

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        return x

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(v)


class RegL1(BregmanRegularizer):
    """L1 norm regularizer."""

    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * torch.norm(x, p=1).item()

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        lamda = lamda if lamda is not None else self.lamda
        return torch.sign(x) * torch.clamp(
            torch.abs(x) - (delta * lamda), min=0
        )

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


class RegL1Pos(BregmanRegularizer):
    """L1 norm regularizer with positivity constraint."""

    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * torch.norm(x, p=1).item()

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        lamda = lamda if lamda is not None else self.lamda
        # Apply soft thresholding first, then clamp to ensure positivity
        soft_thresholded = torch.sign(x) * torch.clamp(
            torch.abs(x) - (delta * lamda), min=0
        )
        return torch.clamp(soft_thresholded, min=0)

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


class RegL1L2(BregmanRegularizer):
    """L1-L2 group sparsity regularizer (group lasso)."""

    def __call__(self, x: torch.Tensor) -> float:
        if x.dim() < 2:
            return 0.0  # Not applicable for vectors
        return (
            self.lamda
            * math.sqrt(x.shape[-1])
            * torch.norm(torch.norm(x, p=2, dim=1), p=1).item()
        )

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        lamda = lamda if lamda is not None else self.lamda
        if x.dim() < 2:
            return x  # Not applicable for vectors
        thresh = delta * lamda * math.sqrt(x.shape[-1])

        nx = torch.norm(x, p=2, dim=1, keepdim=True)
        # Avoid division by zero by adding a small epsilon where the norm is zero
        nx_safe = nx + (nx == 0).float() * 1e-8

        scale = torch.clamp(1 - thresh / nx_safe, min=0)
        return x * scale

    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return torch.zeros_like(x)  # Not applicable for vectors
        thresh = self.lamda * math.sqrt(x.shape[-1])

        nx = torch.norm(x, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        nx_safe = nx + (nx == 0).float() * 1e-8

        return thresh * (x / nx_safe)


class RegL1L2Conv(RegL1L2):
    """L1-L2 regularizer for convolutional layers."""

    def __call__(self, x: torch.Tensor) -> float:
        if x.dim() < 2:
            return 0.0
        return super().__call__(x.view(x.shape[0] * x.shape[1], -1))

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        if x.dim() < 2:
            return x
        original_shape = x.shape
        ret = super().prox(
            x.view(original_shape[0] * original_shape[1], -1),
            delta,
            lamda=lamda,
        )
        return ret.view(original_shape)

    def sub_grad(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return torch.zeros_like(x)
        original_shape = x.shape
        ret = super().sub_grad(
            x.view(original_shape[0] * original_shape[1], -1)
        )
        return ret.view(original_shape)


class RegSoftBernoulli(BregmanRegularizer):
    """Soft Bernoulli regularizer for encouraging sparsity with noise."""

    def __call__(self, x: torch.Tensor) -> float:
        return self.lamda * torch.norm(x, p=1).item()

    def prox(
        self, x: torch.Tensor, delta: float = 1.0, lamda: float = None
    ) -> torch.Tensor:
        lamda = lamda if lamda is not None else self.lamda
        return torch.sign(x) * torch.max(
            torch.clamp(torch.abs(x) - (delta * lamda), min=0),
            torch.bernoulli(0.01 * torch.ones_like(x)),
        )

    def sub_grad(self, v: torch.Tensor) -> torch.Tensor:
        return self.lamda * torch.sign(v)


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
    """Factory function to get a regularizer instance by name.

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
        raise ValueError(
            f"Unknown regularizer: {name}. Available: {list(_REGULARIZERS.keys())}"
        )

    return _REGULARIZERS[name](**kwargs)


# --- Param-group vocabulary: what "actively regularized" means for a group ---


def thresholds_weights(group: dict) -> bool:
    """The group's regularizer thresholds weights; RegNone does not."""
    return not isinstance(group["reg"], RegNone)


def lambda_scale(group: dict) -> float:
    """lambda_scale of a param group; a missing key is a config bug."""
    if "lambda_scale" not in group:
        raise KeyError(
            f"Group '{group.get('name')}' has no lambda_scale; every "
            "Bregman group must set optimizer_settings.lambda_scale."
        )
    return group["lambda_scale"]


def is_regularized(group: dict) -> bool:
    """An actively pruning group: thresholding, with lambda_scale > 0."""
    return thresholds_weights(group) and lambda_scale(group) > 0.0


if __name__ == "__main__":
    # Smoke: RegL1 keeps every weight above the barrier and shrinks it by delta*lamda.
    delta, lamda = 1.0, 0.3
    w0 = torch.tensor([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0])
    reg = RegL1(lamda=lamda)
    v0 = w0 / delta + reg.sub_grad(w0)  # v0 in dJ(w0), as LinBreg builds it
    print(f"RegL1: v0={v0.tolist()}")
    print(f"       w ={reg.prox(delta * v0, delta).tolist()}")
    print(f"barrier delta*lamda = {delta * lamda}")
