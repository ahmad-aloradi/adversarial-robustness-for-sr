"""Bregman optimizers adapted from BregmanLearning repository.

These implement linearized Bregman iterations for sparse neural network
training.
"""
import math
from typing import Optional

import torch

# Absolute (not relative) so the __main__ smoke block runs via `python <file>`.
from src.callbacks.pruning.bregman.bregman_regularizers import (
    BregmanRegularizer,
    RegNone,
)


class LinBreg(torch.optim.Optimizer):
    """Linearized Bregman optimizer.

    Implementation of the baseline algorithm from "A Bregman Learning Framework
    for Sparse Neural Networks" by Bungert et al.

    Momentum mirrors ``torch.optim.SGD`` exactly (same buffer, dampening and
    Nesterov formulas); the only difference is that the momentum-adjusted
    gradient drives the dual variable ``v`` and the weights come from its prox.

    ``weight_decay`` (mu) enters the gradient, so the iteration is LinBreg on
    ``L + mu*||w||^2/2``. Through the prox that is ``w <- (1 - lr*mu*delta)*w``
    on the support and nothing off it, where w is already 0. It cannot go after
    the prox -- w is a readout of v, recomputed every step -- and adding L2 to J
    only rescales delta. It is also the only norm control: the L1 prox
    translates v by a constant delta*lamda, which picks the support without
    bounding the size of what survives. See docs/pruning.md 1.1.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        momentum: float = 0.0,
        dampening: float = 0.0,  # SGD dampening on the momentum buffer
        nesterov: bool = False,  # Nesterov look-ahead on the momentum step
        weight_decay: float = 0.0,  # mu: L2 on the surviving weights, into the dual
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if momentum < 0.0:
            raise ValueError("Invalid momentum value")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening == 0"
            )

        if reg is None:
            reg = RegNone()

        defaults = dict(
            lr=lr,
            reg=reg,
            delta=delta,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            delta = group["delta"]
            reg = group["reg"]
            step_size = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["sub_grad"] = self.initialize_sub_grad(p, reg, delta)
                    state["momentum_buffer"] = None

                state["step"] += 1
                sub_grad = state["sub_grad"]

                # mu reaches the support only: prox makes p exactly 0 elsewhere.
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # SGD momentum on the gradient (see torch.optim.SGD).
                d_grad = grad
                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    if buf is None:
                        # First step seeds the buffer: b_1 = g
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        # b_k = μ·b_{k-1} + (1 − dampening)·g
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    # nesterov: g ← g + μ·b_k ;  plain: g ← b_k
                    d_grad = grad.add(buf, alpha=momentum) if nesterov else buf

                # Dual update: v^(k+1) = v^(k) − τ·g   (v = sub_grad)
                sub_grad.add_(d_grad, alpha=-step_size)

                # Primal update (prox): θ^(k+1) = prox(δ·v^(k+1))
                p.copy_(reg.prox(delta * sub_grad, delta))

        return loss

    def initialize_sub_grad(
        self, p: torch.Tensor, reg: BregmanRegularizer, delta: float
    ):
        """Initialize subgradient for Bregman iterations."""
        p_init = p.data.clone()
        return 1 / delta * p_init + reg.sub_grad(p_init)

    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group["reg"]

            for p in group["params"]:
                group_reg_val += reg(p)

            reg_vals.append(group_reg_val)

        return reg_vals


class AdaBreg(torch.optim.Optimizer):
    """Adaptive Bregman optimizer (Adam-style acceleration).

    Combines adaptive moment estimation with Bregman iterations. ``eps`` is
    the gradient size below which the Adam step degrades to being
    proportional to the gradient; with an over-sized classifier head, raising
    it to ~1e-4 stops never-vanishing phantom-class pushes from accumulating
    (see docs/bregman_phantom_classes.md).

    ``weight_decay`` (mu) enters the gradient before the moments, so the
    denominator divides ``mu*w`` and ``grad L`` alike: mu sets the decay's share
    of the numerator, not a rate, and LinBreg's ``(1 - lr*mu*delta)`` does not
    hold here. Where ``mu*w`` dominates the numerator the dual step is
    ``lr*sign(w)`` for any mu. See LinBreg for why it must live in the dual.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,  # mu: L2 on the surviving weights, into the dual
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")

        if reg is None:
            reg = RegNone()

        defaults = dict(
            lr=lr,
            reg=reg,
            delta=delta,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            delta = group["delta"]
            reg = group["reg"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["sub_grad"] = self.initialize_sub_grad(p, reg, delta)
                    state["exp_avg"] = torch.zeros_like(state["sub_grad"])
                    state["exp_avg_sq"] = torch.zeros_like(state["sub_grad"])

                state["step"] += 1
                step = state["step"]

                sub_grad = state["sub_grad"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # mu reaches the support only: prox makes p exactly 0 elsewhere.
                # It enters before the moments, so Adam normalizes it too.
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
                step_size = lr / bias_correction1

                # Dual update: v^(k+1) = v^(k) − τ·adam_step
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Primal update (prox): θ^(k+1) = prox(δ·v^(k+1))
                p.copy_(reg.prox(delta * sub_grad, delta))

        return loss

    def initialize_sub_grad(
        self, p: torch.Tensor, reg: BregmanRegularizer, delta: float
    ):
        """Initialize subgradient for Bregman iterations."""
        p_init = p.data.clone()
        return 1 / delta * p_init + reg.sub_grad(p_init)

    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group["reg"]

            for p in group["params"]:
                group_reg_val += reg(p)

            reg_vals.append(group_reg_val)

        return reg_vals


class ProxSGD(torch.optim.Optimizer):
    """Proximal SGD optimizer.

    Standard proximal gradient method for comparison.

    ``weight_decay`` (mu) enters the gradient, as in ``torch.optim.SGD``: there
    is no dual here, w carries the state, so the shrink accumulates on w itself.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        momentum: float = 0.0,
        dampening: float = 0.0,  # SGD dampening on the momentum buffer
        nesterov: bool = False,  # Nesterov look-ahead on the momentum step
        weight_decay: float = 0.0,  # mu: L2 on the surviving weights
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")

        if reg is None:
            reg = RegNone()

        defaults = dict(
            lr=lr,
            reg=reg,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            reg = group["reg"]
            step_size = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = None

                # mu reaches the support only: prox makes p exactly 0 elsewhere.
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # SGD momentum on the gradient (see torch.optim.SGD).
                d_grad = grad
                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    if buf is None:
                        # First step seeds the buffer: b_1 = g
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        # b_k = μ·b_{k-1} + (1 − dampening)·g
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    # nesterov: g ← g + μ·b_k ;  plain: g ← b_k
                    d_grad = grad.add(buf, alpha=momentum) if nesterov else buf

                # Gradient step
                p.add_(d_grad, alpha=-step_size)

                # Proximal step
                p.data = reg.prox(p.data, step_size)

        return loss

    @torch.no_grad()
    def evaluate_reg(self):
        """Evaluate regularization terms."""
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            reg = group["reg"]

            for p in group["params"]:
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
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name]


if __name__ == "__main__":
    # Smoke: L1 Bregman drives a tiny dense layer sparse for both optimizers.
    import torch.nn as nn

    from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
    from src.callbacks.pruning.shared_prune_utils import compute_sparsity

    for cls, lr in ((AdaBreg, 1e-2), (LinBreg, 1e-1)):
        torch.manual_seed(0)
        layer = nn.Linear(64, 32)
        opt = cls(layer.parameters(), lr=lr, reg=RegL1(lamda=0.2))
        for _ in range(150):
            x, y = torch.randn(16, 64), torch.randn(16, 32)
            opt.zero_grad()
            ((layer(x) - y) ** 2).mean().backward()
            opt.step()
        sp = compute_sparsity(list(layer.parameters()), threshold=1e-12)
        print(f"{cls.__name__}: sparsity after 150 steps = {sp:.1%}")
