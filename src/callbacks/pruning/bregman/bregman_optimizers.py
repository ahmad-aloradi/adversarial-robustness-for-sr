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

    ``dual_leak`` (gamma) turns the dual into a finite-memory integrator:
    ``v <- (1 - gamma) * v - lr * g`` on regularized groups. Survival then
    requires a sustained average gradient above ``gamma * lambda / lr`` instead
    of an unbounded lifetime sum, so weights whose demand has faded are
    garbage-collected. ``0.0`` (default) is the classic perfect-memory dual.
    Unregularized groups (RegNone: biases, norms) never leak — a classifier
    bias must drift freely to turn never-used classes off.
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
        dual_leak: float = 0.0,  # per-step dual forgetting factor (gamma)
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if momentum < 0.0:
            raise ValueError("Invalid momentum value")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening == 0"
            )
        if not 0.0 <= dual_leak < 1.0:
            raise ValueError(f"dual_leak must be in [0, 1), got {dual_leak}")

        if reg is None:
            reg = RegNone()

        defaults = dict(
            lr=lr,
            reg=reg,
            delta=delta,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            dual_leak=dual_leak,
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
            leak = group["dual_leak"] if not isinstance(reg, RegNone) else 0.0

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

                # Dual update: v^(k+1) = (1 − γ)·v^(k) − τ·g   (v = sub_grad)
                if leak > 0.0:
                    sub_grad.mul_(1.0 - leak)
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


# Re-born weights enter just past the prox threshold, so they start tiny.
BIRTH_MARGIN = 0.01


class AdaBreg(torch.optim.Optimizer):
    """Adaptive Bregman optimizer (Adam-style acceleration).

    Combines adaptive moment estimation with Bregman iterations.

    ``dual_leak`` (gamma) enables the hybrid birth rule on regularized groups.
    Adam-preconditioned dual steps are magnitude-blind (any sign-consistent
    gradient moves at full speed), so a leak inside them cannot separate real
    demand from a never-ending drip; the revival evidence must be the raw
    gradient. In hybrid mode, pruned weights stop taking Adam dual steps:
    their dual decays by ``(1 - gamma)`` and they are re-born only when a
    gamma-horizon EMA of the raw gradient exceeds
    ``birth_scale * gamma * lambda / lr`` — the same sustained-demand
    bar the LinBreg leak implies. ``dual_leak=0.0`` (default) is classic
    AdaBreg. Unregularized groups (RegNone) are never affected.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        dual_leak: float = 0.0,  # gamma: enables hybrid birth rule when > 0
        birth_scale: float = 0.5,  # scales the sustained-demand birth bar
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if not 0.0 <= dual_leak < 1.0:
            raise ValueError(f"dual_leak must be in [0, 1), got {dual_leak}")
        if birth_scale <= 0.0:
            raise ValueError(f"birth_scale must be > 0, got {birth_scale}")

        if reg is None:
            reg = RegNone()

        defaults = dict(
            lr=lr,
            reg=reg,
            delta=delta,
            betas=betas,
            eps=eps,
            dual_leak=dual_leak,
            birth_scale=birth_scale,
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
            leak = (
                group["dual_leak"] if not isinstance(reg, RegNone) else 0.0
            )

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

                if leak > 0.0:
                    self._hybrid_dual_update(
                        p, state, group, grad, exp_avg, denom, step_size
                    )
                else:
                    # Dual update: v^(k+1) = v^(k) − τ·adam_step
                    sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Primal update (prox): θ^(k+1) = prox(δ·v^(k+1))
                p.copy_(reg.prox(delta * sub_grad, delta))

        return loss

    @staticmethod
    def _hybrid_dual_update(p, state, group, grad, exp_avg, denom, step_size):
        """Dual step with the birth rule: active weights take the Adam step;
        pruned weights decay and are re-born only on sustained raw-gradient
        demand above the bar (see class docstring)."""
        leak = group["dual_leak"]
        lamda = group["reg"].lamda
        assert lamda > 0.0, f"hybrid birth needs reg.lamda > 0, got {lamda}"

        if "grad_ema" not in state:
            state["grad_ema"] = torch.zeros_like(p)
        grad_ema = state["grad_ema"]
        grad_ema.mul_(1.0 - leak).add_(grad, alpha=leak)

        sub_grad = state["sub_grad"]
        dead = p.data == 0
        adam_dual = sub_grad - step_size * exp_avg / denom
        new_dual = torch.where(dead, sub_grad * (1.0 - leak), adam_dual)

        bar = group["birth_scale"] * leak * lamda / group["lr"]
        born = dead & (grad_ema.abs() > bar)
        # Dual moves against the gradient; born entries start just past λ.
        born_dual = -torch.sign(grad_ema) * (lamda * (1.0 + BIRTH_MARGIN))
        sub_grad.copy_(torch.where(born, born_dual, new_dual))

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


class AdaBregW(AdaBreg):
    """Adaptive Bregman optimizer with decoupled weight decay.

    Extends AdaBreg with AdamW-style decoupled weight decay to control the
    magnitude of surviving weights, while L1 proximal controls sparsity. Weight
    decay is applied directly to weights after the proximal step, keeping it
    independent from the subgradient accumulation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3,
    ):
        self.weight_decay = weight_decay
        if weight_decay <= 0.0:
            if weight_decay == 0:
                msg = f"{weight_decay} is set to zero. If you wish to use no weigth decay, use AdaBreg instead of AdabregW"
            else:
                msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(f"{msg}")
        super().__init__(
            params, lr=lr, reg=reg, delta=delta, betas=betas, eps=eps
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with decoupled weight decay."""
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
            wd = group.get("weight_decay", self.weight_decay)

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

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

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

                # Decoupled weight decay: shrink surviving weights
                assert wd > 0, "Weight decay must be positive for AdaBregW"
                p.mul_(1 - lr * wd)

        return loss


class ProxSGD(torch.optim.Optimizer):
    """Proximal SGD optimizer.

    Standard proximal gradient method for comparison.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")

        if reg is None:
            reg = RegNone()

        defaults = dict(lr=lr, reg=reg)
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                # Gradient step
                p.add_(-step_size * grad)
                # Proximal step
                p.copy_(reg.prox(p.data, step_size))

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
    "adabregw": AdaBregW,
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
