"""Bregman optimizers adapted from BregmanLearning repository.

These implement linearized Bregman iterations for sparse neural network
training.
"""
import math
from typing import Optional, Union

import torch

# Absolute (not relative) so the __main__ smoke block runs via `python <file>`.
from src.callbacks.pruning.bregman.bregman_regularizers import (
    BregmanRegularizer,
    RegL1L2,
    RegNone,
)

_MOVE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def weight_masked_prox_arg(
    sub_grad: torch.Tensor,
    weight: torch.Tensor,
    delta: float,
    w_p: float,
    wp_mode: str = "blend",
) -> torch.Tensor:
    """Geometric-mean readout for the "weight_masking" primal step.

    Reads out a magnitude between the standard candidate c = delta·sub_grad and
    the latched geometric mean sqrt(|weight|·|c|), with the per-element gate
    g ∈ [0, 1] plugged into:

        arg = sign(c) · sqrt( ((1 - g)·|weight| + g·|c|) · |c| )

        g = 1 -> arg = c                           (standard Bregman readout)
        g = 0 -> arg = sign(c)·sqrt(|weight|·|c|)   (latch; weight=0 stays 0)

    w_p ∈ [0, 1] sets g in one of two ways:

    "blend" (default): g = w_p, a deterministic geometric blend. Every weight
        gets a single readout that lags the candidate toward |weight|, i.e.
        graded damping applied to all live weights at once.

    "probabilistic": g ~ Bernoulli(w_p) per element, drawn fresh each step.
        Each weight then takes one of the two *exact* endpoints — the true
        Bregman readout with probability w_p, the latch otherwise. w_p is thus
        the fraction of the support still allowed its reversible step; annealing
        w_p 1 -> 0 commits the support stochastically, element by element. The
        draw consumes from the global torch RNG, so seed the run to reproduce.

    Both modes share the endpoints exactly (w_p ∈ {0, 1} is identical) and the
    nonzero fixed point |c|: feeding |weight| = |c| back returns |c| for any
    gate. delta sits inside the root (sqrt(delta) on the cross term) so the
    g=0 fixed point stays delta·sub_grad rather than delta^2·sub_grad. The
    radicand is non-negative by construction, so sqrt is NaN-safe for any sign
    of sub_grad; the result takes the candidate's sign.
    """
    delta_sub_grad = delta * sub_grad
    if wp_mode == "blend":
        g = w_p
    elif wp_mode == "probabilistic":
        g = torch.bernoulli(torch.full_like(weight, float(w_p)))
    else:
        raise ValueError(
            f"wp_mode must be 'blend' or 'probabilistic', got {wp_mode!r}"
        )
    weighted_mag = (1.0 - g) * weight.abs() + g * delta_sub_grad.abs()
    return torch.sign(sub_grad) * torch.sqrt(
        weighted_mag * delta_sub_grad.abs()
    )


class _MovementReweightMixin:
    """Reweighted-l1 per-weight thresholds for the Bregman prox optimizers.

    Each weight gets its own soft-threshold ``lamda * a_i`` (Candes-Wakin-Boyd
    reweighted l1) with ``a_i`` driven by a movement importance (Sanh et al.),
    redistributing a fixed sparsity budget toward task-useful weights. The
    global rho controller still pins the rate; ``a`` only moves *where* the cuts
    fall. All defaults are the no-op, so the off path is byte-identical.
    """

    def _setup_movement(
        self,
        movement_reweight: bool,
        move_importance: str,
        move_beta: float,
        move_eps: float,
        move_clip: tuple,
        move_warmup_steps: int,
        move_dtype: Optional[str],
    ) -> None:
        assert move_importance in ("movement_signed", "taylor_abs"), (
            f"move_importance must be 'movement_signed' or 'taylor_abs', "
            f"got {move_importance!r}"
        )
        a_min, a_max = move_clip
        assert (
            0.0 <= a_min < a_max
        ), f"need 0 <= a_min < a_max, got {move_clip}"
        assert 0.0 <= move_beta < 1.0, "move_beta must be in [0, 1)"
        assert move_eps > 0.0, "move_eps must be positive"
        assert move_warmup_steps >= 0, "move_warmup_steps must be >= 0"
        assert move_dtype is None or move_dtype.lower() in _MOVE_DTYPES, (
            f"move_dtype must be one of {sorted(_MOVE_DTYPES)} or None, "
            f"got {move_dtype!r}"
        )
        self.movement_reweight = movement_reweight
        self.move_importance = move_importance
        self.move_beta = move_beta
        self.move_eps = move_eps
        self.move_a_min = a_min
        self.move_a_max = a_max
        self.move_warmup_steps = move_warmup_steps
        self._move_dtype = (
            None if move_dtype is None else _MOVE_DTYPES[move_dtype.lower()]
        )

    @torch.no_grad()
    def _reweighted_lamda(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        reg: BregmanRegularizer,
    ) -> Union[torch.Tensor, float]:
        """Per-weight soft-threshold for reweighted-l1.

        Returns the scalar ``reg.lamda`` (no allocation) when reweighting is
        off, the prox does no thresholding (RegNone), or still in warmup;
        otherwise updates the importance EMA in place and returns the per-weight
        tensor ``reg.lamda * a``.
        """
        if not self.movement_reweight or isinstance(reg, RegNone):
            return reg.lamda
        # A group-lasso prox thresholds by row norm; a per-weight a is undefined.
        assert not isinstance(reg, RegL1L2), (
            "movement_reweight needs an element-wise L1 reg (e.g. RegL1); got "
            f"{type(reg).__name__}."
        )

        if "move_ema" not in state:
            dtype = self._move_dtype or p.dtype
            state["move_ema"] = torch.zeros_like(p, dtype=dtype)

        # Importance EMA from the raw gradient and the pre-update weight.
        move_ema = state["move_ema"]
        if self.move_importance == "movement_signed":
            move_ema.mul_(self.move_beta).addcmul_(
                grad, p, value=-(1.0 - self.move_beta)
            )
        else:  # taylor_abs
            move_ema.mul_(self.move_beta).add_(
                grad.mul(p).abs_(), alpha=1.0 - self.move_beta
            )

        if state["step"] <= self.move_warmup_steps:
            return reg.lamda

        return reg.lamda * self._movement_multiplier(p, move_ema)

    @torch.no_grad()
    def _movement_multiplier(
        self, p: torch.Tensor, move_ema: torch.Tensor
    ) -> torch.Tensor:
        """Mean-unit, clipped, revival-protected per-weight multiplier ``a``.

        Normalized over the live support only: dead weights have imp~=0, so
        including them would collapse every live multiplier. ``a=1`` on the dead
        support keeps Bregman revival on the baseline threshold.
        """
        live = p != 0
        if not live.any():
            return torch.ones_like(p, dtype=torch.float32)

        # fp32 copy: keep the in-place ops off the buffer and the mean accurate.
        imp = move_ema.to(dtype=torch.float32, copy=True)
        if self.move_importance == "movement_signed":
            imp.clamp_(min=0.0)
        r = imp.add_(self.move_eps).reciprocal_()
        r.div_(r[live].mean())
        a = r.clamp_(self.move_a_min, self.move_a_max)
        a[~live] = 1.0
        return a


class LinBreg(_MovementReweightMixin, torch.optim.Optimizer):
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
        momentum: float = 0.0,
        movement_reweight: bool = False,
        move_importance: str = "movement_signed",
        move_beta: float = 0.9,
        move_eps: float = 1e-8,
        move_clip: tuple = (0.1, 10.0),
        move_warmup_steps: int = 0,
        move_dtype: Optional[str] = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")

        if reg is None:
            reg = RegNone()

        self._setup_movement(
            movement_reweight,
            move_importance,
            move_beta,
            move_eps,
            move_clip,
            move_warmup_steps,
            move_dtype,
        )
        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, "rescale_mode", "none")
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["sub_grad"] = self.initialize_sub_grad(p, reg, delta)
                    state["momentum_buffer"] = None

                # Update step
                state["step"] += 1

                # Get current subgradient
                sub_grad = state["sub_grad"]

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

                # Dual update: p̃^(k+1) = p^(k) − τ∇L(θ^(k))
                if momentum > 0.0:
                    mom_buff = state["momentum_buffer"]
                    if state["momentum_buffer"] is None:
                        mom_buff = torch.zeros_like(grad)

                    mom_buff.mul_(momentum)
                    mom_buff.add_((1 - momentum) * step_size * grad)
                    state["momentum_buffer"] = mom_buff
                    sub_grad.add_(-mom_buff)
                else:
                    sub_grad.add_(-step_size * grad)

                # Scalar reg.lamda when off (byte-identical), else per-weight.
                lamda = self._reweighted_lamda(p, grad, state, reg)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta, lamda=lamda)
                else:
                    prox_result = reg.prox(
                        delta * sub_grad, delta, lamda=lamda
                    )
                if use_nestrov_update:
                    prox_result *= 1 / (reg.lamda + 1e-12)
                p.copy_(prox_result)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

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


class AdaBreg(_MovementReweightMixin, torch.optim.Optimizer):
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
        eps: float = 1e-8,
        movement_reweight: bool = False,
        move_importance: str = "movement_signed",
        move_beta: float = 0.9,
        move_eps: float = 1e-8,
        move_clip: tuple = (0.1, 10.0),
        move_warmup_steps: int = 0,
        move_dtype: Optional[str] = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")

        if reg is None:
            reg = RegNone()

        self._setup_movement(
            movement_reweight,
            move_importance,
            move_beta,
            move_eps,
            move_clip,
            move_warmup_steps,
            move_dtype,
        )
        defaults = dict(lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)
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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, "rescale_mode", "none")
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

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

                # Update step
                state["step"] += 1
                step = state["step"]

                # Get state variables
                sub_grad = state["sub_grad"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )

                # Compute step size
                step_size = lr / bias_correction1

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Scalar reg.lamda when off (byte-identical), else per-weight.
                lamda = self._reweighted_lamda(p, grad, state, reg)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta, lamda=lamda)
                else:
                    prox_result = reg.prox(
                        delta * sub_grad, delta, lamda=lamda
                    )

                if use_nestrov_update:
                    prox_result *= 1 / (reg.lamda + 1e-12)
                p.copy_(prox_result)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

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


class AdaBregW(AdaBreg):
    """Adaptive Bregman optimizer with decoupled weight decay.

    Extends AdaBreg with AdamW-style decoupled weight decay to control the
    magnitude of surviving weights, while L1 proximal controls sparsity. Weight
    decay is applied directly to weights before the proximal step, keeping it
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
        movement_reweight: bool = False,
        move_importance: str = "movement_signed",
        move_beta: float = 0.9,
        move_eps: float = 1e-8,
        move_clip: tuple = (0.1, 10.0),
        move_warmup_steps: int = 0,
        move_dtype: Optional[str] = None,
    ):
        self.weight_decay = weight_decay
        if weight_decay <= 0.0:
            if weight_decay == 0:
                msg = f"{weight_decay} is set to zero. If you wish to use no weigth decay, use AdaBreg instead of AdabregW"
            else:
                msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(f"{msg}")
        super().__init__(
            params,
            lr=lr,
            reg=reg,
            delta=delta,
            betas=betas,
            eps=eps,
            movement_reweight=movement_reweight,
            move_importance=move_importance,
            move_beta=move_beta,
            move_eps=move_eps,
            move_clip=move_clip,
            move_warmup_steps=move_warmup_steps,
            move_dtype=move_dtype,
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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, "rescale_mode", "none")
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

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

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )

                # Compute step size
                step_size = lr / bias_correction1

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Scalar reg.lamda when off (byte-identical), else per-weight.
                lamda = self._reweighted_lamda(p, grad, state, reg)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta, lamda=lamda)
                else:
                    prox_result = reg.prox(
                        delta * sub_grad, delta, lamda=lamda
                    )
                if use_nestrov_update:
                    prox_result *= 1 / (reg.lamda + 1e-12)
                p.copy_(prox_result)

                # Decoupled weight decay: shrink surviving weights
                assert wd > 0, "Weight decay must be positive for AdaBregW"
                p.mul_(1 - lr * wd)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

        return loss


class AdaBregL2(AdaBreg):
    """Adaptive Bregman optimizer with coupled L2 regularization.

    Extends AdaBreg by adding weight_decay * p to the gradient before the
    Adam moment updates, exactly like standard Adam's weight_decay (coupled).
    This means L2 regularization flows through the subgradient accumulation,
    unlike AdaBregW which applies decay after the proximal step (decoupled).

    Comparison:
    - AdaBregW: p = prox(delta * v); p *= (1 - lr * wd)  [decoupled]
    - AdaBregL2: grad += wd * p; v -= adam_step(grad); p = prox(delta * v)  [coupled]
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
        movement_reweight: bool = False,
        move_importance: str = "movement_signed",
        move_beta: float = 0.9,
        move_eps: float = 1e-8,
        move_clip: tuple = (0.1, 10.0),
        move_warmup_steps: int = 0,
        move_dtype: Optional[str] = None,
    ):
        self.weight_decay = weight_decay
        if weight_decay <= 0.0:
            if weight_decay == 0:
                msg = f"{weight_decay} is set to zero. If you wish to use no weight decay, use AdaBreg instead of AdaBregL2"
            else:
                msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(f"{msg}")
        super().__init__(
            params,
            lr=lr,
            reg=reg,
            delta=delta,
            betas=betas,
            eps=eps,
            movement_reweight=movement_reweight,
            move_importance=move_importance,
            move_beta=move_beta,
            move_eps=move_eps,
            move_clip=move_clip,
            move_warmup_steps=move_warmup_steps,
            move_dtype=move_dtype,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with coupled L2
        regularization."""
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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, "rescale_mode", "none")
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Coupled L2: add weight decay to gradient
                if wd > 0:
                    grad = grad.add(p.data, alpha=wd)

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

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )

                # Compute step size
                step_size = lr / bias_correction1

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step (L2-augmented gradient)
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Importance uses the raw grad, not the L2-augmented one.
                lamda = self._reweighted_lamda(p, p.grad.data, state, reg)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta, lamda=lamda)
                else:
                    prox_result = reg.prox(
                        delta * sub_grad, delta, lamda=lamda
                    )
                if use_nestrov_update:
                    prox_result *= 1 / (reg.lamda + 1e-12)
                p.copy_(prox_result)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

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
    "adabregl2": AdaBregL2,
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
    # Smoke: reweighted-l1 still drives sparsity on a tiny dense layer.
    import torch.nn as nn

    from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
    from src.callbacks.pruning.shared_prune_utils import compute_sparsity

    for cls, lr in ((AdaBreg, 1e-2), (LinBreg, 1e-1)):
        torch.manual_seed(0)
        layer = nn.Linear(64, 32)
        opt = cls(
            layer.parameters(),
            lr=lr,
            reg=RegL1(lamda=0.2),
            movement_reweight=True,
            move_warmup_steps=5,
        )
        for _ in range(150):
            x, y = torch.randn(16, 64), torch.randn(16, 32)
            opt.zero_grad()
            ((layer(x) - y) ** 2).mean().backward()
            opt.step()
        sp = compute_sparsity(list(layer.parameters()), threshold=1e-12)
        print(
            f"{cls.__name__}: sparsity after 150 reweighted steps = {sp:.1%}"
        )
