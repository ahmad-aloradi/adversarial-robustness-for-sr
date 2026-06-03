"""
Bregman optimizers adapted from BregmanLearning repository.
These implement linearized Bregman iterations for sparse neural network training.
"""
import torch
import math
from typing import Optional
from .bregman_regularizers import BregmanRegularizer, RegNone


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
    return torch.sign(sub_grad) * torch.sqrt(weighted_mag * delta_sub_grad.abs())


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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, 'rescale_mode', 'none')
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

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

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

                # Dual update: p̃^(k+1) = p^(k) − τ∇L(θ^(k))
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

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta)
                else:
                    prox_result = reg.prox(delta * sub_grad, delta)
                if use_nestrov_update:
                    prox_result *= (1 / (reg.lamda + 1e-12))
                p.copy_(prox_result)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

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
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, 'rescale_mode', 'none')
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

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

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

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

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta)
                else:
                    prox_result = reg.prox(delta * sub_grad, delta)

                if use_nestrov_update:
                    prox_result *= (1 / (reg.lamda + 1e-12))
                p.copy_(prox_result)

            if use_subgrad_correction or use_nestrov_update:
                reg.step_lamda_state()

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


class AdaBregW(AdaBreg):
    """Adaptive Bregman optimizer with decoupled weight decay.

    Extends AdaBreg with AdamW-style decoupled weight decay to control the
    magnitude of surviving weights, while L1 proximal controls sparsity.
    Weight decay is applied directly to weights before the proximal step,
    keeping it independent from the subgradient accumulation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        reg: Optional[BregmanRegularizer] = None,
        delta: float = 1.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3
    ):
        self.weight_decay = weight_decay
        if weight_decay <= 0.0:
            if weight_decay == 0:
                msg = f'{weight_decay} is set to zero. If you wish to use no weigth decay, use AdaBreg instead of AdabregW'
            else:
                msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(f"{msg}")
        super().__init__(params, lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with decoupled weight decay."""
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
            wd = group.get('weight_decay', self.weight_decay)
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, 'rescale_mode', 'none')
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

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

                state['step'] += 1
                step = state['step']

                sub_grad = state['sub_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

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

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta)
                else:
                    prox_result = reg.prox(delta * sub_grad, delta)
                if use_nestrov_update:
                    prox_result *= (1 / (reg.lamda + 1e-12))
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
    ):
        self.weight_decay = weight_decay
        if weight_decay <= 0.0:
            if weight_decay == 0:
                msg = f'{weight_decay} is set to zero. If you wish to use no weight decay, use AdaBreg instead of AdaBregL2'
            else:
                msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(f"{msg}")
        super().__init__(params, lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with coupled L2 regularization."""
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
            wd = group.get('weight_decay', self.weight_decay)
            # rescaling mode when λ changes between steps
            rescale_mode = getattr(reg, 'rescale_mode', 'none')
            use_subgrad_correction = rescale_mode == "subgradient_correction"
            use_nestrov_update = rescale_mode == "nestrovs_adaptive_update"
            use_weight_gradient_masking = rescale_mode == "weight_masking"

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Coupled L2: add weight decay to gradient
                if wd > 0:
                    grad = grad.add(p.data, alpha=wd)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['sub_grad'] = self.initialize_sub_grad(p, reg, delta)
                    state['exp_avg'] = torch.zeros_like(state['sub_grad'])
                    state['exp_avg_sq'] = torch.zeros_like(state['sub_grad'])

                state['step'] += 1
                step = state['step']

                sub_grad = state['sub_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Subgradient correction (before dual update)
                if use_subgrad_correction:
                    reg.apply_subgradient_correction(sub_grad, p.data)

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

                # Dual update: p̃^(k+1) = p^(k) − τ·adam_step (L2-augmented gradient)
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)

                # Primal update (Proximal step): θ^(k+1) = ∇(φ^(k))*(p̃^(k+1))
                if use_weight_gradient_masking:
                    prox_arg = weight_masked_prox_arg(
                        sub_grad, p.data, delta, reg.w_p, reg.wp_mode
                    )
                    prox_result = reg.prox(prox_arg, delta)
                else:
                    prox_result = reg.prox(delta * sub_grad, delta)
                if use_nestrov_update:
                    prox_result *= (1 / (reg.lamda + 1e-12))
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
            reg = group['reg']
            
            for p in group['params']:
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
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    return OPTIMIZER_REGISTRY[name]