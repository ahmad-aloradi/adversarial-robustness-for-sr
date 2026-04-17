"""
Mini-simulation of Bregman pruning dynamics comparing LinBreg vs AdaBreg
across all three rescale modes: none, nesterov_adaptive_update, subgradient_correction.
ProxSGD (standard proximal gradient) is included as a non-Bregman baseline.

LinBreg: v -= lr * grad;         w = prox(δv)
AdaBreg: v -= lr * m̂/(√v̂+ε);     w = prox(δv)   (Adam-style adaptive step)
ProxSGD: w = prox(w - lr * grad, lr)            (no dual variable; rescale modes N/A)

Rescale modes (applied when λ changes between steps):
  none:                     standard prox, no correction
  nesterov_adaptive_update: w = prox(δv) / λ
  subgradient_correction:   AFTER the dual step, additively patch the support
                            of v with the λ shift: v[nz] += (λ_new − λ_old)·sign(w[nz]);
                            clamp the zero coordinates to [−λ_new, λ_new].
                            Valid for EN R(w) = λ‖w‖₁ + (1/2δ)‖w‖²,
                            because only the λ·sign(w) piece of ∂R depends on λ.
                            The additive form commutes with the gradient step, so
                            placing it after `v -= lr·g` preserves the gradient
                            contribution (unlike the ratio form).
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Literal

# ── Simulation parameters ──────────────────────────────────────────
N = 20_000
T = 20_000
SUBGRAD_MATHOD: Literal['method1', 'method2', 'method3', 'method4', 'method4_adaptive'] = 'method4_adaptive'

sigma_w = 1.0         # scale of nonzero weights at init
init_density = 0.01   # fraction of nonzero weights (1% → 99% sparse)
sigma_grad = 0.1      # gradient noise std

# Lambda (and scheduler)
delta = 1.0
lambda_init_linbreg = 0.01
lambda_init_adabreg = 0.1
update_frequency = 5
acceleration_factor = 1.0
target_sparsity = 0.9
proxsgd_warmup_steps = 10000

min_lambda = 1e-6
max_lambda = 1e6


# LinBreg hyperparams
lr_linbreg = 0.1

# Adam hyperparams (for AdaBreg)
lr_adabreg = 1e-2
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

lr_proxsgd = 0.01
lambda_init_proxsgd = 1e-9

np.random.seed(42)

# ── Shared init ─────────────────────────────────────────────────────
# 99% sparse init: Bernoulli mask selects 1% of weights as nonzero
mask = (np.random.rand(N) < init_density).astype(float)
w_init = np.random.randn(N) * sigma_w * mask
w_init_proxsgd = np.random.randn(N)
all_grads = np.random.randn(T, N) * sigma_grad

SNAPSHOT_STEPS = [0, 50, 1000, 5000, 7000, 10000, 15000, 20000, T - 1]
PRINT_STEPS = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, T - 1]
SPARSITY_ZOOM = 1000


# ── Helpers ─────────────────────────────────────────────────────────
def prox_l1(x, delta, lamda):
    # Resolvent of R(u) = λ‖u‖₁ + (1/2)‖u‖² (Elastic Net) used by LinBreg/AdaBreg.
    return np.sign(x) * np.maximum(np.abs(x) - delta * lamda, 0.0) / (1 + delta)


def soft_threshold(x, threshold):
    # Pure L1 prox for classical ProxSGD: argmin_u λ‖u‖₁ + (1/(2α))‖u - x‖²,
    # threshold = α·λ. No EN shrinkage term — that belongs to the Bregman flow.
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def lambda_step(lamda, current_sparsity, step):
    if step % update_frequency != 0:
        return lamda
    diff = current_sparsity - target_sparsity
    if current_sparsity < target_sparsity:
        lamda *= 1 + acceleration_factor * abs(diff)
    elif current_sparsity > target_sparsity:
        lamda /= 1 + acceleration_factor * abs(diff)
    return np.clip(lamda, min_lambda, max_lambda)


def make_history():
    return {'sparsity': [], 'lambda': [], 'mean_w': [], 'mean_v': [],
            'max_w': [], 'snapshots': {}}


def record(history, w, v, lamda, t):
    sparsity = np.mean(np.abs(w) < 1e-8)
    history['sparsity'].append(sparsity)
    history['lambda'].append(lamda)
    history['mean_w'].append(np.mean(np.abs(w)))
    history['mean_v'].append(np.mean(np.abs(v)))
    history['max_w'].append(np.max(np.abs(w)))
    if t in SNAPSHOT_STEPS:
        history['snapshots'][t] = w.copy()
    return sparsity


# ── Generic runner ──────────────────────────────────────────────────
def run(optimizer, mode, w_init, all_grads):
    """
    optimizer: 'linbreg', 'adabreg', or 'proxsgd'
    mode: 'none', 'nesterov', or 'subgrad_correction'
          (rescale modes are Bregman-specific; 'proxsgd' should only use 'none')
    """
    w = w_init.copy()
    if optimizer == 'linbreg':
        lamda = lambda_init_linbreg
    elif optimizer == 'adabreg':
        lamda = lambda_init_adabreg
    else:  # proxsgd
        lamda = lambda_init_proxsgd
    prev_lamda = lamda
    # ProxSGD has no dual variable; keep v=0 so mean_v recording stays meaningful.
    if optimizer == 'proxsgd':
        v = np.zeros(N)
    else:
        v = (1.0 / delta) * w + lamda * np.sign(w)

    # Adam state
    m = np.zeros(N)
    m2 = np.zeros(N)

    history = make_history()

    for t in range(T):
        sparsity = record(history, w, v, lamda, t)

        step = t + 1
        grad = all_grads[t]

        # Method 1, 2 & 4 (before the dual step)
        if optimizer != 'proxsgd' and mode == 'subgrad_correction':
            if lamda != prev_lamda:
                ratio = lamda / (prev_lamda + 1e-12)

                # Method 1: Linear combination (valid sub-gradient)
                if SUBGRAD_MATHOD == 'method1':
                    v = ratio * v + (1 - ratio) * w
                # Method 2: correction by clipping smaller weights (also valid sub-gradient)
                if SUBGRAD_MATHOD == 'method2':
                    idx = np.abs(w) > 0.
                    v[idx]  = ratio * v[idx] + (1 - ratio) * w[idx]
                    v[~idx] = np.clip(v[~idx], -lamda, lamda)
                # Method 4: Predict w' from pre-gradient v under λ_new, then apply
                # the fixed-point difference. Placed before the dual step.
                if SUBGRAD_MATHOD == 'method4':
                    w_pred = prox_l1(delta * v, delta, lamda)
                    v += (w_pred - w) / delta \
                       + lamda * np.sign(w_pred) - prev_lamda * np.sign(w)

        # ── Dual update ──
        # For method4_adaptive + adabreg: compute moments first, peek at
        # adam_step, apply correction, then do the dual update.
        if optimizer == 'linbreg':
            # method4_adaptive falls back to standard method4 for LinBreg
            if mode == 'subgrad_correction' and SUBGRAD_MATHOD == 'method4_adaptive' and lamda != prev_lamda:
                w_pred = prox_l1(delta * v, delta, lamda)
                v += (w_pred - w) / delta \
                   + lamda * np.sign(w_pred) - prev_lamda * np.sign(w)
            v -= lr_linbreg * grad
        
        elif optimizer == 'adabreg':
            m = beta1 * m + (1 - beta1) * grad
            m2 = beta2 * m2 + (1 - beta2) * grad**2
            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step
            denom = np.sqrt(m2 / bc2) + eps
            step_size = lr_adabreg / bc1
            # method4_adaptive: peek at Adam step for better w_pred
            if mode == 'subgrad_correction' and SUBGRAD_MATHOD == 'method4_adaptive' and lamda != prev_lamda:
                adam_step = step_size * m / denom
                v_post = v - adam_step
                w_pred = prox_l1(delta * v_post, delta, lamda)
                v += (w_pred - w) / delta \
                   + lamda * np.sign(w_pred) - prev_lamda * np.sign(w)
            v -= step_size * m / denom

        # ── Subgradient correction (AFTER dual step, additive form) ──
        # Method 3: rescaling non-zero by lambdas difference (after the dual step)
        if SUBGRAD_MATHOD == 'method3':
            if mode == 'subgrad_correction' and lamda != prev_lamda:
                dlam = lamda - prev_lamda
                idx = np.abs(w) > 0.
                v[idx] += dlam * np.sign(w[idx])

        # ── Primal update (mode-dependent) ──
        if optimizer == 'proxsgd':
            # ProxSGD with pure L1:
            #   w ← w - lr·∇L;  w ← soft_threshold(w, lr·λ)
            # threshold = lr·λ follows from prox at stepsize α for argmin L(w)+λ‖w‖₁.
            w = soft_threshold(w - lr_proxsgd * grad, lr_proxsgd * lamda)
        else:
            w = prox_l1(delta * v, delta, lamda)
            if mode == 'nesterov':
                w = w / (lamda + 1e-12)

        # ── Lambda update ──
        prev_lamda = lamda
        if optimizer != 'proxsgd':
            lamda = lambda_step(lamda, sparsity, t)
        else:
            if step > proxsgd_warmup_steps:
                lamda = lambda_step(lamda, sparsity, t)

    # Final record
    record(history, w, v, lamda, T)
    return history


# ── Run all configurations ────────────────────────────────────────
# LinBreg/AdaBreg × {none, nesterov, subgrad_correction}, plus ProxSGD/none.
configs = OrderedDict()
for opt in ['linbreg', 'adabreg']:
    for mode in ['none', 'nesterov', 'subgrad_correction']:
        label = f"{opt}/{mode}"
        print(f"Running {label}...")
        configs[label] = run(opt, mode, w_init, all_grads)

# ProxSGD has no dual variable, so rescale modes don't apply — only 'none'.
print("Running proxsgd/none...")
configs['proxsgd/none'] = run('proxsgd', 'none', w_init_proxsgd, all_grads)

# ── Color/style scheme ─────────────────────────────────────────────
STYLES = {
    'linbreg/none':                {'color': '#2166ac', 'ls': '-',  'label': 'LinBreg/none'},
    'linbreg/nesterov':            {'color': "#7991a9", 'ls': '--', 'label': 'LinBreg/nesterov'},
    'linbreg/subgrad_correction':  {'color': "#022547", 'ls': ':',  'label': 'LinBreg/subgrad'},
    'adabreg/none':                {'color': '#b2182b', 'ls': '-',  'label': 'AdaBreg/none'},
    'adabreg/nesterov':            {'color': '#b2182b', 'ls': '--', 'label': 'AdaBreg/nesterov'},
    'adabreg/subgrad_correction':  {'color': "#54030d", 'ls': ':',  'label': 'AdaBreg/subgrad'},
    'proxsgd/none':                {'color': '#1a7a3e', 'ls': '-',  'label': 'ProxSGD/none'},
}


def plot_all(ax, key, xlim=None):
    for name, h in configs.items():
        s = STYLES[name]
        data = h[key][:-1] if key in ('sparsity', 'lambda') else h[key]
        ax.plot(data, color=s['color'], linestyle=s['ls'], linewidth=1.2, label=s['label'])
    if xlim:
        ax.set_xlim(xlim)


# ── Figure 1: Full run overview ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# Sparsity
ax = axes[0, 0]
plot_all(ax, 'sparsity')
ax.axhline(target_sparsity, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Sparsity')
ax.set_title('Sparsity Convergence')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Lambda
ax = axes[0, 1]
plot_all(ax, 'lambda')
ax.set_xlabel('Step')
ax.set_ylabel('λ')
ax.set_title('Lambda Evolution')
ax.set_yscale('log')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Mean |w|
ax = axes[0, 2]
plot_all(ax, 'mean_w')
ax.set_xlabel('Step')
ax.set_ylabel('mean |w|')
ax.set_title('Mean Weight Magnitude')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Max |w|
ax = axes[1, 0]
plot_all(ax, 'max_w')
ax.set_xlabel('Step')
ax.set_ylabel('max |w|')
ax.set_title('Max Weight Magnitude')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Mean |v| (subgradient)
ax = axes[1, 1]
plot_all(ax, 'mean_v')
ax.set_xlabel('Step')
ax.set_ylabel('mean |v|')
ax.set_title('Mean Subgradient Magnitude')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Sparsity zoom (first 1000 steps)
ax = axes[1, 2]
plot_all(ax, 'sparsity', xlim=(0, SPARSITY_ZOOM))
ax.axhline(target_sparsity, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Sparsity')
ax.set_title(f'Sparsity (first {SPARSITY_ZOOM} steps)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/bregman_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Weight distributions at key snapshots ────────────────
dist_configs = ['linbreg/none',  'linbreg/nesterov' , 'linbreg/subgrad_correction' , 'adabreg/none', 'adabreg/nesterov', 'adabreg/subgrad_correction', 'proxsgd/none']
# dist_configs = ['linbreg/none',  'linbreg/subgrad_correction' , 'adabreg/none', 'adabreg/subgrad_correction']
dist_steps = [0, 5000, T - 1]

fig2, axes2 = plt.subplots(len(dist_configs), len(dist_steps), figsize=(18, 12))
fig2.suptitle('Weight Distributions at Key Steps', fontsize=13, fontweight='bold')

for row, cname in enumerate(dist_configs):
    h = configs.get(cname, None)
    if h is None:
        continue
    s = STYLES[cname]
    for col, t in enumerate(dist_steps):
        ax = axes2[row, col]
        if t in h['snapshots']:
            ws = h['snapshots'][t]
            nonzero = ws[np.abs(ws) > 1e-8]
            zero_frac = 1 - len(nonzero) / len(ws)
            ax.hist(ws, bins=100, density=True, alpha=0.7, color=s['color'], edgecolor='none')
            if len(nonzero) > 0 and len(nonzero) < len(ws):
                ax.hist(nonzero, bins=50, density=True, alpha=0.5, color='orange',
                        edgecolor='none', label=f'nonzero ({len(nonzero)})')
                ax.legend(fontsize=8)
            ax.set_title(f'{s["label"]} t={t} (sparse={zero_frac:.1%})')
        else:
            ax.set_title(f'{s["label"]} t={t} (no snapshot)')
        ax.set_xlabel('Weight value')
        if col == 0:
            ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/bregman_simulation_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Print summary table ────────────────────────────────────────────
print()
print("=" * 110)
print("BREGMAN SIMULATION SUMMARY")
print("=" * 110)
print(f"N={N}, T={T}, δ={delta}, lr (LinBreg)={lr_linbreg}, lr (AdaBreg)={lr_adabreg}, lr (ProxSGD)={lr_proxsgd}, σ_w={sigma_w}, σ_grad={sigma_grad}")
print(f"λ₀ LinBreg={lambda_init_linbreg}, λ₀ AdaBreg={lambda_init_adabreg}, λ₀ ProxSGD={lambda_init_proxsgd}, target={target_sparsity}, accel={acceleration_factor}")
print()

# Per-config summary
for name, h in configs.items():
    final_s = h['sparsity'][-1]
    final_l = h['lambda'][-1]
    converged = abs(final_s - target_sparsity) < 0.02
    first_reach = next((t for t, s in enumerate(h['sparsity']) if s >= target_sparsity - 0.01), None)
    last_s = h['sparsity'][-500:]
    osc = max(last_s) - min(last_s)
    last_l = h['lambda'][-500:]
    l_ratio = max(last_l) / (min(last_l) + 1e-12)

    print(f"  {STYLES[name]['label']:.<30s} sparsity={final_s:.4f}  λ={final_l:.4f}  "
          f"max|w|={h['max_w'][-1]:.4f}  mean|w|={h['mean_w'][-1]:.6f}  "
          f"converged={'Y' if converged else 'N'}  "
          f"first@1%={first_reach}  osc={osc:.4f}  λ_ratio={l_ratio:.2f}x")

# Detailed step-by-step for each config
print()
print("-" * 110)
hdr = f"{'Step':>6}"
for name in configs:
    hdr += f" | {STYLES[name]['label']:>18s}"
print(f"{'':>6}   {'Sparsity':^110s}")
print(hdr)
print("-" * 110)
for t in PRINT_STEPS:
    if t >= T:
        continue
    row = f"{t:>6d}"
    for name, h in configs.items():
        row += f" | {h['sparsity'][t]:>8.4f} λ={h['lambda'][t]:>7.3f}"
    print(row)
print("-" * 110)
