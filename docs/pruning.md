# Neural Network Compression Methods

This project implements two complementary approaches for neural network compression to create efficient speaker recognition models:

1. **Bregman Learning Framework** - A sparsity-inducing training approach
2. **Magnitude-Based Pruning** - A classical pruning method with advanced scheduling

---

## 1. Bregman Learning Framework

### Overview

The Bregman learning framework implements sparsity-inducing optimization during training, based on the work by [Bungert et al. (2022)](https://www.jmlr.org/papers/volume23/21-0545/21-0545.pdf). This approach achieves compression by incorporating adaptive regularization that encourages sparse weight distributions while maintaining model performance.

**Reference Implementation:** [TimRoith/BregmanLearning](https://github.com/TimRoith/BregmanLearning/tree/main/notebooks)

### Key Components

#### 1.1 Bregman Optimizers

Located in `src/callbacks/pruning/bregman/bregman_optimizers.py`:

- **LinBreg**: Linear Bregman iteration — momentum-based variant of gradient descent
- **AdaBreg**: Adaptive Bregman iteration — Adam-style per-parameter step sizes
- **ProxSGD**: Proximal SGD baseline (thresholds the weights, not the dual)

All of them support **parameter groups** with different regularization strategies.

##### Weight decay (μ) in a Bregman iteration

Notation: `τ = lr`, `δ = delta`, `μ = weight_decay`, `m = momentum`,
`S_a(x) = sign(x)·max(|x| − a, 0)`.

**The iteration is dual.** With the elastic net `J(w) = λ‖w‖₁ + ‖w‖₂²/(2δ)`,
LinBreg and AdaBreg carry `v ∈ ∂J(w)` as their state and read `w` back off it:

```text
v⁰ = w⁰/δ + λ·sign(w⁰) ∈ ∂J(w⁰)      # initialize_sub_grad
v ← v − τ·g                           # sub_grad.add_(d_grad, alpha=-lr)
w ← ∇J*(v) = S_δλ(δ·v)                # p.copy_(reg.prox(delta * sub_grad, delta))
```

###### The master equation

Fix a coordinate on the support and hold `s = sign(v)`. There the prox is affine
in `v`, and a difference cancels its constant offset. `g` is the momentum- or
Adam-processed loss step alone — LinBreg and AdaBreg add `μw` to the dual as a
separate term (AdamW-style: Loshchilov & Hutter, ICLR 2019), so it never shares
`g`'s buffer:

```text
(A)   w  = δ·(v − λ·s)
(B)   Δw = δ·Δv                       # the λ·s offset drops out
(M)   Δw = −δτ·g − δτμ·w              # since Δv = −τ·g − τμ·w
```

**Inside a sign cell, LinBreg is gradient descent on `w` with step `δτ`, plus a
decay term independent of `g`.** λ is absent from (M): it acts only when a
coordinate changes cell — enters or leaves the support. That is the debiasing,
and every result below is a substitution into (M).

| `g` | `Δw = −δτ·g − δτμ·w` gives |
| --- | --- |
| `∇L` (momentum = 0) | `w⁺ = (1 − δτμ)·w − δτ·∇L` |
| `b`, momentum's buffer of `∇L` alone | `w⁺ = (1 − δτμ)·w − δτ·b` — the same `(1 − δτμ)` factor, for any momentum, since `μw` never enters `b` |
| `∇L = 0` (any momentum) | `w⁺ = (1 − δτμ)·w` — ordinary multiplicative decay |
| `w = 0`, off the support | `μ·w = 0`; the decay term is absent, no mask needed |

Stationarity follows too: `Δw = 0 ⟺ g = −μw`, which is stationarity of
`L + μ‖w‖₂²/2`. At `∇L = 0` that forces `w = 0`, i.e. `|v| ≤ λ` — μ drives a
coordinate onto the threshold and out of the support, so it raises the sparsity a
given λ reaches. `fixed_lambda` is therefore calibrated per μ; the adaptive arms
re-derive λ online and are unaffected.

###### Why μ must enter the dual update, before the prox

```text
(G)   w = ∇J*(v)                      # w is a function of v alone: a post-prox
                                      # edit to w is overwritten at the next step
(H)   J + μ‖w‖₂²/2                    # sends 1/(2δ) to 1/(2δ) + μ/2, i.e.
                                      # δ to δ/(1 + δμ). J enters only through
                                      # ∇J*, so this rescales δ, constant in k
```

μ must land in the dual update `v ← v − τ(...)`, before the prox — (G)/(H) rule
out any post-prox edit to `w`. It does not have to share a buffer with `g`:
LinBreg and AdaBreg add it as an independent term, so every step it reaches the
dual at the full rate `τμ`, never smeared by momentum or normalized by Adam's
denominator. Equivalently: the loss part of the iteration is unchanged, run on
`L` (not `L + μ‖w‖₂²/2` — μ is no longer folded into the smooth loss term, it
sits beside it), which keeps the published convergence result for the `∇L`
part untouched by μ.

###### AdaBreg: μ is decoupled from Adam's normalization

`exp_avg`/`exp_avg_sq` are built from `∇L` alone — `μw` never enters them, so
Adam's `√Ŝ` denominator never sees it. Substituting `g = adam_step =
m̂/(√Ŝ + ε)` into (M):

```text
(D)   w⁺ = (1 − δτμ)·w − δτ·adam_step
```

At `∇L = 0`, `exp_avg` and `exp_avg_sq` stay at `0` forever — nothing ever
feeds them — so `adam_step = 0/ε = 0` and:

```text
(E)   w⁺ = (1 − δτμ)·w                # the same identity as LinBreg
```

μ is a fixed rate here, exactly as it is for LinBreg — not a share of Adam's
numerator that vanishes into `sign(w)` regardless of magnitude.

###### What is established, and what is not

(D) shows the `(1 − δτμ)` rate is the same substitution into (M) whatever `g`
is — a property of the dual update, true by construction: the momentum buffer
and the Adam moments only ever accumulate `∇L`, so `μw`'s contribution to (M)
never touches them. That is definitional, verifiable by reading the code, and
does not depend on being on the support.

Its translation into a *w-space* claim — "LinBreg and AdaBreg decay `w`
identically" — does depend on the support precondition, same as every other
row in the master-equation table. It is not left as an analogy:
`test_decay_only_trajectory_is_identical_across_momentum_and_adam`
(`tests/test_bregman_optimizer_correctness.py`) drives LinBreg (no momentum),
LinBreg (momentum=0.9) and AdaBreg for 60 steps at `∇L = 0` and asserts the
three trajectories are bit-for-bit equal (`torch.equal`, not `allclose`) —
the momentum buffer and the Adam moments both collapse to an exact `+0.0`
contribution when fed nothing but zeros, so all three machineries reduce to
the identical no-op and only the shared `−τμ·w` term survives.

What this does not cover: a coordinate crossing the support boundary or
flipping sign, where (B) fails. Decay alone cannot cause that crossing —
`v_{k+1} = v_k(1 − τμδ) + τμδλs` is a contraction toward the fixed point
`v* = λs`, so for `0 < τμδ < 1` it approaches the boundary geometrically and
never reaches it in exact arithmetic (confirmed numerically: 60 steps at
`τμδ = 0.05` leaves every coordinate strictly inside the support). A crossing
needs `∇L` or a λ update to push past it, and that mechanism is identical for
LinBreg and AdaBreg — the prox and λ are shared, so a crossing is symmetric
across optimizers, not something this decoupling treats differently.

###### ProxSGD: the offset does not cancel

The prox is applied to the iterate, not to the state, so (B) never happens:

```text
(F)   w⁺ = S_τλ(w − τ·g) = w − τ·g − τλ·s
```

Side by side, the `−τλ·s` is the whole difference:

```text
LinBreg   Δw = −δτ·g
ProxSGD   Δw = −τ·g − τλ·s
```

That term is the LASSO bias, re-applied every step. LinBreg pays it once, as a
coordinate change in (A).

###### In the recipes

μ defaults to `0.0`, the published algorithm. Every arm is configured with the L2
of the baseline it is compared against — `5e-4` on the img recipes, `1e-4` on the
SV ones — so on the support the two decay identically, and off it Bregman decays
nothing, because there is nothing there. No recipe overrides `delta`, so `δ = 1`
and the rate is `τμ`, regardless of momentum.

#### 1.2 Bregman Regularizers

Located in `src/callbacks/pruning/bregman/bregman_regularizers.py` (others are not used):

- **RegNone**: No regularization (standard training)
- **RegL1**: L1 norm regularizer for unstructured sparsity
- **RegL1L2Conv**: Specialized group sparsity for convolutional layers

Each regularizer implements:
- `__call__(x)`: Computes regularization value
- `prox(x, delta)`: Proximal operator for Bregman updates
- `sub_grad(v)`: Subgradient for gradient-based methods

#### 1.3 Lambda Scheduler

Located in `src/callbacks/pruning/bregman/lambda_scheduler.py`:

A feedback controller on the regularization strength λ. It carries its own setpoint from config; `BregmanPruner` only steps it every `update_frequency` steps.

```python
lambda_scheduler:
  _target_: src.callbacks.pruning.bregman.lambda_scheduler.LambdaScheduler
  target_sparsity: 0.9       # setpoint λ drives toward
  initial_sparsity: 0.99     # where the run starts, so the opening gap is known
  initial_lambda: 1e-2
  alpha_0: 1.0               # the step size before any decay
  update_frequency: 50       # steps between λ updates
```

**The update.** With `gap = target − sparsity`, λ is multiplied by `1 + α·|gap|` while the model is short of the target and divided by it once past. The move is relative, so the controller behaves the same at any λ scale and reaches any setpoint from any `initial_lambda`.

**The step size.** `α = alpha_0 · gamma^max(C − 1, 0)`, where `C` counts the updates whose gap changed sign. Each overshoot shrinks the steps, so λ settles instead of ringing and α tends to zero over a long run. A gap that shrinks, grows or hovers keeps its sign and leaves α alone. The model starts at `initial_sparsity`, on one side of the setpoint, so crossing #1 is the approach and not an overshoot — at `initial_sparsity: 0.99` and `target_sparsity: 0.9` the support grows through the target in the first updates, and that crossing costs no α. `gamma` defaults to 0.95 and is a constructor argument only — not yet wired to a config key.

λ steers on the sparsity of every weight tensor — all but norms and biases (`WHICH_SPARSITY_PERCENTAGE` in `bregman_pruner.py`, default `pruned`), so `target_sparsity: 0.9` means the same thing as the magnitude pruner's `amount: 0.9` and RigL's. Switching the constant to `overall` means pointing `_bregman_sparsity_metric` at `sparsity` instead.

Metrics: `bregman/global_lambda` (live λ), `bregman/lambda_delta` (applied `Δλ`, zero between updates), `bregman/lambda_delta_over_lambda` (the relative move), `bregman/lambda_gap` (`target − sparsity` at the last update), `bregman/lambda_crossings` (`C`), `bregman/alpha` (the α of the last update).

**Note**: the target is not guaranteed to be reached. The optimizer term and the regularizer term pull against each other in the weight update, so the Bregman optimizer, `lr`, and the LR schedule all move the reachable sparsity.

#### 1.3b Target ramp

`TargetScheduler` (`src/callbacks/pruning/bregman/target_scheduler.py`) is a separate callback that raises the controller's setpoint from `initial_sparsity` to its configured target over `epochs_to_ramp` epochs, then holds. It reads the endpoint off `BregmanPruner.lambda_scheduler` at train start and writes one float on it per epoch — nothing in the Bregman stack imports it.

```yaml
callbacks:
  target_scheduler:
    _target_: src.callbacks.pruning.bregman.target_scheduler.TargetScheduler
    initial_sparsity: 0.5 # ramp start; the end is the controller's target
    epochs_to_ramp: 80
    schedule_type: cubic # Zhu & Gupta / GraNet Eq. 1 cubic ramp (default), or `linear`/`constant`
```

The ramp drives `PruningScheduler` (§2.3) itself, so a Bregman ramp and a gradual-pruning run that share `initial_sparsity` and length aim at the same sparsity in the same epoch — that comparison is what the recipes are for, and the img recipes are aligned for it (see [image_benchmarks.md](image_benchmarks.md)). Metric: `bregman/ramp_target`.

```bash
python src/train.py experiment=img/bregman_adabreg_progressive datamodule=datasets/cifar100
```

#### 1.4 Pruning Manager

Located in `src/callbacks/pruning/utils/pruning_manager.py`:

Manages parameter groups and applies structured/unstructured pruning based on sparsity thresholds (for the algorithm's initalization). The fine0grained control of initilization might be later deprecated and hardcode untructured pruning as it could be an overkill.

#### 1.5 BregmanPruner Callback

Located in `src/callbacks/pruning/bregman/bregman_pruner.py`:

Orchestrates the entire Bregman learning process:
- Applies the initial sparsity via the pruning manager
- Refuses at fit start an `overall`-steered `target_sparsity` above the prunable fraction, which the gates could never reach
- Steps the lambda scheduler each batch and broadcasts `λ · lambda_scale` to the groups
- Publishes `sparsity` (whole-model) and `bregman/pruned_sparsity` (all weight tensors, norms and biases aside) for the gates and the run artifacts
- Handles checkpoint save/load of the scheduler state

Everything the callback writes out — the per-step metric series and the fit-start configuration/group dumps — lives in `bregman_report.py`; the "is this group actively regularized" predicates live in `bregman_regularizers.py`. Neither affects training.

### Usage Example

`configs/experiment/sv/sv_bregman_adabreg.yaml` (SV) and `configs/experiment/img/bregman_adabreg.yaml` (image) are the canonical recipes. Both declare the target once and reuse it for the controller and the gates:

```yaml
callbacks:
  model_pruning:
    _target_: src.callbacks.pruning.bregman.bregman_pruner.BregmanPruner
    target_sparsity: ${_bregman_target_sparsity} # setpoint + gate band centre, over all weight tensors
    sparsity_threshold: 1e-12
    verbose: 2
    lambda_scheduler: # null for a fixed lambda (the *_fixed variants)
      _target_: src.callbacks.pruning.bregman.lambda_scheduler.LambdaScheduler
      _partial_: true
      initial_lambda: ${_bregman_lambda}
      target_sparsity: 0.9
      initial_sparsity: 0.99
      alpha_0: 1.0
      update_frequency: 50

module:
  optimizer:
    _target_: src.callbacks.pruning.bregman.bregman_optimizers.AdaBreg
    _partial_: true
    lr: 1e-2

  model:
    pruning_groups:
      - name: conv_layers
        layer_types: ["torch.nn.Conv1d", "torch.nn.Conv2d"]
        param_names: ["weight"]
        optimizer_settings:
          reg:
            _target_: src.callbacks.pruning.bregman.bregman_regularizers.RegL1
            lamda: ${_bregman_lambda}
          lambda_scale: 1.0
        pruning_config:
          pruning_type: "unstructured"
          sparsity_rate: ${_bregman_initial_sparsity}

      - name: norm_params # kept dense: RegNone + scale 0
        layer_types: ["torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d", "torch.nn.LayerNorm"]
        optimizer_settings:
          reg:
            _target_: src.callbacks.pruning.bregman.bregman_regularizers.RegNone
          lambda_scale: 0.0
```

Every regularized group must set `lambda_scale`, and the last group must be the `is_fallback: True` catch-all. See the config itself for the full group list (linear, classifier, bias, fallback).

### Training Workflow

1. **Initialization**: Model parameters are assigned to groups based on layer type and name patterns
2. **Training Loop**:
   - Forward pass computes loss
   - Backward pass computes gradients
   - Bregman optimizer applies proximal operator using the regularizer
   - Lambda scheduler adjusts regularization strength based on current sparsity
3. **Sparsity Tracking**: BregmanPruner logs sparsity metrics per group and overall
4. **Checkpoint Handling**: Scheduler state is saved/restored for resuming training
---

## 2. Magnitude-Based Pruning

### Overview

Classical pruning method that removes weights with the smallest magnitudes, either all at once or gradually over training epochs. This implementation includes advanced features like checkpoint compatibility and metric tracker management.

### Key Components

#### 2.1 MagnitudePruner Callback

Located in `src/callbacks/pruning/prune.py`:

Main callback that orchestrates the pruning process with the following features:

- **Pruning Methods**: L1 unstructured, L1 structured, Ln structured
- **Scheduled Pruning**: Gradual sparsity ramping over epochs
- **Global/Local Pruning**: Prune across all parameters or per-layer
- **Permanent Pruning**: Option to make pruning permanent at training end
- **Checkpoint Compatibility**: Handles resumption from pruned checkpoints

**Key Parameters:**
```python
callbacks:
  model_pruning:
    _target_: src.callbacks.pruning.prune.MagnitudePruner
    pruning_fn: "l1_unstructured"    # Pruning strategy
    amount: 0.5                      # 50% final sparsity
    initial_amount: 0.5              # Sparsity the ramp starts from (img recipes: 0.5)
    scheduled_pruning: true          # Enable gradual ramping
    schedule_type: "cubic"           # The rate of increasing sparsity [linear, constant, cubic] (default: cubic)
    epochs_to_ramp: 10               # Epochs to reach target sparsity
    use_global_unstructured: true    # Global vs local pruning
    make_pruning_permanent: true     # Fuse masks at training end
    min_param_elements: 100          # Skip layers with small number of parameters
    verbose: 1
```

#### 2.2 Parameter Manager

Located in `src/callbacks/pruning/parameter_manager.py`:

Manages parameter selection and validation using a hybrid strategy:

1. **Allowlist**: Standard prunable layers (Conv, Linear, LSTM, GRU, Embedding)
2. **Blocklist**: Protected layers (BatchNorm, LayerNorm, etc.)
3. **Duck Typing**: Custom layers not in Blocklist with `weight` parameter are considered prunable

Features:
- Automatic parameter discovery
- Size-based filtering (skip small parameters)
- Bias handling (optional)
- Detailed logging of prunable and skipped parameters

#### 2.3 Pruning Scheduler

Located in `src/callbacks/pruning/scheduler.py`:

Implements various sparsity ramping schedules:

- **Linear**: Uniformly increase from initial to final sparsity
- **Constant**: Prune the same amount of weights in each epoch
- **Cubic** (default): Zhu & Gupta's GMP schedule, `S_t = S_f + (S_i - S_f)(1 - progress)^3` — also GraNet's Eq. 1 (`dst_schedules.cubic_prune_rate`)


```python
scheduler = PruningScheduler(
    schedule_type="cubic",
    final_sparsity=0.8,
    epochs_to_ramp=20,
    initial_sparsity=0.0
)
```

#### 2.4 Checkpoint Handler

Located in `src/callbacks/pruning/checkpoint_handler.py`:

It has two primary tasks:
1. Handling the saving/loading ckpt when training was interrupted during training (e.g., loading state dict of the model and optimizer).
2. It tracks the pruning schedule and ensures it is resumed from where it stopped.

Both goals ensure seamless loading of pruned checkpoints into unpruned models:

- Detects pruned checkpoints (parameters ending with `_orig`)
- Reconstructs pruning structure with `Identity` masks
- Auto-fuses weights when loading pruned checkpoint into clean model
- Maintains parameter order for optimizer compatibility

### Pruning Workflow

#### Training from Scratch

1. **Epoch Start** (`on_train_epoch_start`):
   - Compute target sparsity from scheduler
   - Remove existing masks (if resuming)
   - Apply pruning to reach target
   - Verify sparsity jump is monotonic

2. **Epoch End** (`on_train_epoch_end`):
   - Publish `pruning/sparsity` (all weight tensors, norms and biases aside) and `sparsity` (whole model) for the gates

3. **Training End** (`on_train_end`):
   - Optionally make pruning permanent (fuse masks into weights)

#### Resuming from Checkpoint

1. **Load Checkpoint** (`on_load_checkpoint`):
   - Restore pruning structure via `PrunedCheckpointHandler`
   - Load scheduler state
   - Reconstruct `Identity` masks for pruned parameters

2. **Continue Training**:
   - Scheduler continues from saved state
   - Sparsity maintained or increased (never decreased)

### Sparsity Enforcement

- `on_train_epoch_start` refuses to prune when the measured sparsity already exceeds `amount * (1 + tolerance)`.
- `_verify_sparsity_jump` refuses a re-prune that drops sparsity by more than 5 points from above 0.1.

---

## The stem: `prune_first_layer`

`callbacks.model_pruning.prune_first_layer` drops the model's first weight tensor from the target list. Every recipe sets `true`, because every method's own implementation sparsifies the stem:

| Recipe | Source | Evidence |
| --- | --- | --- |
| `pruning_rigl`, `pruning_set`, `pruning_static` | google-research/rigl | the dense stem is the paper's **Uniform** rule (sec. 3); the recipes run ERK, which "applies to all layers" (README) |
| `pruning_granet` | VITA-Group/GraNet | `--rm-first` is `store_true` and no published command passes it |
| `pruning_snip`, `pruning_snip_iter` | namhoonlee/snip-public, naver/force | SNIP masks every `w`/`b` key; FORCE takes every `nn.Conv2d`/`nn.Linear` |
| `soft_threshold` | RAIVNLab/STR | `--first-layer-dense` defaults to `False`; STR's own ResNet-50 budget puts `conv1` at 51–78 % sparse |
| `pruning_mag_*`, `sv_pruning_mag_*` | Gale et al. 2019 | GMP sparsifies "all convolutional and fully-connected layers" |

The Bregman stack has no such flag: `TimRoith/BregmanLearning` excludes no layer, so every weight tensor carries the regularizer.

`stem_weight` reads the module walk, not a layer-type list and not a filtered target list: the first module with a trainable `weight` Parameter that is not a norm layer. So a custom encoder stem counts, and a stem below `min_param_elements` costs only itself.

At `false` the held-dense tensor stays in the reported denominator at full size, spending its share of the budget — which is how RigL accounts for a dense first layer, and why `pool_sparsity` scales the target the callback actually applies.

---

## 3. Sparsity Gating

Located in `src/callbacks/pruning/sparsity_gating.py`, shared by both stacks. Off-target epochs must not be selected on, so three callbacks read the sparsity the pruner publishes each epoch and gate on it:

| Callback | Out of band |
|---|---|
| `SparsityGatedModelCheckpoint` | skips top-k saving; `last.ckpt` keeps saving, so resume is intact |
| `SparsityGatedEarlyStopping` | skips the check, so patience never accrues on an off-target metric |
| `RampValidationGate` | zeroes `limit_val_batches`, skipping the validation pass |

**The band is relative:** in band iff `(1 - tolerance) * target <= sparsity <= (1 + tolerance) * target`. Each gate takes its own `tolerance`; `1.0` spans every sparsity, which disables that gate.

The two selection gates take `tolerance: 0.005` (0.5%, declared once as `_bregman_tolerance` / `_pruning_tolerance`) on every recipe whose sparsity is controlled, so an off-target epoch never becomes the evaluated model. `RampValidationGate` takes `1.0` on the img recipes, so the per-epoch curve is measured over the whole ramp; it takes the same 0.5% band where a validation pass is too expensive to spend off-target — ImageNet, and every SV recipe. The `*_fixed` recipes open all three: a static λ reaches whatever sparsity it reaches, so there is no setpoint to band around.

Point all three at the same metric as the pruner steers on: `bregman/pruned_sparsity` (Bregman) or `pruning/sparsity` (magnitude); `sparsity` is whole-model.

---

## Comparison: Bregman vs Magnitude Pruning

| Aspect | Bregman Learning | Magnitude Pruning |
|--------|------------------|-------------------|
| **Approach** | Regularization-based | Weight removal |
| **Timing** | During training | Before/during training |
| **Adaptivity** | Automatic λ adjustment | Manual schedule design |
| **Granularity** | Parameter group control | Global/local options |
| **Flexibility** | Multiple regularizer types | Limited to magnitude |
| **Recommended Use** | Training from scratch | Fine-tuning or iterative pruning |


---

## Troubleshooting

### Bregman Learning

**Issue**: Sparsity not reaching target
- **Solution**: Increase `alpha_0`, or lower `update_frequency` so λ updates more often

**Issue**: Training unstable
- **Solution**: Decrease `initial_lambda` or try using a different regularizer (e.g., `RegL1L2` instead of `RegL1`)

**Issue**: Some layers not being pruned
- **Solution**: Check `lambda_scale` in group config (should be > 0)

### Magnitude Pruning

**Issue**: Checkpoint loading fails
- **Solution**: Ensure `PrunedCheckpointHandler` is in callbacks list

**Issue**: Sparsity jumps unexpectedly
- **Solution**: Check for checkpoint resumption issues; verify scheduler state

**Issue**: Early stopping triggered during ramp-up
- **Solution**: Increase `patience` or ensure metric tracker management is enabled

---

## File Structure

```
src/callbacks/pruning/
├── bregman/
│   ├── bregman_optimizers.py      # LinBreg, AdaBreg, ProxSGD
│   ├── bregman_regularizers.py    # L1, L1L2, etc. regularizers
│   ├── bregman_pruner.py          # Main Bregman callback
│   └── lambda_scheduler.py        # Adaptive λ controller
├── utils/
│   ├── pruning_manager.py         # Parameter group management
│   └── sparsity_applier.py        # Initial sparsity per group
├── prune.py                        # MagnitudePruner callback
├── parameter_manager.py            # Parameter selection for magnitude pruning
├── scheduler.py                    # Sparsity scheduling
├── shared_prune_utils.py           # compute_sparsity, shared by both stacks
├── sparsity_gating.py              # Gated checkpoint/early-stop/validation
└── checkpoint_handler.py           # Checkpoint compatibility

scripts/
└── make_pruning_permanent.py      # Post-training weight fusion

configs/experiment/sv/
├── sv_bregman_adabreg.yaml        # Bregman parent (adaptive λ, AdaBreg)
├── sv_bregman_adabreg_fixed.yaml  # Same recipe, fixed λ
└── sv_pruning_mag_unstruct.yaml   # Magnitude pruning
```
