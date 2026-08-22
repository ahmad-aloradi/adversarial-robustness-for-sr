# Neural Network Compression Methods

Two sparsity stacks, both Lightning callbacks, both orthogonal to the task module:

1. **Bregman learning** — sparsity-inducing training ([Bungert et al., JMLR 2022](https://www.jmlr.org/papers/volume23/21-0545/21-0545.pdf); reference implementation [TimRoith/BregmanLearning](https://github.com/TimRoith/BregmanLearning))
2. **Magnitude pruning** — gradual magnitude pruning behind a sparsity ramp

Dynamic sparse training (RigL, SET, Static-ERK, SNIP, GraNet) and STR live in [sparse_training.md](sparse_training.md). The image recipes live in [image_benchmarks.md](image_benchmarks.md).

```bash
python src/train.py experiment=img/bregman_adabreg datamodule=datasets/cifar100
python src/train.py experiment=img/pruning_mag_unstruct datamodule=datasets/cifar100
python src/train.py experiment=sv/sv_bregman_adabreg
```

---

## 1. Bregman learning

### 1.1 Optimizers

`src/callbacks/pruning/bregman/bregman_optimizers.py`:

- **LinBreg** — linearized Bregman iteration, SGD momentum on the dual
- **AdaBreg** — the same iteration with Adam-style per-parameter steps
- **ProxSGD** — proximal SGD baseline; it thresholds `w`, not the dual

Each takes one `reg` and one `lambda_scale` per parameter group.

#### Weight decay (μ) in a Bregman iteration

Notation: `τ = lr`, `δ = delta`, `μ = weight_decay`, `S_a(x) = sign(x)·max(|x| − a, 0)`.

**The iteration is dual.** With the elastic net `J(w) = λ‖w‖₁ + ‖w‖₂²/(2δ)`, LinBreg and AdaBreg carry `v ∈ ∂J(w)` as their state and read `w` back off it:

```text
v⁰ = w⁰/δ + λ·sign(w⁰) ∈ ∂J(w⁰)      # initialize_sub_grad
v ← v − τ·g − τμ·w                    # sub_grad
w ← ∇J*(v) = S_δλ(δ·v)                # p.copy_(reg.prox(delta * sub_grad, delta))
```

**Where μ enters depends on the arm.** AdaBreg is always decoupled AdamW-style (Loshchilov & Hutter, ICLR 2019): `g` is the Adam step on `∇L` alone and `−τμ·w` is the separate term written above. LinBreg is coupled by default: `μ·w` joins `∇L` *inside* `g`, exactly as `torch.optim.SGD` does it, and `decoupled_weight_decay=True` moves it back out to the separate term.

Coupling matters because momentum integrates it. At the same nominal μ, coupled decay is `1/(1 − momentum)` times decoupled — 10x at `momentum=0.9`. LinBreg's default is coupled so that one `weight_decay` value means the same thing in a LinBreg arm and in the SGD baselines it is benchmarked against. AdaBreg has no coupled path, so at `lr=5e-3` its decay is ~100x weaker than a `lr=0.05` SGD baseline at the same μ; its μ must be raised to match. `test_linbreg_default_mu_matches_sgd_bit_for_bit` asserts the LinBreg default against `torch.optim.SGD` over 50 steps.

##### The master equation

Fix a coordinate on the support and hold `s = sign(v)`. There the prox is affine in `v`, and a difference cancels its constant offset:

```text
(A)   w  = δ·(v − λ·s)
(B)   Δw = δ·Δv                       # the λ·s offset drops out
(M)   Δw = −δτ·g − δτμ·w
```

**Inside a sign cell, LinBreg is gradient descent on `w` with step `δτ`, plus a decay term independent of `g`.** λ is absent from (M): it acts only when a coordinate enters or leaves the support. That is the debiasing, and every row below is a substitution into (M).

| `g` | `Δw = −δτ·g − δτμ·w` gives |
| --- | --- |
| `∇L` (momentum = 0) | `w⁺ = (1 − δτμ)·w − δτ·∇L` |
| `b`, momentum's buffer of `∇L` alone (`decoupled_weight_decay=True`) | `w⁺ = (1 − δτμ)·w − δτ·b` — the same `(1 − δτμ)` factor, for any momentum, since `μw` never enters `b` |
| `b`, momentum's buffer of `∇L + μw` (LinBreg's default) | `w⁺ = w − δτ·b`; `μw` is inside `b`, so the decay is smeared across steps and reaches `δτμ/(1 − momentum)` per step |
| `adam_step = m̂/(√Ŝ + ε)` | `w⁺ = (1 − δτμ)·w − δτ·adam_step` — `exp_avg`/`exp_avg_sq` see `∇L` alone, so Adam's denominator never divides μ |
| `g = 0` (any momentum, or Adam at `∇L = 0`) | `w⁺ = (1 − δτμ)·w` — ordinary multiplicative decay, identical across the three |
| `w = 0`, off the support | `μ·w = 0`; the decay term is absent, no mask needed |

`test_decay_only_trajectory_is_identical_across_momentum_and_adam` in `tests/test_bregman_optimizer_correctness.py` asserts row four bit-for-bit over 60 steps.

Stationarity follows too: `Δw = 0 ⟺ g = −μw`, which is stationarity of `L + μ‖w‖₂²/2`. At `∇L = 0` that forces `w = 0`, i.e. `|v| ≤ λ` — μ drives a coordinate onto the threshold and out of the support, so it raises the sparsity a given λ reaches. `fixed_lambda` is therefore calibrated per μ; the adaptive arms re-derive λ online and are unaffected.

##### Why μ must enter the dual update, before the prox

```text
(G)   w = ∇J*(v)                      # w is a function of v alone: a post-prox
                                      # edit to w is overwritten at the next step
(H)   J + μ‖w‖₂²/2                    # sends 1/(2δ) to 1/(2δ) + μ/2, i.e.
                                      # δ to δ/(1 + δμ). J enters only through
                                      # ∇J*, so this rescales δ, constant in k
```

(G) and (H) rule out any post-prox edit to `w`, in both arms. What they do not fix is whether μ shares a buffer with `g`. Decoupled, it reaches the dual every step at the full rate `τμ`, never smeared by momentum, never normalized by Adam's denominator. Coupled, it rides `g`'s buffer and inherits its gain.

##### ProxSGD: the offset does not cancel

The prox is applied to the iterate, not to the state, so (B) never happens:

```text
LinBreg   Δw = −δτ·g
ProxSGD   Δw = −τ·g − τλ·s            # w⁺ = S_τλ(w − τ·g)
```

The `−τλ·s` is the LASSO bias, re-applied every step. LinBreg pays it once, as a coordinate change in (A).

##### In the recipes

μ defaults to `0.0`, the published algorithm. Every arm takes the L2 of the baseline it is compared against — `5e-4` on the img recipes, `1e-4` on the SV ones. No recipe overrides `delta`, so `δ = 1` and the rate is `τμ`.

### 1.2 Regularizers

`bregman_regularizers.py` — **RegNone** (no thresholding), **RegL1** (unstructured), **RegL1L2Conv** (group sparsity per conv filter). Each supplies `__call__`, `prox` and `sub_grad`. A group holds its regularizer dense with `RegNone` plus `lambda_scale: 0.0`.

### 1.3 Lambda scheduler

`lambda_scheduler.py` — a feedback controller on λ, stepped every `update_frequency` steps by `BregmanPruner`.

**The update.** With `gap = target − sparsity`, λ is multiplied by `1 + α·|gap|` while the model is short of the target and divided by it once past. The move is relative, so the controller behaves the same at any λ scale and reaches any setpoint from any `initial_lambda`.

**The step size.** `α = alpha_0 · gamma^max(C − 1, 0)`, where `C` counts the updates whose gap changed sign. Each overshoot shrinks the steps, so λ settles instead of ringing. A gap that shrinks, grows or hovers keeps its sign and leaves α alone. The model starts at `initial_sparsity`, on one side of the setpoint, so crossing #1 is the approach and costs no α. `gamma` defaults to 0.95 and is a constructor argument only.

λ steers on the sparsity of every weight tensor — all but norms and biases (`WHICH_SPARSITY_PERCENTAGE` in `bregman_pruner.py`, default `pruned`), so `target_sparsity: 0.9` means the same as the magnitude pruner's `amount: 0.9` and RigL's.

`QuantileLambdaScheduler` is the alternative: it reads λ off the K-th order statistic of `|v|`, so the prox keeps exactly K weights instead of converging to a measured sparsity.

Metrics: `bregman/global_lambda`, `bregman/lambda_delta`, `bregman/lambda_delta_over_lambda`, `bregman/lambda_gap`, `bregman/lambda_crossings`, `bregman/alpha`.

**The target is not guaranteed.** The loss term and the regularizer pull against each other, so the optimizer, `lr` and the LR schedule all move the reachable sparsity.

### 1.4 Target ramp

`TargetScheduler` is a separate callback. It raises the controller's setpoint from `initial_sparsity` to the configured target over `epochs_to_ramp` epochs, then holds. It reads the endpoint off `BregmanPruner.lambda_scheduler` at train start and writes one float per epoch; nothing in the Bregman stack imports it.

The ramp drives `PruningScheduler` (§2), so a Bregman ramp and a gradual-pruning run that share `initial_sparsity` and length aim at the same sparsity in the same epoch. Metric: `bregman/ramp_target`.

```bash
python src/train.py experiment=img/bregman_adabreg_progressive datamodule=datasets/cifar100
```

### 1.5 BregmanPruner

`bregman_pruner.py` applies the initial sparsity through `PruningManager`, steps the λ scheduler, broadcasts `λ · lambda_scale` to every thresholding group, publishes `sparsity` and `bregman/pruned_sparsity`, and checkpoints the scheduler state. `bregman_report.py` holds everything it writes out; neither affects training.

`configs/experiment/img/bregman_adabreg.yaml` and `configs/experiment/sv/sv_bregman_adabreg.yaml` are the canonical recipes. Read the `pruning_groups` block there: every regularized group must set `lambda_scale`, and the last group must be the `is_fallback: True` catch-all.

---

## 2. Magnitude pruning

`src/callbacks/pruning/prune.py` — `MagnitudePruner` re-derives the masks at every epoch start through `torch.nn.utils.prune`. It takes `l1_unstructured` (global or per-layer), `ln_structured` and the random variants, and fuses the masks into the weights at train end when `make_pruning_permanent` is set.

**The ramp.** `scheduler.py` maps epoch to target sparsity:

- **cubic** (default) — Zhu & Gupta's GMP schedule, `S_t = S_f + (S_i − S_f)(1 − progress)³`, also GraNet's Eq. 1 (`dst_schedules.cubic_prune_rate`)
- **linear** — walk the sparsity value
- **constant** — walk the surviving fraction in log space, so each epoch prunes the same share

**Parameter selection.** `parameter_manager.py` allows Conv, Linear, LSTM, GRU and Embedding, blocks every norm layer, and accepts a custom layer that carries a `weight` Parameter. It skips tensors under `min_param_elements`, and biases unless `prune_bias` is set.

**Enforcement.** `on_train_epoch_start` raises when the measured sparsity already exceeds `amount * (1 + tolerance)`. `_verify_sparsity_jump` raises on a re-prune that drops sparsity by more than 5 points from above 0.1.

**Checkpoints.** `checkpoint_handler.py` reconstructs the `_orig`/`_mask` re-parametrization on resume, and fuses the pair back into `weight` when a pruned checkpoint loads into an un-pruned model. Add it with `+callbacks=[checkpoint_handler]`.

```bash
python scripts/make_pruning_permanent.py --input path/to/last.ckpt   # strip masks from a finished ckpt
```

---

## 3. The stem: `prune_first_layer`

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

## 4. Sparsity gating

`src/callbacks/pruning/sparsity_gating.py`, shared by both stacks. Off-target epochs must not be selected on, so each callback reads the sparsity the pruner publishes and gates on it:

| Callback | Out of band |
|---|---|
| `SparsityGatedModelCheckpoint` | skips top-k saving; `last.ckpt` keeps saving, so resume is intact |
| `SparsityGatedEarlyStopping` | skips the check, so patience never accrues on an off-target metric |
| `RampValidationGate` | zeroes `limit_val_batches`, skipping the validation pass |

**The band is relative:** in band iff `(1 - tolerance) * target <= sparsity <= (1 + tolerance) * target`. Each gate takes its own `tolerance`; `1.0` spans every sparsity, which disables that gate.

A gate takes `tolerance: 0.005` (0.5 %, declared once as `_bregman_tolerance` / `_pruning_tolerance`) on every recipe whose sparsity is controlled, so an off-target epoch never becomes the evaluated model. The img recipes run no early stopping, so `SparsityGatedModelCheckpoint` is their only selection gate; the SV recipes gate both. `RampValidationGate` takes `1.0` on the img recipes, so the per-epoch curve is measured over the whole ramp; it takes the same 0.5 % band where a validation pass is too expensive to spend off-target — ImageNet, and every SV recipe. The `*_fixed` recipes open every gate they carry: a static λ reaches whatever sparsity it reaches, so there is no setpoint to band around.

Point every gate at the metric the pruner steers on: `bregman/pruned_sparsity` (Bregman) or `pruning/sparsity` (magnitude). `sparsity` is whole-model.
