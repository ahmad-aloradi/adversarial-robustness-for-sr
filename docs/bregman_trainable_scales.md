# AdaBreg trainable per-layer scales

`sv/sv_bregman_adabreg_trainable_scales` starts dense, drives to a fixed target,
and **lets each layer learn its share of the sparsity** via a scalar per-layer
scale.

It inherits `sv/sv_bregman_adabreg` (encoder: WeSpeaker ECAPA-TDNN on CN-Celeb;
loss: AAM-softmax). Only `audio_encoder.encoder` and `classifier` weights are
regularized; BatchNorm/bias stay dense FP32.

```bash
python src/train.py +experiment=sv/sv_bregman_adabreg_trainable_scales
```

---

## 0. Foundation: AdaBreg + L1 prox

Bregman learning minimizes `L(θ) + J(θ)` with `J(θ) = λ‖θ‖₁`, but instead of
adding the penalty to the loss (which only shrinks weights) it keeps a **dual
variable** `v` (accumulated subgradient) and reads the weights off it through a
soft-threshold. Sub-threshold signal stays *exactly* zero → real sparsity.

The dual represents the subgradient of the elastic-net functional
`J_δ(θ) = λ‖θ‖₁ + (1/2δ)‖θ‖₂²`. Initialization:

```
v⁰ = (1/δ)·θ⁰ + λ·sign(θ⁰)            # = ∂J_δ(θ⁰)
```

Per step, AdaBreg does an Adam-preconditioned **dual** update, then a proximal
**primal** readout:

```
g       = ∇_θ L(θ^k)
m^k     = β₁·m^{k-1} + (1-β₁)·g                              # 1st moment
u^k     = β₂·u^{k-1} + (1-β₂)·g²                             # 2nd moment
v^{k+1} = v^k − (lr/(1-β₁^k)) · m^k / (√(u^k/(1-β₂^k)) + ε)  # dual update
θ̃^{k+1} = prox_{δλ‖·‖₁}(δ·v^{k+1})                          # primal readout
```

With `RegL1`, the prox is soft-thresholding:

```
prox(x, δ) = sign(x) · max(|x| − δλ, 0)
```

`δ = 1` here. `LinBreg` is identical with the Adam block replaced by
`v^{k+1} = v^k − lr·g`. The weight is always a *thresholded view* of the dual;
trainable scales change the per-layer `λ`.

---

## Trainable per-layer scales

`configs/experiment/sv/sv_bregman_adabreg_trainable_scales.yaml`

### Decouple level from allocation

One global `LambdaScheduler` owns the sparsity **level** `λ_global`; each prunable
layer `g` owns its **share** through a scalar **linear** scale factor `c_g`. The
L1 strength the prox actually applies to layer `g` is:

```
reg.lamda_g = λ_global · c_g          # per-layer threshold t_g = δ · λ_global · c_g
```

`λ_global` is common to all layers (the scheduler's job); `c_g` redistributes
pressure between them. Scales start neutral (`c_g = 1`).

### The scales are ordinary parameters of the same optimizer

`create_scale_params` registers one scalar `nn.Parameter` per matching layer in
`pl_module.bregman_scales` (an `nn.ParameterDict`), created in
`LightningModule.setup` so the keys exist for a strict resume load. They train in
their own `RegNone` group (prox = identity) at lr `scale_lr`, alongside the model
weights — no separate optimizer, EMA, or controller.

### The missing chain rule → closed-form hypergradient

`λ` acts *inside* the `@torch.no_grad()` prox, so autograd never sees `c_g`. The
pruner injects the gradient in closed form in `on_before_optimizer_step`. Because
the factor is linear, `∂t_g/∂c_g = δ·λ_global` (constant — no `c_g` factor), so
for a live weight `∂θ_i/∂c_g = −δ·λ_global·sign(θ_i)`, hence:

```
∂L/∂c_g = −δ · λ_global · Σ_{i live} grad_i·sign(θ_i)   +   scale_decay · (c_g − 1)
```

- `sign(0)=0` drops dead weights with no extra masking.
- `λ_global` is the scheduler's level (not `reg.lamda_g = λ_global·c_g`, which is
  undefined per-`c_g` once a protected layer floors at `c_g = 0`).
- `scale_decay · (c_g − 1)` is a soft L2 prior pulling `c_g → 1` (neutral), which
  gives a finite equilibrium (below) so the only bound needed is the floor.

After the optimizer steps `c_g`, `on_train_batch_end` floors it at `scale_min`
(domain bound `λ_eff ≥ 0`; no upper bound) and re-syncs `c_g` into `lambda_scale`.

### What the hypergradient does

`Σ grad_i·sign(θ_i)` asks whether the loss wants this layer's live weights
*larger* (away from zero) or *smaller*:

- Over-parameterized layer (loss indifferent / wants shrinkage) → hypergradient
  pushes `c_g` **up** → higher threshold → the layer is free to die (SE/pool
  layers).
- Critical layer fighting the threshold → `c_g` pushed **down** → less pressure →
  the layer stays dense (the classifier); at the extreme it floors at `c_g = 0`
  (no pruning pressure at all).

The level stays fixed by the scheduler; only the cross-layer *distribution* is
learned.

### Mode and target

The controller drives `λ_global` toward a fixed `_bregman_target_sparsity` from
step 0, so the scales allocate over the whole run. `_bregman_target_sparsity:
0.99` is an **overall-model** target (global single-scheduler mode); the
controller and validation gate measure whole-model sparsity. The dense BN/bias
floor caps the reachable maximum near ~0.996, so 0.99 is attainable.

A `LinBreg` counterpart exists at
`sv/sv_bregman_linbreg_trainable_scales` (same allocation machinery, plain
linearized-Bregman dual update).

### Why a linear factor

The factor is linear, so its hypergradient is independent of `c_g` and the update
is additive. The equilibrium

```
c_g* = 1 + (δ·λ_global / scale_decay) · signal_g
```

is finite and **proportional to the signal** — the allocation auto-adapts to the
signal's range, with `scale_decay` the single knob for how wide it spreads. The
only bound is the domain floor `c_g ≥ 0` (`λ_eff ≥ 0`), where a fully-protected
layer settles.

### Visualizing the allocation

Trainable-scales runs write `trainable_scales_history.csv` (one `c_g` per layer
per epoch) to the run dir and log each layer to the tracker (not just
`bregman/scale_min` / `scale_max`). Plot the per-layer evolution with:

```bash
python scripts/visualize_scale_evolution.py \
    --run_dirs '/path/to/sv_bregman_adabreg_trainable_scales*' \
    --output results/scale_evolution
```

`src/vis/scale_evolution.py:render_scale_evolution` draws one line per layer
(colored by final scale), mirroring the layerwise-sparsity renders.

The encoder front-end (Fbank, per-utt norm) and BatchNorm/bias stay dense, and
sparsity/EER are reported through the `BregmanPruner` logging.
