# Sparse-training baselines

Six published methods that reach a sparsity target without the Bregman
machinery. Five share one callback (`src/callbacks/pruning/dst_pruner.py`); STR
has its own (`src/callbacks/pruning/str_pruner.py`).

```bash
python src/train.py experiment=img/pruning_rigl datamodule=datasets/cifar100
python src/train.py experiment=img/pruning_snip_iter callbacks.model_pruning.amount=0.95
python src/train.py experiment=img/soft_threshold module.optimizer.weight_decay=9.051757813e-5
```

## What the methods are

| Experiment | Paper | Mask at init | Mask updates | Sparsity |
|---|---|---|---|---|
| `pruning_rigl` | Evci et al., ICML 2020 | ERK layer budget, random | every 100 steps: drop smallest, regrow largest gradient | fixed |
| `pruning_set` | Mocanu et al., Nat. Comm. 2018 | ERK layer budget, random | same, but regrow at random | fixed |
| `pruning_static` | RigL's Static-ERK control | ERK layer budget, random | none | fixed |
| `pruning_snip` | Lee et al., ICLR 2019 | one global \|w·grad\| ranking | none | fixed |
| `pruning_snip_iter` | de Jorge et al., ICLR 2021 | the same ranking over 100 steps | none | fixed |
| `pruning_granet` | Liu et al., NeurIPS 2021 | dense | every 1000 steps: cubic prune + RigL regrowth | 0 → target |
| `soft_threshold` | Kusupati et al., ICML 2020 | dense, learned threshold per layer | continuous | **outcome, not a target** |

`pruning_rigl.yaml` is the parent; the others are thin children that change one
or two knobs. The schedules live in `dst_schedules.py` and the layer budget in
`utils/erk_sparsity.py`, both runnable on their own:

```bash
python -m src.callbacks.pruning.dst_pruner
python src/callbacks/pruning/utils/erk_sparsity.py
```

## What "sparsity" means here

`amount` and the reported `pruning/sparsity` both mean **zeros over every weight
tensor — all but norms and biases**. Layers held dense count in that denominator
at full size with no zeros, so the number is comparable across every method in
the benchmark, including Bregman and magnitude pruning.

`prune_first_layer: false` (the default) holds the stem conv dense, which is
RigL's rule on CIFAR and at 99 % everywhere. It costs 1,728 of the 111,644
weights a 99 % budget allows on ResNet-18 — 1.5 %, so it is always affordable.
The classifier **is** pruned: holding it dense too would eat 46 % of the budget
on CIFAR-100 and more than all of it on ImageNet.

## SNIP collapses at 99 %, and that is the finding

One-shot SNIP ranks `|w·grad|` globally, across layers. On a BatchNorm network
each layer's *summed* saliency is nearly constant (measured on ResNet-18 at
init: 5.6–11.6 across every conv), because BatchNorm makes the loss invariant to
a layer's weight scale. So the *per-weight* score falls like 1 / layer size, and
a global ranking spends the budget on the small early layers:

| Layer | Weights | Kept at a 99 % budget |
|---|---|---|
| `layer1.0.conv1` | 36,864 | 46.2 % |
| `layer2.0.conv2` | 147,456 | 7.7 % |
| `layer3.1.conv2` | 589,824 | 0.018 % |
| `layer4.1.conv2` | 2,359,296 | 0 % |

`layer4` holds 75 % of the network and gets nothing, so the run trains a
collapsed model. That is the documented layer collapse of one-shot pruning at
init (Tanaka et al. 2020; de Jorge et al. 2021), not a bug, so it is logged and
swept like any other rate: CIFAR-100 / ResNet-18 gives 39.6 % test top-1 at 99 %
against 77.7 % at 90 %.

`snip_iterations` is the fix and works at any sparsity: walk the density down
`(1 - amount) ** (t / T)` and rescore the **masked** network at each step. An
over-pruned layer's gradients grow and its saliency recovers before it collapses.
Measured on ResNet-18 at 99 %: `T=1` collapses 5 layers, `T=10` and `T=100`
collapse none. `T=1` reproduces the paper's single ranking exactly, so the two
rows come from one code path.

## STR has no sparsity target

STR replaces each weight with `sign(w)·relu(|w| − sigmoid(s))` for a learnable
per-layer scalar `s`. Weight decay pulls `s` toward 0 (threshold up), the task
loss pushes back, and the balance sets the sparsity. There is nothing to set it
to — one `λ` decays `w` and `s` alike, so sweep `module.optimizer.weight_decay`
over Table 10's grid and report where each run lands.

### Why the paper's `s_init: -3200` does not transfer

Below −89 `sigmoid(s)` underflows to exactly 0, so the loss gradient on `s` is
*zero* and only decay moves it, multiplicatively:
`s(t) = s_init · exp(−Σlr · wd / (1 − momentum))`. Call that exponent the decay
budget. The paper's ResNet-50/ImageNet config (`lr 0.256`, 100 epochs, batch 256,
momentum 0.875, `wd 2.2518e-5`, no separate `--st-decay`) has a budget of 11.5,
and `ln(3200/89) = 3.6` of it is spent underflowed — the first third of the run
is dense by design. This recipe (`lr 0.1`, 200 epochs, batch 128, momentum 0.9,
45k train images) integrates to `Σlr = 3,538`, so the budget is `35,376 · wd`:
**at the paper's `wd` it is 0.80, never reaches 3.6, and the run finishes 100 %
dense.** Raising `wd` to 6e-4 to compensate overshoots the other way — budget 21,
enough to drive every threshold to `sigmoid(0) = 0.5`, above every weight in the
network. That run reached 94 % sparsity at epoch 42 and then thresholded
`layer3.0.downsample.0` away entirely.

So `s_init: -8.0` here (threshold 3.4e-4, ~1 % of weights below it at init): the
delay is gone and `wd` only has to hold the balance, not also travel 3,200 units.

### What is left, and it is the method's

- **The balance is not stationary.** `s` settles where `wd·|s|` matches the loss's
  pull. That pull is proportional to the surviving weight count and to the
  gradient scale, both of which shrink as the run converges, so the sparsity
  creeps up and the least-defended layer (a 1x1 shortcut) collapses first.
- **A collapsed layer never comes back.** The relu zeroes the gradient to both `w`
  and `s`; decay then raises the threshold and shrinks `w`. Both deepen it. Logged
  as `str/collapsed_layers`. Unlike Bregman, where the dual keeps integrating and
  a zeroed weight can return, this is an absorbing state.
- **The knob is exponentially sensitive.** The threshold is `exp(s)` and `wd` acts
  on `log|s|`, so control runs through two nested exponentials, and nothing feeds
  the achieved sparsity back. That is why STR is a sweep, not a target.

At `on_train_end` every ckpt except `last.ckpt` gets `sigmoid(s)` written into
`weight` and the `s` key dropped (`--dense-conv-model` in the original repo), so `eval.py`
and the robustness scripts load the original ResNet. `last.ckpt` keeps its `s` so
an interrupted run can resume.

## Deviations from the papers

- **Gradient clipping is off** for every sparse recipe, and now for `dense_sgd`
  too. A clip scales the loss gradient but never the weight decay or the
  regularizer, which shrinks the live weights' steps ~6× at 99 % sparsity and
  biases every STR threshold upward. No original implementation clips.
- **The stem is dense, the head is not** (above). RigL keeps the first layer
  dense; it prunes the classifier on ImageNet, and so do we everywhere.
- **`pruning_set`** is SET's growth criterion under RigL's schedule — the
  controlled ablation of "does gradient-guided regrowth matter", not SET's
  published recipe (α 0.5, per-epoch updates, random init).

## Checking a run

```bash
pytest tests/test_dst_pruner.py tests/test_str_pruner.py tests/test_dst_schedules.py \
       tests/test_erk_sparsity.py tests/test_first_layer_dense.py
```

At every epoch end the DST callback asserts the achieved sparsity is inside
`tolerance` (a fraction of the target, the same band the checkpoint, early-stop
and validation gates use). A drift means a mask update lost or gained weights.
