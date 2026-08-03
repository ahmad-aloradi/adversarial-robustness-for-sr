# The AdaBreg phantom stripe: cause, proof, fixes

When the classifier head has more rows than the dataset has classes, AdaBreg
keeps one weight in every spare row — all in the same column, the bright
vertical stripe in the mask heatmaps. On cifar10 that is 9,990 weights, **7% of
everything the model keeps** (9.4% on TinyImageNet), paid for by the
convolutions. Cause, in one line: the spare rows push forever, the push lands on
the one scale parameter that is untaxed, shared by all spare rows, and visible
to the head — and AdaBreg's step size ignores how small the push is. The fix
is one config line — raising Adam's `eps` to 1e-4 — verified at scale in §5:
every phantom row dies and accuracy *rises* 0.49. Every number below is
measured from checkpoints or per-epoch logs; §7 reproduces them.

## 1. Setup and end state

- The head maps 512 pooled features to one score per class: a weight row plus a
  bias per class. It is deliberately over-sized — 10,000 rows over cifar10's 10
  classes. A row whose class never appears as a label is a **phantom**.
- Bregman taxes conv and linear *weights* only. Norm scales `γ`, norm shifts `β`
  and all biases carry the identity prox — free — and no weight decay is set.
- A phantom is pushed the same way on every example: softmax gives it
  probability `p > 0` and cross-entropy pushes its score down by `p`, forever.
  A real class stops pushing once it wins; a phantom never stops.

| run (10k head) | phantom rows kept | stripe col | largest `β` | largest `γ` | test acc |
|---|--:|--:|--:|--:|--:|
| cifar10 · AdaBreg · sr99 · BN | 9990 / 9990 | 407 | **111.6** | **38.3** | 94.02 |
| cifar10 · AdaBreg · sr99 · BN · **eps 1e-4** | **0** | none | 1.2 | 3.7 | **94.51** |
| tinyimagenet · AdaBreg · sr99 · BN | 9800 / 9800 | 404 | **165.9** | **87.6** | 61.62 |
| cifar10 · LinBreg · sr99 | 0 | none | 0.04 | 1.17 | 94.10 |
| cifar10 · magnitude · sr99 | 0 | none | 0.44 | 0.84 | 93.65 |
| cifar10 · dense SGD, 10k head | nothing pruned | none | 0.13 | 1.35 | 96.15 |

The end state is identical in every stripe run: each phantom row keeps exactly
one weight, all in the same column, all negative, zero anywhere else, and the
stripe column has the largest `β`, the largest `γ` (one exception: TinyImageNet
AdaBregW ranks 29th in `γ`) and the largest pooled feature of all 512. sr95
stripes identically, so this is not an artifact of the extreme target.

The sr99 BN checkpoint shows what the column *is*: its conv input is nearly
silent (running mean −0.29, running variance 0.024), so the norm output is
essentially the constant `β = 111.6`, and the pooled feature is 92.4 (channel
median 3.0), on for 100% of inputs. A negative weight on a large constant is a
per-row constant subtraction — the stripe is a hand-built bias.

## 2. The mechanism, link by link

**(1) The push is real at first, and never turns off.** At init the phantoms
hold ~99.9% of the softmax mass, so suppressing them is genuinely what the loss
demands early on. But `p` only decays asymptotically; the push never reaches
zero.

**(2) Only three places can hold the suppression, and one is privileged.**

- the head weight — taxed by the prox;
- the row's own bias — free, but per-row: its push is a single row's `p`;
- the norm affine of the feature's channel — free **and shared**: its push is
  the sum over all 9,990 rows. Measured lever: `Σ|w|` down the stripe column is
  **1,207**, so the channel affine feels ~1,200× the push of any one bias.

That pressure is purely phantom pressure: no real row uses column 407 (real
classes sit in 43 other columns), and pooling keeps channels separate, so the
loss reaches `β`/`γ` of channel 407 *only* through phantom scores. Nothing
pushes back.

**(3) AdaBreg's step ignores the size of the push.** The update divides the
push by its own recent size (Adam), so any sign-consistent push — however
small — moves its parameter at roughly the learning rate every step, until the
push drops below `eps = 1e-8`. A never-off push becomes constant-rate drift.
LinBreg's step is proportional to the push, so the same never-off push moves
nothing:

| per-epoch log, sr99, 10k head | after ep 1 | ep 30 | ep 80 | ep 320 |
|---|--:|--:|--:|--:|
| AdaBreg · largest `γ` (init 1.0) | 2.2 | 24.8 | 34.0 | **38.3** |
| AdaBreg · norm-params L2 (init 69.3) | 94.2 | 148.4 | 191.9 | **239.8** |
| AdaBreg · head sparsity | 0.957 | **0.998** | 0.998 | 0.998 |
| LinBreg · largest `γ` | — | 1.27 | 1.44 | **1.45** |
| LinBreg · norm-params L2 | — | 69.3 | 69.3 | **69.5** |

One epoch of AdaBreg moves the free parameters further than LinBreg's entire
run. The end-of-run brake is `eps`, not an equilibrium: `β` still creeps
111.52 → 111.64 over epochs 289–327, consistent with the per-row `p` (~5e-11)
sitting far below `eps` while the 1,207× lever keeps the channel's summed push
barely alive.

**(4) The head tax herds everything into one column.** By epoch 30 the head is
at 99.8% sparsity — about one weight per row. All phantom rows are pushed in
the same direction (`p` times the feature vector, features ≥ 0 after ReLU), so
they rank the 512 columns identically and their single survivors land together.
Then the loop closes: the shared column concentrates the summed push onto one
channel's affine, the channel inflates, suppression per unit of weight gets
cheaper there, and the ranking is locked. About two-thirds of the total `γ`
growth happens during the head crush (epochs 1–30) — stripe formation and norm
blow-up are one process, not cause and effect. The "large norm → large pooled
feature → stripe" arrow is the correct *inference-time* reading of a training
loop with no first mover except the tax on the head.

**(5) Why the last norm layer.** Every earlier `γ`/`β` is re-normalized by the
next BatchNorm downstream, so the head never sees its scale; only
`layer4.1.bn2` faces the head directly. Measured across all 20 norm layers of
the trained net: `β` ≤ 5.2 and `|γ|` ≤ 8.6 everywhere except the head-facing
layer at 111.6 / 38.3.

Both ingredients are necessary:

| largest `γ` at end | 10-class head | 10,000-row head |
|---|--:|--:|
| AdaBreg (size-blind step) | 4.13 | **38.33** |
| LinBreg (step ∝ push) | — | 1.45 |
| dense SGD | — | 1.35 |

Phantoms without the size-blind step: nothing moves. The size-blind step
without phantoms: mild drift. Together: the stripe.

## 3. The stripe was never necessary

The head bias is free and never pruned. Zero the stripe column, then give every
phantom row one bias shift fitted to minimize cross-entropy (2,560 test
images):

| cifar10 · AdaBreg · BN | cross-entropy | accuracy |
|---|--:|--:|
| as trained | 0.2233 | 0.9398 |
| stripe zeroed | 0.3613 | 0.9398 |
| stripe zeroed + bias shift −15.85 | **0.2233** | 0.9398 |

The loss returns to baseline to seven decimals (all-LN run: same, at shift
−19.84). So the stripe lost a race, not an argument: bias and stripe weight
drift at the same parameter-space rate under AdaBreg, but one unit of stripe
weight buys 92–285 score units (the feature's size) while one unit of bias buys
1. Achieved by the end: −11.2 of suppression from the stripe, −0.59 from the
bias. Zeroing the stripe with no compensation at all changes nothing at
inference — accuracy identical, phantoms win 0 / 2,560 on cifar10, all-LN, and
TinyImageNet — because real classes outscore phantoms by ~30 either way.

## 4. What it costs

| cifar10, sr99 | conv kept | head kept | of which phantom | total kept |
|---|--:|--:|--:|--:|
| AdaBreg | 133,464 | 10,092 | **9,990** | 143,556 |
| LinBreg | 142,009 | 1,562 | 0 | 143,571 |

The stripe's 9,990 weights come out of the convolutions' budget. Pricing the
refund from the local trend (5% kept scores 94.93, 1% kept scores 94.02): a 7%
budget refund is worth roughly **+0.04 accuracy** on cifar10. Real waste, small
payoff at this operating point; TinyImageNet carries 9.4% unmeasured. AdaBreg
still beats LinBreg by 11 points on TinyImageNet while carrying the stripe — a
reclaimable inefficiency, not a broken method.

A second cost is controller stability: thousands of near-identical weights sit
barely above one global `λ` and switch on and off as a block, so sparsity
overshoots and rebounds.

## 5. The fix, verified at scale

Raise Adam's `eps` from 1e-8 to 1e-4: `++module.optimizer.eps=1e-4`, no code
change (`RUN_STRIPE_FIX_EXPS` in `scripts/fabfile.py` holds the recipe). `eps`
is the push size below which the Adam step stops being amplified and becomes
proportional to the push again. The phantom pushes start large — above the
floor — and get handled honestly: the free bias walks down and the real
classes take the softmax. Within a few epochs each phantom's push collapses to
~1e-6 and under, below the floor, and stops counting: its dual freezes while
`λ` keeps rising, so every phantom row dies. The norm channel is protected by
the same floor — its only pressure was the summed phantom push. Real gradients
sit above 1e-4 for most of training, so real learning keeps identical steps:
the floor is surgical.

Real runs, sr99 · 10k head · BN · seed 42:

| run | phantom rows dead | stripe | largest `β` / `γ` | real weights/row | conv kept | test acc |
|---|--:|--|--:|--:|--:|--:|
| AdaBreg baseline | 0 / 9990 | col 407 | 111.6 / 38.3 | 10.2 | 133,464 | 94.02 |
| + `eps` 1e-4 | **9990 / 9990** | none | 1.2 / 3.7 | 63.9 | **142,930** | **94.51** |
| + per-tensor denominator | 9990 / 9990 | none | −0.2 / 4.1 | 91.4 | — | 92.53 |
| + relative eps floor | 0 / 9990 | col 158 | 7.8 / 7.3 | 63.3 | — | 94.02 |

The refund is visible in the winner: the 9,990 junk weights went back to the
convolutions (+9,466) and the real head rows (10 → 64 each), worth **+0.49**
accuracy — more than the +0.04 budget-trend estimate, plausibly because the
whole feature scale is sane again (single seed; TinyImageNet, which carries
9.4%, is not yet re-run).

Two alternatives were piloted as optimizer knobs and removed again after these
runs:

- **Per-tensor denominator** (one scalar Adam denominator per tensor,
  NovoGrad-shaped): kills the stripe the same way, but strips per-coordinate
  adaptivity from every real weight too — 1.5 accuracy points.
- **Relative eps floor** (`denom += mean(√v)` per tensor): fails at scale. In
  a head that is 99.9% phantom coordinates the floor is computed from the very
  pushes it should suppress and shrinks with them, while the stripe coordinate
  sits *above* the tensor mean and keeps full-rate steps. Outcome: norms tamed
  (`β` 7.8), stripe fully intact.

Still useful regardless of optimizer:

- **Right-size the head.** No phantoms, no stripe (10-class control: `γ`
  settles at 4.13). The 10k head is a stress stand-in for SV-scale class
  counts, so this dodges the benchmark rather than fixing the method.
- **Delete the stripe column after training.** Free and proven — accuracy
  unchanged to four decimals, no real class uses the column — but the conv
  budget it displaced during training is not refunded.
- Taming only the free parameters (plain-SGD group, or decay on norm affine
  and bias) stops the blow-up but not the waste — the shared column survives.

Ruled out by measurement:

- **Sign consistency as the trigger** (`AdaBregD`, removed): 277 of 512
  channels are more sign-consistent than the stripe column; its *size* singles
  it out.
- **The bias as the off-switch**: LinBreg kills every phantom while barely
  moving the bias (L2 2.58 → 2.67); rows die because the accumulated push
  stays under `λ`, not because the bias suppresses them.
- **BatchNorm statistics**: all-LayerNorm reproduces the stripe exactly; only
  the `β`/`γ` split moves.
- **Extreme-target artifact**: sr95 stripes identically (column 242, `β` 84.1).
- **Dual-space decay** (`AdaBregW`, leaky dual): TinyImageNet AdaBregW still
  stripes (column 73) — decay shrinks magnitude but leaves the size-blind
  accumulation untouched.

Set aside by argument, not measurement:

- **AMSGrad / beta retuning** — still a saturating per-coordinate ratio;
  size-blindness is untouched.
- **Trust-ratio scaling (LARS/LAMB)** — rescales the update per layer but keeps
  Adam's elementwise ratio inside it, so it stays size-blind within a tensor.
- **Sign-momentum steps (Lion)** — a pure sign update is maximally size-blind
  and would sharpen the pathology.

## 6. The bias-less SV head

Carried over from the SV runs, not re-measured here. The cosine SV head has no
bias, and fails the other way: an all-zero row still scores cosine 0, which
beats a trained wrong class, so phantoms keep collecting probability
regardless. Measured: 0 dead phantom rows under either Bregman variant; under
LinBreg all 1,211 phantoms keep one weight each in column 7. These runs change
two things at once (cosine vs linear, no bias vs bias), so they cannot isolate
the bias. LinBreg's other stripes there are real feature selection, not this
defect: 106 of 192 columns on VoxCeleb ECAPA, each shared by hundreds of real
speakers.

## 7. Reproduce

The striped baseline, the `eps` fix and AdaBregW all survive on TinyImageNet;
the cifar10 runs behind the tables above are archived under `old_exps/`, and
its AdaBreg baseline is no longer on disk.

```bash
# per-run anatomy JSONs, heatmaps, feature plots — baseline vs eps fix vs AdaBregW
python scripts/visualize_structured_vs_unstructured.py \
    --base_dirs /data/aloradad/results/tinyimagenet/resnet18/old_exps \
                /data/aloradad/results/tinyimagenet/resnet18 \
    --experiments 'bregman_adabreg*sr99-classifier_10k*' 'pruning_mag_unstruct*sr99-classifier_10k' \
    --head_anatomy --activations --data_dir data \
    --output results/img/tinyimagenet/resnet18/head_anatomy

# one run on its own
python src/vis/head_anatomy.py <run_dir>/seed_42
```

- `head_anatomy.json` — every checkpoint-level scalar quoted above.
- `weight_norms.csv` in each run dir — the per-epoch trajectories in §2.
- Reading the mask heatmap: bright = survived; rows are classes, columns are
  input features. A bright vertical line is one feature kept across many
  classes.
