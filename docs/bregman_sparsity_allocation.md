# Why Bregman misallocates sparsity — and the fixes

Findings from the phantom-classifier study (2026-07), the fixes now in the
code, and how to run the benchmark. All fixes are **off by default**; defaults
reproduce the old behavior exactly.

## The benchmark: classes that cannot exist

The classifier head is deliberately larger than the number of classes that
occur in training:

- VoxCeleb: head has 7205 rows, only 5994 speakers exist → 1211 phantom rows.
- TinyImageNet study: head inflated to 10000 rows, 200 real classes → 9800
  phantom rows.

A phantom class never appears as a label, so its row of weights should end up
all-zero. This gives a rare thing: a sparsity task with a known ground truth.
Magnitude pruning gets it right (exactly 9800/10000 rows dead). Bregman did
not — and each failure below is one reason why.

## Problem 1 — Bregman never forgets

- Bregman decides which weights live by the **dual**: a running total of every
  gradient the weight ever received. A weight revives once that total passes
  the threshold λ.
- Example: a dripping tap fills any bucket, no matter how slow the drip —
  unless the bucket has a hole. Magnitude pruning's bucket has a hole: weight
  decay constantly shrinks weights, so only *current* demand keeps a weight
  alive. Bregman's bucket has no hole; a tiny push repeated forever always
  wins.
- Measured signature: phantom junk in the head stayed at the same density
  while λ was raised 16× — the junk is old accumulated total, out of the
  threshold's reach.

## Problem 2 — phantom classes are pushed in one direction only

- Softmax never assigns exactly zero probability. A class that is never the
  answer gets a small "push your score down" gradient on **every batch**, in
  the same direction, forever.
- A real speaker's weights get pushed *up* on their own utterances and *down*
  on everyone else's — the pushes cancel once the model fits, and the total
  stops growing. Phantom pushes never cancel.
- Combined with Problem 1: real weights' totals stall below λ; phantom totals
  grow without bound and must cross it. Bregman ends up preferring the most
  useless rows in the network.

## Problem 3 — no other weight can take over a phantom's job

- Pruning works on the encoder because it is redundant: remove a weight, the
  surviving weights absorb its function, and the gradient at the removed
  weight fades to zero.
- Class j's score is controlled by row j alone. No other weight can lower it.
  So the "please push this score down" demand can never be absorbed — unless
  the head has a per-class offset (see Problem 4).

## Problem 4 — the SV cosine head has no off switch (and a gradient bomb)

- The AAM head scores by cosine and has **no bias**. A fully-pruned row is
  frozen at cosine 0 — which is a *better* score than trained wrong classes
  (cosine −0.2 × scale 32 = logit −6.4; e⁰ vs e⁻⁶·⁴ ≈ 600×). The phantom
  block soaks up most of the wrong-class probability and its push never
  quenches. The image head has a bias, which is exactly why plain LinBreg
  works there: cross-entropy drives phantom biases down and the push dies.
- Cosine also ignores weight size: rows are rescaled to unit length, and the
  gradient grows like 1/‖row‖. Measured on the real head: healthy row 0.37,
  1%-density row 6.3, exactly-zero row 8.8e10 (the `eps=1e-12` guard). There
  is no "no floor" option — 1e-12 *is* a floor, just one that manufactures
  huge gradients and makes row death impossible.

## Problem 5 — AdaBreg's momenta are magnitude-blind

- Adam divides each gradient by its own typical size. Speed then depends only
  on **sign-consistency**: a gradient of 0.000001 with the same sign every
  step moves the dual at full learning-rate speed; a large but sign-flipping
  gradient barely moves it.
- So the bias off-switch that saves LinBreg on the image head does not save
  AdaBreg: the push shrinks in size but keeps its sign, and Adam ignores size.
- Evidence: image 10k head, LinBreg kills all 9800 phantom rows; AdaBreg
  revives every one of them.

## Why sparsity oscillates late in training

- The λ controller raises λ to cut weights. Cutting a phantom row *restores*
  its push (its score snaps back up, probability returns). More cutting →
  more comeback pressure → no λ has a resting point → sparsity and accuracy
  cycle. The controller is fine; this block of weights is the only place in
  the network where cutting increases the pressure to return.

## The fixes

1. **`dual_leak` on LinBreg** (`module.optimizer.dual_leak`, default 0.0).
   One line: `v ← (1−γ)·v − lr·g`. The total now fades with horizon ~1/γ
   steps, so survival needs *sustained current* demand above `γλ/lr` — the
   same statistic weight decay gives magnitude pruning. Example: γ=0.001 ≈
   1000-step memory; the phantom drip falls below the bar and the rows die;
   a genuinely useful weight has large current demand and revives quickly.
2. **Hybrid birth rule on AdaBreg** (same `dual_leak` knob). Pruned weights
   stop taking magnitude-blind Adam steps; they are re-born only when a
   γ-horizon average of the **raw** gradient exceeds the bar. Death stays
   Bregman, birth becomes gradient-evidence (gradual-pruning style).
3. **Optional classifier bias** (`module.model.classifier.bias=true`,
   default false). Gives every class a weight-free way to say "I never
   occur" — the off switch the image head always had. Unregularized (falls
   into the no-prune bias group), clamped so the AAM margin math stays valid.
4. **Optional norm gate** (`module.model.classifier.norm_gate=0.3`, default
   0.0 = exact cosine). Scores by `cos · ‖W‖/(‖W‖+ρ)`: a two-weight row is
   no longer inflated to a full-confidence direction, gradients stay bounded,
   and a row can actually reach zero. Needed alongside the leak on cosine
   heads — the leak alone shrinks rows into the 1/‖row‖ blow-up and they
   storm back.

## Proof on the toy benchmark (20 real + 180 phantom classes)

| Setup | phantom rows dead | accuracy |
|---|---|---|
| cosine head, vanilla LinBreg (≈ current SV) | 0% | 0.84 |
| linear head + bias, vanilla AdaBreg (≈ current img) | 0% | 0.78 |
| gate + bias + LinBreg leak | **100%** | 0.84 at 24% fewer weights |
| bias + AdaBreg hybrid | **100%** | 0.71 (bar untuned) |
| magnitude + weight decay (reference) | 87% | 0.81 |

The leak result holds over a 20× range of γ. Vanilla's accuracy edge is
carried by junk weight it should not have; at matched weight budget the leak
matches or beats it.

## Running the real benchmark

`fab run_img` with `RUN_LEAKY_BREGMAN_10K = True` (scripts/fabfile.py)
launches TinyImageNet + ResNet-18 + 10k head at sr99:

- `bregman_linbreg … -classifier_10k-leaky` — dual leak γ=1e-3
- `bregman_adabreg … -classifier_10k-hybrid` — birth rule γ=1e-3

Success criteria against the existing vanilla `-classifier_10k` runs:

- `net.fc.weight` `fully_zero_row_frac` = 0.98 in the mask summary
  (vanilla: LinBreg 0.98, AdaBreg 0.00).
- no late-training sparsity oscillation on the AdaBreg run.
- validation accuracy at the 99% target no worse than vanilla.
