# Why the classifier bias matters for Bregman pruning

Short version: an over-sized classifier head has rows for classes that never
occur. Without a bias, Bregman keeps those useless rows alive and prunes useful
weights instead. A per-class bias gives every class a weight-free "off switch",
which lets the useless rows die. This is now on for the SV head
(`CosineClassifier`, `bias: true`).

## The problem: phantom rows

- Heads are usually bigger than the number of real classes:
  - VoxCeleb: 7205 rows, 5994 speakers → 1211 rows that never occur.
  - The 10k-head study: 10000 rows, 200 real classes → 9800 that never occur.
- A class that never appears as a label is a **phantom**. Its row of weights
  should end up all-zero. Magnitude pruning gets this right. Bregman does not —
  it tends to keep phantom rows and prune real ones.

## Why Bregman keeps the wrong rows (no bias)

- Softmax never gives exactly zero probability. A class that is never the
  answer still gets a tiny "push your score down" gradient on **every batch, in
  the same direction, forever.**
- A real class cancels out: pushed up on its own examples, down on everyone
  else's. A phantom's push never cancels.
- Bregman decides which weights live by a **running total** of the gradients a
  weight has received. The phantom's one-directional push piles up without
  limit and crosses the survival threshold — so Bregman ends up preferring the
  most useless rows in the network.
- On a cosine head (the SV case) it is worse: with no bias, a fully-pruned row
  sits at cosine 0, which **scores higher** than a trained wrong class (whose
  cosine is negative). The phantom rows soak up the wrong-class probability, so
  their push never quiets down.

## Why the bias fixes it

- The bias is one number **per class** — shape `(out_neurons,)`, added to each
  class's own score and broadcast over the batch (row-wise on the head). It
  carries **no weights** and is left unregularized (never pruned).
- The per-class shape is what makes the off switch work: each phantom class has
  its **own** offset that cross-entropy can drive down independently. A
  per-*feature* bias of shape `(D,)` would instead be shared by every class at
  once and could not single out a phantom — that version would not help.
- Cross-entropy drives a phantom class's bias very negative → its probability
  drops to ~0 → its push **quiets down** → the row stops receiving gradient.
  (The score saturates at the clamp floor of −1, i.e. logit −1 × scale ≈ 0
  probability, so the push is fully quenched.)
- With the push gone, Bregman's L1 shrinks the idle row to zero and it **stays
  dead**. The bias absorbs the demand at the source, so no weight is needed to
  hold the phantom's score down.
- This is exactly why plain LinBreg already works on a linear head that has a
  bias, and fails on the bias-free cosine head. Adding the bias closes that gap.

What we changed: the SV classifier is now `CosineClassifier` with `bias: true`.
The bias falls through to the unregularized group (only the weights are
pruned), and it is clamped so the output stays a valid cosine for scoring.

Caveat — AdaBreg is different: the bias quiets the push, but AdaBreg builds its
running total with Adam, which reacts to the gradient's **direction, not its
size**. A vanishing-but-same-sign push still moves it at full speed, so AdaBreg
can keep phantom rows alive even with the bias. The bias is a fix for
LinBreg-style optimizers; AdaBreg needs a separate one. LinBreg is the safer
choice for these heads.

## Reading the AdaBreg mask heatmap

The heatmap shows surviving (non-zero) weights per layer. **Row axis = output
units** (filters, or classes for the head); **column axis = input units**;
bright = weights survived.

### The stripes you see (conv1)

- The first panel, `net.conv1`, shows a few bright **horizontal bands**. Those
  are surviving output **filters**.
- `conv1` sees only 3 inputs (RGB). At 99% sparsity ~80% of its 64 filters are
  fully dead; ~13 survive. A surviving filter keeps weights across all 3 inputs,
  so it draws a full-width line — a stripe.
- Whole filters survive (a structured look) even though the method is
  unstructured because a conv filter is useless unless enough of its weights
  survive **together**. The few survivors concentrate into a handful of filters
  instead of scattering. `conv1` shows this most because it is tiny.
- **This is not an AdaBreg fault.** Every method keeps some input filters:
  LinBreg keeps ~16, magnitude keeps ~34, AdaBreg ~13. AdaBreg's stripes are
  simply the fewest and faintest.

### The real AdaBreg problem is not a stripe — it is the head

The failure hides in the last panel, `net.fc` (10000 classes):

| run | phantom rows fully dead |
|---|---|
| LinBreg | 98.0% (all 9800 phantoms → clean) |
| Magnitude | 98.0% |
| **AdaBreg** | **0.0%** |

- LinBreg and magnitude zero out every phantom row and keep only the ~200 real
  classes — a clean result (a sparse set of bright real-class rows over black).
- **AdaBreg zeros no rows.** It smears its surviving 1% of weights across **all
  10000 classes**, keeping every phantom row a little alive. So its head panel
  is a faint, even haze with no clean split between real and phantom — the
  opposite of a tidy stripe.
- Cause: the magnitude-blind Adam total (see the caveat above). Across the whole
  network AdaBreg is the **least structured** of the three methods — its few
  survivors are spread thinnest.

Takeaway: the bright `conv1` stripe is healthy, expected structure that every
method keeps. The AdaBreg problem is the phantom rows it **cannot** clear, which
appear as a full-height haze in the head panel rather than a stripe. The bias
re-applied here is the off switch that lets those rows die under LinBreg.
