# Post-hoc robustness benchmarks (Gaussian noise + adversarial attacks)

`scripts/eval_robustness.py` adds two robustness numbers to every finished
image run, evaluated on the **exact checkpoint whose clean `test_accuracy`
is recorded in the run's `results.json`** (not the averaged checkpoint):

1. **Corruptions** — accuracy on the published corruption benchmarks
   (CIFAR-10-C / CIFAR-100-C / Tiny-ImageNet-C, Hendrycks & Dietterich
   2019): **every corruption type on disk × severities 1–5**, reported per
   type and per severity, plus the mean per severity over all types. **No
   noise is added here**: the corrupted images are read as shipped, and
   `severities` selects which pre-corrupted block to read. Only these three
   datasets have a published `-C` set; MNIST is out of scope.
2. **Adversarial attacks** — robust accuracy under
   [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
   attacks via a generic adapter. The default configuration runs
   **AutoAttack** (Linf, ε = 8/255, `version=standard`: APGD-CE + APGD-T +
   FAB-T + Square) on the **full test set**.

**Adversarial attacks need no training.** AutoAttack is a test-time attack
ensemble run against the frozen checkpoint; nothing in the training
pipeline changes. (Adversarial *training* would only be needed to produce
robust models, which is out of scope here.)

## Conventions

- Both benchmarks operate in **[0,1] pixel space**: the trailing
  `Normalize` of `transforms.eval` is popped from the data pipeline and
  re-applied inside the model (`src/robustness/normalization.py`). ε is
  therefore a fraction of the full dynamic range (ε = 8/255 ≈ 0.0314).
- The evaluation refuses to proceed if the recomputed clean accuracy
  deviates more than `2e-3` from the recorded `test_accuracy` (wrong
  checkpoint / wrong transform pipeline guard). The check is skipped when
  you pass an explicit `ckpt_path=`.
- Magnitude-pruned checkpoints (with `weight_orig`/`weight_mask`
  re-parametrization keys) are collapsed to plain tensors before loading
  (`collapse_pruning_reparam` in
  `src/callbacks/pruning/shared_prune_utils.py`); dense and Bregman
  checkpoints load unchanged.
- **How many corruption types** depends on the dataset: CIFAR-10-C and
  CIFAR-100-C ship **19** (the 15 benchmark corruptions plus 4 extra —
  `gaussian_blur`, `saturate`, `spatter`, `speckle_noise`), Tiny-ImageNet-C
  ships only the **15**. The list is discovered from disk, so the
  per-severity mean always covers what is actually there; `n_types` is
  recorded alongside it. Note the published mCE protocol averages the 15
  benchmark corruptions only — the mean here is over all types found.
- torchattacks 3.5.1 derives the APGD-T/FAB-T target-class count as
  `n_classes − 1`; the adapter caps it at 9 (the original AutoAttack
  convention), otherwise CIFAR-100/TinyImageNet would be 10–20× slower
  than the published protocol.

## Setup

```bash
pip install torchattacks==3.5.1          # into the training env

bash scripts/datasets/prep_cifar10_c.sh        # ~2.9 GB from Zenodo
bash scripts/datasets/prep_cifar100_c.sh       # ~2.9 GB
bash scripts/datasets/prep_tiny_imagenet_c.sh  # ~7.8 GB
```

## Running

Single run (smoke first — 200 attacked examples, csv logging only):

```bash
python scripts/eval_robustness.py \
    exp_dir=logs/train/runs/cifar10/wrn28_10/augmentation/dense_sgd-CosineAnnealing/seed_42 \
    robustness=default \
    robustness.attacks.autoattack.n_examples=200 \
    robustness.loggers=[csv]
```

Full defaults (full test set, wandb + csv + tensorboard):

```bash
python scripts/eval_robustness.py exp_dir=/path/to/run robustness=default
```

Whole sweep over `logs/train/runs` (skips unfinished runs and runs that
already carry a `robustness` block; `FORCE=1` redoes them):

```bash
bash scripts/run_robustness_sweep.sh
FORCE=1 bash scripts/run_robustness_sweep.sh robustness.loggers=[csv]
```

Useful overrides:

- `robustness.attacks.autoattack.n_examples=1000` — attack a subset
  (RobustBench-style) instead of the full test set.
- `robustness.batch_size=256` — evaluation batch size (routed to
  `datamodule.loaders.valid.batch_size`; `loaders.test` interpolates it).
- `robustness.loggers=[csv,tensorboard]` — skip the wandb resume.
- `ckpt_path=...` — evaluate a different checkpoint (disables the clean
  sanity check).

Runtime guidance: the corruption grid is plain forward passes, but it is
19 types × 5 severities × 10k images (95 test-set passes) for CIFAR — cut it
with `robustness.corruption.severities=[1,3,5]` when iterating.
AutoAttack on a full 10k test set is hours per run on one GPU — start with
`n_examples=200` to validate the setup, then launch the sweep. Runs are
processed sequentially because one AutoAttack saturates a GPU by itself.

## Where results land

1. **`results.json`** gains a `robustness` block (read-merge-write; the
   training entries are untouched, and a re-train that rewrites
   results.json correctly invalidates the block):

```json
"robustness": {
    "corruption": {
        "types": {"gaussian_noise": {"severity_1": 0.91, "severity_5": 0.57},
                  "fog": {"severity_1": 0.94, "severity_5": 0.79}},
        "mean": {"severity_1": 0.92, "severity_5": 0.68},
        "n_types": 19},
    "attacks": {"autoattack": {"name": "AutoAttack", "accuracy": 0.01,
                               "n_examples": 10000,
                               "kwargs": {"norm": "Linf", "eps": 0.0314,
                                          "version": "standard",
                                          "n_classes": 10, "seed": 0}}},
    "metadata": {"checkpoint_path": "...", "ckpt_epoch": 193,
                 "ckpt_global_step": 70400,
                 "clean_accuracy": 0.9727,
                 "clean_accuracy_reference": 0.9728,
                 "batch_size": 128, "torchattacks_version": "3.5.1",
                 "date": "..."}
}
```

2. **The run's own loggers** (rebuilt from `{exp_dir}/.hydra/config.yaml`,
   `save_dir` re-anchored to the run): the original **wandb** dashboard
   entry is resumed by its parsed id, **csv** appends a fresh
   `csv/version_1/metrics.csv` (already picked up by
   `scripts/visualize.py`'s `version_*` glob), **tensorboard** writes a new
   version under `{exp_dir}/tensorboard/`. Metrics are logged at
   `step = ckpt_global_step`:

   - `robust/clean`
   - `robust/corruption/{type}/severity_1..5` (one per corruption type) —
     **CSV only**; the per-type breakdown would be 19×5 series on a dashboard
   - `robust/corruption/mean/severity_1..5` (mean over all types)
   - `robust/attack/AutoAttack_Linf_eps0.03137` (the budget is embedded in
     the key so re-runs at other budgets never overwrite it)

   wandb and tensorboard get the summary only (clean, the per-severity mean,
   and attacks); CSV keeps the full per-type breakdown.

## Adding another attack

The adapter is generic — any torchattacks class works from config alone:

```bash
python scripts/eval_robustness.py exp_dir=... robustness=default \
    +robustness.attacks.pgd.name=PGD \
    +robustness.attacks.pgd.n_examples=null \
    '+robustness.attacks.pgd.kwargs={eps: 0.0314, alpha: 0.0078, steps: 10}'
```

or add a sibling of `autoattack:` in `configs/robustness/default.yaml`.

## Tests

```bash
pytest tests/test_robustness.py -m "not slow"   # unit tests, fabricated data
pytest tests/test_robustness.py -m slow         # AutoAttack smoke + full in-process pipeline
```
