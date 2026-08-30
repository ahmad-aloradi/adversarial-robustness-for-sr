# Image benchmarks (CIFAR-10/100, MNIST, TinyImageNet, ImageNet-1k)

Image-classification recipes for validating the Bregman pruning stack against
published numbers.

- One backbone for every dataset (ResNet-18 default) — datamodules adapt the
  data, the model never changes.
- One experiment file = one method. Dataset and backbone are config-group swaps.
- All datamodules emit 3-channel images; head width follows `num_classes`, stem
  follows input size.

```bash
python src/train.py experiment=img/dense_sgd                                   # CIFAR-10 (default)
python src/train.py experiment=img/bregman_adabreg datamodule=datasets/cifar100
python src/train.py experiment=img/dense_sgd module/img_model=wrn28_10 datamodule=datasets/tinyimagenet
```

## Data prep

CIFAR/MNIST auto-download. Offline nodes: pre-fetch on a login node.

```bash
bash scripts/datasets/prep_cifar10.sh      # or prep_cifar100 / prep_mnist
bash scripts/datasets/prep_tiny_imagenet.sh
bash scripts/datasets/prep_imagenet.sh     # ~150 GB (~300 GB peak)
```

- All take `--data-dir DIR` (default the repo's `data/`), which must match
  `paths.data_dir`. Cluster jobs use `--data-dir $WOODY_DIR/datasets`.
- TinyImageNet: prep builds `val_structured/<wnid>/` from `val_annotations.txt`;
  that is the test set (`test/` is unlabeled).
- ImageNet: prep sorts `val/` into `<wnid>/` folders matching `train/`. `val/` is
  the test set — the ILSVRC test labels were never released.
- ImageNet prep is safe to re-run and asserts 1281167 / 50000 images at the end.
  If that assert fires, delete the named split folder and re-run.

## Datasets

| Dataset | Input | Classes | Epochs | Reference top-1 |
|---|---|---|---|---|
| CIFAR-10 | 32×32 | 10 | 200 | ≈95.5% |
| CIFAR-100 | 32×32 | 100 | 200 | ≈77–78% |
| MNIST | 32×32 (padded, gray→3ch) | 10 | 120 | ≈99.5% |
| TinyImageNet | 64×64 | 200 | 200 | ≈60–65% |
| ImageNet-1k | 224×224 | 1000 | 100 | — (see below) |

- Transforms, batch size and the epoch budget live in
  `configs/datamodule/datasets/<name>.yaml`; experiments read them, so swapping
  the dataset swaps the whole budget. The lr and its schedule live in the
  experiment config, not the dataset one.
- `datamodule.augmentation=false` gives the train split the eval pipeline.
- Validation is a class-stratified carve from train, sized by
  `valid_dataset.split` (10%; 2% on ImageNet, where 10% would hold back 128k
  images) and seeded by `valid_dataset.split_seed`.
- No ImageNet reference number: the published ≈70% top-1 at 90 epochs uses weight
  decay 1e-4, this shared recipe keeps CIFAR's 5e-4. First run is calibration.

## Experiments

| Experiment | Optimizer | λ regime |
|---|---|---|
| `dense_sgd` | SGD | — (dense baseline) |
| `bregman_adabreg` | AdaBreg | adaptive λ (feedback loop → target sparsity) |
| `bregman_adabreg_fixed` | AdaBreg | fixed λ, no scheduler |
| `bregman_adabreg_progressive` | AdaBreg | adaptive λ, target ramped 0.5 → target |
| `bregman_linbreg` | LinBreg | adaptive λ |
| `bregman_linbreg_fixed` | LinBreg | fixed λ |
| `bregman_linbreg_progressive` | LinBreg | adaptive λ, target ramped 0.5 → target |
| `proxsgd` | ProxSGD | adaptive λ; dense start |
| `proxsgd_fixed` | ProxSGD | fixed λ |
| `pruning_mag_struct` | SGD | gradual magnitude pruning, `ln_structured`; 50% sparse start |
| `pruning_mag_unstruct` | SGD | gradual magnitude pruning, global `l1_unstructured` |
| `pruning_rigl` | SGD | RigL — sparse from step 0, regrow on the gradient |
| `pruning_set` | SGD | SET — RigL with random regrowth |
| `pruning_static` | SGD | Static-ERK — one random mask, never updated |
| `pruning_snip` | SGD | SNIP — one-shot \|w·grad\| ranking at init |
| `pruning_snip_iter` | SGD | iterative SNIP — the same ranking over 100 steps |
| `pruning_granet` | SGD | GraNet — 50% sparse start, cubic ramp, RigL regrowth |
| `soft_threshold` | SGD | STR — learned per-layer threshold, sparsity is an outcome |

- `bregman_adabreg.yaml` is the parent — it holds the only full ResNet-18
  `pruning_groups` block; variants swap the optimizer and λ source.
  `pruning_mag_unstruct` inherits `pruning_mag_struct`.
- `pruning_rigl.yaml` is the parent of the six sparse-training baselines; see
  [sparse_training.md](sparse_training.md).
- Target sparsity: `_bregman_target_sparsity` (Bregman) or
  `callbacks.model_pruning.amount` (the rest). A fixed-λ run sets λ directly
  with `_bregman_fixed_lambda`.

### The ramped recipes share one ramp

`pruning_mag_struct` / `_unstruct`, `pruning_granet` and both `*_progressive`
recipes share start (0.5), shape (cubic) and end, so at epoch *t* they all aim
at one sparsity — which is what makes their per-epoch validation curves
comparable. Those curves exist because the validation gate is open on the img
recipes while the selection gates band the target; see [pruning.md](pruning.md)
§3.

Every ramp knob reads `_sparsity_ramp_epochs`, so one override moves them all:

| Experiment | ramp knob reading `_sparsity_ramp_epochs` |
| --- | --- |
| `pruning_mag_struct` / `_unstruct` | `callbacks.model_pruning.epochs_to_ramp` |
| `pruning_granet` | `callbacks.model_pruning.final_prune_epoch` |
| `bregman_{adabreg,linbreg}_progressive` | `_bregman_ramp_epochs` |

```bash
python src/train.py experiment=img/pruning_granet datamodule=datasets/cifar100 _sparsity_ramp_epochs=40
```

### What "sparsity" means

Every method reports zeros over **all weight tensors, norms and biases aside**.
Every recipe sparsifies all of them, so one `amount` means the same thing for
RigL, GraNet, magnitude pruning and Bregman alike. A layer held dense would
still count in that denominator at full size with no zeros. `sparsity` is the
whole model including BatchNorm and biases, which no method sparsifies.

That one quantity is published under two keys, and each gate must read the one
its own pruner writes — `_bregman_sparsity_metric` exists for exactly this:

| Pruner | Gate metric |
| --- | --- |
| magnitude, DST (RigL/SET/Static/SNIP/GraNet), STR | `pruning/sparsity` |
| Bregman (AdaBreg/LinBreg/ProxSGD) | `bregman/pruned_sparsity` |

### Pruning groups

- Groups match by module **type**, not name: torchvision names downsample layers
  `layerX.0.downsample.0/.1` (no "conv"/"bn" substring), so a name regex would
  misroute downsample BatchNorm γ into the RegL1 group and shrink it.
- That is why every backbone below works with no config change — covered by
  `tests/test_vision_resnet.py` / `tests/test_wide_resnet.py`.
- `conv1` and `fc` are both pruned. Add a name to a group's
  `exclude_module_name_patterns` to keep a layer dense.

## Backbones

Config group `configs/module/img_model/` at package `module.model.net`, default
`resnet18`. Override on any experiment: `module/img_model=<name>`.

| `img_model` name | Builder | Notes |
|---|---|---|
| `resnet18` (default) | `vision_resnet.build_resnet` | He 2016 / Bungert baseline |
| `resnet34`, `resnet50`, `resnet101`, `resnet152` | `vision_resnet.build_resnet` | deeper torchvision variants |
| `wide_resnet50_2`, `wide_resnet101_2` | `vision_resnet.build_resnet` | torchvision wide ResNets, 2× Bottleneck width |
| `wrn28_10` | `wide_resnet.build_wide_resnet` | CIFAR WRN-28-10 benchmark config |

- ResNet stem follows the dataset: 3×3 stride-1 without maxpool at ≤64px,
  torchvision's stock 7×7 stride-2 above. A new dataset config needs its `name`
  in `_IMAGE_SIZES` (`src/modules/models/vision_resnet.py`).
- `wrn28_10` is Zagoruyko & Komodakis' WRN-28-10 (pre-activation blocks, no
  bottleneck) — a different architecture from `wide_resnet50_2`. A forward
  pre-hook accepts 32 and 64 px only; use `wide_resnet50_2` at ImageNet scale.

## λ calibration

- The λ table (`src/utils/bregman_utils.py`) was calibrated on speech — treat the
  first image runs as calibration.
- Adaptive runs re-calibrate λ online. Fixed runs do not: `fixed_lambda` needs a
  per-dataset sweep to reach the target.

## References

He et al., *Deep Residual Learning* (2016) · Zagoruyko & Komodakis, *Wide
Residual Networks* (2016) · Bungert et al., *A Bregman Learning Framework for
Sparse Neural Networks* (2022).
