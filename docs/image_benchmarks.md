# Image benchmarks (CIFAR-10/100, MNIST, TinyImageNet)

Standard image-classification recipes for validating the Bregman pruning stack
against citable published numbers. One fixed backbone (ResNet-18 by default)
runs unchanged on every dataset: all datamodules emit 3-channel images at
32x32 (CIFAR, padded MNIST) or 64x64 (TinyImageNet), so only the classifier
head width follows `num_classes`. Each experiment file defines one *method*;
the dataset is swapped as a config group:

```bash
python src/train.py experiment=img/dense_sgd                                   # CIFAR-10 (default)
python src/train.py experiment=img/dense_sgd datamodule=datasets/tinyimagenet
```

The pruning stack is unchanged from the speaker-verification task — only the
datamodule/module/config surface is new.

## Backbones

Backbone is a swappable Hydra config group, `configs/module/img_model/`, at
package `module.model.net` (mirrors how `configs/module/sv_model/` swaps the
SV encoder). `configs/module/img.yaml` defaults to `resnet18`; an experiment
config overrides it wholesale via `override /module/img_model: <name>` (a
config-group swap, not a key-by-key merge, so it never leaks a stale kwarg
from the previous backbone, e.g. WRN's `depth`/`widen_factor` vs. ResNet's
`arch`).

| `img_model` name | Builder | Notes |
|---|---|---|
| `resnet18` (default) | `vision_resnet.build_resnet` | small-image stem (3×3 stride-1 `conv1`, no maxpool); matches He et al., 2016 / Bungert et al. |
| `resnet34`, `resnet50`, `resnet101`, `resnet152` | `vision_resnet.build_resnet` | same stem swap, deeper torchvision variants |
| `wide_resnet50_2`, `wide_resnet101_2` | `vision_resnet.build_resnet` | torchvision's ImageNet-style wide ResNets (2x Bottleneck width) |
| `wrn28_10` | `wide_resnet.build_wide_resnet` | Zagoruyko & Komodakis' CIFAR WRN benchmark config (see below) |

Every backbone consumes the shared 3-channel input contract (`in_channels: 3`
in the `img_model` configs); the datamodules adapt the data, never the model —
MNIST is padded to 32×32 and replicated to 3 channels in its transforms.

`wrn28_10` is Zagoruyko & Komodakis' Wide-ResNet-28-10 — architecturally
distinct from torchvision's `wide_resnet50_2` (pre-activation blocks, no
bottleneck, built for 32×32 from the start) and the standard CIFAR-10/100
WRN benchmark configuration. It wraps `pytorchcv`'s reference implementation
(`pytorchcv.models.wrn_cifar`), whose fixed 8×8 final-pool kernel only matches
a 32×32 input; `build_wide_resnet` replaces it with global average pooling
(same weights, no new parameters, identical result on CIFAR's 8×8 case), so
the same backbone also covers TinyImageNet's 64×64 (16×16 feature map). A
forward pre-hook asserts a supported square input (32 or 64) so a mismatch
fails loud instead of an opaque matmul shape error.

Selecting a backbone for a new experiment (any dataset, any recipe — dense,
Bregman, or magnitude pruning) needs no other changes: the compression
callbacks match parameters by module **type**, not name (see below), so they
route Bottleneck/pre-activation parameters exactly like BasicBlock ones.
There is no dedicated experiment file per backbone — override the config
group directly on top of any experiment:

```bash
python src/train.py experiment=img/dense_sgd module/img_model=resnet50
python src/train.py experiment=img/dense_sgd module/img_model=wrn28_10
python src/train.py experiment=img/dense_sgd module/img_model=wrn28_10 datamodule=datasets/tinyimagenet
```

## Recipes

| Dataset | Train augmentation | Normalize (mean / std) | Optimizer | Epochs |
|---|---|---|---|---|
| CIFAR-10 | RandomCrop(32, pad 4) + HFlip | (.4914,.4822,.4465)/(.2470,.2435,.2616) | SGD lr 0.1, mom 0.9, wd 5e-4, cosine, batch 128 | 200 (≈95.5%) |
| CIFAR-100 | same | (.5071,.4865,.4409)/(.2673,.2564,.2762) | same | 200 (≈77–78%) |
| MNIST | none (Pad 28→32, gray→3ch) | (.1307)/(.3081) ×3 channels | same | 50 (≈99.5%) |
| TinyImageNet | RandomCrop(64, pad 4) + HFlip | (.4802,.4481,.3975)/(.2770,.2691,.2821) | same, batch 256 | 100 (≈60–65%) |

Eval transforms are ToTensor + Normalize only (plus MNIST's pad/replicate).
Validation is a class-stratified 10% carved from the train set (every class
appears in both, seeded by `valid_dataset.split_seed`). Test: CIFAR/MNIST use
the official test split; TinyImageNet has no public test labels, so its labeled
`val_structured` serves as the test set.

The epoch budget lives in the dataset config (`max_epochs` in
`configs/datamodule/datasets/<name>.yaml`); experiments read it via
`${datamodule.max_epochs}`, so swapping the dataset swaps the budget too.

## Data prep

CIFAR/MNIST auto-download on first run. On offline compute nodes, pre-fetch on a
login node:

```bash
bash scripts/datasets/prep_cifar10.sh      # or prep_cifar100 / prep_mnist
bash scripts/datasets/prep_tiny_imagenet.sh
```

TinyImageNet is not a torchvision dataset: the prep downloads the archive and
builds `val_structured/<wnid>/` from `val_annotations.txt` (train is already
ImageFolder-shaped) so train and the `val_structured` test set share one
`class_to_idx`.

## Experiments

One file per method — the dataset is not part of the experiment name:

| Experiment | Optimizer | λ / schedule regime |
|---|---|---|
| `dense_sgd` | SGD | — (dense baseline) |
| `bregman_adabreg` | AdaBreg | adaptive λ (feedback controller → target sparsity) |
| `bregman_adabreg_fixed` | AdaBreg | fixed `fixed_lambda`, no scheduler |
| `bregman_linbreg` | LinBreg | adaptive λ |
| `bregman_linbreg_fixed` | LinBreg | fixed λ |
| `pruning_mag_struct` | SGD | gradual magnitude pruning, `ln_structured`, constant rate |
| `pruning_mag_unstruct` | SGD | gradual magnitude pruning, global `l1_unstructured`, constant rate |

```bash
python src/train.py experiment=img/dense_sgd                                  # CIFAR-10 (default)
python src/train.py experiment=img/bregman_adabreg datamodule=datasets/cifar100
python src/train.py experiment=img/bregman_linbreg datamodule=datasets/mnist
python src/train.py experiment=img/pruning_mag_unstruct datamodule=datasets/tinyimagenet
```

`bregman_adabreg.yaml` is the parent: it holds the only full ResNet-18
`pruning_groups` block. The other Bregman variants inherit it and change only
the optimizer and λ source; `pruning_mag_unstruct` likewise inherits
`pruning_mag_struct`.

### Pruning groups (type-only matching)

Groups match by module **type**, not name. torchvision names downsample layers
`layerX.0.downsample.0/.1` (no "conv"/"bn" substring), so a name regex would
misroute downsample BatchNorm γ into the RegL1 group and shrink it. `conv1`
(stem) and `fc` (head) are pruned; add a module name to a group's
`exclude_module_name_patterns` to keep it dense. Type-only matching is why
this generalizes to every backbone in the table above without any config
changes — verified directly for Bottleneck's `downsample.0/.1` (resnet50,
wide_resnet50_2) and WRN's pre-activation blocks in
`tests/test_vision_resnet.py` / `tests/test_wide_resnet.py`.

## λ calibration note

The λ table (`src/utils/bregman_utils.py`) was calibrated on speech. Adaptive
runs re-calibrate λ online — treat the first CIFAR runs as calibration. Fixed
runs do **not** self-calibrate: the `fixed_lambda` values will likely need a
per-dataset sweep before the fixed variants reach the 0.9 sparsity target.

## References

- He et al., *Deep Residual Learning for Image Recognition*, 2016.
- Zagoruyko & Komodakis, *Wide Residual Networks*, 2016.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2017.
- Bungert et al., *A Bregman Learning Framework for Sparse Neural Networks*, 2022.
