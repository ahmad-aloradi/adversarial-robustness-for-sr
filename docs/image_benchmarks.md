# Image benchmarks (CIFAR-10/100, MNIST, TinyImageNet)

Standard image-classification recipes for validating the Bregman pruning stack
against citable published numbers. ResNet-18 is the default backbone across
datasets; the pruning stack is unchanged from the speaker-verification task —
only the datamodule/module/config surface is new.

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
| `wrn28_10` | `wide_resnet.build_wide_resnet` | CIFAR-only (see below) |

All `vision_resnet.build_resnet` variants take `in_channels` (1 for MNIST, 3
otherwise) and work at any input resolution (adaptive pooling), so they run
unchanged across all four datasets.

`wrn28_10` is Zagoruyko & Komodakis' Wide-ResNet-28-10 — architecturally
distinct from torchvision's `wide_resnet50_2` (pre-activation blocks, no
bottleneck, built for 32×32 from the start) and the standard CIFAR-10/100
WRN benchmark configuration. It wraps `pytorchcv`'s reference implementation
(`pytorchcv.models.wrn_cifar`), which pools with a **fixed 8×8 kernel** — so
it only accepts 32×32 input (CIFAR-10/100); a forward pre-hook asserts this
so an MNIST/TinyImageNet mismatch fails loud instead of an opaque matmul
shape error. Use `vision_resnet.build_resnet` for those datasets instead.

Selecting a backbone for a new experiment (any dataset, any recipe — dense,
Bregman, or magnitude pruning) needs no other changes: the compression
callbacks match parameters by module **type**, not name (see below), so they
route Bottleneck/pre-activation parameters exactly like BasicBlock ones.

```bash
python src/train.py experiment=img/cifar10_dense_sgd_resnet50
python src/train.py experiment=img/cifar10_dense_sgd_wrn28_10
```

## Recipes

| Dataset | Train augmentation | Normalize (mean / std) | Optimizer | Epochs |
|---|---|---|---|---|
| CIFAR-10 | RandomCrop(32, pad 4) + HFlip | (.4914,.4822,.4465)/(.2470,.2435,.2616) | SGD lr 0.1, mom 0.9, wd 5e-4, cosine, batch 128 | 200 (≈95.5%) |
| CIFAR-100 | same | (.5071,.4865,.4409)/(.2673,.2564,.2762) | same | 200 (≈77–78%) |
| MNIST | none | (.1307)/(.3081), 1 channel | same | 30 dense / 50 Bregman (≈99.5%) |
| TinyImageNet | RandomCrop(64, pad 4) + HFlip | (.4802,.4481,.3975)/(.2770,.2691,.2821) | same, batch 256 | 100 (≈60–65%) |

Eval transforms are ToTensor + Normalize only. Validation uses the official test
split (CIFAR/MNIST literature practice); TinyImageNet uses its official val split
(test labels are not public).

## Data prep

CIFAR/MNIST auto-download on first run. On offline compute nodes, pre-fetch on a
login node:

```bash
bash scripts/datasets/prep_cifar10.sh      # or prep_cifar100 / prep_mnist
bash scripts/datasets/prep_tiny_imagenet.sh
```

TinyImageNet is not a torchvision dataset: the prep downloads the archive and
builds `val_structured/<wnid>/` from `val_annotations.txt` (train is already
ImageFolder-shaped) so train and val share one `class_to_idx`.

## Experiments

Per dataset: a dense SGD baseline plus four Bregman variants — AdaBreg and
LinBreg, each in an adaptive-λ and a fixed-λ flavour.

| Experiment suffix | Optimizer | λ regime |
|---|---|---|
| `_dense_sgd` | SGD | — (dense baseline) |
| `_bregman_adabreg` | AdaBreg | adaptive (feedback controller → target sparsity) |
| `_bregman_adabreg_fixed` | AdaBreg | fixed `fixed_lambda`, no scheduler |
| `_bregman_linbreg` | LinBreg | adaptive |
| `_bregman_linbreg_fixed` | LinBreg | fixed |

```bash
python src/train.py experiment=img/cifar10_dense_sgd
python src/train.py experiment=img/cifar10_bregman_adabreg
python src/train.py experiment=img/cifar10_bregman_adabreg_fixed
python src/train.py experiment=img/cifar10_bregman_linbreg
python src/train.py experiment=img/cifar10_bregman_linbreg_fixed
# swap cifar10 -> cifar100 / mnist / tinyimagenet
```

`cifar10_bregman_adabreg.yaml` is the parent: it holds the only full ResNet-18
`pruning_groups` block. The other three CIFAR-10 variants and all cross-dataset
files inherit it and change only the optimizer, λ source, dataset, or epoch
budget.

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
