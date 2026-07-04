# Image benchmarks (CIFAR-10/100, MNIST, TinyImageNet)

Standard image-classification recipes for validating the Bregman pruning stack
against citable published numbers. Same ResNet-18 backbone across datasets; the
pruning stack is unchanged from the speaker-verification task — only the
datamodule/module/config surface is new.

## Backbone

`src/modules/models/vision_resnet.py:build_resnet18` — torchvision ResNet-18
with the small-image stem (3×3 stride-1 `conv1`, no maxpool), `in_channels`
configurable (1 for MNIST, 3 otherwise). This is the widely cited CIFAR setup
(He et al., 2016) and matches Bungert et al.'s Bregman-learning experiments.

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

### ResNet-18 pruning groups (type-only matching)

Groups match by module **type**, not name. torchvision names downsample layers
`layerX.0.downsample.0/.1` (no "conv"/"bn" substring), so a name regex would
misroute downsample BatchNorm γ into the RegL1 group and shrink it. `conv1`
(stem) and `fc` (head) are pruned; add a module name to a group's
`exclude_module_name_patterns` to keep it dense.

## λ calibration note

The λ table (`src/utils/bregman_utils.py`) was calibrated on speech. Adaptive
runs re-calibrate λ online — treat the first CIFAR runs as calibration. Fixed
runs do **not** self-calibrate: the `fixed_lambda` values will likely need a
per-dataset sweep before the fixed variants reach the 0.9 sparsity target.

## References

- He et al., *Deep Residual Learning for Image Recognition*, 2016.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2017.
- Bungert et al., *A Bregman Learning Framework for Sparse Neural Networks*, 2022.
