"""Torchvision ResNet family with the standard small-image stem.

The stock torchvision ResNet down-samples 4x in its stem (7x7 stride-2 conv
+ 3x3 stride-2 maxpool), which throws away most of a 32x32 CIFAR image before
the first residual block. The widely cited CIFAR recipe (and Bungert et al.'s
Bregman setup) replaces the stem with a 3x3 stride-1 conv and drops the
maxpool, so the spatial resolution is preserved into `layer1`. Every variant
below shares this exact stem (conv1 is always a 64-channel 7x7 stride-2 conv
regardless of depth/width), so the same replacement applies uniformly.

`dataset_name` picks between the two stems via `_IMAGE_SIZES`, so one backbone
config serves every dataset.

Run standalone to compare the stem variants by their pre-pool feature map:

    python src/modules/models/vision_resnet.py
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    wide_resnet50_2,
    wide_resnet101_2,
)

_ARCHS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    # widened-bottleneck ImageNet ResNets; the CIFAR WRN-28-10 lives in wide_resnet.py
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}

# Side length each benchmark datamodule emits, keyed by its `name`. MNIST is
# padded 28 -> 32 upstream; TinyImageNet crops to 64; ImageNet crops to 224.
_IMAGE_SIZES = {
    "mnist": 32,
    "cifar10": 32,
    "cifar100": 32,
    "tinyimagenet": 64,
    "imagenet": 224,
}


def build_resnet(
    arch: str,
    num_classes: int,
    dataset_name: str,
    in_channels: int = 3,
) -> nn.Module:
    """Torchvision ResNet with a size-appropriate stem and a fresh head.

    Args:
        arch: one of "resnet18", "resnet34", "resnet50", "resnet101",
            "resnet152", "wide_resnet50_2", "wide_resnet101_2".
        num_classes: number of output classes (fc head width).
        dataset_name: the datamodule's `name`; `_IMAGE_SIZES` turns it into the
            input side length, which picks the stem.
        in_channels: input channels — 3 for RGB, 1 for grayscale. Only the
            small-image stem is rebuilt, so above 64px it must be 3.

    Returns the torchvision model whose stem matches the dataset: 3x3 stride-1
    with no maxpool at or below 64px, torchvision's stock 7x7 stride-2 above.
    """
    assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
    assert in_channels in (
        1,
        3,
    ), f"in_channels must be 1 or 3, got {in_channels}"
    assert (
        arch in _ARCHS
    ), f"unknown arch {arch!r}, expected one of {sorted(_ARCHS)}"
    assert (
        dataset_name in _IMAGE_SIZES
    ), f"unknown dataset {dataset_name!r}, expected one of {sorted(_IMAGE_SIZES)}"
    small_stem = _IMAGE_SIZES[dataset_name] <= 64
    assert small_stem or in_channels == 3, "the stock stem is RGB-only"

    model = _ARCHS[arch](weights=None, num_classes=num_classes)

    # a 4x-downsampling stem would leave a 32x32 image at 8x8 before layer1
    if small_stem:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            model.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        model.maxpool = nn.Identity()
    return model


def _features_before_pool(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run stem + all four stages, stopping just before the global pool."""
    for name, layer in net.named_children():
        if name == "avgpool":
            return x
        x = layer(x)
    raise AssertionError("avgpool child not found on torchvision ResNet")


if __name__ == "__main__":

    # one dataset per distinct input size; the rest repeat 32x32
    for dataset_name in ("cifar10", "tinyimagenet", "imagenet"):
        size = _IMAGE_SIZES[dataset_name]
        x = torch.randn(2, 3, size, size)
        print(f"\n== {dataset_name}, input {tuple(x.shape)} ==")
        for arch in ("resnet18", "resnet50"):
            net = build_resnet(arch, 10, dataset_name).eval()
            with torch.no_grad():
                feat = _features_before_pool(net, x)
            c, h, w = feat.shape[1:]
            stem = "small" if size <= 64 else "stock"
            print(
                f"  {arch:16s} ({stem} stem): pre-pool {tuple(feat.shape)} "
                f"= {c}ch x {h*w} spatial"
            )
