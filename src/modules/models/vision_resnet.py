"""Torchvision ResNet family with the standard small-image stem.

The stock torchvision ResNet down-samples 4x in its stem (7x7 stride-2 conv
+ 3x3 stride-2 maxpool), which throws away most of a 32x32 CIFAR image before
the first residual block. The widely cited CIFAR recipe (and Bungert et al.'s
Bregman setup) replaces the stem with a 3x3 stride-1 conv and drops the
maxpool, so the spatial resolution is preserved into `layer1`. Every variant
below shares this exact stem (conv1 is always a 64-channel 7x7 stride-2 conv
regardless of depth/width), so the same replacement applies uniformly.

Run standalone to check output shapes:

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
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}


def build_resnet(
    arch: str, num_classes: int, in_channels: int = 3
) -> nn.Module:
    """Torchvision ResNet variant with a small-image stem and a fresh head.

    Args:
        arch: one of "resnet18", "resnet34", "resnet50", "resnet101",
            "resnet152", "wide_resnet50_2", "wide_resnet101_2".
        num_classes: number of output classes (fc head width).
        in_channels: input channels — 3 for RGB, 1 for grayscale (the
            benchmark datamodules all emit 3-channel images).

    Returns the torchvision model whose `conv1` is a 3x3 stride-1 conv over
    `in_channels` and whose `maxpool` is removed.
    """
    assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
    assert in_channels in (
        1,
        3,
    ), f"in_channels must be 1 or 3, got {in_channels}"
    assert (
        arch in _ARCHS
    ), f"unknown arch {arch!r}, expected one of {sorted(_ARCHS)}"

    model = _ARCHS[arch](weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    nn.init.kaiming_normal_(
        model.conv1.weight, mode="fan_out", nonlinearity="relu"
    )
    model.maxpool = nn.Identity()
    return model


if __name__ == "__main__":
    for arch in ("resnet18", "resnet50", "wide_resnet50_2"):
        net = build_resnet(arch, num_classes=10, in_channels=3)
        out = net(torch.randn(2, 3, 32, 32))
        print(f"{arch}: input (2,3,32,32) -> logits {tuple(out.shape)}")
