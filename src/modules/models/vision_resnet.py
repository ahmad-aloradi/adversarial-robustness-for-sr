"""torchvision ResNet-18 with the standard small-image stem.

The stock torchvision ResNet-18 down-samples 4x in its stem (7x7 stride-2 conv
+ 3x3 stride-2 maxpool), which throws away most of a 32x32 CIFAR image before
the first residual block. The widely cited CIFAR recipe (and Bungert et al.'s
Bregman setup) replaces the stem with a 3x3 stride-1 conv and drops the
maxpool, so the spatial resolution is preserved into `layer1`.

Run standalone to check output shapes:

    python src/modules/models/vision_resnet.py
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int, in_channels: int = 3) -> nn.Module:
    """ResNet-18 with a small-image stem and a fresh classifier head.

    Args:
        num_classes: number of output classes (fc head width).
        in_channels: input channels — 3 for RGB (CIFAR/TinyImageNet), 1 for
            grayscale (MNIST).

    Returns a torchvision ResNet-18 whose `conv1` is a 3x3 stride-1 conv over
    `in_channels` and whose `maxpool` is removed.
    """
    assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
    assert in_channels in (1, 3), f"in_channels must be 1 or 3, got {in_channels}"

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    nn.init.kaiming_normal_(
        model.conv1.weight, mode="fan_out", nonlinearity="relu"
    )
    model.maxpool = nn.Identity()
    return model


if __name__ == "__main__":
    for name, (c, hw, n) in {
        "CIFAR-10 (32x32 RGB)": (3, 32, 10),
        "MNIST (28x28 gray)": (1, 28, 10),
        "TinyImageNet (64x64 RGB)": (3, 64, 200),
    }.items():
        net = build_resnet18(num_classes=n, in_channels=c)
        out = net(torch.randn(2, c, hw, hw))
        print(f"{name}: input (2,{c},{hw},{hw}) -> logits {tuple(out.shape)}")
