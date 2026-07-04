"""Wide-ResNet (WRN) for CIFAR, via pytorchcv's reference implementation.

Zagoruyko & Komodakis, "Wide Residual Networks" (2016). WRN-28-10
(depth=28, widen_factor=10) is the standard CIFAR-10/100 benchmark
configuration, matching Bungert et al.'s Bregman-learning experiments.

pytorchcv's `CIFARWRN` pools with a fixed 8x8 kernel (i.e. assumes a 32x32
input downsampled by its two stride-2 stages to 8x8) — CIFAR-only, not for
MNIST (28x28) or TinyImageNet (64x64); use `vision_resnet.build_resnet` for
those instead.

Run standalone to check output shapes:

    python src/modules/models/wide_resnet.py
"""

import torch
import torch.nn as nn
from pytorchcv.models.wrn_cifar import get_wrn_cifar

_INPUT_SIZE = 32


def _assert_input_size(module: nn.Module, inputs) -> None:
    (x,) = inputs
    assert x.shape[-2:] == (_INPUT_SIZE, _INPUT_SIZE), (
        f"WRN-CIFAR expects {_INPUT_SIZE}x{_INPUT_SIZE} input (pytorchcv's "
        f"CIFARWRN pools with a fixed 8x8 kernel), got {tuple(x.shape[-2:])}"
    )


def build_wide_resnet(
    depth: int = 28,
    widen_factor: int = 10,
    num_classes: int = 10,
    in_channels: int = 3,
) -> nn.Module:
    """WRN-`depth`-`widen_factor` (Zagoruyko & Komodakis, 2016), CIFAR-only.

    Args:
        depth: total conv depth, must be 6n+4 (28 -> n=4 blocks/stage).
        widen_factor: channel-width multiplier over the base [16,32,64]
            (the paper's "k"; 10 gives the standard WRN-28-10 CIFAR recipe).
        num_classes: number of output classes (fc head width).
        in_channels: input channels — 3 for RGB, 1 for grayscale.

    Thin wrapper around ``pytorchcv.models.wrn_cifar.get_wrn_cifar``; a
    forward pre-hook asserts 32x32 input so a resolution mismatch fails
    loud instead of surfacing as an opaque matmul shape error.
    """
    assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
    assert in_channels in (1, 3), (
        f"in_channels must be 1 or 3, got {in_channels}"
    )
    model = get_wrn_cifar(
        num_classes=num_classes,
        blocks=depth,
        width_factor=widen_factor,
        in_channels=in_channels,
        in_size=(_INPUT_SIZE, _INPUT_SIZE),
        model_name=f"wrn{depth}_{widen_factor}",
    )
    model.register_forward_pre_hook(_assert_input_size)
    return model


if __name__ == "__main__":
    net = build_wide_resnet(num_classes=10, in_channels=3)
    out = net(torch.randn(2, 3, _INPUT_SIZE, _INPUT_SIZE))
    print(f"WRN-28-10: input (2,3,32,32) -> logits {tuple(out.shape)}")
