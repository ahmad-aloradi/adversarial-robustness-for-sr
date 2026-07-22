"""Wide-ResNet (WRN) for CIFAR/TinyImageNet, via pytorchcv's reference impl.

Zagoruyko & Komodakis, "Wide Residual Networks" (2016). WRN-28-10
(depth=28, widen_factor=10) is the standard CIFAR-10/100 benchmark
configuration, matching Bungert et al.'s Bregman-learning experiments.

pytorchcv's `CIFARWRN` pools with a fixed 8x8 kernel, hardcoding the
assumption of a 32x32 input downsampled by its two stride-2 stages.
1. We swap that for `AdaptiveAvgPool2d(1)` — identical output to the
   fixed kernel on an actual 8x8 feature map, but resolution-agnostic —
   i.e., works on TinyImageNet's 64x64 (16x16 feature map).
2. MNIST enters at 32x32 too (padded, 3-channel — see its datamodule
   transforms), so one WRN covers every benchmark dataset.

Run standalone to check output shapes:

    python src/modules/models/wide_resnet.py
"""

if __name__ == "__main__":
    # Make the `src.*` import below resolve when running this file directly.
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

import torch
import torch.nn as nn
from pytorchcv.models.wrn_cifar import get_wrn_cifar

from src.utils.torch_utils import convert_norm

# All benchmark datasets enter at 32x32 (CIFAR, padded MNIST) or 64x64
# (TinyImageNet).
_SUPPORTED_INPUT_SIZES = (32, 64)


def _assert_input_size(module: nn.Module, inputs) -> None:
    (x,) = inputs
    h, w = x.shape[-2:]
    assert h == w and h in _SUPPORTED_INPUT_SIZES, (
        f"WRN-CIFAR expects a square input in {_SUPPORTED_INPUT_SIZES}, "
        f"got {tuple(x.shape[-2:])}"
    )


def build_wide_resnet(
    depth: int = 28,
    widen_factor: int = 10,
    num_classes: int = 10,
    in_channels: int = 3,
    norm: str = "bn",
) -> nn.Module:
    """WRN-`depth`-`widen_factor` (Zagoruyko & Komodakis, 2016).

    Args:
        depth: total conv depth, must be 6n+4 (28 -> n=4 blocks/stage).
        widen_factor: channel-width multiplier over the base [16,32,64]
            (the paper's "k"; 10 gives the standard WRN-28-10 CIFAR recipe).
        num_classes: number of output classes (fc head width).
        in_channels: input channels — 3 for RGB, 1 for grayscale.
        norm: "bn" keeps BatchNorm; "ln_last" makes only the last norm layer
            (`features.post_activ.bn`, which feeds the head) a LayerNorm;
            "ln_all" makes all of them LayerNorm.

    pytorchcv's reference model pools its last feature map with a fixed 8x8
    kernel, which only matches one input size:
      - CIFAR (32x32 in) -> 2 stride-2 stages -> 8x8 feature map: fits.
      - TinyImageNet (64x64 in) -> same stages -> 16x16 feature map: doesn't.
    Fixed pool is replaced by global average pooling (mean over whatever
    feature map it gets --> 1 value per channel).
    """
    assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
    assert in_channels in (
        1,
        3,
    ), f"in_channels must be 1 or 3, got {in_channels}"
    model = get_wrn_cifar(
        num_classes=num_classes,
        blocks=depth,
        width_factor=widen_factor,
        in_channels=in_channels,
        model_name=f"wrn{depth}_{widen_factor}",
    )
    model.features.final_pool = nn.AdaptiveAvgPool2d(output_size=1)
    model.register_forward_pre_hook(_assert_input_size)
    return convert_norm(model, norm)


if __name__ == "__main__":
    net = build_wide_resnet(num_classes=10, in_channels=3)
    for size in _SUPPORTED_INPUT_SIZES:
        out = net(torch.randn(2, 3, size, size))
        print(
            f"WRN-28-10: input (2,3,{size},{size}) -> logits {tuple(out.shape)}"
        )
