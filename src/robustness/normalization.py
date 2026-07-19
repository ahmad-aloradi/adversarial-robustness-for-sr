"""Move input normalization from the data pipeline into the model.

Adversarial attacks (torchattacks) and the corruption datasets both operate
on [0,1] pixel-space images, so the trailing ``Normalize`` of the training
config's ``transforms.eval`` must be popped from the loader and re-applied
inside the model. ``extract_normalization`` does the popping (config side),
``NormalizedModel`` does the re-applying (model side).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_NORMALIZE_TARGET = "torchvision.transforms.Normalize"


class NormalizedModel(nn.Module):
    """Wrap ``net`` so it accepts [0,1] images and normalizes internally."""

    def __init__(self, net: nn.Module, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        assert (
            mean.ndim == 1 and mean.shape == std.shape
        ), f"mean/std must be 1-D and matching, got {mean.shape}/{std.shape}"
        self.net = net
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
        assert (
            x.shape[1] == self.mean.shape[1]
        ), f"expected {self.mean.shape[1]} channels, got {x.shape[1]}"
        # Small tolerance: interpolation/augment-free eval pipelines emit
        # exact [0,1], but float noise addition may leave 1+eps before clamp.
        # detach(): attacks call this on requires_grad inputs.
        xmin, xmax = float(x.detach().min()), float(x.detach().max())
        assert xmin >= -1e-4 and xmax <= 1 + 1e-4, (
            f"input outside [0,1]: min={xmin}, max={xmax}"
            " — was Normalize left in the data pipeline?"
        )
        return self.net((x - self.mean) / self.std)


def extract_normalization(eval_specs) -> tuple[list, list[float], list[float]]:
    """Split a ``transforms.eval`` spec list into (specs-without-Normalize,
    mean, std).

    The training configs place ``Normalize`` last in ``transforms.eval``;
    everything before it (ToTensor, and any dataset-specific resizing) is
    kept verbatim. Anything else is a config drift we refuse to guess around.
    """
    specs = list(eval_specs)
    assert specs, "transforms.eval is empty"
    last = specs[-1]
    target = last.get("_target_") if hasattr(last, "get") else None
    assert (
        target == _NORMALIZE_TARGET
    ), f"expected last eval transform to be {_NORMALIZE_TARGET}, got {target!r}"
    mean = [float(v) for v in last["mean"]]
    std = [float(v) for v in last["std"]]
    return specs[:-1], mean, std


if __name__ == "__main__":
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.LazyLinear(10))
    mean, std = [0.49, 0.48, 0.45], [0.25, 0.24, 0.26]
    wrapped = NormalizedModel(net, mean, std).eval()
    x = torch.rand(2, 3, 32, 32)
    from torchvision.transforms import Normalize

    with torch.no_grad():
        direct = net(Normalize(mean, std)(x))
        via_wrap = wrapped(x)
    print("max |diff|:", float((direct - via_wrap).abs().max()))  # ~0

    specs = [
        {"_target_": "torchvision.transforms.ToTensor"},
        {"_target_": _NORMALIZE_TARGET, "mean": mean, "std": std},
    ]
    rest, m, s = extract_normalization(specs)
    print("popped:", len(rest) == 1 and m == mean and s == std)  # True
