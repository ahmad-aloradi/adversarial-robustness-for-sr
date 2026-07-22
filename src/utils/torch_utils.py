import torch
from torch import nn
from typing import Sequence, Any, Union

# Norm layouts accepted by the image backbones (see convert_norm).
NORM_MODES = ("bn", "ln_last", "ln_all")


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim of an (N, C, H, W) feature map.

    Drop-in for BatchNorm2d — same constructor argument (channel count), same
    input/output shape — but the statistics come from the channels at each
    spatial position, so they are batch- and resolution-independent. Subclasses
    nn.LayerNorm so isinstance-based norm detection (pruning groups, weight
    trackers) picks it up unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"LayerNorm2d expects (N, C, H, W), got {tuple(x.shape)}"
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def convert_norm(model: nn.Module, norm: str) -> nn.Module:
    """Replace BatchNorm2d layers with `LayerNorm2d`, in place.

    Args:
        model: any module — BatchNorm2d layers are found at any depth.
        norm: "bn" leaves the model untouched; "ln_last" converts only the last
            BatchNorm2d in registration order (`layer4[1].bn2` on ResNet18,
            `features.post_activ.bn` on WRN); "ln_all" converts every one.

    Returns the same `model` object, mutated.
    """
    assert norm in NORM_MODES, f"norm must be one of {NORM_MODES}, got {norm!r}"
    if norm == "bn":
        return model

    names = [name for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    assert names, "no BatchNorm2d layer to convert"
    for name in names if norm == "ln_all" else names[-1:]:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, LayerNorm2d(getattr(parent, attr).num_features))
    return model


class Permute(nn.Module):
    """
    A torch.nn.Module to permute a tensor's dimensions.
    """

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class Squeeze(nn.Module):
    """
    A torch.nn.Module to squeeze a tensor at a specified dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


class UnSqueeze(nn.Module):
    """
    A torch.nn.Module to unsqueeze a tensor at a specified dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


class SelectFromTuple(nn.Module):
    """
    A module to select one or more items from a sequence (e.g., a tuple).

    Args:
        *indices (int): One or more integer indices of the items to select.
    """
    def __init__(self, *indices: int):
        super().__init__()
        if not indices:
            raise ValueError("SelectFromTuple requires at least one index to be specified.")
        self.indices = indices

    def forward(self, x: Sequence[Any]) -> Union[Any, tuple[Any, ...]]:
        """
        Selects items from the input sequence.

        Args:
            x (Sequence[Any]): The input sequence (e.g., tuple or list).

        Returns:
            - A single item if one index was provided during initialization.
            - A tuple of items if multiple indices were provided.
        """
        if not isinstance(x, Sequence):
            raise TypeError(f"Input must be a sequence (e.g., tuple), but got {type(x).__name__}")

        if len(self.indices) == 1:
            return x[self.indices[0]]

        # return a new tuple if self.indices > 1.
        return tuple(x[i] for i in self.indices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(indices={self.indices})"


if __name__ == "__main__":
    for mode in NORM_MODES:
        net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
        )
        kinds = [type(m).__name__ for m in convert_norm(net, mode) if not isinstance(m, nn.Conv2d)]
        out = net(torch.randn(2, 3, 16, 16))
        print(f"{mode:8s} norms={kinds} out={tuple(out.shape)}")
