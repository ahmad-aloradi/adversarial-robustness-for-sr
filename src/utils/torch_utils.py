from torch import nn
from typing import Sequence, Any, Union


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