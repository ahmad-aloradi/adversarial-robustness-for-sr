"""
SparsityApplier: A tool for applying structured and unstructured sparsity.
"""

import torch
from typing import Literal

class SparsityApplier:
    """
    Handles the application of structured and unstructured sparsity to a tensor.

    This class provides a generic way to apply sparsity by inferring the correct
    dimension for structured pruning based on semantic descriptions, making it
    adaptable to various layer types and model architectures.

    Args:
        pruning_type (Literal["unstructured", "structured"]):
            - "unstructured": Zeros out individual weights (element-wise).
            - "structured": Zeros out entire groups of weights along a dimension.
        sparsity_rate (float): The fraction of elements to zero out (0.0 to 1.0).
        structured_method (Literal["channel", "filter"]): The method for structured pruning.
            This is used to automatically determine the pruning dimension.
            - "channel": Prunes along dimension 0. This typically corresponds to:
                - Output channels in a Conv layer (shape: [out_channels, in_channels, ...])
                - Output neurons in a Linear layer (shape: [out_features, in_features])
            - "filter": Prunes along dimension 1. This typically corresponds to:
                - Input channels (filters) in a Conv layer.
                - Input features in a Linear layer.
    """
    def __init__(
        self,
        pruning_type: Literal["unstructured", "structured"] = "unstructured",
        sparsity_rate: float = 0.0,
        structured_method: Literal["channel", "filter"] = "channel",
    ):
        if not (0.0 <= sparsity_rate <= 1.0):
            raise ValueError("`sparsity_rate` must be between 0.0 and 1.0.")

        self.pruning_type = pruning_type
        self.sparsity_rate = sparsity_rate
        self.structured_method = structured_method

    def apply(self, param: torch.nn.Parameter) -> None:
        """
        Applies the configured sparsity to the given parameter in-place.
        """
        if not param.requires_grad or self.sparsity_rate == 0.0:
            return

        with torch.no_grad():
            if self.sparsity_rate == 1.0:
                param.data.zero_()
                return

            if self.pruning_type == "unstructured":
                self._apply_unstructured(param)
            elif self.pruning_type == "structured":
                self._apply_structured(param)

    def _apply_unstructured(self, param: torch.nn.Parameter):
        """Applies element-wise sparsity."""
        keep_prob = 1.0 - self.sparsity_rate
        mask = torch.bernoulli(torch.full_like(param.data, keep_prob))
        param.data.mul_(mask)

    def _apply_structured(self, param: torch.nn.Parameter):
        """Applies structured sparsity by automatically inferring the dimension."""
        if param.dim() < 2:
            # Silently skip structured pruning for 1D tensors (e.g., biases)
            # as it's not well-defined.
            return

        # Infer pruning dimension from the chosen method
        if self.structured_method == "channel":
            pruning_dim = 0
        elif self.structured_method == "filter":
            pruning_dim = 1
        else:
            raise ValueError(f"Unknown structured_method: '{self.structured_method}'")

        keep_prob = 1.0 - self.sparsity_rate
        dim_size = param.shape[pruning_dim]

        # 1. Create a 1D mask for the specified dimension
        mask_1d = torch.bernoulli(torch.full((dim_size,), keep_prob, device=param.device))

        # 2. Reshape the mask to broadcast it across the other dimensions
        broadcast_shape = [1] * param.dim()
        broadcast_shape[pruning_dim] = dim_size
        mask = mask_1d.view(broadcast_shape)

        # 3. Apply the mask
        param.data.mul_(mask)