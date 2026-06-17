"""Shape records for prunable weights, read off a live torch module.

The only place that reads weight shapes off a model; used to enumerate the
prunable layers when registering one trainable per-layer log-scale each.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from pytorch_lightning import LightningModule

from src.callbacks.pruning.utils.pruning_manager import module_param_matches


@dataclass
class LayerShape:
    """Shape record for one prunable weight tensor.

    fan_in is weight.shape[1] (already in-channels-per-group for grouped
    convs), fan_out is weight.shape[0], kernel_dims is weight.shape[2:] (empty
    for Linear), n_params is weight.numel().
    """

    name: str
    fan_in: int
    fan_out: int
    kernel_dims: Tuple[int, ...]
    n_params: int


def collect_prunable_layer_shapes(
    pl_module: LightningModule,
    config: Dict[str, Any],
    resolved_types: tuple,
) -> List[LayerShape]:
    """LayerShape for every weight matching a prunable group config.

    Uses the same predicate as PruningManager so the collected layer set is
    exactly the regularized set.

    Inputs:
        pl_module: the model.
        config: a group config (layer_types/param_names/module_name_patterns).
        resolved_types: config["layer_types"] resolved to classes (the caller
            resolves once via ``_resolve_layer_types``).
    Output:
        One LayerShape per matching weight, named by its owning module.
    """
    shapes: List[LayerShape] = []
    for mod_name, mod in pl_module.named_modules():
        for p_name, p_obj in mod.named_parameters(recurse=False):
            if not p_obj.requires_grad:
                continue
            if not module_param_matches(
                mod_name, mod, p_name, config, resolved_types
            ):
                continue
            assert p_obj.ndim >= 2, (
                f"a trainable per-layer scale needs a weight with "
                f"fan_in/fan_out, got shape {tuple(p_obj.shape)} for "
                f"{mod_name}.{p_name}"
            )
            shapes.append(
                LayerShape(
                    name=mod_name,
                    fan_in=p_obj.shape[1],
                    fan_out=p_obj.shape[0],
                    kernel_dims=tuple(p_obj.shape[2:]),
                    n_params=p_obj.numel(),
                )
            )
    return shapes


if __name__ == "__main__":
    import torch.nn as nn

    from src.callbacks.pruning.utils.pruning_manager import (
        _resolve_layer_types,
    )

    model = nn.Sequential(nn.Conv1d(8, 16, 3), nn.Flatten(), nn.Linear(16, 4))
    cfg = {
        "param_names": ["weight"],
        "layer_types": ["torch.nn.Conv1d", "torch.nn.Linear"],
    }
    types = _resolve_layer_types(cfg["layer_types"])
    for layer_shape in collect_prunable_layer_shapes(model, cfg, types):
        print(layer_shape)
