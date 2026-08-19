"""Unit tests for prune_first_layer across the three pruning selectors.

Every published baseline in this repo sparsifies the stem, so the flag is
``true`` everywhere; a recipe that turns it off must still hold back the *same*
tensor in all three stacks. Covers:
- DST, magnitude pruning and STR agree on the target list, flag on and off.
- Off drops exactly the stem weight, and nothing else — the head is pruned.
- `stem_weight` reads the module walk, not a layer-type list or a filtered
  target list, so an encoder whose first layer owns a `weight` Parameter is
  still the stem, and one below `min_param_elements` costs only itself.
"""
import pytest
import torch
import torch.nn as nn

from src.callbacks.pruning.dst_pruner import DynamicSparsePruner
from src.callbacks.pruning.parameter_manager import (
    ParameterManager,
    stem_weight,
)
from src.callbacks.pruning.prune import MagnitudePruner
from src.callbacks.pruning.str_pruner import STRPruner


def _net():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def _qualnames(model, targets):
    lookup = {id(m): n for n, m in model.named_modules()}
    return [f"{lookup[id(m)]}.{name}" for m, name in targets]


def _dst_targets(model, flag):
    pruner = DynamicSparsePruner(amount=0.9, prune_first_layer=flag, verbose=0)
    pruner._collect(model)
    return [key for key, _, _ in pruner._targets]


def _magnitude_targets(model, flag):
    pruner = MagnitudePruner(
        amount=0.9, epochs_to_ramp=None, prune_first_layer=flag, verbose=0
    )
    return _qualnames(model, pruner.manager.collect_parameters(model, None))


def _str_targets(model, flag):
    pruner = STRPruner(prune_first_layer=flag, verbose=0)
    pruner._substitute(model)
    return [f"{name}.weight" for name, _, _ in pruner._layers]


# =============================================================================
# 1. The three ParameterManager selectors
# =============================================================================


@pytest.mark.parametrize("flag", [True, False])
def test_selectors_agree(flag):
    dst = _dst_targets(_net(), flag)
    magnitude = _magnitude_targets(_net(), flag)
    strs = _str_targets(_net(), flag)
    assert dst == magnitude == strs, (
        f"selectors disagree at prune_first_layer={flag}: "
        f"dst={dst} magnitude={magnitude} str={strs}"
    )


def test_off_drops_exactly_the_stem():
    everything = _dst_targets(_net(), True)
    trimmed = _dst_targets(_net(), False)
    assert everything == ["0.weight", "2.weight", "5.weight"]
    assert trimmed == [
        "2.weight",
        "5.weight",
    ], f"expected the stem gone and the head kept, got {trimmed}"


class BareStem(nn.Module):
    """A first layer owning a `weight` Parameter without being an nn.Conv2d."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 3, 3, 3))


def test_stem_is_not_read_off_a_layer_type_list():
    model = nn.Sequential(BareStem(), nn.Conv2d(16, 32, 3), nn.Linear(32, 10))
    assert stem_weight(model) == (model[0], "weight")


def test_stem_skips_norm_layers():
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 8, 3))
    assert stem_weight(model) == (model[1], "weight")


def test_no_weights_raises():
    with pytest.raises(ValueError, match="no trainable weight tensor"):
        stem_weight(nn.Sequential(nn.ReLU()))


def test_stem_below_min_param_elements_costs_only_itself():
    """The size filter must not promote the second layer into the stem's slot."""
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3), nn.Conv2d(8, 32, 3), nn.Linear(32, 10)
    )
    assert model[0].weight.numel() < 100
    manager = ParameterManager(prune_first_layer=False, min_param_elements=100)
    kept = manager.collect_parameters(model)
    assert [m for m, _ in kept] == [model[1], model[2]]
