"""Unit tests for prune_first_layer across all four selectors.

The stem conv is held dense by RigL (on CIFAR throughout, everywhere at 99 %
sparsity) and, in this repo, by every img recipe, so the comparison is only
honest if the four selectors hold back the *same* tensor. Covers:
- DST, magnitude pruning and STR agree on the target list, flag on and off.
- Off drops exactly the stem weight, and nothing else — the head is pruned.
- Bregman routes the stem into a group that neither regularizes nor sparsifies,
  and agrees with the other three on a stem its group config never matches.
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
from src.callbacks.pruning.utils.pruning_manager import PruningManager


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


# =============================================================================
# 2. The Bregman selector
# =============================================================================


_GROUPS = [
    {
        "name": "conv_layers",
        "layer_types": ["torch.nn.Conv2d"],
        "param_names": ["weight"],
        "optimizer_settings": {"lambda_scale": 1.0},
        "pruning_config": {"sparsity_rate": 0.9},
    },
    {
        "name": "linear_layers",
        "layer_types": ["torch.nn.Linear"],
        "param_names": ["weight"],
        "optimizer_settings": {"lambda_scale": 1.0},
        "pruning_config": {"sparsity_rate": 0.9},
    },
    {
        "name": "norm_params",
        "layer_types": ["torch.nn.BatchNorm2d"],
        "optimizer_settings": {"lambda_scale": 0.0},
    },
    {
        "name": "bias_params",
        "param_names": ["bias"],
        "optimizer_settings": {"lambda_scale": 0.0},
    },
    {
        "name": "fallback",
        "is_fallback": True,
        "module_name_patterns": [".*"],
        "param_names": ["weight"],
        "optimizer_settings": {"lambda_scale": 1.0},
    },
]


def _grouped(model, flag):
    manager = PruningManager(model, _GROUPS, prune_first_layer=flag)
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {
        g["config"]["name"]: [by_id[id(p)] for p in g["params"]]
        for g in manager.processed_groups
    }, manager


def test_bregman_reserves_the_stem():
    model = _net()
    groups, manager = _grouped(model, flag=False)
    assert groups["first_dense"] == ["0.weight"]
    assert groups["conv_layers"] == ["2.weight"]
    assert groups["linear_layers"] == ["5.weight"]

    manager.apply_initial_sparsity()
    assert float(model[0].weight.abs().min()) > 0, "stem was sparsified"
    # SparsityApplier draws a Bernoulli mask, so the rate scatters around 0.9.
    assert float((model[2].weight == 0).float().mean()) == pytest.approx(
        0.9, abs=0.03
    )


def test_bregman_keeps_every_parameter():
    model = _net()
    for flag in (True, False):
        groups, _ = _grouped(model, flag)
        assigned = sum(len(names) for names in groups.values())
        assert assigned == len(list(model.parameters())), (
            f"prune_first_layer={flag} lost parameters: "
            f"{assigned} of {len(list(model.parameters()))}"
        )


def test_bregman_agrees_with_the_pruners_on_a_custom_stem():
    """The cross-check `test_selectors_agree` cannot make: a stem the group
    config routes to its fallback is still the tensor Bregman holds dense."""
    model = nn.Sequential(BareStem(), nn.Conv2d(16, 32, 3), nn.Linear(32, 10))
    groups, _ = _grouped(model, flag=False)
    by_id = {id(p): n for n, p in model.named_parameters()}

    assert groups["first_dense"] == [by_id[id(model[0].weight)]]
    held = ParameterManager(prune_first_layer=False)
    held.collect_parameters(model)
    assert (model[0], "weight") not in held.prunable_params


def test_bregman_without_an_unregularized_group_raises():
    only_regularized = [g for g in _GROUPS if g["name"] != "norm_params"]
    only_regularized = [
        g for g in only_regularized if g["name"] != "bias_params"
    ]
    with pytest.raises(ValueError, match="lambda_scale: 0"):
        PruningManager(_net(), only_regularized, prune_first_layer=False)
