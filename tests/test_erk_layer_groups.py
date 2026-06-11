"""Tests for PruningManager's auto_per_layer_erk expansion.

A group with auto_per_layer_erk becomes one optimizer param group per prunable
weight, each carrying its ERK initial sparsity_rate and an attached
erk_target_sparsity. Norm/bias stay out of the ERK set. Covers structure,
target-vs-solver agreement, disjoint coverage, fresh reg per layer, and that
applying the per-layer initial sparsity tracks each layer's ERK target.
"""
import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.erk_sparsity import (
    LayerShape,
    solve_erk_densities,
)
from src.callbacks.pruning.utils.pruning_manager import PruningManager

_REGL1 = "src.callbacks.pruning.bregman.bregman_regularizers.RegL1"
_REGNONE = "src.callbacks.pruning.bregman.bregman_regularizers.RegNone"


class TinyEncoderModule(LightningModule):
    """Conv0 (k=3), conv1 (k=5), a BN, and a Linear head."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=5)
        self.bn = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        return x


def _configs(target_sparsity=0.9, mode="erk", initial=None):
    return [
        {
            "name": "erk",
            "layer_types": ["torch.nn.Conv1d", "torch.nn.Linear"],
            "param_names": ["weight"],
            "auto_per_layer_erk": {
                "mode": mode,
                "target_sparsity": target_sparsity,
                "initial_target_sparsity": initial,
            },
            "optimizer_settings": {
                "reg": {"_target_": _REGL1, "lamda": 0.1},
                "lambda_scale": 1.0,
            },
            "pruning_config": {"pruning_type": "unstructured"},
        },
        {
            "name": "norm",
            "layer_types": ["torch.nn.BatchNorm1d"],
            "optimizer_settings": {"reg": {"_target_": _REGNONE}},
        },
        {
            "name": "fallback",
            "is_fallback": True,
            "param_names": ["weight", "bias"],
            "module_name_patterns": [".*"],
            "optimizer_settings": {},
            "pruning_config": {
                "pruning_type": "unstructured",
                "sparsity_rate": 0.0,
            },
        },
    ]


def _layer_shapes(model):
    return [
        LayerShape(
            "conv0",
            fan_in=64,
            fan_out=128,
            kernel_dims=(3,),
            n_params=64 * 128 * 3,
        ),
        LayerShape(
            "conv1",
            fan_in=128,
            fan_out=128,
            kernel_dims=(5,),
            n_params=128 * 128 * 5,
        ),
        LayerShape(
            "fc", fan_in=128, fan_out=64, kernel_dims=(), n_params=128 * 64
        ),
    ]


# =============================================================================
# 1. Structure
# =============================================================================


def test_one_group_per_prunable_weight():
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs())
    groups = manager.get_optimizer_param_groups()

    erk_groups = [g for g in groups if "erk_target_sparsity" in g]
    assert len(erk_groups) == 3  # conv0, conv1, fc
    names = [g["name"] for g in erk_groups]
    assert len(set(names)) == 3  # unique
    for g in erk_groups:
        assert len(g["params"]) == 1


def test_bn_and_bias_excluded_from_erk():
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs())
    groups = manager.get_optimizer_param_groups()

    erk_params = {
        id(p)
        for g in groups
        if "erk_target_sparsity" in g
        for p in g["params"]
    }
    assert id(model.bn.weight) not in erk_params
    assert id(model.conv0.bias) not in erk_params


# =============================================================================
# 2. Targets match the solver, and the set is disjoint + complete
# =============================================================================


def test_targets_match_solver():
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs(target_sparsity=0.9, mode="erk"))
    groups = manager.get_optimizer_param_groups()

    densities = solve_erk_densities(_layer_shapes(model), 0.9, "erk")
    name_to_target = {
        "conv0": 1 - densities["conv0"],
        "conv1": 1 - densities["conv1"],
        "fc": 1 - densities["fc"],
    }

    for g in groups:
        if "erk_target_sparsity" not in g:
            continue
        layer = g["name"].split("__")[-1]  # "erk__conv0" -> "conv0"
        assert abs(g["erk_target_sparsity"] - name_to_target[layer]) < 1e-9


def test_erk_set_is_disjoint_and_covers_all_weights():
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs())
    groups = manager.get_optimizer_param_groups()

    erk_param_ids = [
        id(p)
        for g in groups
        if "erk_target_sparsity" in g
        for p in g["params"]
    ]
    assert len(erk_param_ids) == len(set(erk_param_ids))  # no overlap

    expected = {
        id(model.conv0.weight),
        id(model.conv1.weight),
        id(model.fc.weight),
    }
    assert set(erk_param_ids) == expected


# =============================================================================
# 3. Fresh reg per layer (each gets its own lamda)
# =============================================================================


def test_fresh_reg_instance_per_layer():
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs())
    groups = manager.get_optimizer_param_groups()

    regs = [
        instantiate(g["reg"]) for g in groups if "erk_target_sparsity" in g
    ]
    assert len(regs) == 3
    assert regs[0] is not regs[1] and regs[1] is not regs[2]
    regs[0].lamda = 99.0
    assert regs[1].lamda == 0.1  # independent


# =============================================================================
# 4. Applying per-layer initial sparsity tracks each layer's ERK target
# =============================================================================


def test_initial_sparsity_tracks_erk_target():
    torch.manual_seed(0)
    model = TinyEncoderModule()
    manager = PruningManager(model, _configs(target_sparsity=0.9, mode="erk"))
    manager.apply_initial_sparsity()

    densities = solve_erk_densities(_layer_shapes(model), 0.9, "erk")
    # bernoulli mask => achieved sparsity is near the target (binomial noise);
    # large layers keep the band tight.
    for name, weight in (
        ("conv0", model.conv0.weight),
        ("conv1", model.conv1.weight),
        ("fc", model.fc.weight),
    ):
        achieved = compute_sparsity([weight], threshold=1e-12)
        assert abs(achieved - (1 - densities[name])) < 0.02


# =============================================================================
# 5. The speechbrain Classifier (raw weight Parameter) is ERK-controlled
# =============================================================================


def test_speechbrain_classifier_is_expanded():
    from speechbrain.lobes.models.ECAPA_TDNN import Classifier

    class WithClassifier(LightningModule):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(80, 256, kernel_size=5)
            self.classifier = Classifier(input_size=256, out_neurons=100)

        def forward(self, x):
            return x

    configs = _configs()
    configs[0]["layer_types"].append(
        "speechbrain.lobes.models.ECAPA_TDNN.Classifier"
    )
    model = WithClassifier()
    manager = PruningManager(model, configs)
    groups = manager.get_optimizer_param_groups()

    erk_names = {
        g["name"].split("__")[-1] for g in groups if "erk_target_sparsity" in g
    }
    assert erk_names == {"conv", "classifier"}
    erk_params = {
        id(p)
        for g in groups
        if "erk_target_sparsity" in g
        for p in g["params"]
    }
    assert id(model.classifier.weight) in erk_params


# =============================================================================
# 6. Misconfiguration fails loud
# =============================================================================


def test_unresolvable_layer_type_module_raises():
    configs = _configs()
    configs[0]["layer_types"].append("torch.nn.layernorm.LayerNorm")
    with pytest.raises(ImportError, match="torch.nn.layernorm"):
        PruningManager(TinyEncoderModule(), configs)


def test_unresolvable_layer_type_class_raises():
    configs = _configs()
    configs[1]["layer_types"] = ["torch.nn.NoSuchNorm"]
    with pytest.raises(AttributeError, match="NoSuchNorm"):
        PruningManager(TinyEncoderModule(), configs)


def test_erk_group_without_layer_types_asserts():
    configs = _configs()
    del configs[0]["layer_types"]
    with pytest.raises(AssertionError, match="layer_types"):
        PruningManager(TinyEncoderModule(), configs)
