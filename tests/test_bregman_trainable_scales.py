"""Tests for trainable per-layer Bregman lambda scales.

A ``trainable_scales`` group expands into one optimizer group per prunable
weight (each carrying a trainable_scale marker, no per-layer target, so the
pruner stays in global single-scheduler mode) plus one bregman_log_scales
RegNone group holding the scalar log-scale nn.Parameters. The scales are trained
by the same optimizer; the only non-stock piece is the closed-form hypergradient
the pruner injects.

Covers: expansion structure, neutral init, the injected gradient's closed form,
the "RegNone group == plain Adam" identity that the no-new-machinery claim rests
on, the clamp/sync/decay dynamics, and checkpoint round-trip of scales + moments.
"""
import math

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegNone
from src.callbacks.pruning.utils.pruning_manager import (
    LOG_SCALES_ATTR,
    PruningManager,
    create_log_scale_params,
)

_REGL1 = "src.callbacks.pruning.bregman.bregman_regularizers.RegL1"
_REGNONE = "src.callbacks.pruning.bregman.bregman_regularizers.RegNone"


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=5)
        self.bn = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return x


def _trainable_configs(initial_sparsity=0.0, scale_lr=1e-3):
    return [
        {
            "name": "ts",
            "layer_types": ["torch.nn.Conv1d", "torch.nn.Linear"],
            "param_names": ["weight"],
            "trainable_scales": {
                "scale_lr": scale_lr,
                "scale_decay": 1e-4,
                "scale_clamp": [0.25, 4.0],
                "initial_sparsity": initial_sparsity,
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


def _build(model, configs):
    """Register scales, build the manager + an AdaBreg over its groups."""
    create_log_scale_params(model, configs)
    manager = PruningManager(model, configs)
    groups = manager.get_optimizer_param_groups()
    for g in groups:
        if isinstance(g.get("reg"), dict):
            g["reg"] = instantiate(g["reg"])
    opt = AdaBreg(groups, lr=1e-2)
    return manager, opt


def _make_pruner(manager, opt, model):
    pruner = BregmanPruner(verbose=0)
    pruner.manager = manager
    pruner._optimizer = opt
    pruner._setup_trainable_scales(opt, model)
    return pruner


# =============================================================================
# 1. Expansion structure
# =============================================================================


def test_trainable_groups_carry_marker():
    model = TinyModel()
    create_log_scale_params(model, _trainable_configs())
    manager = PruningManager(model, _trainable_configs())
    groups = manager.get_optimizer_param_groups()

    weight_groups = [g for g in groups if g.get("trainable_scale")]
    assert len(weight_groups) == 3  # conv0, conv1, fc
    for g in weight_groups:
        assert "erk_target_sparsity" not in g
        assert "trainable_scale_key" in g
        assert g["scale_decay"] == 1e-4
        assert list(g["scale_clamp"]) == [0.25, 4.0]


def test_one_log_scales_group_regnone_with_lr():
    model = TinyModel()
    create_log_scale_params(model, _trainable_configs(scale_lr=7e-4))
    manager = PruningManager(model, _trainable_configs(scale_lr=7e-4))
    groups = manager.get_optimizer_param_groups()

    scale_groups = [g for g in groups if g["name"] == LOG_SCALES_ATTR]
    assert len(scale_groups) == 1
    sg = scale_groups[0]
    assert sg["lambda_scale"] == 0.0
    assert sg["lr"] == 7e-4
    assert sg["reg"]["_target_"].endswith("RegNone")
    # One scale param per prunable weight.
    assert len(sg["params"]) == 3


def test_scale_keys_round_trip_to_layer_names():
    model = TinyModel()
    create_log_scale_params(model, _trainable_configs())
    manager = PruningManager(model, _trainable_configs())
    groups = manager.get_optimizer_param_groups()

    keys = {
        g["trainable_scale_key"] for g in groups if g.get("trainable_scale")
    }
    assert keys == set(getattr(model, LOG_SCALES_ATTR).keys())
    assert keys == {"conv0", "conv1", "fc"}  # no dots here, so unchanged


def test_scales_seed_zero():
    """Every scale starts neutral: s = 0 (e^{s} = 1)."""
    model = TinyModel()
    create_log_scale_params(model, _trainable_configs())
    sd = getattr(model, LOG_SCALES_ATTR)
    for v in sd.values():
        assert v.dim() == 0  # one scalar per layer
        assert v.item() == 0.0


def test_two_trainable_scale_groups_raise():
    model = TinyModel()
    configs = _trainable_configs()
    configs.insert(1, dict(configs[0], name="ts2"))
    with pytest.raises(ValueError, match="At most one trainable_scales"):
        create_log_scale_params(model, configs)


# =============================================================================
# 2. The injected hypergradient matches its closed form
# =============================================================================


def test_injected_gradient_matches_closed_form():
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)

    decay = pruner._scale_decay
    sd = getattr(model, LOG_SCALES_ATTR)
    # Give each scale a non-zero value (exercises the decay term) and each
    # weight a known grad with some exactly-zero entries (must drop out).
    expected = {}
    for group in opt.param_groups:
        if not group.get("trainable_scale"):
            continue
        key = group["trainable_scale_key"]
        sd[key].data.fill_(0.5)
        group["reg"].lamda = 0.5
        p = group["params"][0]
        p.data = torch.randn_like(p)
        p.data.view(-1)[:3] = 0.0  # dead weights
        p.grad = torch.randn_like(p)
        live = p.data != 0
        signal = (p.grad[live] * torch.sign(p.data[live])).sum()
        expected[key] = (
            -group["delta"] * group["reg"].lamda * signal + decay * 0.5
        )

    pruner.on_before_optimizer_step(None, model, opt)

    for group in opt.param_groups:
        if not group.get("trainable_scale"):
            continue
        key = group["trainable_scale_key"]
        assert sd[key].grad.item() == pytest.approx(
            expected[key].item(), abs=1e-6
        )


def test_zero_weights_contribute_nothing():
    torch.manual_seed(1)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, LOG_SCALES_ATTR)

    group = next(g for g in opt.param_groups if g.get("trainable_scale"))
    key = group["trainable_scale_key"]
    group["reg"].lamda = 1.0
    p = group["params"][0]
    p.data = torch.zeros_like(p)  # whole layer dead
    p.grad = torch.randn_like(p)  # large grads, all on dead weights

    pruner.on_before_optimizer_step(None, model, opt)
    # signal == 0 => grad is purely the decay term (s == 0 here => exactly 0).
    assert sd[key].grad.item() == pytest.approx(0.0, abs=1e-7)


# =============================================================================
# 3. No new machinery: a RegNone group's step is a plain Adam step
# =============================================================================


def test_regnone_group_equals_adam_step():
    s = nn.Parameter(torch.tensor(0.3))
    s_ref = nn.Parameter(torch.tensor(0.3))
    opt = AdaBreg(
        [
            {
                "name": LOG_SCALES_ATTR,
                "params": [s],
                "reg": RegNone(),
                "lambda_scale": 0.0,
                "lr": 1e-3,
            }
        ],
        lr=1e-3,
    )
    opt_ref = torch.optim.Adam([s_ref], lr=1e-3)

    for step in range(6):
        g = torch.tensor(0.7 - 0.05 * step)
        s.grad = g.clone()
        s_ref.grad = g.clone()
        opt.step()
        opt_ref.step()
        assert s.item() == pytest.approx(s_ref.item(), abs=1e-7)


# =============================================================================
# 4. Clamp / sync / decay dynamics
# =============================================================================


def test_sync_sets_lambda_scale_to_exp_s():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, LOG_SCALES_ATTR)

    group = next(g for g in opt.param_groups if g.get("trainable_scale"))
    sd[group["trainable_scale_key"]].data.fill_(math.log(1.7))
    pruner._sync_trainable_scales(opt)
    assert group["lambda_scale"] == pytest.approx(1.7, abs=1e-6)
    # The downstream contract the scheduler relies on: reg.lamda = λ · e^{s}.
    reg_lamda = 0.02 * group["lambda_scale"]
    assert reg_lamda == pytest.approx(0.02 * 1.7, abs=1e-9)


def test_clamp_caps_the_scale():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs(scale_lr=0.1))
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, LOG_SCALES_ATTR)

    group = next(g for g in opt.param_groups if g.get("trainable_scale"))
    key = group["trainable_scale_key"]
    group["reg"].lamda = 1.0
    p = group["params"][0]

    for _ in range(200):
        p.data = torch.ones_like(p)  # positive weights
        p.grad = torch.ones_like(p)  # positive grad => shrink-benefit => s up
        pruner.on_before_optimizer_step(None, model, opt)
        opt.step()
        pruner._sync_trainable_scales(opt)

    assert sd[key].item() <= math.log(4.0) + 1e-6
    assert group["lambda_scale"] <= 4.0 + 1e-6


def test_positive_shrink_signal_raises_scale():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs(scale_lr=0.05))
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, LOG_SCALES_ATTR)

    group = next(g for g in opt.param_groups if g.get("trainable_scale"))
    key = group["trainable_scale_key"]
    group["reg"].lamda = 1.0
    p = group["params"][0]
    start = sd[key].item()

    for _ in range(10):
        p.data = torch.ones_like(p)
        p.grad = torch.ones_like(p)
        pruner.on_before_optimizer_step(None, model, opt)
        opt.step()
        pruner._sync_trainable_scales(opt)

    assert sd[key].item() > start


def test_dead_layer_scale_decays_toward_one():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs(scale_lr=0.1))
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, LOG_SCALES_ATTR)

    group = next(g for g in opt.param_groups if g.get("trainable_scale"))
    key = group["trainable_scale_key"]
    sd[key].data.fill_(math.log(2.0))  # start at scale 2
    group["reg"].lamda = 1.0
    p = group["params"][0]
    start = sd[key].item()

    for _ in range(20):
        p.data = torch.zeros_like(p)  # dead => zero signal => pure decay
        p.grad = torch.randn_like(p)
        pruner.on_before_optimizer_step(None, model, opt)
        opt.step()
        pruner._sync_trainable_scales(opt)

    assert sd[key].item() < start  # pulled back toward 0 (scale 1)


# =============================================================================
# 5. Checkpoint round-trip: scales + Adam moments restored by standard load
# =============================================================================


def _run_steps(pruner, opt, model, n):
    for _ in range(n):
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.grad = torch.randn_like(p)
        pruner.on_before_optimizer_step(None, model, opt)
        opt.step()
        pruner._sync_trainable_scales(opt)


def test_scales_and_moments_restore_bit_exact():
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    _run_steps(pruner, opt, model, 4)

    model_sd = {k: v.clone() for k, v in model.state_dict().items()}
    opt_sd = opt.state_dict()

    torch.manual_seed(99)  # different init, must be overwritten by load
    model2 = TinyModel()
    manager2, opt2 = _build(model2, _trainable_configs())
    model2.load_state_dict(model_sd, strict=True)
    opt2.load_state_dict(opt_sd)

    sd1 = getattr(model, LOG_SCALES_ATTR)
    sd2 = getattr(model2, LOG_SCALES_ATTR)
    for key in sd1:
        assert torch.equal(sd1[key].data, sd2[key].data)

    # The scale params' Adam moments live in the optimizer state.
    for p1, p2 in zip(
        next(g for g in opt.param_groups if g["name"] == LOG_SCALES_ATTR)[
            "params"
        ],
        next(g for g in opt2.param_groups if g["name"] == LOG_SCALES_ATTR)[
            "params"
        ],
    ):
        st1, st2 = opt.state[p1], opt2.state[p2]
        assert torch.equal(st1["exp_avg"], st2["exp_avg"])
        assert torch.equal(st1["exp_avg_sq"], st2["exp_avg_sq"])


def test_loading_no_scales_checkpoint_into_trainable_raises():
    model = TinyModel()
    create_log_scale_params(model, _trainable_configs())
    full = model.state_dict()
    stripped = {
        k: v for k, v in full.items() if not k.startswith(LOG_SCALES_ATTR)
    }
    with pytest.raises(RuntimeError, match=LOG_SCALES_ATTR):
        model.load_state_dict(stripped, strict=True)
