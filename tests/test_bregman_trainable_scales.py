"""Tests for trainable per-layer Bregman allocation (water-filling).

A ``trainable_scales`` group expands into one optimizer group per prunable
weight (each carrying a trainable_scale marker, no per-layer target, so the
pruner stays in global single-scheduler mode) plus one bregman_log_scales
RegNone group holding the scalar log-scale nn.Parameters s_g. The scales are
trained by the same optimizer; the only non-stock piece is the water-filling
gradient the pruner injects: ρ_g = −S_g/κ_g, centered over live layers. The
effective scale is ``lambda_scale = c_g = exp(s_g)``; the gauge Σ_live s_g = 0
fixes the level/allocation split; there is no floor.

Covers: expansion structure, neutral init (s = 0), the injected gradient's
closed form, the dead-layer freeze (no grad, excluded from the mean, no
divergence), the gauge re-centering + numeric clamp, the "RegNone group ==
plain Adam" identity the no-new-machinery claim rests on, checkpoint round-trip
of scales + moments, and the fail-loud rename guard.
"""

import math

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import (
    SCALE_BAND,
    SCALE_CLAMP,
    BregmanPruner,
)
from src.callbacks.pruning.bregman.bregman_regularizers import RegNone
from src.callbacks.pruning.utils.pruning_manager import (
    SCALES_ATTR,
    PruningManager,
    create_scale_params,
)

_REGL1 = "src.callbacks.pruning.bregman.bregman_regularizers.RegL1"
_REGNONE = "src.callbacks.pruning.bregman.bregman_regularizers.RegNone"
# Global sparsity level the per-layer scales allocate (the scheduler's job).
_LAM_GLOBAL = 0.3


class _StubScheduler:
    """Minimal global-lambda source for tests that bypass on_fit_start."""

    def __init__(self, lam):
        self._lam = lam

    def get_lambda(self):
        return self._lam

    def step(self, current_sparsity, target_sparsity=None, current_step=None):
        return self._lam


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
    create_scale_params(model, configs)
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
    pruner.lambda_scheduler = _StubScheduler(_LAM_GLOBAL)
    pruner._setup_trainable_scales(opt, model)
    return pruner


def _trainable_groups(opt):
    return [g for g in opt.param_groups if g.get("trainable_scale")]


def _band(group):
    """SCALE_BAND·t_g at s = 0 (c = 1): the kill-rate band edge."""
    return SCALE_BAND * group["delta"] * _LAM_GLOBAL * 1.0


# =============================================================================
# 1. Expansion structure
# =============================================================================


def test_trainable_groups_carry_marker():
    model = TinyModel()
    create_scale_params(model, _trainable_configs())
    manager = PruningManager(model, _trainable_configs())
    groups = manager.get_optimizer_param_groups()

    weight_groups = [g for g in groups if g.get("trainable_scale")]
    assert len(weight_groups) == 3  # conv0, conv1, fc
    for g in weight_groups:
        assert "erk_target_sparsity" not in g
        assert "trainable_scale_key" in g
        assert "scale_decay" not in g  # the anchor prior is gone
        assert "scale_min" not in g  # no floor in log-space


def test_one_scales_group_regnone_with_lr():
    model = TinyModel()
    create_scale_params(model, _trainable_configs(scale_lr=7e-4))
    manager = PruningManager(model, _trainable_configs(scale_lr=7e-4))
    groups = manager.get_optimizer_param_groups()

    scale_groups = [g for g in groups if g["name"] == SCALES_ATTR]
    assert len(scale_groups) == 1
    sg = scale_groups[0]
    assert sg["lambda_scale"] == 0.0
    assert sg["lr"] == 7e-4
    assert sg["reg"]["_target_"].endswith("RegNone")
    # One log-scale param per prunable weight.
    assert len(sg["params"]) == 3


def test_scale_keys_round_trip_to_layer_names():
    model = TinyModel()
    create_scale_params(model, _trainable_configs())
    manager = PruningManager(model, _trainable_configs())
    groups = manager.get_optimizer_param_groups()

    keys = {
        g["trainable_scale_key"] for g in groups if g.get("trainable_scale")
    }
    assert keys == set(getattr(model, SCALES_ATTR).keys())
    assert keys == {"conv0", "conv1", "fc"}  # no dots here, so unchanged


def test_scales_seed_zero():
    """Every log-scale starts neutral: s = 0 (c = exp(0) = 1, λ_eff = λ_global)."""
    model = TinyModel()
    create_scale_params(model, _trainable_configs())
    sd = getattr(model, SCALES_ATTR)
    for v in sd.values():
        assert v.dim() == 0  # one scalar per layer
        assert v.item() == 0.0


def test_two_trainable_scale_groups_raise():
    model = TinyModel()
    configs = _trainable_configs()
    configs.insert(1, dict(configs[0], name="ts2"))
    with pytest.raises(ValueError, match="At most one trainable_scales"):
        create_scale_params(model, configs)


# =============================================================================
# 2. The injected hypergradient matches the water-filling closed form
# =============================================================================


def test_injected_gradient_matches_waterfilling():
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, SCALES_ATTR)

    # Give each layer four in-band survivors (counted in κ), two out-of-band
    # survivors (live but not in κ), and a per-layer grad scale so ρ differs
    # and the live-mean centering is non-trivial.
    base_grad = torch.tensor([0.2, -0.3, 0.4, -0.1, 0.7, -0.5])
    trainable = _trainable_groups(opt)
    expected_rho = {}
    for idx, group in enumerate(trainable):
        key = group["trainable_scale_key"]
        band = (
            SCALE_BAND
            * group["delta"]
            * _LAM_GLOBAL
            * math.exp(float(sd[key]))
        )
        p = group["params"][0]
        flat = p.data.view(-1)
        flat.zero_()
        flat[0], flat[1], flat[2], flat[3] = (
            0.5 * band,
            0.5 * band,
            0.9 * band,
            0.1 * band,
        )
        flat[4], flat[5] = 5.0 * band, 3.0 * band  # survivors, out of band
        p.grad = torch.zeros_like(p)
        p.grad.view(-1)[:6] = base_grad * (idx + 1)

        signal = (p.grad * torch.sign(p.data)).sum()
        n_band = ((p.data.abs() > 0) & (p.data.abs() <= band)).sum()
        kappa = n_band / band
        expected_rho[key] = float(-signal / kappa)

    rho_bar = sum(expected_rho.values()) / len(expected_rho)

    pruner.on_before_optimizer_step(None, model, opt)

    for group in trainable:
        key = group["trainable_scale_key"]
        assert sd[key].grad is not None
        assert sd[key].grad.item() == pytest.approx(
            expected_rho[key] - rho_bar, abs=1e-6
        )


# =============================================================================
# 3. Dead-layer freeze (the mandatory fix)
# =============================================================================


def test_dead_layer_frozen_and_excluded_from_mean():
    torch.manual_seed(1)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, SCALES_ATTR)
    trainable = _trainable_groups(opt)

    for idx, group in enumerate(trainable):
        p = group["params"][0]
        if idx == 0:
            p.data = torch.zeros_like(p)  # fully dead, no survivors in band
            p.grad = torch.randn_like(p)  # grads on dead weights are ignored
        else:
            p.data = torch.full_like(p, 0.5 * _band(group))  # all in band
            p.grad = torch.randn_like(p)

    pruner.on_before_optimizer_step(None, model, opt)

    dead_key = trainable[0]["trainable_scale_key"]
    assert sd[dead_key].grad is None  # frozen
    assert dead_key not in pruner._live_scale_names
    for group in trainable[1:]:
        assert sd[group["trainable_scale_key"]].grad is not None
    assert len(pruner._live_scale_names) == len(trainable) - 1


def test_dead_layer_does_not_diverge_over_steps():
    torch.manual_seed(2)
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs(scale_lr=0.1))
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, SCALES_ATTR)
    trainable = _trainable_groups(opt)
    dead_key = trainable[0]["trainable_scale_key"]
    s_dead_start = float(sd[dead_key])

    for _ in range(50):
        for idx, group in enumerate(trainable):
            key = group["trainable_scale_key"]
            p = group["params"][0]
            band = (
                SCALE_BAND
                * group["delta"]
                * _LAM_GLOBAL
                * math.exp(float(sd[key]))
            )
            if idx == 0:
                p.data = torch.zeros_like(p)  # permanently dead
                p.grad = torch.randn_like(p)
            else:
                p.data = torch.full_like(p, 0.5 * band)
                p.grad = torch.randn_like(p)
        pruner.on_before_optimizer_step(None, model, opt)
        opt.step()
        pruner._sync_trainable_scales(opt)

    # No grad ever, excluded from the live mean, so the dead s stays put — the
    # geomean S10.1 divergence does not occur.
    assert sd[dead_key].grad is None
    assert float(sd[dead_key]) == pytest.approx(s_dead_start, abs=1e-6)


# =============================================================================
# 4. Gauge re-centering, numeric clamp, lambda_scale = exp(s)
# =============================================================================


def test_sync_recenters_live_and_sets_exp_lambda_scale():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, SCALES_ATTR)
    trainable = _trainable_groups(opt)

    values = [0.3, 0.0, -0.6]
    for group, v in zip(trainable, values):
        sd[group["trainable_scale_key"]].data.fill_(v)

    pruner._sync_trainable_scales(opt)

    centered = [v - sum(values) / len(values) for v in values]
    s_sum = sum(float(sd[g["trainable_scale_key"]]) for g in trainable)
    assert s_sum == pytest.approx(0.0, abs=1e-6)  # gauge: Σ_live s = 0
    for group, cs in zip(trainable, centered):
        key = group["trainable_scale_key"]
        assert float(sd[key]) == pytest.approx(cs, abs=1e-6)
        assert group["lambda_scale"] == pytest.approx(math.exp(cs), abs=1e-6)


def test_sync_clamps_log_scale_to_numeric_range():
    model = TinyModel()
    manager, opt = _build(model, _trainable_configs())
    pruner = _make_pruner(manager, opt, model)
    sd = getattr(model, SCALES_ATTR)
    keys = [g["trainable_scale_key"] for g in _trainable_groups(opt)]

    sd[keys[0]].data.fill_(1000.0)  # far past the clamp, even after centering
    sd[keys[1]].data.fill_(0.0)
    sd[keys[2]].data.fill_(0.0)
    pruner._sync_trainable_scales(opt)

    for key in keys:
        assert abs(float(sd[key])) <= SCALE_CLAMP + 1e-6


# =============================================================================
# 5. No new machinery: a RegNone group's step is a plain Adam step
# =============================================================================


def test_regnone_group_equals_adam_step():
    s = nn.Parameter(torch.tensor(0.3))
    s_ref = nn.Parameter(torch.tensor(0.3))
    opt = AdaBreg(
        [
            {
                "name": SCALES_ATTR,
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
# 6. Checkpoint round-trip + fail-loud rename guard
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

    sd1 = getattr(model, SCALES_ATTR)
    sd2 = getattr(model2, SCALES_ATTR)
    for key in sd1:
        assert torch.equal(sd1[key].data, sd2[key].data)

    # The scale params' Adam moments live in the optimizer state.
    for p1, p2 in zip(
        next(g for g in opt.param_groups if g["name"] == SCALES_ATTR)[
            "params"
        ],
        next(g for g in opt2.param_groups if g["name"] == SCALES_ATTR)[
            "params"
        ],
    ):
        st1, st2 = opt.state[p1], opt2.state[p2]
        assert torch.equal(st1["exp_avg"], st2["exp_avg"])
        assert torch.equal(st1["exp_avg_sq"], st2["exp_avg_sq"])


def test_loading_no_scales_checkpoint_into_trainable_raises():
    model = TinyModel()
    create_scale_params(model, _trainable_configs())
    full = model.state_dict()
    stripped = {k: v for k, v in full.items() if not k.startswith(SCALES_ATTR)}
    with pytest.raises(RuntimeError, match=SCALES_ATTR):
        model.load_state_dict(stripped, strict=True)


def test_pre_log_scale_checkpoint_fails_loud():
    """A c-valued checkpoint keyed 'bregman_scales.*' must not silently load
    into the log-scale 'bregman_log_scales.*' state — the key mismatch raises.
    """
    model = TinyModel()
    create_scale_params(model, _trainable_configs())
    current = model.state_dict()
    old_keyed = {}
    for key, value in current.items():
        if key.startswith(SCALES_ATTR):
            old_keyed["bregman_scales" + key[len(SCALES_ATTR) :]] = value
        else:
            old_keyed[key] = value
    with pytest.raises(RuntimeError):
        model.load_state_dict(old_keyed, strict=True)
