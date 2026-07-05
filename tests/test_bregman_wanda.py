"""Tests for Wanda ordering (activation-weighted L1 Bregman groups).

A group config with ``expand_per_layer: true`` expands into one optimizer
group per prunable weight; each group's ``RegL1ColWeighted`` holds that
layer's per-input-column weights, so weight (i, j) survives the prox iff
``col_weights[j] * |v_ij| > lamda``. ``BregmanPruner`` wires forward
hooks (``ActivationStats``), arms them every ``act_update_freq`` batches,
and folds the geomean-1 normalized per-channel stds into the regs before
the optimizer step. The stats are derived state: nothing is checkpointed.

Covers: the weighted prox/sub_grad math and its shape contract, the
activation-std capture (arming, train-mode gating, hook removal, grouped
convs), the per-layer expansion, and the pruner integration incl. the
refresh cadence, the dual resync when a channel goes constant, and a
bit-exact resume.
"""

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from src.callbacks.pruning.bregman.activation_stats import (
    MAX_RATIO,
    ActivationStats,
    normalize_geomean,
)
from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import (
    RegL1,
    RegL1ColWeighted,
    get_regularizer,
)
from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.pruning_manager import PruningManager

_REGL1CW = (
    "src.callbacks.pruning.bregman.bregman_regularizers.RegL1ColWeighted"
)
_REGNONE = "src.callbacks.pruning.bregman.bregman_regularizers.RegNone"
SPARSITY_THRESHOLD = 1e-12  # |w| below this counts as zero


class _StubTrainer:
    """Minimal trainer surface for driving the pruner hooks manually."""

    def __init__(self):
        self.global_step = 0


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=5)
        self.bn = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        h = torch.relu(self.conv0(x))
        h = torch.relu(self.conv1(h))
        return self.fc(h.mean(dim=2))


def _wanda_configs(sparsity_rate=0.0):
    return [
        {
            "name": "wanda",
            "layer_types": ["torch.nn.Conv1d", "torch.nn.Linear"],
            "param_names": ["weight"],
            "expand_per_layer": True,
            "optimizer_settings": {
                "reg": {"_target_": _REGL1CW, "lamda": 0.1},
                "lambda_scale": 1.0,
            },
            "pruning_config": {
                "pruning_type": "unstructured",
                "sparsity_rate": sparsity_rate,
            },
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
    manager = PruningManager(model, configs)
    groups = manager.get_optimizer_param_groups()
    for g in groups:
        if isinstance(g.get("reg"), dict):
            g["reg"] = instantiate(g["reg"])
    opt = AdaBreg(groups, lr=1e-2)
    return manager, opt


def _make_pruner(manager, opt, model, act_update_freq=1):
    pruner = BregmanPruner(verbose=0, act_update_freq=act_update_freq)
    pruner.manager = manager
    pruner._optimizer = opt
    pruner._setup_activation_stats(opt, model)
    return pruner


def _train_step(model, opt, pruner, trainer, x):
    pruner.on_train_batch_start(trainer, model, x, 0)
    loss = model(x).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    pruner.on_before_optimizer_step(trainer, model, opt)
    opt.step()
    trainer.global_step += 1


def _wanda_groups(opt):
    return [
        g
        for g in opt.param_groups
        if isinstance(g.get("reg"), RegL1ColWeighted)
    ]


def _reg_for(opt, weight):
    for g in opt.param_groups:
        if isinstance(g.get("reg"), RegL1ColWeighted) and g["params"][0] is weight:
            return g["reg"]
    raise KeyError("no RegL1ColWeighted group owns this weight")


# =============================================================================
# 1. RegL1ColWeighted math
# =============================================================================


def test_unit_col_weights_match_regl1():
    plain = RegL1(lamda=0.3)
    weighted = RegL1ColWeighted(lamda=0.3)
    for shape in [(16, 8), (16, 8, 3)]:
        v = torch.randn(*shape)
        weighted.col_weights = torch.ones(shape[1])
        assert torch.equal(plain.prox(v, 1.0), weighted.prox(v, 1.0))
        assert torch.equal(plain.sub_grad(v), weighted.sub_grad(v))


def test_silent_column_gets_bounded_max_threshold():
    weighted = RegL1ColWeighted(lamda=0.1)
    weighted.col_weights = torch.ones(8)
    weighted.col_weights[3] = 1.0 / MAX_RATIO  # silent-channel floor
    v = torch.randn(16, 8)  # ~unit-scale weights
    out = weighted.prox(v, 1.0)
    # silent column threshold = lamda * MAX_RATIO = 10 -> unit-scale die;
    # threshold is bounded (no 1e12 blow-up in the recovered weight).
    assert (out[:, 3] == 0).all()
    assert (out[:, :3] != 0).any()  # live columns (thresh 0.1) keep weights


def test_conv_threshold_broadcasts_per_column():
    weighted = RegL1ColWeighted(lamda=0.2)
    col_weights = torch.tensor([0.5, 1.0, 2.0, 4.0])
    weighted.col_weights = col_weights
    v = torch.randn(6, 4, 3)
    expected_thresh = (0.2 / col_weights).view(1, 4, 1)
    expected = torch.sign(v) * torch.clamp(v.abs() - expected_thresh, min=0)
    assert torch.equal(weighted.prox(v, 1.0), expected)


def test_unset_or_mismatched_col_weights_raise():
    weighted = RegL1ColWeighted(lamda=0.1)
    with pytest.raises(AssertionError):
        weighted.prox(torch.randn(4, 4), 1.0)
    weighted.col_weights = torch.ones(3)
    with pytest.raises(AssertionError):
        weighted.prox(torch.randn(4, 4), 1.0)


def test_registered_in_factory():
    assert isinstance(
        get_regularizer("l1_col_weighted"), RegL1ColWeighted
    )


# =============================================================================
# 2. Activation statistics
# =============================================================================


def test_normalize_geomean_is_geomean_one():
    act_std = torch.rand(16) + 0.1
    weights = normalize_geomean(act_std)
    geo_mean = torch.exp(torch.log(weights).mean())
    assert torch.isclose(geo_mean, torch.tensor(1.0), atol=1e-5)


def test_constant_channel_maps_to_floor():
    act_std = torch.rand(8) + 0.1
    act_std[2] = 0.0
    weights = normalize_geomean(act_std)
    assert float(weights[2]) == pytest.approx(1.0 / MAX_RATIO, rel=1e-3)
    assert (weights > 0).all()


def test_all_constant_maps_to_uniform_floor():
    weights = normalize_geomean(torch.zeros(8))
    assert (weights == torch.full((8,), 1.0 / MAX_RATIO)).all()


def test_channel_std_matches_manual():
    conv1d = nn.Conv1d(4, 8, 3)
    conv2d = nn.Conv2d(4, 8, 3)
    linear = nn.Linear(4, 8)
    stats = ActivationStats(
        {"c1": conv1d, "c2": conv2d, "fc": linear}
    )
    stats.arm()
    x1 = torch.randn(2, 4, 10)
    x2 = torch.randn(2, 4, 10, 10)
    x3 = torch.randn(2, 5, 4)  # (B, T, F): flattened over leading dims
    conv1d(x1), conv2d(x2), linear(x3)
    assert torch.allclose(
        stats._raw_std["c1"], x1.std(dim=(0, 2), unbiased=False)
    )
    assert torch.allclose(
        stats._raw_std["c2"], x2.std(dim=(0, 2, 3), unbiased=False)
    )
    assert torch.allclose(
        stats._raw_std["fc"],
        x3.reshape(-1, 4).std(dim=0, unbiased=False),
    )
    stats.remove()


def test_capture_requires_arm_and_train_mode():
    linear = nn.Linear(4, 2)
    stats = ActivationStats({"fc": linear})
    linear(torch.randn(2, 4))  # not armed
    assert not stats.fresh()
    stats.arm()
    linear.eval()
    linear(torch.randn(2, 4))  # armed but eval mode
    assert not stats.fresh()
    linear.train()
    linear(torch.randn(2, 4))
    assert stats.fresh()
    stats.remove()


def test_removed_hooks_never_capture():
    linear = nn.Linear(4, 2)
    stats = ActivationStats({"fc": linear})
    stats.remove()
    stats.arm()
    linear(torch.randn(2, 4))
    assert not stats.fresh()


def test_grouped_conv_rejected():
    with pytest.raises(AssertionError):
        ActivationStats({"g": nn.Conv1d(8, 16, 3, groups=2)})


# =============================================================================
# 3. Per-layer group expansion
# =============================================================================


def test_expansion_yields_one_group_per_weight():
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs())
    wanda = _wanda_groups(opt)
    names = sorted(g["name"] for g in wanda)
    assert names == ["wanda__conv0", "wanda__conv1", "wanda__fc"]
    for g in wanda:
        assert len(g["params"]) == 1


def test_expanded_groups_inherit_sparsity_rate():
    # SparsityApplier masks each element via an independent Bernoulli draw
    # (not top-k), so the achieved sparsity is a random variable around the
    # target; seed for a reproducible draw and size the tolerance to the
    # smallest group (fc, 128 elements: std = sqrt(0.5*0.5/128) ~= 0.044).
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs(sparsity_rate=0.5))
    manager.apply_initial_sparsity()
    for g in _wanda_groups(opt):
        sparsity = compute_sparsity(
            g["params"], threshold=SPARSITY_THRESHOLD
        )
        assert abs(sparsity - 0.5) < 0.15, g["name"]


def test_reg_instances_are_distinct():
    model = TinyModel()
    _, opt = _build(model, _wanda_configs())
    regs = [g["reg"] for g in _wanda_groups(opt)]
    assert len({id(r) for r in regs}) == len(regs) == 3


def test_expansion_rejects_non_weight_params():
    model = TinyModel()
    configs = _wanda_configs()
    configs[0]["param_names"] = ["weight", "bias"]
    with pytest.raises(AssertionError):
        PruningManager(model, configs)


# =============================================================================
# 4. Pruner integration
# =============================================================================


def test_setup_initializes_ones_col_weights():
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs())
    _make_pruner(manager, opt, model)
    for g in _wanda_groups(opt):
        weight = g["params"][0]
        assert torch.equal(
            g["reg"].col_weights, torch.ones(weight.shape[1])
        )


def test_first_batch_folds_real_stats_before_step():
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs())
    pruner = _make_pruner(manager, opt, model)
    trainer = _StubTrainer()
    x = torch.randn(4, 8, 16)
    pruner.on_train_batch_start(trainer, model, x, 0)
    model(x).pow(2).mean().backward()
    pruner.on_before_optimizer_step(trainer, model, opt)
    for g in _wanda_groups(opt):
        assert not torch.equal(
            g["reg"].col_weights, torch.ones_like(g["reg"].col_weights)
        ), g["name"]


def test_refresh_cadence_follows_act_update_freq():
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs())
    pruner = _make_pruner(manager, opt, model, act_update_freq=3)
    trainer = _StubTrainer()
    fc_reg = next(
        g["reg"] for g in _wanda_groups(opt) if g["name"] == "wanda__fc"
    )
    changed = []
    for step in range(6):
        before = fc_reg.col_weights.clone()
        # shifting input scale so a refresh visibly changes the stats
        x = (1.0 + step) * torch.randn(4, 8, 16)
        _train_step(model, opt, pruner, trainer, x)
        changed.append(not torch.equal(before, fc_reg.col_weights))
    # refresh on steps 0 (pending), 3 (freq); holds in between
    assert changed == [True, False, False, True, False, False]


def test_channel_going_constant_gets_floor_without_release():
    """A channel that goes constant gets the strongest bounded threshold
    (col_weight at the 1/MAX_RATIO floor). The resync rebuilds the dual under
    the new col_weights, so the current weights are preserved on that refresh
    (Bregman invariant) instead of being released as a huge/zero jump."""
    torch.manual_seed(0)
    model = TinyModel()
    manager, opt = _build(model, _wanda_configs())
    pruner = _make_pruner(manager, opt, model, act_update_freq=1)
    trainer = _StubTrainer()
    conv0_reg = _reg_for(opt, model.conv0.weight)
    for step in range(3):
        _train_step(model, opt, pruner, trainer, torch.randn(4, 8, 16))
    assert (model.conv0.weight[:, 5, :].abs() > SPARSITY_THRESHOLD).any()

    x = torch.randn(4, 8, 16)
    x[:, 5] = 0.7  # channel 5 goes constant mid-training
    _train_step(model, opt, pruner, trainer, x)

    # dead channel -> floor col_weight -> strongest (bounded) threshold
    assert conv0_reg.col_weights[5] == pytest.approx(1.0 / MAX_RATIO)
    assert conv0_reg.col_weights[5] <= conv0_reg.col_weights.min() + 1e-12
    # resync preserved the column rather than releasing it (the fixed bug)
    assert (model.conv0.weight[:, 5, :].abs() > SPARSITY_THRESHOLD).any()


def test_resume_is_bit_exact_at_refresh_boundary():
    def run(n_steps, resume_at=None):
        torch.manual_seed(0)
        model = TinyModel()
        manager, opt = _build(model, _wanda_configs())
        pruner = _make_pruner(manager, opt, model, act_update_freq=1)
        trainer = _StubTrainer()
        data_gen = torch.Generator().manual_seed(1)
        for step in range(n_steps):
            if step == resume_at:
                model_state = model.state_dict()
                opt_state = opt.state_dict()
                torch.manual_seed(0)
                model = TinyModel()
                manager, opt = _build(model, _wanda_configs())
                model.load_state_dict(model_state, strict=True)
                opt.load_state_dict(opt_state)
                pruner = _make_pruner(
                    manager, opt, model, act_update_freq=1
                )
                trainer = _StubTrainer()
                trainer.global_step = step
            x = torch.randn(4, 8, 16, generator=data_gen)
            _train_step(model, opt, pruner, trainer, x)
        return model

    reference = run(6)
    resumed = run(6, resume_at=3)
    for (name, p_ref), (_, p_res) in zip(
        reference.named_parameters(), resumed.named_parameters()
    ):
        assert torch.equal(p_ref, p_res), name
