"""Resume contract for the per-layer ERK schedulers.

Run N steps, checkpoint, reload into a fresh pruner+optimizer, run M more, and
assert each layer's lambda matches a never-interrupted reference. Also checks
the namespaced checkpoint key and that a layer-set mismatch fails loud.
"""
import functools
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.utils.pruning_manager import PruningManager

_REGL1 = "src.callbacks.pruning.bregman.bregman_regularizers.RegL1"


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv1 = nn.Conv1d(64, 64, kernel_size=5)
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        return x


def _configs(target=0.8, initial=0.95, mode="erk"):
    return [
        {
            "name": "erk",
            "layer_types": ["torch.nn.Conv1d", "torch.nn.Linear"],
            "param_names": ["weight"],
            "auto_per_layer_erk": {
                "mode": mode,
                "target_sparsity": target,
                "initial_target_sparsity": initial,
            },
            "optimizer_settings": {
                "reg": {"_target_": _REGL1, "lamda": 1.0},
                "lambda_scale": 1.0,
            },
            "pruning_config": {"pruning_type": "unstructured"},
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


def _setup(model, target=0.8):
    manager = PruningManager(model, _configs(target=target))
    groups = manager.get_optimizer_param_groups()
    for g in groups:
        if isinstance(g.get("reg"), dict):
            g["reg"] = instantiate(g["reg"])
    opt = AdaBreg(groups, lr=1e-2)
    return manager, opt


def _make_pruner(opt, manager, target=0.8):
    # Template partial: per-layer clones override only target_sparsity.
    template = functools.partial(
        LambdaScheduler,
        initial_lambda=1.0,
        update_frequency=1,
        damping_zone=0.0,  # no latch/damping => clean determinism
        max_relative_change=None,
        acceleration_factor=0.5,
    )
    pruner = BregmanPruner(
        lambda_scheduler=template, target_sparsity=target, verbose=0
    )
    pruner._erk_mode = True
    pruner._optimizer = opt
    pruner.manager = manager
    return pruner


def _run(pruner, opt, n_steps, start_step):
    for i in range(n_steps):
        trainer = SimpleNamespace(optimizers=[opt], global_step=start_step + i)
        pruner._step_layer_schedulers(trainer)
    return start_step + n_steps


def _lambdas(pruner):
    return {
        name: s.get_lambda() for name, s in pruner._layer_schedulers.items()
    }


def test_resume_matches_uninterrupted_reference():
    torch.manual_seed(0)
    model = TinyModel()
    manager_ref, opt_ref = _setup(model)
    manager_ref.apply_initial_sparsity()  # freeze a sparse model (weights fixed)

    # Reference: 30 uninterrupted steps.
    ref = _make_pruner(opt_ref, manager_ref)
    ref._setup_layer_schedulers(opt_ref, is_resuming=False)
    for s in ref._layer_schedulers.values():
        s.resolve_warmup_steps(10)
    _run(ref, opt_ref, 30, 0)
    ref_lambdas = _lambdas(ref)

    # Interrupted A: 10 steps, then checkpoint.
    manager_a, opt_a = _setup(model)
    pruner_a = _make_pruner(opt_a, manager_a)
    pruner_a._setup_layer_schedulers(opt_a, is_resuming=False)
    for s in pruner_a._layer_schedulers.values():
        s.resolve_warmup_steps(10)
    _run(pruner_a, opt_a, 10, 0)
    ckpt = {}
    pruner_a.on_save_checkpoint(None, None, ckpt)
    assert "bregman_erk_layer_scheduler_states" in ckpt
    assert set(ckpt["bregman_erk_layer_scheduler_states"]) == set(ref_lambdas)

    # Interrupted B: reload, run the remaining 20 steps.
    manager_b, opt_b = _setup(model)
    pruner_b = _make_pruner(opt_b, manager_b)
    pruner_b.on_load_checkpoint(None, None, ckpt)
    pruner_b._setup_layer_schedulers(opt_b, is_resuming=True)
    _run(pruner_b, opt_b, 20, 10)
    b_lambdas = _lambdas(pruner_b)

    assert set(b_lambdas) == set(ref_lambdas)
    for name in ref_lambdas:
        assert abs(b_lambdas[name] - ref_lambdas[name]) < 1e-9


def test_resume_without_erk_states_raises():
    # A checkpoint from a non-ERK (or pre-ERK) run must not silently restart
    # every per-layer controller from the template defaults.
    torch.manual_seed(3)
    model = TinyModel()
    manager, opt = _setup(model)

    pruner = _make_pruner(opt, manager)
    pruner.on_load_checkpoint(None, None, {})  # no ERK key in the checkpoint
    with pytest.raises(ValueError, match="bregman_erk_layer_scheduler_states"):
        pruner._setup_layer_schedulers(opt, is_resuming=True)


def test_erk_group_without_active_regularizer_raises():
    # Every ERK group needs reg with lambda_scale > 0, or its scheduler would
    # have no lambda to drive.
    torch.manual_seed(4)
    model = TinyModel()
    manager, opt = _setup(model)
    erk_group = next(g for g in opt.param_groups if "erk_target_sparsity" in g)
    erk_group["lambda_scale"] = 0.0

    pruner = _make_pruner(opt, manager)
    with pytest.raises(ValueError, match="no active regularizer"):
        pruner._setup_layer_schedulers(opt, is_resuming=False)


def test_group_missing_lambda_scale_raises():
    torch.manual_seed(5)
    model = TinyModel()
    manager, opt = _setup(model)
    erk_group = next(g for g in opt.param_groups if "erk_target_sparsity" in g)
    del erk_group["lambda_scale"]

    pruner = _make_pruner(opt, manager)
    with pytest.raises(KeyError, match="lambda_scale"):
        pruner._setup_layer_schedulers(opt, is_resuming=False)


def test_resume_layer_set_mismatch_raises():
    torch.manual_seed(1)
    model = TinyModel()
    manager, opt = _setup(model)
    manager.apply_initial_sparsity()

    pruner = _make_pruner(opt, manager)
    pruner._ckpt_layer_states = {"erk__does_not_exist": {"lambda_value": 1.0}}
    with pytest.raises(ValueError, match="layer-set mismatch"):
        pruner._setup_layer_schedulers(opt, is_resuming=True)


def test_regularized_group_without_erk_target_raises():
    # A RegL1 group lacking erk_target_sparsity would freeze in ERK mode.
    torch.manual_seed(2)
    model = TinyModel()
    manager, opt = _setup(model)
    # Inject a fake regularized, non-ERK group.
    rogue_reg = instantiate({"_target_": _REGL1, "lamda": 1.0})
    opt.param_groups.append(
        {"name": "rogue", "params": [], "reg": rogue_reg, "lambda_scale": 1.0}
    )
    pruner = _make_pruner(opt, manager)
    with pytest.raises(ValueError, match="erk_target_sparsity"):
        pruner._setup_layer_schedulers(opt, is_resuming=False)
