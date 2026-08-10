"""Unit tests for STRPruner (Kusupati et al., ICML 2020).

Covers:
- soft_threshold matches the upstream formula elementwise and is
  differentiable in the threshold, which is what trains it.
- Substitution replaces exactly the ParameterManager targets, keeps the
  weights, and is idempotent across a second setup.
- A module type STR cannot reparameterize raises instead of staying dense.
- sparsities() reads the thresholded weights while the stored weights are
  still dense.
- Weight decay drives the threshold up (the paper's sparsity knob).
- apply_thresholds writes sigmoid(s) into w, drops s, and the result loads
  strict=True into the original model with real zeros.
"""
import json
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.str_pruner import (
    STRConv2d,
    STRLinear,
    STRPruner,
    apply_thresholds,
    soft_threshold,
)


def _net():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def _substituted(**kwargs):
    net = _net()
    kwargs.setdefault("prune_first_layer", True)
    pruner = STRPruner(verbose=0, **kwargs)
    pruner._substitute(net)
    return net, pruner


# =============================================================================
# 1. The reparameterization
# =============================================================================


def test_soft_threshold_matches_upstream_formula():
    weight = torch.randn(64)
    threshold = torch.tensor([[-1.0]])
    expected = torch.sign(weight) * torch.relu(
        torch.abs(weight) - torch.sigmoid(threshold)
    )
    assert torch.equal(soft_threshold(weight, threshold), expected)


def test_soft_threshold_zeroes_below_the_threshold():
    weight = torch.tensor([-0.4, -0.6, 0.0, 0.4, 0.6])
    threshold = torch.zeros(1)  # sigmoid(0) = 0.5
    out = soft_threshold(weight, threshold)
    assert torch.equal(
        out == 0, torch.tensor([True, False, True, True, False])
    )


def test_threshold_receives_a_gradient():
    """The task loss must be able to push the threshold back down."""
    # -3 puts sigmoid(s)=0.047 under the init scale, so weights survive and
    # relu stays active; a saturating threshold would zero the gradient.
    layer = STRConv2d(3, 4, 3, s_init=-3.0)
    layer(torch.randn(2, 3, 8, 8)).square().mean().backward()
    assert layer.sparse_threshold.grad is not None
    assert float(layer.sparse_threshold.grad.abs()) > 0


def test_saturated_init_has_no_threshold_gradient():
    """At s_init=-3200 sigmoid is flat, so only weight decay moves s: this is
    why sparsity starts late rather than immediately."""
    layer = STRConv2d(3, 4, 3, s_init=-3200.0)
    assert float(torch.sigmoid(layer.sparse_threshold)) == 0.0
    layer(torch.randn(2, 3, 8, 8)).square().mean().backward()
    assert float(layer.sparse_threshold.grad.abs()) == 0.0


# =============================================================================
# 2. Substitution
# =============================================================================


def test_substitution_covers_the_parameter_manager_targets():
    net, pruner = _substituted()
    assert [name for name, _, _ in pruner._layers] == ["0", "2", "5"]
    assert isinstance(net[0], STRConv2d)
    assert isinstance(net[2], STRConv2d)
    assert isinstance(net[5], STRLinear)


def test_substitution_preserves_weights_and_shapes():
    net = _net()
    before = {name: p.clone() for name, p in net.state_dict().items()}
    pruner = STRPruner(verbose=0, prune_first_layer=True)
    pruner._substitute(net)
    for name, value in before.items():
        assert torch.equal(net.state_dict()[name], value), name
    net(torch.randn(2, 3, 8, 8))  # still runs


def test_setup_is_idempotent_across_stages():
    net = _net()
    pruner = STRPruner(verbose=0, prune_first_layer=True)
    trainer, module = MagicMock(spec=Trainer), net
    pruner.setup(trainer, module, stage="fit")
    first = [id(layer) for _, layer, _ in pruner._layers]
    pruner.setup(trainer, module, stage="fit")
    pruner.setup(trainer, module, stage="test")
    assert [id(layer) for _, layer, _ in pruner._layers] == first


def test_unsupported_layer_type_raises():
    """Leaving a targeted layer dense would make the sparsity figure wrong."""
    net = nn.Sequential(nn.Conv1d(4, 32, 5))
    with pytest.raises(TypeError, match="Conv2d and Linear only"):
        STRPruner(verbose=0, prune_first_layer=True)._substitute(net)


def test_bias_target_raises():
    # 128 outputs so the bias clears min_param_elements and is really targeted.
    net = nn.Sequential(nn.Conv2d(3, 128, 3))
    with pytest.raises(ValueError, match="thresholds weights only"):
        STRPruner(
            verbose=0, prune_bias=True, prune_first_layer=True
        )._substitute(net)


# =============================================================================
# 3. Measurement
# =============================================================================


def test_sparsity_is_read_from_the_thresholded_weights():
    net, pruner = _substituted(
        s_init=0.0
    )  # sigmoid(0) = 0.5, prunes nearly all
    overall, pruned = pruner.sparsities()
    assert pruned > 0.9
    # The stored weights are untouched, so the naive reading says ~0.
    assert compute_sparsity(net) < 0.01
    assert 0.0 < overall < pruned


def test_saturated_init_starts_dense():
    _, pruner = _substituted(s_init=-3200.0)
    assert pruner.sparsities()[1] == 0.0


def test_thresholds_are_reported_per_layer():
    _, pruner = _substituted(s_init=-3.0)
    thresholds = pruner.thresholds()
    assert set(thresholds) == {"0", "2", "5"}
    assert all(
        v == pytest.approx(float(torch.sigmoid(torch.tensor(-3.0))))
        for v in thresholds.values()
    )


def test_every_layer_collapsed_is_reported():
    _, pruner = _substituted(s_init=0.0)  # sigmoid(0)=0.5 exceeds every weight
    assert set(pruner._collapsed_layers()) == {n for n, _, _ in pruner._layers}


def test_one_collapsed_layer_is_reported_not_fatal():
    """A ResNet block falls back to its other path, so STR may drop a layer."""
    _, pruner = _substituted(s_init=-3200.0)
    with torch.no_grad():
        pruner._layers[0][1].sparse_threshold.fill_(0.0)  # threshold 0.5
    collapsed = pruner._collapsed_layers()
    assert set(collapsed) == {pruner._layers[0][0]}
    assert collapsed[pruner._layers[0][0]] == pytest.approx(0.5)


def test_live_layers_report_no_collapse():
    _, pruner = _substituted(s_init=-3200.0)
    assert pruner._collapsed_layers() == {}


def test_weight_decay_drives_the_threshold_up():
    """The paper's knob: decay on s walks it toward 0, so sigmoid(s) grows."""
    _, pruner = _substituted(s_init=-8.0)
    layer = pruner._layers[0][1]
    optimizer = torch.optim.SGD(
        [layer.sparse_threshold], lr=0.1, weight_decay=0.5
    )
    before = float(layer.sparse_threshold)
    for _ in range(50):
        optimizer.zero_grad()
        layer.sparse_threshold.grad = torch.zeros_like(layer.sparse_threshold)
        optimizer.step()
    assert float(layer.sparse_threshold) > before
    assert float(torch.sigmoid(layer.sparse_threshold)) > torch.sigmoid(
        torch.tensor(before)
    )


# =============================================================================
# 4. Applying thresholds
# =============================================================================


def test_apply_thresholds_drops_the_threshold_key():
    net, pruner = _substituted(s_init=0.0)
    applied = apply_thresholds(net.state_dict())

    assert not any(k.endswith(".sparse_threshold") for k in applied)
    original = _net()
    original.load_state_dict(applied, strict=True)
    assert compute_sparsity(original) == pytest.approx(
        pruner.sparsities()[0], abs=1e-6
    )


def test_apply_leaves_an_original_state_dict_untouched():
    original = _net().state_dict()
    assert apply_thresholds(original).keys() == original.keys()
    for key, value in original.items():
        assert torch.equal(apply_thresholds(original)[key], value)


def test_apply_thresholds_to_model_rewrites_the_live_model():
    net, pruner = _substituted(s_init=0.0)
    expected = pruner.sparsities()[1]
    pruner._apply_thresholds_to_model(net)

    assert isinstance(net[0], nn.Conv2d) and not isinstance(net[0], STRConv2d)
    assert isinstance(net[5], nn.Linear) and not isinstance(net[5], STRLinear)
    weights = [net[0].weight, net[2].weight, net[5].weight]
    zeros = sum(int((w == 0).sum()) for w in weights)
    total = sum(w.numel() for w in weights)
    assert zeros / total == pytest.approx(expected, abs=1e-6)
    _net().load_state_dict(net.state_dict(), strict=True)


def test_apply_thresholds_to_model_follows_a_moved_model():
    """Substitution runs before Lightning moves the model, so the layer the
    str_layer displaced has to be brought back to where the str_layer ended
    up."""
    net, pruner = _substituted(s_init=0.0)
    net.to(torch.float64)
    pruner._apply_thresholds_to_model(net)

    for index in (0, 2, 5):
        assert (
            net[index].weight.dtype == torch.float64
        ), f"net[{index}] came back at the pre-move dtype"


# =============================================================================
# 5. End to end
# =============================================================================


class _TinyClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = _net()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        return {"loss": nn.functional.cross_entropy(self(images), targets)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=5e-4)


def _fit(tmp_path, **trainer_kwargs):
    model = _TinyClassifier()
    pruner = STRPruner(s_init=-3.0, prune_first_layer=True, verbose=0)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(16, 3, 8, 8), torch.randint(0, 10, (16,))
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        callbacks=[pruner],
        logger=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        **trainer_kwargs,
    )
    trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=8))


def test_run_refuses_to_start_without_checkpointing(tmp_path):
    with pytest.raises(AssertionError, match="checkpointing must be enabled"):
        _fit(tmp_path, enable_checkpointing=False)


def test_run_refuses_to_start_on_a_versioned_last_ckpt(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "last-v1.ckpt").touch()
    with pytest.raises(AssertionError, match="versioned last"):
        _fit(tmp_path)


@pytest.mark.slow
def test_mini_fit_writes_loadable_checkpoints(tmp_path):
    model = _TinyClassifier()
    # -3 puts the threshold inside the init scale, so a 2-epoch run is already
    # partly sparse without any layer collapsing.
    pruner = STRPruner(s_init=-3.0, prune_first_layer=True, verbose=0)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 3, 8, 8), torch.randint(0, 10, (32,))
    )
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        callbacks=[pruner],
        logger=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=8))

    thresholds = json.loads((tmp_path / "str_thresholds.json").read_text())
    assert set(thresholds) == {"net.0", "net.2", "net.5"}

    checkpoints = [
        p
        for p in (tmp_path / "checkpoints").glob("*.ckpt")
        if p.name != "last.ckpt"
    ]
    assert checkpoints, "no checkpoint was written"
    for path in checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        assert not any(k.endswith(".sparse_threshold") for k in state)
        _TinyClassifier().load_state_dict(state, strict=True)
        assert (
            0.0 < compute_sparsity(_restored(state)) < 1.0
        ), f"{path.name} carries no zeros"
    assert not list(
        (tmp_path / "checkpoints").glob("*.tmp")
    ), "a temp ckpt was left behind"


@pytest.mark.slow
def test_last_ckpt_stays_resumable(tmp_path):
    """Applying the thresholds drops the s keys, but a resumed run re-
    substitutes STR layers in setup and would then fail a strict load."""
    model = _TinyClassifier()
    pruner = STRPruner(s_init=-3.0, prune_first_layer=True, verbose=0)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 3, 8, 8), torch.randint(0, 10, (32,))
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        callbacks=[pruner, pl.callbacks.ModelCheckpoint(save_last=True)],
        logger=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, loader)

    last = tmp_path / "checkpoints" / "last.ckpt"
    state = torch.load(last, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    assert any(
        k.endswith(".sparse_threshold") for k in state
    ), "last.ckpt lost its s and can no longer be resumed from"

    resumed = STRPruner(s_init=-3.0, prune_first_layer=True, verbose=0)
    Trainer(
        max_epochs=3,
        accelerator="cpu",
        callbacks=[resumed],
        logger=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    ).fit(_TinyClassifier(), loader, ckpt_path=last)


def _restored(state):
    model = _TinyClassifier()
    model.load_state_dict(state, strict=True)
    return model
