"""Unit tests for DynamicSparsePruner (RigL / SET / Static-ERK / SNIP /
GraNet).

Covers what separates dynamic sparse training from gradual magnitude pruning:
the mask hits the target at step 0 and never drifts, redistribution is
count-preserving per layer, grown weights start at 0 with zeroed momentum, the
gradient is never masked, and the model state_dict stays dense-with-zeros.
"""
import logging
import os
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torchmetrics import Accuracy

from src.callbacks.pruning.dst_pruner import DynamicSparsePruner
from src.callbacks.pruning.shared_prune_utils import compute_sparsity

TOTAL_STEPS = 1000


@pytest.fixture(autouse=True)
def _global_seed():
    os.environ["PL_GLOBAL_SEED"] = "42"
    torch.manual_seed(42)


def _net():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def _ready_pruner(net, **kwargs):
    """Collect targets and build masks without a live trainer."""
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("prune_first_layer", True)
    pruner = DynamicSparsePruner(**kwargs)
    pruner._collect(net)
    pruner._build_masks(net)
    # What on_train_start derives, for a 10-epoch run of TOTAL_STEPS steps.
    pruner._total_steps = TOTAL_STEPS
    if pruner.initial_amount is not None:
        pruner._final_update = int(
            pruner.final_prune_epoch
            * (TOTAL_STEPS / 10)
            / pruner.update_frequency
        )
    return pruner


def _target_params(pruner):
    """The prunable set in ``compute_sparsity``'s (module, name) form."""
    return [(module, name) for _, module, name in pruner._targets]


def _with_gradients(net, optimizer=None):
    """Populate .grad on every parameter with one real backward pass."""
    net.zero_grad(set_to_none=True)
    out = net(torch.randn(4, 3, 8, 8))
    out.square().mean().backward()
    return optimizer


def _sgd(net):
    return torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )


# =============================================================================
# 1. Initial mask
# =============================================================================


@pytest.mark.parametrize("amount", [0.5, 0.9, 0.99])
def test_initial_mask_hits_target(amount):
    net = _net()
    pruner = _ready_pruner(net, amount=amount)
    pruner._apply_masks(_sgd(net))
    achieved = compute_sparsity(_target_params(pruner))
    assert achieved == pytest.approx(amount, abs=0.005)


def test_erk_is_not_uniform():
    """ERK gives the parameter-heavy layer the higher sparsity."""
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, mask_init="erk")
    sparsities = {
        key: 1 - float(mask.float().mean())
        for key, mask in pruner._masks.items()
    }
    assert sparsities["2.weight"] > sparsities["5.weight"], (
        f"the 16->32 conv should be sparser than the 32->10 head, got "
        f"{sparsities}"
    )


# =============================================================================
# 2. Redistribution is count-preserving
# =============================================================================


@pytest.mark.parametrize("growth", ["gradient", "random"])
def test_active_counts_are_constant(growth):
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth=growth)
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    before = {k: int(m.sum()) for k, m in pruner._masks.items()}

    for step in range(100, 600, 100):
        _with_gradients(net)
        pruner._redistribute(step, optimizer)

    after = {k: int(m.sum()) for k, m in pruner._masks.items()}
    assert after == before, "redistribution must not change any active count"


@pytest.mark.parametrize("growth", ["gradient", "random"])
def test_sparsity_does_not_drift_over_many_updates(growth):
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth=growth)
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)

    for step in range(100, TOTAL_STEPS, 100):
        _with_gradients(net)
        pruner._redistribute(step, optimizer)
        optimizer.step()
        pruner._apply_masks(optimizer)
        pruner._assert_on_target()

    assert pruner.pruned_sparsity() == pytest.approx(0.9, abs=0.005)


def test_grown_weights_are_zero_with_zero_momentum():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    optimizer = _sgd(net)

    # Give every parameter a momentum buffer to zero.
    _with_gradients(net)
    optimizer.step()
    pruner._apply_masks(optimizer)

    before = {k: m.clone() for k, m in pruner._masks.items()}
    _with_gradients(net)
    pruner._redistribute(100, optimizer)

    grew_somewhere = False
    for key, param in pruner._iter_targets():
        grown = pruner._masks[key] & ~before[key]
        if not bool(grown.any()):
            continue
        grew_somewhere = True
        assert torch.all(param[grown] == 0.0), f"{key}: grown weight not zero"
        buffer = optimizer.state[param]["momentum_buffer"]
        assert torch.all(
            buffer[grown] == 0.0
        ), f"{key}: grown weight kept stale momentum"
    assert grew_somewhere, "the test never exercised a growth event"


def test_grown_weights_are_still_zero_after_the_next_optimizer_step():
    """RigL grows at zero, and the mask update runs after ``optimizer.step()``,
    so a new connection must not move on the gradient that selected it."""
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    optimizer = _sgd(net)
    trainer = MagicMock(spec=Trainer)
    trainer.optimizers = [optimizer]

    _with_gradients(net)
    optimizer.step()
    pruner._apply_masks(optimizer)

    before = {k: m.clone() for k, m in pruner._masks.items()}
    trainer.global_step = 100
    _with_gradients(net)
    optimizer.step()
    pruner.on_train_batch_end(trainer, MagicMock(), None, None, 0)

    grew_somewhere = False
    for key, param in pruner._iter_targets():
        grown = pruner._masks[key] & ~before[key]
        if not bool(grown.any()):
            continue
        grew_somewhere = True
        assert torch.all(
            param[grown] == 0.0
        ), f"{key}: grown weight already moved"
    assert grew_somewhere, "the test never exercised a growth event"


def test_rigl_grows_the_top_gradient_inactive_positions():
    """After the drop step, the grown set is the largest |grad| among the
    positions inactive at that moment."""
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth="gradient")
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    _with_gradients(net)

    key = "2.weight"
    param = dict(pruner._iter_targets())[key]
    mask_before = pruner._masks[key].clone()
    grad = param.grad.clone()

    pruner._redistribute(100, optimizer)
    mask_after = pruner._masks[key]
    grown = (mask_after & ~mask_before).flatten().nonzero().squeeze(1)
    dropped = (mask_before & ~mask_after).flatten().nonzero().squeeze(1)
    assert grown.numel() == dropped.numel() > 0

    # Inactive after the drop step: never active, or just dropped.
    post_death = mask_before.flatten().clone()
    post_death[dropped] = False
    candidates = (~post_death).nonzero().squeeze(1)
    scores = grad.flatten().abs()[candidates]
    expected = candidates[
        torch.topk(scores, grown.numel(), largest=True).indices
    ]
    assert set(grown.tolist()) == set(expected.tolist())


def test_set_grows_only_inactive_positions():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth="random")
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    _with_gradients(net)

    before = {k: m.clone() for k, m in pruner._masks.items()}
    pruner._redistribute(100, optimizer)
    for key in pruner._masks:
        grown = pruner._masks[key] & ~before[key]
        dropped = before[key] & ~pruner._masks[key]
        assert int(grown.sum()) == int(dropped.sum())


def test_random_growth_differs_between_updates():
    """The RNG offset must advance, or every update replays one permutation."""
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth="random")
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)

    _with_gradients(net)
    before = pruner._masks["2.weight"].clone()
    pruner._redistribute(100, optimizer)
    first = (pruner._masks["2.weight"] & ~before).flatten().nonzero()

    _with_gradients(net)
    before = pruner._masks["2.weight"].clone()
    pruner._redistribute(200, optimizer)
    second = (pruner._masks["2.weight"] & ~before).flatten().nonzero()

    assert not torch.equal(first, second)


# =============================================================================
# 3. Mask enforcement
# =============================================================================


def test_masked_weights_stay_zero_under_momentum_and_weight_decay():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)

    for _ in range(20):
        _with_gradients(net)
        optimizer.step()
        pruner._apply_masks(optimizer)

    for key, param in pruner._iter_targets():
        assert torch.all(
            param[~pruner._masks[key]] == 0.0
        ), f"{key}: a masked weight moved off zero"


def test_static_leaves_the_mask_bit_identical():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, update_frequency=None)
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    before = {k: m.clone() for k, m in pruner._masks.items()}

    trainer = MagicMock(spec=Trainer)
    trainer.optimizers = [optimizer]
    for step in range(1, 501):
        trainer.global_step = step
        _with_gradients(net)
        optimizer.step()
        pruner.on_train_batch_end(trainer, MagicMock(), None, None, 0)

    for key in pruner._masks:
        assert torch.equal(before[key], pruner._masks[key]), f"{key} changed"


def test_gradients_are_never_masked():
    """RigL grows on the dense gradient, so the pruner leaves .grad alone."""
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    _with_gradients(net)

    param = dict(pruner._iter_targets())["2.weight"]
    grad_before = param.grad.clone()

    trainer = MagicMock(spec=Trainer)
    trainer.global_step = 100
    trainer.optimizers = [optimizer]
    pruner.on_train_batch_end(trainer, MagicMock(), None, None, 0)

    assert torch.equal(param.grad, grad_before)
    assert torch.any(param.grad[~pruner._masks["2.weight"]] != 0.0)


# =============================================================================
# 4. GraNet's cubic ramp
# =============================================================================


def test_granet_ramps_sparsity_and_stops_at_the_window():
    net = _net()
    pruner = _ready_pruner(
        net,
        amount=0.9,
        initial_amount=0.0,
        update_frequency=100,
        drop_fraction=0.5,
        drop_end_fraction=1.0,
        drop_end_value=0.005,
        final_prune_epoch=6,
    )
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    assert pruner.pruned_sparsity() == pytest.approx(0.0)

    trainer = MagicMock(spec=Trainer)
    trainer.optimizers = [optimizer]
    seen = []
    for step in range(100, TOTAL_STEPS + 1, 100):
        trainer.global_step = step
        _with_gradients(net)
        # Same order Lightning uses: step, then the hook that masks and updates.
        optimizer.step()
        pruner.on_train_batch_end(trainer, MagicMock(), None, None, 0)
        seen.append(pruner.pruned_sparsity())

    assert seen == sorted(seen), f"sparsity must be monotone, got {seen}"
    assert seen[-1] == pytest.approx(0.9, abs=0.01)
    # The ramp ends at epoch 6 of 10, so the tail is flat.
    assert seen[-1] == pytest.approx(seen[-3], abs=0.01)


def test_a_fixed_target_never_enters_the_ramp():
    """``_final_update`` stays 0 without ``initial_amount``, and a mask update
    index is 1 or more, so RigL never reaches the cubic."""
    pruner = _ready_pruner(_net(), amount=0.9)
    assert pruner._final_update == 0
    assert pruner._current_target == 0.9


# =============================================================================
# 5. SNIP
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
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


def _snip_trainer():
    dataset = torch.utils.data.TensorDataset(
        torch.randn(16, 3, 8, 8), torch.randint(0, 10, (16,))
    )
    trainer = MagicMock(spec=Trainer)
    trainer.world_size = 1
    trainer.train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8
    )
    return trainer


def test_snip_hits_the_target_and_picks_its_own_layer_budget():
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9, mask_init="snip", prune_first_layer=True, verbose=0
    )
    pruner._collect(model)
    pruner._build_masks(model, _snip_trainer())
    pruner._apply_masks(_sgd(model))

    assert compute_sparsity(_target_params(pruner)) == pytest.approx(
        0.9, abs=0.005
    )
    per_layer = {
        key: 1 - float(mask.float().mean())
        for key, mask in pruner._masks.items()
    }
    assert len({round(v, 3) for v in per_layer.values()}) > 1, (
        f"a global ranking must produce different layer sparsities, got "
        f"{per_layer}"
    )


class _StatefulClassifier(pl.LightningModule):
    """BatchNorm and a metric, like img.py: a scoring forward moves both."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.net = _net()
        self.train_metric = Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.net(self.norm(images))
        self.train_metric(logits, targets)
        return {"loss": nn.functional.cross_entropy(logits, targets)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_snip_leaves_batchnorm_stats_and_metrics_untouched():
    """SNIP measures the init; scoring must not move what training reports."""
    model = _StatefulClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9,
        mask_init="snip",
        snip_iterations=5,
        prune_first_layer=True,
        verbose=0,
    )
    pruner._collect(model)
    before = (
        model.norm.running_mean.clone(),
        model.norm.running_var.clone(),
        model.norm.num_batches_tracked.clone(),
    )

    pruner._build_masks(model, _snip_trainer())

    after = (
        model.norm.running_mean,
        model.norm.running_var,
        model.norm.num_batches_tracked,
    )
    for name, was, now in zip(("mean", "var", "count"), before, after):
        assert torch.equal(was, now), f"scoring moved BatchNorm running {name}"

    # A metric holding SNIP's batches would report below 1.0 on a perfect one.
    logits = torch.eye(10) * 10.0
    model.train_metric(logits, torch.arange(10))
    assert float(model.train_metric.compute()) == 1.0


def test_snip_clears_gradients_it_created():
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9, mask_init="snip", prune_first_layer=True, verbose=0
    )
    pruner._collect(model)
    pruner._build_masks(model, _snip_trainer())
    assert all(p.grad is None for p in model.parameters())


def test_snip_without_a_trainer_raises():
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9, mask_init="snip", prune_first_layer=True, verbose=0
    )
    pruner._collect(model)
    with pytest.raises(AssertionError, match="needs a live trainer"):
        pruner._build_masks(model, None)


def test_snip_layer_collapse_is_reported_not_fatal(caplog):
    """A global ranking collapses the deep layers first and nothing regrows
    them; the run continues because that accuracy is the benchmark's
    finding."""
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.999, mask_init="snip", prune_first_layer=True, verbose=0
    )
    pruner._collect(model)
    with caplog.at_level(logging.WARNING):
        pruner._build_masks(model, _snip_trainer())
    assert "SNIP collapsed" in caplog.text
    assert "output units" in caplog.text
    assert "snip_iterations" in caplog.text
    assert pruner._masks


def test_iterative_snip_hits_the_same_target():
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9,
        mask_init="snip",
        snip_iterations=10,
        prune_first_layer=True,
        verbose=0,
    )
    pruner._collect(model)
    pruner._build_masks(model, _snip_trainer())
    assert pruner.pruned_sparsity() == pytest.approx(0.9, abs=0.005)


def test_iterative_snip_only_ever_shrinks_the_mask():
    """Rescoring must never regrow a weight; growth is FORCE, not SNIP."""
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9,
        mask_init="snip",
        snip_iterations=4,
        prune_first_layer=True,
        verbose=0,
    )
    pruner._collect(model)
    seen = []
    original = pruner._snip_scores

    def record(model_, batch, masks):
        seen.append({key: mask.clone() for key, mask in masks.items()})
        return original(model_, batch, masks)

    pruner._snip_scores = record
    pruner._build_masks(model, _snip_trainer())
    seen.append(pruner._masks)
    for earlier, later in zip(seen, seen[1:]):
        for key in earlier:
            assert bool(
                (later[key] <= earlier[key]).all()
            ), f"{key} regrew a weight between prune steps"


def test_one_iteration_is_the_papers_single_global_ranking():
    torch.manual_seed(0)
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9, mask_init="snip", prune_first_layer=True, verbose=0
    )
    pruner._collect(model)
    pruner._build_masks(model, _snip_trainer())

    # The reference: |w * grad| on the untouched weights, one global threshold.
    torch.manual_seed(0)
    reference = _TinyClassifier()
    images, targets = next(iter(_snip_trainer().train_dataloader))
    nn.functional.cross_entropy(reference(images), targets).backward()
    scores = {
        name: (p.grad * p).abs()
        for name, p in reference.named_parameters()
        if name.endswith(".weight") and p.dim() > 1
    }
    flat = torch.cat([s.flatten() for s in scores.values()])
    n_keep = round(0.1 * flat.numel())
    threshold = torch.topk(flat, n_keep, sorted=True).values[-1]

    for key, mask in pruner._masks.items():
        assert torch.equal(mask, scores[key] >= threshold), key


# =============================================================================
# 6. A misspelled mode is caught at its dispatch site
# =============================================================================


def test_unknown_mask_init_raises():
    net = _net()
    pruner = DynamicSparsePruner(
        amount=0.9, mask_init="ekr", prune_first_layer=True, verbose=0
    )
    pruner._collect(net)
    with pytest.raises(ValueError, match="mask_init is 'erk' or 'snip'"):
        pruner._build_masks(net)


def test_unknown_growth_raises():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9, growth="gradiant")
    optimizer = _sgd(net)
    pruner._apply_masks(optimizer)
    _with_gradients(net)
    with pytest.raises(ValueError, match="growth is 'gradient' or 'random'"):
        pruner._redistribute(100, optimizer)


# =============================================================================
# 7. Checkpointing
# =============================================================================


def test_state_dict_round_trip_restores_the_mask():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    state = pruner.state_dict()

    fresh = DynamicSparsePruner(amount=0.9, prune_first_layer=True, verbose=0)
    fresh._collect(net)
    fresh.load_state_dict(state)
    fresh._restore_masks_to(torch.device("cpu"))

    for key in pruner._masks:
        assert torch.equal(pruner._masks[key], fresh._masks[key])


def test_model_state_dict_stays_dense_and_loads_strict():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    pruner._apply_masks(_sgd(net))

    state = net.state_dict()
    assert not any(
        k.endswith("_mask") or k.endswith("_orig") for k in state
    ), "the model state_dict must carry no pruning reparameterization keys"
    _net().load_state_dict(state, strict=True)


def test_restore_rejects_a_mask_set_that_does_not_match():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    pruner._masks.pop("2.weight")
    with pytest.raises(
        AssertionError, match="cover exactly the target weights"
    ):
        pruner._restore_masks_to(torch.device("cpu"))


def test_restore_rejects_an_extra_mask():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    pruner._masks["9.weight"] = torch.ones(4, 4, dtype=torch.bool)
    with pytest.raises(
        AssertionError, match="cover exactly the target weights"
    ):
        pruner._restore_masks_to(torch.device("cpu"))


def test_restore_rejects_a_reshaped_mask():
    net = _net()
    pruner = _ready_pruner(net, amount=0.9)
    pruner._masks["2.weight"] = torch.ones(3, 3, dtype=torch.bool)
    with pytest.raises(AssertionError, match="restored mask is"):
        pruner._restore_masks_to(torch.device("cpu"))


# =============================================================================
# 8. End to end
# =============================================================================


@pytest.mark.slow
def test_mini_fit_holds_the_target(tmp_path):
    model = _TinyClassifier()
    pruner = DynamicSparsePruner(
        amount=0.9,
        update_frequency=2,
        drop_end_fraction=1.0,
        prune_first_layer=True,
        verbose=0,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 3, 8, 8), torch.randint(0, 10, (32,))
    )
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        callbacks=[pruner],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=8))

    assert pruner.pruned_sparsity() == pytest.approx(0.9, abs=0.005)
    # Zeros on disk are a superset of the mask: a regrown weight may still
    # sit at 0 if its gradient was 0.
    assert compute_sparsity(_target_params(pruner)) >= pruner.pruned_sparsity()
    for key, param in pruner._iter_targets():
        assert torch.all(param[~pruner._masks[key]] == 0.0), key
