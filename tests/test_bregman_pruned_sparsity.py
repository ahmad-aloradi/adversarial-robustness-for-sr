"""Unit tests for BregmanPruner's epoch-end measurements.

Regression guard: in a pure-regularization run (sparsity_rate=0) the
applier-based ``manager.get_pruned_parameters()`` is empty, which used to make
``bregman/pruned_sparsity`` report 0.000 even while ``bregman/sparsity`` showed
~0.99. The metric must instead measure the groups that carry an active
regularizer (lambda_scale > 0).

Support turnover is measured on the same groups: births against the new
support, deaths against the old.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1, RegNone
from src.callbacks.pruning.shared_prune_utils import reported_sparsities


def _make_optimizer(groups):
    return torch.optim.SGD(groups, lr=0.1)


def test_pruned_sparsity_measures_regularized_groups():
    """pruned_sparsity reflects the regularized weights, not the applier."""
    w_sparse = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0]))  # 50% zero
    w_dense = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # 0% zero

    pruner = BregmanPruner(target_sparsity=0.9)
    pruner._optimizer = _make_optimizer(
        [
            {
                "params": [w_sparse],
                "reg": RegL1(lamda=0.1),
                "lambda_scale": 1.0,
            },
            {
                "params": [w_dense],
                "reg": RegL1(lamda=0.1),
                "lambda_scale": 1.0,
            },
        ]
    )

    # 2 zeros out of 8 regularized elements.
    assert pruner._pruned_sparsity() == pytest.approx(0.25)


def test_pruned_sparsity_excludes_unregularized_groups():
    """Groups with lambda_scale == 0 (norm/bias) are excluded entirely."""
    w_reg = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))  # 25% zero
    bias = nn.Parameter(torch.zeros(4))  # all zero, but unregularized

    pruner = BregmanPruner(target_sparsity=0.9)
    pruner._optimizer = _make_optimizer(
        [
            {"params": [w_reg], "reg": RegL1(lamda=0.1), "lambda_scale": 1.0},
            {"params": [bias], "reg": RegNone(lamda=0.1), "lambda_scale": 0.0},
        ]
    )

    # If the all-zero bias group leaked in, this would be 5/8 = 0.625.
    assert pruner._pruned_sparsity() == pytest.approx(0.25)


def test_pruned_sparsity_zero_when_no_regularized_groups():
    """No active regularizer -> empty set -> 0.0 (compute_sparsity guard)."""
    bias = nn.Parameter(torch.zeros(4))

    pruner = BregmanPruner(target_sparsity=0.9)
    pruner._optimizer = _make_optimizer(
        [{"params": [bias], "reg": RegNone(lamda=0.1), "lambda_scale": 0.0}]
    )

    assert pruner._pruned_sparsity() == 0.0


def test_pruned_sparsity_asserts_when_uninitialized():
    """Calling before on_fit_start stored the optimizer is a hard error."""
    pruner = BregmanPruner(target_sparsity=0.9)
    with pytest.raises(AssertionError):
        pruner._pruned_sparsity()


class _FakeTrainer:
    def __init__(self):
        self.callback_metrics = {}
        self.current_epoch = 0


class _FakeModule:
    """Enough LightningModule surface for on_train_epoch_end: parameters to
    measure, and a log_dict that records instead of writing to a logger."""

    logging_params = {"on_step": False, "on_epoch": True, "sync_dist": True}

    def __init__(self, params):
        self._params = params
        self.logged = {}

    def parameters(self):
        return iter(self._params)

    def log_dict(self, metrics, **kwargs):
        self.logged.update(metrics)


def _turnover_pruner(w):
    """A pruner wired to one regularized parameter, past on_fit_start."""
    pruner = BregmanPruner(target_sparsity=0.9, verbose=0)
    pruner._optimizer = _make_optimizer(
        [{"params": [w], "reg": RegL1(lamda=0.1), "lambda_scale": 1.0}]
    )
    pruner._initialized = True
    module = _FakeModule([w])
    pruner.manager = type("_M", (), {"pl_module": module})()
    return pruner, _FakeTrainer(), module


def test_support_turnover_counts_births_and_deaths():
    """Births are measured against the new support, deaths against the old."""
    w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    pruner, trainer, module = _turnover_pruner(w)

    pruner.on_train_epoch_end(trainer, module)  # stores the first snapshot
    # One survivor dies, two zeros are born: support 3 -> 4.
    w.data = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    pruner.on_train_epoch_end(trainer, module)

    assert module.logged["bregman/support_births"] == pytest.approx(2 / 4)
    assert module.logged["bregman/support_deaths"] == pytest.approx(1 / 3)


def test_support_turnover_separates_the_two_denominators():
    """A pure prune births nothing and normalizes deaths on the old support."""
    w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    pruner, trainer, module = _turnover_pruner(w)

    pruner.on_train_epoch_end(trainer, module)
    w.data = torch.tensor([1.0, 1.0, 0.0, 0.0])
    pruner.on_train_epoch_end(trainer, module)

    assert module.logged["bregman/support_births"] == 0.0
    assert module.logged["bregman/support_deaths"] == pytest.approx(2 / 4)


def test_support_turnover_is_silent_on_the_first_epoch():
    """Nothing to compare against, so nothing is logged."""
    w = nn.Parameter(torch.tensor([1.0, 0.0, 1.0]))
    pruner, trainer, module = _turnover_pruner(w)

    pruner.on_train_epoch_end(trainer, module)

    assert "bregman/support_births" not in module.logged
    assert pruner._prev_support is not None


def test_support_turnover_rejects_an_empty_support():
    """A support that empties out is the bug, not a case to report on."""
    w = nn.Parameter(torch.tensor([1.0, 1.0]))
    pruner, trainer, module = _turnover_pruner(w)

    pruner.on_train_epoch_end(trainer, module)
    w.data = torch.zeros(2)
    with pytest.raises(AssertionError, match="non-empty support"):
        pruner.on_train_epoch_end(trainer, module)


# ---------------------------------------------------------------------------
# reported_sparsities — the pair src/modules/{img,sv}.py write out
# ---------------------------------------------------------------------------


class _ReportingModule(nn.Module):
    """A model with norms and biases, plus the callback list the report reads."""

    def __init__(self, callbacks):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(20, 20), nn.BatchNorm1d(20), nn.Linear(20, 20)
        )
        for layer in (self.body[0], self.body[2]):
            layer.weight.data.fill_(1.0)
            layer.weight.data[:10] = 0.0  # half the weights of each Linear
            layer.bias.data.fill_(1.0)
        self.trainer = SimpleNamespace(callbacks=callbacks)


def test_reported_sparsities_measures_weights_without_a_pruner():
    """Both figures come from the model when no callback claims the weights."""
    overall, pruned = reported_sparsities(_ReportingModule([]))

    # 400 zeros of 800 Linear weights. overall adds the 80 norm and bias entries,
    # 20 of which the BatchNorm bias starts at zero — no method put them there.
    assert pruned == pytest.approx(0.5)
    assert overall == pytest.approx(420 / 880)


def test_reported_sparsities_takes_the_pruners_own_figure():
    """Only the callback knows which weights it holds dense, so it states the
    pruned figure."""
    w = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # 75% zero
    pruner = BregmanPruner(target_sparsity=0.9)
    pruner._optimizer = _make_optimizer(
        [{"params": [w], "reg": RegL1(lamda=0.1), "lambda_scale": 1.0}]
    )

    overall, pruned = reported_sparsities(_ReportingModule([pruner]))

    assert pruned == pytest.approx(0.75)  # the model's own weights read 0.5
    assert overall == pytest.approx(420 / 880)  # unchanged: the model states it
