"""Unit tests for BregmanPruner._pruned_sparsity.

Regression guard: in a pure-regularization run (sparsity_rate=0) the
applier-based ``manager.get_pruned_parameters()`` is empty, which used to make
``bregman/pruned_sparsity`` report 0.000 even while ``bregman/sparsity`` showed
~0.99. The metric must instead measure the groups that carry an active
regularizer (lambda_scale > 0).
"""
import pytest
import torch
import torch.nn as nn

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1, RegNone


def _make_optimizer(groups):
    return torch.optim.SGD(groups, lr=0.1)


def test_pruned_sparsity_measures_regularized_groups():
    """pruned_sparsity reflects the regularized weights, not the applier."""
    w_sparse = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0]))  # 50% zero
    w_dense = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # 0% zero

    pruner = BregmanPruner()
    pruner._optimizer = _make_optimizer(
        [
            {"params": [w_sparse], "reg": RegL1(lamda=0.1), "lambda_scale": 1.0},
            {"params": [w_dense], "reg": RegL1(lamda=0.1), "lambda_scale": 1.0},
        ]
    )

    # 2 zeros out of 8 regularized elements.
    assert pruner._pruned_sparsity() == pytest.approx(0.25)


def test_pruned_sparsity_excludes_unregularized_groups():
    """Groups with lambda_scale == 0 (norm/bias) are excluded entirely."""
    w_reg = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))  # 25% zero
    bias = nn.Parameter(torch.zeros(4))  # all zero, but unregularized

    pruner = BregmanPruner()
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

    pruner = BregmanPruner()
    pruner._optimizer = _make_optimizer(
        [{"params": [bias], "reg": RegNone(lamda=0.1), "lambda_scale": 0.0}]
    )

    assert pruner._pruned_sparsity() == 0.0


def test_pruned_sparsity_asserts_when_uninitialized():
    """Calling before on_fit_start stored the optimizer is a hard error."""
    pruner = BregmanPruner()
    with pytest.raises(AssertionError):
        pruner._pruned_sparsity()
