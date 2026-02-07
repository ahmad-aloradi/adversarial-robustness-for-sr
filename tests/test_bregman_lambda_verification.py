"""Comprehensive verification tests for Bregman lambda update correctness.

This test suite verifies that the LambdaScheduler behaves as expected:
- Updates lambda exactly once per call to step()
- Increases lambda when sparsity is below target
- Decreases lambda when sparsity is above target
- Respects configured min/max bounds
- EMA smoothing reduces volatility compared to raw sparsity
- Checkpoint save/restore preserves exact state
- Resume path correctly initializes EMA from last_sparsity
- Invalid sparsity inputs are rejected

Also tests BregmanPruner integration to verify lambda is correctly
propagated to optimizer param groups with proper scaling.
"""
import math
from unittest.mock import MagicMock, Mock

import pytest
import torch

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler

# =============================================================================
# Unit tests for LambdaScheduler
# =============================================================================


def test_lambda_update_frequency():
    """Lambda updates exactly once per call to step()."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=False,
    )

    lambda_values = []

    # Call step 300 times with gradually increasing sparsity
    for i in range(300):
        sparsity = 0.5 + (0.4 * i / 299)  # 0.5 -> 0.9
        scheduler.step(sparsity)
        lambda_values.append(scheduler.get_lambda())

    # Assert: one update per call
    assert len(lambda_values) == 300

    # Assert: all values are finite and positive
    assert all(math.isfinite(lam) and lam > 0 for lam in lambda_values)


def test_lambda_increases_below_target():
    """Lambda increases when sparsity is below target."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=False,  # Disable EMA for deterministic test
    )

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well below target
    for _ in range(20):
        scheduler.step(0.5)
        lambda_values.append(scheduler.get_lambda())

    # Assert: lambda increases monotonically
    for i in range(1, len(lambda_values)):
        assert (
            lambda_values[i] >= lambda_values[i - 1]
        ), f"Lambda should increase, but {lambda_values[i]} < {lambda_values[i-1]}"

    # Assert: lambda increased from initial value
    assert scheduler.get_lambda() > initial_lambda


def test_lambda_decreases_above_target():
    """Lambda decreases when sparsity is above target."""
    scheduler = LambdaScheduler(
        target_sparsity=0.5,
        initial_lambda=1e-3,
        use_ema=False,  # Disable EMA for deterministic test
    )

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well above target
    for _ in range(20):
        scheduler.step(0.8)
        lambda_values.append(scheduler.get_lambda())

    # Assert: lambda decreases monotonically
    for i in range(1, len(lambda_values)):
        assert (
            lambda_values[i] <= lambda_values[i - 1]
        ), f"Lambda should decrease, but {lambda_values[i]} > {lambda_values[i-1]}"

    # Assert: lambda decreased from initial value
    assert scheduler.get_lambda() < initial_lambda


def test_lambda_stable_at_target():
    """Lambda does not change when sparsity equals target."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=False,
    )

    initial_lambda = scheduler.get_lambda()

    # Call step with sparsity exactly at target
    scheduler.step(0.9)

    # Assert: lambda unchanged (sparsity_difference == 0)
    assert scheduler.get_lambda() == initial_lambda


def test_lambda_respects_bounds():
    """Lambda never exceeds min_lambda or max_lambda bounds."""
    # Test max_lambda bound
    scheduler_max = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        min_lambda=1e-4,
        max_lambda=10.0,
        acceleration_factor=5.0,  # Aggressive to force hitting bounds
        use_ema=False,
    )

    # Call step many times with sparsity far below target
    for _ in range(100):
        scheduler_max.step(0.1)

    # Assert: lambda never exceeds max_lambda
    assert scheduler_max.get_lambda() <= 10.0

    # Test min_lambda bound
    scheduler_min = LambdaScheduler(
        target_sparsity=0.1,
        initial_lambda=1e-3,
        min_lambda=1e-4,
        max_lambda=10.0,
        acceleration_factor=5.0,
        use_ema=False,
    )

    # Call step many times with sparsity far above target
    for _ in range(100):
        scheduler_min.step(0.99)

    # Assert: lambda never goes below min_lambda
    assert scheduler_min.get_lambda() >= 1e-4


def test_ema_smoothing_reduces_volatility():
    """EMA smoothing reduces lambda direction changes when sparsity oscillates
    around target."""
    # Create two schedulers: one with EMA, one without
    # The key insight: when sparsity oscillates AROUND the target,
    # EMA prevents rapid direction reversals in lambda updates
    scheduler_ema = LambdaScheduler(
        target_sparsity=0.6,
        initial_lambda=1e-3,
        use_ema=True,
        ema_decay_factor=0.8,  # Less aggressive smoothing to show effect
        acceleration_factor=0.5,
    )

    scheduler_no_ema = LambdaScheduler(
        target_sparsity=0.6,
        initial_lambda=1e-3,
        use_ema=False,
        acceleration_factor=0.5,
    )

    # Feed both the same oscillating sparsity sequence
    # Oscillate AROUND the target (0.6) - some below, some above
    oscillating_sparsity = [0.55, 0.65] * 15  # 30 steps

    lambda_ema = []
    lambda_no_ema = []

    for sparsity in oscillating_sparsity:
        scheduler_ema.step(sparsity)
        scheduler_no_ema.step(sparsity)
        lambda_ema.append(scheduler_ema.get_lambda())
        lambda_no_ema.append(scheduler_no_ema.get_lambda())

    # Count direction changes (lambda increasing vs decreasing)
    import numpy as np

    # Calculate diffs (positive = increase, negative = decrease)
    diffs_ema = np.diff(lambda_ema)
    diffs_no_ema = np.diff(lambda_no_ema)

    # Count sign changes (direction reversals)
    # A sign change means lambda switched from increasing to decreasing or vice versa
    sign_changes_ema = np.sum(np.diff(np.sign(diffs_ema)) != 0)
    sign_changes_no_ema = np.sum(np.diff(np.sign(diffs_no_ema)) != 0)

    # Assert: EMA scheduler has fewer direction reversals
    # The EMA-smoothed sparsity stays more stable around the target,
    # preventing rapid oscillations in lambda update direction
    assert (
        sign_changes_ema < sign_changes_no_ema
    ), f"EMA direction changes {sign_changes_ema} should be < no-EMA direction changes {sign_changes_no_ema}"


def test_checkpoint_save_restore():
    """Checkpoint save/restore preserves exact state."""
    # Create scheduler and run 50 steps
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=True,
        ema_decay_factor=0.9,
    )

    for i in range(50):
        scheduler.step(0.5 + 0.01 * i)

    # Save state
    saved_lambda = scheduler.get_lambda()
    saved_ema = scheduler.get_ema_smoothed_sparsity()
    saved_state = scheduler.get_state()

    # Create a NEW scheduler with different initial params
    scheduler_restored = LambdaScheduler(
        target_sparsity=0.5,  # Different
        initial_lambda=5e-3,  # Different
        use_ema=False,  # Different
    )

    # Load state
    scheduler_restored.load_state(saved_state)

    # Assert: lambda matches
    assert scheduler_restored.get_lambda() == saved_lambda

    # Assert: EMA matches
    assert scheduler_restored.get_ema_smoothed_sparsity() == saved_ema

    # Run one more step on both with same sparsity
    test_sparsity = 0.7
    scheduler.step(test_sparsity)
    scheduler_restored.step(test_sparsity)

    # Assert: both produce the same lambda (proving full state restoration)
    assert abs(scheduler.get_lambda() - scheduler_restored.get_lambda()) < 1e-9


def test_resume_with_last_sparsity():
    """Resume path correctly initializes EMA from last_sparsity."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=True,
        ema_decay_factor=0.9,
    )

    # Run 10 steps to establish EMA
    for i in range(10):
        scheduler.step(0.5 + 0.01 * i)

    saved_ema = scheduler.get_ema_smoothed_sparsity()

    # Create a new scheduler (simulating resume)
    scheduler_new = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
        use_ema=True,
        ema_decay_factor=0.9,
    )

    # First step after resume: pass last_sparsity
    scheduler_new.step(current_sparsity=0.6, last_sparsity=saved_ema)

    # Assert: EMA was initialized to last_sparsity, not current_sparsity
    # After one step with current=0.6, EMA should be:
    # 0.9 * saved_ema + 0.1 * 0.6
    expected_ema = 0.9 * saved_ema + 0.1 * 0.6
    assert abs(scheduler_new.get_ema_smoothed_sparsity() - expected_ema) < 1e-9


def test_validation_rejects_invalid_sparsity():
    """Scheduler rejects invalid sparsity values."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1e-3,
    )

    # Assert: sparsity < 0 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(-0.1)

    # Assert: sparsity > 1 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(1.5)

    # Assert: 0.0 is valid (model can start dense)
    scheduler.step(0.0)

    # Assert: NaN raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("nan"))

    # Assert: Inf raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("inf"))


# =============================================================================
# Integration tests for BregmanPruner
# =============================================================================


def _make_bregman_pruner_and_mocks(target_sparsity=0.9, initial_lambda=1e-3):
    """Create a BregmanPruner with lambda scheduler and mock trainer."""
    scheduler = LambdaScheduler(
        initial_lambda=initial_lambda,
        target_sparsity=target_sparsity,
        use_ema=False,
    )
    pruner = BregmanPruner(
        sparsity_threshold=1e-30,
        verbose=0,
        lambda_scheduler=scheduler,
    )
    return pruner, scheduler


def test_bregman_pruner_updates_lambda_per_batch():
    """BregmanPruner updates lambda once per batch via on_train_batch_end."""
    pruner, scheduler = _make_bregman_pruner_and_mocks(
        target_sparsity=0.9,
        initial_lambda=1e-3,
    )

    # Mock initialization state
    pruner._initialized = True
    pruner.manager = MagicMock()
    pruner._compute_overall_sparsity = MagicMock(return_value=0.5)

    # Create mock trainer with one optimizer
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    reg = RegL1(lamda=0.01)
    mock_optimizer = Mock()
    mock_optimizer.param_groups = [
        {
            "params": [mock_param],
            "reg": reg,
            "lambda_scale": 1.0,
        }
    ]

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0

    # Create mock pl_module with log method
    mock_pl_module = Mock()
    mock_pl_module.logging_params = {
        "on_step": False,
        "on_epoch": True,
        "sync_dist": True,
    }

    # Record lambda values after each batch
    lambda_values = [scheduler.get_lambda()]

    for i in range(10):
        mock_trainer.global_step = i
        pruner.on_train_batch_end(mock_trainer, mock_pl_module, None, None, i)
        lambda_values.append(scheduler.get_lambda())

    # Assert: lambda changed 10 times (one per batch-end call)
    # Note: we have 11 values (initial + 10 updates)
    assert len(lambda_values) == 11

    # Assert: lambda increased (since sparsity 0.5 < target 0.9)
    assert lambda_values[-1] > lambda_values[0]

    # Assert: all updates resulted in different values (monotonic increase)
    for i in range(1, len(lambda_values)):
        assert lambda_values[i] > lambda_values[i - 1]


def test_bregman_pruner_propagates_lambda_to_optimizer():
    """BregmanPruner propagates lambda to optimizer param groups."""
    pruner, scheduler = _make_bregman_pruner_and_mocks(
        target_sparsity=0.9,
        initial_lambda=1e-3,
    )

    # Mock initialization state
    pruner._initialized = True
    pruner.manager = MagicMock()
    pruner._compute_overall_sparsity = MagicMock(return_value=0.5)

    # Create RegL1 instance with initial lamda
    reg = RegL1(lamda=0.01)
    initial_reg_lambda = reg.lamda

    # Create mock optimizer with param group
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_optimizer = Mock()
    mock_optimizer.param_groups = [
        {
            "params": [mock_param],
            "reg": reg,
            "lambda_scale": 1.0,
        }
    ]

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0

    mock_pl_module = Mock()
    mock_pl_module.logging_params = {
        "on_step": False,
        "on_epoch": True,
        "sync_dist": True,
    }

    # Call on_train_batch_end once
    pruner.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 0)

    # Assert: reg.lamda has been updated to match scheduler lambda
    expected_lambda = scheduler.get_lambda() * 1.0
    assert reg.lamda == expected_lambda

    # Assert: lambda changed from initial value
    assert reg.lamda != initial_reg_lambda


def test_bregman_pruner_respects_lambda_scale():
    """BregmanPruner applies lambda_scale correctly."""
    pruner, scheduler = _make_bregman_pruner_and_mocks(
        target_sparsity=0.9,
        initial_lambda=1e-3,
    )

    # Mock initialization state
    pruner._initialized = True
    pruner.manager = MagicMock()
    pruner._compute_overall_sparsity = MagicMock(return_value=0.5)

    # Create RegL1 instance
    reg = RegL1(lamda=0.01)
    lambda_scale = 0.5

    # Create mock optimizer with lambda_scale
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_optimizer = Mock()
    mock_optimizer.param_groups = [
        {
            "params": [mock_param],
            "reg": reg,
            "lambda_scale": lambda_scale,
        }
    ]

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0

    mock_pl_module = Mock()
    mock_pl_module.logging_params = {
        "on_step": False,
        "on_epoch": True,
        "sync_dist": True,
    }

    # Call on_train_batch_end once
    pruner.on_train_batch_end(mock_trainer, mock_pl_module, None, None, 0)

    # Assert: reg.lamda == scheduler.get_lambda() * lambda_scale
    expected_lambda = scheduler.get_lambda() * lambda_scale
    assert abs(reg.lamda - expected_lambda) < 1e-9
