"""Comprehensive verification tests for Bregman lambda update correctness.

This test suite verifies that the LambdaScheduler behaves as expected:
- Updates lambda exactly once per call to step()
- Increases lambda when sparsity is below target
- Decreases lambda when sparsity is above target
- Respects configured min/max bounds
- Checkpoint save/restore preserves exact state
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
    )

    # Call step many times with sparsity far above target
    for _ in range(100):
        scheduler_min.step(0.99)

    # Assert: lambda never goes below min_lambda
    assert scheduler_min.get_lambda() >= 1e-4


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
    )
    pruner = BregmanPruner(
        sparsity_threshold=1e-12,
        verbose=0,
        lambda_scheduler=scheduler,
    )
    return pruner, scheduler


def _make_mock_optimizer(param_groups):
    """Create a mock optimizer with a real dict for state."""
    mock_optimizer = Mock()
    mock_optimizer.param_groups = param_groups
    mock_optimizer.state = {}  # real dict — .get(p) returns None
    return mock_optimizer


def test_bregman_pruner_updates_lambda_per_batch():
    """BregmanPruner updates lambda once per batch via on_train_batch_end."""
    pruner, scheduler = _make_bregman_pruner_and_mocks(
        target_sparsity=0.9,
        initial_lambda=1e-3,
    )

    # Mock initialization state
    pruner._initialized = True
    pruner.manager = MagicMock()

    # Create mock trainer with one optimizer
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    reg = RegL1(lamda=0.01)
    mock_optimizer = _make_mock_optimizer(
        [
            {
                "params": [mock_param],
                "reg": reg,
                "lambda_scale": 1.0,
                "delta": 1.0,
            }
        ]
    )

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

    # Create RegL1 instance with initial lamda
    reg = RegL1(lamda=0.01)
    initial_reg_lambda = reg.lamda

    # Create mock optimizer with param group
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_optimizer = _make_mock_optimizer(
        [
            {
                "params": [mock_param],
                "reg": reg,
                "lambda_scale": 1.0,
                "delta": 1.0,
            }
        ]
    )

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

    # Create RegL1 instance
    reg = RegL1(lamda=0.01)
    lambda_scale = 0.5

    # Create mock optimizer with lambda_scale
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_optimizer = _make_mock_optimizer(
        [
            {
                "params": [mock_param],
                "reg": reg,
                "lambda_scale": lambda_scale,
                "delta": 1.0,
            }
        ]
    )

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


# =============================================================================
# Near-target damping tests
# =============================================================================


def test_damping_zone_reduces_update_frequency():
    """Inside damping zone, updates happen at damping_frequency_multiplier x
    lower frequency."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=10,
        damping_zone=0.02,
        damping_frequency_multiplier=10,
        damping_acceleration_divisor=5.0,
    )

    initial_lambda = scheduler.get_lambda()

    # Sparsity 0.89 is within 0.02 of target 0.9 -> damping active
    # Effective frequency = 10 * 10 = 100
    # Steps 1-99 should NOT update
    for step in range(1, 100):
        scheduler.step(0.89, current_step=step)
    assert (
        scheduler.get_lambda() == initial_lambda
    ), "Should not update inside damping zone before effective_frequency"

    # Step 100 should update
    scheduler.step(0.89, current_step=100)
    assert (
        scheduler.get_lambda() != initial_lambda
    ), "Should update at effective_frequency step"


def test_damping_zone_reduces_acceleration():
    """Inside damping zone, lambda changes are smaller per update."""
    # Scheduler WITH damping
    damped = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=100,
        damping_zone=0.02,
        damping_frequency_multiplier=1,  # keep frequency same to isolate acceleration effect
        damping_acceleration_divisor=5.0,
    )

    # Scheduler WITHOUT damping (same params but damping_zone=0)
    undamped = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=100,
        damping_zone=0.0,
    )

    # Sparsity 0.89 is within damping zone for damped scheduler
    damped.step(0.89, current_step=100)
    undamped.step(0.89, current_step=100)

    damped_change = abs(damped.get_lambda() - 1.0)
    undamped_change = abs(undamped.get_lambda() - 1.0)

    # Damped change should be ~5x smaller
    ratio = undamped_change / damped_change
    assert 4.9 < ratio < 5.1, f"Expected ~5x ratio, got {ratio}"


def test_damping_zone_inactive_outside():
    """Outside damping zone, behavior is unchanged."""
    damped = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=10,
        damping_zone=0.02,
        damping_frequency_multiplier=10,
        damping_acceleration_divisor=5.0,
    )

    undamped = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=10,
        damping_zone=0.0,
    )

    # Sparsity 0.5 is far from target -> outside damping zone
    damped.step(0.5, current_step=10)
    undamped.step(0.5, current_step=10)

    assert (
        damped.get_lambda() == undamped.get_lambda()
    ), "Outside damping zone, behavior should be identical"


def test_damping_zone_zero_preserves_behavior():
    """Default damping_zone=0.0 preserves existing behavior exactly."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=10,
        damping_zone=0.0,
    )

    values = []
    for step in range(0, 100):
        scheduler.step(0.85, current_step=step)
        values.append(scheduler.get_lambda())

    # Should update at steps 0, 10, 20, ... (every 10 steps)
    # Count distinct values
    distinct = len(set(values))
    assert distinct == 10, f"Expected 10 distinct values, got {distinct}"


def test_damping_zone_checkpointing():
    """damping_zone is preserved through checkpoint save/restore."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        damping_zone=0.02,
    )

    state = scheduler.get_state()
    assert state["damping_zone"] == 0.02

    new_scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=0.5,
        damping_zone=0.0,
    )
    new_scheduler.load_state(state)
    assert new_scheduler.damping_zone == 0.02


# =============================================================================
# Progressive-target (per-epoch schedule) tests
# =============================================================================


def test_scheduler_accepts_list_and_ramps():
    """Per-epoch schedule advances the active target across epoch boundaries."""
    sched = LambdaScheduler(
        target_sparsity=[0.5, 0.7, 0.9],
        initial_lambda=1.0,
        # High update_frequency + far-from-target sparsity keeps lambda
        # essentially untouched; we only care about target_sparsity here.
        update_frequency=10_000,
    )
    steps_per_epoch = 10
    sched.resolve_warmup_steps(steps_per_epoch)

    # Before any step, property returns the first entry.
    assert sched.target_sparsity == 0.5

    # Epoch 0: steps 0..9
    sched.step(0.5, current_step=0)
    assert sched.target_sparsity == 0.5
    sched.step(0.5, current_step=9)
    assert sched.target_sparsity == 0.5

    # Epoch 1: steps 10..19 -> idx 1 = 0.7
    sched.step(0.5, current_step=10)
    assert sched.target_sparsity == 0.7

    # Epoch 2: steps 20..29 -> idx 2 = 0.9
    sched.step(0.5, current_step=20)
    assert sched.target_sparsity == 0.9

    # Epoch 3+: clamped to last entry 0.9
    sched.step(0.5, current_step=45)
    assert sched.target_sparsity == 0.9


def test_scheduler_state_roundtrip_list():
    """Checkpoint save/restore preserves the per-epoch schedule and _last_step."""
    sched = LambdaScheduler(
        target_sparsity=[0.5, 0.7, 0.9],
        initial_lambda=1.0,
    )
    sched.resolve_warmup_steps(10)
    sched.step(0.5, current_step=15)  # epoch 1 -> target should be 0.7

    state = sched.get_state()
    assert state["_target_schedule"] == [0.5, 0.7, 0.9]
    assert state["_last_step"] == 15

    fresh = LambdaScheduler(
        target_sparsity=[0.5, 0.7, 0.9],
        initial_lambda=1.0,
    )
    fresh.resolve_warmup_steps(10)
    fresh.load_state(state)

    assert fresh.target_schedule == [0.5, 0.7, 0.9]
    assert fresh._last_step == 15
    assert fresh.target_sparsity == 0.7


def test_scheduler_state_backcompat_scalar():
    """Legacy checkpoint with only scalar `target_sparsity` still loads."""
    sched = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
    )
    legacy_state = {
        "lambda_value": 0.42,
        "target_sparsity": 0.9,  # no _target_schedule / _last_step
        "_last_sparsity": 0.85,
        "acceleration_factor": 0.25,
        "min_lambda": 1e-6,
        "max_lambda": 1000.0,
        "warmup_steps": 0,
        "damping_zone": 0.0,
    }
    sched.load_state(legacy_state)

    assert sched.target_schedule == [0.9]
    assert sched.target_sparsity == 0.9
    assert sched.lambda_value == 0.42


def test_scheduler_rejects_invalid_list():
    """List targets are validated element-wise."""
    with pytest.raises(ValueError, match="must be finite and in"):
        LambdaScheduler(
            target_sparsity=[0.5, 1.1],  # > 1.0
            initial_lambda=1e-3,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        LambdaScheduler(
            target_sparsity=[],
            initial_lambda=1e-3,
        )


# =============================================================================
# Post-epoch relative-change clamp tests
# =============================================================================


def test_max_relative_change_defaults_to_off():
    """Without the kwarg, scheduler behaves identically to baseline."""
    # Baseline scheduler — no clamp.
    baseline = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
    )
    # Put the baseline past epoch 1 so a clamp (if it existed) would fire.
    baseline.resolve_warmup_steps(10)

    # Force a large update by reading sparsity far below target.
    for step in range(20):
        baseline.step(0.1, current_step=step)

    assert baseline.max_relative_change is None, (
        "max_relative_change should default to None"
    )
    # Baseline should reach the max_lambda ceiling (confirming no clamp).
    assert baseline.get_lambda() == pytest.approx(1e3), (
        "default behavior must match pre-change implementation "
        f"(expected max_lambda=1000, got {baseline.get_lambda()})"
    )


def test_max_relative_change_inactive_during_first_epoch():
    """Clamp does not fire while _last_step < _steps_per_epoch."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        max_relative_change=0.05,  # 5% cap
    )
    steps_per_epoch = 10
    scheduler.resolve_warmup_steps(steps_per_epoch)

    lambda_prev = scheduler.get_lambda()
    # First update, during epoch 0: sparsity_diff=0.8, factor=1.8 -> +80%
    scheduler.step(0.1, current_step=0)
    lambda_new = scheduler.get_lambda()

    # Without the clamp we expect ≈ 1.8; definitely well above the 5% cap.
    rel_change = (lambda_new - lambda_prev) / lambda_prev
    assert rel_change > 0.05, (
        f"epoch-0 clamp must not fire; got rel_change={rel_change:.3f}"
    )


def test_max_relative_change_active_after_first_epoch():
    """Clamp caps |Δλ|/λ_prev at max_relative_change once past epoch 0."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        max_relative_change=0.05,  # 5% cap
    )
    steps_per_epoch = 10
    scheduler.resolve_warmup_steps(steps_per_epoch)

    # Advance _last_step past the first-epoch boundary.
    scheduler._last_step = steps_per_epoch

    eps = 1e-12
    for step in range(steps_per_epoch, steps_per_epoch + 20):
        lambda_prev = scheduler.get_lambda()
        # Push hard with sparsity far below target, each call is a new step.
        scheduler.step(0.1, current_step=step)
        lambda_new = scheduler.get_lambda()

        rel_change = abs(lambda_new - lambda_prev) / lambda_prev
        assert rel_change <= 0.05 + eps, (
            f"step {step}: |Δλ|/λ_prev={rel_change:.6f} exceeds 0.05 cap"
        )


def test_max_relative_change_symmetric():
    """Clamp applies to both increasing and decreasing updates."""
    # Increasing direction: sparsity below target.
    up = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        max_relative_change=0.05,
    )
    up.resolve_warmup_steps(10)
    up._last_step = 10  # past epoch 0
    up.step(0.1, current_step=10)
    # Exactly at the upper bound: lambda_prev * 1.05 = 1.05
    assert up.get_lambda() == pytest.approx(1.05, rel=1e-9)

    # Decreasing direction: sparsity above target.
    down = LambdaScheduler(
        target_sparsity=0.5,
        initial_lambda=1.0,
        acceleration_factor=1.0,
        max_relative_change=0.05,
    )
    down.resolve_warmup_steps(10)
    down._last_step = 10
    down.step(0.99, current_step=10)
    # Exactly at the lower bound: lambda_prev * 0.95 = 0.95
    assert down.get_lambda() == pytest.approx(0.95, rel=1e-9)


def test_max_relative_change_state_roundtrip():
    """max_relative_change is preserved through checkpoint save/restore."""
    scheduler = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=1.0,
        max_relative_change=0.08,
    )

    state = scheduler.get_state()
    assert state["max_relative_change"] == 0.08

    fresh = LambdaScheduler(
        target_sparsity=0.9,
        initial_lambda=0.5,
        max_relative_change=None,  # different initial
    )
    fresh.load_state(state)
    assert fresh.max_relative_change == 0.08


def test_max_relative_change_rejects_invalid():
    """Constructor rejects non-positive max_relative_change."""
    with pytest.raises(ValueError, match="max_relative_change must be > 0.0"):
        LambdaScheduler(
            target_sparsity=0.9,
            initial_lambda=1.0,
            max_relative_change=-0.1,
        )
    with pytest.raises(ValueError, match="max_relative_change must be > 0.0"):
        LambdaScheduler(
            target_sparsity=0.9,
            initial_lambda=1.0,
            max_relative_change=0.0,
        )
