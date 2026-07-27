"""Comprehensive verification tests for Bregman lambda update correctness.

This test suite verifies that the LambdaScheduler behaves as expected:
- Updates lambda exactly once per call to step()
- Increases lambda when sparsity is below target
- Decreases lambda when sparsity is above target
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


def _controller(initial_lambda=1e-3, target_sparsity=0.9, **kwargs):
    """A scheduler bound to a run, as on_fit_start leaves it."""
    sched = LambdaScheduler(initial_lambda=initial_lambda, **kwargs)
    sched.target_sparsity = target_sparsity
    return sched


def test_lambda_update_frequency():
    """Lambda updates exactly once per call to step()."""
    scheduler = _controller()

    lambda_values = []

    # Call step 300 times with gradually increasing sparsity
    for i in range(300):
        sparsity = 0.5 + (0.4 * i / 299)  # 0.5 -> 0.9
        scheduler.step(sparsity, current_step=i)
        lambda_values.append(scheduler.get_lambda())

    # Assert: one update per call
    assert len(lambda_values) == 300

    # Assert: all values are finite and positive
    assert all(math.isfinite(lam) and lam > 0 for lam in lambda_values)


def test_lambda_increases_below_target():
    """Lambda increases when sparsity is below target."""
    scheduler = _controller()

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well below target
    for step in range(20):
        scheduler.step(0.5, current_step=step)
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
    scheduler = _controller(target_sparsity=0.5)

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well above target
    for step in range(20):
        scheduler.step(0.8, current_step=step)
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
    scheduler = _controller()

    initial_lambda = scheduler.get_lambda()

    # Call step with sparsity exactly at target
    scheduler.step(0.9, current_step=0)

    # Assert: lambda unchanged (sparsity_difference == 0)
    assert scheduler.get_lambda() == initial_lambda


def test_lambda_stays_positive():
    """The asymmetric update keeps lambda > 0 and finite for any accel.

    There is no min_lambda floor anymore: positivity comes from multiplying
    when below target and dividing when above (never <= 0).
    """
    # Far below target: lambda grows but stays finite over a bounded run.
    up = _controller(
        acceleration_factor=2.0,  # aggressive, would overflow a naive 1+a*gap
    )
    for step in range(50):
        up.step(0.1, current_step=step)
    assert up.get_lambda() > 0.0 and math.isfinite(up.get_lambda())

    # Far above target: lambda shrinks geometrically but never reaches 0.
    down = _controller(target_sparsity=0.1, acceleration_factor=2.0)
    for step in range(50):
        down.step(0.99, current_step=step)
    assert down.get_lambda() > 0.0 and math.isfinite(down.get_lambda())


def test_validation_rejects_invalid_sparsity():
    """Scheduler rejects invalid sparsity values."""
    scheduler = _controller()

    # Assert: sparsity < 0 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(-0.1, current_step=0)

    # Assert: sparsity > 1 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(1.5, current_step=0)

    # Assert: 0.0 is valid (model can start dense)
    scheduler.step(0.0, current_step=0)

    # Assert: NaN raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("nan"), current_step=0)

    # Assert: Inf raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("inf"), current_step=0)


# =============================================================================
# Integration tests for BregmanPruner
# =============================================================================


def _make_bregman_pruner_and_mocks(target_sparsity=0.9, initial_lambda=1e-3):
    """Create a BregmanPruner with lambda scheduler and mock trainer."""
    # on_fit_start binds in production; these tests enter at on_train_batch_end.
    scheduler = _controller(
        initial_lambda=initial_lambda, target_sparsity=target_sparsity
    )
    pruner = BregmanPruner(
        sparsity_threshold=1e-12,
        verbose=0,
        lambda_scheduler=scheduler,
        target_sparsity=target_sparsity,
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
                "lr": 1e-2,
            }
        ]
    )

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0
    mock_trainer.callback_metrics = {}  # real dict for the gate-metric publish
    pruner._optimizer = mock_optimizer  # on_fit_start normally stores this

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
                "lr": 1e-2,
            }
        ]
    )

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0
    mock_trainer.callback_metrics = {}  # real dict for the gate-metric publish
    pruner._optimizer = mock_optimizer  # on_fit_start normally stores this

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
                "lr": 1e-2,
            }
        ]
    )

    mock_trainer = Mock()
    mock_trainer.optimizers = [mock_optimizer]
    mock_trainer.global_step = 0
    mock_trainer.callback_metrics = {}  # real dict for the gate-metric publish
    pruner._optimizer = mock_optimizer  # on_fit_start normally stores this

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
# Update frequency and checkpoint state
# =============================================================================


def test_lambda_updates_only_every_update_frequency_steps():
    """Lambda moves on steps divisible by update_frequency, not in between."""
    scheduler = _controller(
        initial_lambda=1.0, acceleration_factor=1.0, update_frequency=10
    )

    values = [scheduler.step(0.85, s) for s in range(100)]

    distinct = len(set(values))
    assert distinct == 10, f"Expected an update every 10 steps, got {distinct}"


def test_checkpoint_round_trip_restores_lambda():
    """lambda_value is the only state a checkpoint has to carry."""
    scheduler = _controller(initial_lambda=1.0)
    scheduler.step(0.5, current_step=0)

    restored = LambdaScheduler(initial_lambda=0.5)
    restored.load_state(scheduler.get_state())

    assert restored.get_lambda() == scheduler.get_lambda()


def test_checkpoint_key_backward_compat():
    """Pruner reads the new namespaced key, falling back to the old one."""
    pruner, scheduler = _make_bregman_pruner_and_mocks(
        target_sparsity=0.9, initial_lambda=1.0
    )

    old_state = scheduler.get_state()
    old_state["lambda_value"] = 0.5
    pruner.on_load_checkpoint(
        Mock(), Mock(), {"lambda_scheduler_state": old_state}
    )
    assert pruner._ckpt_scheduler_state is old_state

    new_state = scheduler.get_state()
    new_state["lambda_value"] = 0.7
    pruner.on_load_checkpoint(
        Mock(),
        Mock(),
        {
            "bregman_lambda_scheduler_state": new_state,
            "lambda_scheduler_state": old_state,  # superseded by the new key
        },
    )
    assert pruner._ckpt_scheduler_state is new_state


# =============================================================================
# Controller setpoint
# =============================================================================


def test_step_feeds_the_measured_sparsity_and_the_global_step():
    """The controller sees the gate metric and the step, in that order."""
    pruner = BregmanPruner(target_sparsity=0.99)
    pruner.lambda_scheduler = Mock()
    pruner.lambda_scheduler.step.return_value = 1.0
    pruner._broadcast_lambda = Mock()

    trainer = Mock(current_epoch=5, global_step=100)
    pruner._optimizer = _make_mock_optimizer(
        [
            {
                "params": [],
                "reg": RegL1(lamda=0.01),
                "lambda_scale": 1.0,
                "lr": 1e-4,
            }
        ]
    )
    pruner._step_lambda_scheduler(trainer, current_sparsity=0.3)

    assert pruner.lambda_scheduler.step.call_args[0] == (0.3, 100)


# =============================================================================
# The relative lambda move and its acceleration factor
# =============================================================================


def test_relative_step_reads_the_gap():
    """dlambda/lambda is acceleration_factor * gap and nothing else."""
    sched = _controller(
        initial_lambda=1.0, target_sparsity=0.9, acceleration_factor=0.5
    )
    sched.step(0.5, current_step=0)  # gap = 0.4
    assert sched.last_delta_over_lambda == pytest.approx(0.5 * 0.4)
    assert sched.last_delta == pytest.approx(0.2)


def test_effective_acceleration_factor_holds_before_the_warmup():
    """Inside the hold window the hook returns the configured factor."""
    sched = _controller(acceleration_factor=0.75, update_frequency=50)
    assert sched.effective_acceleration_factor(0) == 0.75
    assert sched.effective_acceleration_factor(5000) == 0.75


def test_effective_acceleration_factor_decays_after_the_warmup():
    """Past the hold window the factor only falls, never rises."""
    sched = _controller(acceleration_factor=1.0, update_frequency=50)
    warmup = 1000 * 50  # warmup_updates * update_frequency, in global steps
    decayed = [
        sched.effective_acceleration_factor(warmup + n)
        for n in (100, 1000, 10000)
    ]
    assert decayed == sorted(decayed, reverse=True)
    assert all(0.0 < a < 1.0 for a in decayed)


def test_step_reads_the_factor_through_the_hook():
    """step() takes alpha from the hook, so overriding it steers lambda."""

    class DecayingScheduler(LambdaScheduler):
        def effective_acceleration_factor(self, num_updates):
            return self.acceleration_factor / (1 + num_updates)

    sched = DecayingScheduler(
        initial_lambda=1.0, acceleration_factor=1.0, update_frequency=10
    )
    sched.target_sparsity = 0.9

    sched.step(0.4, current_step=0)  # alpha = 1.0
    assert sched.last_delta_over_lambda == pytest.approx(0.5)

    sched.step(0.4, current_step=10)  # alpha = 1/11
    assert sched.last_delta_over_lambda == pytest.approx(0.5 / 11)


def test_the_hook_sees_the_global_step():
    """The hook reads the optimizer step, and only on update steps."""
    seen = []

    class RecordingScheduler(LambdaScheduler):
        def effective_acceleration_factor(self, num_updates):
            seen.append(num_updates)
            return self.acceleration_factor

    sched = RecordingScheduler(initial_lambda=1.0, update_frequency=10)
    sched.target_sparsity = 0.9
    for current_step in range(30):
        sched.step(0.5, current_step)

    assert seen == [0, 10, 20]


def test_lambda_is_reproduced_after_a_resume():
    """lambda is the only state, so a resumed run tracks a fresh one."""
    fresh = _controller(initial_lambda=1.0)
    for current_step in range(10):
        fresh.step(0.0, current_step=current_step)

    interrupted = _controller(initial_lambda=1.0)
    for current_step in range(5):
        interrupted.step(0.0, current_step=current_step)
    resumed = _controller(initial_lambda=1.0)
    resumed.load_state(interrupted.get_state())
    for current_step in range(5, 10):
        resumed.step(0.0, current_step=current_step)

    assert resumed.get_lambda() == pytest.approx(fresh.get_lambda())


def test_step_without_a_setpoint_is_refused():
    """A scheduler with no target_sparsity must not run."""
    sched = LambdaScheduler(initial_lambda=1.0)
    with pytest.raises(AssertionError, match="target_sparsity must be set"):
        sched.step(0.5, current_step=0)


def test_setup_binds_the_target_sparsity():
    """Fit start hands the pruner's setpoint to the controller."""
    scheduler = LambdaScheduler(initial_lambda=1.0)
    pruner = BregmanPruner(target_sparsity=0.9, lambda_scheduler=scheduler)
    pruner._setup_lambda_scheduler(is_resuming=False)

    assert scheduler.target_sparsity == pytest.approx(0.9)


def test_pruner_steps_the_scheduler_and_broadcasts_lambda():
    """End-to-end: one step moves lambda and lands it on the param groups."""
    scheduler = LambdaScheduler(initial_lambda=1.0, acceleration_factor=1.0)
    pruner = BregmanPruner(
        verbose=0, target_sparsity=0.99, lambda_scheduler=scheduler
    )
    optimizer = _make_mock_optimizer(
        [
            {
                "params": [],
                "reg": RegL1(lamda=1.0),
                "lambda_scale": 1.0,
                "lr": 1e-4,
            }
        ]
    )
    trainer = Mock(max_epochs=50, ckpt_path=None, global_step=0)
    pruner._optimizer = optimizer
    pruner._setup_lambda_scheduler(is_resuming=False)
    pruner._step_lambda_scheduler(trainer, current_sparsity=0.0)

    # gap = 0.99 at alpha = 1.0, so lambda nearly doubles.
    assert scheduler.get_lambda() == pytest.approx(1.99)
    assert optimizer.param_groups[0]["reg"].lamda == pytest.approx(1.99)
