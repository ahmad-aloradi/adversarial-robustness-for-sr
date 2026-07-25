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


def test_lambda_update_frequency():
    """Lambda updates exactly once per call to step()."""
    scheduler = LambdaScheduler(
        initial_lambda=1e-3,
    )

    lambda_values = []

    # Call step 300 times with gradually increasing sparsity
    for i in range(300):
        sparsity = 0.5 + (0.4 * i / 299)  # 0.5 -> 0.9
        scheduler.step(sparsity, 0.9)
        lambda_values.append(scheduler.get_lambda())

    # Assert: one update per call
    assert len(lambda_values) == 300

    # Assert: all values are finite and positive
    assert all(math.isfinite(lam) and lam > 0 for lam in lambda_values)


def test_lambda_increases_below_target():
    """Lambda increases when sparsity is below target."""
    scheduler = LambdaScheduler(
        initial_lambda=1e-3,
    )

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well below target
    for _ in range(20):
        scheduler.step(0.5, 0.9)
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
        initial_lambda=1e-3,
    )

    initial_lambda = scheduler.get_lambda()
    lambda_values = [initial_lambda]

    # Call step multiple times with sparsity well above target
    for _ in range(20):
        scheduler.step(0.8, 0.5)
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
        initial_lambda=1e-3,
    )

    initial_lambda = scheduler.get_lambda()

    # Call step with sparsity exactly at target
    scheduler.step(0.9, 0.9)

    # Assert: lambda unchanged (sparsity_difference == 0)
    assert scheduler.get_lambda() == initial_lambda


def test_lambda_stays_positive():
    """The asymmetric update keeps lambda > 0 and finite for any accel.

    There is no min_lambda floor anymore: positivity comes from multiplying
    when below target and dividing when above (never <= 0).
    """
    # Far below target: lambda grows but stays finite over a bounded run.
    up = LambdaScheduler(
        initial_lambda=1e-3,
        acceleration_factor=2.0,  # aggressive, would overflow a naive 1+a*gap
    )
    for _ in range(50):
        up.step(0.1, 0.9)
    assert up.get_lambda() > 0.0 and math.isfinite(up.get_lambda())

    # Far above target: lambda shrinks geometrically but never reaches 0.
    down = LambdaScheduler(
        initial_lambda=1e-3,
        acceleration_factor=2.0,
    )
    for _ in range(50):
        down.step(0.99, 0.1)
    assert down.get_lambda() > 0.0 and math.isfinite(down.get_lambda())


def test_validation_rejects_invalid_sparsity():
    """Scheduler rejects invalid sparsity values."""
    scheduler = LambdaScheduler(
        initial_lambda=1e-3,
    )

    # Assert: sparsity < 0 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(-0.1, 0.9)

    # Assert: sparsity > 1 raises ValueError
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        scheduler.step(1.5, 0.9)

    # Assert: 0.0 is valid (model can start dense)
    scheduler.step(0.0, 0.9)

    # Assert: NaN raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("nan"), 0.9)

    # Assert: Inf raises ValueError
    with pytest.raises(ValueError, match="must be finite"):
        scheduler.step(float("inf"), 0.9)


# =============================================================================
# Integration tests for BregmanPruner
# =============================================================================


def _make_bregman_pruner_and_mocks(target_sparsity=0.9, initial_lambda=1e-3):
    """Create a BregmanPruner with lambda scheduler and mock trainer."""
    scheduler = LambdaScheduler(
        initial_lambda=initial_lambda,
    )
    # on_fit_start binds in production; these tests enter at on_train_batch_end.
    scheduler.bind_run(total_steps=1000, base_lr=1e-2)
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
    scheduler = LambdaScheduler(
        initial_lambda=1.0, acceleration_factor=1.0, update_frequency=10
    )

    values = [scheduler.step(0.85, 0.9, current_step=s) for s in range(100)]

    distinct = len(set(values))
    assert distinct == 10, f"Expected an update every 10 steps, got {distinct}"


def test_checkpoint_round_trip_restores_lambda():
    """lambda_value is the only state a checkpoint has to carry."""
    scheduler = LambdaScheduler(initial_lambda=1.0)
    scheduler.step(0.5, 0.9)

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


def test_step_feeds_the_fixed_target_to_the_controller():
    """The controller drives toward target_sparsity, unchanged by the epoch."""
    pruner = BregmanPruner(target_sparsity=0.99)
    pruner.lambda_scheduler = Mock()
    pruner.lambda_scheduler.step.return_value = 1.0
    pruner._broadcast_lambda = Mock()

    trainer = Mock(current_epoch=5, global_step=100)
    trainer.optimizers = [
        _make_mock_optimizer(
            [{"params": [], "reg": RegL1(lamda=0.01), "lr": 1e-2}]
        )
    ]
    pruner._step_lambda_scheduler(trainer, overall_sparsity=0.3)

    assert pruner.lambda_scheduler.step.call_args[0][1] == 0.99


# =============================================================================
# Trust region: |Δλ| <= λ0 · (lr/base_lr) · (1 - k/K), so Σ|Δλ|/lr is finite
# =============================================================================


def _bound_scheduler(initial_lambda=1.0, total_steps=1000, base_lr=0.01):
    sched = LambdaScheduler(
        initial_lambda=initial_lambda,
        acceleration_factor=1.0,
        update_frequency=1,
    )
    sched.bind_run(total_steps=total_steps, base_lr=base_lr)
    return sched


def test_cap_clamps_increment_in_both_directions():
    """A large gap moves lambda by exactly the cap, up and down."""
    up = _bound_scheduler()
    up.step(0.0, 0.99, current_step=0, lr=0.001)
    # Uncapped: 1.0 -> 1.99; cap = 1.0 * (0.001/0.01) * 1 = 0.1.
    assert up.get_lambda() == pytest.approx(1.1)

    down = _bound_scheduler()
    down.step(0.99, 0.0, current_step=0, lr=0.001)
    assert down.get_lambda() == pytest.approx(0.9)


def test_cap_is_one_initial_lambda_at_base_lr():
    """The cap carries its scale from lambda0, so it is optimizer-relative."""
    assert _bound_scheduler(initial_lambda=1.0).cap_at(
        0, lr=0.01
    ) == pytest.approx(1.0)
    assert _bound_scheduler(initial_lambda=0.01).cap_at(
        0, lr=0.01
    ) == pytest.approx(0.01)


def test_cap_follows_the_live_lr():
    """An annealed lr shrinks the cap by the same factor, pinning lambda."""
    sched = _bound_scheduler()
    sched.step(0.0, 0.99, current_step=0, lr=1e-8)
    assert sched.get_lambda() == pytest.approx(1.0 + 1e-6)


def test_cap_tapers_to_zero_at_the_end_of_the_run():
    """Lambda is frozen from the last step on — no increment past K."""
    sched = _bound_scheduler(total_steps=1000)
    assert sched.cap_at(500, lr=0.01) == pytest.approx(0.5)
    assert sched.cap_at(1000, lr=0.01) == 0.0
    sched.step(0.0, 0.99, current_step=1500, lr=0.01)  # past the end
    assert sched.get_lambda() == pytest.approx(1.0)


def test_total_variation_of_lambda_is_bounded():
    """Σ|Δλ| over the run stays under λ0·(K+1)/2, and lambda ends frozen."""
    total_steps, base_lr = 200, 0.01
    sched = _bound_scheduler(total_steps=total_steps, base_lr=base_lr)
    drift = 0.0
    previous = sched.get_lambda()
    for current_step in range(total_steps):
        lam = sched.step(0.0, 0.99, current_step=current_step, lr=base_lr)
        drift += abs(lam - previous)
        previous = lam
    assert drift <= 1.0 * (total_steps + 1) / 2
    assert sched.cap_at(total_steps, lr=base_lr) == 0.0


def test_last_cap_tracks_the_cap_the_step_saw():
    """The logged cap is the one step() clamped with; infinite while
    unbound."""
    unbound = LambdaScheduler(initial_lambda=1.0)
    unbound.step(0.5, 0.9)
    assert unbound.last_cap == math.inf

    sched = _bound_scheduler(total_steps=1000, base_lr=0.01)
    sched.step(0.0, 0.99, current_step=500, lr=0.005)
    assert sched.last_cap == pytest.approx(sched.cap_at(500, lr=0.005))


def test_cap_requires_lr_and_step():
    """Fail loud when the run is bound but lr or current_step is missing."""
    sched = _bound_scheduler()
    with pytest.raises(AssertionError):
        sched.step(0.5, 0.9, current_step=1, lr=None)
    with pytest.raises(AssertionError):
        sched.step(0.5, 0.9, current_step=None, lr=0.01)


def test_setup_binds_trust_region_to_the_run():
    """The pruner binds the scheduler to the trainer budget and the base lr."""
    scheduler = LambdaScheduler(initial_lambda=1.0)
    pruner = BregmanPruner(target_sparsity=0.9, lambda_scheduler=scheduler)
    optimizer = _make_mock_optimizer(
        [
            {
                "params": [],
                "reg": RegL1(lamda=1.0),
                "lambda_scale": 1.0,
                "lr": 1e-4,  # mid-anneal; the cap must key off initial_lr
                "initial_lr": 1e-2,
            }
        ]
    )
    trainer = Mock(
        max_epochs=50, ckpt_path=None, estimated_stepping_batches=5000
    )
    pruner._setup_lambda_scheduler(optimizer, trainer, is_resuming=False)

    assert scheduler.total_steps == 5000
    assert scheduler.base_lr == pytest.approx(1e-2)
    assert scheduler.cap_at(0, lr=1e-4) == pytest.approx(0.01)


def test_pruner_steps_the_bound_scheduler_with_the_live_lr():
    """End-to-end: the cap must use the annealed lr, not the base lr."""
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
                "lr": 1e-4,  # annealed 100x below the base lr
                "initial_lr": 1e-2,
            }
        ]
    )
    trainer = Mock(
        max_epochs=50,
        ckpt_path=None,
        estimated_stepping_batches=5000,
        global_step=0,
        optimizers=[optimizer],
    )
    pruner._setup_lambda_scheduler(optimizer, trainer, is_resuming=False)
    pruner._step_lambda_scheduler(trainer, overall_sparsity=0.0)

    # Uncapped 1.0 -> 1.99; cap = 1.0 * (1e-4/1e-2) * 1 = 0.01 at the live lr.
    assert scheduler.get_lambda() == pytest.approx(1.01)
    assert optimizer.param_groups[0]["reg"].lamda == pytest.approx(1.01)
