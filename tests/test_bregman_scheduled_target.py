"""Verification tests for Bregman scheduled target relaxation mode.

This test suite verifies that:
- LambdaScheduler.update_target() evolves target_sparsity according to schedule
- Schedule formulas match PruningScheduler exactly (linear and constant)
- Fixed-target mode (backward compatibility) is unaffected
- Checkpoint save/restore preserves schedule state
- BregmanPruner integrates scheduled updates and validation suppression correctly
"""
from unittest.mock import MagicMock, Mock

import pytest
import torch

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.scheduler import PruningScheduler
from src.callbacks.pruning.utils.pruning_manager import PruningManager

# =============================================================================
# LambdaScheduler schedule tests
# =============================================================================


def test_linear_schedule_target_evolves():
    """Linear schedule evolves target_sparsity monotonically from initial to
    final (upward ramp)."""
    scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )

    # Initial state
    assert scheduler.target_sparsity == 0.0
    assert scheduler.is_scheduled
    assert not scheduler.schedule_complete

    targets = []
    for epoch in range(10):
        target = scheduler.update_target(epoch)
        targets.append(target)
        # Expected formula: initial + (final - initial) * (epoch + 1) / epochs_to_ramp
        expected = 0.0 + (0.9 - 0.0) * (epoch + 1) / 10
        assert (
            abs(target - expected) < 1e-9
        ), f"Epoch {epoch}: {target} != {expected}"

    # After epoch 0: closer to 0.0 than 0.9
    assert 0.0 < targets[0] < 0.9

    # After epoch 9: equals final target
    assert abs(targets[9] - 0.9) < 1e-9

    # Target increases monotonically (ramping upward)
    for i in range(1, len(targets)):
        assert targets[i] >= targets[i - 1]

    # Schedule is complete
    assert scheduler.schedule_complete


def test_constant_schedule_matches_pruning_scheduler():
    """Constant schedule produces identical values to PruningScheduler."""
    initial = 0.0
    final = 0.9
    epochs_to_ramp = 10

    ls = LambdaScheduler(
        schedule_type="constant",
        initial_target_sparsity=initial,
        final_target_sparsity=final,
        epochs_to_ramp=epochs_to_ramp,
    )
    ps = PruningScheduler(
        schedule_type="constant",
        initial_sparsity=initial,
        final_sparsity=final,
        epochs_to_ramp=epochs_to_ramp,
    )

    for epoch in range(epochs_to_ramp + 2):
        ls_val = ls.update_target(epoch)
        ps_val = ps.get_target_sparsity(epoch)
        assert abs(ls_val - ps_val) < 1e-9, (
            f"Mismatch at epoch {epoch}: LambdaScheduler={ls_val}, "
            f"PruningScheduler={ps_val}"
        )


def test_schedule_holds_final_target_after_ramp():
    """After ramp completes, target stays at final_target_sparsity."""
    scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=5,
    )

    # Complete the ramp
    for epoch in range(5):
        scheduler.update_target(epoch)

    # Verify final target reached
    assert abs(scheduler.target_sparsity - 0.9) < 1e-9

    # Continue beyond ramp
    for epoch in range(5, 15):
        target = scheduler.update_target(epoch)
        assert (
            abs(target - 0.9) < 1e-9
        ), f"Epoch {epoch}: target should hold at 0.9, got {target}"


def test_fixed_mode_unaffected():
    """Fixed-target mode (no schedule) is backward compatible."""
    scheduler = LambdaScheduler(target_sparsity=0.85)

    # No schedule active
    assert not scheduler.is_scheduled
    assert scheduler.schedule_complete

    initial_target = scheduler.target_sparsity

    # update_target is a no-op
    for epoch in range(10):
        target = scheduler.update_target(epoch)
        assert target == initial_target


def test_lambda_chases_moving_target():
    """Lambda adjusts as target evolves during scheduled mode (upward ramp)."""
    scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=5,
        acceleration_factor=0.25,
        use_ema=False,
    )

    # Epoch 0: target ~0.18, sparsity=0.5 (above target)
    scheduler.update_target(0)
    target_epoch_0 = scheduler.target_sparsity
    initial_lambda = scheduler.get_lambda()

    # Step with sparsity above target -> lambda should decrease
    for _ in range(20):
        scheduler.step(0.5)

    assert (
        scheduler.get_lambda() < initial_lambda
    ), "Lambda should decrease when sparsity > target"

    # Epoch 4: target evolves to 0.9, same sparsity=0.5 now BELOW target
    scheduler.update_target(4)
    target_epoch_4 = scheduler.target_sparsity
    assert target_epoch_4 > target_epoch_0  # Target moved higher (upward ramp)

    lambda_before = scheduler.get_lambda()
    # Step with sparsity below target -> lambda should INCREASE
    for _ in range(20):
        scheduler.step(0.5)

    assert (
        scheduler.get_lambda() > lambda_before
    ), "Lambda should increase when sparsity < target (target ramped above)"


def test_schedule_checkpoint_save_restore():
    """Checkpoint save/restore preserves schedule state."""
    # Create scheduler and run for 5 epochs (upward ramp)
    scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )

    for epoch in range(5):
        scheduler.update_target(epoch)

    # Save state
    state = scheduler.get_state()
    assert "_schedule_type" in state
    assert state["_schedule_type"] == "linear"
    assert state["_schedule_epoch"] == 4
    target_at_save = scheduler.target_sparsity

    # Create new scheduler and load state
    scheduler_restored = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )
    scheduler_restored.load_state(state)

    # Verify restored state
    assert scheduler_restored.target_sparsity == target_at_save
    assert scheduler_restored._schedule_epoch == 4

    # Continue from epoch 5 on both
    original_target = scheduler.update_target(5)
    restored_target = scheduler_restored.update_target(5)

    assert abs(original_target - restored_target) < 1e-9


def test_schedule_checkpoint_backward_compatibility():
    """Old checkpoints without schedule fields load gracefully."""
    # Create scheduler with schedule
    scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )

    # Simulate old checkpoint (no schedule fields)
    old_state = {
        "lambda_value": 0.01,
        "target_sparsity": 0.9,
        "_ema_smoothed_sparsity": None,
        "acceleration_factor": 0.25,
        "min_lambda": 1e-6,
        "max_lambda": 1e3,
        "use_ema": True,
        "ema_decay_factor": 0.9,
    }

    scheduler.load_state(old_state)

    # Should default to fixed mode (no schedule)
    assert scheduler._schedule_type is None
    assert not scheduler.is_scheduled
    assert scheduler.schedule_complete


# =============================================================================
# BregmanPruner integration tests
# =============================================================================


def test_bregman_pruner_updates_target_each_epoch():
    """BregmanPruner calls update_target at on_train_epoch_start."""
    # Create scheduled lambda scheduler (upward ramp)
    lambda_scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=5,
    )

    pruner = BregmanPruner(lambda_scheduler=lambda_scheduler)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()

    # Mock pruning_manager
    mock_manager = Mock(spec=PruningManager)
    mock_manager.get_pruned_parameters.return_value = [
        torch.nn.Parameter(torch.zeros(10, 10))
    ]
    mock_manager.processed_groups = []
    pl_module.pruning_manager = mock_manager

    # Initialize pruner
    mock_optimizer = Mock()
    mock_optimizer.param_groups = []
    trainer.optimizers = [mock_optimizer]
    trainer.ckpt_path = None
    pruner.on_fit_start(trainer, pl_module)

    # Simulate epochs
    targets = []
    for epoch in range(5):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, pl_module)
        targets.append(lambda_scheduler.target_sparsity)

    # Verify target evolved (upward ramp)
    assert len(set(targets)) > 1, "Target should change across epochs"
    assert targets[-1] > targets[0], "Target should increase (upward ramp)"


def test_bregman_pruner_suppresses_validation_during_ramp():
    """Validation is suppressed during schedule ramp, restored after."""
    lambda_scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=5,
    )

    pruner = BregmanPruner(lambda_scheduler=lambda_scheduler)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()
    trainer.limit_val_batches = 1.0

    # Mock pruning_manager
    mock_manager = Mock(spec=PruningManager)
    mock_manager.get_pruned_parameters.return_value = [
        torch.nn.Parameter(torch.zeros(10, 10))
    ]
    mock_manager.processed_groups = []
    pl_module.pruning_manager = mock_manager

    # Initialize
    mock_optimizer = Mock()
    mock_optimizer.param_groups = []
    trainer.optimizers = [mock_optimizer]
    trainer.ckpt_path = None
    pruner.on_fit_start(trainer, pl_module)

    # During ramp (epochs 0-3): validation suppressed
    for epoch in range(4):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, pl_module)
        assert (
            trainer.limit_val_batches == 0
        ), f"Epoch {epoch}: validation should be suppressed"

    # After ramp completes (epoch 4): validation restored
    trainer.current_epoch = 4
    pruner.on_train_epoch_start(trainer, pl_module)
    assert trainer.limit_val_batches == 1.0, "Validation should be restored"

    # Subsequent epochs: no change (guard prevents repeated restoration)
    for epoch in range(5, 10):
        trainer.current_epoch = epoch
        original_val = trainer.limit_val_batches
        pruner.on_train_epoch_start(trainer, pl_module)
        assert trainer.limit_val_batches == original_val


def test_bregman_pruner_no_suppression_in_fixed_mode():
    """Fixed-target mode does not suppress validation."""
    lambda_scheduler = LambdaScheduler(target_sparsity=0.9)

    pruner = BregmanPruner(lambda_scheduler=lambda_scheduler)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()
    trainer.limit_val_batches = 1.0

    # Mock pruning_manager
    mock_manager = Mock(spec=PruningManager)
    mock_manager.get_pruned_parameters.return_value = [
        torch.nn.Parameter(torch.zeros(10, 10))
    ]
    mock_manager.processed_groups = []
    pl_module.pruning_manager = mock_manager

    # Initialize
    mock_optimizer = Mock()
    mock_optimizer.param_groups = []
    trainer.optimizers = [mock_optimizer]
    trainer.ckpt_path = None
    pruner.on_fit_start(trainer, pl_module)

    # Run several epochs
    for epoch in range(10):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, pl_module)
        assert (
            trainer.limit_val_batches == 1.0
        ), f"Epoch {epoch}: validation should not be suppressed in fixed mode"


def test_bregman_pruner_checkpoint_preserves_schedule_state():
    """BregmanPruner checkpoint saves and restores schedule state."""
    lambda_scheduler = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )

    pruner = BregmanPruner(lambda_scheduler=lambda_scheduler)

    # Mock trainer and module
    trainer = Mock()
    pl_module = Mock()

    # Mock pruning_manager
    mock_manager = Mock(spec=PruningManager)
    mock_manager.get_pruned_parameters.return_value = [
        torch.nn.Parameter(torch.ones(5, 5) * 0.001)  # Very sparse
    ]
    mock_manager.processed_groups = []
    pl_module.pruning_manager = mock_manager

    # Initialize
    mock_optimizer = Mock()
    mock_optimizer.param_groups = []
    trainer.optimizers = [mock_optimizer]
    trainer.ckpt_path = None
    pruner.on_fit_start(trainer, pl_module)

    # Run 3 epochs to advance schedule
    for epoch in range(3):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, pl_module)

    target_at_save = lambda_scheduler.target_sparsity
    schedule_epoch_at_save = lambda_scheduler._schedule_epoch

    # Save checkpoint
    checkpoint = {}
    pruner.on_save_checkpoint(trainer, pl_module, checkpoint)

    # Verify schedule state in checkpoint
    assert "lambda_scheduler_state" in checkpoint
    sched_state = checkpoint["lambda_scheduler_state"]
    assert sched_state["_schedule_type"] == "linear"
    assert sched_state["_schedule_epoch"] == schedule_epoch_at_save
    assert sched_state["_initial_target_sparsity"] == 0.0
    assert sched_state["_final_target_sparsity"] == 0.9
    assert sched_state["_epochs_to_ramp"] == 10

    # Create new pruner with scheduled lambda scheduler
    lambda_scheduler_new = LambdaScheduler(
        schedule_type="linear",
        initial_target_sparsity=0.0,
        final_target_sparsity=0.9,
        epochs_to_ramp=10,
    )
    pruner_new = BregmanPruner(lambda_scheduler=lambda_scheduler_new)

    # Load checkpoint
    pruner_new.on_load_checkpoint(trainer, pl_module, checkpoint)

    # Re-initialize (simulates on_fit_start after load_checkpoint)
    trainer.ckpt_path = "/fake/path.ckpt"  # Indicate resuming
    pruner_new.on_fit_start(trainer, pl_module)

    # Verify schedule state restored
    assert lambda_scheduler_new.target_sparsity == target_at_save
    assert lambda_scheduler_new._schedule_epoch == schedule_epoch_at_save

    # Continue from next epoch
    trainer.current_epoch = 3
    pruner_new.on_train_epoch_start(trainer, pl_module)

    # Verify schedule continues correctly
    expected_target_epoch_3 = 0.0 + (0.9 - 0.0) * 4 / 10
    assert (
        abs(lambda_scheduler_new.target_sparsity - expected_target_epoch_3)
        < 1e-9
    )
