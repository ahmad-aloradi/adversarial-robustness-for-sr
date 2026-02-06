"""Tests for MagnitudePruner validation suppression during sparsity ramp-up."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.callbacks.pruning.prune import MagnitudePruner


def _make_pruner_and_mocks(final_amount=0.9, epochs_to_ramp=5):
    """Create a MagnitudePruner with scheduled pruning and mock
    trainer/module."""
    pruner = MagnitudePruner(
        pruning_fn="l1_unstructured",
        amount=final_amount,
        scheduled_pruning=True,
        epochs_to_ramp=epochs_to_ramp,
        schedule_type="linear",
        verbose=0,
    )
    # Create a simple model with known weights
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))

    # Mock trainer
    trainer = MagicMock(spec=Trainer)
    trainer.callbacks = []
    trainer.max_epochs = 20
    trainer.limit_val_batches = 1.0  # Default Lightning value
    trainer.current_epoch = 0

    # Mock pl_module
    pl_module = MagicMock(spec=LightningModule)
    pl_module.log = MagicMock()

    # Setup the pruner (collect parameters)
    pruner.setup(trainer, model, stage="fit")
    pruner.on_train_start(trainer, model)

    return pruner, trainer, model


def test_validation_disabled_during_ramp():
    """Test that validation is disabled (limit_val_batches=0) during ramp-
    up."""
    pruner, trainer, model = _make_pruner_and_mocks(
        final_amount=0.9, epochs_to_ramp=5
    )

    # Simulate epochs 0-3 (during ramp)
    for epoch in range(4):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)
        # During ramp-up, validation should be suppressed
        assert (
            trainer.limit_val_batches == 0
        ), f"Epoch {epoch}: Expected limit_val_batches=0 during ramp"


def test_validation_restored_after_target_reached():
    """Test that validation is restored when target sparsity is reached."""
    pruner, trainer, model = _make_pruner_and_mocks(
        final_amount=0.9, epochs_to_ramp=5
    )

    # Simulate full ramp-up
    for epoch in range(6):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)

    # After ramp completes (epochs >= epochs_to_ramp), validation should be restored
    assert (
        trainer.limit_val_batches == 1.0
    ), "Expected limit_val_batches=1.0 after target reached"


def test_original_limit_val_batches_preserved():
    """Test that custom limit_val_batches value is preserved and restored."""
    pruner, trainer, model = _make_pruner_and_mocks(
        final_amount=0.9, epochs_to_ramp=5
    )

    # Set custom value
    trainer.limit_val_batches = 0.5

    # Run through ramp
    for epoch in range(4):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)

    # During ramp, should be 0
    assert trainer.limit_val_batches == 0

    # Continue past ramp
    trainer.current_epoch = 5
    pruner.on_train_epoch_start(trainer, model)

    # After ramp, should restore to original custom value
    assert (
        trainer.limit_val_batches == 0.5
    ), "Expected original custom value 0.5 to be restored"


def test_no_validation_suppression_without_scheduled_pruning():
    """Test that validation is NOT suppressed when scheduled_pruning=False."""
    pruner = MagnitudePruner(
        pruning_fn="l1_unstructured",
        amount=0.9,
        scheduled_pruning=False,  # No scheduled pruning
        epochs_to_ramp=0,  # Must be 0 or None when scheduled_pruning=False
        verbose=0,
    )

    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))

    trainer = MagicMock(spec=Trainer)
    trainer.callbacks = []
    trainer.max_epochs = 20
    trainer.limit_val_batches = 1.0
    trainer.current_epoch = 0

    pruner.setup(trainer, model, stage="fit")
    pruner.on_train_start(trainer, model)

    # Run a few epochs
    for epoch in range(5):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)

    # Validation should NOT be suppressed
    assert (
        trainer.limit_val_batches == 1.0
    ), "Expected limit_val_batches unchanged without scheduled pruning"


def test_early_stopping_reset_on_target_reached():
    """Test that EarlyStopping state is reset when target sparsity is
    reached."""
    pruner, trainer, model = _make_pruner_and_mocks(
        final_amount=0.9, epochs_to_ramp=5
    )

    # Add EarlyStopping callback
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    early_stopping.wait_count = 2
    early_stopping.best_score = torch.tensor(0.5)
    trainer.callbacks = [early_stopping]

    # Run through ramp (epochs 0-4)
    # At epoch 4, target is NOT yet reached because schedule_map[4] < final_amount
    # We need epoch 5 to reach target
    for epoch in range(5):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)
        pl_module = MagicMock(spec=LightningModule)
        pl_module.log = MagicMock()
        pruner.on_train_epoch_end(trainer, pl_module)

    # At this point, EarlyStopping should still be getting reset each epoch (disabled mode)
    # Now continue to epoch 5 where target is actually reached
    trainer.current_epoch = 5
    pruner.on_train_epoch_start(trainer, model)
    pl_module = MagicMock(spec=LightningModule)
    pl_module.log = MagicMock()
    pruner.on_train_epoch_end(trainer, pl_module)

    # EarlyStopping should be reset when validation is re-enabled
    assert early_stopping.wait_count == 0, "Expected wait_count to be reset"
    assert early_stopping.best_score == torch.tensor(
        float("inf")
    ), "Expected best_score to be reset to inf (min mode)"


def test_model_checkpoint_save_top_k_restored():
    """Test that ModelCheckpoint save_top_k is properly restored."""
    pruner, trainer, model = _make_pruner_and_mocks(
        final_amount=0.9, epochs_to_ramp=5
    )

    # Add ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
    )
    trainer.callbacks = [checkpoint]

    # Run through ramp-up (should disable save_top_k)
    for epoch in range(3):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)
        pl_module = MagicMock(spec=LightningModule)
        pl_module.log = MagicMock()
        pruner.on_train_epoch_end(trainer, pl_module)

    # During ramp, save_top_k should be 0
    assert (
        checkpoint.save_top_k == 0
    ), "Expected save_top_k=0 during ramp (checkpoint suppression)"

    # Continue past ramp
    for epoch in range(5, 7):
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)
        pl_module = MagicMock(spec=LightningModule)
        pl_module.log = MagicMock()
        pruner.on_train_epoch_end(trainer, pl_module)

    # After target reached, save_top_k should be restored to original
    assert (
        checkpoint.save_top_k == 3
    ), "Expected save_top_k=3 restored after target reached"
