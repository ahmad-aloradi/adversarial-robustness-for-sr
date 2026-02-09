"""Verification tests for magnitude pruning correctness.

Tests cover PRUNE-01 and PRUNE-03 requirements:
- Sparsity calculation accuracy with known-sparsity tensors
- Threshold handling (1e-12 boundary)
- Mask binary property (all values are 0.0 or 1.0)
- No fully collapsed layers (no layer >= 99% sparsity)
- PyTorch API consistency (MagnitudePruner matches direct torch.nn.utils.prune)
- Monotonic sparsity increase during scheduled ramp
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import LightningModule, Trainer

from src.callbacks.pruning.prune import MagnitudePruner
from src.callbacks.pruning.shared_prune_utils import compute_sparsity


def test_sparsity_calculation_known_values():
    """Test compute_sparsity returns correct values for known-sparsity
    tensors."""
    # Create a model with known sparsity
    model = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
    )

    # Test at 25% sparsity
    with torch.no_grad():
        # Zero out exactly 25% of first layer's weight elements
        weight = model[0].weight
        total_elements = weight.numel()
        num_zeros = int(0.25 * total_elements)
        flat = weight.flatten()
        flat[:num_zeros] = 0.0

    params = [(model[0], "weight")]
    sparsity = compute_sparsity(params)
    assert abs(sparsity - 0.25) < 1e-6, f"Expected 0.25, got {sparsity}"

    # Test at 50% sparsity
    with torch.no_grad():
        weight = model[1].weight
        total_elements = weight.numel()
        num_zeros = int(0.50 * total_elements)
        flat = weight.flatten()
        flat[:num_zeros] = 0.0

    params = [(model[1], "weight")]
    sparsity = compute_sparsity(params)
    assert abs(sparsity - 0.50) < 1e-6, f"Expected 0.50, got {sparsity}"

    # Test at 75% sparsity (both layers)
    with torch.no_grad():
        for module in model:
            weight = module.weight
            total_elements = weight.numel()
            num_zeros = int(0.75 * total_elements)
            flat = weight.flatten()
            flat[:num_zeros] = 0.0

    params = [(model[0], "weight"), (model[1], "weight")]
    sparsity = compute_sparsity(params)
    assert abs(sparsity - 0.75) < 1e-6, f"Expected 0.75, got {sparsity}"


def test_sparsity_calculation_threshold():
    """Test that the 1e-12 threshold correctly classifies near-zero values."""
    model = nn.Conv1d(16, 32, kernel_size=3, padding=1)

    with torch.no_grad():
        weight = model.weight
        total = weight.numel()

        # Set some values to exactly 1e-12, some below, some above
        weight.zero_()
        flat = weight.flatten()

        # Split into thirds
        third = total // 3

        flat[:third] = 1e-13  # Below threshold → should count as zero
        flat[third : 2 * third] = 1e-12  # At threshold → should count as zero
        flat[2 * third :] = 1e-11  # Above threshold → non-zero

    params = [(model, "weight")]
    sparsity = compute_sparsity(params, threshold=1e-12)

    # 2/3 of values should be counted as zero
    expected = 2.0 / 3.0
    assert (
        abs(sparsity - expected) < 0.01
    ), f"Expected ~{expected:.2f}, got {sparsity:.2f}"


def test_masks_are_binary_after_pruning():
    """Test that all pruning masks contain only 0.0 or 1.0 values."""
    for amount in [0.3, 0.7, 0.9]:
        model = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
        )

        pruner = MagnitudePruner(
            pruning_fn="ln_structured",
            amount=amount,
            scheduled_pruning=False,
            epochs_to_ramp=0,
            pruning_dim=1,
            pruning_norm=1,
            verbose=0,
        )

        trainer = MagicMock(spec=Trainer)
        trainer.callbacks = []
        trainer.max_epochs = 10
        trainer.current_epoch = 0

        pruner.setup(trainer, model, stage="fit")
        pruner.on_train_epoch_start(trainer, model)

        # Check all masks are binary
        for module in model:
            assert hasattr(
                module, "weight_mask"
            ), f"Expected weight_mask at amount={amount}"
            mask = module.weight_mask
            unique = torch.unique(mask)
            assert all(
                v in [0.0, 1.0] for v in unique.tolist()
            ), f"Non-binary mask values at amount={amount}: {unique.tolist()}"


def test_no_fully_collapsed_layers():
    """Test that no individual layer exceeds 99% sparsity when pruning at
    90%."""
    model = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.Conv1d(64, 32, kernel_size=3, padding=1),
    )

    pruner = MagnitudePruner(
        pruning_fn="ln_structured",
        amount=0.9,
        scheduled_pruning=False,
        epochs_to_ramp=0,
        pruning_dim=1,
        pruning_norm=1,
        verbose=0,
    )

    trainer = MagicMock(spec=Trainer)
    trainer.callbacks = []
    trainer.max_epochs = 10
    trainer.current_epoch = 0

    pruner.setup(trainer, model, stage="fit")
    pruner.on_train_epoch_start(trainer, model)

    # Check each layer's sparsity
    for i, module in enumerate(model):
        layer_params = [(module, "weight")]
        layer_sparsity = compute_sparsity(layer_params)
        assert (
            layer_sparsity < 0.99
        ), f"Layer {i} collapsed with {layer_sparsity:.2%} sparsity"


def test_pruner_matches_pytorch_reference():
    """Test that MagnitudePruner produces same masks as direct PyTorch
    ln_structured call."""
    # Create two identical models
    torch.manual_seed(42)
    model1 = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
    )

    torch.manual_seed(42)
    model2 = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
    )

    amount = 0.5

    # Apply pruning via MagnitudePruner
    pruner = MagnitudePruner(
        pruning_fn="ln_structured",
        amount=amount,
        scheduled_pruning=False,
        epochs_to_ramp=0,
        pruning_dim=1,
        pruning_norm=1,
        verbose=0,
    )

    trainer = MagicMock(spec=Trainer)
    trainer.callbacks = []
    trainer.max_epochs = 10
    trainer.current_epoch = 0

    pruner.setup(trainer, model1, stage="fit")
    pruner.on_train_epoch_start(trainer, model1)

    # Apply pruning via direct PyTorch API
    for module in model2:
        pytorch_prune.ln_structured(
            module, name="weight", amount=amount, n=1, dim=1
        )

    # Compare masks
    for i, (m1, m2) in enumerate(zip(model1, model2)):
        assert hasattr(m1, "weight_mask"), f"Model1 layer {i} missing mask"
        assert hasattr(m2, "weight_mask"), f"Model2 layer {i} missing mask"

        mask1 = m1.weight_mask
        mask2 = m2.weight_mask

        assert torch.equal(
            mask1, mask2
        ), f"Layer {i} masks differ between MagnitudePruner and PyTorch reference"


def test_sparsity_monotonically_increases_during_ramp():
    """Test that sparsity is non-decreasing at each epoch during scheduled
    ramp."""
    model = nn.Sequential(
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
    )

    pruner = MagnitudePruner(
        pruning_fn="ln_structured",
        amount=0.9,
        scheduled_pruning=True,
        epochs_to_ramp=10,
        schedule_type="linear",
        pruning_dim=1,
        pruning_norm=1,
        verbose=0,
    )

    trainer = MagicMock(spec=Trainer)
    trainer.callbacks = []
    trainer.max_epochs = 20
    trainer.limit_val_batches = 1.0  # Required for validation suppression
    trainer.current_epoch = 0

    pruner.setup(trainer, model, stage="fit")
    pruner.on_train_start(trainer, model)

    sparsities = []
    for epoch in range(12):  # Go beyond ramp to verify it holds
        trainer.current_epoch = epoch
        pruner.on_train_epoch_start(trainer, model)

        params = [(model[0], "weight"), (model[1], "weight")]
        sparsity = compute_sparsity(params)
        sparsities.append(sparsity)

        if epoch > 0:
            assert (
                sparsity >= sparsities[epoch - 1] - 1e-6
            ), f"Sparsity decreased at epoch {epoch}: {sparsities[epoch-1]:.4f} -> {sparsity:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
