"""Integration tests for Bregman mini-training behavior.

This test suite verifies that the full Bregman pipeline (optimizer + pruner + scheduler)
works correctly together in abbreviated training loops:
- Mini-training produces nonzero sparsity
- No NaN or Inf values in parameters
- Per-layer sparsity is non-degenerate
- Lambda evolves correctly during training
- Scheduled target mode works end-to-end
"""
import pytest
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from unittest.mock import Mock

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.pruning_manager import PruningManager


# =============================================================================
# Mini-training framework
# =============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MiniBregmanModule(LightningModule):
    """Minimal LightningModule for Bregman testing."""

    def __init__(self, model, optimizer_config):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.pruning_manager = None

        # Logging params for callback compatibility
        self.logging_params = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": False,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = ((output - y) ** 2).mean()
        return loss

    def configure_optimizers(self):
        # Setup pruning manager first
        self.pruning_manager = PruningManager(
            pl_module=self,
            group_configs=[
                {
                    "name": "linear_weights",
                    "layer_types": ["torch.nn.Linear"],
                    "param_names": ["weight"],
                    "optimizer_settings": {
                        "reg": RegL1(lamda=self.optimizer_config["lambda"]),
                        "lambda_scale": 1.0,
                    },
                    "pruning_config": {
                        "pruning_type": "unstructured",
                        "sparsity_rate": self.optimizer_config.get("initial_sparsity", 0.0),
                    },
                },
                {
                    "name": "other",
                    "is_fallback": True,
                    "optimizer_settings": {},
                    "pruning_config": {
                        "pruning_type": "unstructured",
                        "sparsity_rate": 0.0,
                    },
                },
            ],
        )

        # Get optimizer param groups from manager
        optimizer_param_groups = self.pruning_manager.get_optimizer_param_groups()

        # Create optimizer
        optimizer = AdaBreg(
            optimizer_param_groups,
            lr=self.optimizer_config["lr"],
            delta=1.0,
        )

        return optimizer


def _run_mini_bregman_training(
    target_sparsity=0.7,
    initial_sparsity=0.99,
    num_epochs=10,
    num_batches_per_epoch=20,
    use_ema=False,
    schedule_type=None,
    initial_target=None,
    final_target=None,
    epochs_to_ramp=None,
):
    """Run mini Bregman training loop and return metrics.

    Returns:
        sparsity_per_epoch: List of sparsity values per epoch
        lambda_per_step: List of lambda values per step
        final_params: Final model parameters
        model: The trained model
    """
    torch.manual_seed(42)

    # Create model
    model = SimpleMLP()

    # Configure optimizer
    optimizer_config = {
        "lr": 0.01,
        "lambda": 0.5,
        "initial_sparsity": initial_sparsity,
    }

    # Create Lightning module
    pl_module = MiniBregmanModule(model, optimizer_config)

    # Create lambda scheduler
    if schedule_type is not None:
        scheduler = LambdaScheduler(
            schedule_type=schedule_type,
            initial_target_sparsity=initial_target,
            final_target_sparsity=final_target,
            epochs_to_ramp=epochs_to_ramp,
            initial_lambda=0.1,
            use_ema=use_ema,
        )
    else:
        scheduler = LambdaScheduler(
            target_sparsity=target_sparsity,
            initial_lambda=0.1,
            use_ema=use_ema,
        )

    # Create pruner
    pruner = BregmanPruner(
        sparsity_threshold=1e-12,
        verbose=0,
        lambda_scheduler=scheduler,
    )

    # Initialize optimizer (needed for pruner setup)
    optimizer = pl_module.configure_optimizers()

    # Create mock trainer
    trainer = Mock()
    trainer.optimizers = [optimizer]
    trainer.ckpt_path = None
    trainer.callbacks = []
    trainer.limit_val_batches = 1.0

    # Initialize pruner
    pruner.on_fit_start(trainer, pl_module)

    # Track metrics
    sparsity_per_epoch = []
    lambda_per_step = []

    # Training loop
    for epoch in range(num_epochs):
        trainer.current_epoch = epoch

        # on_train_epoch_start
        pruner.on_train_epoch_start(trainer, pl_module)

        for batch_idx in range(num_batches_per_epoch):
            # Generate fake batch
            x = torch.randn(8, 50)
            y = torch.randn(8, 10)
            batch = (x, y)

            # Forward/backward
            loss = pl_module.training_step(batch, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # on_train_batch_end
            trainer.global_step = epoch * num_batches_per_epoch + batch_idx
            pruner.on_train_batch_end(trainer, pl_module, None, batch, batch_idx)

            # Record lambda
            lambda_per_step.append(scheduler.get_lambda())

        # on_train_epoch_end
        pruner.on_train_epoch_end(trainer, pl_module)

        # Record sparsity
        current_sparsity = pruner._overall_sparsity()
        sparsity_per_epoch.append(current_sparsity)

    return sparsity_per_epoch, lambda_per_step, list(model.parameters()), model


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.slow
def test_bregman_mini_training_produces_sparsity():
    """Mini-training with BregmanPruner produces nonzero sparsity."""
    sparsity_per_epoch, _, final_params, _ = _run_mini_bregman_training(
        target_sparsity=0.7,
        initial_sparsity=0.99,  # Start very sparse (inverse-scale)
        num_epochs=10,
        num_batches_per_epoch=20,
        use_ema=False,
    )

    # Final sparsity should be between target and initial
    # (model starts at 0.99, moves toward 0.7)
    final_sparsity = sparsity_per_epoch[-1]
    assert 0.3 < final_sparsity < 0.99, (
        f"Expected final sparsity in (0.3, 0.99), got {final_sparsity}"
    )


@pytest.mark.slow
def test_bregman_mini_training_no_nan():
    """Mini-training produces no NaN or Inf values."""
    _, _, final_params, _ = _run_mini_bregman_training(
        target_sparsity=0.7,
        initial_sparsity=0.0,
        num_epochs=5,
        num_batches_per_epoch=20,
    )

    # Check all parameters for NaN/Inf
    for param in final_params:
        assert torch.all(torch.isfinite(param)), (
            f"Parameter contains NaN or Inf values"
        )


@pytest.mark.slow
def test_bregman_per_layer_sparsity_not_degenerate():
    """Per-layer sparsity is non-degenerate (no layer fully collapsed)."""
    _, _, final_params, model = _run_mini_bregman_training(
        target_sparsity=0.7,
        initial_sparsity=0.85,  # Start somewhat sparse to avoid collapse
        num_epochs=10,
        num_batches_per_epoch=20,
    )

    # Check per-layer sparsity for linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_sparsity = compute_sparsity([module.weight], threshold=1e-12)

            # Layer should not be 100% sparse (fully collapsed)
            assert weight_sparsity < 0.99, (
                f"Layer {name} is nearly fully sparse (collapsed): {weight_sparsity}"
            )

            # Layer should not be 0% sparse (no regularization applied)
            # Starting from 0.85 initial sparsity, should retain some sparsity
            assert weight_sparsity > 0.1, (
                f"Layer {name} has too little sparsity: {weight_sparsity}"
            )


@pytest.mark.slow
def test_bregman_lambda_evolves_during_training():
    """Lambda values evolve during training to drive sparsity toward target."""
    sparsity_per_epoch, lambda_per_step, _, _ = _run_mini_bregman_training(
        target_sparsity=0.5,
        initial_sparsity=0.99,  # Start too sparse
        num_epochs=10,
        num_batches_per_epoch=20,
        use_ema=False,
    )

    # Lambda should decrease over training
    # (model is too sparse, need to reduce regularization)
    initial_lambda = lambda_per_step[0]
    final_lambda = lambda_per_step[-1]

    assert final_lambda < initial_lambda, (
        f"Expected lambda to decrease, but {initial_lambda} -> {final_lambda}"
    )


@pytest.mark.slow
def test_bregman_scheduled_mode_mini_training():
    """Scheduled target mode ramps target sparsity correctly."""
    sparsity_per_epoch, lambda_per_step, _, _ = _run_mini_bregman_training(
        schedule_type="linear",
        initial_target=0.5,
        final_target=0.9,
        epochs_to_ramp=5,
        initial_sparsity=0.6,  # Start somewhat sparse
        num_epochs=10,
        num_batches_per_epoch=20,
        use_ema=False,
    )

    # Verify final sparsity is significant
    final_sparsity = sparsity_per_epoch[-1]
    assert final_sparsity > 0.4, (
        f"Expected final sparsity > 0.4, got {final_sparsity}"
    )

    # Sparsity should be in reasonable range after scheduled ramp
    # Target goes from 0.5 to 0.9, so final should be closer to 0.9
    assert 0.4 < final_sparsity < 0.95, (
        f"Expected sparsity in (0.4, 0.95), got {final_sparsity}"
    )
