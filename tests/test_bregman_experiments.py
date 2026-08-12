"""Integration tests for Bregman mini-training behavior.

This test suite verifies that the full Bregman pipeline (optimizer + pruner + scheduler)
works correctly together in abbreviated training loops:
- Mini-training produces nonzero sparsity
- No NaN or Inf values in parameters
- Per-layer sparsity is non-degenerate
- Lambda evolves correctly during training
- Scheduled target mode works end-to-end
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.callbacks.pruning.bregman import bregman_pruner as bp
from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.bregman.quantile_lambda_scheduler import (
    QuantileLambdaScheduler,
)
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
            prune_first_layer=True,
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
                        "sparsity_rate": self.optimizer_config.get(
                            "initial_sparsity", 0.0
                        ),
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
        optimizer_param_groups = (
            self.pruning_manager.get_optimizer_param_groups()
        )

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
    scheduler_cls=LambdaScheduler,
    scheduler_kwargs=None,
):
    """Run mini Bregman training loop and return metrics.

    Args:
        scheduler_cls: LambdaScheduler (default) or QuantileLambdaScheduler.
            Both accept target_sparsity/initial_sparsity/initial_lambda, so
            they're interchangeable here.
        scheduler_kwargs: Extra kwargs merged in for the chosen scheduler.

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
    scheduler = scheduler_cls(
        target_sparsity=target_sparsity,
        initial_sparsity=initial_sparsity,
        initial_lambda=0.1,
        **(scheduler_kwargs or {}),
    )

    # Create pruner
    pruner = BregmanPruner(
        sparsity_threshold=1e-12,
        verbose=0,
        lambda_scheduler=scheduler,
        target_sparsity=target_sparsity,
    )

    # Initialize optimizer (needed for pruner setup)
    optimizer = pl_module.configure_optimizers()

    # Create mock trainer
    trainer = Mock()
    trainer.optimizers = [optimizer]
    trainer.ckpt_path = None
    trainer.callbacks = []
    trainer.lr_scheduler_configs = []
    trainer.limit_val_batches = 1.0
    trainer.num_training_batches = num_batches_per_epoch
    trainer.estimated_stepping_batches = num_epochs * num_batches_per_epoch
    trainer.callback_metrics = {}  # on_train_epoch_end writes sparsity here

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
            pruner.on_train_batch_end(
                trainer, pl_module, None, batch, batch_idx
            )

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
    )

    # Final sparsity should be between target and initial
    # (model starts at 0.99, moves toward 0.7)
    final_sparsity = sparsity_per_epoch[-1]
    assert (
        0.3 < final_sparsity < 0.99
    ), f"Expected final sparsity in (0.3, 0.99), got {final_sparsity}"


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
        assert torch.all(
            torch.isfinite(param)
        ), "Parameter contains NaN or Inf values"


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
            weight_sparsity = compute_sparsity(
                [module.weight], threshold=1e-12
            )

            # Layer should not be 100% sparse (fully collapsed)
            assert (
                weight_sparsity < 0.99
            ), f"Layer {name} is nearly fully sparse (collapsed): {weight_sparsity}"

            # Layer should not be 0% sparse (no regularization applied)
            # Starting from 0.85 initial sparsity, should retain some sparsity
            assert (
                weight_sparsity > 0.1
            ), f"Layer {name} has too little sparsity: {weight_sparsity}"


@pytest.mark.slow
def test_bregman_lambda_evolves_during_training():
    """Lambda values evolve during training to drive sparsity toward target."""
    sparsity_per_epoch, lambda_per_step, _, _ = _run_mini_bregman_training(
        target_sparsity=0.5,
        initial_sparsity=0.99,  # Start too sparse
        num_epochs=10,
        num_batches_per_epoch=20,
    )

    # Lambda should decrease over training
    # (model is too sparse, need to reduce regularization)
    initial_lambda = lambda_per_step[0]
    final_lambda = lambda_per_step[-1]

    assert (
        final_lambda < initial_lambda
    ), f"Expected lambda to decrease, but {initial_lambda} -> {final_lambda}"


@pytest.mark.slow
def test_quantile_mini_training_hits_target_sparsity():
    """QuantileLambdaScheduler lands pruned sparsity in a much tighter band
    than LambdaScheduler's -- exact by construction, not by convergence."""
    _, _, _, model = _run_mini_bregman_training(
        target_sparsity=0.7,
        initial_sparsity=0.99,
        num_epochs=10,
        num_batches_per_epoch=20,
        scheduler_cls=QuantileLambdaScheduler,
    )

    pruned_sparsity = compute_sparsity(
        [model.fc1.weight, model.fc2.weight], threshold=1e-12
    )
    assert pruned_sparsity == pytest.approx(0.7, abs=0.02)


@pytest.mark.slow
def test_quantile_mini_training_no_nan():
    """Mini-training under QuantileLambdaScheduler produces no NaN or Inf."""
    _, _, final_params, _ = _run_mini_bregman_training(
        target_sparsity=0.7,
        initial_sparsity=0.0,
        num_epochs=5,
        num_batches_per_epoch=20,
        scheduler_cls=QuantileLambdaScheduler,
    )

    for param in final_params:
        assert torch.all(
            torch.isfinite(param)
        ), "Parameter contains NaN or Inf values"


# =============================================================================
# Target feasibility (validation can only reopen if the target is reachable)
# =============================================================================


def _make_pruner_for_fit(target_sparsity):
    """Real model + optimizer + pruner wired to run on_fit_start.

    SimpleMLP has 1800 prunable Linear weights of 1840 trainable params, so the
    achievable overall-sparsity ceiling is ~0.978.
    """
    model = SimpleMLP()
    pl_module = MiniBregmanModule(
        model, {"lr": 0.01, "lambda": 0.5, "initial_sparsity": 0.0}
    )
    optimizer = pl_module.configure_optimizers()
    scheduler = LambdaScheduler(
        target_sparsity=target_sparsity,
        initial_sparsity=0.0,
        initial_lambda=0.1,
    )
    pruner = BregmanPruner(
        verbose=0,
        lambda_scheduler=scheduler,
        target_sparsity=target_sparsity,
    )

    trainer = Mock()
    trainer.optimizers = [optimizer]
    trainer.ckpt_path = None
    trainer.callbacks = []
    trainer.lr_scheduler_configs = []
    trainer.limit_val_batches = 1.0
    trainer.num_training_batches = 20
    trainer.estimated_stepping_batches = 200
    trainer.callback_metrics = {}
    return pruner, trainer, pl_module


def test_target_above_overall_ceiling_raises(monkeypatch):
    """Steering on overall, a target above the prunable fraction is refused."""
    monkeypatch.setattr(bp, "WHICH_SPARSITY_PERCENTAGE", "overall")
    pruner, trainer, pl_module = _make_pruner_for_fit(target_sparsity=0.99)
    with pytest.raises(AssertionError, match="ceiling"):
        pruner.on_fit_start(trainer, pl_module)


def test_pruned_steering_has_no_overall_ceiling(monkeypatch):
    """The regularized groups can reach 1.0, so no ceiling applies to them."""
    monkeypatch.setattr(bp, "WHICH_SPARSITY_PERCENTAGE", "pruned")
    pruner, trainer, pl_module = _make_pruner_for_fit(target_sparsity=0.99)
    pruner.on_fit_start(trainer, pl_module)
    assert pruner._target_sparsity == 0.99


def test_feasible_target_passes_fit_start():
    """A target below the ceiling initializes without raising."""
    pruner, trainer, pl_module = _make_pruner_for_fit(target_sparsity=0.7)
    pruner.on_fit_start(trainer, pl_module)
    assert pruner._target_sparsity == 0.7
