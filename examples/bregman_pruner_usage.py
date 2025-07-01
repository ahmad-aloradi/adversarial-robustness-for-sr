"""
Usage example for BregmanPruner with PyTorch Lightning.

This example shows how to integrate BregmanPruner with both structured pruning
(SafeModelPruning) and Bregman regularization for achieving target sparsity.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.prune import SafeModelPruning


class ExampleModel(pl.LightningModule):
    """Example model that integrates with BregmanPruner."""
    
    def __init__(self, use_bregman_regularization: bool = True):
        super().__init__()
        self.use_bregman_regularization = use_bregman_regularization
        
        # Example CNN architecture
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Store reference to BregmanPruner (will be set by callback)
        self.bregman_pruner = None
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Standard loss
        loss = nn.functional.cross_entropy(logits, y)
        
        # Add Bregman regularization if enabled
        if self.use_bregman_regularization and self.bregman_pruner is not None:
            reg_loss = self.bregman_pruner.get_regularization_loss(self)
            loss = loss + reg_loss
            
            # Log components
            self.log("train/ce_loss", nn.functional.cross_entropy(logits, y))
            self.log("train/reg_loss", reg_loss)
            self.log("train/sparsity", self.bregman_pruner.get_current_sparsity())
            self.log("train/lambda", self.bregman_pruner.get_current_lambda())
        
        self.log("train/loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def setup_pruning_callbacks(target_sparsity: float = 0.8):
    """
    Setup both structured pruning and Bregman regularization callbacks.
    
    Returns:
        List of callbacks for PyTorch Lightning trainer
    """
    # Parameters to prune (typically conv and linear layers)
    # In practice, this would be set programmatically or via config
    parameters_to_prune = None  # Auto-discover parameters
    
    # Structured pruning callback (removes weights permanently)
    safe_pruning = SafeModelPruning(
        amount=0.2,  # Remove 20% of weights each time
        scheduled_pruning=True,
        initial_amount=0.0,
        final_amount=target_sparsity * 0.8,  # Don't rely only on structured pruning
        epochs_to_ramp=10,
        collect_metrics=True,
        verbose=1,
        parameters_to_prune=parameters_to_prune,
    )
    
    # Bregman regularization callback (encourages sparsity through loss)
    bregman_pruning = BregmanPruner(
        target_sparsity=target_sparsity,
        initial_lambda=1.0,
        lambda_update_rate=0.001,
        update_frequency=100,
        parameters_to_prune=parameters_to_prune,
        verbose=1,
    )
    
    return [safe_pruning, bregman_pruning]


def create_trainer_with_pruning(target_sparsity: float = 0.8):
    """
    Create a PyTorch Lightning trainer with pruning callbacks.
    
    Args:
        target_sparsity: Target sparsity ratio (0.0 to 1.0)
    
    Returns:
        Configured trainer with pruning callbacks
    """
    callbacks = setup_pruning_callbacks(target_sparsity)
    
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        logger=True,  # Enable logging to see sparsity metrics
    )
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Create model and trainer
    model = ExampleModel(use_bregman_regularization=True)
    trainer = create_trainer_with_pruning(target_sparsity=0.8)
    
    # Connect BregmanPruner to model (for getting regularization loss)
    for callback in trainer.callbacks:
        if isinstance(callback, BregmanPruner):
            model.bregman_pruner = callback
            break
    
    # Create dummy data for demonstration
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),  # Images
            torch.randint(0, 10, (100,))  # Labels
        ),
        batch_size=16,
        shuffle=True
    )
    
    print("Starting training with BregmanPruner...")
    print("Expected logs:")
    print("- Epoch X: Sparsity of pruned modules = Y.YYY%")
    print("- Sparsity X.XXX% vs target 80.0% → Lambda ↗/↘ ... → ...")
    print("- Step XXX: Sparsity=X.XXX%, lambda=X.XXXX")
    print()
    
    # trainer.fit(model, train_loader)  # Uncomment to actually train