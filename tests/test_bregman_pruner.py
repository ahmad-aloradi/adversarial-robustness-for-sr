"""
Test for BregmanPruner sparsity calculations.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
import pytest
from unittest.mock import MagicMock

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestBregmanPruner:
    """Test BregmanPruner functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        pruner = BregmanPruner(target_sparsity=0.8)
        assert pruner.target_sparsity == 0.8
        assert pruner.current_lambda == 1.0
        assert pruner._last_computed_sparsity == 0.0
    
    def test_invalid_target_sparsity(self):
        """Test validation of target_sparsity parameter."""
        with pytest.raises(ValueError):
            BregmanPruner(target_sparsity=1.5)
        
        with pytest.raises(ValueError):
            BregmanPruner(target_sparsity=-0.1)
    
    def test_sparsity_calculation_no_pruning(self):
        """Test sparsity calculation on unpruned model."""
        model = SimpleCNN()
        pruner = BregmanPruner()
        
        # Initially, sparsity should be very low (only natural zeros)
        sparsity = pruner._compute_current_sparsity(model)
        assert 0.0 <= sparsity <= 0.1  # Should be minimal natural zeros
    
    def test_sparsity_calculation_with_pytorch_pruning(self):
        """Test sparsity calculation with PyTorch pruning applied."""
        model = SimpleCNN()
        pruner = BregmanPruner()
        
        # Apply structured pruning to conv1
        pytorch_prune.l1_unstructured(model.conv1, name='weight', amount=0.5)
        
        # Compute sparsity - should reflect the 50% pruning
        sparsity = pruner._compute_current_sparsity(model)
        
        # Should be close to 50% sparsity for conv1, but less overall due to other unpruned layers
        assert 0.1 <= sparsity <= 0.8  # Should be significant but not exactly 50%
    
    def test_sparsity_calculation_with_manual_zeros(self):
        """Test sparsity calculation with manually zeroed weights."""
        model = SimpleCNN()
        pruner = BregmanPruner()
        
        # Manually zero out half of conv1 weights
        with torch.no_grad():
            weight = model.conv1.weight
            num_zeros = weight.numel() // 2
            weight.view(-1)[:num_zeros] = 0.0
        
        sparsity = pruner._compute_current_sparsity(model)
        
        # Should reflect the manual zeros
        assert sparsity > 0.1  # Should be significant
    
    def test_lambda_update_mechanism(self):
        """Test lambda update based on sparsity."""
        pruner = BregmanPruner(target_sparsity=0.8, lambda_update_rate=0.01)
        initial_lambda = pruner.current_lambda
        
        # Test when current sparsity is below target
        pruner._update_lambda(current_sparsity=0.3)
        # Lambda should increase
        assert pruner.current_lambda > initial_lambda
        
        # Reset and test when current sparsity is above target
        pruner.current_lambda = initial_lambda
        pruner._update_lambda(current_sparsity=0.9)
        # Lambda should decrease
        assert pruner.current_lambda < initial_lambda
    
    def test_parameter_validation_and_caching(self):
        """Test parameter validation and caching mechanism."""
        model = SimpleCNN()
        pruner = BregmanPruner()
        
        # First call should validate and cache
        params1 = pruner._validate_and_cache_parameters(model)
        
        # Second call should return cached result
        params2 = pruner._validate_and_cache_parameters(model)
        
        assert params1 == params2
        assert len(params1) == 3  # conv1.weight, conv2.weight, fc.weight
        
        # Verify parameters are correct
        param_names = [(type(module).__name__, param_name) for module, param_name in params1]
        expected = [('Conv2d', 'weight'), ('Conv2d', 'weight'), ('Linear', 'weight')]
        assert param_names == expected
    
    def test_regularization_loss_computation(self):
        """Test Bregman regularization loss computation."""
        model = SimpleCNN()
        pruner = BregmanPruner(initial_lambda=2.0)
        
        # Compute regularization loss
        reg_loss = pruner._apply_bregman_regularization(model)
        
        # Should be a tensor
        assert isinstance(reg_loss, torch.Tensor)
        # Should be positive (L1 norm is always positive)
        assert reg_loss.item() > 0
        
        # Test with different lambda
        pruner.current_lambda = 0.5
        reg_loss_lower = pruner._apply_bregman_regularization(model)
        
        # Lower lambda should give lower regularization loss
        assert reg_loss_lower.item() < reg_loss.item()
    
    def test_callback_integration(self):
        """Test integration with PyTorch Lightning trainer mock."""
        model = SimpleCNN()
        pruner = BregmanPruner(update_frequency=1, verbose=1)
        
        # Mock trainer and pl_module
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.global_step = 100
        trainer.logger = None
        
        # Test setup
        pruner.setup(trainer, model, "fit")
        
        # Test epoch start
        pruner.on_train_epoch_start(trainer, model)
        
        # Test batch end (should trigger update due to frequency=1)
        initial_sparsity = pruner.get_current_sparsity()
        pruner.on_train_batch_end(trainer, model, None, None, 0)
        
        # Check that sparsity was computed
        assert pruner.get_current_sparsity() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])