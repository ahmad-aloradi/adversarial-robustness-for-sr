"""
Tests for LambdaScheduler with sparsity smoothing functionality.
"""

import pytest
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler


class TestLambdaScheduler:
    """Test cases for LambdaScheduler sparsity smoothing."""
    
    def test_init_default_parameters(self):
        """Test scheduler initialization with default parameters."""
        scheduler = LambdaScheduler()
        
        assert scheduler.lambda_value == 1e-3
        assert scheduler.target_sparsity == 0.9
        assert scheduler.adjustment_factor == 1.1
        assert scheduler.min_lambda == 1e-6
        assert scheduler.max_lambda == 1e3
        assert scheduler._last_sparsity is None
        
    def test_init_custom_parameters(self):
        """Test scheduler initialization with custom parameters."""
        scheduler = LambdaScheduler(
            initial_lambda=1e-2,
            target_sparsity=0.8,
            adjustment_factor=1.2,
            min_lambda=1e-5,
            max_lambda=1e2
        )
        
        assert scheduler.lambda_value == 1e-2
        assert scheduler.target_sparsity == 0.8
        assert scheduler.adjustment_factor == 1.2
        assert scheduler.min_lambda == 1e-5
        assert scheduler.max_lambda == 1e2
        
    def test_step_normal_sparsity_below_target(self):
        """Test lambda increase when sparsity is below target."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Sparsity below target should increase lambda
        new_lambda = scheduler.step(0.7)
        
        assert new_lambda == 1.1  # 1.0 * 1.1
        assert scheduler.lambda_value == 1.1
        assert scheduler._last_sparsity == 0.7
        
    def test_step_normal_sparsity_above_target(self):
        """Test lambda decrease when sparsity is above target."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Sparsity above target should decrease lambda
        new_lambda = scheduler.step(0.95)
        
        assert new_lambda == 1.0 / 1.1  # 1.0 / 1.1
        assert scheduler.lambda_value == 1.0 / 1.1
        assert scheduler._last_sparsity == 0.95
        
    def test_step_sparsity_at_target(self):
        """Test lambda unchanged when sparsity equals target."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Sparsity at target should not change lambda
        new_lambda = scheduler.step(0.9)
        
        assert new_lambda == 1.0
        assert scheduler.lambda_value == 1.0
        assert scheduler._last_sparsity == 0.9
        
    def test_spurious_zero_filtering_with_valid_last_reading(self):
        """Test that spurious zero readings use last valid sparsity."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # First, establish a valid sparsity reading
        scheduler.step(0.8)  # Below target, lambda should increase
        assert scheduler.lambda_value == 1.1
        assert scheduler._last_sparsity == 0.8
        
        # Now send a spurious zero - should use last sparsity (0.8) instead
        # Since 0.8 < 0.9, lambda should increase again
        original_lambda = scheduler.lambda_value
        new_lambda = scheduler.step(0.0)
        
        assert new_lambda == original_lambda * 1.1  # Used last sparsity (0.8)
        assert scheduler._last_sparsity == 0.8  # Unchanged, zero not stored
        
    def test_spurious_zero_without_last_reading(self):
        """Test spurious zero when no last reading exists."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Send zero with no last reading - should use the zero
        new_lambda = scheduler.step(0.0)
        
        # 0.0 < 0.9, so lambda should increase
        assert new_lambda == 1.1
        assert scheduler._last_sparsity is None  # Zero not stored as last reading
        
    def test_zero_followed_by_valid_reading(self):
        """Test that valid readings after zeros are properly stored."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Start with a spurious zero
        scheduler.step(0.0)
        assert scheduler._last_sparsity is None
        
        # Follow with a valid reading
        scheduler.step(0.85)
        assert scheduler._last_sparsity == 0.85
        
    def test_lambda_clamping_min(self):
        """Test lambda is clamped to minimum value."""
        scheduler = LambdaScheduler(
            initial_lambda=1e-5,
            target_sparsity=0.5,
            min_lambda=1e-6
        )
        
        # Force lambda below minimum by having high sparsity
        scheduler.step(0.95)  # Much higher than target
        
        assert scheduler.lambda_value >= scheduler.min_lambda
        
    def test_lambda_clamping_max(self):
        """Test lambda is clamped to maximum value."""
        scheduler = LambdaScheduler(
            initial_lambda=1e2,
            target_sparsity=0.9,
            max_lambda=1e3,
            adjustment_factor=10.0  # Large factor for quick growth
        )
        
        # Force lambda above maximum by having low sparsity repeatedly
        for _ in range(10):
            scheduler.step(0.1)  # Much lower than target
            
        assert scheduler.lambda_value <= scheduler.max_lambda
        
    def test_get_lambda(self):
        """Test get_lambda method returns current lambda value."""
        scheduler = LambdaScheduler(initial_lambda=2.5)
        
        assert scheduler.get_lambda() == 2.5
        
        scheduler.step(0.7)  # Should change lambda
        assert scheduler.get_lambda() == scheduler.lambda_value
        
    def test_reset(self):
        """Test reset clears the last sparsity reading."""
        scheduler = LambdaScheduler()
        
        # Establish some state
        scheduler.step(0.8)
        assert scheduler._last_sparsity == 0.8
        
        # Reset should clear last sparsity
        scheduler.reset()
        assert scheduler._last_sparsity is None
        
    def test_get_state(self):
        """Test get_state returns complete scheduler state."""
        scheduler = LambdaScheduler(
            initial_lambda=1.5,
            target_sparsity=0.85,
            adjustment_factor=1.2
        )
        
        scheduler.step(0.75)  # Establish some state
        
        state = scheduler.get_state()
        
        expected_keys = {
            'lambda_value', 'target_sparsity', 'last_sparsity',
            'adjustment_factor', 'min_lambda', 'max_lambda'
        }
        assert set(state.keys()) == expected_keys
        assert state['last_sparsity'] == 0.75
        assert state['target_sparsity'] == 0.85
        
    def test_multiple_spurious_zeros_with_smoothing(self):
        """Test multiple consecutive spurious zeros use smoothing consistently."""
        scheduler = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
        
        # Establish valid reading
        scheduler.step(0.8)
        lambda_after_valid = scheduler.lambda_value
        
        # Multiple spurious zeros should all use the same last valid reading
        for _ in range(3):
            scheduler.step(0.0)
            
        # Each zero should have used 0.8 sparsity (below target),
        # so lambda should have increased 3 times
        expected_lambda = lambda_after_valid * (1.1 ** 3)
        assert abs(scheduler.lambda_value - expected_lambda) < 1e-10
        assert scheduler._last_sparsity == 0.8  # Still the original valid reading
        
    def test_effective_sparsity_method(self):
        """Test _get_effective_sparsity method directly."""
        scheduler = LambdaScheduler()
        
        # Without last reading, should return input
        assert scheduler._get_effective_sparsity(0.0) == 0.0
        assert scheduler._get_effective_sparsity(0.5) == 0.5
        
        # Set a last reading
        scheduler._last_sparsity = 0.8
        
        # Non-zero should return as-is
        assert scheduler._get_effective_sparsity(0.7) == 0.7
        
        # Zero should return last reading
        assert scheduler._get_effective_sparsity(0.0) == 0.8