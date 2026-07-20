"""Unit tests for Bregman optimizer and regularizer mathematical correctness.

This test suite verifies that:
- RegL1 proximal operator implements correct soft-thresholding
- RegL1L2Conv implements correct group sparsity
- AdaBreg and LinBreg optimizers induce measurable sparsity
- Optimizers correctly update subgradient state and parameters
"""
import pytest
import torch
import torch.nn as nn

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg, LinBreg
from src.callbacks.pruning.bregman.bregman_regularizers import (
    RegL1,
    RegL1L2Conv,
    RegNone,
)
from src.callbacks.pruning.shared_prune_utils import compute_sparsity

# =============================================================================
# RegL1 proximal operator tests
# =============================================================================


def test_regl1_prox_soft_thresholding():
    """Verify RegL1.prox implements soft-thresholding correctly.

    Soft-thresholding formula: sign(x) * max(|x| - delta*lambda, 0)
    """
    reg = RegL1(lamda=0.2)
    delta = 1.0

    # Test input with known values
    x = torch.tensor([0.5, -0.3, 0.1, -0.8])

    result = reg.prox(x, delta)

    # Expected: threshold = delta * lambda = 1.0 * 0.2 = 0.2
    # Elements with |x| <= 0.2 become 0, others shrink by 0.2
    expected = torch.tensor(
        [
            0.3,  # 0.5 - 0.2 = 0.3
            -0.1,  # -0.3 + 0.2 = -0.1
            0.0,  # |0.1| <= 0.2, becomes 0
            -0.6,  # -0.8 + 0.2 = -0.6
        ]
    )

    assert torch.allclose(result, expected, atol=1e-6)


def test_regl1_prox_zeros_small_weights():
    """Verify that weights smaller than delta*lambda are driven to exactly
    zero."""
    reg = RegL1(lamda=0.5)
    delta = 2.0

    # Threshold = 2.0 * 0.5 = 1.0
    # All values with |x| <= 1.0 should become 0
    x = torch.tensor([0.5, -0.8, 1.0, -1.0, 0.3, 1.5, -1.5])

    result = reg.prox(x, delta)

    # Check that small values are exactly zero
    assert result[0] == 0.0  # |0.5| < 1.0
    assert result[1] == 0.0  # |-0.8| < 1.0
    assert result[2] == 0.0  # |1.0| == 1.0
    assert result[3] == 0.0  # |-1.0| == 1.0
    assert result[4] == 0.0  # |0.3| < 1.0

    # Large values should be non-zero
    assert result[5] != 0.0  # |1.5| > 1.0
    assert result[6] != 0.0  # |-1.5| > 1.0


def test_regl1_subgrad_matches_sign():
    """Verify RegL1.sub_grad returns lambda * sign(v)."""
    reg = RegL1(lamda=0.5)

    v = torch.tensor([1.0, -2.0, 0.5, -0.3, 0.0])

    result = reg.sub_grad(v)

    expected = 0.5 * torch.sign(v)

    assert torch.allclose(result, expected)


# =============================================================================
# RegL1L2Conv proximal operator tests
# =============================================================================


def test_regl1l2conv_prox_group_sparsity():
    """Verify group lasso behavior -- entire rows are zeroed when L2 norm is
    below threshold."""
    reg = RegL1L2Conv(lamda=1.0)
    delta = 1.0

    # Create a 4x3 weight tensor (4 filters/groups, 3 features per group)
    # Make one row have small values (should be zeroed)
    x = torch.tensor(
        [
            [2.0, 3.0, 4.0],  # Large L2 norm
            [0.1, 0.05, 0.08],  # Small L2 norm
            [5.0, 6.0, 7.0],  # Large L2 norm
            [0.2, 0.15, 0.1],  # Small L2 norm
        ]
    )

    result = reg.prox(x, delta)

    # Row 1 and 3 (small L2 norm) should be exactly zero
    assert torch.allclose(result[1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(result[3], torch.zeros(3), atol=1e-6)

    # Row 0 and 2 (large L2 norm) should be non-zero
    assert torch.all(result[0] != 0.0)
    assert torch.all(result[2] != 0.0)


def test_regl1l2conv_prox_preserves_large_groups():
    """Verify that groups with large L2 norm are only scaled, not zeroed."""
    reg = RegL1L2Conv(lamda=0.1)
    delta = 1.0

    # Create a weight tensor with large values
    x = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [15.0, 25.0, 35.0],
        ]
    )

    result = reg.prox(x, delta)

    # All values should be non-zero (scaled down but not zeroed)
    assert torch.all(result != 0.0)

    # Result should be smaller than input (scaled)
    assert torch.all(torch.abs(result) < torch.abs(x))


# =============================================================================
# AdaBreg optimizer tests
# =============================================================================


def test_adabreg_single_step_updates_subgrad():
    """Verify AdaBreg performs one step correctly: initializes subgradient and updates parameters."""
    # Create simple 2-layer network
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 2),
    )

    reg = RegL1(lamda=0.1)
    optimizer = AdaBreg(model.parameters(), lr=0.01, reg=reg, delta=1.0)

    # Forward/backward pass with random data
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    # Before step: check state is empty
    for param in model.parameters():
        assert len(optimizer.state[param]) == 0

    # Perform one optimizer step
    optimizer.step()

    # After step: verify state is initialized
    for param in model.parameters():
        state = optimizer.state[param]
        assert "step" in state
        assert "sub_grad" in state
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["step"] == 1
        assert state["sub_grad"].shape == param.shape


def test_adabreg_induces_sparsity():
    """Train a simple model with AdaBreg + RegL1 and verify sparsity
    increases."""
    torch.manual_seed(42)

    # Simple linear model
    model = nn.Linear(20, 10)

    # Use RegL1 with reasonable lambda
    reg = RegL1(lamda=0.5)
    optimizer = AdaBreg(model.parameters(), lr=0.01, reg=reg, delta=1.0)

    # Train for 100 steps on random data
    for _ in range(100):
        x = torch.randn(8, 20)
        y = torch.randn(8, 10)

        output = model(x)
        loss = ((output - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Measure final sparsity
    final_sparsity = compute_sparsity(
        list(model.parameters()), threshold=1e-12
    )

    # Assert: sparsity should be nonzero (at least 10%)
    assert (
        final_sparsity > 0.1
    ), f"Expected sparsity > 0.1, got {final_sparsity}"


def test_adabreg_no_sparsity_with_regnone():
    """Verify that AdaBreg with RegNone produces no sparsity (prox is
    identity)."""
    torch.manual_seed(42)

    model = nn.Linear(20, 10)

    # Use RegNone (no regularization)
    reg = RegNone()
    optimizer = AdaBreg(model.parameters(), lr=0.01, reg=reg, delta=1.0)

    # Train for 100 steps
    for _ in range(100):
        x = torch.randn(8, 20)
        y = torch.randn(8, 10)

        output = model(x)
        loss = ((output - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Measure final sparsity
    final_sparsity = compute_sparsity(
        list(model.parameters()), threshold=1e-12
    )

    # Assert: sparsity should be near zero (< 1%)
    assert (
        final_sparsity < 0.01
    ), f"Expected sparsity < 0.01, got {final_sparsity}"


# =============================================================================
# LinBreg optimizer tests
# =============================================================================


def test_linbreg_single_step():
    """Verify LinBreg performs one step correctly."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 2),
    )

    reg = RegL1(lamda=0.1)
    optimizer = LinBreg(model.parameters(), lr=0.01, reg=reg, delta=1.0)

    # Forward/backward pass
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    # Perform one step
    optimizer.step()

    # Verify state is initialized
    for param in model.parameters():
        state = optimizer.state[param]
        assert "step" in state
        assert "sub_grad" in state
        assert state["step"] == 1


def test_linbreg_induces_sparsity():
    """Train a simple model with LinBreg + RegL1 and verify sparsity
    increases."""
    torch.manual_seed(42)

    model = nn.Linear(20, 10)

    # LinBreg needs higher lr or stronger lambda to induce sparsity
    reg = RegL1(lamda=1.0)
    optimizer = LinBreg(model.parameters(), lr=0.1, reg=reg, delta=1.0)

    # Train for 100 steps
    for _ in range(100):
        x = torch.randn(8, 20)
        y = torch.randn(8, 10)

        output = model(x)
        loss = ((output - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Measure final sparsity
    final_sparsity = compute_sparsity(
        list(model.parameters()), threshold=1e-12
    )

    # Assert: sparsity should be nonzero (LinBreg is slower than AdaBreg)
    assert (
        final_sparsity > 0.05
    ), f"Expected sparsity > 0.05, got {final_sparsity}"


# =============================================================================
# Cross-implementation consistency
# =============================================================================


def test_adabreg_linbreg_both_induce_sparsity():
    """Run both AdaBreg and LinBreg on same problem and verify both produce
    sparse solutions."""
    torch.manual_seed(42)

    # Create two identical models
    model_ada = nn.Linear(20, 10)
    model_lin = nn.Linear(20, 10)

    # Copy weights to make them start from same initialization
    with torch.no_grad():
        model_lin.weight.copy_(model_ada.weight)
        model_lin.bias.copy_(model_ada.bias)

    # Create optimizers with same lambda (LinBreg gets higher lr)
    reg_ada = RegL1(lamda=0.5)
    reg_lin = RegL1(lamda=0.5)

    optimizer_ada = AdaBreg(
        model_ada.parameters(), lr=0.01, reg=reg_ada, delta=1.0
    )
    optimizer_lin = LinBreg(
        model_lin.parameters(), lr=0.05, reg=reg_lin, delta=1.0
    )

    # Train both for 200 steps on same data
    torch.manual_seed(100)  # Reset seed for data generation
    for _ in range(200):
        x = torch.randn(8, 20)
        y = torch.randn(8, 10)

        # Train AdaBreg model
        output_ada = model_ada(x)
        loss_ada = ((output_ada - y) ** 2).mean()
        optimizer_ada.zero_grad()
        loss_ada.backward()
        optimizer_ada.step()

        # Train LinBreg model (same data)
        output_lin = model_lin(x)
        loss_lin = ((output_lin - y) ** 2).mean()
        optimizer_lin.zero_grad()
        loss_lin.backward()
        optimizer_lin.step()

    # Measure final sparsity for both
    sparsity_ada = compute_sparsity(
        list(model_ada.parameters()), threshold=1e-12
    )
    sparsity_lin = compute_sparsity(
        list(model_lin.parameters()), threshold=1e-12
    )

    # Assert: both should achieve nonzero sparsity
    # AdaBreg converges faster due to adaptive moments
    assert (
        sparsity_ada > 0.1
    ), f"AdaBreg sparsity {sparsity_ada} should be > 0.1"
    # LinBreg is slower but should still achieve meaningful sparsity
    assert (
        sparsity_lin > 0.05
    ), f"LinBreg sparsity {sparsity_lin} should be > 0.05"

    # Both produce sparse solutions (exact values differ due to different update rules)


# =============================================================================
# Subgradient reinitialization tests
# =============================================================================


@pytest.mark.parametrize("OptimizerClass", [LinBreg, AdaBreg])
def test_lambda_change_preserves_subgradients(OptimizerClass):
    """Lambda updates must NOT modify subgradients.

    The scheduler only sets reg.lamda — subgradients preserve accumulated
    gradient history. The new lambda takes effect on the next prox step,
    shifting the sparsity threshold without destroying Bregman's multi-step
    gradient accumulation.
    """
    torch.manual_seed(0)
    model = nn.Linear(20, 10)

    old_lambda = 0.5
    new_lambda = 1.2
    delta = 1.0
    reg = RegL1(lamda=old_lambda)
    optimizer = OptimizerClass(model.parameters(), lr=0.01, reg=reg, delta=delta)

    # Train a few steps to populate optimizer state and drift subgradients
    for _ in range(30):
        x = torch.randn(8, 20)
        y = torch.randn(8, 10)
        output = model(x)
        loss = ((output - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Snapshot subgradients before lambda change
    subgrads_before = {}
    for p in model.parameters():
        state = optimizer.state.get(p)
        if state is not None and "sub_grad" in state:
            subgrads_before[p] = state["sub_grad"].clone()

    # Update lambda (as _step_lambda_scheduler does — only changes reg.lamda)
    reg.lamda = new_lambda

    # Subgradients must be unchanged
    for p, v_before in subgrads_before.items():
        v_after = optimizer.state[p]["sub_grad"]
        assert torch.equal(v_before, v_after), (
            "Subgradients must not change when lambda is updated"
        )


@pytest.mark.parametrize("OptimizerClass", [LinBreg, AdaBreg])
def test_lambda_increase_prunes_via_prox_threshold(OptimizerClass):
    """Increasing lambda prunes weights by raising the prox threshold.

    When lambda increases without subgradient modification, the next prox step
    applies a higher threshold: p_new = max(p + δ(λ_old − λ_new) − δ·lr·grad, 0).
    This naturally drives small weights to zero.
    """
    torch.manual_seed(42)
    model = nn.Linear(50, 20)

    delta = 1.0
    reg = RegL1(lamda=0.1)
    optimizer = OptimizerClass(model.parameters(), lr=0.01, reg=reg, delta=delta)

    # Train to get a model with some weights near zero
    for _ in range(50):
        x = torch.randn(8, 50)
        y = torch.randn(8, 20)
        output = model(x)
        loss = ((output - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Count non-zero weights before lambda increase
    nonzero_before = sum(
        (p.data != 0).sum().item() for p in model.parameters()
    )

    # Increase lambda significantly
    reg.lamda = 5.0

    # Run one optimizer step with the new (higher) lambda
    x = torch.randn(8, 50)
    y = torch.randn(8, 20)
    output = model(x)
    loss = ((output - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Count non-zero weights after — should have fewer (more pruned)
    nonzero_after = sum(
        (p.data != 0).sum().item() for p in model.parameters()
    )

    assert nonzero_after < nonzero_before, (
        f"Lambda increase should prune weights: before={nonzero_before}, after={nonzero_after}"
    )
