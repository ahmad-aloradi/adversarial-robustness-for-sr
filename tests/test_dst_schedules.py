"""Unit tests for the dynamic-sparse-training step schedules.

Pure math, no torch. Covers:
- Cosine drop fraction: starts at alpha, reaches end_value at the end of the
  ramp, holds it afterwards, and decreases monotonically in between.
- The end_value argument reproduces both upstream conventions exactly
  (RigL decays to 0 and stops; GraNet/ITOP decay to 0.005 over the whole run).
- Cubic prune rate: hits both endpoints, is monotone, and clamps outside the
  ramp window.
- Out-of-contract arguments raise.
"""
import math

import pytest

from src.callbacks.pruning.dst_schedules import (
    cosine_drop_fraction,
    cubic_prune_rate,
)

TOTAL = 1000


# =============================================================================
# 1. Cosine drop fraction
# =============================================================================


def test_cosine_starts_at_alpha():
    assert cosine_drop_fraction(0, TOTAL, alpha=0.3) == pytest.approx(0.3)


def test_cosine_hits_end_value_at_the_end_and_holds():
    """Past end_fraction * total_steps the value is exactly end_value."""
    end_step = int(0.75 * TOTAL)
    for step in (end_step, end_step + 1, TOTAL, 10 * TOTAL):
        assert cosine_drop_fraction(step, TOTAL) == 0.0, f"step={step}"


def test_cosine_is_monotone_decreasing():
    values = [cosine_drop_fraction(s, TOTAL) for s in range(0, TOTAL, 5)]
    assert all(
        b <= a for a, b in zip(values, values[1:])
    ), "drop fraction must never increase"


def test_cosine_midpoint_is_half_alpha():
    """Cos(pi/2) = 0, so the halfway point of the ramp sits at alpha / 2."""
    alpha = 0.3
    half = 0.5 * 0.75 * TOTAL
    assert cosine_drop_fraction(
        int(half), TOTAL, alpha=alpha
    ) == pytest.approx(alpha / 2)


def test_cosine_matches_rigl_closed_form():
    """RigL's f_decay: alpha / 2 * (1 + cos(pi * t / T_end)), 0 past T_end."""
    alpha, end_fraction = 0.3, 0.75
    end_step = end_fraction * TOTAL
    for step in (0, 37, 199, 500, 749):
        expected = 0.5 * alpha * (1 + math.cos(math.pi * step / end_step))
        assert cosine_drop_fraction(
            step, TOTAL, alpha=alpha, end_fraction=end_fraction
        ) == pytest.approx(expected), f"step={step}"


def test_cosine_granet_convention_floors_at_eta_min():
    """GraNet decays over the whole run to 0.005 and never reaches 0."""
    kwargs = dict(alpha=0.5, end_fraction=1.0, end_value=0.005)
    assert cosine_drop_fraction(0, TOTAL, **kwargs) == pytest.approx(0.5)
    assert cosine_drop_fraction(TOTAL, TOTAL, **kwargs) == pytest.approx(0.005)
    assert cosine_drop_fraction(TOTAL - 1, TOTAL, **kwargs) > 0.005


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_steps": 0},
        {"step": -1},
        {"end_fraction": 0.0},
        {"end_fraction": 1.5},
        {"end_value": 0.4},  # above alpha=0.3
    ],
)
def test_cosine_out_of_contract_raises(kwargs):
    call = {"step": 10, "total_steps": TOTAL, **kwargs}
    with pytest.raises(AssertionError):
        cosine_drop_fraction(**call)


# =============================================================================
# 2. Cubic prune rate
# =============================================================================


def test_cubic_endpoints():
    """Dense start, GraNet's 0.10 final density."""
    assert cubic_prune_rate(0, 100, 1.0, 0.1) == pytest.approx(0.0)
    assert cubic_prune_rate(100, 100, 1.0, 0.1) == pytest.approx(0.9)


def test_cubic_is_monotone_increasing():
    values = [cubic_prune_rate(i, 100, 1.0, 0.1) for i in range(101)]
    assert all(
        b >= a for a, b in zip(values, values[1:])
    ), "target sparsity must never decrease"


def test_cubic_clamps_after_the_window():
    """At the start it holds the start sparsity; past the end, the target."""
    assert cubic_prune_rate(0, 100, 0.5, 0.1) == pytest.approx(0.5)
    assert cubic_prune_rate(500, 100, 0.5, 0.1) == pytest.approx(0.9)


def test_cubic_matches_granet_closed_form():
    init_density, final_density = 1.0, 0.1
    for i in (0, 13, 55, 99, 100):
        progress = i / 100
        prune_decay = (1 - progress) ** 3
        expected = (1 - init_density) + (init_density - final_density) * (
            1 - prune_decay
        )
        assert cubic_prune_rate(
            i, 100, init_density, final_density
        ) == pytest.approx(expected), f"update={i}"


def test_cubic_front_loads_the_pruning():
    """The cubic removes most of the budget early; half the ramp is well past
    half the sparsity."""
    half = cubic_prune_rate(50, 100, 1.0, 0.1)
    assert half > 0.5 * 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"final_update": 0},  # ramp with no width
        {"init_density": 0.1, "final_density": 0.5},  # final denser than init
        {"final_density": 0.0},
        {"init_density": 1.5},
    ],
)
def test_cubic_out_of_contract_raises(kwargs):
    call = {
        "update_index": 10,
        "final_update": 100,
        "init_density": 1.0,
        "final_density": 0.1,
        **kwargs,
    }
    with pytest.raises(AssertionError):
        cubic_prune_rate(**call)
