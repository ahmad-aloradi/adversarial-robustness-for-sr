"""Unit tests for the ERK/ER layerwise sparsity solve (RigL arXiv:1911.11134).

Pure math, no torch model. Covers:
- ER closed-form on two layers.
- ERK assigns higher sparsity to the larger-kernel layer.
- The kept-parameter budget equals (1 - target) * total for both modes.
- The dense-clamp redistribution: a layer whose raw density exceeds 1 is
  pinned dense and its freed budget raises the others; the total budget holds.
- An unknown mode raises.
"""
import math

import pytest

from src.callbacks.pruning.utils.erk_sparsity import (
    ERK_BUDGET_TOL,
    LayerShape,
    raw_density,
    solve_erk_densities,
)


def _kept(shapes, densities):
    return sum(densities[s.name] * s.n_params for s in shapes)


# =============================================================================
# 1. ER closed form
# =============================================================================


def test_er_two_layer_closed_form():
    """ER density ratio matches (1/fan_in + 1/fan_out) ratio between layers."""
    a = LayerShape(
        "a", fan_in=64, fan_out=64, kernel_dims=(), n_params=64 * 64
    )
    b = LayerShape(
        "b", fan_in=256, fan_out=256, kernel_dims=(), n_params=256 * 256
    )
    dens = solve_erk_densities([a, b], target_sparsity=0.9, mode="er")

    # The small layer keeps a higher density than the big one.
    assert dens["a"] > dens["b"]
    # Density ratio equals the raw-score ratio (neither layer is clamped here).
    score_ratio = raw_density(a, "er") / raw_density(b, "er")
    assert math.isclose(dens["a"] / dens["b"], score_ratio, rel_tol=1e-9)


# =============================================================================
# 2. ERK accounts for the kernel
# =============================================================================


def test_erk_larger_kernel_gets_more_sparsity():
    """Same fan_in/out, bigger kernel => smaller score => higher sparsity."""
    k3 = LayerShape(
        "k3", fan_in=64, fan_out=128, kernel_dims=(3,), n_params=64 * 128 * 3
    )
    k5 = LayerShape(
        "k5", fan_in=64, fan_out=128, kernel_dims=(5,), n_params=64 * 128 * 5
    )
    dens = solve_erk_densities([k3, k5], target_sparsity=0.9, mode="erk")
    assert (1 - dens["k5"]) > (1 - dens["k3"])


def test_erk_linear_equals_er():
    """For a kernel-free (Linear) layer set, ERK and ER coincide."""
    a = LayerShape(
        "a", fan_in=64, fan_out=128, kernel_dims=(), n_params=64 * 128
    )
    b = LayerShape(
        "b", fan_in=128, fan_out=256, kernel_dims=(), n_params=128 * 256
    )
    er = solve_erk_densities([a, b], target_sparsity=0.8, mode="er")
    erk = solve_erk_densities([a, b], target_sparsity=0.8, mode="erk")
    for name in ("a", "b"):
        assert math.isclose(er[name], erk[name], rel_tol=1e-12)


# =============================================================================
# 3. Budget equality
# =============================================================================


@pytest.mark.parametrize("mode", ["er", "erk"])
@pytest.mark.parametrize("target", [0.5, 0.9, 0.99])
def test_budget_equality(mode, target):
    """Kept count lands on (1 - target) * total within tolerance."""
    shapes = [
        LayerShape(
            "c0",
            fan_in=80,
            fan_out=512,
            kernel_dims=(5,),
            n_params=80 * 512 * 5,
        ),
        LayerShape(
            "c1",
            fan_in=512,
            fan_out=512,
            kernel_dims=(3,),
            n_params=512 * 512 * 3,
        ),
        LayerShape(
            "c2",
            fan_in=512,
            fan_out=1536,
            kernel_dims=(1,),
            n_params=512 * 1536,
        ),
        LayerShape(
            "fc", fan_in=1536, fan_out=192, kernel_dims=(), n_params=1536 * 192
        ),
    ]
    total = sum(s.n_params for s in shapes)
    dens = solve_erk_densities(shapes, target_sparsity=target, mode=mode)
    kept = _kept(shapes, dens)
    assert abs(kept - (1 - target) * total) <= ERK_BUDGET_TOL * total


# =============================================================================
# 4. Dense-clamp redistribution
# =============================================================================


def test_dense_clamp_redistributes_budget():
    """A tiny layer whose raw density would exceed 1 is pinned dense; its freed
    budget raises the other layers, and the total budget still holds."""
    tiny = LayerShape("tiny", fan_in=2, fan_out=2, kernel_dims=(), n_params=4)
    big = LayerShape(
        "big",
        fan_in=512,
        fan_out=512,
        kernel_dims=(3,),
        n_params=512 * 512 * 3,
    )
    shapes = [tiny, big]
    total = sum(s.n_params for s in shapes)

    # Low target => large epsilon => the tiny layer's density saturates at 1.0.
    dens = solve_erk_densities(shapes, target_sparsity=0.3, mode="erk")
    assert dens["tiny"] == 1.0  # pinned dense

    # Budget still met overall.
    assert (
        abs(_kept(shapes, dens) - (1 - 0.3) * total) <= ERK_BUDGET_TOL * total
    )

    # Without the tiny layer competing, the big layer is denser than a naive
    # epsilon * score would give if tiny had not been clamped.
    assert 0.0 < dens["big"] <= 1.0


# =============================================================================
# 5. Fail-loud
# =============================================================================


def test_unknown_mode_raises():
    shapes = [
        LayerShape("a", fan_in=8, fan_out=8, kernel_dims=(), n_params=64)
    ]
    with pytest.raises(ValueError, match="Unknown mode"):
        solve_erk_densities(shapes, target_sparsity=0.5, mode="bogus")


@pytest.mark.parametrize("bad_target", [1.0, 1.5, -0.1])
def test_out_of_range_target_raises(bad_target):
    shapes = [
        LayerShape("a", fan_in=8, fan_out=8, kernel_dims=(), n_params=64)
    ]
    with pytest.raises(AssertionError):
        solve_erk_densities(shapes, target_sparsity=bad_target, mode="erk")


# =============================================================================
# 6. Degenerate target: 0.0 keeps everything dense
# =============================================================================


@pytest.mark.parametrize("mode", ["er", "erk"])
def test_target_zero_is_all_dense(mode):
    shapes = [
        LayerShape(
            "a",
            fan_in=64,
            fan_out=128,
            kernel_dims=(3,),
            n_params=64 * 128 * 3,
        ),
        LayerShape(
            "b", fan_in=128, fan_out=128, kernel_dims=(), n_params=128 * 128
        ),
    ]
    dens = solve_erk_densities(shapes, target_sparsity=0.0, mode=mode)
    for d in dens.values():
        assert math.isclose(d, 1.0, rel_tol=ERK_BUDGET_TOL)
