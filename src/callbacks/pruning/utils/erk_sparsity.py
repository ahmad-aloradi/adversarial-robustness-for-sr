"""Layerwise sparsity distribution from RigL's Erdos-Renyi family.

RigL (arXiv:1911.11134, sec 3.1) sets each layer's density (1 - sparsity) from
its shape, then scales all layers by one scalar so the kept-parameter count
matches a global budget. Two scores:

- ER  : (fan_in + fan_out) / (fan_in * fan_out)          -- ignores kernel size
- ERK : (fan_in + fan_out + sum_k) / (fan_in * fan_out * prod_k)
        -- the kernel enters the denominator, so larger-kernel (more-parameter)
        layers get a smaller score => higher sparsity. For Linear (no kernel)
        ERK collapses back to ER.

This module is pure: it operates on shape records (LayerShape), not torch
modules, so it is unit-testable in isolation. The budget is defined over the
prunable layers passed in -- callers must exclude never-pruned params (BN,
bias) so the target is interpreted over the prunable-weight set.

Example:
    >>> shapes = [
    ...     LayerShape("a", fan_in=64, fan_out=128, kernel_dims=(3,), n_params=64*128*3),
    ...     LayerShape("b", fan_in=128, fan_out=128, kernel_dims=(), n_params=128*128),
    ... ]
    >>> dens = solve_erk_densities(shapes, target_sparsity=0.9, mode="erk")
    >>> all(0.0 < d <= 1.0 for d in dens.values())
    True
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Kept-parameter budget equality tolerance, relative to the total prunable
# count: the epsilon-solve must land the achieved kept-count on the budget.
ERK_BUDGET_TOL: float = 1e-6


@dataclass
class LayerShape:
    """Shape record for one prunable weight tensor.

    fan_in is weight.shape[1] (already in-channels-per-group for grouped
    convs), fan_out is weight.shape[0], kernel_dims is weight.shape[2:] (empty
    for Linear), n_params is weight.numel().
    """

    name: str
    fan_in: int
    fan_out: int
    kernel_dims: Tuple[int, ...]
    n_params: int


def raw_density(shape: LayerShape, mode: str) -> float:
    """Unnormalized ER/ERK density score for one layer (before epsilon
    scaling).

    Inputs:
        shape: the layer's LayerShape.
        mode: "er" or "erk".
    Output:
        A positive score; larger => the layer should stay denser.
    """
    fan_in = shape.fan_in
    fan_out = shape.fan_out
    assert fan_in > 0 and fan_out > 0, (
        f"fan_in/fan_out must be positive, got "
        f"fan_in={fan_in}, fan_out={fan_out} (layer={shape.name})"
    )

    if mode == "er":
        return (fan_in + fan_out) / (fan_in * fan_out)

    if mode == "erk":
        # Empty kernel_dims (Linear) => sum 0, prod 1, so ERK == ER.
        kernel_sum = sum(shape.kernel_dims)
        kernel_prod = math.prod(shape.kernel_dims) if shape.kernel_dims else 1
        numerator = fan_in + fan_out + kernel_sum
        denominator = fan_in * fan_out * kernel_prod
        return numerator / denominator

    raise ValueError(f"Unknown mode {mode!r}. Expected one of ['er', 'erk'].")


def solve_erk_densities(
    shapes: List[LayerShape],
    target_sparsity: float,
    mode: str,
) -> Dict[str, float]:
    """Per-layer density (1 - sparsity) realizing the ER/ERK distribution.

    Scales raw scores by a scalar epsilon so the total kept count equals
    (1 - target_sparsity) * total_prunable. Any layer whose scaled density
    exceeds 1.0 is clamped to dense, removed from the pool, its full param
    count subtracted from the budget, and epsilon is re-solved over the rest;
    this repeats until no layer exceeds 1.0 (RigL's dense-clamp redistribution).

    Inputs:
        shapes: the prunable layers the budget is defined over (non-empty).
        target_sparsity: target over these layers, in [0.0, 1.0).
        mode: "er" or "erk".
    Output:
        {layer_name: density in (0.0, 1.0]} for every input layer.
    """
    mode = mode.lower()
    if mode not in ("er", "erk"):
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of ['er', 'erk']."
        )
    assert (
        0.0 <= target_sparsity < 1.0
    ), f"target_sparsity must be in [0.0, 1.0), got {target_sparsity}"
    assert len(shapes) > 0, "solve_erk_densities needs at least one layer"

    total_prunable = sum(s.n_params for s in shapes)
    kept_budget = (1.0 - target_sparsity) * total_prunable
    scores = {s.name: raw_density(s, mode) for s in shapes}
    n_params = {s.name: s.n_params for s in shapes}

    densities: Dict[str, float] = {}
    pool = [s.name for s in shapes]
    remaining_budget = kept_budget

    while pool:
        score_mass = sum(scores[name] * n_params[name] for name in pool)
        assert score_mass > 0, "ERK score mass collapsed to zero"
        epsilon = remaining_budget / score_mass

        clamped = [name for name in pool if epsilon * scores[name] > 1.0]
        if not clamped:
            for name in pool:
                densities[name] = epsilon * scores[name]
            break

        for name in clamped:
            densities[name] = 1.0
            remaining_budget -= n_params[name]
            pool.remove(name)

    achieved_kept = sum(densities[name] * n_params[name] for name in densities)
    assert (
        abs(achieved_kept - kept_budget) <= ERK_BUDGET_TOL * total_prunable
    ), (
        f"ERK budget mismatch: kept {achieved_kept:.1f} != "
        f"target {kept_budget:.1f} (tol={ERK_BUDGET_TOL} * {total_prunable})"
    )
    assert all(
        0.0 < d <= 1.0 + ERK_BUDGET_TOL for d in densities.values()
    ), f"ERK produced a density outside (0, 1]: {densities}"
    return densities


def erk_initial_and_target_sparsity(
    shapes: List[LayerShape],
    target_sparsity: float,
    mode: str,
    initial_target_sparsity: float = None,
) -> Dict[str, Dict[str, float]]:
    """Per-layer target and initial sparsity for the ER/ERK distribution.

    The target sparsity is 1 - density at ``target_sparsity``. When
    ``initial_target_sparsity`` is None the initial sparsity equals the target,
    so each layer starts AT its ERK target (RigL keeps per-layer sparsity
    fixed). When set, the initial mask is solved at that denser budget so the
    random start follows the same ERK shape but begins sparser.

    Inputs:
        shapes: the prunable layers (non-empty).
        target_sparsity: final per-layer target budget, in [0.0, 1.0).
        mode: "er" or "erk".
        initial_target_sparsity: optional denser start budget, or None.
    Output:
        {layer_name: {"target_sparsity": s, "initial_sparsity": s0}}.
    """
    target_densities = solve_erk_densities(shapes, target_sparsity, mode)
    if initial_target_sparsity is None:
        initial_densities = target_densities
    else:
        initial_densities = solve_erk_densities(
            shapes, initial_target_sparsity, mode
        )

    result: Dict[str, Dict[str, float]] = {}
    for shape in shapes:
        result[shape.name] = {
            "target_sparsity": 1.0 - target_densities[shape.name],
            "initial_sparsity": 1.0 - initial_densities[shape.name],
        }
    return result


if __name__ == "__main__":
    demo_shapes = [
        LayerShape(
            "conv_k3",
            fan_in=64,
            fan_out=128,
            kernel_dims=(3,),
            n_params=64 * 128 * 3,
        ),
        LayerShape(
            "conv_k5",
            fan_in=64,
            fan_out=128,
            kernel_dims=(5,),
            n_params=64 * 128 * 5,
        ),
        LayerShape(
            "linear",
            fan_in=128,
            fan_out=256,
            kernel_dims=(),
            n_params=128 * 256,
        ),
    ]
    for demo_mode in ("er", "erk"):
        dens = solve_erk_densities(
            demo_shapes, target_sparsity=0.9, mode=demo_mode
        )
        total = sum(s.n_params for s in demo_shapes)
        kept = sum(dens[s.name] * s.n_params for s in demo_shapes)
        print(f"mode={demo_mode}: overall sparsity={1 - kept / total:.4f}")
        for s in demo_shapes:
            print(f"  {s.name}: sparsity={1 - dens[s.name]:.4f}")
