""" Implementation of: "Rigging the Lottery: Making All Tickets Winners"
(Evci, Gale, Menick, Castro, Elsen, ICML 2020)
https://arxiv.org/abs/1911.11134

Idea (sec 3.1): score each layer's density from its shape, then scale every
score by one epsilon so the kept count hits a global budget. ER scores
(fan_in + fan_out) / (fan_in * fan_out); ERK adds the kernel to both terms, so
larger-kernel layers score lower and come out sparser.

**The budget is over the layers you pass in!** Callers must leave out the
params no method sparsifies (norm weights, biases), or the target means
something other than what every other method reports.

Run it with::

    python src/train.py experiment=img/pruning_rigl datamodule=datasets/mnist

Inspect the solve alone with::

    python src/callbacks/pruning/utils/erk_sparsity.py
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Budget tolerance relative to the total prunable count: the epsilon-solve must land the kept-count on the budget.
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

    >>> shapes = [
    ...     LayerShape("a", fan_in=64, fan_out=128, kernel_dims=(3,), n_params=64*128*3),
    ...     LayerShape("b", fan_in=128, fan_out=128, kernel_dims=(), n_params=128*128),
    ... ]
    >>> dens = solve_erk_densities(shapes, target_sparsity=0.9, mode="erk")
    >>> all(0.0 < d <= 1.0 for d in dens.values())
    True
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
        assert score_mass > 0, "ERK score mass is zero"
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


if __name__ == "__main__":
    demo_shapes = [
        LayerShape("conv_k3", 64, 128, (3,), 64 * 128 * 3),
        LayerShape("conv_k5", 64, 128, (5,), 64 * 128 * 5),
        LayerShape("linear", 128, 256, (), 128 * 256),
    ]
    total = sum(s.n_params for s in demo_shapes)
    for demo_mode in ("er", "erk"):
        dens = solve_erk_densities(demo_shapes, 0.9, demo_mode)
        kept = sum(dens[s.name] * s.n_params for s in demo_shapes)
        print(f"mode={demo_mode}: overall sparsity={1 - kept / total:.4f}")
        for s in demo_shapes:
            print(f"  {s.name}: sparsity={1 - dens[s.name]:.4f}")
