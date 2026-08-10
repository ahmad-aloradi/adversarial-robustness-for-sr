"""The two schedules of dynamic sparse training, from RigL (Evci et al., ICML
2020) and GraNet (Liu et al., NeurIPS 2021).

Idea: ``cosine_drop_fraction`` is how much of a layer's mask is redrawn at an
update; ``cubic_prune_rate`` is GraNet's dense-to-sparse ramp, the sparsity the
mask holds after an update. Magnitude pruning's per-epoch ramp is
``scheduler.py``.

**The two count different things!** ``cosine_drop_fraction`` takes a training
step, ``cubic_prune_rate`` a mask-update index.

Print both with::

    python src/callbacks/pruning/dst_schedules.py
"""
import math


def cosine_drop_fraction(
    step: int,
    total_steps: int,
    alpha: float = 0.3,
    end_fraction: float = 0.75,
    end_value: float = 0.0,
) -> float:
    """Fraction of a layer's active weights redrawn at ``step``.

    Cosine decay from ``alpha`` at step 0 to ``end_value`` at
    ``end_fraction * total_steps``, holding ``end_value`` after that. Both
    upstreams are exact in these last two arguments: RigL decays to 0 over the
    first 75% and then stops updating; GraNet/ITOP decay over the whole run to
    0.005 and never stop.

    >>> cosine_drop_fraction(0, 1000)
    0.3
    >>> round(cosine_drop_fraction(375, 1000), 4)
    0.15
    >>> cosine_drop_fraction(750, 1000), cosine_drop_fraction(900, 1000)
    (0.0, 0.0)
    >>> round(cosine_drop_fraction(1000, 1000, alpha=0.5, end_fraction=1.0, end_value=0.005), 4)
    0.005
    """
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
    assert step >= 0, f"step must be non-negative, got {step}"
    assert (
        0.0 < end_fraction <= 1.0
    ), f"end_fraction must be in (0, 1], got {end_fraction}"
    assert (
        0.0 <= end_value <= alpha
    ), f"end_value must be in [0, alpha={alpha}], got {end_value}"

    end_step = end_fraction * total_steps
    if step >= end_step:
        return end_value
    # ported from google-research/rigl@d39fc7d rigl/sparse_optimizers_base.py:get_drop_fraction ('cosine' branch)
    decay = 0.5 * (1.0 + math.cos(math.pi * step / end_step))
    return end_value + (alpha - end_value) * decay


def cubic_prune_rate(
    update_index: int,
    final_update: int,
    init_density: float,
    final_density: float,
) -> float:
    """Sparsity the mask should hold after mask update ``update_index``.

    The ramp runs from update 0 to ``final_update``, clamped to its endpoints
    outside that window. GraNet writes the cubic on the *prune rate* rather
    than on sparsity, and counts mask updates rather than steps.

    >>> cubic_prune_rate(0, 100, 1.0, 0.1)
    0.0
    >>> cubic_prune_rate(100, 100, 1.0, 0.1)
    0.9
    >>> round(cubic_prune_rate(50, 100, 1.0, 0.1), 4)
    0.7875
    >>> cubic_prune_rate(200, 100, 1.0, 0.1)
    0.9
    """
    assert (
        final_update > 0
    ), f"final_update must be positive, got {final_update}"
    assert 0.0 < final_density <= init_density <= 1.0, (
        f"need 0 < final_density <= init_density <= 1, got "
        f"final_density={final_density}, init_density={init_density}"
    )

    progress = min(max(update_index / final_update, 0.0), 1.0)
    # ported from VITA-Group/GraNet@f338a24 CIFAR/sparselearning/core.py:Masking.pruning
    prune_decay = (1.0 - progress) ** 3
    return (1.0 - init_density) + (init_density - final_density) * (
        1.0 - prune_decay
    )


if __name__ == "__main__":
    total, every = 1000, 100
    final_update = int(total * 0.6875) // every
    print("step  update  rigl_drop  granet_drop  granet_sparsity")
    for step in range(0, total + 1, every):
        rigl = cosine_drop_fraction(step, total)
        granet = cosine_drop_fraction(
            step, total, alpha=0.5, end_fraction=1.0, end_value=0.005
        )
        update = step // every
        sparsity = cubic_prune_rate(update, final_update, 1.0, 0.1)
        print(
            f"{step:5d}  {update:6d}  {rigl:9.4f}  {granet:11.4f}  "
            f"{sparsity:15.4f}"
        )
