"""Composite learning-rate schedule: flat, then cosine to 0.

Idea: hold the base lr for ``constant_epochs``, then cosine-anneal it to 0 over
the rest of the run.

Why: GraNet's plasticity finding (Liu et al., NeurIPS 2021, Fig. 2) shows that
connection regeneration stops helping once the lr has decayed, so the
pruning+regrowth ramp (``final_prune_epoch``) needs the lr held up through it.

Run it with::

    python -m src.utils.lr_schedulers
"""
from typing import Union

import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    SequentialLR,
)


def constant_then_cosine(
    optimizer: torch.optim.Optimizer, constant_epochs: int, max_epochs: int
) -> Union[CosineAnnealingLR, SequentialLR]:
    """Flat lr for ``constant_epochs``, then ``CosineAnnealingLR`` to 0.

    ``constant_epochs=0`` returns a plain ``CosineAnnealingLR`` rather than a
    1-epoch ``SequentialLR``: ``SequentialLR`` only switches sub-schedulers
    once ``last_epoch >= milestones[0]``, which for ``milestones=[0]`` still
    serves one flat epoch before the cosine leg starts, shifting the whole
    trajectory by one epoch relative to a schedule with no hold at all.
    """
    assert (
        0 <= constant_epochs < max_epochs
    ), f"constant_epochs must be in [0, {max_epochs}), got {constant_epochs}"
    if constant_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max_epochs)
    schedulers = [
        ConstantLR(optimizer, factor=1.0, total_iters=constant_epochs),
        CosineAnnealingLR(optimizer, T_max=max_epochs - constant_epochs),
    ]
    return SequentialLR(
        optimizer, schedulers=schedulers, milestones=[constant_epochs]
    )


if __name__ == "__main__":
    demo_params = [torch.nn.Parameter(torch.zeros(1))]
    demo_opt = torch.optim.SGD(demo_params, lr=0.1)
    demo_sched = constant_then_cosine(
        demo_opt, constant_epochs=5, max_epochs=10
    )
    for epoch in range(10):
        print(f"epoch {epoch}: lr={demo_opt.param_groups[0]['lr']:.4f}")
        demo_sched.step()
