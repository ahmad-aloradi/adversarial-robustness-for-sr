"""Composite learning-rate schedules: a first leg, then cosine.

Idea: ``SequentialLR`` runs one leg (a flat hold, or a linear warmup) and hands
over to ``CosineAnnealingLR`` at the milestone.

**``SequentialLR`` needs its sub-schedulers built on the same optimizer!** Hydra
instantiates a nested node before it passes ``optimizer=`` to the outer target,
so the legs cannot be written as YAML — build them here.

Run it with::

    python -m src.utils.lr_schedulers
"""
from typing import Union

import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
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


def warmup_then_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    start_factor: float,
    eta_min: float = 0.0,
) -> SequentialLR:
    """Linear warmup over ``warmup_epochs``, then ``CosineAnnealingLR`` to ``eta_min``.

    The first step serves ``start_factor * lr``. ``LinearLR`` reaches the base
    lr on step ``warmup_epochs``, so the cosine leg gets ``max_epochs -
    warmup_epochs`` steps and no cosine decay applies during the warmup.
    """
    assert (
        0 < warmup_epochs < max_epochs
    ), f"warmup_epochs must be in (0, {max_epochs}), got {warmup_epochs}"
    schedulers = [
        LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=eta_min),
    ]
    return SequentialLR(
        optimizer, schedulers=schedulers, milestones=[warmup_epochs]
    )


if __name__ == "__main__":
    demo_params = [torch.nn.Parameter(torch.zeros(1))]
    demo_opt = torch.optim.SGD(demo_params, lr=0.1)
    demo_sched = warmup_then_cosine(
        demo_opt, warmup_epochs=20, max_epochs=100, start_factor=0.1
    )
    for epoch in range(100):
        print(f"epoch {epoch}: lr={demo_opt.param_groups[0]['lr']:.4f}")
        demo_sched.step()
