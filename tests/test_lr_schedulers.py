"""Unit tests for the constant-then-cosine lr schedule.

Covers:
- ``constant_epochs=0`` returns a plain ``CosineAnnealingLR``, not a 1-epoch
  ``SequentialLR`` -- the latter shifts the whole trajectory by one epoch.
- A positive ``constant_epochs`` holds the base lr flat, then anneals it
  towards 0 with no discontinuity at the switch.
- Out-of-contract arguments raise.
"""
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.lr_schedulers import constant_then_cosine

BASE_LR = 0.1


def _optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)


def _trajectory(scheduler, optimizer, epochs):
    values = []
    for _ in range(epochs):
        values.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return values


def test_zero_constant_epochs_is_a_plain_cosine_scheduler():
    optimizer = _optimizer()
    scheduler = constant_then_cosine(
        optimizer, constant_epochs=0, max_epochs=10
    )
    assert isinstance(scheduler, CosineAnnealingLR)


def test_zero_constant_epochs_matches_plain_cosine_trajectory():
    opt_a, opt_b = _optimizer(), _optimizer()
    scheduler_a = constant_then_cosine(opt_a, constant_epochs=0, max_epochs=10)
    scheduler_b = CosineAnnealingLR(opt_b, T_max=10)
    assert _trajectory(scheduler_a, opt_a, 10) == _trajectory(
        scheduler_b, opt_b, 10
    )


def test_positive_constant_epochs_holds_the_base_lr():
    optimizer = _optimizer()
    scheduler = constant_then_cosine(
        optimizer, constant_epochs=5, max_epochs=10
    )
    values = _trajectory(scheduler, optimizer, 10)
    assert values[:6] == pytest.approx([BASE_LR] * 6)


def test_positive_constant_epochs_anneals_towards_zero_with_no_jump():
    optimizer = _optimizer()
    scheduler = constant_then_cosine(
        optimizer, constant_epochs=5, max_epochs=10
    )
    values = _trajectory(scheduler, optimizer, 10)
    assert values[6] < values[5]  # cosine leg has started descending
    assert values[6] > 0.5 * BASE_LR  # no discontinuous drop at the switch
    assert all(b <= a for a, b in zip(values[5:], values[6:]))
    assert values[-1] < 0.1 * BASE_LR


@pytest.mark.parametrize(
    "constant_epochs, max_epochs", [(-1, 10), (10, 10), (11, 10)]
)
def test_out_of_contract_raises(constant_epochs, max_epochs):
    with pytest.raises(AssertionError):
        constant_then_cosine(_optimizer(), constant_epochs, max_epochs)
