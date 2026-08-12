"""Tests for the Bregman target ramp (`bregman/target_scheduler.py`).

The ramp is a standalone callback: it finds the BregmanPruner in the callback
list, builds a schedule ending at its controller's setpoint, and moves that
setpoint once per epoch. These cover the wiring, the schedule it forwards its
knobs into, and what happens when the pieces it needs are missing.
"""

from unittest.mock import Mock

import pytest

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.bregman.target_scheduler import TargetScheduler
from src.callbacks.pruning.scheduler import PruningScheduler


def _wired(
    epochs_to_ramp=10,
    schedule_type="cubic",
    initial_sparsity=0.0,
    target_sparsity=0.9,
    max_epochs=100,
):
    """A ramp plus the controller it steers, as the configs wire them."""
    controller = LambdaScheduler(
        target_sparsity=target_sparsity, initial_sparsity=initial_sparsity
    )
    ramp = TargetScheduler(
        initial_sparsity=initial_sparsity,
        epochs_to_ramp=epochs_to_ramp,
        schedule_type=schedule_type,
    )
    trainer = Mock(
        callbacks=[
            BregmanPruner(
                target_sparsity=target_sparsity, lambda_scheduler=controller
            )
        ],
        max_epochs=max_epochs,
        current_epoch=0,
    )
    return ramp, controller, trainer


@pytest.mark.parametrize("schedule_type", ["cubic", "linear", "constant"])
def test_setpoint_follows_the_magnitude_pruner_schedule(schedule_type):
    """Epoch for epoch, the ramp aims where a gradual-pruning run of the same
    length is: the comparison the progressive recipes exist for."""
    ramp, controller, trainer = _wired(
        epochs_to_ramp=10, schedule_type=schedule_type
    )
    baseline = PruningScheduler(
        schedule_type=schedule_type, final_sparsity=0.9, epochs_to_ramp=10
    )
    ramp.on_train_start(trainer, Mock())
    for epoch in range(13):
        trainer.current_epoch = epoch
        ramp.on_train_epoch_start(trainer, Mock())
        assert controller.target_sparsity == pytest.approx(
            baseline.get_target_sparsity(epoch)
        )


def test_decay_alpha_off_until_the_ramp_holds():
    """Every intermediate target flips the controller's gap sign on its own;
    only the final, held target should let alpha decay."""
    ramp, controller, trainer = _wired(epochs_to_ramp=10)
    ramp.on_train_start(trainer, Mock())
    for epoch in range(9):
        trainer.current_epoch = epoch
        ramp.on_train_epoch_start(trainer, Mock())
        assert controller.decay_alpha is False
    for epoch in (9, 10, 12):
        trainer.current_epoch = epoch
        ramp.on_train_epoch_start(trainer, Mock())
        assert controller.decay_alpha is True


def test_endpoint_comes_from_the_controller():
    """No final_sparsity config key: the ramp ends at the run's fixed target,
    which is also what the gates band."""
    ramp, _, trainer = _wired(target_sparsity=0.95)
    ramp.on_train_start(trainer, Mock())
    assert ramp.schedule.final_sparsity == 0.95


def test_ramp_starts_where_the_weights_start():
    """A sparse start ramps from that sparsity, not from dense."""
    ramp, controller, trainer = _wired(
        initial_sparsity=0.5, epochs_to_ramp=2, schedule_type="linear"
    )
    ramp.on_train_start(trainer, Mock())
    ramp.on_train_epoch_start(trainer, Mock())
    assert controller.target_sparsity == pytest.approx(0.7)


def test_resume_needs_no_state():
    """A ramp rebuilt mid-run picks the schedule up from the epoch alone."""
    ramp, controller, trainer = _wired(
        epochs_to_ramp=10, schedule_type="linear"
    )
    trainer.current_epoch = 7
    ramp.on_train_start(trainer, Mock())
    ramp.on_train_epoch_start(trainer, Mock())
    assert controller.target_sparsity == pytest.approx(0.72)


def test_ramp_longer_than_the_run_raises():
    ramp, _, trainer = _wired(epochs_to_ramp=80, max_epochs=50)
    with pytest.raises(ValueError, match="epochs_to_ramp"):
        ramp.on_train_start(trainer, Mock())


def test_unknown_schedule_type_raises():
    ramp, _, trainer = _wired(schedule_type="quadratic")
    with pytest.raises(ValueError, match="schedule_type"):
        ramp.on_train_start(trainer, Mock())


def test_fixed_lambda_pruner_raises():
    """Fixed-lambda mode has no setpoint to ramp."""
    trainer = Mock(
        callbacks=[BregmanPruner(target_sparsity=0.9)],
        max_epochs=100,
        current_epoch=0,
    )
    with pytest.raises(AssertionError, match="fixed-lambda"):
        TargetScheduler().on_train_start(trainer, Mock())


def test_without_a_bregman_pruner_raises():
    trainer = Mock(callbacks=[], max_epochs=100, current_epoch=0)
    with pytest.raises(ValueError, match="BregmanPruner"):
        TargetScheduler().on_train_start(trainer, Mock())


def test_start_above_the_endpoint_raises():
    ramp, _, trainer = _wired(initial_sparsity=0.99, target_sparsity=0.9)
    with pytest.raises(AssertionError, match="initial_sparsity"):
        ramp.on_train_start(trainer, Mock())
