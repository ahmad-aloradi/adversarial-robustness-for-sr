"""Per-epoch ramp of the sparsity target the Bregman lambda controller chases.

Idea: start the weights dense and raise the controller's setpoint up to its
configured target over ``epochs_to_ramp`` epochs, then hold — gradual magnitude
pruning's schedule, driven through lambda instead of a mask. It reads
``BregmanPruner.lambda_scheduler`` and writes one float on it per epoch;
nothing in the Bregman stack imports this file.

**The gates keep the fixed final target!** Checkpointing, early stopping and
validation open only inside the band around it, so every epoch of the ramp is
unvalidated.

Run it with::

    python src/train.py experiment=img/bregman_adabreg_progressive datamodule=datasets/cifar100

Inspect the schedule alone with::

    python -m src.callbacks.pruning.bregman.target_scheduler
"""

from typing import Optional

from pytorch_lightning import Callback, LightningModule, Trainer

from src import utils
from src.callbacks.pruning.scheduler import PruningScheduler

from .bregman_pruner import BregmanPruner
from .lambda_scheduler import LambdaScheduler

log = utils.get_pylogger(__name__)


class TargetScheduler(Callback):
    """Ramps the lambda controller's setpoint, then holds it at the target.

    The schedule is ``PruningScheduler``, the magnitude pruner's own
    (``src/callbacks/pruning/scheduler.py``), so a Bregman ramp and a
    gradual-pruning run of the same length aim at the same sparsity in the same
    epoch. ``schedule_type`` takes the values it defines: ``cubic``, ``linear``
    or ``constant``.

    The ramp ends at ``LambdaScheduler.target_sparsity``, read at train start
    rather than configured here: that is the value the gates band.
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        epochs_to_ramp: int = 10,
        schedule_type: str = "cubic",
    ):
        super().__init__()
        assert (
            0.0 <= initial_sparsity < 1.0
        ), f"initial_sparsity must be in [0.0, 1.0), got {initial_sparsity}"
        assert (
            isinstance(epochs_to_ramp, int) and epochs_to_ramp >= 1
        ), f"epochs_to_ramp must be an int >= 1, got {epochs_to_ramp}"
        self.initial_sparsity = float(initial_sparsity)
        self.epochs_to_ramp = int(epochs_to_ramp)
        self.schedule_type = schedule_type
        self.schedule: Optional[PruningScheduler] = None

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Build the schedule that ends at the controller's setpoint.

        Why here: every ``on_fit_start`` has run, so the pruner has built its
        lambda scheduler wherever this callback sits in the list.
        """
        target = self._controller(trainer).target_sparsity
        assert (
            self.initial_sparsity <= target
        ), f"initial_sparsity must be <= the controller's target ({target}), got {self.initial_sparsity}"

        self.schedule = PruningScheduler(
            schedule_type=self.schedule_type,
            final_sparsity=target,
            epochs_to_ramp=self.epochs_to_ramp,
            initial_sparsity=self.initial_sparsity,
        )
        self.schedule.verify_schedule_feasibility(trainer.max_epochs)
        log.info(
            f"Target ramp: {self.initial_sparsity:.3%} -> {target:.3%} over "
            f"{self.epochs_to_ramp} epochs ({self.schedule_type})"
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Move the controller's setpoint to this epoch's ramp value."""
        target = self.schedule.get_target_sparsity(trainer.current_epoch)
        self._controller(trainer).target_sparsity = target
        pl_module.log("bregman/ramp_target", target)

    @staticmethod
    def _controller(trainer: Trainer) -> LambdaScheduler:
        """The lambda controller this ramp steers."""
        for callback in trainer.callbacks:
            if isinstance(callback, BregmanPruner):
                assert callback.lambda_scheduler is not None, (
                    "TargetScheduler ramps a lambda controller's setpoint, but "
                    "BregmanPruner runs without one (fixed-lambda mode)"
                )
                return callback.lambda_scheduler
        raise ValueError(
            "TargetScheduler needs a BregmanPruner callback, found "
            f"{[type(c).__name__ for c in trainer.callbacks]}"
        )


if __name__ == "__main__":
    # Smoke: the cubic ramp a dense start follows to 99%, then holds.
    ramp = TargetScheduler(epochs_to_ramp=10)
    schedule = PruningScheduler(
        ramp.schedule_type, 0.99, ramp.epochs_to_ramp, ramp.initial_sparsity
    )
    for epoch in range(13):
        print(
            f"epoch {epoch:2d}: target={schedule.get_target_sparsity(epoch):.4f}"
        )
