"""Trainer probes shared by task modules and compression callbacks."""

from pytorch_lightning import Trainer


def total_training_steps(trainer: Trainer, needed_for: str) -> int:
    """Optimizer steps the run will take; raises if the budget is unbounded.

    Lightning reports ``inf`` for ``max_epochs=-1`` and falls back to
    ``max_steps`` (-1 by default) for iterable datasets, so both sentinels mean
    "unknown". ``needed_for`` names the caller in the error.

    >>> from unittest.mock import Mock
    >>> total_training_steps(Mock(estimated_stepping_batches=5000), "the demo")
    5000
    """
    total = trainer.estimated_stepping_batches
    if not isinstance(total, int) or total < 1:
        raise RuntimeError(
            f"{needed_for} requires a finite trainer.estimated_stepping_batches, "
            f"got {total!r}. Set trainer.max_epochs, or max_steps for an "
            "iterable dataset."
        )
    return total


if __name__ == "__main__":
    from unittest.mock import Mock

    trainer = Mock(estimated_stepping_batches=1234)
    print(total_training_steps(trainer, "the lambda trust region"))
