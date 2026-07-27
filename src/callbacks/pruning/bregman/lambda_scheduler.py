import math
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """Feedback controller for the Bregman regularization strength λ.

    :meth:`step` raises λ while the model is less sparse than
    ``target_sparsity`` and lowers it when sparser, once every
    ``update_frequency`` steps. The move is relative — ``Δλ/λ`` is ``+α·gap``
    while the model is short of the target and the reciprocal once past it —
    so the controller behaves the same at any λ scale and λ can reach any
    setpoint from any start.

    The factor ``α`` comes from :meth:`effective_acceleration_factor`, which
    reads the global optimizer step, so it can taper over the run.

    >>> sched = LambdaScheduler(initial_lambda=1.0)
    >>> sched.target_sparsity = 0.9  # BregmanPruner sets this at fit start
    >>> sched.step(0.5, current_step=0) > 1.0  # below target -> grows
    True
    >>> sched.effective_acceleration_factor(10)  # constant for now
    0.25
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        acceleration_factor: float = 0.25,
        update_frequency: int = 1,
    ):
        assert (
            initial_lambda > 0.0
        ), f"initial_lambda must be > 0.0, got {initial_lambda}"
        assert (
            acceleration_factor >= 0.0
        ), f"acceleration_factor must be >= 0.0, got {acceleration_factor}"
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"

        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.update_frequency = update_frequency
        self.target_sparsity: Optional[float] = None  # set at fit start
        self.last_delta = 0.0  # Δλ the last step() applied
        self.last_delta_over_lambda = 0.0  # Δλ/λ the last step() applied

    def effective_acceleration_factor(self, num_steps: int) -> float:
        """The factor scaling the relative λ. `num_steps` is the number of optimizer steps."""
        p = 1.0
        warmup_updates = 1e3
        num_updates = num_steps // self.update_frequency

        if num_updates > warmup_updates:
            return self.acceleration_factor / max(1, num_updates - warmup_updates) ** p
        else:
            return self.acceleration_factor

    def step(self, current_sparsity: float, current_step: int) -> float:
        """Move λ one update toward ``target_sparsity``.

        Inputs:
            current_sparsity: measured model sparsity in [0, 1].
            current_step: global step; updates land on multiples of
                ``update_frequency``.

        Output:
            Current lambda value.
        """
        if not math.isfinite(current_sparsity):
            raise ValueError(
                f"Sparsity must be finite, got {current_sparsity}"
            )
        if not 0.0 <= current_sparsity <= 1.0:
            raise ValueError(
                f"Sparsity must be in [0.0, 1.0], got {current_sparsity}"
            )
        assert (
            self.target_sparsity is not None
        ), "target_sparsity must be set before the first lambda update"

        # Zero between updates so the logged series sums exactly.
        self.last_delta = 0.0
        self.last_delta_over_lambda = 0.0
        if current_step % self.update_frequency != 0:
            return self.lambda_value

        # lambda_t+1 = lambda_t * (1 + alpha * |gap|)^sign(gap)
        alpha = self.effective_acceleration_factor(current_step)
        gap = self.target_sparsity - float(current_sparsity)
        factor = 1.0 + alpha * abs(gap)
        delta = self.lambda_value * (
            factor - 1.0 if gap >= 0 else 1.0 / factor - 1.0
        )
        self.last_delta_over_lambda = delta / self.lambda_value
        self.lambda_value += delta
        self.last_delta = delta

        assert math.isfinite(
            self.lambda_value
        ), f"lambda_value became non-finite: {self.lambda_value}"
        return self.lambda_value

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Checkpoint state: the live lambda."""
        return {"lambda_value": self.lambda_value}

    def load_state(self, state: dict) -> None:
        """Restore lambda from a checkpoint; the rest comes from config."""
        self.lambda_value = state["lambda_value"]
        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )


if __name__ == "__main__":
    # Smoke: sparsity lags the target, so lambda climbs until the gap closes.
    sched = LambdaScheduler(
        initial_lambda=1.0,
        acceleration_factor=1.0,
        update_frequency=50,
    )
    sched.target_sparsity = 0.9
    measured = 0.2
    for current_step in range(0, 1000, 50):
        lam = sched.step(measured, current_step)
        measured = min(0.9, measured + 0.05)  # model approaches the target
        print(
            f"step {current_step:4d}: sparsity={measured:.3f} "
            f"lambda={lam:.4f} dlambda={sched.last_delta:.4f} "
            f"dlambda/lambda={sched.last_delta_over_lambda:.4f}"
        )
    print(f"final lambda={sched.get_lambda():.4f}")
