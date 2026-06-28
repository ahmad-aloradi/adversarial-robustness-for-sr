import math
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """Lambda adapter for sparsity-controlled Bregman Learning.

    Each :meth:`step` updates ``lambda`` to achieve a target sparsity: it increases when
    sparsity is below it, decreases when above. Within ``damping_zone`` of
    the target, updates are gentler and less frequent; ``damping_zone=0``
    disables damping.

    >>> sched = LambdaScheduler(initial_lambda=1.0)
    >>> sched.step(0.5, target_sparsity=0.9) > 1.0  # below target -> grows
    True
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        acceleration_factor: float = 0.25,
        update_frequency: int = 1,
        damping_zone: float = 0.1,
        damping_frequency_multiplier: int = 10,
        damping_acceleration_divisor: float = 5.0,
    ):
        assert (
            acceleration_factor >= 0.0
        ), f"acceleration_factor must be >= 0.0, got {acceleration_factor}"
        assert (
            initial_lambda > 0.0
        ), f"initial_lambda must be > 0.0, got {initial_lambda}"
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"
        assert (
            damping_zone >= 0.0
        ), f"damping_zone must be >= 0.0, got {damping_zone}"
        assert (
            damping_frequency_multiplier >= 1 and isinstance(damping_frequency_multiplier, int)
        ), f"Frequency factor must be >= 1 and integer, got {damping_frequency_multiplier}"
        assert (
            damping_acceleration_divisor > 0.0
        ), f"damping_acceleration_divisor must be > 0.0, got {damping_acceleration_divisor}"

        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.update_frequency = update_frequency
        self.damping_zone = damping_zone
        self.damping_frequency_multiplier = damping_frequency_multiplier
        self.damping_acceleration_divisor = damping_acceleration_divisor

    def step(
        self,
        current_sparsity: float,
        target_sparsity: float,
        current_step: Optional[int] = None,
    ) -> float:
        """Update lambda from a sparsity reading toward ``target_sparsity``.

        Inputs:
            current_sparsity: measured model sparsity in [0, 1].
            target_sparsity: setpoint the controller drives toward; supplied by
                the caller each step (the scheduler holds no target).
            current_step: global training step; gates the update frequency.
                ``None`` updates every call.

        Output:
            Current lambda value.
        """
        self._validate_sparsity(current_sparsity)
        sparsity = float(current_sparsity)
        target = float(target_sparsity)

        in_damping_zone = (
            self.damping_zone > 0.0
            and abs(sparsity - target) < self.damping_zone
        )
        effective_frequency = (
            self.update_frequency * self.damping_frequency_multiplier
            if in_damping_zone
            else self.update_frequency
        )
        effective_acceleration = (
            self.acceleration_factor / self.damping_acceleration_divisor
            if in_damping_zone
            else self.acceleration_factor
        )

        if (
            current_step is not None
            and current_step % effective_frequency != 0
        ):
            return self.lambda_value

        # lambda_t+1 = lambda_t * (1 + alpha * |epsilon|)^sign(gap)
        gap = target - sparsity
        if gap > 0:
            self.lambda_value *= 1.0 + effective_acceleration * gap
        elif gap < 0:
            self.lambda_value /= 1.0 + effective_acceleration * (-gap)

        assert math.isfinite(
            self.lambda_value
        ), f"lambda_value became non-finite: {self.lambda_value}"
        return self.lambda_value

    def _validate_sparsity(self, current_sparsity: float) -> None:
        """Validate a sparsity reading: a finite float in [0.0, 1.0]."""
        if not math.isfinite(current_sparsity):
            raise ValueError(
                f"Sparsity must be finite, got {current_sparsity}"
            )
        if not 0.0 <= current_sparsity <= 1.0:
            raise ValueError(
                f"Sparsity must be in [0.0, 1.0], got {current_sparsity}"
            )

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "lambda_value": self.lambda_value,
            "acceleration_factor": self.acceleration_factor,
            "damping_zone": self.damping_zone,
        }

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint.

        ``lambda_value`` is the evolving state and must be present; the damping
        knobs are reconstructed from config but restored here too so a
        checkpoint is self-consistent.
        """
        self.lambda_value = state["lambda_value"]
        self.acceleration_factor = state.get(
            "acceleration_factor", self.acceleration_factor
        )
        self.damping_zone = state.get("damping_zone", self.damping_zone)
        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )


if __name__ == "__main__":
    # Smoke: drive toward a fixed target 0.9; lambda climbs while sparsity lags.
    target = 0.9
    sched = LambdaScheduler(
        initial_lambda=1.0,
        acceleration_factor=0.25,
        update_frequency=1,
    )
    measured = 0.5
    for current_step in range(1, 21):
        lam = sched.step(measured, target, current_step=current_step)
        measured = min(target, measured + 0.02)  # model approaches the target
        if current_step % 5 == 0:
            print(
                f"step {current_step:3d}: sparsity={measured:.3f} "
                f"lambda={lam:.4f}"
            )
    print(f"final lambda={sched.get_lambda():.4f} target={target}")
