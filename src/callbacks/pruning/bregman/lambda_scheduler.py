import math
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)

# Lower bound on the remaining-sparsity fraction in log-space interpolation,
# avoids log(0). Kept local so the Bregman stack stays independent of the
# magnitude-pruning scheduler (src/callbacks/pruning/scheduler.py).
sched_floor: float = 1e-9


def _interpolate_target_sparsity(
    schedule_type: str,
    initial_sparsity: float,
    final_sparsity: float,
    progress: float,
) -> float:
    """Interpolate a target sparsity at a point in a ramp.

    Inputs:
        schedule_type: "linear" or "constant" (log-space on remaining mass).
        initial_sparsity: ramp start (progress 0.0).
        final_sparsity: ramp end (progress 1.0).
        progress: fraction through the ramp, in [0.0, 1.0].

    Output:
        Interpolated target sparsity in [initial_sparsity, final_sparsity].
    """
    assert (
        0.0 <= progress <= 1.0
    ), f"progress must be in [0.0, 1.0], got {progress}"

    if schedule_type == "linear":
        return (
            initial_sparsity + (final_sparsity - initial_sparsity) * progress
        )

    if schedule_type == "constant":
        # log-space interpolation of the surviving-weight fraction: equal
        # relative steps regardless of scale, unlike linear which front-loads
        # coarse sparsity changes.
        remaining_initial = max(1.0 - initial_sparsity, sched_floor)
        remaining_final = max(1.0 - final_sparsity, sched_floor)
        current_remaining = (
            remaining_initial
            * (remaining_final / remaining_initial) ** progress
        )
        return 1.0 - current_remaining

    raise ValueError(
        f"Unknown schedule_type: {schedule_type!r}. "
        "Expected one of ['linear', 'constant']."
    )


class TargetScheduler:
    """Per-epoch ramp of the target sparsity for progressive Bregman pruning.

    Independent of :class:`LambdaScheduler`: it only computes a moving target
    that the caller feeds into the lambda feedback controller each step. The
    target ramps ``initial_sparsity -> final_sparsity`` over ``epochs_to_ramp``
    epochs, then holds at ``final_sparsity``. ``schedule_type="linear"`` walks
    the sparsity value; ``"constant"`` walks the surviving-weight fraction in
    log-space (equal relative steps).

    Stateless w.r.t. training: :meth:`target_at` is a pure function of the epoch,
    so nothing needs checkpointing (Lightning restores ``current_epoch``).

    >>> sched = TargetScheduler(final_sparsity=0.9, initial_sparsity=0.0,
    ...                         epochs_to_ramp=3, schedule_type="linear")
    >>> sched.target_at(0)
    0.0
    >>> round(sched.target_at(1), 3)
    0.3
    >>> sched.target_at(3)
    0.9
    >>> sched.target_at(10)  # held after the ramp
    0.9
    """

    def __init__(
        self,
        final_sparsity: float,
        initial_sparsity: float = 0.0,
        epochs_to_ramp: int = 10,
        schedule_type: str = "constant",
    ):
        assert 0.0 <= initial_sparsity <= final_sparsity <= 1.0, (
            "require 0 <= initial_sparsity <= final_sparsity <= 1, got "
            f"initial={initial_sparsity}, final={final_sparsity}"
        )
        assert (
            isinstance(epochs_to_ramp, int) and epochs_to_ramp >= 1
        ), f"epochs_to_ramp must be an int >= 1, got {epochs_to_ramp}"
        assert schedule_type in (
            "linear",
            "constant",
        ), f"schedule_type must be 'linear' or 'constant', got {schedule_type!r}"
        self.final_sparsity = float(final_sparsity)
        self.initial_sparsity = float(initial_sparsity)
        self.epochs_to_ramp = int(epochs_to_ramp)
        self.schedule_type = schedule_type

    def target_at(self, epoch: int) -> float:
        """Target sparsity at ``epoch``; ramps then holds at final_sparsity."""
        assert epoch >= 0, f"epoch must be >= 0, got {epoch}"
        if epoch >= self.epochs_to_ramp:
            return self.final_sparsity
        progress = epoch / self.epochs_to_ramp
        return _interpolate_target_sparsity(
            self.schedule_type,
            self.initial_sparsity,
            self.final_sparsity,
            progress,
        )

    def verify_schedule_feasibility(self, max_epochs: int) -> None:
        """Fail loud if the ramp cannot complete within the training budget.

        Mirrors ``PruningScheduler.verify_schedule_feasibility``
        (src/callbacks/pruning/scheduler.py) for the magnitude-pruning ramp.
        """
        if max_epochs < self.epochs_to_ramp:
            raise ValueError(
                f"TargetScheduler Error: epochs_to_ramp ({self.epochs_to_ramp}) "
                f"> trainer.max_epochs ({max_epochs}). The ramp cannot "
                "complete, so the gates would never reopen."
            )


class LambdaScheduler:
    """Lambda adapter for sparsity-controlled Bregman Learning.

    Each :meth:`step` updates ``lambda`` to achieve a target sparsity: it increases when
    sparsity is below it, decreases when above. Within ``damping_zone`` of
    the target, updates are gentler and less frequent; ``damping_zone=0``
    disables damping.

    Once :meth:`bind_run` ties the scheduler to a run, every accepted update is
    clamped to the trust region ``|Δλ| <= λ0 · (lr/base_lr) · (1 - k/K)``
    (``k`` = global step, ``K`` = total steps, ``lr`` = live learning rate of
    the regularized groups). λ then has bounded total variation —
    ``Σ |Δλ_k| <= λ0 · (K+1)/2`` while ``lr <= base_lr`` — and the increment is
    exactly 0 from ``K`` on, so λ converges and the tail of training is a
    fixed-λ Bregman iteration. Unbound, the scheduler is the bare controller.

    >>> sched = LambdaScheduler(initial_lambda=1.0)
    >>> sched.step(0.5, target_sparsity=0.9) > 1.0  # below target -> grows
    True
    >>> sched.bind_run(total_steps=1000, base_lr=0.01)
    >>> sched.cap_at(500, lr=0.01)  # half the run left, at base lr
    0.5
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
        assert damping_frequency_multiplier >= 1 and isinstance(
            damping_frequency_multiplier, int
        ), f"Frequency factor must be >= 1 and integer, got {damping_frequency_multiplier}"
        assert (
            damping_acceleration_divisor > 0.0
        ), f"damping_acceleration_divisor must be > 0.0, got {damping_acceleration_divisor}"

        self.initial_lambda = initial_lambda
        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.update_frequency = update_frequency
        self.damping_zone = damping_zone
        self.damping_frequency_multiplier = damping_frequency_multiplier
        self.damping_acceleration_divisor = damping_acceleration_divisor
        self.total_steps: Optional[int] = None
        self.base_lr: Optional[float] = None

    def bind_run(self, total_steps: int, base_lr: float) -> None:
        """Tie the λ trust region to a run; it tapers to 0 at the end."""
        assert total_steps >= 1, f"total_steps must be >= 1, got {total_steps}"
        assert base_lr > 0.0, f"base_lr must be > 0.0, got {base_lr}"
        self.total_steps = total_steps
        self.base_lr = base_lr

    def cap_at(self, current_step: int, lr: float) -> float:
        """Largest |Δλ| allowed now: λ0, scaled by the lr and the run left."""
        return (
            self.initial_lambda
            * (lr / self.base_lr)
            * max(0.0, 1.0 - current_step / self.total_steps)
        )

    def step(
        self,
        current_sparsity: float,
        target_sparsity: float,
        current_step: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> float:
        """Update lambda from a sparsity reading toward ``target_sparsity``.

        Inputs:
            current_sparsity: measured model sparsity in [0, 1].
            target_sparsity: setpoint the controller drives toward; supplied by
                the caller each step (the scheduler holds no target).
            current_step: global training step; gates the update frequency.
                ``None`` updates every call.
            lr: live learning rate of the regularized groups; required once
                :meth:`bind_run` has set the trust region.

        Output:
            Current lambda value.
        """
        self._validate_sparsity(current_sparsity)
        if self.total_steps is not None:
            assert (
                current_step is not None
            ), "bind_run() was called, so step() needs current_step for the taper"
            assert (
                lr is not None and lr >= 0.0
            ), f"bind_run() was called, so step() needs the live lr, got {lr}"
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
        proposed = self.lambda_value
        if gap > 0:
            proposed *= 1.0 + effective_acceleration * gap
        elif gap < 0:
            proposed /= 1.0 + effective_acceleration * (-gap)

        if self.total_steps is None:
            self.lambda_value = proposed
        else:
            # Trust region: Σ|Δλ|/lr stays finite for any LR schedule.
            cap = self.cap_at(current_step, lr)
            self.lambda_value += max(
                -cap, min(cap, proposed - self.lambda_value)
            )

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
    sched.bind_run(total_steps=10000, base_lr=1e-2)
    measured = 0.2
    for current_step in range(1, 10000):
        if current_step % 50 == 0:
            lam = sched.step(
                measured, target, current_step=current_step, lr=1e-2
            )
            measured = min(
                target, measured + 0.002
            )  # model approaches the target
            print(
                f"step {current_step:3d}: sparsity={measured:.3f} "
                f"lambda={lam:.4f} cap={sched.cap_at(current_step, 1e-2):.4f}"
            )
    print(f"final lambda={sched.get_lambda():.4f} target={target}")

    # Smoke: log-space target ramp 0.0 -> 0.99 over 10 epochs, then held.
    ramp = TargetScheduler(final_sparsity=0.99, epochs_to_ramp=10)
    for epoch in range(13):
        print(f"epoch {epoch:2d}: target={ramp.target_at(epoch):.4f}")
