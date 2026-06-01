import math
from numbers import Real
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)

# Lower bound on the remaining-sparsity fraction in log-space interpolation,
# avoids log(0). Same value the magnitude-pruning scheduler uses; duplicated
# here so the Bregman stack stays independent of src/callbacks/pruning/scheduler.py.
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


class LambdaScheduler:
    """Lambda controller for Bregman target-sparsity tracking.

    A feedback controller that updates ``lambda`` once per call to 
    :meth:`step` (per-batch when driven from a batch-end hook):
    lambda is increased when measured sparsity is below the current target and
    decreased when it is above.

    The target the controller chases has two modes:

    - **fixed** (``target_initial_sparsity=None``): ``target_sparsity`` is the
      constant ``target_sparsity`` scalar.
    - **ramp** (``target_initial_sparsity`` set): ``target_sparsity``
      interpolates ``target_initial_sparsity -> target_sparsity`` over
      ``epochs_to_ramp`` epochs and is held at the final value afterward.
      ``ramp_granularity="step"`` advances the target every batch (smooth);
      ``"epoch"`` advances it once per epoch.


    Near the target (within ``damping_zone``) updates become less frequent and
    gentler to reduce oscillation. ``damping_zone`` also doubles as the
    convergence band: the first time sparsity reaches within ``damping_zone`` of
    the *final* target the controller latches ``_converged``, after which
    ``max_relative_change`` caps the per-update relative change in lambda. So
    the initial climb and the whole ramp run unthrottled, and lambda is
    stabilised only once the operating point is reached. ``damping_zone=0``
    disables both the damping and the clamp.

    Parameters
    ----------
    initial_lambda : float
        Initial regularization weight (must be > 0).
    target_sparsity : float
        Final / steady target sparsity. Held constant in fixed mode; the ramp
        endpoint in ramp mode.
    target_initial_sparsity : float, optional
        Ramp start. ``None`` (default) selects fixed mode.
    schedule_type : {"linear", "constant"}, default="constant"
        Ramp interpolation. "constant" is log-space ("exponential").
    epochs_to_ramp : int, optional
        Ramp length in epochs. Required (>= 1) in ramp mode.
    ramp_granularity : {"step", "epoch"}, default="step"
        Whether the ramp target advances per batch or per epoch.
    acceleration_factor : float, default=0.25
        Multiplies the sparsity gap to control update aggressiveness.
    warmup_epochs : int, default=0
        Epochs to hold lambda at ``initial_lambda`` before scheduling begins.
    update_frequency : int, default=1
        Update lambda every this many steps.
    damping_zone : float, default=0.1
        Sparsity distance from the target that activates damping, and the band
        around the final target that latches convergence. 0.0 disables both.
    damping_frequency_multiplier : int, default=10
        Multiplies ``update_frequency`` inside the damping zone.
    damping_acceleration_divisor : float, default=5.0
        Divides ``acceleration_factor`` inside the damping zone.
    max_relative_change : float, optional
        Bounds the per-update relative change in lambda once the controller has
        converged. ``None`` disables the clamp (pure feedback).

    Examples
    --------
    >>> sched = LambdaScheduler(target_sparsity=0.9, initial_lambda=1.0)
    >>> sched.target_sparsity
    0.9
    >>> sched.final_target
    0.9
    >>> ramp = LambdaScheduler(
    ...     target_sparsity=0.9,
    ...     target_initial_sparsity=0.0,
    ...     epochs_to_ramp=2,
    ...     schedule_type="linear",
    ...     ramp_granularity="step",
    ...     initial_lambda=1.0,
    ... )
    >>> ramp.resolve_warmup_steps(10)
    >>> ramp.target_sparsity
    0.0
    >>> _ = ramp.step(0.0, current_step=10)
    >>> round(ramp.target_sparsity, 2)
    0.45
    >>> ramp.final_target
    0.9
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: float = 0.9,
        target_initial_sparsity: Optional[float] = None,
        schedule_type: str = "constant",
        epochs_to_ramp: Optional[int] = None,
        ramp_granularity: str = "step",
        acceleration_factor: float = 0.25,
        warmup_epochs: int = 0,
        update_frequency: int = 1,
        damping_zone: float = 0.1,
        damping_frequency_multiplier: int = 10,
        damping_acceleration_divisor: float = 5.0,
        max_relative_change: Optional[float] = 0.1,
    ):
        self._target_final = self._validated_sparsity(
            target_sparsity, "target_sparsity"
        )
        self._schedule_type = schedule_type
        self._epochs_to_ramp = epochs_to_ramp
        self._ramp_granularity = ramp_granularity
        if target_initial_sparsity is not None:
            self._target_initial = self._validated_sparsity(
                target_initial_sparsity, "target_initial_sparsity"
            )
            if self._target_initial > self._target_final:
                raise ValueError(
                    "ramp requires target_initial_sparsity <= target_sparsity, "
                    f"got initial={self._target_initial}, "
                    f"final={self._target_final}"
                )
            if epochs_to_ramp is None or epochs_to_ramp < 1:
                raise ValueError(
                    "epochs_to_ramp must be an int >= 1 in ramp mode, "
                    f"got {epochs_to_ramp}"
                )
            if schedule_type not in ("linear", "constant"):
                raise ValueError(
                    f"schedule_type must be 'linear' or 'constant', "
                    f"got {schedule_type!r}"
                )
            if ramp_granularity not in ("step", "epoch"):
                raise ValueError(
                    f"ramp_granularity must be 'step' or 'epoch', "
                    f"got {ramp_granularity!r}"
                )
        else:
            self._target_initial = None

        if acceleration_factor < 0.0:
            raise ValueError(
                f"acceleration_factor must be >= 0.0, got {acceleration_factor}"
            )
        if initial_lambda <= 0.0:
            raise ValueError(
                f"initial_lambda must be > 0.0, got {initial_lambda}"
            )

        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.warmup_epochs = warmup_epochs
        # Resolved to actual steps by BregmanPruner via resolve_warmup_steps().
        self.warmup_steps = 0
        # Set by resolve_warmup_steps(); the ramp target and warmup both need
        # the per-epoch batch count.
        self._steps_per_epoch: Optional[int] = None
        self._last_step: int = 0
        # Latched the first time sparsity reaches the final-target band; gates
        # the max_relative_change clamp.
        self._converged: bool = False
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"
        self.update_frequency = update_frequency
        if damping_zone < 0.0:
            raise ValueError(
                f"damping_zone must be >= 0.0, got {damping_zone}"
            )
        if damping_frequency_multiplier < 1:
            raise ValueError(
                "damping_frequency_multiplier must be >= 1, "
                f"got {damping_frequency_multiplier}"
            )
        if damping_acceleration_divisor <= 0.0:
            raise ValueError(
                "damping_acceleration_divisor must be > 0.0, "
                f"got {damping_acceleration_divisor}"
            )
        self.damping_zone = damping_zone
        self.damping_frequency_multiplier = damping_frequency_multiplier
        self.damping_acceleration_divisor = damping_acceleration_divisor
        if max_relative_change is not None and max_relative_change <= 0.0:
            raise ValueError(
                f"max_relative_change must be > 0.0 when set, "
                f"got {max_relative_change}"
            )
        self.max_relative_change = max_relative_change

    @staticmethod
    def _validated_sparsity(value: float, name: str) -> float:
        """Coerce and range-check a target sparsity."""
        if not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, got {type(value)}")
        value = float(value)
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{name}={value} must be finite and in [0.0, 1.0]"
            )
        return value

    @property
    def target_sparsity(self) -> float:
        """Current target sparsity the feedback loop chases.

        Fixed mode returns the constant final target. Ramp mode returns the
        interpolated value; before the batch count is known
        (``_steps_per_epoch`` unset) it returns the ramp start.
        """
        return self._compute_target(self._target_advance_interval())

    @property
    def final_target(self) -> float:
        """Steady target held after the ramp completes (scalar in both
        modes)."""
        return self._target_final

    def _target_advance_interval(self) -> int:
        """Number of steps between successive ramp-target advances.

        Epoch granularity advances at epoch boundaries; step granularity
        advances every ``update_frequency`` steps, in lockstep with the lambda
        updates.
        """
        if self._ramp_granularity == "epoch":
            return self._steps_per_epoch
        return self.update_frequency

    def _ramp_progress(self, advance_interval: int) -> float:
        """Fraction in [0.0, 1.0] through the ramp, measured after warmup."""
        ramp_steps = self._epochs_to_ramp * self._steps_per_epoch
        raw_position = max(0, self._last_step - self.warmup_steps)
        # The endpoint is final once raw_position reaches the ramp length, even
        # when ramp_steps is not a whole multiple of advance_interval — the
        # rounding below would otherwise pull it back short of 1.0.
        if raw_position >= ramp_steps:
            return 1.0
        aligned_position = raw_position - raw_position % advance_interval
        return min(1.0, aligned_position / ramp_steps)

    def _compute_target(self, advance_interval: int) -> float:
        """Target sparsity for a given ramp advance interval."""
        if self._target_initial is None:
            return self._target_final
        if not self._steps_per_epoch:
            return self._target_initial
        progress = self._ramp_progress(advance_interval)
        return _interpolate_target_sparsity(
            self._schedule_type,
            self._target_initial,
            self._target_final,
            progress,
        )

    def step(
        self,
        current_sparsity: float,
        current_step: Optional[int] = None,
    ) -> float:
        """Process a sparsity reading and update lambda.

        Inputs:
            current_sparsity: current model sparsity.
            current_step: global training step; drives warmup, update
                frequency, and the ramp target.

        Output:
            Current lambda value.
        """
        # Ramp mode needs current_step so the moving target can advance.
        if self._target_initial is not None:
            assert current_step is not None, (
                "current_step must be provided in ramp mode; without it the "
                "ramp target never advances"
            )

        # Track the step before any early return so target_sparsity reflects the
        # correct ramp position even during warmup.
        if current_step is not None:
            self._last_step = int(current_step)

        if self.warmup_steps > 0:
            assert (
                current_step is not None
            ), "current_step must be provided when warmup_steps > 0"
            if current_step <= self.warmup_steps:
                if current_step == 0:
                    log.info(
                        f"Warmup phase: holding lambda at "
                        f"{self.lambda_value:.4f} for {self.warmup_steps} "
                        f"steps ({self.warmup_epochs} epochs)."
                    )
                return self.lambda_value
            if current_step == self.warmup_steps + 1:
                log.info(
                    f"Warmup complete. Starting lambda updates with target "
                    f"sparsity {self.target_sparsity:.4f}."
                )

        self._validate_sparsity(current_sparsity)
        sparsity = float(current_sparsity)
        # One consistent target for both the proximity check and the gap;
        # damping never feeds back into the target computation.
        target = self.target_sparsity

        # Damping keys off the undamped target, so it behaves identically for
        # constant and ramped schedules.
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

        # Convergence: within damping_zone of the FINAL target. The
        # relative clamp activates only afterward, so the initial climb and the
        # whole ramp run unthrottled.
        if (
            self.damping_zone > 0.0
            and abs(sparsity - self._target_final) <= self.damping_zone
        ):
            self._converged = True

        # Asymmetric update — multiply below target, divide above — keeps lambda
        # strictly positive for any acceleration, so no floor is needed.
        gap = target - sparsity
        lambda_prev = self.lambda_value
        if gap > 0:
            self.lambda_value *= 1.0 + effective_acceleration * gap
        elif gap < 0:
            self.lambda_value /= 1.0 + effective_acceleration * (-gap)

        if self._converged and self.max_relative_change is not None:
            upper = lambda_prev * (1.0 + self.max_relative_change)
            lower = lambda_prev * (1.0 - self.max_relative_change)
            self.lambda_value = max(lower, min(upper, self.lambda_value))

        # Fail loud rather than silently capping: a non-finite lambda means the
        # target is infeasible or acceleration_factor is far too large.
        assert math.isfinite(self.lambda_value), (
            f"lambda became non-finite ({self.lambda_value}); target may be "
            f"infeasible or acceleration_factor too large"
        )
        return self.lambda_value

    def _validate_sparsity(self, current_sparsity: float) -> None:
        """Validate a sparsity reading.

        Expected domain: a finite float in [0.0, 1.0].
        """
        if not isinstance(current_sparsity, Real):
            raise TypeError(
                f"current_sparsity must be a real number, got {type(current_sparsity)}"
            )
        current_sparsity = float(current_sparsity)
        if not math.isfinite(current_sparsity):
            raise ValueError(
                f"current_sparsity must be finite, got {current_sparsity}"
            )
        if current_sparsity < 0.0 or current_sparsity > 1.0:
            raise ValueError(
                f"current_sparsity must be in [0.0, 1.0], got {current_sparsity}."
            )

    def resolve_warmup_steps(self, steps_per_epoch: int) -> None:
        """Convert warmup_epochs to warmup_steps using the actual batch count.

        Called by BregmanPruner once the trainer is available. Also fixes the
        ramp length, which is measured in ``epochs_to_ramp * steps_per_epoch``.
        """
        self._steps_per_epoch = int(steps_per_epoch)
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.warmup_steps > 0:
            log.info(
                f"Lambda warmup: {self.warmup_epochs} epochs "
                f"× {steps_per_epoch} batches/epoch = {self.warmup_steps} steps"
            )
        # The step-granularity target advances one increment per
        # update_frequency steps; too few increments makes the ramp coarse.
        if (
            self._target_initial is not None
            and self._ramp_granularity == "step"
        ):
            ramp_steps = self._epochs_to_ramp * self._steps_per_epoch
            n_increments = ramp_steps // self.update_frequency
            if n_increments < 2:
                log.warning(
                    f"Target ramp has only {n_increments} increment(s): "
                    f"update_frequency={self.update_frequency} is large "
                    f"relative to ramp_steps={ramp_steps}. Lower "
                    f"update_frequency or lengthen the ramp for a smoother "
                    f"target."
                )

    def verify_ramp_feasibility(self, max_epochs: int) -> None:
        """Raise if a ramp cannot complete within the training budget."""
        if self._target_initial is None:
            return
        # The ramp only starts counting after warmup (_ramp_progress subtracts
        # warmup_steps) and reaches progress=1.0 at the first step of epoch
        # ramp_end_epoch — which exists only when max_epochs is strictly larger.
        ramp_end_epoch = self.warmup_epochs + self._epochs_to_ramp
        if max_epochs <= ramp_end_epoch:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) + epochs_to_ramp "
                f"({self._epochs_to_ramp}) = {ramp_end_epoch} >= "
                f"trainer.max_epochs ({max_epochs}); set max_epochs > "
                f"{ramp_end_epoch} so the ramp can complete."
            )

    def get_epoch_targets(self, max_epochs: int, steps_per_epoch: int) -> list:
        """Target sparsity at the end of each epoch, epochs 0..max_epochs-1.

        Inputs:
            max_epochs: number of training epochs.
            steps_per_epoch: batches per epoch (resolved before this call).

        Output:
            List of ``(epoch_index, target_sparsity)`` pairs.
        """
        if self._target_initial is None:
            return [(epoch, self._target_final) for epoch in range(max_epochs)]
        warmup_steps = self.warmup_epochs * steps_per_epoch
        ramp_steps = self._epochs_to_ramp * steps_per_epoch
        # Derived from the passed steps_per_epoch (not self._steps_per_epoch) so
        # this can run at on_train_start before resolve_warmup_steps.
        if self._ramp_granularity == "epoch":
            advance_interval = steps_per_epoch
        else:
            advance_interval = self.update_frequency
        targets = []
        for epoch in range(max_epochs):
            last_step = (epoch + 1) * steps_per_epoch - 1
            raw_position = max(0, last_step - warmup_steps)
            if raw_position >= ramp_steps:
                progress = 1.0
            else:
                aligned_position = (
                    raw_position - raw_position % advance_interval
                )
                progress = min(1.0, aligned_position / ramp_steps)
            target = _interpolate_target_sparsity(
                self._schedule_type,
                self._target_initial,
                self._target_final,
                progress,
            )
            targets.append((epoch, target))
        return targets

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "lambda_value": self.lambda_value,
            "target_initial_sparsity": self._target_initial,
            "target_final_sparsity": self._target_final,
            "schedule_type": self._schedule_type,
            "epochs_to_ramp": self._epochs_to_ramp,
            "ramp_granularity": self._ramp_granularity,
            "warmup_epochs": self.warmup_epochs,
            "warmup_steps": self.warmup_steps,
            "_steps_per_epoch": self._steps_per_epoch,
            "_last_step": self._last_step,
            "_converged": self._converged,
            "acceleration_factor": self.acceleration_factor,
            "damping_zone": self.damping_zone,
            "max_relative_change": self.max_relative_change,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint.

        Supports new-style ramp checkpoints and collapses legacy schedules
        (a ``_target_schedule`` list or a scalar ``target_sparsity``) to fixed
        mode at the final value. Keys removed in the refactor
        (``min_lambda``/``max_lambda``/``_last_sparsity``) are ignored if a
        pre-refactor checkpoint still carries them.
        """
        self.lambda_value = state["lambda_value"]

        if "target_final_sparsity" in state:
            self._target_final = self._validated_sparsity(
                state["target_final_sparsity"], "target_final_sparsity"
            )
            initial = state.get("target_initial_sparsity")
            self._target_initial = (
                None
                if initial is None
                else self._validated_sparsity(
                    initial, "target_initial_sparsity"
                )
            )
            if (
                self._target_initial is not None
                and self._target_initial > self._target_final
            ):
                raise ValueError(
                    f"Checkpoint has target_initial ({self._target_initial}) > "
                    f"target_final ({self._target_final}); checkpoint may be "
                    f"corrupted."
                )
            self._schedule_type = state.get(
                "schedule_type", self._schedule_type
            )
            self._epochs_to_ramp = state.get(
                "epochs_to_ramp", self._epochs_to_ramp
            )
            self._ramp_granularity = state.get(
                "ramp_granularity", self._ramp_granularity
            )
        elif "_target_schedule" in state:
            # Legacy per-epoch list: collapse to fixed mode at the final value.
            self._target_final = self._validated_sparsity(
                state["_target_schedule"][-1], "target_final_sparsity"
            )
            self._target_initial = None
        elif "target_sparsity" in state:
            # Legacy scalar checkpoint.
            self._target_final = self._validated_sparsity(
                state["target_sparsity"], "target_final_sparsity"
            )
            self._target_initial = None

        self._steps_per_epoch = state.get(
            "_steps_per_epoch", self._steps_per_epoch
        )
        self._last_step = int(state.get("_last_step", self._last_step))
        self._converged = bool(state.get("_converged", self._converged))
        self.acceleration_factor = state.get(
            "acceleration_factor", self.acceleration_factor
        )
        self.warmup_epochs = state.get("warmup_epochs", self.warmup_epochs)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.damping_zone = state.get("damping_zone", self.damping_zone)
        self.max_relative_change = state.get(
            "max_relative_change", self.max_relative_change
        )

        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )


if __name__ == "__main__":
    # Smoke: log-space ramp 0.0 -> 0.99 over 10 epochs at epoch granularity.
    sched = LambdaScheduler(
        initial_lambda=1.0,
        target_sparsity=0.99,
        target_initial_sparsity=0.0,
        schedule_type="constant",
        epochs_to_ramp=10,
        ramp_granularity="epoch",
        update_frequency=1,
    )
    steps_per_epoch = 100
    sched.resolve_warmup_steps(steps_per_epoch)
    prev_target = sched.target_sparsity
    for current_step in range(steps_per_epoch * 30):
        target = sched.target_sparsity
        assert target >= prev_target - 1e-9, "target must be monotonic"
        prev_target = target
        measured_sparsity = max(0.0, target - 0.02)  # model lags the target
        lam = sched.step(measured_sparsity, current_step=current_step)
        if current_step % steps_per_epoch == 0:
            print(
                f"step {current_step:4d}: target={target:.4f} lambda={lam:.4f}"
            )
    print(
        f"final target={sched.target_sparsity:.4f} "
        f"(final_target={sched.final_target}) converged={sched._converged}"
    )
