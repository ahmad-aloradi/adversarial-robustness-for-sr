import math
from collections import deque
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
    :meth:`step` (per-batch when driven from a batch-end hook): lambda is
    increased when measured sparsity is below the current target and decreased
    when it is above. The target the controller chases has two modes:

    - **fixed** (``target_initial_sparsity=None``): ``target_sparsity`` is the
      constant ``target_sparsity`` scalar.
    - **ramp** (``target_initial_sparsity`` set): ``target_sparsity``
      interpolates ``target_initial_sparsity -> target_sparsity`` over
      ``epochs_to_ramp`` epochs and is held at the final value afterward.
      ``ramp_granularity="step"`` updates the target every batch (smooth);
      ``"epoch"`` updates it once per epoch.

    When sparsity enters within ``damping_zone`` of the target, updates become
    less frequent and gentler to reduce oscillation.

    Parameters
    ----------
    initial_lambda : float
        Initial regularization weight.
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
    min_lambda : float, default=1e-6
        Minimum lambda value.
    max_lambda : float, default=1e3
        Maximum lambda value.
    warmup_epochs : int, default=0
        Epochs to hold lambda at ``initial_lambda`` before scheduling begins.
    update_frequency : int, default=1
        Update lambda every this many steps.
    damping_zone : float, default=0.0
        Sparsity distance from target that activates damping. 0.0 disables it.
    damping_frequency_multiplier : int, default=10
        Multiplies ``update_frequency`` inside the damping zone.
    damping_acceleration_divisor : float, default=5.0
        Divides ``acceleration_factor`` inside the damping zone.
    max_relative_change : float, optional
        Bounds the per-update relative change in lambda once the first epoch
        has completed. ``None`` (default) disables the clamp.

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
        min_lambda: float = 1e-6,
        max_lambda: float = 1e3,
        warmup_epochs: int = 0,
        update_frequency: int = 1,
        damping_zone: float = 0.0,
        damping_frequency_multiplier: int = 10,
        damping_acceleration_divisor: float = 5.0,
        max_relative_change: Optional[float] = None,
    ):
        self._target_final = self._validated_sparsity(
            target_sparsity, "target_sparsity"
        )
        self._is_ramp = target_initial_sparsity is not None
        self._schedule_type = schedule_type
        self._epochs_to_ramp = epochs_to_ramp
        self._ramp_granularity = ramp_granularity
        if self._is_ramp:
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
        if min_lambda <= 0.0:
            raise ValueError(f"min_lambda must be > 0.0, got {min_lambda}")
        if max_lambda < min_lambda:
            raise ValueError(
                f"max_lambda must be >= min_lambda, got max_lambda={max_lambda}, "
                f"min_lambda={min_lambda}"
            )
        if not (min_lambda <= initial_lambda <= max_lambda):
            raise ValueError(
                "initial_lambda must be between min_lambda and max_lambda, "
                f"got {initial_lambda}"
            )

        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self._last_sparsity = None
        self.warmup_epochs = warmup_epochs
        # Resolved to actual steps by BregmanPruner via resolve_warmup_steps().
        self.warmup_steps = 0
        # Set by resolve_warmup_steps(); the ramp target and warmup both need
        # the per-epoch batch count.
        self._steps_per_epoch: Optional[int] = None
        self._last_step: int = 0
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"
        self.update_frequency = update_frequency
        self.damping_zone = damping_zone
        self.damping_frequency_multiplier = damping_frequency_multiplier
        self.damping_acceleration_divisor = damping_acceleration_divisor
        if max_relative_change is not None and max_relative_change <= 0.0:
            raise ValueError(
                f"max_relative_change must be > 0.0 when set, "
                f"got {max_relative_change}"
            )
        self.max_relative_change = max_relative_change
        # Lazily sized by detect_uncontrolled_oscillation() on first call.
        self._oscillation_history: Optional[deque] = None

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
        if not self._is_ramp:
            return self._target_final
        if self._steps_per_epoch is None or self._steps_per_epoch == 0:
            return self._target_initial
        progress = self._ramp_progress()
        return _interpolate_target_sparsity(
            self._schedule_type,
            self._target_initial,
            self._target_final,
            progress,
        )

    @property
    def final_target(self) -> float:
        """Steady target held after the ramp completes (scalar in both
        modes)."""
        return self._target_final

    def _ramp_progress(self) -> float:
        """Fraction in [0.0, 1.0] through the ramp, measured after warmup."""
        if self._ramp_granularity == "step":
            ramp_steps = self._epochs_to_ramp * self._steps_per_epoch
            raw_position = max(0, self._last_step - self.warmup_steps)
            # Once the raw position reaches the ramp length the endpoint is
            # final, even if ramp_steps is not divisible by update_frequency
            # (the snap below would otherwise round it back short of 1.0).
            if raw_position >= ramp_steps:
                return 1.0
            # Advance the target only on lambda-update steps (one increment per
            # update_frequency). Otherwise the per-step target races ahead of
            # the slower lambda updates and each lambda correction lands a
            # non-smooth multi-step jump.
            snapped_position = (
                raw_position - raw_position % self.update_frequency
            )
            return min(1.0, snapped_position / ramp_steps)
        if self._ramp_granularity == "epoch":
            current_epoch = self._last_step // self._steps_per_epoch
            epoch_position = max(0, current_epoch - self.warmup_epochs)
            return min(1.0, epoch_position / self._epochs_to_ramp)
        raise ValueError(
            f"Unknown ramp_granularity: {self._ramp_granularity!r}. "
            "Expected one of ['step', 'epoch']."
        )

    def step(
        self,
        current_sparsity: float,
        last_sparsity: Optional[float] = None,
        current_step: Optional[int] = None,
    ) -> float:
        """Process a sparsity reading and update lambda.

        Inputs:
            current_sparsity: current model sparsity.
            last_sparsity: if provided, cached as the last known sparsity
                (used once when resuming from a checkpoint).
            current_step: global training step; drives warmup, update
                frequency, and the ramp target.

        Output:
            Current lambda value.
        """
        # Track current step before any early return so the target_sparsity
        # property reflects the correct ramp position even during warmup.
        if current_step is not None:
            self._last_step = int(current_step)

        if self.warmup_steps > 0:
            assert (
                current_step is not None
            ), "current_step must be provided when warmup_steps > 0"
            if current_step <= self.warmup_steps:
                if current_step == 0:
                    log.info(
                        f"Warmup phase: Holding lambda at "
                        f"{self.lambda_value:.4f} for {self.warmup_steps} "
                        f"steps ({self.warmup_epochs} epochs)."
                    )
                # At this point, self.lambda_value = initial_lambda
                return self.lambda_value
            if current_step == self.warmup_steps + 1:
                log.info(
                    f"Warmup complete. Starting lambda updates with "
                    f"target sparsity {self.target_sparsity:.4f}."
                )

        # Determine effective parameters based on proximity to target
        in_damping_zone = (
            self.damping_zone > 0.0
            and abs(current_sparsity - self.target_sparsity)
            < self.damping_zone
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

        # Only update lambda every effective_frequency steps
        if (
            current_step is not None
            and current_step % effective_frequency != 0
        ):
            return self.lambda_value

        # If resuming from a checkpoint, use provided last_sparsity
        if last_sparsity is not None:
            self._validate_sparsity(last_sparsity)
            self._last_sparsity = float(last_sparsity)

        self._validate_sparsity(current_sparsity)

        sparsity_signal = float(current_sparsity)
        self._last_sparsity = sparsity_signal
        sparsity_difference = sparsity_signal - self.target_sparsity

        lambda_prev = self.lambda_value

        if sparsity_signal < self.target_sparsity:
            # Increase lambda to encourage more sparsity
            self.lambda_value *= 1 + effective_acceleration * abs(
                sparsity_difference
            )
        elif sparsity_signal > self.target_sparsity:
            # Decrease lambda since we're above target
            self.lambda_value /= 1 + effective_acceleration * abs(
                sparsity_difference
            )

        # Relative-change clamp — only active once the first epoch has
        # completed so the initial sparsity settle isn't penalised.
        if (
            self.max_relative_change is not None
            and self._steps_per_epoch is not None
            and self._steps_per_epoch > 0
            and self._last_step >= self._steps_per_epoch
        ):
            upper = lambda_prev * (1.0 + self.max_relative_change)
            lower = lambda_prev * (1.0 - self.max_relative_change)
            if self.lambda_value > upper:
                self.lambda_value = upper
            elif self.lambda_value < lower:
                self.lambda_value = lower

        # Clamp lambda to valid range
        self.lambda_value = max(
            self.min_lambda, min(self.max_lambda, self.lambda_value)
        )

        return self.lambda_value

    def detect_uncontrolled_oscillation(
        self,
        current_sparsity: float,
        tolerance: float = 0.01,
        window_steps: int = 500,
        min_crossings: int = 50,
    ) -> bool:
        """Detect sustained overshoot/undershoot around the target.

        Pure detection: reports whether sparsity has been oscillating around
        the target outside the tolerance band for a sustained window of steps.
        Minor oscillations are expected by construction of the multiplicative
        lambda update, so readings within ``tolerance`` of the target are
        treated as "converged" and are not counted as target crossings

        Intended to be invoked once per step alongside :meth:`step`.

        Parameters
        ----------
        current_sparsity : float
            Current sparsity reading; appended to the oscillation window.
        tolerance : float, default=0.01
            Distance from target within which readings are treated as
            in-band and ignored for crossing counts.
        window_steps : int, default=500
            Size of the rolling detection window, measured in calls to this
            method.
        min_crossings : int, default=50
            Minimum number of out-of-tolerance target crossings within the
            window required to flag the dynamics as uncontrolled.

        Returns
        -------
        bool
            True iff the window was full and the crossing threshold was
            exceeded on this call.
        """
        if window_steps < 2:
            raise ValueError(f"window_steps must be >= 2, got {window_steps}")
        if min_crossings < 1:
            raise ValueError(
                f"min_crossings must be >= 1, got {min_crossings}"
            )
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be >= 0.0, got {tolerance}")
        self._validate_sparsity(current_sparsity)

        if (
            self._oscillation_history is None
            or self._oscillation_history.maxlen != window_steps
        ):
            self._oscillation_history = deque(maxlen=window_steps)

        diff = float(current_sparsity) - self.target_sparsity
        if abs(diff) <= tolerance:
            sign = 0
        else:
            sign = 1 if diff > 0 else -1
        self._oscillation_history.append(sign)

        if len(self._oscillation_history) < window_steps:
            return False

        # Count transitions between opposite non-zero signs; zeros
        # (in-tolerance readings) are skipped so a brief return to target
        # doesn't reset a run of overshoot/undershoot flips.
        crossings = 0
        last_nonzero = 0
        for s in self._oscillation_history:
            if s == 0:
                continue
            if last_nonzero != 0 and s != last_nonzero:
                crossings += 1
            last_nonzero = s

        if crossings < min_crossings:
            return False

        log.warning(
            f"Uncontrolled sparsity oscillation detected: {crossings} target "
            f"crossings over a {window_steps}-step window outside tolerance "
            f"±{tolerance}."
        )
        self._oscillation_history.clear()
        return True

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
        if self._is_ramp and self._ramp_granularity == "step":
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
        if not self._is_ramp:
            return
        # The ramp only starts counting after warmup (_ramp_progress subtracts
        # warmup_steps), so it finishes at warmup_epochs + epochs_to_ramp.
        ramp_end_epoch = self.warmup_epochs + self._epochs_to_ramp
        if max_epochs < ramp_end_epoch:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) + epochs_to_ramp "
                f"({self._epochs_to_ramp}) > trainer.max_epochs "
                f"({max_epochs}); the ramp cannot complete."
            )

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
            "_last_sparsity": self._last_sparsity,
            "acceleration_factor": self.acceleration_factor,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "damping_zone": self.damping_zone,
            "max_relative_change": self.max_relative_change,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint.

        Supports new-style ramp checkpoints and collapses legacy schedules
        (a ``_target_schedule`` list or a scalar ``target_sparsity``) to fixed
        mode at the final value.
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
            self._is_ramp = self._target_initial is not None
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
            self._is_ramp = False
        elif "target_sparsity" in state:
            # Legacy scalar checkpoint.
            self._target_final = self._validated_sparsity(
                state["target_sparsity"], "target_final_sparsity"
            )
            self._target_initial = None
            self._is_ramp = False

        self._steps_per_epoch = state.get(
            "_steps_per_epoch", self._steps_per_epoch
        )
        self._last_step = int(state.get("_last_step", self._last_step))
        self._last_sparsity = state.get("_last_sparsity")
        if self._last_sparsity is not None:
            self._validate_sparsity(self._last_sparsity)
        self.acceleration_factor = state["acceleration_factor"]
        self.min_lambda = state["min_lambda"]
        self.max_lambda = state["max_lambda"]
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
    # Smoke: log-space ramp 0.0 -> 0.99 over 4 epochs at step granularity.
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
        # assert target - prev_target < 0.1, "no per-step target jumps"
        prev_target = target
        measured_sparsity = max(0.0, target - 0.02)  # model lags the target
        lam = sched.step(measured_sparsity, current_step=current_step)
        if current_step % steps_per_epoch == 0:
            print(
                f"step {current_step:4d}: target={target:.4f} lambda={lam:.4f}"
            )
    print(
        f"final target={sched.target_sparsity:.4f} "
        f"(final_target={sched.final_target})"
    )
