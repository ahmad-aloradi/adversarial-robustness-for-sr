import math
from collections import deque
from numbers import Real
from typing import List, Optional, Sequence, Union

from src import utils

log = utils.get_pylogger(__name__)


def _normalize_target_schedule(
    target: Union[float, Sequence[float]]
) -> List[float]:
    """Coerce a scalar or sequence of targets into a validated list."""
    if isinstance(target, Real):
        schedule = [float(target)]
    else:
        try:
            schedule = [float(v) for v in target]
        except TypeError as exc:
            raise TypeError(
                f"target_sparsity must be a float or a sequence of floats, got {type(target)}"
            ) from exc
        if len(schedule) == 0:
            raise ValueError("target_sparsity schedule must not be empty")
    for i, v in enumerate(schedule):
        if not math.isfinite(v) or not (0.0 < v <= 1.0):
            raise ValueError(
                f"target_sparsity[{i}]={v} must be finite and in (0.0, 1.0]"
            )
    return schedule


class LambdaScheduler:
    """Lambda scheduler for Bregman target sparsity control.

    This scheduler updates lambda once per call to :meth:`step` (i.e., per-batch when
    called from a batch-end hook). ``target_sparsity`` may be a single float or a
    per-epoch list; when a list is provided, the i-th entry is the target for
    epoch ``i`` and the final entry is held for all subsequent epochs.

    When sparsity enters within ``damping_zone`` of the target, the scheduler reduces
    oscillation by increasing the effective update frequency (fewer updates) and
    reducing the effective acceleration factor (gentler corrections).

    Parameters
    ----------
    initial_lambda : float
        Initial lambda value for regularization
    target_sparsity : float or list of float
        Target sparsity level to achieve. A list specifies a per-epoch schedule;
        the last value is held for all epochs beyond ``len(list) - 1``.
    acceleration_factor : float, default=0.25
        Factor multiplied by the sparsity difference between current and target
        to control how aggressively to update lambda
    min_lambda : float, default=1e-6
        Minimum lambda value
    max_lambda : float, default=1e3
        Maximum lambda value
    warmup_epochs : int, default=0
        Number of epochs to hold lambda at initial value before scheduling begins.
    update_frequency : int, default=1
        Only update lambda every this many steps.
    damping_zone : float, default=0.0
        Sparsity distance from target to activate damping. 0.0 disables damping.
    damping_frequency_multiplier : int, default=10
        Multiply ``update_frequency`` by this when inside the damping zone.
    damping_acceleration_divisor : float, default=5.0
        Divide ``acceleration_factor`` by this when inside the damping zone.
    max_relative_change : float, optional
        If set, bounds the per-update relative change in lambda such that
        ``|lambda_new - lambda_prev| / lambda_prev <= max_relative_change``.
        Only active once the first epoch has completed
        (``_last_step >= _steps_per_epoch``) to allow sparsity to settle
        from the initial state. ``None`` (default) disables the clamp.
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: Union[float, List[float]] = 0.9,
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
        self._target_schedule = _normalize_target_schedule(target_sparsity)
        if acceleration_factor < 0.0:
            raise ValueError(
                f"acceleration_factor must be >= 0.0, got {acceleration_factor}"
            )
        if min_lambda < 0.0:
            raise ValueError(f"min_lambda must be > 0.0, got {min_lambda}")
        if max_lambda < min_lambda:
            raise ValueError(
                f"max_lambda must be >= min_lambda, got max_lambda={max_lambda}, min_lambda={min_lambda}"
            )
        if not (min_lambda <= initial_lambda <= max_lambda):
            raise ValueError(
                f"initial_lambda must be between min_lambda and max_lambda, got {initial_lambda}"
            )

        self.lambda_value = initial_lambda
        self.acceleration_factor = acceleration_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self._last_sparsity = None
        self.warmup_epochs = warmup_epochs
        # Resolved to actual steps by BregmanPruner.resolve_warmup_steps()
        self.warmup_steps = 0
        # Populated via resolve_warmup_steps() so target_sparsity property can
        # derive the current-epoch target from self._last_step.
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
                f"max_relative_change must be > 0.0 when set, got {max_relative_change}"
            )
        self.max_relative_change = max_relative_change
        # Lazily sized by detect_uncontrolled_oscillation() on first call.
        self._oscillation_history: Optional[deque] = None

    @property
    def target_sparsity(self) -> float:
        """Current-epoch target sparsity derived from ``_last_step``.

        Before training starts (``_steps_per_epoch`` unset), returns the first
        entry of the schedule — this is what consumers log at setup time.
        """
        if self._steps_per_epoch is None or self._steps_per_epoch == 0:
            return self._target_schedule[0]
        epoch = self._last_step // self._steps_per_epoch
        idx = min(epoch, len(self._target_schedule) - 1)
        return self._target_schedule[idx]

    @property
    def target_schedule(self) -> List[float]:
        """Full per-epoch target schedule (at least one element)."""
        return list(self._target_schedule)

    def step(
        self,
        current_sparsity: float,
        last_sparsity: Optional[float] = None,
        current_step: Optional[int] = None,
    ) -> float:
        """Process a sparsity reading and update lambda.

        Parameters
        ----------
        current_sparsity : float
            Current model sparsity
        last_sparsity : float, optional
            If provided, this value is cached as the last known sparsity.

        Returns
        -------
        float
            Current lambda value
        """
        # Track current step before any early return so the target_sparsity
        # property reflects the correct epoch even during warmup.
        if current_step is not None:
            self._last_step = int(current_step)

        if self.warmup_steps > 0 and current_step <= self.warmup_steps:
            if current_step == 0:
                log.info(
                    f"Warmup phase: Holding lambda at {self.lambda_value:.4f} "
                    f"for {self.warmup_steps} steps ({self.warmup_epochs} epochs)."
                )
            # At this point, self.lambda_value = initial_lambda
            return self.lambda_value

        if self.warmup_steps > 0 and current_step == self.warmup_steps + 1:
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
            raise ValueError(
                f"window_steps must be >= 2, got {window_steps}"
            )
        if min_crossings < 1:
            raise ValueError(
                f"min_crossings must be >= 1, got {min_crossings}"
            )
        if tolerance < 0.0:
            raise ValueError(
                f"tolerance must be >= 0.0, got {tolerance}"
            )
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

        Called by BregmanPruner once the trainer is available.
        """
        self._steps_per_epoch = int(steps_per_epoch)
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.warmup_steps > 0:
            log.info(
                f"Lambda warmup: {self.warmup_epochs} epochs "
                f"× {steps_per_epoch} batches/epoch = {self.warmup_steps} steps"
            )

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "lambda_value": self.lambda_value,
            # Kept for backward compatibility with older checkpoints; readers
            # should prefer `_target_schedule`.
            "target_sparsity": self.target_sparsity,
            "_target_schedule": list(self._target_schedule),
            "_last_step": self._last_step,
            "_last_sparsity": self._last_sparsity,
            "acceleration_factor": self.acceleration_factor,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "warmup_steps": self.warmup_steps,
            "damping_zone": self.damping_zone,
            "max_relative_change": self.max_relative_change,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint.

        Supports both new-style checkpoints (with ``_target_schedule``) and
        legacy ones that only stored the scalar ``target_sparsity``.
        """
        self.lambda_value = state["lambda_value"]

        if "_target_schedule" in state:
            self._target_schedule = _normalize_target_schedule(
                state["_target_schedule"]
            )
        elif "target_sparsity" in state:
            # Legacy scalar-only checkpoint.
            self._target_schedule = _normalize_target_schedule(
                state["target_sparsity"]
            )
        self._last_step = int(state.get("_last_step", self._last_step))

        self._last_sparsity = state.get("_last_sparsity")
        if self._last_sparsity is not None:
            # Ensure restored state is consistent with strict Bregman assumptions.
            self._validate_sparsity(self._last_sparsity)
        self.acceleration_factor = state["acceleration_factor"]
        self.min_lambda = state["min_lambda"]
        self.max_lambda = state["max_lambda"]
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.damping_zone = state.get("damping_zone", self.damping_zone)
        self.max_relative_change = state.get(
            "max_relative_change", self.max_relative_change
        )

        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )
