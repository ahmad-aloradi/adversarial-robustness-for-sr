import math
from numbers import Real
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """Lambda scheduler with optional sparsity EMA smoothing and target
    relaxation.

    This scheduler updates lambda once per call to :meth:`step` (i.e., per-batch when
    called from a batch-end hook). Optionally, it applies exponential moving average
    (EMA) smoothing to sparsity readings to reduce oscillations.

    The scheduler also supports a scheduled target mode, where target_sparsity
    evolves from an initial value to a final value over N epochs (linear or constant
    schedule). Typically used to ramp sparsity upward (e.g., 0.0 → 0.9), mirroring
    how magnitude pruning gradually increases sparsity during training.

    Parameters
    ----------
    initial_lambda : float
        Initial lambda value for regularization
    target_sparsity : float
        Target sparsity level to achieve (ignored when schedule_type is set)
    acceleration_factor : float, default=0.25
        Factor multiplied by the sparsity difference between current and target
        to control how aggressively to update lambda
    min_lambda : float, default=1e-6
        Minimum lambda value
    max_lambda : float, default=1e3
        Maximum lambda value
    ema_decay_factor : float, default=0.9
        EMA coefficient $\beta$ for sparsity smoothing.
        Higher values = more smoothing. Set to 0.0 to disable EMA.
    use_ema : bool, default=True
        Whether to use EMA-smoothed sparsity for lambda updates.
    schedule_type : str, optional
        Schedule type for target relaxation ("linear" or "constant").
        When None, target_sparsity is fixed.
    initial_target_sparsity : float, optional
        Starting target sparsity for scheduled mode (e.g., 0.0; required if schedule_type is set)
    final_target_sparsity : float, optional
        Ending target sparsity for scheduled mode (e.g., 0.9; required if schedule_type is set)
    epochs_to_ramp : int, default=10
        Number of epochs for the schedule to complete
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: float = 0.9,
        acceleration_factor: float = 0.25,
        min_lambda: float = 1e-6,
        max_lambda: float = 1e3,
        ema_decay_factor: float = 0.9,
        use_ema: bool = True,
        schedule_type: Optional[str] = None,
        initial_target_sparsity: Optional[float] = None,
        final_target_sparsity: Optional[float] = None,
        epochs_to_ramp: int = 10,
    ):
        if not (0.0 < target_sparsity <= 1.0):
            raise ValueError(
                f"target_sparsity must be in (0.0, 1.0], got {target_sparsity}"
            )
        if acceleration_factor < 0.0:
            raise ValueError(
                f"acceleration_factor must be >= 0.0, got {acceleration_factor}"
            )
        if min_lambda <= 0.0:
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

        # Scheduled target support
        self._schedule_type = schedule_type
        if schedule_type is not None:
            if (
                initial_target_sparsity is None
                or final_target_sparsity is None
            ):
                raise ValueError(
                    "initial_target_sparsity and final_target_sparsity must be provided "
                    "when schedule_type is set"
                )
            if not (0.0 <= initial_target_sparsity <= 1.0):
                raise ValueError(
                    f"initial_target_sparsity must be in [0.0, 1.0], got {initial_target_sparsity}"
                )
            if not (0.0 <= final_target_sparsity <= 1.0):
                raise ValueError(
                    f"final_target_sparsity must be in [0.0, 1.0], got {final_target_sparsity}"
                )
            if schedule_type not in ("linear", "constant"):
                raise ValueError(
                    f"schedule_type must be 'linear' or 'constant', got {schedule_type}"
                )

            self._initial_target_sparsity = initial_target_sparsity
            self._final_target_sparsity = final_target_sparsity
            self._epochs_to_ramp = epochs_to_ramp
            self._schedule_epoch = 0
            self.target_sparsity = initial_target_sparsity
        else:
            self._initial_target_sparsity = None
            self._final_target_sparsity = None
            self._epochs_to_ramp = epochs_to_ramp
            self._schedule_epoch = 0
            self.target_sparsity = target_sparsity

        # EMA smoothing for sparsity
        self.use_ema = use_ema
        self.ema_decay_factor = float(ema_decay_factor)

        if not (0.0 <= self.ema_decay_factor <= 1.0):
            raise ValueError(
                f"ema_decay_factor must be in [0.0, 1.0], got {self.ema_decay_factor}"
            )
        if self.use_ema and self.ema_decay_factor <= 0.0:
            log.warning(
                "ema_decay_factor <= 0.0 with use_ema=True disables EMA smoothing."
            )
        self._ema_smoothed_sparsity: Optional[float] = None

    def step(
        self,
        current_sparsity: float,
        last_sparsity: Optional[float] = None,
    ) -> float:
        """Process a sparsity reading and update lambda.

        EMA is always updated (if enabled), and lambda is updated once per call.

        Parameters
        ----------
        current_sparsity : float
            Current model sparsity
        last_sparsity : float, optional
            If provided, this value is used as the last known sparsity,
            bypassing the internal state. Useful for resuming.

        Returns
        -------
        float
            Current lambda value
        """
        # If resuming from a checkpoint, use provided last_sparsity
        if last_sparsity is not None:
            self._validate_sparsity(last_sparsity)
            self._ema_smoothed_sparsity = float(last_sparsity)

        self._validate_sparsity(current_sparsity)

        # Update EMA (if enabled)
        if self.use_ema and self.ema_decay_factor > 0.0:
            if self._ema_smoothed_sparsity is None:
                self._ema_smoothed_sparsity = float(current_sparsity)
            else:
                self._ema_smoothed_sparsity = (
                    self.ema_decay_factor * self._ema_smoothed_sparsity
                    + (1 - self.ema_decay_factor) * float(current_sparsity)
                )
        else:
            self._ema_smoothed_sparsity = float(current_sparsity)

        # After the update above, _ema_smoothed_sparsity is always the signal we want:
        # - if EMA enabled: EMA-smoothed sparsity
        # - else: current sparsity (stored into _ema_smoothed_sparsity)
        sparsity_signal = self._ema_smoothed_sparsity
        sparsity_difference = sparsity_signal - self.target_sparsity

        # Update lambda based on smoothed sparsity
        if sparsity_signal < self.target_sparsity:
            # Increase lambda to encourage more sparsity
            self.lambda_value *= 1 + self.acceleration_factor * abs(
                sparsity_difference
            )
        elif sparsity_signal > self.target_sparsity:
            # Decrease lambda since we're above target
            self.lambda_value /= 1 + self.acceleration_factor * abs(
                sparsity_difference
            )

        # Clamp lambda to valid range
        self.lambda_value = max(
            self.min_lambda, min(self.max_lambda, self.lambda_value)
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

    def update_target(self, current_epoch: int) -> float:
        """Update target_sparsity based on schedule (if enabled).

        Parameters
        ----------
        current_epoch : int
            Current training epoch

        Returns
        -------
        float
            Updated target sparsity value
        """
        if self._schedule_type is None:
            return self.target_sparsity

        progress = min(1.0, (current_epoch + 1) / self._epochs_to_ramp)

        if self._schedule_type == "linear":
            target = (
                self._initial_target_sparsity
                + (self._final_target_sparsity - self._initial_target_sparsity)
                * progress
            )
        elif self._schedule_type == "constant":
            # Log-space interpolation (constant pruning rate)
            remaining_final = 1.0 - self._final_target_sparsity
            remaining_initial = 1.0 - self._initial_target_sparsity

            if remaining_final <= 1e-9:
                remaining_final = 1e-9
            if remaining_initial <= 1e-9:
                target = 1.0
            else:
                current_remaining = (
                    remaining_initial
                    * (remaining_final / remaining_initial) ** progress
                )
                target = 1.0 - current_remaining

            # Clamp (match PruningScheduler: clamp to final_target_sparsity)
            target = min(max(target, 0.0), self._final_target_sparsity)
        else:
            raise ValueError(f"Unknown schedule_type: {self._schedule_type}")

        self.target_sparsity = target
        self._schedule_epoch = current_epoch

        log.info(
            f"Target sparsity updated: {self.target_sparsity:.4f} "
            f"(epoch {current_epoch}, schedule progress {progress:.1%})"
        )

        return self.target_sparsity

    @property
    def is_scheduled(self) -> bool:
        """Check if target sparsity follows a schedule."""
        return self._schedule_type is not None

    @property
    def schedule_complete(self) -> bool:
        """Check if schedule has completed."""
        return (
            not self.is_scheduled
            or self._schedule_epoch >= self._epochs_to_ramp - 1
        )

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_ema_smoothed_sparsity(self) -> Optional[float]:
        """Get current EMA-smoothed sparsity value."""
        return self._ema_smoothed_sparsity

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "lambda_value": self.lambda_value,
            "target_sparsity": self.target_sparsity,
            "_ema_smoothed_sparsity": self._ema_smoothed_sparsity,
            "acceleration_factor": self.acceleration_factor,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "use_ema": self.use_ema,
            "ema_decay_factor": self.ema_decay_factor,
            "_schedule_type": self._schedule_type,
            "_initial_target_sparsity": self._initial_target_sparsity,
            "_final_target_sparsity": self._final_target_sparsity,
            "_epochs_to_ramp": self._epochs_to_ramp,
            "_schedule_epoch": self._schedule_epoch,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint."""
        self.lambda_value = state["lambda_value"]
        self.target_sparsity = state["target_sparsity"]
        self._ema_smoothed_sparsity = state.get("_ema_smoothed_sparsity")
        if self._ema_smoothed_sparsity is not None:
            # Ensure restored state is consistent with strict Bregman assumptions.
            self._validate_sparsity(self._ema_smoothed_sparsity)
        self.acceleration_factor = state["acceleration_factor"]
        self.min_lambda = state["min_lambda"]
        self.max_lambda = state["max_lambda"]
        self.use_ema = state["use_ema"]
        self.ema_decay_factor = float(state["ema_decay_factor"])

        # Restore schedule state
        self._schedule_type = state["_schedule_type"]
        self._initial_target_sparsity = state["_initial_target_sparsity"]
        self._final_target_sparsity = state["_final_target_sparsity"]
        self._epochs_to_ramp = state["_epochs_to_ramp"]
        self._schedule_epoch = state["_schedule_epoch"]

        sparsity_ema_str = (
            f"{self._ema_smoothed_sparsity:.4f}"
            if self._ema_smoothed_sparsity is not None
            else "None"
        )
        schedule_info = ""
        if self._schedule_type is not None:
            schedule_info = (
                f", schedule={self._schedule_type}, "
                f"epoch={self._schedule_epoch}/{self._epochs_to_ramp}"
            )
        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}, "
            f"sparsity_ema={sparsity_ema_str}{schedule_info}"
        )
