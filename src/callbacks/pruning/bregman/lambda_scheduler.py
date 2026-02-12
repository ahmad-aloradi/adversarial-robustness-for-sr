import math
from numbers import Real
from typing import Optional

from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """Lambda scheduler for Bregman target sparsity control.

    This scheduler updates lambda once per call to :meth:`step` (i.e., per-batch when
    called from a batch-end hook). The scheduler maintains a fixed `target_sparsity` 
    and updates lambda once per call to :meth:`step` (typically called at each batch end).

    Parameters
    ----------
    initial_lambda : float
        Initial lambda value for regularization
    target_sparsity : float
        Target sparsity level to achieve.
    acceleration_factor : float, default=0.25
        Factor multiplied by the sparsity difference between current and target
        to control how aggressively to update lambda
    min_lambda : float, default=1e-6
        Minimum lambda value
    max_lambda : float, default=1e3
        Maximum lambda value
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: float = 0.9,
        acceleration_factor: float = 0.25,
        min_lambda: float = 1e-6,
        max_lambda: float = 1e3,
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
        self._last_sparsity = None
        self.target_sparsity = target_sparsity

    def step(
        self,
        current_sparsity: float,
        last_sparsity: Optional[float] = None,
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
        # If resuming from a checkpoint, use provided last_sparsity
        if last_sparsity is not None:
            self._validate_sparsity(last_sparsity)
            self._last_sparsity = float(last_sparsity)

        self._validate_sparsity(current_sparsity)

        sparsity_signal = float(current_sparsity)
        self._last_sparsity = sparsity_signal
        sparsity_difference = sparsity_signal - self.target_sparsity

        if sparsity_signal < self.target_sparsity:
            # Increase lambda to encourage more sparsity
            self.lambda_value *= 1 + self.acceleration_factor * abs(sparsity_difference)
        elif sparsity_signal > self.target_sparsity:
            # Decrease lambda since we're above target
            self.lambda_value /= 1 + self.acceleration_factor * abs(sparsity_difference)

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


    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "lambda_value": self.lambda_value,
            "target_sparsity": self.target_sparsity,
            "_last_sparsity": self._last_sparsity,
            "acceleration_factor": self.acceleration_factor,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint."""
        self.lambda_value = state["lambda_value"]
        self.target_sparsity = state["target_sparsity"]
        self._last_sparsity = state.get("_last_sparsity")
        if self._last_sparsity is not None:
            # Ensure restored state is consistent with strict Bregman assumptions.
            self._validate_sparsity(self._last_sparsity)
        self.acceleration_factor = state["acceleration_factor"]
        self.min_lambda = state["min_lambda"]
        self.max_lambda = state["max_lambda"]

        log.info(f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}")
