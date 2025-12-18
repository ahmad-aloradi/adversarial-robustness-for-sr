from typing import Optional
from src import utils

log = utils.get_pylogger(__name__)


class LambdaScheduler:
    """
    Lambda scheduler with optional sparsity EMA smoothing.

    This scheduler updates lambda once per call to :meth:`step` (i.e., per-batch when
    called from a batch-end hook). Optionally, it applies exponential moving average
    (EMA) smoothing to sparsity readings to reduce oscillations.

    Parameters
    ----------
    initial_lambda : float
        Initial lambda value for regularization
    target_sparsity : float
        Target sparsity level to achieve
    acceleration_factor : float, default=0.25
        Factor multiplied by the sparsity difference between current and target
        to control how aggressively to update lambda
    min_lambda : float, default=1e-6
        Minimum lambda value
    max_lambda : float, default=1e3
        Maximum lambda value
    sparsity_ema_decay : float, default=0.9
        Decay factor for exponential moving average of sparsity readings.
        Higher values = more smoothing. Set to 0.0 to disable EMA.
    use_ema : bool, default=True
        Whether to use EMA-smoothed sparsity for lambda updates.
    """

    def __init__(
        self,
        initial_lambda: float = 1e-3,
        target_sparsity: float = 0.9,
        acceleration_factor: float = 0.25,
        min_lambda: float = 1e-6,
        max_lambda: float = 1e3,
        sparsity_ema_decay: float = 0.9,
        use_ema: bool = True,
    ):
        self.lambda_value = initial_lambda
        self.target_sparsity = target_sparsity
        self.acceleration_factor = acceleration_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # EMA smoothing for sparsity
        self.use_ema = use_ema
        self.sparsity_ema_decay = sparsity_ema_decay
        if self.use_ema and self.sparsity_ema_decay <= 0.0:
            log.warning(
                "sparsity_ema_decay <= 0.0 with use_ema=True disables EMA smoothing."
            )
        self._sparsity_ema: Optional[float] = None
        self._last_sparsity: Optional[float] = None


    def step(
        self,
        current_sparsity: float,
        last_sparsity: Optional[float] = None,
    ) -> float:
        """
        Process a sparsity reading and update lambda.

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
            self._last_sparsity = last_sparsity
            self._sparsity_ema = last_sparsity

        # Handle spurious zero readings
        effective_sparsity = self._get_sparsity(current_sparsity)
        
        # Update EMA (if enabled)
        if self.use_ema and self.sparsity_ema_decay > 0.0:
            if self._sparsity_ema is None:
                self._sparsity_ema = effective_sparsity
            else:
                self._sparsity_ema = (
                    self.sparsity_ema_decay * self._sparsity_ema
                    + (1 - self.sparsity_ema_decay) * effective_sparsity
                )
        else:
            # Keep EMA state consistent for logging/inspection.
            self._sparsity_ema = effective_sparsity
        
        # Store valid sparsity reading
        if current_sparsity > 0.0:
            self._last_sparsity = current_sparsity

        # Choose sparsity signal for lambda update
        sparsity_signal = self._sparsity_ema if (self.use_ema and self._sparsity_ema is not None) else effective_sparsity
        sparsity_difference = sparsity_signal - self.target_sparsity

        # Update lambda based on smoothed sparsity
        if sparsity_signal < self.target_sparsity:
            # Increase lambda to encourage more sparsity
            self.lambda_value *= 1 + self.acceleration_factor * abs(sparsity_difference)
        elif sparsity_signal > self.target_sparsity:
            # Decrease lambda since we're above target
            self.lambda_value /= 1 + self.acceleration_factor * abs(sparsity_difference)

        # Clamp lambda to valid range
        self.lambda_value = max(self.min_lambda, min(self.max_lambda, self.lambda_value))
        
        return self.lambda_value

    def _get_sparsity(self, current_sparsity: float) -> float:
        """
        Get model sparsity with safety mechanism for spurious zeros.

        Parameters
        ----------
        current_sparsity : float
            Raw sparsity reading

        Returns
        -------
        float
            Effective sparsity after safety check
        """
        # If current reading is exactly 0.0 and we have a valid last reading,
        # use the last reading to avoid spurious zeros
        if current_sparsity == 0.0 and self._last_sparsity is not None:
            log.warning(
                f"Spurious zero sparsity detected, using last valid reading: "
                f"{self._last_sparsity:.4f}"
            )
            return self._last_sparsity

        return current_sparsity

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value
    
    def get_smoothed_sparsity(self) -> Optional[float]:
        """Get current EMA-smoothed sparsity value."""
        return self._sparsity_ema

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'lambda_value': self.lambda_value,
            'target_sparsity': self.target_sparsity,
            '_last_sparsity': self._last_sparsity,
            '_sparsity_ema': self._sparsity_ema,
            'acceleration_factor': self.acceleration_factor,
            'min_lambda': self.min_lambda,
            'max_lambda': self.max_lambda,
            'use_ema': self.use_ema,
            'sparsity_ema_decay': self.sparsity_ema_decay,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from a checkpoint."""
        self.lambda_value = state['lambda_value']
        self.target_sparsity = state['target_sparsity']
        self._last_sparsity = state.get('_last_sparsity')
        self._sparsity_ema = state.get('_sparsity_ema')
        self.acceleration_factor = state['acceleration_factor']
        self.min_lambda = state['min_lambda']
        self.max_lambda = state['max_lambda']
        self.use_ema = state.get('use_ema', True)
        self.sparsity_ema_decay = state.get('sparsity_ema_decay', 0.9)
        sparsity_ema_str = f"{self._sparsity_ema:.4f}" if self._sparsity_ema is not None else "None"
        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}, "
            f"sparsity_ema={sparsity_ema_str}"
        )
