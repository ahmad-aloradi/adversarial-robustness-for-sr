import math

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

    The step size ``α`` starts at ``alpha_0`` and, while ``decay_alpha`` is set,
    decays by ``gamma`` on every overshoot, so λ settles instead of ringing.
    The first crossing is the approach from ``initial_sparsity``, not an
    overshoot, so it leaves α alone (see :meth:`update_alpha`).

    >>> sched = LambdaScheduler(0.9, initial_sparsity=0.5, initial_lambda=1.0)
    >>> sched.step(0.5, current_step=0) > 1.0  # below target -> grows
    True
    >>> sched.update_alpha(-0.1) == 0.25  # approach ends -> α holds
    True
    >>> sched.update_alpha(0.05) < 0.25  # overshoot -> α decays
    True
    """

    def __init__(
        self,
        target_sparsity: float,
        initial_sparsity: float,
        initial_lambda: float = 1e-3,
        alpha_0: float = 0.25,
        gamma: float = 0.95,
        update_frequency: int = 1,
    ):
        assert (
            0.0 <= target_sparsity <= 1.0
        ), f"target_sparsity must be in [0.0, 1.0], got {target_sparsity}"
        assert (
            0.0 <= initial_sparsity <= 1.0
        ), f"initial_sparsity must be in [0.0, 1.0], got {initial_sparsity}"
        assert (
            initial_lambda > 0.0
        ), f"initial_lambda must be > 0.0, got {initial_lambda}"
        assert alpha_0 >= 0.0, f"alpha_0 must be >= 0.0, got {alpha_0}"
        assert 0.0 < gamma < 1.0, f"gamma must be in (0.0, 1.0), got {gamma}"
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"

        self.lambda_value = initial_lambda
        self.target_sparsity = target_sparsity
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.update_frequency = update_frequency

        self.last_delta = 0.0  # Δλ the last step() applied
        self.last_delta_over_lambda = 0.0  # Δλ/λ the last step() applied
        self.alpha = alpha_0  # α the last update scaled by
        self.crossings = 0  # updates whose gap flipped sign; the first is the approach
        self.decay_alpha = True  # False while a TargetScheduler ramp moves the setpoint
        self.gap = target_sparsity - initial_sparsity
        self.prev_gap = self.gap

    def update_alpha(self, gap: float) -> float:
        """Record ``gap``; while ``decay_alpha``, set α to
        ``alpha_0 · gamma**overshoots`` on every sign flip.

        Sparsity starts at ``initial_sparsity``, on one side of the setpoint,
        so its first crossing is the approach. Overshoot follows it.
        """
        self.prev_gap, self.gap = self.gap, gap
        if self.decay_alpha:
            if self.gap * self.prev_gap < 0.0:
                self.crossings += 1
            overshoots = max(self.crossings - 1, 0)
            self.alpha = self.alpha_0 * self.gamma**overshoots
        return self.alpha

    def step(self, current_sparsity: float, current_step: int) -> float:
        """Move λ one update toward ``target_sparsity``.

        Inputs:
            current_sparsity: measured model sparsity in [0, 1].
            current_step: global step; updates land on multiples of
                ``update_frequency``.

        Output:
            Current lambda value.
        """
        if not 0.0 <= current_sparsity <= 1.0:
            raise ValueError(
                f"Sparsity must be in [0.0, 1.0], got {current_sparsity}"
            )

        # Zero between updates so the logged series sums exactly.
        self.last_delta = 0.0
        self.last_delta_over_lambda = 0.0

        # Update λ only every ``update_frequency`` steps.
        if current_step % self.update_frequency != 0:
            return self.lambda_value

        # λ_t+1 = λ_t * (1 + α * |gap|)^sign(gap); gap := target - current
        gap = self.target_sparsity - float(current_sparsity)
        alpha = self.update_alpha(gap)
        factor = 1.0 + alpha * abs(gap)
        delta = self.lambda_value * (
            factor - 1.0 if gap >= 0 else 1.0 / factor - 1.0
        )
        self.last_delta_over_lambda = delta / self.lambda_value
        self.lambda_value += delta
        self.last_delta = delta

        assert math.isfinite(self.lambda_value), f"Infinite λ: {self.lambda_value}"
        return self.lambda_value

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Checkpoint state: the live lambda and the α decay bookkeeping."""
        return {
            "lambda_value": self.lambda_value,
            "alpha": self.alpha,
            "crossings": self.crossings,
            "gap": self.gap,
            "prev_gap": self.prev_gap,
        }

    def load_state(self, state: dict) -> None:
        """Restore lambda and the α decay bookkeeping; rest comes from
        config."""
        self.lambda_value = state["lambda_value"]
        self.alpha = state["alpha"]
        self.crossings = state["crossings"]
        self.gap = state["gap"]
        self.prev_gap = state["prev_gap"]
        log.info(
            f"LambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )


if __name__ == "__main__":
    # Smoke: sparsity lags the target, so lambda climbs until the gap closes.
    sched = LambdaScheduler(
        target_sparsity=0.9,
        initial_sparsity=0.2,
        initial_lambda=1.0,
        alpha_0=1.0,
        update_frequency=50,
    )
    measured = 0.2
    for current_step in range(0, 1000, 50):
        lam = sched.step(measured, current_step)
        measured = min(0.9, measured + 0.05)  # model approaches the target
        print(
            f"step {current_step:4d}: sparsity={measured:.3f} "
            f"lambda={lam:.4f} gap={sched.gap:+.3f} alpha={sched.alpha:.4f} "
            f"crossings={sched.crossings} dlambda={sched.last_delta:.4f}"
        )
    print(f"final lambda={sched.get_lambda():.4f}")
