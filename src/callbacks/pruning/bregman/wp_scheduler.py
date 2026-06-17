"""
Global explore->commit annealer for the per-element Bernoulli probability
``w_p`` used by the LinBreg/AdaBreg primal readout.

``w_p`` in [0, 1] is the keep-probability of the per-element Bernoulli gate:
each step a weight takes its standard Bregman prox step with probability
``w_p``, or is frozen at its current value (a zero stays zero) with probability
``1 - w_p``. ``w_p = 1`` is fully reversible standard Bregman; ``w_p = 0``
latches the support. Annealing ``w_p`` from ~1 down to ~0 over training keeps
the support reversible early (exploration; overshoots self-heal) and freezes it
late (commitment; stable, non-oscillating mask).

The annealer is a pure function of training progress in [0, 1]. It is
independent of the lambda scheduler: the BregmanPruner supplies progress from
``trainer.global_step / trainer.estimated_stepping_batches`` and pushes the
value onto each regularizer's ``w_p`` attribute.

    >>> ann = WpAnnealer(schedule="linear")
    >>> ann.value_at(0.0), ann.value_at(0.5), ann.value_at(1.0)
    (1.0, 0.5, 0.0)
"""
import math


class WpAnnealer:
    """Anneal ``w_p`` from ``w_p_init`` to ``w_p_final`` over a training window.

    Parameters
    ----------
    w_p_init : float, default=1.0
        Value held before ``start_fraction``. 1.0 reproduces standard Bregman.
    w_p_final : float, default=0.0
        Value held after ``end_fraction``. 0.0 fully latches the support.
    start_fraction : float, default=0.0
        Training-progress fraction at which the anneal begins; ``w_p`` is held
        at ``w_p_init`` over ``[0, start_fraction)``. With ``w_p_init=1`` this
        is the *reversible explore window* — the readout is exact standard
        Bregman, so the support can still migrate (dead weights revive,
        overshoots self-heal). ``start_fraction = 0`` removes the held window;
        the anneal begins immediately (``cosine`` still lingers near the init
        for a few steps via its zero start-slope, ``linear`` does not).
    end_fraction : float, default=1.0
        Training-progress fraction at which the anneal completes; ``w_p`` is
        held at ``w_p_final`` over ``(end_fraction, 1]``. With ``w_p_final=0``
        this is the *commit window* where the support is fully latched.
    schedule : {"linear", "cosine"}, default="cosine"
        Interpolation between the endpoints. "cosine" is smooth (zero slope at
        both ends); "linear" is constant-rate.

    Notes
    -----
    The anneal does not assume a dense start. When a run begins from a nonzero
    initial sparsity (a random mask), it is the reversible explore window
    (``start_fraction`` with ``w_p_init = 1``), not a dense init, that lets the
    support redistribute off that mask before it latches. Shrinking
    ``start_fraction`` toward 0 on a sparse start freezes that random mask
    early, so keep ``start_fraction > 0`` whenever the model starts sparse.
    """

    def __init__(
        self,
        w_p_init: float = 1.0,
        w_p_final: float = 0.0,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
        schedule: str = "cosine",
    ):
        for name, value in (("w_p_init", w_p_init), ("w_p_final", w_p_final)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not (0.0 <= start_fraction < end_fraction <= 1.0):
            raise ValueError(
                "require 0 <= start_fraction < end_fraction <= 1, got "
                f"start={start_fraction}, end={end_fraction}"
            )
        if schedule not in ("linear", "cosine"):
            raise ValueError(
                f"schedule must be 'linear' or 'cosine', got {schedule!r}"
            )
        self.w_p_init = float(w_p_init)
        self.w_p_final = float(w_p_final)
        self.start_fraction = float(start_fraction)
        self.end_fraction = float(end_fraction)
        self.schedule = schedule

    def value_at(self, progress: float) -> float:
        """``w_p`` at a training-progress fraction in [0, 1]."""
        assert (
            0.0 <= progress <= 1.0
        ), f"progress must be in [0, 1], got {progress}"
        if progress <= self.start_fraction:
            return self.w_p_init
        if progress >= self.end_fraction:
            return self.w_p_final
        local = (progress - self.start_fraction) / (
            self.end_fraction - self.start_fraction
        )
        if self.schedule == "linear":
            frac = local
        else:
            frac = 0.5 * (1.0 - math.cos(math.pi * local))
        return self.w_p_init + (self.w_p_final - self.w_p_init) * frac


if __name__ == "__main__":
    for sched in ("linear", "cosine"):
        ann = WpAnnealer(schedule=sched, start_fraction=0.1, end_fraction=0.9)
        row = " ".join(
            f"{p:.2f}->{ann.value_at(p):.3f}"
            for p in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        print(f"{sched:6s}: {row}")

    for sched in ("linear", "cosine"):
        ann = WpAnnealer(schedule=sched, start_fraction=0.0, end_fraction=1.0)
        row = " ".join(
            f"{p:.2f}->{ann.value_at(p):.3f}"
            for p in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        print(f"{sched:6s}: {row}")
