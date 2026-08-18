import math
from typing import List, Optional

import torch

from src import utils
from src.callbacks.pruning.bregman.bregman_regularizers import lambda_scale

log = utils.get_pylogger(__name__)


class QuantileLambdaScheduler:
    """Sets λ to the K-th order statistic of ``|v|``, K from
    ``target_sparsity``.

    Unlike ``LambdaScheduler``, λ is read off the current dual distribution
    instead of estimated from a measured sparsity, so the prox keeps exactly
    K of N regularized weights every time :meth:`step` fires. λ follows the
    order statistic in both directions (docs/bregman_phantom_classes.md).

    :meth:`bind` must run once before :meth:`step`: computing K needs each
    param's Bregman dual (``optimizer.state[p]["sub_grad"]``), which
    ``step``'s ``(sparsity, step)`` signature has no path to.

    >>> import torch
    >>> from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
    >>> from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
    >>> w = torch.nn.Parameter(torch.linspace(0.1, 1.0, steps=10))
    >>> opt = AdaBreg([{"params": [w], "reg": RegL1(lamda=0.0), "lambda_scale": 1.0}], lr=1e-2)
    >>> w.grad = torch.zeros_like(w)
    >>> _ = opt.step()  # materializes sub_grad == w (zero grad, zero reg)
    >>> sched = QuantileLambdaScheduler(target_sparsity=0.5, initial_lambda=0.01)
    >>> sched.bind(opt, [w])
    >>> lam = sched.step(current_sparsity=0.0, current_step=0)
    >>> sched.last_k, sched.n  # keep the top half of 10
    (5, 10)
    """

    def __init__(
        self,
        target_sparsity: float,
        initial_lambda: float = 1e-2,
        update_frequency: int = 50,
        initial_sparsity: float = 0.0,  # accepted, unused: LambdaScheduler's config node merges this in
        alpha_0: float = 1.0,  # accepted, unused: same reason -- no feedback gain to seed here
    ):
        assert (
            0.0 <= target_sparsity <= 1.0
        ), f"target_sparsity must be in [0.0, 1.0], got {target_sparsity}"
        assert (
            initial_lambda > 0.0
        ), f"initial_lambda must be > 0.0, got {initial_lambda}"
        assert (
            update_frequency >= 1
        ), f"update_frequency must be >= 1, got {update_frequency}"

        self.lambda_value = initial_lambda
        self.target_sparsity = target_sparsity
        self.update_frequency = update_frequency

        self.last_k = 0  # survivors the last step() selected
        self.n = 0  # total regularized elements; set by bind()
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._params: List[torch.nn.Parameter] = []
        self._scale: List[float] = []

    def bind(
        self,
        optimizer: torch.optim.Optimizer,
        params: List[torch.nn.Parameter],
    ) -> None:
        """Attach the optimizer and the params this scheduler thresholds.

        Why: RegL1.prox's survive condition is ``|v| > lambda * lambda_scale``,
        so per-param scale must be captured once here to normalize the order
        statistic correctly if ``lambda_scale`` is ever non-uniform.
        """
        assert (
            len(params) > 0
        ), f"QuantileLambdaScheduler.bind() needs regularized parameters, got {len(params)}"
        # Only look up scale for the passed (already-regularized) params: an
        # unregularized group (e.g. RegNone) may skip lambda_scale entirely,
        # since nothing upstream needs it there either (bregman_regularizers.is_regularized).
        param_ids = {id(p) for p in params}
        scale_by_id = {
            id(p): lambda_scale(group)
            for group in optimizer.param_groups
            for p in group["params"]
            if id(p) in param_ids
        }
        self._optimizer = optimizer
        self._params = list(params)
        self._scale = [scale_by_id[id(p)] for p in self._params]
        self.n = sum(p.numel() for p in self._params)

    def step(self, current_sparsity: float, current_step: int) -> float:
        """Move λ to the exact K-th-largest-|v| threshold for this update.

        Inputs:
            current_sparsity: unused -- the quantile computes its own exact
                count instead of trusting a measurement.
            current_step: global step; updates land on multiples of
                ``update_frequency``.

        Output:
            Current lambda value.
        """
        if current_step % self.update_frequency != 0:
            return self.lambda_value

        assert (
            self._optimizer is not None
        ), "QuantileLambdaScheduler.step() called before bind()"
        v = torch.cat(
            [
                (self._optimizer.state[p]["sub_grad"].abs() / scale).flatten()
                for p, scale in zip(self._params, self._scale)
            ]
        )
        k = max(1, min(self.n, round((1.0 - self.target_sparsity) * self.n)))
        self.last_k = k
        # topk, not kthvalue: CUDA kthvalue runs one thread block per slice, so it takes 113 ms on a flat 21M slice against 0.6 ms.
        self.lambda_value = (
            0.0
            if k >= self.n
            else torch.topk(v, k + 1, sorted=False).values.min().item()
        )

        assert math.isfinite(
            self.lambda_value
        ), f"Infinite λ: {self.lambda_value}"
        return self.lambda_value

    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value

    def get_state(self) -> dict:
        """Checkpoint state: the live lambda. ``n``/``last_k`` are recomputed
        by ``bind()`` from the live model on every run, fresh or resumed, so
        persisting them would risk drifting from what's loaded.
        """
        return {"lambda_value": self.lambda_value}

    def load_state(self, state: dict) -> None:
        """Restore lambda; the rest comes from config and the next
        ``bind()``."""
        self.lambda_value = state["lambda_value"]
        log.info(
            f"QuantileLambdaScheduler state restored. lambda={self.lambda_value:.4f}"
        )


if __name__ == "__main__":
    # Smoke: a widening population of dual values, thresholded to a fixed 30% target.
    from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
    from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
    from src.callbacks.pruning.shared_prune_utils import compute_sparsity

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(2000))
    reg = RegL1(lamda=0.01)
    optimizer = AdaBreg(
        [{"params": [w], "reg": reg, "lambda_scale": 1.0}], lr=1e-2
    )
    sched = QuantileLambdaScheduler(
        target_sparsity=0.3, initial_lambda=0.01, update_frequency=1
    )
    sched.bind(optimizer, [w])
    for step in range(5):
        w.grad = torch.randn_like(w) * 0.1
        optimizer.step()
        reg.lamda = sched.step(
            current_sparsity=0.0, current_step=step
        )  # broadcast, as BregmanPruner does
        achieved = compute_sparsity([w], threshold=1e-12)
        print(
            f"step {step}: n={sched.n} k={sched.last_k} lambda={reg.lamda:.4f} achieved_sparsity={achieved:.4f}"
        )
