"""Unit tests for QuantileLambdaScheduler: exact-by-construction lambda from
the K-th order statistic of |v|, in place of LambdaScheduler's feedback
estimate. Mirrors test_bregman_lambda_verification.py's style.
"""

import pytest
import torch

from src.callbacks.pruning.bregman.bregman_optimizers import AdaBreg
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
from src.callbacks.pruning.bregman.quantile_lambda_scheduler import (
    QuantileLambdaScheduler,
)
from src.callbacks.pruning.shared_prune_utils import compute_sparsity


def _controller(target_sparsity=0.5, initial_lambda=1e-2, **kwargs):
    """A scheduler carrying its setpoint, as the config builds it."""
    return QuantileLambdaScheduler(
        target_sparsity=target_sparsity,
        initial_lambda=initial_lambda,
        **kwargs
    )


def _materialize_sub_grad(param: torch.nn.Parameter, reg=None) -> AdaBreg:
    """One zero-gradient AdaBreg step: sub_grad is seeded from param, unmoved."""
    reg = reg or RegL1(lamda=0.0)
    optimizer = AdaBreg(
        [{"params": [param], "reg": reg, "lambda_scale": 1.0}], lr=1e-2
    )
    param.grad = torch.zeros_like(param)
    optimizer.step()
    return optimizer


def test_bind_computes_population_size():
    """N is the summed numel of the bound params, available before any step."""
    a = torch.nn.Parameter(torch.randn(4))
    b = torch.nn.Parameter(torch.randn(6))
    optimizer = _materialize_sub_grad(a)
    optimizer.add_param_group(
        {"params": [b], "reg": RegL1(lamda=0.0), "lambda_scale": 1.0}
    )

    scheduler = _controller()
    scheduler.bind(optimizer, [a, b])

    assert scheduler.n == 10


def test_bind_rejects_empty_params():
    """An empty regularized-param list is a config bug, not a silent no-op."""
    optimizer = _materialize_sub_grad(torch.nn.Parameter(torch.randn(4)))
    scheduler = _controller()

    with pytest.raises(AssertionError, match="needs regularized parameters"):
        scheduler.bind(optimizer, [])


def test_step_before_bind_raises():
    """Step() needs the optimizer bind() attaches; nothing to threshold without
    it."""
    scheduler = _controller()

    with pytest.raises(AssertionError, match="called before bind"):
        scheduler.step(current_sparsity=0.0, current_step=0)


def test_step_selects_exact_k_survivors():
    """The prox at the returned lambda keeps exactly K of N, by
    construction."""
    w = torch.nn.Parameter(torch.linspace(0.1, 1.0, steps=10))
    reg = RegL1(lamda=0.0)
    optimizer = _materialize_sub_grad(
        w, reg=reg
    )  # sub_grad == w exactly (zero grad, zero reg)
    scheduler = _controller(target_sparsity=0.5)
    scheduler.bind(optimizer, [w])

    lam = scheduler.step(current_sparsity=0.0, current_step=0)

    w.data.copy_(reg.prox(w.data, delta=1.0, lamda=lam))
    assert compute_sparsity([w], threshold=1e-12) == pytest.approx(0.5)
    assert scheduler.last_k == 5


def test_scale_aware_threshold_matches_lambda_scale():
    """Order statistic is on |v|/lambda_scale, matching RegL1.prox's own
    survive rule."""
    a = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    b = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    reg_a, reg_b = RegL1(lamda=0.0), RegL1(lamda=0.0)
    optimizer = AdaBreg(
        [
            {"params": [a], "reg": reg_a, "lambda_scale": 1.0},
            {"params": [b], "reg": reg_b, "lambda_scale": 2.0},
        ],
        lr=1e-2,
    )
    a.grad, b.grad = torch.zeros_like(a), torch.zeros_like(b)
    optimizer.step()  # sub_grad == a, b exactly

    scheduler = _controller(target_sparsity=0.5)  # k = round(0.5 * 8) = 4
    scheduler.bind(optimizer, [a, b])
    lam = scheduler.step(current_sparsity=0.0, current_step=0)

    a.data.copy_(reg_a.prox(a.data, delta=1.0, lamda=lam))
    b.data.copy_(reg_b.prox(b.data, delta=1.0, lamda=lam * 2.0))
    survivors = int((a.abs() > 0).sum()) + int((b.abs() > 0).sum())
    assert survivors == 4  # not the 3-vs-5 split raw-|v| ranking would give


def test_step_updates_only_every_update_frequency_steps():
    """Lambda moves on steps divisible by update_frequency, not in between.

    Needs sub_grad to actually evolve between update instants (unlike the other
    tests' single zero-gradient step) to tell "cached" from "recomputed but
    coincidentally equal".
    """
    w = torch.nn.Parameter(torch.randn(50))
    reg = RegL1(lamda=0.0)
    optimizer = AdaBreg(
        [{"params": [w], "reg": reg, "lambda_scale": 1.0}], lr=1e-2
    )
    scheduler = _controller(update_frequency=10)
    scheduler.bind(optimizer, [w])

    values = []
    for step in range(100):
        w.grad = torch.randn_like(w) * 0.1
        optimizer.step()
        values.append(scheduler.step(0.0, step))

    assert len(set(values)) == 10


def test_get_state_load_state_round_trip():
    """lambda_value is the only state a checkpoint has to carry."""
    w = torch.nn.Parameter(torch.randn(20))
    optimizer = _materialize_sub_grad(w)
    scheduler = _controller(initial_lambda=1.0)
    scheduler.bind(optimizer, [w])
    scheduler.step(current_sparsity=0.0, current_step=0)

    restored = _controller(initial_lambda=0.5)
    restored.load_state(scheduler.get_state())

    assert restored.get_lambda() == scheduler.get_lambda()


def test_constructor_rejects_target_sparsity_outside_unit_interval():
    """A scheduler that could never band its target must not be built."""
    with pytest.raises(AssertionError, match="target_sparsity must be in"):
        _controller(target_sparsity=1.5)


def test_accepted_but_unused_kwargs_do_not_change_lambda():
    """initial_sparsity/alpha_0 ride along from LambdaScheduler's config node
    but must not perturb the quantile threshold."""
    w = torch.nn.Parameter(torch.randn(30))
    optimizer_a = _materialize_sub_grad(w.clone().detach().requires_grad_())
    scheduler_a = _controller(initial_sparsity=0.0, alpha_0=1.0)
    scheduler_a.bind(optimizer_a, list(optimizer_a.param_groups[0]["params"]))

    optimizer_b = _materialize_sub_grad(w.clone().detach().requires_grad_())
    scheduler_b = _controller(initial_sparsity=0.99, alpha_0=0.25)
    scheduler_b.bind(optimizer_b, list(optimizer_b.param_groups[0]["params"]))

    lam_a = scheduler_a.step(current_sparsity=0.0, current_step=0)
    lam_b = scheduler_b.step(current_sparsity=0.0, current_step=0)

    assert lam_a == pytest.approx(lam_b)


if __name__ == "__main__":
    # Smoke: exercised by pytest above; this mirrors CLAUDE.md's
    # standalone-debug convention for the module it tests.
    test_step_selects_exact_k_survivors()
    print("QuantileLambdaScheduler unit tests: smoke pass")
