"""Tests for the Bernoulli w_p readout and the w_p anneal.

Covers:
- WpAnnealer: endpoints, monotonicity, linear/cosine shape, active window.
- bernoulli_masked_readout:
    * w_p = 1 returns the standard prox step for every element.
    * w_p = 0 freezes every element at its current value (a zero stays zero).
    * 0 < w_p < 1 picks one of the two exact endpoints per element (never a
      blend); the fraction taking the standard step tracks w_p.
- optimizer step honoring reg.bernoulli_mask (LinBreg, AdaBreg, AdaBregW,
  AdaBregL2): w_p = 1 reproduces the unmasked step; the readout stays NaN-free
  under negative duals.
- the BregmanPruner anneals w_p when a wp_annealer is configured.
"""
import pytest
import torch

from src.callbacks.pruning.bregman.bregman_optimizers import (
    AdaBreg,
    AdaBregL2,
    AdaBregW,
    LinBreg,
    bernoulli_masked_readout,
)
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1
from src.callbacks.pruning.bregman.wp_scheduler import WpAnnealer


# --------------------------------------------------------------------------- #
# WpAnnealer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_annealer_endpoints(schedule):
    ann = WpAnnealer(w_p_init=1.0, w_p_final=0.0, schedule=schedule)
    assert ann.value_at(0.0) == 1.0
    assert ann.value_at(1.0) == 0.0


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_annealer_monotonic_non_increasing(schedule):
    ann = WpAnnealer(w_p_init=1.0, w_p_final=0.0, schedule=schedule)
    grid = [i / 50 for i in range(51)]
    values = [ann.value_at(p) for p in grid]
    for prev, cur in zip(values, values[1:]):
        assert cur <= prev + 1e-12
    assert all(0.0 <= v <= 1.0 for v in values)


def test_annealer_midpoints():
    # Both schedules pass through the endpoint average at the window midpoint.
    assert WpAnnealer(schedule="linear").value_at(0.5) == pytest.approx(0.5)
    assert WpAnnealer(schedule="cosine").value_at(0.5) == pytest.approx(0.5)


def test_annealer_window_holds_outside():
    ann = WpAnnealer(start_fraction=0.2, end_fraction=0.8, schedule="linear")
    assert ann.value_at(0.1) == 1.0  # before the window: held at init
    assert ann.value_at(0.9) == 0.0  # after the window: held at final
    assert ann.value_at(0.5) == pytest.approx(0.5)  # halfway through window


def test_annealer_rejects_bad_args():
    with pytest.raises(ValueError):
        WpAnnealer(w_p_init=1.5)
    with pytest.raises(ValueError):
        WpAnnealer(start_fraction=0.8, end_fraction=0.2)
    with pytest.raises(ValueError):
        WpAnnealer(schedule="quadratic")


# --------------------------------------------------------------------------- #
# bernoulli_masked_readout (per-element gate)
# --------------------------------------------------------------------------- #
def test_readout_wp1_is_standard():
    """w_p = 1 -> Bernoulli(1) is all ones -> the standard prox step."""
    standard = torch.randn(8, 8)
    weight = torch.randn(8, 8)
    out = bernoulli_masked_readout(standard, weight, 1.0)
    assert torch.equal(out, standard)


def test_readout_wp0_freezes():
    """w_p = 0 -> Bernoulli(0) is all zeros -> freeze at current value; a zero
    weight stays zero."""
    standard = torch.randn(2, 2)
    weight = torch.tensor([[0.0, 0.5], [0.0, -0.3]])
    out = bernoulli_masked_readout(standard, weight, 0.0)
    assert torch.equal(out, weight)


def test_readout_picks_one_of_two_exact_endpoints():
    """For 0 < w_p < 1 every element equals EITHER the standard step OR the
    frozen weight -- never an in-between blend."""
    torch.manual_seed(0)
    standard = torch.randn(64, 64)
    weight = torch.randn(64, 64)  # distinct from standard generically
    out = bernoulli_masked_readout(standard, weight, 0.5)
    is_std = torch.isclose(out, standard, atol=1e-6)
    is_w = torch.isclose(out, weight, atol=1e-6)
    assert (is_std | is_w).all()
    assert is_std.any() and is_w.any()  # both branches occur at w_p=0.5


def test_readout_fraction_tracks_wp():
    """The fraction of elements taking the standard step approximates w_p."""
    torch.manual_seed(0)
    standard = torch.randn(200, 200)
    weight = torch.randn(200, 200)
    out = bernoulli_masked_readout(standard, weight, 0.3)
    frac_std = torch.isclose(out, standard, atol=1e-6).float().mean().item()
    assert abs(frac_std - 0.3) < 0.05


# --------------------------------------------------------------------------- #
# Optimizer step honoring reg.bernoulli_mask
# --------------------------------------------------------------------------- #
def _run_single_step(opt_cls, bernoulli_mask, w_p, init, grad, **opt_kwargs):
    """One optimizer step on a fresh param; returns the updated weight."""
    p = torch.nn.Parameter(init.clone())
    reg = RegL1(lamda=0.01)
    reg.bernoulli_mask = bernoulli_mask
    reg.w_p = w_p
    opt = opt_cls([p], lr=0.1, reg=reg, delta=1.0, **opt_kwargs)
    p.grad = grad.clone()
    opt.step()
    return p.detach().clone()


@pytest.mark.parametrize("opt_cls", [LinBreg, AdaBreg, AdaBregW, AdaBregL2])
def test_wp1_matches_unmasked_step(opt_cls):
    """w_p = 1 with the mask on reproduces the unmasked step exactly."""
    torch.manual_seed(0)
    init = torch.randn(8, 8)
    grad = torch.randn(8, 8)
    unmasked = _run_single_step(opt_cls, False, 1.0, init, grad)
    masked = _run_single_step(opt_cls, True, 1.0, init, grad)
    assert torch.allclose(unmasked, masked, atol=1e-6)


@pytest.mark.parametrize("opt_cls", [LinBreg, AdaBreg])
def test_wp0_freezes_weight(opt_cls):
    """w_p = 0 freezes every weight at its pre-step value (zeros stay zero)."""
    init = torch.tensor([[0.0, 0.5], [0.0, -0.3]])
    grad = torch.randn(2, 2)
    out = _run_single_step(opt_cls, True, 0.0, init, grad)
    assert torch.equal(out, init)


@pytest.mark.parametrize("opt_cls", [LinBreg, AdaBreg])
def test_no_nan_with_negative_dual(opt_cls):
    """The masked readout stays NaN-free under negative duals."""
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(16, 16))
    reg = RegL1(lamda=0.01)
    reg.bernoulli_mask = True
    reg.w_p = 0.3
    opt = opt_cls([p], lr=0.5, reg=reg, delta=1.0)
    for _ in range(5):
        p.grad = torch.randn_like(p) * 5.0  # large grads drive duals negative
        opt.step()
    assert torch.isfinite(p).all()


# --------------------------------------------------------------------------- #
# Pruner wiring
# --------------------------------------------------------------------------- #
def test_pruner_anneals_wp_without_lambda_scheduler():
    """End-to-end wiring: a wp_annealer drives reg.w_p from 1->0 over the run
    with no lambda_scheduler configured (the two are orthogonal)."""
    from types import SimpleNamespace

    from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner

    p = torch.nn.Parameter(torch.randn(4, 4))
    reg = RegL1(lamda=0.01)
    opt = LinBreg(
        [{"params": [p], "reg": reg, "lambda_scale": 1.0}],
        lr=0.1,
        reg=reg,
        delta=1.0,
    )

    pruner = BregmanPruner(
        wp_annealer=WpAnnealer(schedule="linear"),
        lambda_scheduler=None,  # explicitly no scheduler
    )
    pruner._optimizer = opt
    assert pruner._wp_anneal_active()

    trainer = SimpleNamespace(
        optimizers=[opt], global_step=0, estimated_stepping_batches=10
    )

    pruner._apply_wp_to_groups(opt, pruner.wp_annealer.value_at(0.0))
    assert reg.w_p == 1.0  # start fully reversible

    trainer.global_step = 5
    pruner._step_wp_annealer(trainer)
    assert reg.w_p == pytest.approx(0.5)  # halfway: linear midpoint

    trainer.global_step = 10
    pruner._step_wp_annealer(trainer)
    assert reg.w_p == 0.0  # committed/latched by the end
