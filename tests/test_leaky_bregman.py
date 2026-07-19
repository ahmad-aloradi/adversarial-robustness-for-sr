"""Finite-memory Bregman: dual leak (LinBreg), birth rule (AdaBreg), and the
CosineClassifier options. The phantom-row scenario is a weight pinned at zero
receiving a tiny one-sided gradient: perfect-memory duals must revive it,
finite-memory duals must not.
"""
import pytest
import torch

from src.callbacks.pruning.bregman.bregman_optimizers import (
    AdaBreg,
    AdaBregW,
    LinBreg,
)
from src.callbacks.pruning.bregman.bregman_regularizers import RegL1

LAM = 0.5
LR = 0.1


def _step_with_grad(opt, p, grad):
    p.grad = grad.clone()
    opt.step()


def test_linbreg_leak_zero_keeps_dual_frozen_without_gradient_decay():
    """gamma=0: zero gradient leaves the dual (and weight) untouched."""
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = LinBreg([p], lr=LR, reg=RegL1(lamda=LAM), dual_leak=0.0)
    for _ in range(50):
        _step_with_grad(opt, p, torch.zeros(1))
    assert torch.allclose(p.data, torch.tensor([1.0]))


def test_linbreg_leak_decays_stale_active_weight():
    """gamma>0: a weight whose gradient support vanished is collected."""
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = LinBreg([p], lr=LR, reg=RegL1(lamda=LAM), dual_leak=0.05)
    for _ in range(400):
        _step_with_grad(opt, p, torch.zeros(1))
    assert p.data.item() == 0.0


def test_linbreg_one_sided_drip_revives_only_without_leak():
    """A pinned weight under a tiny one-sided push: vanilla must revive it
    (integral diverges), leak must hold it dead (drip below the bar)."""
    drip = torch.tensor([1e-3])  # sustained bar at gamma=0.05: 0.05*0.5/0.1=0.25

    p_vanilla = torch.nn.Parameter(torch.zeros(1))
    opt_vanilla = LinBreg([p_vanilla], lr=LR, reg=RegL1(lamda=LAM))

    p_leaky = torch.nn.Parameter(torch.zeros(1))
    opt_leaky = LinBreg([p_leaky], lr=LR, reg=RegL1(lamda=LAM), dual_leak=0.05)

    for _ in range(20000):
        _step_with_grad(opt_vanilla, p_vanilla, drip)
        _step_with_grad(opt_leaky, p_leaky, drip)

    assert p_vanilla.data.item() != 0.0, "perfect memory must integrate past λ"
    assert p_leaky.data.item() == 0.0, "leak must hold the drip below λ"


def test_linbreg_leak_spares_regnone_groups():
    """Unregularized params (e.g. the classifier bias) must never leak."""
    p = torch.nn.Parameter(torch.tensor([2.0]))
    opt = LinBreg([p], lr=LR, dual_leak=0.5)  # reg defaults to RegNone
    for _ in range(20):
        _step_with_grad(opt, p, torch.zeros(1))
    assert torch.allclose(p.data, torch.tensor([2.0]))


def test_linbreg_rejects_bad_leak():
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="dual_leak"):
        LinBreg([p], lr=LR, dual_leak=1.0)


def test_adabreg_vanilla_revives_consistent_drip():
    """Adam-preconditioned dual moves at full speed on any sign-consistent
    gradient, however tiny — the phantom failure this work removes."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdaBreg([p], lr=LR, reg=RegL1(lamda=LAM))
    for _ in range(200):
        _step_with_grad(opt, p, torch.tensor([1e-6]))
    assert p.data.item() != 0.0


def test_adabreg_hybrid_birth_requires_sustained_demand():
    """Hybrid: a strong sustained gradient re-births its weight; a tiny drip
    stays dead. Bar = birth_scale * gamma * lambda / lr = 0.025."""
    p = torch.nn.Parameter(torch.zeros(2))
    opt = AdaBreg(
        [p], lr=LR, reg=RegL1(lamda=LAM), dual_leak=0.01, birth_scale=0.5
    )
    grad = torch.tensor([1.0, 1e-3])  # strong demand vs sub-bar drip
    for _ in range(2000):
        _step_with_grad(opt, p, grad)
    assert p.data[0].item() != 0.0, "strong sustained demand must be born"
    assert p.data[1].item() == 0.0, "sub-bar drip must stay dead"


def test_adabreg_leak_zero_matches_vanilla_trajectory():
    """dual_leak=0 must reproduce the classic AdaBreg update exactly."""
    torch.manual_seed(0)
    grads = [torch.randn(4) for _ in range(20)]

    p_a = torch.nn.Parameter(torch.ones(4))
    p_b = torch.nn.Parameter(torch.ones(4))
    opt_a = AdaBreg([p_a], lr=LR, reg=RegL1(lamda=0.1))
    opt_b = AdaBreg([p_b], lr=LR, reg=RegL1(lamda=0.1), dual_leak=0.0)
    for g in grads:
        _step_with_grad(opt_a, p_a, g)
        _step_with_grad(opt_b, p_b, g)
    assert torch.equal(p_a.data, p_b.data)


def test_adabregw_signature_has_no_dual_leak():
    """AdaBregW does not implement the hybrid rule; passing dual_leak must
    fail loudly instead of being silently ignored."""
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(TypeError):
        AdaBregW([p], lr=LR, weight_decay=1e-3, dual_leak=0.01)


class TestCosineClassifier:
    def _make(self, **kwargs):
        from src.modules.models.cosine_classifier import CosineClassifier

        torch.manual_seed(0)
        return CosineClassifier(input_size=32, out_neurons=10, **kwargs)

    def test_defaults_match_speechbrain(self):
        from speechbrain.lobes.models.ECAPA_TDNN import Classifier

        torch.manual_seed(0)
        ref = Classifier(input_size=32, out_neurons=10)
        clf = self._make()
        clf.weight.data.copy_(ref.weight.data)
        x = torch.randn(4, 1, 32)
        assert torch.allclose(clf(x), ref(x))

    def test_bias_output_stays_in_cosine_range(self):
        clf = self._make(bias=True)
        clf.bias.data.fill_(5.0)  # force the clamp path
        out = clf(torch.randn(4, 1, 32))
        assert out.shape == (4, 1, 10)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_bias_is_trainable_and_off_by_default(self):
        assert self._make().bias is None
        clf = self._make(bias=True)
        out = clf(torch.randn(4, 1, 32))
        out.sum().backward()
        assert clf.bias.grad is not None

    def test_norm_gate_attenuates_small_rows(self):
        clf_cos = self._make()
        clf_gate = self._make(norm_gate=0.3)
        clf_gate.weight.data.copy_(clf_cos.weight.data)
        clf_gate.weight.data[0] *= 1e-3  # tiny row: confidence must collapse
        clf_cos.weight.data[0] *= 1e-3
        x = torch.randn(4, 1, 32)
        gated, cos = clf_gate(x), clf_cos(x)
        assert gated[..., 0].abs().max() < 0.01 * cos[..., 0].abs().max()

    def test_norm_gate_bounds_gradient_of_zero_row(self):
        """Exact cosine amplifies a zero row's gradient by 1/eps; the gate
        must keep it at the healthy-row scale."""
        clf = self._make(norm_gate=0.3)
        clf.weight.data[0] = 0.0
        x = torch.randn(4, 1, 32)
        clf(x).sum().backward()
        zero_row_grad = clf.weight.grad[0].norm()
        healthy_grad = clf.weight.grad[1:].norm(dim=1).mean()
        assert zero_row_grad < 100 * healthy_grad
