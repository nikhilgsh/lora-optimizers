"""Behavioral equivalence: `CurvatureWhitenLoRA(precond_method="gram_ns")` must
track the validated eigh/QR default within Newton-Schulz tolerance.

The protagonist's small-side Shampoo inverse-sqrt `S_a^{-1/2}` can be computed by the
QR eigenbasis (`eigh`, the validated default) or the Polar-Express Gram NS
(`gram_ns`, `gram_ns_inv_sqrt`). They compute the SAME operator, so the parameter
trajectories must match — this is the "trajectory-equivalent" half of the strict
adoption rule (docs/notes/inverse_sqrt_variant_plan.md). With `precond_refresh_every=1`
the eigh path refreshes its eigenbasis every step, so both paths use a *fresh*
inverse-sqrt and the only gap is the NS residual + the per-pair λ_max scale that
washes out through polar scale-invariance and the σ_max rescale.

Tolerance: ~3e-3 relative on the (A, B) parameters after N steps — the gram_ns
residual at 8 iters; tighter than this would falsely fail on NS noise, looser would
not catch a real dispatch bug. Stays finite throughout (the spectral guardrail).
"""
import copy

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import CurvatureWhitenLoRA
from lora_playground.utils import collect_lora_pairs


class _FakePair(nn.Module):
    """One (A, B) LoRA pair in the PEFT convention (lora_A: r×d_in, lora_B: d_out×r)
    with a NONZERO B so the small-side curvature Q_B is full-rank from step 1."""
    def __init__(self, r, d_in, d_out, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        a = torch.empty(r, d_in); nn.init.kaiming_uniform_(a)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(a)
            self.lora_B["default"].weight.copy_(0.02 * torch.randn(d_out, r, generator=g))


class _FakeModel(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.adapters = nn.ModuleList([_FakePair(r, di, do, seed=i)
                                       for i, (r, di, do) in enumerate(specs)])


# Protagonist config (kl-diag-polar-lora), minus the inverse-sqrt method.
_PROTO = dict(lr=1e-3, betas=(0.95, 0.999), delta=1e-4, curvature_beta=0.99,
              kl_coupled=True, soap_v=False, diag_metric=True, use_polar=True,
              cw_nesterov=True, cw_picard_iters=1, polar_method="polar_express",
              ns_steps=8, precond_refresh_every=1)


def test_gram_ns_matches_eigh_trajectory():
    specs = [(16, 64, 64), (16, 96, 48), (16, 48, 96)]
    torch.manual_seed(0)
    m0 = _FakeModel(specs)
    m_e, m_g = copy.deepcopy(m0), copy.deepcopy(m0)
    opt_e = CurvatureWhitenLoRA(m_e, precond_method="eigh", **_PROTO)
    opt_g = CurvatureWhitenLoRA(m_g, precond_method="gram_ns", higham_iters=8, **_PROTO)
    pairs_e, pairs_g = collect_lora_pairs(m_e), collect_lora_pairs(m_g)

    for step in range(1, 9):
        g = torch.Generator().manual_seed(1000 + step)
        for (Ae, Be), (Ag, Bg) in zip(pairs_e, pairs_g):
            gA = torch.randn(Ae.shape, generator=g)
            gB = torch.randn(Be.shape, generator=g)
            Ae.grad, Be.grad = gA.clone(), gB.clone()
            Ag.grad, Bg.grad = gA.clone(), gB.clone()
        opt_e.step(); opt_g.step()
        opt_e.zero_grad(); opt_g.zero_grad()
        for (Ae, Be), (Ag, Bg) in zip(pairs_e, pairs_g):
            assert torch.isfinite(Ag).all() and torch.isfinite(Bg).all(), \
                f"gram_ns produced non-finite params at step {step}"
            relA = (Ae - Ag).norm() / Ae.norm().clamp_min(1e-30)
            relB = (Be - Bg).norm() / Be.norm().clamp_min(1e-30)
            assert relA < 3e-3, f"A diverged at step {step}: relA={relA:.2e}"
            assert relB < 3e-3, f"B diverged at step {step}: relB={relB:.2e}"


@pytest.mark.parametrize("soap_v", [True])
def test_gram_ns_requires_soap_v_false(soap_v):
    """gram_ns (like higham) needs soap_v=False — the SOAP v̂ path consumes the
    eigenbasis the NS methods do not produce. Building it with soap_v=True raises."""
    m = _FakeModel([(8, 32, 32)])
    with pytest.raises(ValueError, match="requires soap_v=False"):
        CurvatureWhitenLoRA(m, precond_method="gram_ns", soap_v=soap_v,
                            kl_coupled=True, diag_metric=True, use_polar=True)
