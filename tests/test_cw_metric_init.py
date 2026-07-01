"""cw_metric_init={zero,ones} for CurvatureWhitenLoRA's diagonal metric EMAs
D_in (=Q) / D_out (=P).

Algebra (see notebooks/cw_metric_init_analysis.ipynb): both inits make step 1 use
the identity metric — zero hits the _rdinv xmax≈0 fallback, ones normalizes the
all-ones EMA to the same identity — so the step-1 update is identical up to the
O(eps) σ_max-rescale floor. They then diverge through the warmup as the ones-init
β₂ᵗ identity prior on the curvature shape decays. CPU, tiny, deterministic.
"""
import copy
import pytest
import torch
import torch.nn as nn

from lora_playground.optim import CurvatureWhitenLoRA


class _LoraLin(nn.Module):
    def __init__(self, d_in, d_out, r, b_zero):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        if b_zero:
            nn.init.zeros_(self.lora_B["default"].weight)  # PEFT init


class _Model(nn.Module):
    def __init__(self, b_zero):
        super().__init__()
        self.l1 = _LoraLin(48, 32, 8, b_zero)
        self.l2 = _LoraLin(64, 32, 8, b_zero)
        self.l3 = _LoraLin(32, 48, 8, b_zero)


# Production protagonist flags (protagonist_config / sweep_protagonist_generic.sh).
_PROD = dict(kl_coupled=True, diag_metric=True, soap_v=False, use_polar=True,
             precond_method="gram_ns", higham_iters=8, polar_method="polar_express",
             ns_steps=8, cw_nesterov=True, curvature_beta=0.99,
             precond_refresh_every=10, delta=1e-4, betas=(0.9, 0.999), lr=1e-2)


def _two_runs(b_zero, nsteps, init="ones"):
    torch.manual_seed(0)
    m0 = _Model(b_zero)
    m1 = copy.deepcopy(m0)
    o0 = CurvatureWhitenLoRA(m0, cw_metric_init="zero", **_PROD)
    o1 = CurvatureWhitenLoRA(m1, cw_metric_init=init, **_PROD)
    g = torch.Generator().manual_seed(1)
    rel = []
    for _ in range(nsteps):
        grads = [(torch.randn(A.shape, generator=g), torch.randn(B.shape, generator=g))
                 for A, B in o0.pairs]
        for (A0, B0), (A1, B1), (gA, gB) in zip(o0.pairs, o1.pairs, grads):
            A0.grad, B0.grad = gA.clone(), gB.clone()
            A1.grad, B1.grad = gA.clone(), gB.clone()
        o0.step(); o1.step()
        num = den = 0.0
        for (A0, B0), (A1, B1) in zip(o0.pairs, o1.pairs):
            assert torch.isfinite(A1).all() and torch.isfinite(B1).all()
            num = max(num, (A0 - A1).abs().max().item(), (B0 - B1).abs().max().item())
            den = max(den, A0.abs().max().item(), B0.abs().max().item())
        rel.append(num / max(den, 1e-12))
    return rel


@pytest.mark.parametrize("b_zero", [True, False])
def test_step1_identical(b_zero):
    # Step 1 must match across inits up to the O(eps) σ_max floor.
    rel = _two_runs(b_zero, nsteps=1)
    assert rel[0] < 1e-5, f"step-1 rel diff {rel[0]:.2e} too large"


@pytest.mark.parametrize("b_zero", [True, False])
def test_diverges_after_step1(b_zero):
    # The flag must actually change later steps (guard against a future no-op).
    rel = _two_runs(b_zero, nsteps=8)
    assert rel[0] < 1e-5
    assert rel[-1] > 1e-3, f"no divergence by step 8 (rel={rel[-1]:.2e})"


@pytest.mark.parametrize("b_zero", [True, False])
def test_delta_close_to_zero_TOY_SCALE_ONLY(b_zero):
    # CAVEAT: this passes only because this toy model's curvature is O(1), so the
    # δ=1e-4 init sits BELOW its crossover. At PRODUCTION scale the curvature metric
    # is ~1e-7, so δ=1e-4 is ~1000× too big and "delta" behaves like "ones" (+0.019
    # eval loss), NOT like zero. Do NOT read this as "delta reproduces zero" — it does
    # not in production. The prior-free branch-free init is a FLOAT ε ≪ 1e-7 (≈1e-10),
    # not δ. Kept as a regression that the init plumbing works, with the scale warning.
    rel = _two_runs(b_zero, nsteps=50, init="delta")
    assert rel[0] < 1e-5, f"step-1 rel diff {rel[0]:.2e}"


def test_float_eps_init_value():
    # cw_metric_init accepts a float ε → P0=Q0=εI (the branch-free prior-free init).
    m = _Model(b_zero=True)
    o = CurvatureWhitenLoRA(m, cw_metric_init="1e-10", **_PROD)
    st = o.pair_state[0]
    assert torch.all(st["D_in"] == 1e-10) and torch.all(st["D_out"] == 1e-10)


def test_default_is_eps():
    # Shipped default is P0=Q0=εI at ε=1e-12 (branch-free, prior-free), not "zero".
    m = _Model(b_zero=True)
    o = CurvatureWhitenLoRA(m, **_PROD)
    assert o.cw_metric_init == "1e-12"
    assert torch.all(o.pair_state[0]["D_in"] == 1e-12)


def test_ones_init_value():
    m = _Model(b_zero=True)
    o = CurvatureWhitenLoRA(m, cw_metric_init="ones", **_PROD)
    st = o.pair_state[0]
    assert torch.all(st["D_in"] == 1.0) and torch.all(st["D_out"] == 1.0)


def test_delta_init_value():
    m = _Model(b_zero=True)
    o = CurvatureWhitenLoRA(m, cw_metric_init="delta", **_PROD)
    st = o.pair_state[0]
    assert torch.all(st["D_in"] == _PROD["delta"]) and torch.all(st["D_out"] == _PROD["delta"])


def test_bad_value_raises():
    m = _Model(b_zero=True)
    with pytest.raises(ValueError, match="cw_metric_init"):
        CurvatureWhitenLoRA(m, cw_metric_init="identity", **_PROD)
