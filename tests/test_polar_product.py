"""CPU-only unit tests for PolarProductLoRA and AdamPolarProductLoRA.

These optimizers implement the closed-form polar update under the
spectral-product norm from docs/theory/main.tex lines 622-660.

Key behavioral test: when LoRA factors are constructed to be semi-orthogonal,
the spectral square-root preconditioner is the identity (S = I, S^{-1/2} = I)
and the optimizer reduces to per-factor polar (NS) — i.e., Muon-like.
"""
import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import (
    AdamPolarProductLoRA,
    PolarProductLoRA,
    _newton_schulz,
)


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        torch.nn.init.normal_(self.lora_B["default"].weight, std=0.05)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    m = TinyLoRAModel()
    x = torch.randn(3, 8)
    target = torch.randn(3, 8)
    return m, x, target


@pytest.mark.parametrize("OptCls", [PolarProductLoRA, AdamPolarProductLoRA])
def test_step_runs_and_changes_params(OptCls):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = OptCls(m, lr=1e-2)
    out = m(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    diffs = [float((a - b).abs().sum()) for a, b in zip(pre, post)]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params after step"
    assert max(diffs) > 0.0, "No params changed"


@pytest.mark.parametrize("OptCls", [PolarProductLoRA, AdamPolarProductLoRA])
def test_zero_grad_no_finite_failure(OptCls):
    """With zero grads on a step the optimizer must not produce NaN/Inf."""
    m, x, target = _make()
    opt = OptCls(m, lr=1e-2)
    out = m(x)
    loss = (out * 0.0).sum()
    loss.backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param after zero-grad step"


@pytest.mark.parametrize("OptCls", [PolarProductLoRA, AdamPolarProductLoRA])
def test_determinism(OptCls):
    def run():
        m, x, target = _make(seed=42)
        opt = OptCls(m, lr=1e-3)
        for _ in range(3):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    a = run()
    b = run()
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=0.0)


def test_orthogonal_factors_reduce_to_per_factor_polar():
    """Behavioral equivalence with the lemma's degenerate case:
    when (A·Aᵀ) and (BᵀB) are scaled identities, S^{-1/2} ∝ I and the
    spectral-product update reduces to per-factor polar(∇).

    We construct a tiny model with A having orthonormal rows (so AAᵀ = I)
    and B having orthonormal columns (so BᵀB = I), then check that one
    PolarProductLoRA step matches a hand-computed `lr * polar(∇)` per factor.
    """
    torch.manual_seed(0)
    d_in, d_out, r = 8, 6, 4
    m = TinyLoRAModel(d_in=d_in, d_out=d_out, r=r)

    # Force A to have orthonormal rows (r × d_in, r ≤ d_in): use QR of a Gaussian.
    # Force B to have orthonormal columns (d_out × r, r ≤ d_out): same trick.
    for sub in (m.l0, m.l1):
        d_in_l = sub.lora_A["default"].weight.shape[1]
        d_out_l = sub.lora_B["default"].weight.shape[0]
        r_l = sub.lora_A["default"].weight.shape[0]
        # A: orthonormal rows
        Q_A = torch.linalg.qr(torch.randn(d_in_l, r_l))[0]   # (d_in, r) with orthonormal cols
        sub.lora_A["default"].weight.data.copy_(Q_A.T)        # (r, d_in) with orthonormal rows
        # B: orthonormal columns
        Q_B = torch.linalg.qr(torch.randn(d_out_l, r_l))[0]   # (d_out, r) with orthonormal cols
        sub.lora_B["default"].weight.data.copy_(Q_B)

    # Sanity: AAᵀ = I, BᵀB = I per construction
    for sub in (m.l0, m.l1):
        A = sub.lora_A["default"].weight
        B = sub.lora_B["default"].weight
        assert torch.allclose(A @ A.T, torch.eye(r), atol=1e-5), "A not row-orthonormal"
        assert torch.allclose(B.T @ B, torch.eye(r), atol=1e-5), "B not col-orthonormal"

    # Compute a step
    x = torch.randn(3, d_in)
    target = torch.randn(3, d_in)
    pre = {n: p.detach().clone() for n, p in m.named_parameters()}
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()

    lr = 1e-2
    delta = 1e-6
    # Hand-compute expected per-factor polar update for orthogonal init.
    # Note: with delta=1e-6 added in spdify, S^{-1/2} ≈ (1 + δ)^{-1/2} ≈ 1 − δ/2,
    # negligible at this precision. So expected dA = -lr * polar(∇A), dB = -lr * polar(∇B).
    expected_changes = {}
    for layer_name, sub in [("l0", m.l0), ("l1", m.l1)]:
        gA = sub.lora_A["default"].weight.grad.float()
        gB = sub.lora_B["default"].weight.grad.float()
        expected_changes[f"{layer_name}.lora_A.default.weight"] = -lr * _newton_schulz(gA, nsteps=5)
        expected_changes[f"{layer_name}.lora_B.default.weight"] = -lr * _newton_schulz(gB, nsteps=5)

    # Apply PolarProductLoRA step
    opt = PolarProductLoRA(m, lr=lr, delta=delta, ns_steps=5)
    opt.step()
    post = {n: p.detach().clone() for n, p in m.named_parameters()}

    # Verify each factor's actual update matches the per-factor polar
    # within tolerance accounting for the δ-shift.
    for n, p_pre in pre.items():
        if n not in expected_changes:
            continue
        actual_change = post[n] - p_pre
        expected = expected_changes[n]
        rel_err = (actual_change - expected).norm() / (expected.norm() + 1e-30)
        assert rel_err < 0.02, (
            f"At orthogonal init, optimizer should reduce to per-factor polar "
            f"but {n} relative-error vs hand-computed = {rel_err:.4f}"
        )
