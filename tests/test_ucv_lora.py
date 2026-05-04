"""CPU-only unit tests for the UCV^T orthogonal-core LoRA layer + optimizer.

Spec: docs/notes/polar_product/orthogonal_core_lora_2026_05_03.md.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import AdamOrthogonalCoreLoRA
from lora_playground.ucv_layer import UCVLinear, inject_ucv_adapters
from lora_playground.utils import collect_ucv_triples


class _TwoLinear(nn.Module):
    def __init__(self, d_in=8, d_hidden=12, d_out=8):
        super().__init__()
        self.l0 = nn.Linear(d_in, d_hidden, bias=False)
        self.l1 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0, r=4, alpha=4):
    torch.manual_seed(seed)
    m = _TwoLinear()
    inject_ucv_adapters(m, target_modules="all-linear", r=r, alpha=alpha)
    x = torch.randn(3, 8)
    target = torch.randn(3, 8)
    return m, x, target


def test_injection_replaces_linears():
    m, _, _ = _make()
    assert isinstance(m.l0, UCVLinear)
    assert isinstance(m.l1, UCVLinear)
    triples = collect_ucv_triples(m)
    assert len(triples) == 2


def test_init_invariants():
    """U^T U ≈ I, V^T V ≈ I, C = 0 (so adapter output is zero at step 0)."""
    m, x, _ = _make()
    for U, C, V in collect_ucv_triples(m):
        I = torch.eye(U.shape[1])
        assert torch.allclose(U.T @ U, I, atol=1e-5)
        assert torch.allclose(V.T @ V, I, atol=1e-5)
        assert torch.all(C == 0)
    # Adapter is identity at init: forward must equal frozen base forward.
    base_only = m.l1.base(m.l0.base(x))
    out = m(x)
    assert torch.allclose(out, base_only, atol=1e-6)


def test_only_ucv_params_require_grad():
    m, _, _ = _make()
    train_names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all("ucv_" in n for n in train_names), train_names
    assert len(train_names) == 6  # 2 layers × {U, C, V}


def test_step_runs_and_changes_params():
    m, x, target = _make()
    pre_C = [C.detach().clone() for U, C, V in collect_ucv_triples(m)]
    pre_U = [U.detach().clone() for U, C, V in collect_ucv_triples(m)]
    opt = AdamOrthogonalCoreLoRA(m, lr=1e-2)
    out = m(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    triples = collect_ucv_triples(m)
    for (U, C, V), C0, U0 in zip(triples, pre_C, pre_U):
        assert torch.isfinite(U).all() and torch.isfinite(C).all() and torch.isfinite(V).all()
        # C should move (it was zero with nonzero gradient through adapter? Actually
        # C=0 means adapter output is 0, so loss grad w.r.t. C is non-trivial only
        # if dL/d(adapter_out) is nonzero — which it is since base forward has loss).
        # But U, V grads also flow. We at minimum require U or C changed.
        assert (U - U0).abs().sum() > 0 or (C - C0).abs().sum() > 0


def test_stiefel_preserved_after_step():
    """After one step, U and V should still have orthonormal columns within
    NS-iteration tolerance."""
    m, x, target = _make()
    opt = AdamOrthogonalCoreLoRA(m, lr=1e-2, ns_steps=5)
    out = m(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    for U, _C, V in collect_ucv_triples(m):
        I_U = torch.eye(U.shape[1])
        I_V = torch.eye(V.shape[1])
        # 5 NS iters drive ||X^T X - I||_F to ~1e-6 from a near-orthonormal start.
        assert (U.T @ U - I_U).abs().max() < 1e-3
        assert (V.T @ V - I_V).abs().max() < 1e-3


def test_zero_grad_no_finite_failure():
    m, x, _ = _make()
    opt = AdamOrthogonalCoreLoRA(m, lr=1e-2)
    out = m(x)
    loss = (out * 0.0).sum()
    loss.backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all()


def test_determinism():
    def run():
        m, x, target = _make(seed=42)
        opt = AdamOrthogonalCoreLoRA(m, lr=1e-2)
        for _ in range(3):
            out = m(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters() if p.requires_grad]

    a = run()
    b = run()
    for x_a, x_b in zip(a, b):
        assert torch.allclose(x_a, x_b, atol=1e-6)
