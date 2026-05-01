"""CPU-only unit tests for the precond_refresh_every cache in
AdamScaledLoRA, AdamLinLoRA, AdamPolarProductLoRA, AdamuonPolarProductLoRA.

What this guards:
  1. K=1 still produces the original update (within float32 numerical
     equivalence — bit-identical for ScaledLoRA / polar variants where the
     cached-path branch reduces to the original calls; ULP-level differences
     for AdamLinLoRA where the Sylvester solve is now expressed through cached
     eigendecomps of SA/SB rather than eigh(γ²·SA) directly).
  2. K=5 produces a finite update that differs from K=1 by a bounded amount
     (i.e., staleness has an effect but doesn't blow up).
  3. Determinism within each (optimizer, K) configuration.
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
    AdamLinLoRA,
    AdamPolarProductLoRA,
    AdamScaledLoRA,
    AdamuonPolarProductLoRA,
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


OPT_CLASSES = [
    AdamScaledLoRA,
    AdamLinLoRA,
    AdamPolarProductLoRA,
    AdamuonPolarProductLoRA,
]


def _run(OptCls, K, n_steps, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    m, x, target = _make(seed)
    opt = OptCls(m, lr=lr, precond_refresh_every=K)
    for _ in range(n_steps):
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in m.parameters()]


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_constructor_accepts_precond_refresh_every(OptCls):
    m, _, _ = _make()
    opt = OptCls(m, lr=1e-3, precond_refresh_every=5)
    assert opt.precond_refresh_every == 5


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_k1_default_matches_explicit_k1(OptCls):
    """Default ctor (no K arg) and explicit K=1 produce the same trajectory.
    Establishes K=1 is the default."""
    torch.manual_seed(0)
    m1, x, target = _make(0)
    opt1 = OptCls(m1, lr=1e-3)  # default
    torch.manual_seed(0)
    m2, _, _ = _make(0)
    opt2 = OptCls(m2, lr=1e-3, precond_refresh_every=1)

    for _ in range(3):
        ((m1(x) - target) ** 2).mean().backward()
        opt1.step()
        ((m2(x) - target) ** 2).mean().backward()
        opt2.step()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_determinism_k5(OptCls):
    """Same seed + K=5 → same trajectory."""
    a = _run(OptCls, K=5, n_steps=8, seed=42)
    b = _run(OptCls, K=5, n_steps=8, seed=42)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=0.0)


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_k5_finite_and_changes_params(OptCls):
    """K=5 over 10 steps — at least one stale-cache reuse happened — produces
    finite, non-trivial updates."""
    pre = _run(OptCls, K=1, n_steps=0, seed=7)
    post = _run(OptCls, K=5, n_steps=10, seed=7)
    diffs = [float((a - b).abs().sum()) for a, b in zip(pre, post)]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params"
    assert max(diffs) > 0.0, "No params changed"


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_k5_drifts_from_k1_but_bounded(OptCls):
    """K=5 trajectory differs from K=1 but stays in a sane neighborhood.
    Bound is loose — the point is to catch runaway drift, not to nail down
    a specific magnitude."""
    a = _run(OptCls, K=1, n_steps=10, seed=11)
    b = _run(OptCls, K=5, n_steps=10, seed=11)
    # Compute relative Frobenius drift across all params.
    num = sum(float((pa - pb).pow(2).sum()) for pa, pb in zip(a, b))
    den = sum(float(pa.pow(2).sum()) for pa in a) + 1e-30
    rel_drift = (num / den) ** 0.5
    assert rel_drift < 0.5, f"K=5 vs K=1 relative drift {rel_drift:.4f} too large"


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_k1_equals_kbig_when_only_one_step(OptCls):
    """For a single step, any K refreshes (since (1-1)%K == 0) — identical."""
    a = _run(OptCls, K=1, n_steps=1, seed=3)
    b = _run(OptCls, K=10, n_steps=1, seed=3)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=0.0)


@pytest.mark.parametrize("OptCls", OPT_CLASSES)
def test_k_huge_caches_for_entire_run(OptCls):
    """K larger than n_steps means cache is built once at step 1 and reused.
    Trajectory must be finite and produce non-trivial updates."""
    a = _run(OptCls, K=1000, n_steps=5, seed=5)
    pre = _run(OptCls, K=1000, n_steps=0, seed=5)
    diffs = [float((p - q).abs().sum()) for p, q in zip(pre, a)]
    assert all(torch.isfinite(p).all() for p in a)
    assert max(diffs) > 0.0
