"""Tests for CurvatureWhitenLoRA (two-sided curvature-whitened momentum).

Covers: step runs and changes params (both polar arms), zero-grad finiteness,
determinism, the polar toggle's orthogonalization behavior, and factory dispatch
under both registered names.
"""
import torch
import torch.nn as nn
import pytest

from lora_playground.optim import CurvatureWhitenLoRA, build_optimizer


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


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar)
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params after step"
    assert max(float((a - b).abs().sum()) for a, b in zip(pre, post)) > 0.0, "No params changed"


@pytest.mark.parametrize("use_polar", [False, True])
def test_zero_grad_no_finite_failure(use_polar):
    m, x, target = _make()
    opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar)
    loss = (m(x) * 0.0).sum()
    loss.backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param after zero-grad step"


@pytest.mark.parametrize("use_polar", [False, True])
def test_determinism(use_polar):
    def run():
        m, x, target = _make(seed=42)
        opt = CurvatureWhitenLoRA(m, lr=1e-3, use_polar=use_polar)
        for _ in range(3):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pa, pb in zip(run(), run()):
        assert torch.allclose(pa, pb, atol=0.0)


def test_polar_arm_orthogonalizes_update():
    """use_polar=True must flatten the per-factor update spectrum (σ ratio ≈ 1);
    use_polar=False (plain whiten) leaves it spread. Read the applied ΔA directly
    by capturing pre/post on a wide factor where the polar has work to do.
    """
    def applied_dA(use_polar):
        torch.manual_seed(1)
        # r=4 ≤ d_in=16 so ΔA is wide; polar → orthonormal rows (flat σ).
        sub = _FakeLoRALinear(16, 6, 4)

        class _M(nn.Module):
            def __init__(s): super().__init__(); s.l0 = sub
            def forward(s, x): return s.l0(x)
        m = _M()
        # ns_steps=10 so Newton-Schulz fully converges (5 iters under-flattens a
        # wide matrix). Drive with full-rank random gradients over several steps:
        # step 1 uses an identity eigenbasis (plain Adam) by construction, and a
        # backprop-through-this-toy gradient is rank-deficient — both make a
        # single-step update degenerate. refresh_every=1 → Q non-identity from
        # step 2; we read the last step's ΔA, which is full row-rank.
        opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar, ns_steps=10,
                                  precond_refresh_every=1)
        wA = sub.lora_A["default"].weight
        wB = sub.lora_B["default"].weight
        g = torch.Generator().manual_seed(7)
        before = None
        for _ in range(4):
            wA.grad = torch.randn(wA.shape, generator=g)
            wB.grad = torch.randn(wB.shape, generator=g)
            before = wA.detach().clone()
            opt.step()
        return (wA.detach() - before).float()

    s_polar = torch.linalg.svdvals(applied_dA(True))
    s_plain = torch.linalg.svdvals(applied_dA(False))
    ratio_polar = float(s_polar[0] / s_polar[-1])
    ratio_plain = float(s_plain[0] / s_plain[-1])
    # Polar arm flattens the update spectrum (σ ratio → 1). Plain SOAP (Adam in
    # the eigenbasis) is fairly isotropic but retains real spectral spread, so
    # the polar still does visible work — it is strictly flatter than plain.
    assert ratio_polar < 1.3, f"polar arm should flatten spectrum, got σ ratio {ratio_polar:.3f}"
    assert ratio_plain > 1.4 and ratio_plain > ratio_polar, (
        f"no-polar arm should be more spread than the orthogonalized polar arm "
        f"(got plain {ratio_plain:.2f} vs polar {ratio_polar:.2f})")


@pytest.mark.parametrize("name,expect_polar", [
    ("curvature-whiten-lora", False),
    ("curvature-whiten-polar-lora", True),
])
def test_factory_dispatch(name, expect_polar):
    m, _, _ = _make()
    opt = build_optimizer(m, name, lr=1e-3, curvature_beta=0.99, muon_ns_steps=5)
    assert isinstance(opt, CurvatureWhitenLoRA)
    assert opt.use_polar is expect_polar
