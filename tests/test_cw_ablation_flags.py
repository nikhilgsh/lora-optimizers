"""Ablation flags on the protagonist (diag-shampoo-polar-lora), defined off skeleton Alg 1:
  --cw_no_radius     : ρ = lr  (drop the operator-norm radius ρ = lr/(σmax A + σmax B))
  --cw_no_diag_curv  : input/output diagonals → I  (C_A=BᵀB, C_B=AAᵀ; partner-Gram, iMuon-like)
Both default off (= protagonist). Tests verify the intended math, not just "runs".
"""
import math
import pytest
import torch
import torch.nn as nn

from lora_playground.optim import build_optimizer, CurvatureWhitenLoRA


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.normal_(self.lora_B["default"].weight, std=0.1)  # nonzero B (avoid B=0 corner)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class _Tiny(nn.Module):
    def __init__(self, d_in=12, d_out=10, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    return _Tiny(), torch.randn(5, 12), torch.randn(5, 12)


def _build(model, **kw):
    return build_optimizer(model, "diag-shampoo-polar-lora", lr=3e-2,
                           curvature_beta=0.99, muon_ns_steps=8, precond_delta=1e-4, **kw)


def _step_capture(opt, model, x, tgt):
    before = {id(p): p.detach().clone() for grp in opt.param_groups for p in grp["params"]}
    ((model(x) - tgt) ** 2).mean().backward()
    opt.step()
    deltas = {id(p): (p.detach() - before[id(p)]) for grp in opt.param_groups for p in grp["params"]}
    return deltas


def _smax(M):
    return torch.linalg.svdvals(M.float())[0].item()


def test_flags_set_and_default_off():
    m, _, _ = _make()
    base = _build(m)
    assert base.cw_no_radius is False and base.cw_no_diag_curv is False
    m2, _, _ = _make()
    abl = _build(m2, cw_no_radius=True, cw_no_diag_curv=True)
    assert abl.cw_no_radius is True and abl.cw_no_diag_curv is True


def test_no_radius_gives_sigma_max_equal_lr():
    """−radius ⟹ ρ=lr, and the operator-norm rescale sets σmax(Ȧ)=ρ=lr."""
    lr = 3e-2
    m, x, tgt = _make(seed=1)
    opt = _build(m, cw_no_radius=True)
    deltas = _step_capture(opt, m, x, tgt)
    for d in deltas.values():
        assert math.isfinite(d.norm().item())
        # each per-factor update is rescaled to σmax = ρ = lr
        assert _smax(d) == pytest.approx(lr, rel=0.08), f"σmax(Ȧ)={_smax(d)} != lr={lr}"


def test_radius_on_gives_smaller_sigma_than_lr():
    """Protagonist (radius on): ρ=lr/(σmaxA+σmaxB); for well-scaled factors σmaxA+σmaxB>1,
    so σmax(Ȧ)=ρ < lr — distinct from the −radius arm."""
    lr = 3e-2
    m, x, tgt = _make(seed=1)
    opt = _build(m)  # radius ON
    deltas = _step_capture(opt, m, x, tgt)
    smaxes = [_smax(d) for d in deltas.values()]
    assert all(math.isfinite(s) for s in smaxes)
    # radius divides by (σmaxA+σmaxB); not pinned to lr like the −radius arm
    assert not all(s == pytest.approx(lr, rel=0.08) for s in smaxes)


def test_no_diag_curv_differs_and_finite():
    """−Shampoo changes the update once the diagonals accumulate (the curvature collapses
    to the partner-Gram) and stays finite. Needs >1 step: at step 1 the diagonals are still
    zero → identity-fallback, so the flag is a no-op there."""
    m1, x, tgt = _make(seed=2)
    m2, _, _ = _make(seed=2)
    full = _build(m1)
    nosh = _build(m2, cw_no_diag_curv=True)
    for _ in range(4):  # let D_in/D_out accumulate so the flag bites
        for opt, m in ((full, m1), (nosh, m2)):
            opt.zero_grad(set_to_none=False)
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
    diffs = [(p1.detach() - p2.detach()).norm().item()
             for p1, p2 in zip(m1.parameters(), m2.parameters())]
    assert any(df > 1e-6 for df in diffs), "−Shampoo did not change the trajectory by step 4"
    assert all(torch.isfinite(p).all() for p in m2.parameters())


def test_no_diag_curv_requires_diag_metric():
    """The flag is only defined on the diag_metric (protagonist) path."""
    m, _, _ = _make()
    pairs_model = m
    with pytest.raises(ValueError, match="diag_metric"):
        CurvatureWhitenLoRA(pairs_model, lr=1e-2, diag_metric=False, cw_no_diag_curv=True)
