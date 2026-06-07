"""Tests for KL-diag-LoRA — option (b) of kl_shampoo_polar_derivation.md
§"Cross-coupling": CurvatureWhitenLoRA with ``diag_metric=True`` (plus
``kl_coupled=True, soap_v=False``).

Option (b) commits to the single global diagonal metric (P,Q)=(D_out, D_in): the
dense KL small side S_curv is replaced by the conjugate-diagonal-weighted geometric
Gram, M_A = Bᵀ diag(D_out) B and M_B = A diag(D_in) Aᵀ, recomputed each step. The
diagonal KL coupling is unchanged but now whitens by M⁻¹, giving a self-consistent
two-sided program.

Covers: step runs & changes params, multistep finiteness, the batched↔per-pair
equivalence on the diag path (both code paths were edited), the defining identity
L_A == Bᵀ diag(D_out) B, factory dispatch, and that diag (b) actually differs from
dense kl-shampoo (a).
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


def _diag_opt(m, lr=1e-2, use_polar=True, **kw):
    return CurvatureWhitenLoRA(m, lr=lr, use_polar=use_polar,
                               kl_coupled=True, soap_v=False, diag_metric=True, **kw)


def test_requires_kl_coupled():
    m, _, _ = _make()
    with pytest.raises(ValueError):
        CurvatureWhitenLoRA(m, kl_coupled=False, soap_v=False, diag_metric=True)


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    before = [p.detach().clone() for p in m.parameters()]
    opt = _diag_opt(m, use_polar=use_polar)
    for _ in range(3):
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    after = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert all(torch.isfinite(p).all() for p in m.parameters())


@pytest.mark.parametrize("use_polar", [False, True])
def test_batched_matches_per_pair(use_polar):
    """Grouped batched path and the per-pair oracle must agree on the diag path —
    both were edited for diag_metric."""
    def run(batched):
        m, x, target = _make(seed=3)
        opt = _diag_opt(m, lr=1e-2, use_polar=use_polar, precond_refresh_every=2)
        opt._batched_step = batched
        for _ in range(4):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pg, pp in zip(run(True), run(False)):
        assert torch.allclose(pg, pp, atol=1e-5, rtol=1e-4), "batched vs per-pair diag mismatch"


def test_small_side_is_geometric_gram():
    """The defining identity: after a step, the stored small-side factor L_A equals
    Bᵀ diag(D_out) B computed from the factor and diagonal as they were at the START
    of that step (option b), NOT a dense EMA of g g^T."""
    m, x, target = _make(seed=7)
    opt = _diag_opt(m, lr=1e-2, use_polar=True)
    # Warm up so D_out is non-zero (step 1 has D_out=0 ⇒ M_A=0).
    for _ in range(3):
        ((m(x) - target) ** 2).mean().backward()
        opt.step(); opt.zero_grad()
    # Snapshot the (B, D_out) the next step will consume, per pair.
    snap = []
    for i, (A, B) in enumerate(opt.pairs):
        snap.append((B.detach().float().clone(), opt.pair_state[i]['D_out'].clone()))
    ((m(x) - target) ** 2).mean().backward()
    opt.step(); opt.zero_grad()
    for i, (B_pre, Dout_pre) in enumerate(snap):
        M_A = B_pre.transpose(-2, -1) @ (Dout_pre.unsqueeze(-1) * B_pre)
        assert torch.allclose(opt.pair_state[i]['L_A'], M_A, atol=1e-5, rtol=1e-4), \
            f"pair {i}: L_A is not Bᵀ diag(D_out) B"


def test_diag_differs_from_dense_kl():
    """Option (b) (diag_metric) must produce a different trajectory than dense
    kl-shampoo (a) — same seed/data, different small-side metric."""
    def run(diag):
        m, x, target = _make(seed=11)
        opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=True, kl_coupled=True,
                                  soap_v=False, diag_metric=diag)
        for _ in range(5):
            ((m(x) - target) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]

    a = run(False)  # dense kl-shampoo-polar
    b = run(True)   # diag (option b)
    assert any(not torch.allclose(pa, pb, atol=1e-4) for pa, pb in zip(a, b)), \
        "diag_metric produced the same params as dense kl — substitution not in effect"


@pytest.mark.parametrize("name,polar", [("kl-diag-lora", False), ("kl-diag-polar-lora", True)])
def test_factory_dispatch(name, polar):
    m, _, _ = _make()
    opt = build_optimizer(
        m, optimizer_type=name, lr=1e-3,
        precond_delta=1e-3, curvature_beta=0.99, muon_ns_steps=5,
        precond_refresh_every=10,
    )
    assert isinstance(opt, CurvatureWhitenLoRA)
    assert opt.diag_metric is True
    assert opt.kl_coupled is True
    assert opt.soap_v is False
    assert opt.use_polar is polar
