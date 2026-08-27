"""Tight equivalence: CurvatureWhitenLoRA's grouped batched step
(`_batched_step=True`, default) must match the per-pair reference
(`_batched_step=False`) — both use the blessed spectral lib, so they agree to
batched-vs-looped float reduction order. Exercises a same-shape group of 2 plus
singleton groups. CPU, tiny, deterministic.
"""
import copy
import torch
import torch.nn as nn

from lora_playground.optim import CurvatureWhitenLoRA


class _LoraLin(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = _LoraLin(32, 16, 4)   # shape S
        self.l2 = _LoraLin(64, 16, 4)   # singleton (different d_in)
        self.l3 = _LoraLin(32, 16, 4)   # shape S again → group of 2 with l1
        self.l4 = _LoraLin(16, 48, 4)   # singleton (different d_out)


def _run(use_polar, **extra):
    torch.manual_seed(0)
    m_batched = _Model()
    m_ref = copy.deepcopy(m_batched)
    common = dict(lr=1e-3, use_polar=use_polar, precond_refresh_every=3, delta=1e-3, **extra)
    o_b = CurvatureWhitenLoRA(m_batched, **common); o_b._batched_step = True
    o_r = CurvatureWhitenLoRA(m_ref, **common);     o_r._batched_step = False
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        grads = [(torch.randn(A.shape, generator=g), torch.randn(B.shape, generator=g))
                 for A, B in o_b.pairs]
        for (Ab, Bb), (Ar, Br), (gA, gB) in zip(o_b.pairs, o_r.pairs, grads):
            Ab.grad, Bb.grad = gA.clone(), gB.clone()
            Ar.grad, Br.grad = gA.clone(), gB.clone()
        o_b.step()
        o_r.step()
    for k, ((Ab, Bb), (Ar, Br)) in enumerate(zip(o_b.pairs, o_r.pairs)):
        assert torch.isfinite(Ab).all() and torch.isfinite(Bb).all(), f"pair {k}: non-finite"
        assert torch.allclose(Ab, Ar, atol=1e-4, rtol=1e-3), \
            f"pair {k} A: max|Δ|={(Ab - Ar).abs().max().item():.2e}"
        assert torch.allclose(Bb, Br, atol=1e-4, rtol=1e-3), \
            f"pair {k} B: max|Δ|={(Bb - Br).abs().max().item():.2e}"


def test_grouped_matches_per_pair_polar():
    _run(use_polar=True)


def test_grouped_matches_per_pair_no_polar():
    _run(use_polar=False)


def test_grouped_matches_per_pair_no_radius():
    # ablation --cw_no_radius (ρ=lr): grouped and per-pair must still agree
    _run(use_polar=True, soap_v=False, cw_nesterov=True, cw_no_radius=True)


def test_grouped_matches_per_pair_no_diag_curv():
    # ablation --cw_no_diag_curv (P=Q=I → C_A=BᵀB): requires diag_metric; paths must agree
    _run(use_polar=True, soap_v=False, cw_nesterov=True, diag_metric=True, cw_no_diag_curv=True)


def test_grouped_matches_per_pair_diag_shampoo():
    # The diag-shampoo arm: diag_metric=True, soap_v=False, kl_coupled=False.
    # Touches the else-branch P_A/Q_B clobber guard in BOTH step paths, so the
    # batched↔per-pair equivalence must still hold.
    _run(use_polar=True, diag_metric=True, soap_v=False, kl_coupled=False)


def test_grouped_matches_per_pair_flat_outer():
    # The kl-diag-polar-flatout arm: flat_outer=True skips the un-whiten in BOTH
    # step paths (dX ∝ φ(z)). Equivalence must hold across the edited branch.
    _run(use_polar=True, diag_metric=True, soap_v=False, kl_coupled=True, flat_outer=True)


def test_grouped_matches_per_pair_solved_rho():
    # cw_solved_rho: the post-Picard solved-magnitude rescale (prodsum power
    # iter + quadratic root) runs in BOTH step paths; equivalence must hold.
    _run(use_polar=True, diag_metric=True, soap_v=False, kl_coupled=True,
         cw_nesterov=True, cw_solved_rho=True)
