"""Tests for the shared Tier-1 factor diagnostics (lora_playground.optim_diagnostics)."""

import math

import torch

from lora_playground.optim_diagnostics import factor_diagnostics, FACTOR_DIAG_KEYS


def test_keys_present_and_finite():
    torch.manual_seed(0)
    A = torch.randn(8, 64)      # (r, d_in)
    B = torch.randn(128, 8)     # (d_out, r)
    rec = factor_diagnostics(A, B)
    assert set(rec) == set(FACTOR_DIAG_KEYS)
    assert all(math.isfinite(rec[k]) for k in rec)


def test_balanced_pair_has_small_residual():
    # AAᵀ == BᵀB exactly ⇒ balance_resid == 0.
    torch.manual_seed(1)
    M = torch.randn(4, 4)
    A = M                       # AAᵀ = M Mᵀ
    B = M.T                     # BᵀB = (Mᵀ)ᵀ Mᵀ = M Mᵀ  → equal to AAᵀ
    rec = factor_diagnostics(A, B)
    assert rec["balance_resid"] < 1e-5


def test_imbalanced_pair_has_large_residual():
    # Scale one factor up, the other down (a gauge move): product unchanged,
    # but AAᵀ and BᵀB diverge ⇒ balance_resid grows.
    torch.manual_seed(2)
    M = torch.randn(4, 4)
    A = 10.0 * M
    B = (M.T) / 10.0
    rec = factor_diagnostics(A, B)
    assert rec["balance_resid"] > 0.5


def test_sigma_max_matches_svd():
    torch.manual_seed(3)
    A = torch.randn(6, 32)
    B = torch.randn(48, 6)
    rec = factor_diagnostics(A, B)
    sa = torch.linalg.svdvals(A.float())[0].item()
    sb = torch.linalg.svdvals(B.float())[0].item()
    assert abs(rec["sigma_max_A"] - sa) < 1e-3 * sa
    assert abs(rec["sigma_max_B"] - sb) < 1e-3 * sb


def test_stable_rank_bounds():
    # 1 <= stable rank <= min(r, dim). Orthonormal columns ⇒ stable rank == r.
    Q = torch.linalg.qr(torch.randn(64, 8))[0]   # (64, 8), orthonormal cols
    A = Q.T                                        # (8, 64), σ all 1 ⇒ sr = 8
    B = Q                                          # (64, 8)
    rec = factor_diagnostics(A, B)
    assert abs(rec["stable_rank_A"] - 8.0) < 1e-3
    assert abs(rec["stable_rank_B"] - 8.0) < 1e-3
