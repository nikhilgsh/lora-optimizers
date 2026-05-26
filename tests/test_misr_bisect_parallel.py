"""Tests for the K-way parallel-grid MISR-bisect variant.

Compares `_ssc_misr_bisect_batched_kpar` against the sequential
`_ssc_misr_bisect_batched`. Sequential K=3 bisection refines the log-c
bracket to log_window / 2^K = log_window/8 resolution. Parallel-grid with
K candidates on the full ±log_window bracket has residual log_window/(K-1)
(closest grid point is at most half the spacing = log_window/(K-1)).

To match seq-K=3 (~log_window/8), parallel needs K such that
log_window/(K-1) ≲ log_window/4, i.e. K ≥ 9 (1/8 of bracket on each side).

The agreement tolerance below is therefore set to one parallel grid step
in log-c: |log(c_par) - log(c_seq)| ≤ log_window / (K_par - 1).
"""
import math

import numpy as np
import pytest
import torch

from lora_playground.optim import (
    _ssc_misr_batched,
    _ssc_misr_bisect_batched,
    _ssc_misr_bisect_batched_kpar,
)


def _make_inputs(N=4, r=8, d=16, seed=0):
    """Build a small batched X with σ_max(X) = 1 (MISR contract)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, r, d, generator=g, dtype=torch.float64)
    # Pre-rescale each pair so σ_max(X_n) = 1 — that's the MISR convention.
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    S = S / S[..., :1]  # σ_max → 1
    X = U @ torch.diag_embed(S) @ Vt
    return X.float()


def _bulk_seed_c(X, kappa, eps=1e-12):
    """Cheap warm-start c via the bulk closed form (no eigvalsh required)."""
    F2 = (X ** 2).sum(dim=(-2, -1)).clamp_min(eps)
    r = X.shape[-2]
    mu = F2 / r  # since σ_max = 1
    # c_bulk = sqrt(μ(1-κ)/(κ-μ)), valid when μ < κ. Clamp for safety.
    num = (mu * (1.0 - kappa)).clamp_min(eps)
    den = (kappa - mu).clamp_min(eps)
    return (num / den).sqrt()


def test_parallel_grid_matches_sequential_within_one_step():
    torch.manual_seed(0)
    N, r, d = 4, 8, 16
    kappa = 0.6
    log_window = 0.5
    K_seq = 3
    K_par = 9  # 9 candidates ⇒ residual log_window/(K_par-1) = 0.5/8 = 0.0625

    X = _make_inputs(N=N, r=r, d=d, seed=0)
    c_init = _bulk_seed_c(X, kappa)

    out_seq, c_seq = _ssc_misr_bisect_batched(
        X, kappa=kappa, K=K_seq, nsteps=10,
        c_init=c_init, log_window=log_window,
    )
    out_par, c_par = _ssc_misr_bisect_batched_kpar(
        X, kappa=kappa, K=K_par, nsteps=10,
        c_init=c_init, log_window=log_window,
    )

    # Bracket is [log(c_init) - log_window, log(c_init) + log_window], width
    # 2·log_window. Parallel-K_par grid spacing = 2·log_window / (K_par-1).
    # Sequential residual ≈ 2·log_window / 2^K_seq. Allow the sum (each
    # method's solution can be off by its own residual from the true c*).
    tol_log = (2 * log_window) / (K_par - 1) + (2 * log_window) / (2 ** K_seq)
    log_err = (c_par.log() - c_seq.log()).abs()
    assert torch.all(log_err <= tol_log + 1e-5), (
        f"Parallel-K={K_par} should land within one grid step (log_err ≤ "
        f"{tol_log:.4f}) of sequential-K={K_seq}; got max={log_err.max():.4f}"
    )


def test_parallel_grid_output_consistent_with_solved_c():
    """The H returned by the parallel-grid path must equal _ssc_misr_batched(X, c_solved)."""
    torch.manual_seed(0)
    N, r, d = 4, 8, 16
    kappa = 0.6
    X = _make_inputs(N=N, r=r, d=d, seed=1)
    c_init = _bulk_seed_c(X, kappa)

    out_par, c_par = _ssc_misr_bisect_batched_kpar(
        X, kappa=kappa, K=9, nsteps=10,
        c_init=c_init, log_window=0.5,
    )
    # Recompute H_c(X) directly with the solved c.
    out_direct = _ssc_misr_batched(X, c=c_par, nsteps=10)
    # Same MISR formula, same c, same X ⇒ should be bit-identical.
    diff = (out_par - out_direct).abs().max().item()
    assert diff < 1e-5, f"parallel-grid winner-H disagrees with MISR(c_solved): max|Δ|={diff:.2e}"


def test_parallel_grid_requires_c_init():
    X = _make_inputs(N=2, r=4, d=8)
    with pytest.raises(ValueError, match="c_init"):
        _ssc_misr_bisect_batched_kpar(X, kappa=0.6, K=3, nsteps=5, c_init=None)


def test_parallel_grid_K1_degenerate():
    """K=1 collapses to a single candidate at log(c_init) — should return c_init."""
    torch.manual_seed(0)
    X = _make_inputs(N=3, r=4, d=8, seed=2)
    c_init = _bulk_seed_c(X, kappa=0.6)
    out, c_solved = _ssc_misr_bisect_batched_kpar(
        X, kappa=0.6, K=1, nsteps=5, c_init=c_init, log_window=0.5,
    )
    assert torch.allclose(
        c_solved, c_init.clamp(1e-3, 1e3), atol=1e-6,
    ), "K=1 should return clamped c_init"
