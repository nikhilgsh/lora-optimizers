"""Tests for K-way batched parallel-grid MISR bisect (optimization C).

The parallel variant `_ssc_misr_bisect_batched_kpar` evaluates K log-spaced
candidate c values per pair in ONE batched MISR launch (vs K sequential
launches in `_ssc_misr_bisect_batched`). These tests assert:

  (1) Parallel-K with a warm c_init lands within log_window of the seq-K
      converged c (since the parallel grid covers the same bracket).
  (2) The output H returned by parallel-K equals MISR applied with the
      winning c — i.e., post-selection picks consistent (H, c) pairs.
  (3) c_solved values lie on the prescribed log-c grid.
  (4) Edge case: K=1 returns exactly c_init (single-point grid).
"""

import math

import pytest
import torch

from lora_playground.optim import (
    _ssc_misr_batched,
    _ssc_misr_bisect_batched,
    _ssc_misr_bisect_batched_kpar,
)


def _rescaled_X(N=4, r=8, d=32, seed=0):
    """Random batch of (N, r, d) matrices pre-rescaled so σ_max(X) ≈ 1
    per pair (matches the §2.5-rescaled input convention assumed by
    `_ssc_misr_batched`'s closed-form σ_max(H_c) = c/√(c²+1))."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, r, d, generator=g, dtype=torch.float32)
    # Per-pair max-singular-value rescale.
    sigma_max = torch.linalg.svdvals(X)[:, 0]  # (N,)
    return X / sigma_max.view(N, 1, 1)


def test_kpar_K1_returns_c_init():
    X = _rescaled_X(N=3, r=4, d=16, seed=1)
    c_init = torch.tensor([0.3, 0.5, 0.8])
    out, c_solved = _ssc_misr_bisect_batched_kpar(
        X, kappa=0.5, K=1, nsteps=8, c_init=c_init, log_window=0.5,
    )
    assert torch.allclose(c_solved, c_init, atol=1e-6)
    H_ref = _ssc_misr_batched(X, c=c_solved, nsteps=8)
    assert torch.allclose(out, H_ref, atol=1e-6)


def test_kpar_output_matches_misr_at_solved_c():
    """Property (2): for each pair, the returned H equals MISR(X, c_solved)."""
    X = _rescaled_X(N=5, r=8, d=32, seed=2)
    c_init = torch.full((5,), 0.4)
    out, c_solved = _ssc_misr_bisect_batched_kpar(
        X, kappa=0.6, K=5, nsteps=10, c_init=c_init, log_window=0.5,
    )
    H_ref = _ssc_misr_batched(X, c=c_solved, nsteps=10)
    # The selected H[winner, pair_idx] from the (K, N, r, d) MISR batch must
    # match a fresh MISR call with the same c — same kernel, same inputs.
    assert torch.allclose(out, H_ref, atol=1e-5)


def test_kpar_c_on_log_grid():
    """Property (3): c_solved lies on log_c_init + linspace(-w, +w, K)."""
    X = _rescaled_X(N=6, r=8, d=32, seed=3)
    c_init = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    K, log_window = 5, 0.5
    _, c_solved = _ssc_misr_bisect_batched_kpar(
        X, kappa=0.6, K=K, nsteps=10, c_init=c_init, log_window=log_window,
    )
    offsets = torch.linspace(-log_window, log_window, K)
    grid = c_init.log().unsqueeze(0) + offsets.unsqueeze(1)  # (K, N)
    grid = grid.exp()
    # c_solved[i] must equal grid[k, i] for some k.
    for i in range(c_init.numel()):
        diffs = (grid[:, i] - c_solved[i]).abs()
        assert diffs.min() < 1e-5, (
            f"pair {i}: c_solved={c_solved[i].item()} not on grid {grid[:, i].tolist()}"
        )


def test_kpar_within_log_window_of_sequential():
    """Property (1): parallel-K lands within log_window of seq-K converged c
    (both started from the same c_init, same bracket). They won't agree
    exactly — sequential refines to log_window/2^K while parallel quantizes
    to log_window/(K-1) — but both stay within the initial bracket."""
    X = _rescaled_X(N=8, r=8, d=32, seed=4)
    c_init = torch.full((8,), 0.4)
    kappa, K, log_window = 0.6, 3, 0.5
    _, c_seq = _ssc_misr_bisect_batched(
        X, kappa=kappa, K=K, nsteps=10, c_init=c_init, log_window=log_window,
    )
    _, c_par = _ssc_misr_bisect_batched_kpar(
        X, kappa=kappa, K=K, nsteps=10, c_init=c_init, log_window=log_window,
    )
    # Both must lie in the bracket [log(c_init) - w, log(c_init) + w].
    log_init = c_init.log()
    for c, name in [(c_seq, "seq"), (c_par, "par")]:
        diff = (c.log() - log_init).abs()
        assert (diff <= log_window + 1e-5).all(), (
            f"{name} c outside bracket: max log-diff {diff.max().item()}"
        )
    # Parallel-K=3 grid step is log_window (3 points span 2*log_window).
    # Sequential-K=3 refines to ~log_window/8. So |log(c_par) - log(c_seq)|
    # is at most one parallel-grid step.
    par_step = 2 * log_window / (K - 1)
    log_diff = (c_par.log() - c_seq.log()).abs()
    assert (log_diff <= par_step + 1e-5).all(), (
        f"par vs seq diff exceeds one par-grid step {par_step}: "
        f"max {log_diff.max().item()}"
    )


def test_kpar_requires_c_init():
    X = _rescaled_X(N=2, r=4, d=16, seed=5)
    with pytest.raises(ValueError, match="c_init"):
        _ssc_misr_bisect_batched_kpar(
            X, kappa=0.5, K=3, nsteps=8, c_init=None, log_window=0.5,
        )


def test_kpar_shapes_preserved_with_leading_dims():
    """Caller passes X shaped (M, N, r, d) with leading-batch and N-pair
    dims — output should preserve leading dims."""
    X = _rescaled_X(N=6, r=8, d=32, seed=6).reshape(2, 3, 8, 32)
    c_init = torch.full((2, 3), 0.4)
    out, c_solved = _ssc_misr_bisect_batched_kpar(
        X, kappa=0.5, K=3, nsteps=8, c_init=c_init, log_window=0.5,
    )
    assert out.shape == X.shape
    assert c_solved.shape == (2, 3)
