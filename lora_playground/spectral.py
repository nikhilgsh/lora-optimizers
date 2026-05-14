"""Spectral-norm (σ_max) primitives used across optimizers.

See `docs/notes/sigma_max_estimation.md` for the cost / fragility / accuracy
trade-offs and the call-site → primitive mapping.

Two computational shapes:

  A. Plain σ_max(M) — largest singular value of a single matrix. Served by
     `sigma_max_power_iter` and `sigma_max_power_iter_batched`. Robust, no
     damping, no factorization. Cost ≈ n_iters × matvec in min(m,n)-space.

  B. σ_max(X·Y) without forming X·Y — for `X: (m, r), Y: (r, n)` where the
     product would be `(m, n)`. Three primitives:

       - `sigma_max_krylov_chol(G_outer, G_inner)`: Krylov-accelerated power
         iter on M = L^T G_outer L with L = chol(G_inner + ε·(trace/r)·I).
         Cheapest per-iter (r-space matvecs) but Cholesky can fail when
         G_inner + ε·I is numerically indefinite — bumped `eps` default
         from the historical 1e-12 to 1e-6 to keep above the bf16-accum
         noise floor.

       - `sigma_max_warm_power_iter_unfactored(X, Y, *, v_init)`: power iter
         on the product structure directly, no Cholesky, no damping. Matvecs
         in (m,n)-space — flops are ~10-30× higher per iter than Krylov-chol
         at chord-direction's typical r=64..256, but warm-start can close
         much of that gap. Numerically unconditional.

       - `sigma_max_multimoment_upper(G_outer, G_inner)`: Su eq 12 strict
         upper bound from two moments. 2 r-space matmuls. No factorization.
         Slightly conservative (tightness depends on spectrum concentration).

The historical `_sigma_max_via_chol_eigh` (heavy-diagnostics only) is
included for parity but flagged not-for-hot-path: it calls `eigvalsh` on a
single (r, r) matrix per call, which kernel-storms when invoked across many
pairs.

**Rules** (see decision doc):
  - Never propose `eigh`/`solve`/`lu` on batched (..., r, r) at r ≥ 64.
  - Bench before refactoring; flop counts are suggestive but not
    load-bearing — kernel launch / memory effects can dominate.
"""
from __future__ import annotations

import torch


# ─── Shape A: plain σ_max ────────────────────────────────────────────────────


def sigma_max_power_iter(M, v_init=None, n_iters=8, eps=1e-30):
    """Estimate σ_max(M) = ‖M‖_op via power iteration on the smaller-side Gram.

    For an (m, n) matrix, iterates on the (k, k) Gram where k = min(m, n).
    Convergence rate is (σ_2/σ_1)^(2·n_iters); on Gaussian random matrices,
    cold start at n_iters=8 gives ~5% p95 rel-error, n_iters=16 gives ~3%.
    With a warm-started v_init across optimizer steps (top singular vector
    barely moves when factor updates are small), n_iters=3 reaches the
    same accuracy.

    Args:
        M: (m, n) tensor on any device, any float dtype.
        v_init: optional warm-start vector of length min(m, n). If None,
            random init. After this call returns, the caller should cache
            the returned v for the next call.
        n_iters: number of M·M^T (or M^T·M) applications.

    Returns:
        (sigma, v_new): scalar 0-dim tensor σ_max estimate, and the
        converged singular vector. Both live on M's device — no
        GPU→CPU sync. Caller does `float(sigma)` only at logging time.

    Verified by `tests/test_sigma_max_power_iter.py`.
    """
    if M.numel() == 0:
        return torch.zeros((), device=M.device, dtype=torch.float32), None
    Mf = M.float() if M.dtype != torch.float32 else M
    m, n = Mf.shape
    if m <= n:
        if (v_init is None or v_init.shape != (m,)
                or v_init.device != Mf.device or v_init.dtype != torch.float32):
            v = torch.randn(m, device=Mf.device, dtype=torch.float32)
        else:
            v = v_init
        v = v / (v.norm() + eps)
        for _ in range(n_iters):
            v = Mf @ (Mf.T @ v)
            v = v / (v.norm() + eps)
        sigma = (Mf.T @ v).norm()
    else:
        if (v_init is None or v_init.shape != (n,)
                or v_init.device != Mf.device or v_init.dtype != torch.float32):
            v = torch.randn(n, device=Mf.device, dtype=torch.float32)
        else:
            v = v_init
        v = v / (v.norm() + eps)
        for _ in range(n_iters):
            v = Mf.T @ (Mf @ v)
            v = v / (v.norm() + eps)
        sigma = (Mf @ v).norm()
    return sigma, v


def sigma_max_power_iter_batched(M, v_init=None, n_iters=8, eps=1e-30):
    """Batched version of `sigma_max_power_iter`. M: (..., m, n).

    Returns (sigma: (...,), v: (..., k)) where k = min(m, n). Iterates
    on the smaller-side Gram via bmm; one launch per iter regardless of
    batch size, vs N launches for the per-pair loop. ~10× faster on
    LoRA-shape inputs at OLMo r=64 (verified by spectral_chord bench).

    Convention matches the per-pair version (v lives on M's device, no
    GPU→CPU sync). Caller supplies `v_init` as the warm-start across
    optimizer steps; for fresh runs pass None and a deterministic init
    (M @ ones-like) is used.
    """
    if M.numel() == 0:
        return torch.zeros(M.shape[:-2], device=M.device, dtype=torch.float32), None
    Mf = M.float() if M.dtype != torch.float32 else M
    *batch, m, n = Mf.shape
    smaller_m = m <= n
    k = m if smaller_m else n
    expected_shape = (*batch, k)
    # Deterministic fallback: M·ones (smaller_m=True) or M^T·ones — gives
    # near-perfect overlap with the dominant singular vector for typical
    # SPD/Gauss-like factor matrices.
    if smaller_m:
        ones_n = torch.ones(*batch, n, device=Mf.device, dtype=torch.float32)
        v_fallback = (Mf @ ones_n.unsqueeze(-1)).squeeze(-1)
    else:
        ones_m = torch.ones(*batch, m, device=Mf.device, dtype=torch.float32)
        v_fallback = (Mf.transpose(-2, -1) @ ones_m.unsqueeze(-1)).squeeze(-1)
    if (v_init is None
            or v_init.shape != expected_shape
            or v_init.device != Mf.device
            or v_init.dtype != torch.float32):
        v = v_fallback
    else:
        # Sticky-zero guard: if a previous step ran with M ≈ 0 (e.g. chord-tight
        # whiten at LoRA step 1 with Init[A] has u_A = 0 because gA = 0 when
        # B = 0), the stored warm-start vector is zero. Power-iter preserves
        # zero across iterations, so the next call would return σ = 0 even when
        # M is nonzero — and downstream divisions by sigma+eps explode u_A and
        # collapse dA. Per-batch fallback to deterministic init when warm-start
        # is degenerate.
        v_norm = v_init.norm(dim=-1, keepdim=True)
        degenerate = v_norm <= eps
        v = torch.where(degenerate, v_fallback, v_init)
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    if smaller_m:
        for _ in range(n_iters):
            tmp = (Mf.transpose(-2, -1) @ v.unsqueeze(-1)).squeeze(-1)
            v = (Mf @ tmp.unsqueeze(-1)).squeeze(-1)
            v = v / (v.norm(dim=-1, keepdim=True) + eps)
        sigma = (Mf.transpose(-2, -1) @ v.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
    else:
        for _ in range(n_iters):
            tmp = (Mf @ v.unsqueeze(-1)).squeeze(-1)
            v = (Mf.transpose(-2, -1) @ tmp.unsqueeze(-1)).squeeze(-1)
            v = v / (v.norm(dim=-1, keepdim=True) + eps)
        sigma = (Mf @ v.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
    return sigma, v


# ─── Shape B: σ_max of a product XY via Gram form ────────────────────────────


def sigma_max_krylov_chol(G_outer, G_inner, eps=1e-6,
                          n_iters=16, k_sub=5):
    """σ_max(XY) for X,Y with G_outer = X^T X, G_inner = Y Y^T.

    Krylov-accelerated symmetric power-iter on M = L^T G_outer L (PSD,
    L = chol(G_inner + ε·(trace/r)·I)). Power-iter generates a sequence;
    the last k_sub vectors are QR-orthonormalized and we solve a small
    (k_sub × k_sub) eigvalsh on Q^T M Q for the top eigenvalue =
    σ_max²(XY). Su's Krylov idea (kexue.fm/archives/11736 §2).

    Verified <1e-4 rel-err on batched (n_pairs=112, r=256) structured
    inputs at n_iters=16, k_sub=5.

    Why Krylov-chol over an exact `eigvalsh(M)`: cuSOLVER's batched
    `eigvalsh` falls back to per-matrix syevd at r ≥ 64 (~10-20 ms each ×
    112 pairs = 1-2 s per call; 3 calls/step in chord-direction → 4-6 s/step
    at r=256). Krylov stays in batched matmul-only ops.

    Fragility: the Cholesky setup fails when G_inner is sufficiently
    rank-deficient that `G_inner + ε·(trace/r)·I` is still numerically
    indefinite under bf16 accumulation noise. **eps default bumped from
    1e-12 to 1e-6 (~1 decade above fp32 accumulation noise floor)** to keep
    Cholesky reliable on h100 / blackwell bf16. Perturbation to σ_max in
    the well-conditioned regime: ≲ ε·κ(G_inner) ≈ 1e-4 worst case.

    Supports leading batch dims on both inputs.
    """
    last = G_inner.shape[-1]
    diag = G_inner.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)
    diag_load = (eps * diag.clamp_min(1e-30)).unsqueeze(-1)
    I_r = torch.eye(last, dtype=G_inner.dtype, device=G_inner.device)
    damped = G_inner + diag_load * I_r
    Lc = torch.linalg.cholesky(damped)
    M = Lc.transpose(-1, -2) @ G_outer @ Lc          # (..., r, r) sym PSD
    batch_shape = M.shape[:-2]
    r = M.shape[-1]
    k_eff = min(k_sub, r)
    v = torch.ones(*batch_shape, r, device=M.device, dtype=M.dtype)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-30)
    history = []
    for it in range(n_iters):
        v = (M @ v.unsqueeze(-1)).squeeze(-1)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-30)
        if it >= n_iters - k_eff:
            history.append(v)
    V = torch.stack(history, dim=-1)                 # (..., r, k)
    Q, _ = torch.linalg.qr(V, mode='reduced')        # (..., r, k)
    M_proj = Q.transpose(-1, -2) @ M @ Q             # (..., k, k)
    sigma_sq = torch.linalg.eigvalsh(M_proj).max(dim=-1).values
    return sigma_sq.clamp_min(0.0).sqrt()


def sigma_max_power_iter_nonsym(G_outer, G_inner, v_init=None, n_iters=16,
                                 eps=1e-30):
    """σ_max(XY) by plain power iter on N = G_outer · G_inner (r×r, non-symmetric).

    For X, Y with G_outer = X^T X, G_inner = Y Y^T, the matrix N is r×r
    and non-symmetric, but its eigenvalues are real and non-negative (N
    is similar to the symmetric PSD G_outer^{1/2} · G_inner · G_outer^{1/2}).
    The largest eigenvalue of N equals σ_max²(XY).

    Algorithm: form N once, then power iter v ← N v / ‖N v‖ for n_iters.
    Return σ_max(XY) ≈ √‖N v‖ from the Rayleigh quotient on the converged
    final iterate (unit-norm v ⇒ ‖N v‖ ≈ λ_max(N) at convergence).
    Convergence rate (|λ_2|/|λ_1|)^T per matvec — at n_iters=16 with
    typical chord-direction spectra this gives <1e-3 rel-error.

    Why not the Krylov last-k projection trick? It seemed cheap and tight
    (smaller k_sub × k_sub eigvals on the projection of N onto the
    last-k iterates' QR basis) but fails in practice: when power iter has
    fully converged, the last-k iterates are nearly parallel to the
    dominant eigenvector, the k_sub × k_sub projection has repeated
    eigenvalues, and cuSOLVER's iterative eigvals raises
    `_LinAlgError: ill-conditioned or repeated eigenvalues`. Observed
    crashing the optimizer at chord-direction k=3 r=64 lr=1e-2 step
    ~3800. The plain Rayleigh quotient on the final iterate is robust
    and accurate enough.

    Robust to rank-deficient G_inner — no damping, no factorization;
    σ_max(XY) is always finite by submultiplicativity even when G_inner
    is singular.

    Args:
        G_outer: (..., r, r) SPD. = X^T X.
        G_inner: (..., r, r) SPD (possibly rank-deficient). = Y Y^T.
        v_init: optional warm-start vector of shape (..., r).
        n_iters: number of N matvecs (default 16).

    Returns:
        (sigma, v_new): σ_max(XY) and converged dominant eigenvector of N
        for warm-starting the next call.
    """
    N = G_outer @ G_inner                              # (..., r, r) non-symmetric
    batch_shape = N.shape[:-2]
    r = N.shape[-1]
    if (v_init is None
            or v_init.shape != (*batch_shape, r)
            or v_init.device != N.device
            or v_init.dtype != N.dtype):
        v = torch.ones(*batch_shape, r, device=N.device, dtype=N.dtype)
    else:
        v = v_init
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    for _ in range(n_iters):
        v = (N @ v.unsqueeze(-1)).squeeze(-1)
        v = v / (v.norm(dim=-1, keepdim=True) + eps)
    # Rayleigh quotient: ‖N v‖ ≈ λ_max(N) = σ²_max(XY) for converged unit v.
    Nv = (N @ v.unsqueeze(-1)).squeeze(-1)
    sigma_sq = Nv.norm(dim=-1)
    return sigma_sq.clamp_min(0.0).sqrt(), v


def sigma_max_warm_power_iter_unfactored(X, Y, v_init=None, n_iters=8,
                                          eps=1e-30):
    """σ_max(X·Y) by power iter on the product structure, no Cholesky.

    For X: (..., m, r), Y: (..., r, n), computes σ_max(X·Y) = ‖XY‖_op.
    The product XY is NOT formed explicitly. Each power-iter step is
    4 matvecs:
        y  = Y @ v       # (..., r,)
        u  = X @ y       # (..., m,)
        σ  = ‖u‖
        u  = u / σ
        y' = X^T @ u     # (..., r,)
        v  = Y^T @ y'    # (..., n,)
        v  = v / ‖v‖

    Numerically unconditional — σ_max(XY) is always finite (≤ σ_max(X) ·
    σ_max(Y) by submultiplicativity); no factorization, no damping, no
    "what does σ_max mean when G_inner is rank-deficient" pathology.

    Cost vs `sigma_max_krylov_chol`: matvecs are in (m,n)-space rather
    than r-space, so per-iter flops are ~10-30× higher at chord-direction's
    r/m/n ratios. Warm-started v_init from the previous optimizer step
    (top singular vector barely moves) lets n_iters=2-3 match cold n_iters=8.

    Args:
        X: (..., m, r) — left factor of the product.
        Y: (..., r, n) — right factor.
        v_init: warm-start vector of shape (..., n). None → deterministic
            init from Y @ ones.
        n_iters: number of power-iter passes (full 4-matvec cycles).
        eps: norm clamp for division stability.

    Returns:
        (sigma, v_new): σ_max(XY) and updated right-singular-vector for
        warm-starting the next call.
    """
    if X.numel() == 0 or Y.numel() == 0:
        batch_shape = X.shape[:-2] if X.numel() > 0 else Y.shape[:-2]
        return (torch.zeros(*batch_shape, device=X.device, dtype=torch.float32),
                None)
    Xf = X.float() if X.dtype != torch.float32 else X
    Yf = Y.float() if Y.dtype != torch.float32 else Y
    *batch, m, r = Xf.shape
    *_, r_y, n = Yf.shape
    assert r == r_y, f"X r={r} != Y r={r_y}"
    # Deterministic warm-start: v_fallback = Y^T @ (X^T @ ones) projected to (n,)
    # Equivalent to one half-iter from "all ones" in m-space.
    ones_m = torch.ones(*batch, m, device=Xf.device, dtype=torch.float32)
    half_iter = (Xf.transpose(-2, -1) @ ones_m.unsqueeze(-1)).squeeze(-1)  # (..., r)
    v_fallback = (Yf.transpose(-2, -1) @ half_iter.unsqueeze(-1)).squeeze(-1)  # (..., n)
    expected_shape = (*batch, n)
    if (v_init is None
            or v_init.shape != expected_shape
            or v_init.device != Xf.device
            or v_init.dtype != torch.float32):
        v = v_fallback
    else:
        v_norm = v_init.norm(dim=-1, keepdim=True)
        degenerate = v_norm <= eps
        v = torch.where(degenerate, v_fallback, v_init)
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    sigma = None
    for _ in range(n_iters):
        y = (Yf @ v.unsqueeze(-1)).squeeze(-1)            # (..., r)
        u = (Xf @ y.unsqueeze(-1)).squeeze(-1)            # (..., m)
        sigma = u.norm(dim=-1)
        u = u / (sigma.unsqueeze(-1) + eps)
        y_prime = (Xf.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1)  # (..., r)
        v_new = (Yf.transpose(-2, -1) @ y_prime.unsqueeze(-1)).squeeze(-1)  # (..., n)
        v = v_new / (v_new.norm(dim=-1, keepdim=True) + eps)
    # Final σ estimate: ‖X·(Y·v)‖, no extra normalization needed.
    if sigma is None:
        y = (Yf @ v.unsqueeze(-1)).squeeze(-1)
        u = (Xf @ y.unsqueeze(-1)).squeeze(-1)
        sigma = u.norm(dim=-1)
    return sigma, v


def sigma_max_multimoment_upper(G_outer, G_inner):
    """Strict upper bound on σ_max(XY) via Su's multi-moment formula (eq 12,
    kexue.fm/archives/11736).

    For X, Y with G_outer = X^T X, G_inner = Y Y^T, the spectral norm of
    X·Y satisfies:

        σ_max(XY) ≤ sqrt((S₂ + sqrt((m-1)·(m·S₄ - S₂²))) / m)

    where
        m  = rank dimension (typically r for shape-B usage)
        S₂ = sum of squared singular values of XY
           = tr(X^T X · Y Y^T) = tr(G_outer · G_inner)
        S₄ = sum of σᵢ⁴(XY)
           = tr((X^T X · Y Y^T)²) = tr((G_outer · G_inner)²)

    Bound is tight when XY's spectrum is concentrated in σ_max (low effective
    rank); loose when the spectrum is flat. For chord-direction's
    rank-deficient B early in training, tight; for well-spread later
    iterates, slightly loose.

    Cost: two batched r×r matmuls + two traces + scalar ops. No
    Cholesky, no factorization, no iteration. Strict upper bound by
    construction — never underestimates.

    Args:
        G_outer: (..., r, r) — X^T X.
        G_inner: (..., r, r) — Y Y^T.

    Returns:
        sigma_upper: (...,) — upper bound on σ_max(XY).
    """
    # Form M = G_outer · G_inner (non-symmetric r×r). σ²ᵢ(XY) are eigenvalues
    # of M. S₂ = tr(M), S₄ = tr(M²) = sum_{i,j} M_ij · M_ji.
    M = G_outer @ G_inner                              # (..., r, r)
    r = M.shape[-1]
    # S2 = trace(M) = sum of diag
    S2 = M.diagonal(dim1=-2, dim2=-1).sum(dim=-1)      # (...,)
    # S4 = trace(M @ M) = sum_{i,j} M_ij · M_ji = (M * M.T).sum()
    # (Cheaper than forming M @ M explicitly.)
    S4 = (M * M.transpose(-1, -2)).sum(dim=(-1, -2))   # (...,)
    # Multi-moment bound (eq 12):
    #   σ_max ≤ sqrt((S2 + sqrt((m-1)·(m·S4 - S2²))) / m)
    m = float(r)
    inner = ((m - 1.0) * (m * S4 - S2 * S2)).clamp_min(0.0)
    return ((S2 + inner.sqrt()) / m).clamp_min(0.0).sqrt()


# ─── Diagnostic-only (single-pair, NOT for hot path) ─────────────────────────


def sigma_max_via_chol_eigh_single(G_outer, G_inner, eps=1e-12):
    """Exact σ_max(XY) on a single (r, r) Gram pair via Cholesky + eigvalsh.

    Used by heavy-diagnostics only — NOT for the hot path. `eigvalsh` on a
    batched (..., r, r) tensor kernel-storms at r ≥ 64 (see decision doc).
    Calling this once per pair in a Python loop (as the diagnostic logger
    does) is fine; calling it inside a batched op is not.

    Returns a Python float on CPU after one sync. Inputs must be 2-D.
    """
    diag_load = eps * G_inner.diagonal().abs().mean().clamp_min(1e-30)
    damped = G_inner + diag_load * torch.eye(
        G_inner.shape[-1], dtype=G_inner.dtype, device=G_inner.device)
    Lc = torch.linalg.cholesky(damped)
    M = Lc.T @ G_outer @ Lc
    return float(torch.linalg.eigvalsh(M).clamp_min(0.0).max().sqrt())
