"""Tests for Soft Spectral Clipping (SSC) primitives.

H_c(X) = U diag(h_c(σ)) V^T  with  h_c(σ) = σ / sqrt(1 + (σ/c)²)
       = (I + X X^T / c²)^{-1/2} · X    (operator form)

`_ssc_svd` is the reference. `_ssc_misr_batched` is a Newton-Schulz Algorithm 2
(SPECTRA) computation of the inverse square root.
"""

import torch

from lora_playground.optim import _ssc_misr_batched, _ssc_svd


def _ssc_eigh_ref(M, c):
    """Manual reference via eigh on (I + X X^T / c²). Used to cross-check SVD path."""
    Mf = M.float()
    tall = Mf.shape[-2] > Mf.shape[-1]
    if tall:
        Mf = Mf.transpose(-2, -1)
    r = Mf.shape[-2]
    I_r = torch.eye(r, dtype=Mf.dtype, device=Mf.device).expand(*Mf.shape[:-2], r, r)
    G = I_r + (Mf @ Mf.transpose(-2, -1)) / (c * c)
    # G is symmetric PSD; G^{-1/2} via eigendecomposition.
    evals, evecs = torch.linalg.eigh(G)
    inv_sqrt = (evecs * evals.clamp_min(1e-30).rsqrt().unsqueeze(-2)) @ evecs.transpose(-2, -1)
    out = inv_sqrt @ Mf
    return out.transpose(-2, -1) if tall else out


def test_ssc_svd_matches_eigh_reference():
    torch.manual_seed(0)
    # `_ssc_svd` casts to float32 internally for SVD stability; tolerance is
    # set to the float32 SVD precision (~1e-6), not double precision.
    for shape in [(16, 64), (8, 8), (4, 32)]:
        for c in [0.3, 1.0, 5.0]:
            M = torch.randn(*shape)
            out_svd = _ssc_svd(M, c).float()
            out_ref = _ssc_eigh_ref(M, c).float()
            err = (out_svd - out_ref).norm() / out_ref.norm().clamp_min(1e-30)
            assert err < 1e-5, f"shape={shape} c={c}: rel err {err:.3e}"


def test_ssc_misr_matches_svd_on_normalized_input():
    torch.manual_seed(1)
    # Pre-rescale so σ_max ≤ 1 (the pipeline invariant).
    for shape in [(8, 16), (16, 64), (4, 4)]:
        M = torch.randn(*shape)
        M = M / torch.linalg.matrix_norm(M, ord=2)
        for c in [0.3, 0.5, 1.0, 3.0]:
            out_ref = _ssc_svd(M, c).float()
            out_ns = _ssc_misr_batched(M, c, nsteps=15)
            err = (out_ns - out_ref).norm() / out_ref.norm().clamp_min(1e-30)
            assert err < 1e-4, f"shape={shape} c={c}: rel err {err:.3e}"


def test_ssc_misr_batched_leading_dims():
    torch.manual_seed(2)
    X = torch.randn(3, 5, 8, 16)
    X = X / torch.linalg.matrix_norm(X, ord=2).unsqueeze(-1).unsqueeze(-1)
    out_ref = _ssc_svd(X, 0.5).float()
    out_ns = _ssc_misr_batched(X, 0.5, nsteps=15)
    err = (out_ns - out_ref).norm() / out_ref.norm()
    assert err < 1e-4


def test_ssc_limit_large_c_identity():
    # c → ∞ ⇒ h_c(σ) → σ ⇒ H_c(X) → X.
    torch.manual_seed(3)
    X = torch.randn(8, 16)
    X = X / torch.linalg.matrix_norm(X, ord=2)
    out = _ssc_svd(X, c=1e6)
    err = (out - X).norm() / X.norm()
    assert err < 1e-6


def test_ssc_limit_small_c_op_norm_bounded_by_c():
    # For any X, ‖H_c(X)‖_op = max σ / sqrt(1 + (σ/c)²) ≤ c (the saturation level).
    torch.manual_seed(4)
    X = torch.randn(8, 16) * 10.0  # large σ
    for c in [0.05, 0.1, 0.3]:
        out = _ssc_svd(X, c)
        op_norm = torch.linalg.matrix_norm(out, ord=2).item()
        assert op_norm <= c + 1e-5, f"c={c}: ‖H_c‖_op = {op_norm:.4e} > c"
