"""The `pre_norm="spec"` path of `_newton_schulz` (used only by
curvature-whiten-polar) now computes σ_max via the library power-iter
(`spectral.sigma_max_power_iter`) instead of a full SVD. Guard that the swap
(a) never dangerously underestimates (NS needs scaled σ < √3) and (b) leaves
the NS output orthogonalizing as before. CPU, tiny tensors, deterministic.
"""
import torch

from lora_playground.optim import _newton_schulz
from lora_playground.spectral import sigma_max_power_iter


def test_poweriter_sigma_max_safe_for_ns_basin():
    # Rayleigh-style power-iter is a lower bound on σ_max; the NS danger is
    # underestimate → scaled σ > 1. NS still converges while scaled σ < √3, i.e.
    # est must stay > true/√3 ≈ true/1.73. Check across shapes incl. near-tie.
    cases = [torch.randn(r, d, generator=torch.Generator().manual_seed(s)).float()
             for (r, d) in [(8, 8), (8, 32), (16, 64), (64, 256)] for s in range(4)]
    for X in cases:
        est = sigma_max_power_iter(X, n_iters=8)[0].item()
        true = torch.linalg.matrix_norm(X.double(), ord=2).item()
        assert est <= true * 1.01, f"σ_max overestimate: est={est} true={true}"
        assert est > true / 1.7, f"underestimate too large (scaled σ≥√3): est={est} true={true}"


def test_ns_spec_still_orthogonalizes_and_matches_svd_prenorm():
    # NS output rows ~orthonormal (YYᵀ ≈ I_r for r ≤ d); the power-iter spec path
    # matches an exact-SVD pre-norm reference closely (the pre-norm scale washes
    # out through the iteration + the caller's downstream renormalization).
    for s in range(4):
        X = torch.randn(16, 64, generator=torch.Generator().manual_seed(s)).float()
        Y = _newton_schulz(X.clone(), nsteps=5, pre_norm="spec")
        Xr = X / (torch.linalg.matrix_norm(X, ord=2) + 1e-7)
        for _ in range(5):
            Xr = 1.5 * Xr - 0.5 * Xr @ Xr.T @ Xr
        assert torch.allclose(Y, Xr, atol=2e-3, rtol=2e-2), \
            f"seed={s}: max|Δ| vs SVD-prenorm ref = {(Y - Xr).abs().max().item():.2e}"
        off = (Y @ Y.T - torch.eye(16)).abs().max().item()
        assert off < 0.15, f"seed={s}: rows not ~orthonormal, max off-I={off:.3f}"
