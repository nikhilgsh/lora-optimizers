"""CPU-only unit tests for the LoRSUM / F-LoRSUM proximal subspace iteration
subroutines (PSI-LoRA paper eq. 10 and 14)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.utils import f_lorsum, lorsum, truncated_svd


def _decompose_dense(W, rank):
    """Get rank-r factors (A, B) such that B @ A ≈ W in Frobenius norm.
    Returns A: (r, d_in), B: (d_out, r)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    B = U[:, :rank] * S[:rank]
    A = Vh[:rank]
    return A, B


def test_lorsum_recovers_truncated_svd_with_many_iters():
    """LoRSUM with K large should converge to the rank-r truncated SVD of the
    sum of factors. Test on a known rank-r matrix."""
    torch.manual_seed(0)
    d_out, d_in, r = 12, 8, 3
    # Build a target W = factor_out @ factor_in with rank exactly r
    factor_in = torch.randn(r, d_in)
    factor_out = torch.randn(d_out, r)
    W_target = factor_out @ factor_in

    # Initial guess: random rank-r factors
    A_init = torch.randn(r, d_in)
    B_init = torch.randn(d_out, r)

    # LoRSUM with the prox center being the random init and ONE term equal to W_target.
    # Coefficients (1, 0): the prox center is the first factor; we want to project
    # exactly onto the second factor with the prox shrinking the role of the first.
    factors = [(A_init, B_init), (factor_in, factor_out)]
    A_new, B_new = lorsum(
        factors=factors,
        coefficients=[0.0, 1.0],   # ignore prox center as a contribution; only the gradient term matters
        num_iters=20,
        lmbd=1e-6,                 # near-zero proximal regularization
    )
    W_approx = B_new @ A_new
    err = (W_approx - W_target).norm() / W_target.norm()
    assert err < 1e-3, f"LoRSUM failed to converge to target: rel err={err.item():.4e}"


def test_lorsum_proximal_pull_zero_coeffs():
    """With all coefficients zero on the gradient terms but nonzero ρ on the prox
    center, the solution must equal the prox center."""
    torch.manual_seed(1)
    d_out, d_in, r = 6, 4, 2
    A0 = torch.randn(r, d_in)
    B0 = torch.randn(d_out, r)
    extra_in = torch.randn(r, d_in)
    extra_out = torch.randn(d_out, r)
    A_new, B_new = lorsum(
        factors=[(A0, B0), (extra_in, extra_out)],
        coefficients=[1.0, 0.0],   # only prox center contributes
        num_iters=5,
        lmbd=1.0,
    )
    # B_new @ A_new ≈ B0 @ A0 (the projection's prox center)
    assert torch.allclose(B_new @ A_new, B0 @ A0, atol=1e-3)


def test_f_lorsum_reduces_to_lorsum_for_unit_metrics():
    """When D_U = D_V = ones (and γ=1 so M_U = M_V = I), F-LoRSUM should produce
    nearly the same result as LoRSUM."""
    torch.manual_seed(2)
    d_out, d_in, r = 8, 6, 2
    A0 = torch.randn(r, d_in)
    B0 = torch.randn(d_out, r)
    factor_in = torch.randn(r, d_in)
    factor_out = torch.randn(d_out, r)
    factors = [(A0, B0), (factor_in, factor_out)]
    coeffs = [1.0, -0.1]

    A_pl, B_pl = lorsum(factors, coeffs, num_iters=3, lmbd=1e-2)
    D_U = torch.ones(d_out)
    D_V = torch.ones(d_in)
    A_fl, B_fl = f_lorsum(factors, coeffs, D_U=D_U, D_V=D_V,
                          num_iters=3, lmbd=1e-2, gamma=1.0, delta=0.0)

    err = (B_fl @ A_fl - B_pl @ A_pl).norm() / max((B_pl @ A_pl).norm().item(), 1e-9)
    assert err < 1e-3, f"F-LoRSUM (unit metrics) ≠ LoRSUM: rel err={err.item():.4e}"


def test_f_lorsum_runs_with_metrics():
    """Smoke test: F-LoRSUM with non-trivial metrics returns sane shapes and
    finite values."""
    torch.manual_seed(3)
    d_out, d_in, r = 8, 6, 2
    A0 = torch.randn(r, d_in)
    B0 = torch.randn(d_out, r)
    factor_in = torch.randn(r, d_in)
    factor_out = torch.randn(d_out, r)
    D_U = torch.rand(d_out) + 0.5
    D_V = torch.rand(d_in) + 0.5
    A_new, B_new = f_lorsum(
        factors=[(A0, B0), (factor_in, factor_out)],
        coefficients=[1.0, -0.05],
        D_U=D_U, D_V=D_V,
        num_iters=2, lmbd=1e-3, gamma=0.5, delta=1e-5,
    )
    assert A_new.shape == (r, d_in)
    assert B_new.shape == (d_out, r)
    assert torch.isfinite(A_new).all() and torch.isfinite(B_new).all()
