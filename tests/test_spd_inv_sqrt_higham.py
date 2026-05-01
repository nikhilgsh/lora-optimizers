"""Unit tests for spd_inv_sqrt_higham — the Newton-Schulz inverse-sqrt
replacement for spd_frac_power_inv(H, gamma=0.5)."""
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.utils import spd_frac_power_inv, spd_inv_sqrt_higham


def _wishart(r, d_in, seed=0):
    """Realistic LoRA-factor Gram: A is (r, d_in), d_in >> r → well-conditioned."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, d_in, generator=g) * (1.0 / d_in ** 0.5)
    return A @ A.T + 1e-6 * torch.eye(r)


@pytest.mark.parametrize("r,d_in", [(16, 2048), (64, 2048), (256, 2048)])
def test_higham_matches_eigh_on_wishart(r, d_in):
    """For realistic LoRA-factor Gram (Wishart with d_in >> r, well-conditioned),
    Higham with default 10 NS iters should match eigh-based reference to float32 noise."""
    H = _wishart(r, d_in, seed=0)
    ref = spd_frac_power_inv(H, gamma=0.5, eps=1e-6)
    approx = spd_inv_sqrt_higham(H, eps=1e-6)
    rel_err = (approx - ref).norm() / ref.norm()
    assert rel_err < 1e-4, f"Higham at r={r}: rel_err {rel_err:.2e} too large"


def _conditioned_spd(r, kappa, seed=0):
    """SPD with condition number ≈ kappa, log-spaced eigenvalues."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(r, r, generator=g))
    evals = torch.logspace(0, -torch.tensor(float(kappa)).log10().item(), r)
    return Q @ torch.diag(evals) @ Q.T + 1e-8 * torch.eye(r)


@pytest.mark.parametrize("kappa", [10, 100, 200])
def test_higham_handles_realistic_conditioning(kappa):
    """During training, SB develops condition numbers up to ~200 (observed in
    polar-product r=64 sweep diagnostics). Default 10 iters must handle this."""
    r = 64
    H = _conditioned_spd(r, kappa, seed=3)
    approx = spd_inv_sqrt_higham(H, eps=1e-8)
    recon = approx @ H @ approx
    err = (recon - torch.eye(r)).norm() / (r ** 0.5)
    assert err < 1e-3, f"κ={kappa}: reconstruction error {err:.2e} too large"


@pytest.mark.parametrize("r,d_in", [(16, 2048), (64, 2048), (256, 2048)])
def test_higham_reconstruction(r, d_in):
    """Z @ H @ Z should be ≈ I. This is the actual semantics we care about
    (we're computing H^{-1/2}, not necessarily matching eigh's output bit-for-bit)."""
    H = _wishart(r, d_in, seed=1)
    approx = spd_inv_sqrt_higham(H, n_iters=5, eps=1e-6)
    recon = approx @ H @ approx
    err = (recon - torch.eye(r)).norm() / (r ** 0.5)
    assert err < 1e-4, f"Reconstruction error {err:.2e} too large at r={r}"


def test_higham_handles_identity():
    H = torch.eye(64)
    approx = spd_inv_sqrt_higham(H, n_iters=5, eps=1e-6)
    rel_err = (approx - torch.eye(64)).norm() / (64 ** 0.5)
    assert rel_err < 1e-5, f"Identity case off by {rel_err:.2e}"


def test_higham_diagonal():
    diag_vals = torch.tensor([0.1, 0.5, 1.0, 2.0, 4.0])
    H = torch.diag(diag_vals)
    approx = spd_inv_sqrt_higham(H, n_iters=10, eps=1e-7)
    expected = torch.diag(diag_vals.pow(-0.5))
    rel_err = (approx - expected).norm() / expected.norm()
    assert rel_err < 1e-3, f"Diagonal case off by {rel_err:.2e}"


def test_higham_determinism():
    """Same input + same seed for power iter → same output. The power iter uses
    a torch global-state randn under the hood; we set the seed before each call."""
    H = _wishart(64, 2048, seed=2)
    torch.manual_seed(0)
    a = spd_inv_sqrt_higham(H, n_iters=5)
    torch.manual_seed(0)
    b = spd_inv_sqrt_higham(H, n_iters=5)
    assert torch.allclose(a, b, atol=0.0)
