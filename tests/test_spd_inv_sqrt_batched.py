"""Equivalence tests for `spd_inv_sqrt_higham_batched` vs the per-matrix
loop. The batched version is the hot path for the polar-product
optimizer's `precond_refresh` step; regression-protect equivalence here.
"""
import pytest
import torch

from lora_playground.utils import (
    spd_inv_sqrt_higham,
    spd_inv_sqrt_higham_batched,
)


def _spd_batch(N, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(N, n, 4 * n, generator=g, dtype=torch.float32)
    return M @ M.transpose(-2, -1) / (4 * n)


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("n_iters", [5, 10])
def test_batched_matches_loop(n, n_iters):
    """At converged iter counts, batched higham equals per-matrix higham
    within the iteration's residual band. Power iteration uses random init
    per matrix so loop and batched see different `v`s; the equivalence
    target is "both converged to the same fixed point", not bit-exact."""
    torch.manual_seed(0)
    H = _spd_batch(N=8, n=n)
    Z_loop = torch.stack([
        spd_inv_sqrt_higham(H[i], n_iters=n_iters) for i in range(H.shape[0])
    ])
    Z_batched = spd_inv_sqrt_higham_batched(H, n_iters=n_iters)
    err = (Z_loop - Z_batched).abs().max().item()
    rel = err / (Z_loop.abs().max().item() + 1e-30)
    # n_iters=5 saturates at ~1e-3 error vs the true H^{-1/2}; both
    # implementations sit in that band.
    assert rel < 5e-3, f"batched/loop relative error {rel} too large"


def test_batched_recovers_inverse_square_root():
    """Z @ H @ Z ≈ I (the actual property we want)."""
    H = _spd_batch(N=8, n=32)
    Z = spd_inv_sqrt_higham_batched(H, n_iters=10)
    ZHZ = Z @ H @ Z
    I = torch.eye(32).expand_as(ZHZ)
    rel = (ZHZ - I).norm() / I.norm()
    assert rel < 5e-4, f"Z H Z deviates from I: rel={rel}"


def test_batched_handles_arbitrary_leading_dims():
    """Should accept (..., n, n)."""
    torch.manual_seed(0)
    H = _spd_batch(N=12, n=16).reshape(3, 4, 16, 16)
    Z = spd_inv_sqrt_higham_batched(H, n_iters=5)
    assert Z.shape == H.shape
