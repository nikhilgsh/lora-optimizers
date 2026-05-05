"""Equivalence + property tests for `_newton_schulz_batched`.

The batched implementation is bit-near-equal to per-matrix
`_newton_schulz` and is the hot path for the polar pipeline's
across-pair speedup. Regression-protect equivalence here so
future kernel rewrites can't silently drift.
"""
import pytest
import torch

from lora_playground.optim import _newton_schulz, _newton_schulz_batched


@pytest.mark.parametrize("shape", [
    (16, 2048),    # A-side, group 1 of OLMo r=16
    (2048, 16),    # B-side, group 1
    (8192, 16),    # B-side, group 2 (large)
    (16, 8192),    # A-side, group 3 (large)
    (32, 32),      # square
    (4, 8),        # tiny non-square (wide)
    (8, 4),        # tiny non-square (tall)
])
@pytest.mark.parametrize("nsteps", [1, 3, 5])
def test_batched_matches_loop(shape, nsteps):
    """Batched output equals per-matrix loop within fp32 noise (1e-5)."""
    torch.manual_seed(0)
    N = 5
    m, n = shape
    X = torch.randn(N, m, n, dtype=torch.float32)
    Y_loop = torch.stack([_newton_schulz(X[i], nsteps=nsteps) for i in range(N)])
    Y_batched = _newton_schulz_batched(X, nsteps=nsteps)
    assert Y_batched.shape == Y_loop.shape
    assert torch.allclose(Y_loop, Y_batched, atol=1e-5, rtol=1e-5), \
        f"loop vs batched diverge: max_err={(Y_loop-Y_batched).abs().max()}"


def test_batched_dtype_promotion_to_float32():
    """Batched NS internally promotes to float32 (matches _newton_schulz)."""
    X = torch.randn(3, 16, 32, dtype=torch.bfloat16)
    Y = _newton_schulz_batched(X, nsteps=3)
    assert Y.dtype == torch.float32


def test_batched_handles_arbitrary_leading_dims():
    """Should accept (..., m, n) for arbitrary leading batch shape."""
    torch.manual_seed(0)
    X = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    Y = _newton_schulz_batched(X, nsteps=3)
    assert Y.shape == X.shape
    # Compare to (8, 16, 32)-flattened version
    Y_flat = _newton_schulz_batched(X.reshape(8, 16, 32), nsteps=3)
    assert torch.allclose(Y, Y_flat.reshape(2, 4, 16, 32), atol=1e-6)


def test_batched_orthogonality_property():
    """For non-degenerate input X (m ≥ n), output should be near-semi-orthogonal:
    Y^T Y ≈ I_n. NS converges to the polar; for tall X the polar has
    orthonormal columns. (Not bit-exact at j=5; ~1% acceptable.)"""
    torch.manual_seed(0)
    X = torch.randn(4, 64, 8, dtype=torch.float32)
    Y = _newton_schulz_batched(X, nsteps=5)
    YtY = Y.transpose(-2, -1) @ Y       # (4, 8, 8)
    I = torch.eye(8).expand(4, 8, 8)
    err = (YtY - I).abs().max()
    assert err < 0.05, f"orthogonality error too large: {err}"


@pytest.mark.parametrize("shape", [(16, 64), (64, 16), (16, 256), (256, 16)])
def test_batched_bf16_matches_fp32_orthogonality(shape):
    """bf16 NS converges to the same orthogonality residual as fp32 NS.
    The mantissa floor (~7 bits) caps residual at ~1e-3, but at j=5 fp32
    NS bottoms at ~1e-2 anyway — so bf16 is functionally identical for
    Algorithm 1's polar-direction needs.

    Validates the modded-nanogpt pattern (`train_gpt.py:187`: iterate in
    bf16) on our LoRA shape regime. Verified at the bench-script level
    by `scripts/bench_ns_bf16.py`.
    """
    torch.manual_seed(0)
    m, n = shape
    X = torch.randn(8, m, n, dtype=torch.float32)
    Y_fp32 = _newton_schulz_batched(X, nsteps=5)
    Y_bf16 = _newton_schulz_batched(X, nsteps=5, dtype=torch.bfloat16).float()
    # Both should be very close in direction; orthogonality residuals
    # differ by less than 5e-3.
    tall = m >= n
    M_f = Y_fp32.transpose(-2, -1) @ Y_fp32 if tall else Y_fp32 @ Y_fp32.transpose(-2, -1)
    M_b = Y_bf16.transpose(-2, -1) @ Y_bf16 if tall else Y_bf16 @ Y_bf16.transpose(-2, -1)
    I = torch.eye(M_f.shape[-1]).expand_as(M_f)
    err_fp32 = (M_f - I).flatten(-2).norm(dim=-1).max()
    err_bf16 = (M_b - I).flatten(-2).norm(dim=-1).max()
    # bf16 may be marginally worse (more rounding), but only by <5e-3.
    assert err_bf16 < err_fp32 + 5e-3, (
        f"bf16 NS orthogonality much worse than fp32: "
        f"fp32={err_fp32:.2e}, bf16={err_bf16:.2e}")
    # And the bf16 output is in dtype bf16 (caller decides whether to cast).
    assert _newton_schulz_batched(X, nsteps=3, dtype=torch.bfloat16).dtype == torch.bfloat16
