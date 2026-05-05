"""Equivalence test for `unwhiten_rescale_frob_batched` vs the per-pair loop
inlined in `_polar_pipeline`. Hot path of the polar-product step;
regression-protect equivalence.
"""
import pytest
import torch

from lora_playground._batched_polar import unwhiten_rescale_frob_batched


def _loop_unwhiten_rescale(P_A, P_B, SA_half_inv, SB_half_inv, u_A, u_B,
                           lr, lora_plus_mul):
    """Per-pair version, mirrors the inlined block in `_polar_pipeline`
    at polar_norm_dir='frob'."""
    N = P_A.shape[0]
    dA_l, dB_l = [], []
    for i in range(N):
        geo_A = SB_half_inv[i] @ P_A[i]
        geo_B = P_B[i] @ SA_half_inv[i]
        uA_norm = u_A[i].norm()
        uB_norm = u_B[i].norm()
        gA_norm = geo_A.norm() + 1e-30
        gB_norm = geo_B.norm() + 1e-30
        dA_l.append(-lr * (uA_norm / gA_norm) * geo_A)
        dB_l.append(-lora_plus_mul * lr * (uB_norm / gB_norm) * geo_B)
    return torch.stack(dA_l), torch.stack(dB_l)


@pytest.mark.parametrize("N,r,d_in,d_out", [
    (4, 16, 64, 64),
    (8, 16, 32, 128),     # rectangular pair (d_out > d_in)
    (8, 16, 128, 32),     # rectangular pair (d_in > d_out)
    (16, 32, 16, 16),     # tiny, fewer pairs
])
@pytest.mark.parametrize("lora_plus_mul", [1.0, 2.0])
def test_batched_matches_loop(N, r, d_in, d_out, lora_plus_mul):
    torch.manual_seed(0)
    P_A = torch.randn(N, r, d_in, dtype=torch.float32)
    P_B = torch.randn(N, d_out, r, dtype=torch.float32)
    # SA/SB half-invs are SPD r×r; use a quick symmetric construct.
    SA_seed = torch.randn(N, r, r, dtype=torch.float32)
    SB_seed = torch.randn(N, r, r, dtype=torch.float32)
    SA = 0.5 * (SA_seed + SA_seed.transpose(-2, -1))
    SB = 0.5 * (SB_seed + SB_seed.transpose(-2, -1))
    u_A = torch.randn(N, r, d_in, dtype=torch.float32)
    u_B = torch.randn(N, d_out, r, dtype=torch.float32)
    lr = 1e-3

    dA_loop, dB_loop = _loop_unwhiten_rescale(
        P_A, P_B, SA, SB, u_A, u_B, lr, lora_plus_mul)
    dA_b, dB_b = unwhiten_rescale_frob_batched(
        P_A, P_B, SA, SB, u_A, u_B, lr, lora_plus_multiplier=lora_plus_mul)
    assert dA_b.shape == dA_loop.shape
    assert dB_b.shape == dB_loop.shape
    assert torch.allclose(dA_loop, dA_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dB_loop, dB_b, atol=1e-5, rtol=1e-5)


def test_batched_lora_plus_multiplier_only_affects_dB():
    """lora_plus_multiplier scales B-side step only — dA unchanged."""
    torch.manual_seed(0)
    P_A = torch.randn(4, 16, 32)
    P_B = torch.randn(4, 32, 16)
    SA = torch.eye(16).expand(4, 16, 16).contiguous()
    SB = torch.eye(16).expand(4, 16, 16).contiguous()
    u_A = torch.randn(4, 16, 32)
    u_B = torch.randn(4, 32, 16)

    dA1, dB1 = unwhiten_rescale_frob_batched(P_A, P_B, SA, SB, u_A, u_B,
                                              1e-3, lora_plus_multiplier=1.0)
    dA2, dB2 = unwhiten_rescale_frob_batched(P_A, P_B, SA, SB, u_A, u_B,
                                              1e-3, lora_plus_multiplier=2.0)
    assert torch.allclose(dA1, dA2, atol=1e-7)
    assert torch.allclose(dB1 * 2.0, dB2, atol=1e-7)
