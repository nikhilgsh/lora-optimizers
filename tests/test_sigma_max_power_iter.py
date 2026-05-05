"""Unit tests for _sigma_max_power_iter — verifies accuracy + warm-start.

Run on CPU; the function is device-agnostic and the bench script in
docs/papers/sigma_max_bench.py covers GPU timing separately.
"""
import torch
import pytest
from lora_playground.optim import _sigma_max_power_iter


def _ground_truth(M):
    return float(torch.linalg.matrix_norm(M, ord=2))


@pytest.mark.parametrize("shape", [(128, 2048), (4096, 128), (16, 2048),
                                    (256, 4096), (32, 32)])
def test_cold_start_accuracy(shape):
    """Cold-start (no warm vector) at n_iters=16 should be within 8% p95
    on Gaussian random matrices. Power iter on Gaussian Gram converges as
    (σ_2/σ_1)^(2k) which is slow because σ_2 ≈ 0.95·σ_1; tight bounds
    require warm-starting (test below) or higher n_iters."""
    torch.manual_seed(0)
    errs = []
    for _ in range(20):
        M = torch.randn(*shape)
        true_sigma = _ground_truth(M)
        sigma_t, v = _sigma_max_power_iter(M, n_iters=16)
        est = float(sigma_t)
        errs.append(abs(est - true_sigma) / true_sigma)
    p95 = sorted(errs)[int(0.95 * len(errs))]
    assert p95 < 0.08, f"p95 rel-err {p95:.4f} > 0.08 at shape={shape}"


@pytest.mark.parametrize("shape", [(128, 2048), (4096, 128), (16, 2048)])
def test_warm_start_beats_cold_start_at_same_n_iters(shape):
    """The claim: with v_init carried across calls under small-perturbation
    M-updates, n_iters=3 with warm-start should be MORE accurate than
    n_iters=3 cold-start (where each cold call resamples random init).
    This is the property the optimizer relies on — top σ-vector barely
    moves between steps, so the cached vector accelerates convergence."""
    torch.manual_seed(1)
    M = torch.randn(*shape)
    # Cold pass (high iters) → warm vector seeded near the truth.
    _, v_warm = _sigma_max_power_iter(M, n_iters=24)

    eta = 1e-3
    n_steps = 50
    warm_errs = []
    cold_errs = []
    for _ in range(n_steps):
        # Perturb M by O(η)-scale noise (LoRA-step-like update magnitude).
        M = M + eta * torch.randn_like(M) * float(torch.linalg.matrix_norm(M, ord='fro'))
        true_sigma = _ground_truth(M)
        sigma_warm, v_warm = _sigma_max_power_iter(M, v_init=v_warm, n_iters=3)
        sigma_cold, _ = _sigma_max_power_iter(M, n_iters=3)
        warm_errs.append(abs(float(sigma_warm) - true_sigma) / true_sigma)
        cold_errs.append(abs(float(sigma_cold) - true_sigma) / true_sigma)
    warm_p95 = sorted(warm_errs)[int(0.95 * len(warm_errs))]
    cold_p95 = sorted(cold_errs)[int(0.95 * len(cold_errs))]
    assert warm_p95 < cold_p95, (
        f"warm-start k=3 p95 {warm_p95:.4f} not better than cold-start "
        f"k=3 p95 {cold_p95:.4f} at shape={shape}"
    )
    # Warm-start should be in the 5-10% band (loose-but-useful range).
    assert warm_p95 < 0.10, f"warm-start p95 {warm_p95:.4f} too loose at shape={shape}"


def test_returns_gpu_tensor_no_sync():
    """Returned σ should be a 0-dim tensor on the same device as M (no
    GPU→CPU sync inside the function)."""
    M = torch.randn(128, 2048)
    sigma, v = _sigma_max_power_iter(M, n_iters=8)
    assert isinstance(sigma, torch.Tensor)
    assert sigma.dim() == 0  # scalar
    assert sigma.device == M.device
    assert v.shape == (128,)
    assert v.device == M.device


def test_warm_start_vector_shape_and_persistence():
    """Returned v can be passed back as v_init on the next call.
    Warm-started call refines the estimate further (closer to truth, which
    power iter approaches from below)."""
    torch.manual_seed(0)
    M = torch.randn(4096, 128)
    true_sigma = _ground_truth(M)
    sigma1, v = _sigma_max_power_iter(M, n_iters=8)
    # Same M, warm-started: should be ≥ sigma1 (closer to truth from below)
    # AND closer to truth than the cold start.
    sigma2, v = _sigma_max_power_iter(M, v_init=v, n_iters=3)
    assert float(sigma2) >= float(sigma1) - 1e-5
    err1 = abs(float(sigma1) - true_sigma) / true_sigma
    err2 = abs(float(sigma2) - true_sigma) / true_sigma
    assert err2 <= err1, (
        f"warm-start n=3 (err {err2:.4f}) should be at least as accurate "
        f"as cold n=8 (err {err1:.4f})"
    )


def test_empty_matrix():
    M = torch.zeros(0, 5)
    sigma, v = _sigma_max_power_iter(M, n_iters=8)
    assert float(sigma) == 0.0


def test_zero_matrix():
    M = torch.zeros(128, 2048)
    sigma, v = _sigma_max_power_iter(M, n_iters=8)
    assert float(sigma) == 0.0


def test_dtype_handling():
    """bf16 input is cast to fp32 internally; result is fp32 tensor."""
    M = torch.randn(128, 2048).to(torch.bfloat16)
    sigma, v = _sigma_max_power_iter(M, n_iters=8)
    assert sigma.dtype == torch.float32
    assert v.dtype == torch.float32
