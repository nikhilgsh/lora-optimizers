"""Unit tests for _sigma_max_power_iter — verifies accuracy + warm-start.

Run on CPU; the function is device-agnostic and the bench script in
docs/papers/sigma_max_bench.py covers GPU timing separately.
"""
import torch
import pytest
import lora_playground.spectral as spectral
from lora_playground.optim import (
    _sigma_max_power_iter,
    _sigma_max_power_iter_batched,
)


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


def test_batched_cold_start_uses_non_null_fallback_when_ones_cancels():
    """Regression for Site-C σmax underestimation.

    The batched helper used M·1 as its deterministic start. If row/column sums
    cancel exactly, that start is zero and power iteration returns σ=0 for a
    nonzero matrix. In chord-tight-clean this makes the update scale behave
    like ρ / 1e-30, which can overflow the next Picard cross-term.
    """
    M = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    sigma, v = _sigma_max_power_iter_batched(M, n_iters=8)
    true_sigma = torch.linalg.svdvals(M)[..., 0]
    torch.testing.assert_close(sigma, true_sigma, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(v).all()


def test_batched_warm_start_recovers_when_vector_enters_nullspace():
    """A valid-shape warm start can be useless for the current matrix.

    If the cached vector is in the current smaller-side Gram nullspace, the
    first power-iteration matvec is zero. The helper must fall back per batch
    instead of returning σ=0 for a nonzero matrix.
    """
    M = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]])
    v_init = torch.tensor([[0.0, 1.0]])
    sigma, v = _sigma_max_power_iter_batched(M, v_init=v_init, n_iters=8)
    true_sigma = torch.linalg.svdvals(M)[..., 0]
    torch.testing.assert_close(sigma, true_sigma, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(v).all()


def test_batched_estimate_has_row_norm_floor_for_bad_warm_start():
    """Even before iteration, σmax(M) is at least every row (and column) norm.

    This pins the safety floor used by optimizer rescaling. A bad warm start
    can make ||Mᵀv|| much smaller than this bound; returning that underestimate
    would over-scale the update.
    """
    M = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    v_init = torch.tensor([[0.0, 1.0]])
    sigma, _ = _sigma_max_power_iter_batched(M, v_init=v_init, n_iters=0)
    assert float(sigma[0]) == 5.0


def test_batched_floor_is_two_sided_col_norm_can_dominate():
    """Known-positive for the two-sided floor: a column L2 norm is also a
    lower bound on σmax. Here the largest column norm (√2) exceeds both the
    iterated estimate (1.0, warm start aligned with a single row at n_iters=0)
    and the largest row norm (1.0). A smaller-side-only floor would return
    1.0; the two-sided floor must return √2 — the true σmax of this rank-1 M.
    """
    M = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # row norms 1, col-0 norm √2
    v_init = torch.tensor([[1.0, 0.0]])                      # ‖Mᵀv‖ = 1 at n_iters=0
    sigma, v, info = _sigma_max_power_iter_batched(
        M, v_init=v_init, n_iters=0, return_info=True,
    )
    true_sigma = float(torch.linalg.svdvals(M[0])[0])        # √2
    assert abs(float(sigma[0]) - true_sigma) < 1e-6
    assert float(info["sigma_raw"][0]) == pytest.approx(1.0)
    assert float(info["sigma_floor"][0]) == pytest.approx(true_sigma)
    assert bool(info["floor"][0])
    assert bool(info["guard_hit"][0])
    assert torch.isfinite(v).all()


def test_batched_return_info_reports_guard_hits():
    M = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    v_init = torch.tensor([[0.0, 1.0]])
    sigma, _, info = _sigma_max_power_iter_batched(
        M, v_init=v_init, n_iters=0, return_info=True,
    )
    assert float(sigma[0]) == 5.0
    assert bool(info["guard_hit"][0])
    assert bool(info["floor"][0])
    assert not bool(info["iter_fallback"][0])
    assert not bool(info["ones_fallback"][0])
    assert not bool(info["warm_start_fallback"][0])
    assert float(info["sigma_raw"][0]) == 0.0
    assert float(info["sigma_floor"][0]) == 5.0


def test_batched_guard_can_be_disabled_for_counterfactual(monkeypatch):
    monkeypatch.setattr(spectral, "_DISABLE_SIGMA_MAX_GUARD", True)
    M = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    sigma, _, info = _sigma_max_power_iter_batched(
        M, n_iters=8, return_info=True,
    )
    assert float(sigma[0]) == 0.0
    assert bool(info["guard_disabled"])
    assert not bool(info["guard_hit"][0])


def test_dtype_handling():
    """bf16 input is cast to fp32 internally; result is fp32 tensor."""
    M = torch.randn(128, 2048).to(torch.bfloat16)
    sigma, v = _sigma_max_power_iter(M, n_iters=8)
    assert sigma.dtype == torch.float32
    assert v.dtype == torch.float32
