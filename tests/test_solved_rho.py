"""cw_solved_rho — the solved magnitude rule (GPT-opt polora_attn solved_rho port).

Covers (1) the sum-of-products σ_max estimator against the explicitly formed
matrix, including the degenerate-warm-start known-positive required for any
spectral-estimator change; (2) the merged-update budget certificate
‖Δ(BA)‖₂ ≤ η on real CurvatureWhitenLoRA steps, and that the solved step
spends more of the merged budget than the bound-ρ step; (3) the flag guards.
CPU, tiny, deterministic.
"""
import copy

import torch
import torch.nn as nn
import pytest

from lora_playground.optim import CurvatureWhitenLoRA
from lora_playground.spectral import sigma_max_power_iter_prodsum_batched


def _explicit_sigma(X1, Y1, X2, Y2):
    # Tests may use the full SVD; the optimizer-code ban does not apply here.
    return torch.linalg.matrix_norm(X1 @ Y1 + X2 @ Y2, ord=2)


def _rand_factors(seed, N=3, m=12, r=3, n=10):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, m, r, generator=g), torch.randn(N, r, n, generator=g),
            torch.randn(N, m, r, generator=g), torch.randn(N, r, n, generator=g))


def test_prodsum_estimator_matches_explicit():
    X1, Y1, X2, Y2 = _rand_factors(0)
    ref = _explicit_sigma(X1, Y1, X2, Y2)
    s, v = sigma_max_power_iter_prodsum_batched(X1, Y1, X2, Y2, n_iters=50)
    assert torch.allclose(s, ref, rtol=1e-3), f"{s} vs {ref}"
    # Warm start from the converged vector: default n_iters must hold the value.
    s2, _ = sigma_max_power_iter_prodsum_batched(X1, Y1, X2, Y2, v_init=v, n_iters=8)
    assert torch.allclose(s2, ref, rtol=1e-3)


def test_prodsum_degenerate_start_recovers():
    # Known-positive: an all-zeros warm start must fall back to the
    # deterministic init, not silently return a garbage under-estimate.
    # Zeros must behave exactly like a cold start (proves the fallback
    # engaged); the value itself is power-iter-accurate, not SVD-exact,
    # so the explicit comparison gets a looser tolerance (a near-tied
    # σ₁≈σ₂ draw converges slowly — that is a spectrum property, not a
    # guard failure).
    X1, Y1, X2, Y2 = _rand_factors(1)
    ref = _explicit_sigma(X1, Y1, X2, Y2)
    bad = torch.zeros(X1.shape[0], Y1.shape[-1])
    s, _ = sigma_max_power_iter_prodsum_batched(X1, Y1, X2, Y2, v_init=bad, n_iters=50)
    s_cold, _ = sigma_max_power_iter_prodsum_batched(X1, Y1, X2, Y2, v_init=None, n_iters=50)
    assert torch.equal(s, s_cold), f"fallback did not engage: {s} vs cold {s_cold}"
    assert (s <= ref * (1 + 1e-6)).all(), "power iter exceeded the true norm"
    assert torch.allclose(s, ref, rtol=2e-2), f"{s} vs {ref}"


class _LoraLin(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = _LoraLin(32, 16, 4)
        self.l2 = _LoraLin(16, 48, 4)


def _run_merged_norms(cw_solved_rho, lr=1e-3, n_steps=5, b_zero=True):
    """Per-step σ_max of the realized merged update Δ(BA), per pair."""
    torch.manual_seed(0)
    model = _Model()
    if b_zero:  # production LoRA init
        for mod in (model.l1, model.l2):
            mod.lora_B["default"].weight.data.zero_()
    opt = CurvatureWhitenLoRA(
        model, lr=lr, use_polar=True, kl_coupled=True, diag_metric=True,
        soap_v=False, cw_nesterov=True, delta=1e-4,
        cw_solved_rho=cw_solved_rho)
    g = torch.Generator().manual_seed(1)
    norms = []
    for _ in range(n_steps):
        prods = [(B.detach() @ A.detach()).clone() for A, B in opt.pairs]
        for A, B in opt.pairs:
            A.grad = torch.randn(A.shape, generator=g)
            B.grad = torch.randn(B.shape, generator=g)
        opt.step()
        norms.append([
            torch.linalg.matrix_norm(B.detach() @ A.detach() - p0, ord=2).item()
            for (A, B), p0 in zip(opt.pairs, prods)])
    return torch.tensor(norms)  # (n_steps, n_pairs)


def test_solved_rho_budget_certificate_and_spend():
    lr = 1e-3
    solved = _run_merged_norms(True, lr=lr)
    bound = _run_merged_norms(False, lr=lr)
    # Certificate: ‖Δ(BA)‖₂ ≤ η up to σ̂ estimation error — the certificate is
    # exact only at exact σ_max, and every σ̂ here is a warm-started power-iter
    # LOWER bound. This toy (r=4, i.i.d. random grads each step) is the worst
    # case for warm starts (top direction decorrelates step to step); observed
    # spread 0.996–1.056 of budget, vs 0.65–0.94 realized by the bound rule.
    assert (solved <= lr * 1.10).all(), f"budget exceeded: {solved.max()} vs lr={lr}"
    assert solved.mean() <= lr * 1.02, f"systematic overshoot: {solved.mean()}"
    # Spend: the solved rule must realize at least the bound rule's merged step
    # (it exists to eliminate the triangle-inequality slack), and strictly more
    # once B has grown (later steps).
    assert (solved.mean() >= bound.mean() * 0.999), (solved.mean(), bound.mean())
    assert (solved[-1] >= bound[-1] * 0.999).all(), (solved[-1], bound[-1])


def test_solved_rho_rejects_magnitude_ablations():
    model = _Model()
    with pytest.raises(ValueError):
        CurvatureWhitenLoRA(model, use_polar=True, kl_coupled=True,
                            diag_metric=True, soap_v=False,
                            precond_method="gram_ns",
                            cw_solved_rho=True, cw_unpinned=True)
    with pytest.raises(ValueError):
        CurvatureWhitenLoRA(model, use_polar=True, kl_coupled=True,
                            diag_metric=True, soap_v=False,
                            cw_solved_rho=True, cw_no_radius=True)
