"""Tests for the bf16 mixed-precision path of `spd_inv_sqrt_higham_batched`
(`compute_dtype=torch.bfloat16`).

Three tiers of coverage, mirroring `tests/test_ns_gram.py`:

  Equivalence — bf16 + fp32-bookend on random SPD must reproduce the
                fp32 reference output to within a few-percent
                relative elementwise error. This is the production-
                relevant question: "does bf16 give the same
                preconditioner as fp32?". An *absolute* Iannazzo-
                residual bound conflates bf16 quality with damping
                bias (under `eps_relative=ε`, the returned Z is
                `H_damped^{-1/2}`, not `H^{-1/2}` — `Z·H·Z = I -
                O(ε)·Z²` for the undamped input; fp32 sees this
                same residual). bf16-vs-fp32 elementwise is the
                clean test.

  Tier 1     — real S_A, S_B Grams from the chord-tight r=64 k=3
                snapshot (top-20 worst-conditioned per
                `scripts/build_higham_test_fixtures.py`). bf16 must
                match fp32's Z elementwise on every cell.

  Tier 3     — synthetic logspace spectrum with cond ∈ {1e2, 1e4, 1e6}.
                bf16 stays finite; bf16 elementwise rel-err vs fp32
                stays below a per-cond tolerance.

CPU-only; bf16 on CPU is sufficient for numerical validation (the
speed win is GPU-only). All tests use `eps_relative=True, eps=1e-2`
— the production setting from `scripts/sweep/sweep_chord_tight_*`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lora_playground.utils import spd_inv_sqrt_higham_batched


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "higham_real_grams.pt"
)
PROD_EPS = 1e-2
PROD_EPS_REL = True


def _relerr(A: torch.Tensor, A_ref: torch.Tensor) -> float:
    """`‖A - A_ref‖_F / ‖A_ref‖_F` — the production-relevant metric for
    "does bf16 reproduce fp32?". Uses Frobenius rather than max-abs so
    a few large-singular-value entries don't dominate."""
    return float(
        (A.float() - A_ref.float()).norm()
        / (A_ref.float().norm() + 1e-30)
    )


def _make_spd_random(N: int, r: int, log_cond: float, seed: int = 0):
    """Random SPD `H` of shape (N, r, r) with cond(H) ≈ 10**log_cond.
    Built as U·diag(σ²)·U^T where U is a random orthogonal frame and
    σ = logspace(0, -log_cond/2, r) — so the singular values of `H`
    (which equal its eigenvalues) span the requested cond range."""
    g = torch.Generator().manual_seed(seed)
    sing = torch.logspace(0, -log_cond / 2, r).pow(2)  # eigenvalues
    H_list = []
    for _ in range(N):
        M = torch.randn(r, r, generator=g)
        U, _ = torch.linalg.qr(M)
        H_list.append(U @ torch.diag(sing) @ U.transpose(-2, -1))
    return torch.stack(H_list).float()


def _run_both(H: torch.Tensor):
    """Run Higham in fp32 reference and bf16 mixed-precision; return
    `(Z_fp32, Z_bf16)`."""
    Z_fp32 = spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=PROD_EPS, eps_relative=PROD_EPS_REL,
        compute_dtype=None,
    )
    Z_bf16 = spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=PROD_EPS, eps_relative=PROD_EPS_REL,
        compute_dtype=torch.bfloat16,
    )
    return Z_fp32, Z_bf16


# ─── Equivalence: bf16+bookend vs fp32 on random SPD ────────────────────────


@pytest.mark.parametrize("log_cond,bound", [(2.0, 2e-2), (4.0, 5e-2)])
def test_equivalence_bf16_matches_fp32_on_random_spd(log_cond, bound):
    """bf16's Z must reproduce fp32's Z within a few-percent rel-err.
    bf16's 7-bit mantissa → ~1% per matmul; the 10-iter Higham body
    runs ~24 bf16 matmuls (first/last iters stay fp32), so 2–5%
    aggregate is the realistic ceiling."""
    H = _make_spd_random(N=4, r=32, log_cond=log_cond, seed=0)
    Z_fp32, Z_bf16 = _run_both(H)
    assert torch.isfinite(Z_bf16).all(), "bf16 path produced non-finite output"
    rel = _relerr(Z_bf16, Z_fp32)
    assert rel < bound, (
        f"bf16 vs fp32 rel-err {rel:.2e} exceeds {bound} at cond≈1e{int(log_cond)}"
    )


# ─── Tier 1: real chord-tight r=64 Grams ─────────────────────────────────────


def _load_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"fixture missing: {FIXTURE_PATH} — run "
            "`python scripts/build_higham_test_fixtures.py` first."
        )
    return torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)


def test_tier1_bf16_finite_on_real_grams():
    """bf16 + fp32-bookend produces finite output on every real S in
    the corpus, including the worst-conditioned cell."""
    payload = _load_fixture()
    bad = []
    for S, meta in zip(payload["S_grams"], payload["metadata"]):
        Z = spd_inv_sqrt_higham_batched(
            S.unsqueeze(0), n_iters=10, eps=PROD_EPS,
            eps_relative=PROD_EPS_REL, compute_dtype=torch.bfloat16,
        )
        if not torch.isfinite(Z).all():
            bad.append((meta, "non-finite"))
    assert not bad, f"bf16 produced non-finite output: {bad[:3]}"


def test_tier1_bf16_matches_fp32_on_real_grams():
    """Tier-1 contract: on real chord-tight r=64 Grams, bf16's Z must
    reproduce fp32's Z to within a few-percent Frobenius rel-err on
    every cell. 5% is the band consistent with bf16 mantissa drift
    across the 10-iter Higham body."""
    payload = _load_fixture()
    worst = 0.0
    worst_meta = None
    for S, meta in zip(payload["S_grams"], payload["metadata"]):
        S_b = S.unsqueeze(0)
        Z_fp32, Z_bf16 = _run_both(S_b)
        rel = _relerr(Z_bf16, Z_fp32)
        if rel > worst:
            worst = rel
            worst_meta = (meta, rel)
    assert worst < 5e-2, (
        f"bf16 vs fp32 rel-err exceeded 5% on real Gram: "
        f"worst={worst:.2e}, meta={worst_meta}"
    )


# ─── Tier 3: synthetic cond stress ──────────────────────────────────────────


@pytest.mark.parametrize("log_cond", [2.0, 4.0, 6.0])
def test_tier3_bf16_finite_at_synthetic_cond(log_cond):
    """bf16 + fp32-bookend stays finite at controlled cond up to 1e6.
    Under `eps_relative=1e-2` damping the effective cond is bounded
    at ~100 regardless of input, but the input still flows through
    the H/s scaling in low precision, so this verifies no overflow
    at the input boundary."""
    H = _make_spd_random(N=2, r=64, log_cond=log_cond, seed=int(log_cond))
    Z = spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=PROD_EPS, eps_relative=PROD_EPS_REL,
        compute_dtype=torch.bfloat16,
    )
    assert torch.isfinite(Z).all(), (
        f"bf16 produced non-finite output at cond=1e{int(log_cond)}"
    )


@pytest.mark.parametrize("log_cond,bound", [(2.0, 2e-2), (4.0, 5e-2), (6.0, 1e-1)])
def test_tier3_bf16_matches_fp32_at_synthetic_cond(log_cond, bound):
    """Tier 3 stress: bf16-vs-fp32 rel-err at increasing input cond.
    With `eps_relative=1e-2` the effective cond is capped near 100,
    but extreme-cond input still amplifies bf16 mantissa noise
    through the initial H/s scaling. Bounds: 2% at cond=1e2, 5% at
    cond=1e4, 10% at cond=1e6."""
    H = _make_spd_random(N=2, r=64, log_cond=log_cond, seed=int(log_cond) + 10)
    Z_fp32, Z_bf16 = _run_both(H)
    rel = _relerr(Z_bf16, Z_fp32)
    assert rel < bound, (
        f"bf16 vs fp32 rel-err {rel:.2e} exceeded {bound} at cond=1e{int(log_cond)}"
    )


# ─── fp16 + production eps_abs=1e-6 damping (the actual ship setting) ───────


def _run_fp16_vs_fp32_eps_abs(H, eps=1e-6):
    """Run both paths under production `eps_relative=False, eps=1e-6`
    (the train.py default — the `eps_relative=1e-2` snapshot setting
    is sweep-specific). Effective cond can reach raw_cond under
    absolute damping, so this is the more adversarial regime for the
    fp16-polish variant."""
    Z_fp32 = spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=eps, eps_relative=False, compute_dtype=None,
    )
    Z_fp16 = spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=eps, eps_relative=False,
        compute_dtype=torch.float16,
    )
    return Z_fp32, Z_fp16


@pytest.mark.parametrize("log_cond,bound", [(2.0, 5e-3), (3.0, 2e-2), (6.0, 1e-1)])
def test_fp16_polish_matches_fp32_under_production_eps_abs(log_cond, bound):
    """The production-relevant case: eps_abs=1e-6 damping (train.py
    default), variant B (fp16 inner + 1 fp32 polish). Bounds derived
    from the Blackwell bench (`scripts/bench/bench_higham_variants.py`):
    rel-err ≈ 2e-3 at cond=1e2, ~1e-2 at cond=1e3, ~3e-2 at cond=1e6
    (production worst-case is ~1e3 per the chord-tight r=64 snapshot
    audit). Bounds include 2-5× headroom over benched numbers."""
    H = _make_spd_random(N=4, r=64, log_cond=log_cond, seed=int(log_cond) + 20)
    Z_fp32, Z_fp16 = _run_fp16_vs_fp32_eps_abs(H)
    assert torch.isfinite(Z_fp16).all(), "fp16+polish produced non-finite output"
    rel = _relerr(Z_fp16, Z_fp32)
    assert rel < bound, (
        f"fp16+polish vs fp32 rel-err {rel:.2e} exceeded {bound} "
        f"at cond=1e{int(log_cond)}, eps_abs=1e-6"
    )
