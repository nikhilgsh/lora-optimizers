"""Tests for `_newton_schulz_gram_batched` (Dao 2026 Stabilized Gram NS).

Three tiers of coverage:

  Tier 1 — real chord-tight r=64 X_eff matrices loaded from the fixture
           at tests/fixtures/gram_ns_real_inputs.pt (built by
           scripts/build_gram_ns_test_fixtures.py). The audit at fixture
           build time shows ~0.9% of real Grams have cond(G) > 1e4 —
           exactly the regime that broke spd_power_batched in TF32.

  Tier 3 — synthetic stress with cond(G) = 1e4 by construction
           (logspace singular spectrum), matching Dao's failure case.

  Equivalence — fp32-mode gram NS matches fp32 rectangular NS to fp32
           noise floor (~1e-5). Both should be bit-equivalent modulo
           floating-point rounding because they implement the same
           polynomial.

CPU-only; the fixture is small (top-20 worst-conditioned matrices).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lora_playground.optim import (
    _newton_schulz_batched,
    _newton_schulz_gram_batched,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "gram_ns_real_inputs.pt"


def _max_rel_err(Y: torch.Tensor, Y_ref: torch.Tensor) -> float:
    """max |Y - Y_ref| / max |Y_ref|. Tests "gram approximates rect" — the
    only correctness criterion that matters per project convention. NS=5 is
    deliberately sub-orthogonalizing; "orthogonality residual" is NOT a
    meaningful target."""
    err = (Y - Y_ref).abs().max().item()
    scale = Y_ref.abs().max().item() + 1e-30
    return err / scale


# ─── Equivalence: fp32 gram vs fp32 rect ──────────────────────────────────────


@pytest.mark.parametrize("shape", [(4, 16, 2048), (4, 2048, 16), (2, 32, 8192)])
def test_gram_fp32_matches_rect_fp32(shape):
    """fp32 + no restart should produce the same polynomial as rect NS in fp32.

    Tolerance is the fp32 noise floor for a 5-iter NS composition (~1e-5);
    both paths apply the same scalar polynomial to each singular value, so
    differences are pure fpu rounding.
    """
    torch.manual_seed(0)
    X = torch.randn(*shape)
    Y_rect = _newton_schulz_batched(X, nsteps=5, dtype=None)
    Y_gram = _newton_schulz_gram_batched(
        X, nsteps=5, dtype=torch.float32, restart_at=None, safety_factor=1.0,
    )
    err = (Y_rect - Y_gram).abs().max().item()
    rel = err / (Y_rect.abs().max().item() + 1e-30)
    assert err < 1e-4, f"fp32 gram ↔ rect diverge: max_abs={err:.2e}, rel={rel:.2e}"


def test_gram_fp32_matches_rect_on_random_batch():
    """fp32 path on a random batch matches rect fp32 to fp32 noise on every
    pair in the batch (the equivalence contract)."""
    torch.manual_seed(42)
    X = torch.randn(8, 32, 2048)
    Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
    Y = _newton_schulz_gram_batched(
        X, nsteps=5, dtype=torch.float32, restart_at=None, safety_factor=1.0,
    )
    assert _max_rel_err(Y, Y_ref) < 1e-5


# ─── Tier 1: real chord-tight r=64 X_eff matrices ─────────────────────────────


def _load_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"fixture missing: {FIXTURE_PATH} — run "
            "`python scripts/build_gram_ns_test_fixtures.py` first."
        )
    return torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)


def test_tier1_fp16_restart_finite_on_real_inputs():
    """fp16 + restart at τ=2 produces finite output on every real X_eff in
    the corpus, including the worst-conditioned (cond(G) ≈ 1.4e5)."""
    payload = _load_fixture()
    bad = []
    for X, meta in zip(payload["X_effs"], payload["metadata"]):
        Y = _newton_schulz_gram_batched(
            X, nsteps=5, dtype=torch.float16, restart_at=2, safety_factor=1.05,
        )
        if not torch.isfinite(Y).all():
            bad.append((meta, "non-finite"))
    assert not bad, f"fp16+restart produced non-finite output: {bad[:3]}"


def test_tier1_fp16_restart_matches_fp32_within_polar_tolerance():
    """fp16 + restart on real X_eff matches fp32 rect within Dao's polar
    tolerance — NS residual well below the 0.01 val-ppl threshold the paper
    reports as the quality bound."""
    payload = _load_fixture()
    diffs = []
    for X, meta in zip(payload["X_effs"], payload["metadata"]):
        Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
        Y_test = _newton_schulz_gram_batched(
            X, nsteps=5, dtype=torch.float16, restart_at=2, safety_factor=1.05,
        )
        err = (Y_ref - Y_test).abs().max().item()
        rel = err / (Y_ref.abs().max().item() + 1e-30)
        diffs.append((rel, meta["cond_G"], meta))
    worst = max(diffs, key=lambda x: x[0])
    # Loose bound — fp16 noise floor in a 5-iter NS composition on a r=64
    # matrix with cond up to 1.4e5 is realistically ~1e-2 rel-err.
    assert worst[0] < 5e-2, (
        f"fp16+restart drift too large on real input: rel_err={worst[0]:.2e} "
        f"on cond_G={worst[1]:.2e}, meta={worst[2]}"
    )


def test_tier1_fp32_safety_matches_rect_fp32():
    """Safety-mode (fp32 + TF32-off) gram matches rect fp32 to fp32 noise."""
    payload = _load_fixture()
    worst = 0.0
    for X, meta in zip(payload["X_effs"], payload["metadata"]):
        Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
        Y_test = _newton_schulz_gram_batched(
            X, nsteps=5, dtype=torch.float32, restart_at=None, safety_factor=1.0,
        )
        err = (Y_ref - Y_test).abs().max().item()
        worst = max(worst, err)
    assert worst < 1e-4, f"fp32 gram drift from fp32 rect: max_abs={worst:.2e}"


# ─── Tier 3: synthetic cond=1e4 stress ────────────────────────────────────────


def _make_cond_stress(r: int, d: int, log_decades: float, seed: int = 0) -> torch.Tensor:
    """Build X with cond(X) = 10**log_decades, controlled singular spectrum."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(r, r, generator=g))
    V, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    sing = torch.logspace(0, -log_decades, r)
    return U @ torch.diag(sing) @ V.transpose(-2, -1)


def test_tier3_fp16_restart_handles_cond_1e4():
    """fp16 + restart at τ=2 stays finite on cond=1e4 input and tracks rect-fp32."""
    X = _make_cond_stress(r=64, d=2048, log_decades=4.0)
    Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
    Y = _newton_schulz_gram_batched(
        X, nsteps=5, dtype=torch.float16, restart_at=2, safety_factor=1.05,
    )
    assert torch.isfinite(Y).all(), "fp16+restart non-finite on cond=1e4"
    assert _max_rel_err(Y, Y_ref) < 5e-2, "fp16+restart drift from rect-fp32"


def test_tier3_fp32_handles_cond_1e4():
    """fp32 + TF32-off stays finite and tight on cond=1e4 input."""
    X = _make_cond_stress(r=64, d=2048, log_decades=4.0)
    Y = _newton_schulz_gram_batched(
        X, nsteps=5, dtype=torch.float32, restart_at=None, safety_factor=1.0,
    )
    assert torch.isfinite(Y).all(), "fp32 produced non-finite on cond=1e4"
    Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
    err = (Y - Y_ref).abs().max().item()
    assert err < 1e-4, f"fp32 gram drift from fp32 rect on cond=1e4: {err:.2e}"


def test_tier3_fp16_no_restart_still_tracks_on_cond_1e4_cubic_NS5():
    """Note for future: with cubic Muon (a=1.5, b=-0.5) at NS=5, even fp16
    WITHOUT restart converges on cond=1e4 inputs. Dao's instability story
    (spurious negatives blowing up exponentially) is for Polar Express
    coefficients (15/8, -10/8, 3/8) where h(0)=15/8≈1.875 gives ~3.5x
    per-iter compounding; cubic Muon has h(0)=1.5, giving ~2.25x — much
    slower. At T=5 the negative-eigenvalue blowup doesn't reach the basin.
    Restart is still enabled by default as a hedge for: (a) cond values
    higher than we've measured, (b) future swaps to Polar Express, (c)
    Dao's explicit recommendation. This test documents the current regime
    so future readers know NOT to remove restart based on this single-case
    evidence."""
    X = _make_cond_stress(r=64, d=2048, log_decades=4.0)
    Y_ref = _newton_schulz_batched(X, nsteps=5, dtype=None)
    Y = _newton_schulz_gram_batched(
        X, nsteps=5, dtype=torch.float16, restart_at=None, safety_factor=1.05,
    )
    assert torch.isfinite(Y).all()
    # Loose bound — verifies the iteration is at least functional w/o restart.
    rel = _max_rel_err(Y, Y_ref)
    assert rel < 1e-1, (
        f"naive fp16 gram NS unexpectedly large drift {rel:.2e} — investigate "
        "before treating restart as optional."
    )
