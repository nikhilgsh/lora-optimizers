"""Tests for the `pre_norm` kwarg on polar primitives.

Backstory: `_chord_tight_clean_polar_pipeline` does a §2.5 spec-norm so
σ_max(X) = 1 going into the polar map. The polar primitives historically
applied an internal Frobenius pre-norm regardless, shrinking σ_max further
to 1/(safety_factor·√(stable_rank)) and leaving 5-iter Schulz incomplete
(whitening_fraction ≈ 0.72 instead of ≈ 1.0). The fix added a `pre_norm`
kwarg with values {"frob","spec","none"}; the chord-tight-clean path now
passes "none". These tests pin the three behaviors.
"""
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import (
    _newton_schulz,
    _newton_schulz_batched,
    _newton_schulz_gram_batched,
    _polar_express_gram_batched,
)


def _power_law_X(r=64, d=512, alpha=0.32, seed=0):
    """Matrix with σ_max=1, power-law spectrum σ_i = i^(-alpha) (after rescale).
    For alpha=0.32, stable_rank ≈ 10 — matches measured production training."""
    torch.manual_seed(seed)
    i = torch.arange(1, r + 1, dtype=torch.float32)
    sigma = i.pow(-alpha)
    sigma = sigma / sigma[0]  # σ_max = 1
    U, _ = torch.linalg.qr(torch.randn(d, r))
    V, _ = torch.linalg.qr(torch.randn(r, r))
    return ((U[:, :r] * sigma) @ V.T).T  # (r, d)


def _whitening_fraction(P):
    """‖P‖_F / σ_max(P) / √min(r,d). True polar = 1.0; identity ≈ stable_rank/√r."""
    smax = torch.linalg.matrix_norm(P, ord=2).item()
    fnorm = P.norm().item()
    r_eff = float(min(P.shape[-2], P.shape[-1]))
    return fnorm / (smax * (r_eff ** 0.5) + 1e-30)


@pytest.mark.parametrize("fn", [_newton_schulz_gram_batched, _polar_express_gram_batched])
def test_pre_norm_none_more_complete_than_frob(fn):
    """On a spec-normed input (σ_max=1), pre_norm='none' produces a more
    Muon-faithful polar than pre_norm='frob' (the legacy behavior). Whitening
    fraction is the load-bearing signal: 'none' should be significantly closer
    to 1.0 (true polar) than 'frob' on a power-law spectrum."""
    X = _power_law_X(r=64, d=512).unsqueeze(0).cuda() if torch.cuda.is_available() else _power_law_X(r=64, d=512).unsqueeze(0)
    P_frob = fn(X.clone(), nsteps=5, pre_norm="frob").float()
    P_none = fn(X.clone(), nsteps=5, pre_norm="none").float()
    wf_frob = _whitening_fraction(P_frob[0])
    wf_none = _whitening_fraction(P_none[0])
    assert wf_none > wf_frob, (
        f"pre_norm='none' should be more complete than 'frob' on spec-normed input; "
        f"got wf_none={wf_none:.3f}, wf_frob={wf_frob:.3f}"
    )
    # Strict relative inequality holds for both NS and PE. Absolute gap differs:
    # NS-5 frob ≈ 0.72, NS-5 none ≈ 0.95 (large gap; explains the 9000-step
    # leaderboard loss shift). PE-5 frob ≈ 0.87, PE-5 none ≈ 0.89 (smaller
    # gap; PE's quintic polynomial is less sensitive to the Frob shrinkage
    # than the cubic Muon polynomial). For correctness of the fix, only the
    # inequality is load-bearing.
    assert wf_none > 0.85, f"pre_norm='none' should approach true polar: {wf_none:.3f}"


def test_pre_norm_spec_matches_none_when_already_spec_normed():
    """If input already has σ_max=1, pre_norm='spec' (which divides by σ_max)
    is a no-op modulo safety_factor — should match pre_norm='none' bit-for-bit
    (up to fp32 noise from the power-iter estimation)."""
    X = _power_law_X(r=64, d=512).unsqueeze(0)
    P_spec = _newton_schulz_gram_batched(X.clone(), nsteps=5, pre_norm="spec").float()
    P_none = _newton_schulz_gram_batched(X.clone(), nsteps=5, pre_norm="none").float()
    # Power-iter sigma_max is approximate; allow loose tolerance.
    diff = (P_spec - P_none).norm() / P_none.norm()
    assert diff < 5e-3, f"'spec' and 'none' should match on σ_max=1 input: rel diff {diff:.4e}"


def test_pre_norm_frob_is_scale_invariant():
    """Frob pre-norm: P(α·X) == P(X) for any α>0 (the whole point of Muon)."""
    X = _power_law_X(r=64, d=512).unsqueeze(0)
    P1 = _newton_schulz_gram_batched(X.clone(), nsteps=5, pre_norm="frob").float()
    P2 = _newton_schulz_gram_batched((X.clone() * 0.3), nsteps=5, pre_norm="frob").float()
    diff = (P1 - P2).norm() / P1.norm()
    assert diff < 5e-4, f"frob should be scale-invariant: rel diff {diff:.4e}"


def test_pre_norm_invalid_raises():
    X = _power_law_X(r=8, d=32).unsqueeze(0)
    with pytest.raises(ValueError, match="pre_norm"):
        _newton_schulz_gram_batched(X, nsteps=5, pre_norm="bogus")
    with pytest.raises(ValueError, match="pre_norm"):
        _polar_express_gram_batched(X, nsteps=5, pre_norm="bogus")
    with pytest.raises(ValueError, match="pre_norm"):
        _newton_schulz_batched(X, nsteps=5, pre_norm="bogus")
    with pytest.raises(ValueError, match="pre_norm"):
        _newton_schulz(X[0], nsteps=5, pre_norm="bogus")


def test_pre_norm_default_is_frob_backward_compatible():
    """The default kwarg value must remain 'frob' so existing callers
    (MuonLoRA, chord-tight non-clean, etc.) get identical behavior."""
    X = _power_law_X(r=64, d=512).unsqueeze(0)
    P_default = _newton_schulz_gram_batched(X.clone(), nsteps=5).float()
    P_frob = _newton_schulz_gram_batched(X.clone(), nsteps=5, pre_norm="frob").float()
    diff = (P_default - P_frob).norm() / P_frob.norm()
    assert diff < 1e-6, f"default must equal pre_norm='frob': rel diff {diff:.4e}"
