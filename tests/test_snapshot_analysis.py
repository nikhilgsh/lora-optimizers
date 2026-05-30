"""CPU-only smoke tests for `lora_playground.snapshot_analysis`.

These don't touch real `/mnt/ceph` snapshots — they synthesize a minimal
optimizer state-dict in-memory so the package's small helpers are exercised
without needing the cluster filesystem.
"""
from __future__ import annotations

import math

import pytest
import torch

from lora_playground.snapshot_analysis import (
    Mtilde,
    _polar_uvt,
    _prerescale_unit_op,
    clear_snapshot_cache,
    load_snapshot,
    newton_schulz_polar,
    normalized_sigmas,
    normalized_sigmas_x,
    spd_half_inv,
    stable_rank,
    whitened_NS_input,
)
from lora_playground.snapshot_analysis.moments import BETA1, BETA2, EPS
from lora_playground.snapshot_analysis.calibration import (
    _agreement_kappa_for_pair,
    _chord_residual_norm_sq,
    _linear_residual_norm_sq,
    _simulate_chord_tight_ssc_update,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_pair() -> dict:
    """Synthetic LoRA pair with non-trivial moments at step t=50."""
    torch.manual_seed(0)
    r, d_in, d_out = 4, 8, 6
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)
    m_A = torch.randn(r, d_in) * 0.1
    v_A = torch.randn(r, d_in).pow(2) * 1e-3 + 1e-5
    m_B = torch.randn(d_out, r) * 0.1
    v_B = torch.randn(d_out, r).pow(2) * 1e-3 + 1e-5
    t = 50
    # u_A as produced by an Adam step with bias correction.
    m_A_bc = m_A / (1 - BETA1 ** t)
    v_A_bc = v_A / (1 - BETA2 ** t)
    u_A = m_A_bc / (v_A_bc.sqrt() + EPS)
    m_B_bc = m_B / (1 - BETA1 ** t)
    v_B_bc = v_B / (1 - BETA2 ** t)
    u_B = m_B_bc / (v_B_bc.sqrt() + EPS)
    return {
        'A': A, 'B': B,
        'm_A': m_A, 'v_A': v_A,
        'm_B': m_B, 'v_B': v_B,
        'u_A': u_A, 'u_B': u_B,
        'step': t,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_Mtilde_matches_u_A(fake_pair):
    """Mtilde(bias_correct=True) reconstructs the stored u_A."""
    Mt = Mtilde(fake_pair, 'A', bias_correct=True)
    drift = (Mt - fake_pair['u_A']).abs().max().item()
    assert drift < 1e-6, f'Mtilde drift {drift:.2e} too large'


def test_Mtilde_no_bc_differs(fake_pair):
    """Mtilde(bias_correct=False) ≠ u_A (paper's M̃ definition)."""
    Mt = Mtilde(fake_pair, 'A', bias_correct=False)
    diff = (Mt - fake_pair['u_A']).abs().max().item()
    assert diff > 1e-3, 'no-bc and bc Mtilde should differ noticeably'


def test_normalized_sigmas_unit_norm(fake_pair):
    s = normalized_sigmas(fake_pair['m_A'])
    assert math.isclose(float((s ** 2).sum()), 1.0, abs_tol=1e-6)


def test_normalized_sigmas_x_shape(fake_pair):
    x, s = normalized_sigmas_x(fake_pair['m_A'])
    assert x.shape == s.shape
    assert x[0] == 0.0 and abs(x[-1] - 1.0) < 1e-9


def test_stable_rank_identity():
    """stable_rank(M) = ‖M‖_F² / σ_max². Compare to direct computation."""
    torch.manual_seed(1)
    M = torch.randn(8, 12)
    s = torch.linalg.svdvals(M)
    expected = (s.pow(2).sum() / s[0].pow(2)).item()
    assert math.isclose(stable_rank(M), expected, rel_tol=1e-5)


def test_stable_rank_orthogonal_eq_rank():
    """Stable rank of an orthogonal projector equals its rank."""
    torch.manual_seed(2)
    Q, _ = torch.linalg.qr(torch.randn(10, 6))  # (10, 6) with orthonormal columns
    sr = stable_rank(Q)
    assert math.isclose(sr, 6.0, rel_tol=1e-4)


def test_spd_half_inv_is_inverse_sqrt():
    """spd_half_inv(S) @ S @ spd_half_inv(S) ≈ I within damping budget."""
    torch.manual_seed(3)
    X = torch.randn(7, 12)
    S = X @ X.T
    Z = spd_half_inv(S, delta_abs=1e-8)
    I_approx = Z @ S @ Z
    err = (I_approx - torch.eye(7)).norm().item()
    assert err < 1e-4, f'‖Z S Z − I‖ = {err:.2e}'


def test_whitened_NS_input_shape(fake_pair):
    X = whitened_NS_input(fake_pair, 'A')
    r, d_in = fake_pair['A'].shape
    assert X.shape == (r, d_in)
    assert X.dtype == torch.float32


def test_prerescale_unit_op_sigma_max_is_one():
    torch.manual_seed(4)
    X = torch.randn(8, 12) * 3.7
    Y = _prerescale_unit_op(X)
    s_max = torch.linalg.matrix_norm(Y, ord=2).item()
    assert math.isclose(s_max, 1.0, abs_tol=1e-6)


def test_polar_uvt_is_orthogonal():
    torch.manual_seed(5)
    X = torch.randn(10, 6)
    P = _polar_uvt(X)
    # P has orthonormal columns ⇒ P^T P = I_6
    err = (P.T @ P - torch.eye(6)).norm().item()
    assert err < 1e-5


def test_newton_schulz_polar_singular_values_approach_one():
    """After enough NS steps, σ_i should be near 1 for full-rank input."""
    torch.manual_seed(6)
    X = torch.randn(8, 12)
    Y = newton_schulz_polar(X, nsteps=10)
    s = torch.linalg.svdvals(Y)
    # All singular values should be in [0, ~1] and the max close to 1.
    assert s.max().item() <= 1.001
    assert s.max().item() > 0.99


def test_load_snapshot_cache_returns_same_object():
    """Two calls with the same (step, root) return the cached dict."""
    pytest.importorskip('torch')
    from pathlib import Path
    from lora_playground.snapshot_analysis import SNAP_ROOT, RUNS
    if not RUNS:
        pytest.skip('no snapshot dirs present')
    clear_snapshot_cache()
    try:
        a = load_snapshot(2000, root=SNAP_ROOT)
    except FileNotFoundError:
        pytest.skip('step 2000 snapshot not on disk')
    b = load_snapshot(2000, root=SNAP_ROOT)
    assert a is b, 'LRU cache should return the same dict object'
    clear_snapshot_cache()


def test_local_objective_residual_gram_identities():
    """Snapshot c-sweeps score residuals without materializing full layers."""
    torch.manual_seed(7)
    r, d_in, d_out = 5, 11, 13
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)
    dA = torch.randn(r, d_in) * 0.03
    dB = torch.randn(d_out, r) * 0.03

    dense_linear = B @ dA + dB @ A
    dense_chord = dense_linear + dB @ dA

    assert torch.allclose(
        _linear_residual_norm_sq(A, B, dA, dB),
        dense_linear.square().sum(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        _chord_residual_norm_sq(A, B, dA, dB),
        dense_chord.square().sum(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_simulate_chord_tight_ssc_update_returns_finite_metrics(fake_pair):
    out = _simulate_chord_tight_ssc_update(
        fake_pair, lr=3e-2, c=0.3, picard_iters=2, delta_abs=1e-6,
    )
    for key in (
        'obj_scaled_tangent', 'obj_scaled_chord',
        'obj_raw_tangent', 'obj_raw_chord',
        'sr_XA_eff', 'sr_XB_eff', 'rho',
    ):
        assert math.isfinite(out[key]), key


def test_agreement_kappa_for_pair_returns_finite_metrics(fake_pair):
    out = _agreement_kappa_for_pair(
        fake_pair,
        lr="3e-2",
        lora_r=4,
        ns=5,
        variant="unit-test",
        step=50,
        pair_index=0,
        delta_abs=1e-6,
        device=None,
    )
    for key in (
        'q_agree', 'snr_op',
        'kappa_agree_opfloor', 'kappa_agree_meanfloor',
        'kappa_input_A', 'kappa_input_B',
        'kappa_target_opfloor_A', 'kappa_target_opfloor_B',
        'c_opfloor_A', 'c_opfloor_B',
    ):
        assert math.isfinite(out[key]), key
    assert 0.0 <= out['q_agree'] <= 1.0
    assert out['kappa_target_opfloor_A'] >= out['kappa_input_A']
    assert out['kappa_target_opfloor_B'] >= out['kappa_input_B']
