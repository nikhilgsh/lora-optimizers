"""CPU tests for lora_playground.lmo_diagnostics (approximate-LMO rho scores)."""

import torch

from lora_playground.lmo_diagnostics import (
    eps_polar,
    gram_singular_values,
    lmo_scores,
    nuclear_norm,
    rho,
    t_fisher_racs,
    t_frob,
    t_msign,
    t_polar_k,
    t_reg_alg1,
    t_reg_oneside,
)


def _rand(r, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(r, d, generator=g, dtype=torch.float64)


def test_nuclear_norm_matches_svdvals():
    H = _rand(6, 40)
    assert torch.allclose(nuclear_norm(H), torch.linalg.svdvals(H).sum(), atol=1e-9)


def test_gram_singular_values_match_svdvals_short_side():
    H = _rand(6, 40)
    sv = gram_singular_values(H)
    assert torch.allclose(sv, torch.linalg.svdvals(H), atol=1e-9)
    # Transpose-invariant: the tall orientation gives the same spectrum.
    assert torch.allclose(gram_singular_values(H.T), sv, atol=1e-9)


def test_msign_attains_rho_one():
    """rho == 1 exactly for the exact polar — the calibration of the whole scale."""
    for seed in range(3):
        H = _rand(8, 50, seed)
        assert abs(rho(H, t_msign(H)).item() - 1.0) < 1e-10


def test_rho_is_scale_invariant_in_the_operator_output():
    """A magnitude rule downstream of T cannot move rho."""
    H = _rand(8, 50)
    Z = t_reg_oneside(H)
    base = rho(H, Z).item()
    for c in (1e-6, 3.0, 1e6):
        assert abs(rho(H, Z * c).item() - base) < 1e-10


def test_row_normalization_is_exact_iff_rows_orthogonal():
    """H = D Z with Z row-orthonormal => diag(HH^T)^{-1/2} H = Z = msign(H)."""
    Z = torch.linalg.qr(_rand(60, 6).T.contiguous().T, mode="reduced")[0].T  # (6,60) rows orthonormal
    Z = Z.to(torch.float64)
    D = torch.diag(torch.tensor([5.0, 0.2, 1.0, 3.0, 0.05, 2.0], dtype=torch.float64))
    H = D @ Z
    assert eps_polar(Z).item() < 1e-10               # already scaled-orthogonal
    assert abs(rho(H, t_reg_oneside(H)).item() - 1.0) < 1e-9
    # A generic H does NOT hit 1.
    assert rho(_rand(6, 60), t_reg_oneside(_rand(6, 60))).item() < 0.999


def test_fisher_racs_is_blind_to_sign_pattern():
    """RACS reads only H^(elementwise 2), so two H's with equal squares get equal scales."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    H2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    # Same entrywise squares => the transform acts by the SAME diagonal scales,
    # so T(H) differs only by inheriting H's own signs.
    T1, T2 = t_fisher_racs(H1), t_fisher_racs(H2)
    assert torch.allclose(T1.abs(), T2.abs(), atol=1e-12)


def test_reg_alg1_differs_from_oneside_even_at_t1():
    """Eq. (3) and Algorithm 1 (t=1) are different operators — the paper's ambiguity."""
    H = _rand(6, 40)
    assert not torch.allclose(t_reg_oneside(H), t_reg_alg1(H, iters=1), atol=1e-8)


def test_zero_row_does_not_produce_nan():
    H = _rand(5, 30)
    H[2] = 0.0
    for T in (t_reg_oneside(H), t_reg_alg1(H, iters=2), t_fisher_racs(H)):
        assert torch.isfinite(T).all()
    assert torch.isfinite(rho(H, t_reg_oneside(H))).all()


def test_zero_matrix_scores_zero_not_nan():
    H = torch.zeros(4, 20, dtype=torch.float64)
    assert rho(H, t_frob(H)).item() == 0.0


def test_polar_k_increases_toward_one():
    """More PolarExpress steps => closer to the exact LMO optimum."""
    H = _rand(8, 64)
    rs = [rho(H, t_polar_k(H, k)).item() for k in (1, 2, 4, 8)]
    assert all(0.0 < x <= 1.0 + 1e-6 for x in rs)
    assert rs[0] < rs[-1]
    assert rs[-1] > 0.99


def test_lmo_scores_keys_and_ranges():
    out = lmo_scores(_rand(8, 64), polar_ks=(1, 4), reg_alg1_iters=(1,))
    for k in ("eps_polar", "rho_frob", "rho_reg_oneside", "rho_reg_alg1_t1",
              "rho_fisher_racs", "rho_polar_k1", "rho_polar_k4", "rho_msign"):
        assert k in out, k
    assert abs(out["rho_msign"] - 1.0) < 1e-9
    for k, v in out.items():
        if k.startswith("rho_"):
            assert 0.0 <= v <= 1.0 + 1e-9, (k, v)


def test_rho_rejects_non_2d_in_score_suite():
    try:
        lmo_scores(torch.zeros(2, 4, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 3-D input")
