"""Tests for KL-diag-LoRA — option (b) of kl_shampoo_polar_derivation.md
§"Cross-coupling": CurvatureWhitenLoRA with ``diag_metric=True`` (plus
``kl_coupled=True, soap_v=False``).

Option (b) commits to the single global diagonal metric (P,Q)=(D_out, D_in): the
dense KL small side S_curv is replaced by the conjugate-diagonal-weighted geometric
Gram, M_A = Bᵀ diag(D_out) B and M_B = A diag(D_in) Aᵀ, recomputed each step. The
diagonal KL coupling is unchanged but now whitens by M⁻¹, giving a self-consistent
two-sided program.

Covers: step runs & changes params, multistep finiteness, the batched↔per-pair
equivalence on the diag path (both code paths were edited), the defining identity
L_A == Bᵀ diag(D_out) B, factory dispatch, and that diag (b) actually differs from
dense kl-shampoo (a).
"""
import copy

import torch
import torch.nn as nn
import pytest

from lora_playground.optim import CurvatureWhitenLoRA, build_optimizer


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        torch.nn.init.normal_(self.lora_B["default"].weight, std=0.05)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    m = TinyLoRAModel()
    x = torch.randn(3, 8)
    target = torch.randn(3, 8)
    return m, x, target


def _diag_opt(m, lr=1e-2, use_polar=True, **kw):
    return CurvatureWhitenLoRA(m, lr=lr, use_polar=use_polar,
                               kl_coupled=True, soap_v=False, diag_metric=True, **kw)


def _ssc_history_opt(m, lr=1e-2, k=1, **kw):
    return _diag_opt(
        m,
        lr=lr,
        use_polar=True,
        cw_picard_iters=k,
        cw_picard_mode="history_seeded",
        polar_method="ssc",
        ssc_kappa=0.75,
        ssc_kappa_solver="stable_rank",
        cw_eigh_seed=False,
        **kw,
    )


def test_requires_kl_coupled():
    m, _, _ = _make()
    with pytest.raises(ValueError):
        CurvatureWhitenLoRA(m, kl_coupled=False, soap_v=False, diag_metric=True)


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    before = [p.detach().clone() for p in m.parameters()]
    opt = _diag_opt(m, use_polar=use_polar)
    for _ in range(3):
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    after = [p.detach().clone() for p in m.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert all(torch.isfinite(p).all() for p in m.parameters())


@pytest.mark.parametrize("use_polar", [False, True])
def test_batched_matches_per_pair(use_polar):
    """Grouped batched path and the per-pair oracle must agree on the diag path —
    both were edited for diag_metric."""
    def run(batched):
        m, x, target = _make(seed=3)
        opt = _diag_opt(m, lr=1e-2, use_polar=use_polar, precond_refresh_every=2)
        opt._batched_step = batched
        for _ in range(4):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pg, pp in zip(run(True), run(False)):
        assert torch.allclose(pg, pp, atol=1e-5, rtol=1e-4), "batched vs per-pair diag mismatch"


def test_small_side_is_geometric_gram():
    """The defining identity: after a step, the stored small-side factor L_A equals
    Bᵀ diag(M_out) B, where M_out = (D_out/D_out.max() + δ) is the relative-damped
    output diagonal — the SAME metric the whitening and the Picard cross use (metric
    coherence). Built from the factor and diagonal as they were at the START of that
    step (option b), NOT a dense EMA of g gᵀ and NOT the raw D_out (which vanishes
    early when D_out ≈ 0)."""
    m, x, target = _make(seed=7)
    opt = _diag_opt(m, lr=1e-2, use_polar=True)
    # Warm up so D_out is non-zero (step 1 has D_out=0 ⇒ M_out=δ via _rdinv floor).
    for _ in range(3):
        ((m(x) - target) ** 2).mean().backward()
        opt.step(); opt.zero_grad()
    # Snapshot the (B, D_out) the next step will consume, per pair.
    snap = []
    for i, (A, B) in enumerate(opt.pairs):
        snap.append((B.detach().float().clone(), opt.pair_state[i]['D_out'].clone()))
    ((m(x) - target) ** 2).mean().backward()
    opt.step(); opt.zero_grad()
    for i, (B_pre, Dout_pre) in enumerate(snap):
        # M_out = _rdinv(D_out)^(-2), mirroring the code exactly (incl. the xmax≈0 floor).
        M_out = opt._rdinv(Dout_pre).pow(-2)
        M_A = B_pre.transpose(-2, -1) @ (M_out.unsqueeze(-1) * B_pre)
        assert torch.allclose(opt.pair_state[i]['L_A'], M_A, atol=1e-5, rtol=1e-4), \
            f"pair {i}: L_A is not Bᵀ diag(M_out) B (relative-damped)"


def test_diag_differs_from_dense_kl():
    """Option (b) (diag_metric) must produce a different trajectory than dense
    kl-shampoo (a) — same seed/data, different small-side metric."""
    def run(diag):
        m, x, target = _make(seed=11)
        opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=True, kl_coupled=True,
                                  soap_v=False, diag_metric=diag)
        for _ in range(5):
            ((m(x) - target) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]

    a = run(False)  # dense kl-shampoo-polar
    b = run(True)   # diag (option b)
    assert any(not torch.allclose(pa, pb, atol=1e-4) for pa, pb in zip(a, b)), \
        "diag_metric produced the same params as dense kl — substitution not in effect"


@pytest.mark.parametrize(
    "name,polar,mode,method",
    [
        ("kl-diag-lora", False, "iterated", "ns"),
        ("kl-diag-polar-lora", True, "iterated", "ns"),
        ("kl-diag-ssc-history-picard-lora", True, "history_seeded", "ssc"),
    ],
)
def test_factory_dispatch(name, polar, mode, method):
    m, _, _ = _make()
    opt = build_optimizer(
        m, optimizer_type=name, lr=1e-3,
        precond_delta=1e-3, curvature_beta=0.99, muon_ns_steps=5,
        precond_refresh_every=10,
    )
    assert isinstance(opt, CurvatureWhitenLoRA)
    assert opt.diag_metric is True
    assert opt.kl_coupled is True
    assert opt.soap_v is False
    assert opt.use_polar is polar
    assert opt.cw_picard_mode == mode
    assert opt.polar_method == method
    if name == "kl-diag-ssc-history-picard-lora":
        assert opt.ssc_kappa == 0.75
        assert opt.ssc_kappa_solver == "stable_rank"
        assert opt.cw_eigh_seed is False
        assert opt._q_initialized is True


@pytest.mark.parametrize("k", [1, 2])
def test_ssc_history_picard_uses_one_operator_pass_per_step(monkeypatch, k):
    """History-seeded k=2 may add one skinny cross term, but must not pay for a
    second SSC/polar operator call in the current step."""
    m, x, target = _make(seed=13)
    opt = _ssc_history_opt(m, k=k)
    opt._batched_step = False
    calls = []
    original = opt._polar_ns_guarded

    def counted(Z, states, key, side=None):
        calls.append((side, tuple(Z.shape)))
        return original(Z, states, key, side=side)

    monkeypatch.setattr(opt, "_polar_ns_guarded", counted)
    ((m(x) - target) ** 2).mean().backward()
    opt.step(); opt.zero_grad()

    assert len(calls) == 2 * len(opt.pairs)
    assert sorted(side for side, _shape in calls) == ["A", "A", "B", "B"]


def test_ssc_history_k2_engages_after_seed_step():
    """The first k=2 step has a zero history seed; subsequent steps must differ
    from k=1 if the stored physical update is actually used."""
    def run(k):
        m, x, target = _make(seed=17)
        opt = _ssc_history_opt(m, k=k)
        for _ in range(4):
            ((m(x) - target) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]

    k1 = run(1)
    k2 = run(2)
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(k1, k2)), \
        "history-seeded k=2 matched k=1 after the seed step"


def test_ssc_history_k2_tracks_exact_picard_correction():
    """History-seeded k=2 is the cheap production proxy for exact same-step k=2.

    From the same frozen state and gradient, compare only the Picard correction
    vectors: exact-k2 minus k1 versus history-k2 minus k1. After the first seed
    steps, the history correction should point in nearly the same direction as
    the exact same-step Picard correction and have comparable norm. This is the
    numerical guard that the efficient variant is not just "different from k1".
    """
    def run_from_state(model, opt, param_snap, state_snap, grad_snap, k, mode):
        for p, ps, g in zip(model.parameters(), param_snap, grad_snap):
            p.data.copy_(ps)
            p.grad = g.clone()
        opt.pair_state = copy.deepcopy(state_snap)
        opt.cw_picard_iters = k
        opt.cw_picard_mode = mode
        opt.step()
        opt.zero_grad()
        return torch.cat([
            (p.detach() - ps).flatten()
            for p, ps in zip(model.parameters(), param_snap)
        ])

    m, x, target = _make(seed=23)
    opt = _ssc_history_opt(m, k=1, lr=1e-2, precond_refresh_every=2)
    opt._batched_step = False
    cosines = []
    norm_ratios = []
    rel_errors = []
    for step in range(1, 12):
        ((m(x) - target) ** 2).mean().backward()
        param_snap = [p.detach().clone() for p in m.parameters()]
        grad_snap = [p.grad.detach().clone() for p in m.parameters()]
        state_snap = copy.deepcopy(opt.pair_state)

        d1 = run_from_state(m, opt, param_snap, state_snap, grad_snap,
                            k=1, mode="history_seeded")
        d_exact = run_from_state(m, opt, param_snap, state_snap, grad_snap,
                                 k=2, mode="iterated")
        d_hist = run_from_state(m, opt, param_snap, state_snap, grad_snap,
                                k=2, mode="history_seeded")
        exact_corr = d_exact - d1
        hist_corr = d_hist - d1
        exact_norm = exact_corr.norm().clamp_min(1e-30)
        hist_norm = hist_corr.norm().clamp_min(1e-30)

        if step >= 4:
            cosines.append(torch.dot(exact_corr, hist_corr) / (exact_norm * hist_norm))
            norm_ratios.append(hist_norm / exact_norm)
            rel_errors.append((hist_corr - exact_corr).norm() / exact_norm)

        # Advance the reference trajectory with the production history path so
        # cw_prev_dA/cw_prev_dB remain realistic for the next comparison.
        for p, ps, g in zip(m.parameters(), param_snap, grad_snap):
            p.data.copy_(ps)
            p.grad = g.clone()
        opt.pair_state = copy.deepcopy(state_snap)
        opt.cw_picard_iters = 2
        opt.cw_picard_mode = "history_seeded"
        opt.step()
        opt.zero_grad()

    mean_cos = torch.stack(cosines).mean()
    mean_ratio = torch.stack(norm_ratios).mean()
    mean_rel_error = torch.stack(rel_errors).mean()
    assert mean_cos > 0.95, f"history correction points away from exact Picard: cos={mean_cos:.3f}"
    assert 0.75 < mean_ratio < 1.25, f"history correction has wrong norm ratio: {mean_ratio:.3f}"
    assert mean_rel_error < 0.35, f"history correction too far from exact Picard: {mean_rel_error:.3f}"


def test_ssc_history_diagnostics_report_clipping_and_cross_terms():
    m, x, target = _make(seed=29)
    opt = _ssc_history_opt(
        m,
        k=2,
        log_basic_diagnostics=True,
        diagnostics_every=1,
    )
    opt._batched_step = False
    for _ in range(2):
        ((m(x) - target) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()

    records = opt._last_cw_diag_records
    assert records
    for rec in records:
        assert "ssc_c_A" in rec and "ssc_c_B" in rec
        assert 0.0 < rec["ssc_top_shrink_A"] <= 1.0
        assert 0.0 < rec["ssc_top_shrink_B"] <= 1.0
        assert "cw_picard_cross_A_over_base" in rec
        assert "cw_picard_cross_B_over_base" in rec
        assert rec["cw_picard_cross_A_over_base"] >= 0.0
        assert rec["cw_picard_cross_B_over_base"] >= 0.0


def test_ssc_history_factory_step_avoids_svd_and_eigh(monkeypatch):
    """The efficient named variant must not call SVD/eigh in construction or step.
    The class's standard first-step eigenseed is disabled for this path."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("SVD/eigh is forbidden on kl-diag-ssc-history-picard-lora")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)

    m, x, target = _make(seed=19)
    opt = build_optimizer(
        m,
        optimizer_type="kl-diag-ssc-history-picard-lora",
        lr=1e-3,
        cw_picard_iters=2,
        precond_refresh_every=1,
    )
    for _ in range(2):
        ((m(x) - target) ** 2).mean().backward()
        opt.step(); opt.zero_grad()
