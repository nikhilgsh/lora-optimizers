"""Ablation flags on the curvature-whiten polar family (protagonist = kl-diag-polar-lora;
diag-shampoo-polar-lora is now the −diag-curv-adjacent ablation base). Flags off skeleton Alg 1:
  --cw_no_radius     : ρ = lr  (drop the operator-norm radius ρ = lr/(σmax A + σmax B))
  --cw_no_diag_curv  : input/output diagonals → I  (C_A=BᵀB, C_B=AAᵀ; partner-Gram, iMuon-like)
Both default off (= full protagonist). The flags forward identically for the kl-diag and
diag-shampoo branches, so the tests exercise both bases. Tests verify the intended math,
not just "runs".
"""
import math
import pytest
import torch
import torch.nn as nn

from lora_playground.optim import build_optimizer, CurvatureWhitenLoRA


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.normal_(self.lora_B["default"].weight, std=0.1)  # nonzero B (avoid B=0 corner)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class _Tiny(nn.Module):
    def __init__(self, d_in=12, d_out=10, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    return _Tiny(), torch.randn(5, 12), torch.randn(5, 12)


def _build(model, **kw):
    return build_optimizer(model, "diag-shampoo-polar-lora", lr=3e-2,
                           curvature_beta=0.99, muon_ns_steps=8, precond_delta=1e-4, cw_nesterov=True, **kw)


def _build_kl(model, **kw):
    return build_optimizer(model, "kl-diag-polar-lora", lr=3e-2,
                           curvature_beta=0.99, muon_ns_steps=8, precond_delta=1e-4, cw_nesterov=True, **kw)


def _step_capture(opt, model, x, tgt):
    before = {id(p): p.detach().clone() for grp in opt.param_groups for p in grp["params"]}
    ((model(x) - tgt) ** 2).mean().backward()
    opt.step()
    deltas = {id(p): (p.detach() - before[id(p)]) for grp in opt.param_groups for p in grp["params"]}
    return deltas


def _smax(M):
    return torch.linalg.svdvals(M.float())[0].item()


def test_flags_set_and_default_off():
    m, _, _ = _make()
    base = _build(m)
    assert base.cw_no_radius is False and base.cw_no_diag_curv is False
    m2, _, _ = _make()
    abl = _build(m2, cw_no_radius=True, cw_no_diag_curv=True)
    assert abl.cw_no_radius is True and abl.cw_no_diag_curv is True


def test_no_radius_gives_sigma_max_equal_lr():
    """−radius ⟹ ρ=lr, and the operator-norm rescale sets σmax(Ȧ)=ρ=lr."""
    lr = 3e-2
    m, x, tgt = _make(seed=1)
    opt = _build(m, cw_no_radius=True)
    deltas = _step_capture(opt, m, x, tgt)
    for d in deltas.values():
        assert math.isfinite(d.norm().item())
        # each per-factor update is rescaled to σmax = ρ = lr
        assert _smax(d) == pytest.approx(lr, rel=0.08), f"σmax(Ȧ)={_smax(d)} != lr={lr}"


def test_radius_on_gives_smaller_sigma_than_lr():
    """Protagonist (radius on): ρ=lr/(σmaxA+σmaxB); for well-scaled factors σmaxA+σmaxB>1,
    so σmax(Ȧ)=ρ < lr — distinct from the −radius arm."""
    lr = 3e-2
    m, x, tgt = _make(seed=1)
    opt = _build(m)  # radius ON
    deltas = _step_capture(opt, m, x, tgt)
    smaxes = [_smax(d) for d in deltas.values()]
    assert all(math.isfinite(s) for s in smaxes)
    # radius divides by (σmaxA+σmaxB); not pinned to lr like the −radius arm
    assert not all(s == pytest.approx(lr, rel=0.08) for s in smaxes)


def test_no_diag_curv_differs_and_finite():
    """−Shampoo changes the update once the diagonals accumulate (the curvature collapses
    to the partner-Gram) and stays finite. Needs >1 step: at step 1 the diagonals are still
    zero → identity-fallback, so the flag is a no-op there."""
    m1, x, tgt = _make(seed=2)
    m2, _, _ = _make(seed=2)
    full = _build(m1)
    nosh = _build(m2, cw_no_diag_curv=True)
    for _ in range(4):  # let D_in/D_out accumulate so the flag bites
        for opt, m in ((full, m1), (nosh, m2)):
            opt.zero_grad(set_to_none=False)
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
    diffs = [(p1.detach() - p2.detach()).norm().item()
             for p1, p2 in zip(m1.parameters(), m2.parameters())]
    assert any(df > 1e-6 for df in diffs), "−Shampoo did not change the trajectory by step 4"
    assert all(torch.isfinite(p).all() for p in m2.parameters())


def test_no_radius_grouped_with_diagnostics_no_crash():
    """Regression: cw_no_radius on the GROUPED (higham) path with basic diagnostics ON.
    The diagnostic record indexes ρ[j]; pre-fix ρ was a scalar float in the cw_no_radius
    branch → 'float object is not subscriptable' once diagnostics fired (~step 100 in
    production). The -radius sweep runs exactly this path (higham + --log_basic_diagnostics
    + --optim_diagnostics_every), so it must run and stay finite. The default-eigh,
    no-diagnostics test above does NOT exercise this branch."""
    m, x, tgt = _make(seed=3)
    opt = _build(m, cw_no_radius=True, precond_method="higham",
                 log_basic_diagnostics=True, optim_diagnostics_every=1)
    for _ in range(3):
        opt.zero_grad(set_to_none=False)
        ((m(x) - tgt) ** 2).mean().backward()
        opt.step()  # must not raise (ρ[j] indexing on the grouped diagnostic path)
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_no_diag_curv_requires_diag_metric():
    """The flag is only defined on the diag_metric (protagonist) path."""
    m, _, _ = _make()
    pairs_model = m
    with pytest.raises(ValueError, match="diag_metric"):
        CurvatureWhitenLoRA(pairs_model, lr=1e-2, diag_metric=False, cw_no_diag_curv=True)


@pytest.mark.parametrize("opt_name", [
    "kl-shampoo-polar-lora", "kl-shampoo-lora", "kl-diag-polar-lora",
])
def test_cw_nesterov_honored_by_kl_branches(opt_name):
    """Regression: --cw_nesterov was passed to the optimizer in the diag-shampoo branch
    ONLY; the kl-* branches (soap_v=False, so Nesterov IS valid) dropped it and used the
    False default while train.py logged args.cw_nesterov=True — a silent provenance bug
    that confounded kl-vs-diag comparisons. The kl-* branches must now honor the flag."""
    from lora_playground.optim import build_optimizer
    for want in (True, False):
        m, _, _ = _make()
        opt = build_optimizer(m, opt_name, lr=1e-2, curvature_beta=0.99,
                              muon_ns_steps=5, precond_delta=1e-4, cw_nesterov=want)
        assert opt.cw_nesterov is want, f"{opt_name} ignored cw_nesterov={want}"


@pytest.mark.parametrize("flag", ["cw_no_radius", "cw_no_diag_curv"])
def test_kl_diag_honors_ablation_flags(flag):
    """Regression: the ablation flags (cw_no_radius / cw_no_diag_curv) were wired
    into the diag-shampoo-polar-lora branch ONLY; the kl-diag-polar-lora branch
    (the protagonist as of the kl-diag switch) silently DROPPED them, so an
    ablation sweep on kl-diag would run the FULL protagonist while the config event
    claimed the flag was set — the exact provenance bug class as cw_nesterov. The
    kl-diag branch must honor both flags."""
    from lora_playground.optim import build_optimizer
    m, _, _ = _make()
    opt = build_optimizer(m, "kl-diag-polar-lora", lr=1e-2, curvature_beta=0.99,
                          muon_ns_steps=8, precond_delta=1e-4, cw_nesterov=True,
                          **{flag: True})
    assert getattr(opt, flag) is True, f"kl-diag-polar-lora dropped {flag}"


def test_kl_diag_no_shampoo_matches_diag_shampoo_no_shampoo():
    """The −Shampoo arm (cw_no_diag_curv) is base-independent: forcing the large-axis
    diagonals to I (dinA=doutB=1) collapses both diag-shampoo and kl-diag to the same
    partner-Gram-only update — the kl_coupled D_in/D_out EMAs are still accumulated but never
    read (overwritten to ones each step). So kl-diag −Shampoo must step IDENTICALLY to
    diag-shampoo −Shampoo (justifies reusing the existing −Shampoo run for both bases)."""
    m1, x, tgt = _make(seed=7)
    m2, _, _ = _make(seed=7)
    kl = _build_kl(m1, cw_no_diag_curv=True)
    ds = _build(m2, cw_no_diag_curv=True)
    for _ in range(5):  # let diagonals accumulate so the coupling path is exercised
        for opt, m in ((kl, m1), (ds, m2)):
            opt.zero_grad(set_to_none=False)
            ((m(x) - tgt) ** 2).mean().backward()
            opt.step()
    diffs = [(p1.detach() - p2.detach()).abs().max().item()
             for p1, p2 in zip(m1.parameters(), m2.parameters())]
    assert all(df < 1e-6 for df in diffs), f"kl-diag −Shampoo != diag-shampoo −Shampoo: {diffs}"


def _build_kl_gramns(model, **kw):
    return build_optimizer(model, "kl-diag-polar-lora", lr=3e-2,
                           curvature_beta=0.99, muon_ns_steps=8, precond_delta=1e-4,
                           cw_nesterov=True, precond_method="gram_ns", **kw)


def test_unpinned_requires_gram_ns():
    """cw_unpinned (−pin) needs the TRUE-SCALE gram_ns inverse-sqrt; the eigh/higham paths
    damp relatively, which only the σ_max(W) pin (now removed) would reabsorb — so they are
    rejected at construction."""
    m, _, _ = _make()
    with pytest.raises(ValueError, match="gram_ns"):
        _build_kl(m, cw_unpinned=True, precond_method="eigh")


def test_unpinned_flag_forwards():
    """The kl-diag protagonist must honor cw_unpinned (provenance-bug class)."""
    m, _, _ = _make()
    opt = _build_kl_gramns(m, cw_unpinned=True)
    assert opt.cw_unpinned is True
    assert _build_kl_gramns(_make()[0]).cw_unpinned is False


def test_unpinned_removes_the_pin():
    """−pin (cw_unpinned): dX = −η·W applied RAW (no σ_max(W) rescale), so σmax(Ȧ) is the
    native family magnitude, NOT pinned to ρ=lr. Contrast cw_no_radius: same ρ=lr but the
    σ_max(W) rescale pins σmax(Ȧ)=lr. Both on the gram_ns + partner-Gram (cw_no_diag_curv)
    path, nonzero-B init (std=0.1) so the unpinned whitener is finite."""
    lr = 3e-2
    m1, x, tgt = _make(seed=11)
    m2, _, _ = _make(seed=11)
    pinned = _build_kl_gramns(m1, cw_no_radius=True, cw_no_diag_curv=True)   # ρ=lr, PINNED
    unpin = _build_kl_gramns(m2, cw_unpinned=True, cw_no_diag_curv=True)      # ρ=lr, NO pin
    dp = _step_capture(pinned, m1, x, tgt)
    du = _step_capture(unpin, m2, x, tgt)
    for d in list(dp.values()) + list(du.values()):
        assert math.isfinite(d.norm().item())
    # pinned: every factor rescaled to σmax = lr; unpinned: at least one factor (the A-side,
    # whitened by the small B) far exceeds lr because nothing rescales it.
    assert all(_smax(d) == pytest.approx(lr, rel=0.08) for d in dp.values()), "pinned arm not pinned"
    assert any(_smax(d) > lr * 1.5 for d in du.values()), "unpinned arm still pinned to lr"


def test_curvature_whiten_does_not_apply_nesterov_and_logs_effective():
    """curvature-whiten is soap_v=True (SOAP path); cw_nesterov is incompatible (the
    optimizer raises if both are set), so the build does NOT wire it — opt.cw_nesterov
    stays False even when requested. Provenance is kept honest by logging the EFFECTIVE
    optimizer attr (train.py), not args.cw_nesterov. This asserts the effective value
    that the config event reads."""
    from lora_playground.optim import build_optimizer
    m, _, _ = _make()
    opt = build_optimizer(m, "curvature-whiten-polar-lora", lr=1e-2, curvature_beta=0.99,
                          muon_ns_steps=5, precond_delta=1e-4, cw_nesterov=True)
    assert opt.cw_nesterov is False  # silently not applied; the config logs this False


def test_the_per_pair_path_refuses_cw_unpinned():
    """Known-positive for the second branch of the `_batched_step` dispatch guard.

    `_cw_apply_per_pair` never reads `cw_unpinned` (verified by grep over both
    function bodies), so it neither flattens rho nor skips the sigma_max(W)
    rescale — it silently runs the PINNED step while the config says unpinned.
    The guard refuses instead. Its sibling branch (a non-eigh precond_method)
    has `test_the_per_pair_oracle_refuses_a_production_precond_method` in
    tests/test_precond_msign_branches.py; this one had only a docstring, so half
    the guard was an unverified detector.

    Note cw_unpinned REQUIRES precond_method="gram_ns"
    (`test_unpinned_requires_gram_ns`), so the precond_method branch would fire
    on this config first. The guard checks precond_method before cw_unpinned, so
    this test pins the ordering too: if they were swapped, the cw_unpinned
    message would never be reachable.
    """
    m, _, _ = _make(seed=5)
    opt = _build_kl_gramns(m, cw_unpinned=True, cw_no_diag_curv=True)
    opt._batched_step = False
    for A, B in opt.pairs:
        A.grad = torch.zeros_like(A)
        B.grad = torch.zeros_like(B)
    with pytest.raises(NotImplementedError, match="precond_method|cw_unpinned"):
        opt.step()
