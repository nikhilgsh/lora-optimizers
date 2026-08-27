"""Tests for KL-Shampoo-LoRA (soap_curvature_whitening.md, experiment 5).

KL-Shampoo-LoRA is CurvatureWhitenLoRA with two flags flipped:
``kl_coupled=True`` (curvature factors accumulated by the coupled KL fixed
point, Prop 4, instead of the one-sided EMA(gg^T)/diag(g^T g)) and
``soap_v=False`` (drop the SOAP v̂; inner core is the closed-form Shampoo
whitening S^{-1/2} m̂ D^{-1/2}). Both arms via the polar toggle.

Covers: step runs & changes params (both arms), zero-grad finiteness,
determinism, factory dispatch for both names, the batched↔per-pair equivalence
on the KL path, and the warmup KL-Gram identity (the 1/d normalizers + the
identity-inverse fallback when the factors are still zero).
"""
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


def _kl_opt(m, lr=1e-2, use_polar=False, **kw):
    return CurvatureWhitenLoRA(m, lr=lr, use_polar=use_polar,
                               kl_coupled=True, soap_v=False, **kw)


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = _kl_opt(m, lr=1e-2, use_polar=use_polar)
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params after step"
    assert max(float((a - b).abs().sum()) for a, b in zip(pre, post)) > 0.0, "No params changed"


@pytest.mark.parametrize("use_polar", [False, True])
def test_zero_grad_no_finite_failure(use_polar):
    m, x, target = _make()
    opt = _kl_opt(m, lr=1e-2, use_polar=use_polar)
    loss = (m(x) * 0.0).sum()
    loss.backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param after zero-grad step"


@pytest.mark.parametrize("use_polar", [False, True])
def test_multistep_finite(use_polar):
    m, x, target = _make(seed=7)
    opt = _kl_opt(m, lr=1e-2, use_polar=use_polar, precond_refresh_every=2)
    for _ in range(6):
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param over multiple KL steps"


@pytest.mark.parametrize("use_polar", [False, True])
def test_determinism(use_polar):
    def run():
        m, x, target = _make(seed=42)
        opt = _kl_opt(m, lr=1e-3, use_polar=use_polar)
        for _ in range(3):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pa, pb in zip(run(), run()):
        assert torch.allclose(pa, pb, atol=0.0)


@pytest.mark.parametrize("use_polar", [False, True])
def test_kl_path_is_companion_independent(use_polar, companion_independent):
    """Batching must not let one pair's KL update depend on its shape-group
    companions — guards the coupled-EMA / Shampoo-core orientation through the
    stacked bmm.

    This replaces a grouped-vs-per-pair comparison against
    `_cw_apply_per_pair`, a second implementation of the same step that has
    since been deleted. That comparison could not catch a bug both copies
    shared, and it ran only at `precond_method="eigh"`, which no production
    sweep uses; this one is bit-exact and config-agnostic.
    """
    companion_independent(lr=1e-2, use_polar=use_polar, kl_coupled=True,
                          soap_v=False, precond_refresh_every=2)


def test_warmup_kl_gram_identity():
    """On step 1 the factors are zero ⇒ the relative-damped inverses fall back to
    identity, so the coupled KL Gram update reduces to its one-sided warm form
    with the 1/d normalizer:
        P_A  = (1-β_c)/d_in * g_A g_Aᵀ
        Q = (1-β_c)/r    * diag(g_Aᵀ g_A)
    This pins both the 1/d normalizers and the identity-inverse warmup fallback.
    """
    cb = 0.99
    m, x, target = _make(seed=11)
    opt = _kl_opt(m, lr=1e-3, curvature_beta=cb)
    A, B = opt.pairs[0]
    r, d_in = A.shape
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    gA = A.grad.detach().float().clone()  # captured before step() zeros it
    opt.step()
    st = opt.pair_state[0]
    expected_PA = (1.0 - cb) / d_in * (gA @ gA.T)
    expected_Q = (1.0 - cb) / r * (gA * gA).sum(dim=0)
    assert torch.allclose(st["P_A"], expected_PA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["Q"], expected_Q, atol=1e-6, rtol=1e-5)


def test_coupled_kl_gram_update_nontrivial():
    """Pin the coupled KL Gram EMA (Prop 4) against injected NON-identity factors.

    The warmup test only sees identity inverses, and grouped-vs-per-pair
    equivalence can't catch a CONSISTENT orientation/normalizer bug (same error
    in both paths). This recomputes the four coupled EMA targets from the
    pre-update factors using the reference formula and the optimizer's own
    relative-damped inverse, then checks the post-step state matches:
        P_A ← β P_A + ((1-β)/d_in) g_A Q^{-1} g_A^T
        Q ← β Q + ((1-β)/r) diag(g_A^T S_curv,A^{-1} g_A)
    (B-side symmetric with 1/d_out and 1/r). precond_refresh_every is set huge so
    the injected eigenbasis Q is not overwritten mid-step.
    """
    cb = 0.9
    m, x, target = _make(seed=5)
    opt = _kl_opt(m, lr=1e-3, curvature_beta=cb, precond_refresh_every=10_000)
    opt._q_initialized = True  # no eigh reseed; keep the injected Q
    A, B = opt.pairs[0]
    r, d_in = A.shape
    d_out = B.shape[0]
    st = opt.pair_state[0]

    torch.manual_seed(1)
    UA = torch.linalg.qr(torch.randn(r, r))[0]
    UB = torch.linalg.qr(torch.randn(r, r))[0]
    lamA_eig = torch.tensor([4.0, 1.0, 0.5, 2.0])[:r]
    lamB_eig = torch.tensor([1.0, 3.0, 0.25, 1.5])[:r]
    st["U_A"].copy_(UA)
    st["U_B"].copy_(UB)
    st["P_A"].copy_(UA @ torch.diag(lamA_eig) @ UA.T)
    st["Q_B"].copy_(UB @ torch.diag(lamB_eig) @ UB.T)
    st["Q"].copy_(torch.linspace(0.5, 3.0, d_in))
    st["P"].copy_(torch.linspace(0.4, 2.5, d_out))

    PA0 = st["P_A"].clone(); Q0 = st["Q"].clone()
    QB0 = st["Q_B"].clone(); P0 = st["P"].clone()

    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    gA = A.grad.detach().float().clone()
    gB = B.grad.detach().float().clone()
    opt.step()

    # Reference inverses from the PRE-update factors (matches the code path).
    lamA = opt._rdinv((UA * (PA0 @ UA)).sum(dim=0))      # (-1/2)
    lamB = opt._rdinv((UB * (QB0 @ UB)).sum(dim=0))
    Q_inv = opt._rdinv(Q0).square()
    P_inv = opt._rdinv(P0).square()
    SAinv = UA @ (torch.diag(lamA * lamA) @ UA.T)
    RBinv = UB @ (torch.diag(lamB * lamB) @ UB.T)

    exp_PA = cb * PA0 + ((1 - cb) / d_in) * ((gA * Q_inv.unsqueeze(0)) @ gA.T)
    exp_Q = cb * Q0 + ((1 - cb) / r) * (gA * (SAinv @ gA)).sum(dim=0)
    exp_QB = cb * QB0 + ((1 - cb) / d_out) * (gB.T @ (gB * P_inv.unsqueeze(-1)))
    exp_P = cb * P0 + ((1 - cb) / r) * (gB * (gB @ RBinv)).sum(dim=1)

    assert torch.allclose(st["P_A"], exp_PA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["Q"], exp_Q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["Q_B"], exp_QB, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["P"], exp_P, atol=1e-6, rtol=1e-5)


def test_polar_ns_guard_survives_sigma_max_underestimate():
    """Regression for the r256 polar NaN: a σ_max UNDER-estimate sends zA/σ̂ into
    the Newton-Schulz above its convergence basin → NaN. First show the unguarded
    normalization NaNs on a wide-spectrum matrix when σ_max is grossly low, then
    show _polar_ns_guarded stays finite and orthogonalizes (σ_max(out) ≈ 1)."""
    from lora_playground.optim import _newton_schulz_gram_batched
    torch.manual_seed(0)
    U, _, Vh = torch.linalg.svd(torch.randn(32, 128), full_matrices=False)
    S = torch.linspace(1.0, 0.02, U.shape[1])
    Z = (U * S) @ Vh
    Z = Z / torch.linalg.matrix_norm(Z, ord=2) * 5.0          # σ_max(Z) = 5
    Zb = Z.unsqueeze(0)

    # Unguarded with a 10x under-estimate (σ̂=0.5): NS input σ_max=10 → blows up.
    bad = _newton_schulz_gram_batched(Zb / 0.5, nsteps=5, dtype=torch.float32,
                                      pre_norm="none", safety_factor=1.0)
    assert not torch.isfinite(bad).all(), "expected unguarded NS to diverge (premise)"

    # Guarded path, with a stale warm vector seeded into the bottom subspace so the
    # warm power-iter under-estimates σ_max — the guard must still return finite.
    m, _, _ = _make()
    opt = _kl_opt(m, use_polar=True)
    st = {"v_sigma_test": Vh[-1].clone()}                     # near-nullspace start
    out = opt._polar_ns_guarded(Zb, [st], "v_sigma_test")
    assert torch.isfinite(out).all(), "guarded polar produced non-finite output"
    assert abs(float(torch.linalg.matrix_norm(out[0], ord=2)) - 1.0) < 0.15, \
        "guarded polar did not orthogonalize"


def test_smax_warm_floored_and_accurate():
    """Regression for the shared-estimator fix: _smax_warm must never return below
    the max(row,col) L2 lower bound (the floor that guards a stale warm vector at
    EVERY call site), and must be accurate (not the chronic ~15% under-estimate the
    warm 3-iter start gave that overscaled the rescale into a NaN)."""
    m, _, _ = _make()
    opt = _kl_opt(m, use_polar=True)
    torch.manual_seed(0)
    U, _, Vh = torch.linalg.svd(torch.randn(16, 64), full_matrices=False)
    M = ((U * torch.linspace(1.0, 0.02, 16)) @ Vh).unsqueeze(0) * 5.0   # σ_max = 5
    true = float(torch.linalg.matrix_norm(M[0], ord=2))
    rn = float(M.pow(2).sum(-1).amax(-1).sqrt()[0])
    cn = float(M.pow(2).sum(-2).amax(-1).sqrt()[0])
    floor = max(rn, cn)
    # Even seeded with a near-zero (worst-case stale) warm vector, the estimate
    # must stay >= the deterministic floor and within 5% above the true σ_max.
    st = {"v_sigma_test": torch.zeros(64)}
    s = float(opt._smax_warm(M, [st], "v_sigma_test")[0])
    assert s >= floor - 1e-5, f"below row/col floor: {s} < {floor}"
    assert s <= true * 1.05, f"over-estimates: {s} > {true}"
    assert s >= true * 0.9, f"under-estimates (chronic-bias regression): {s} < {true}"


@pytest.mark.parametrize("name,polar", [
    ("kl-shampoo-lora", False),
    ("kl-shampoo-polar-lora", True),
])
def test_factory_dispatch(name, polar):
    m, _, _ = _make()
    opt = build_optimizer(
        m, optimizer_type=name, lr=1e-3,
        precond_delta=1e-3, curvature_beta=0.99, muon_ns_steps=5,
        precond_refresh_every=10,
    )
    assert isinstance(opt, CurvatureWhitenLoRA)
    assert opt.kl_coupled is True
    assert opt.soap_v is False
    assert opt.use_polar is polar


@pytest.mark.parametrize("use_polar", [False, True])
def test_picard_converges_by_k2(use_polar):
    """The Picard fixed-point iteration contracts: the k=2→k=3 increment is far
    smaller than the k=1→k=2 increment, so k=2 captures the correction and
    k>2 is unnecessary."""
    import copy
    m, x, target = _make(seed=3)
    opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar, kl_coupled=True,
                              soap_v=False, diag_metric=True)
    opt.cw_picard_iters = 1
    for _ in range(12):
        for p in m.parameters():
            p.grad = None
        (((m(x) - target) ** 2).mean() * 10.0).backward()
        opt.step()
    snap = ([p.detach().clone() for p in m.parameters()], copy.deepcopy(opt.pair_state))

    def upd(k):
        opt.cw_picard_iters = k
        with torch.no_grad():
            for p, s in zip(m.parameters(), snap[0]):
                p.copy_(s)
        opt.pair_state = copy.deepcopy(snap[1])
        pre = [p.detach().clone() for p in m.parameters()]
        for p in m.parameters():
            p.grad = None
        (((m(x) - target) ** 2).mean() * 10.0).backward()
        opt.step()
        return torch.cat([(p.detach() - s).flatten() for p, s in zip(m.parameters(), pre)])

    d = {k: upd(k) for k in (1, 2, 3)}
    inc_12 = (d[2] - d[1]).norm()
    inc_23 = (d[3] - d[2]).norm()
    assert inc_23 < 0.25 * inc_12, (
        f"Picard not contracting by k=2: inc(2->3)={inc_23:.3e} vs inc(1->2)={inc_12:.3e}")
