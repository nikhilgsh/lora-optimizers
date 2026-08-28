"""The two orthogonal selectors on the curvature-whiten family.

``precond`` picks what fills the two r x r slots::

    product     C_B = B^T P B,  C_A = A Q A^T
    one-sided   C_B = C_A = I_r
    factorwise  C_B = P_A,      C_A = Q_B   (EMAs of G_A Q^-1 G_A^T / G_B^T P^-1 G_B)
    factorwise-diag  C_B = Diag(P_A), C_A = Diag(Q_B)

``msign`` picks how accurately the matrix sign is applied to the whitened momenta::

    full   U = msign(Z)
    diag   U_A = rownorm(Z_A),  U_B = colnorm(Z_B)

They are orthogonal by construction; these tests pin that, and pin the two
properties that are easy to break silently:

  1. `precond` DEFAULTS to inheriting each spec's `diag_metric`. Five specs pin
     it False, so a "product" default would flip them. The equivalence tests
     below are the guard.
  2. `one-sided` puts the identity in the p, q ESTIMATOR as well as the
     direction. A scalar slot cI cancels from the direction (msign is
     scale-invariant, and the rho/sigma_max rescale removes c^-1/2) but NOT from
     the estimator, so "the direction looks right" is not evidence.
"""
import pytest
import torch
import torch.nn as nn

from lora_playground.optim import (
    MSIGN_CHOICES, PRECOND_CHOICES, CurvatureWhitenLoRA, build_optimizer,
)


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.normal_(self.lora_B["default"].weight, std=0.1)

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


_KW = dict(lr=3e-2, curvature_beta=0.99, ns_steps=8, delta=1e-4,
           cw_nesterov=True, kl_coupled=True, soap_v=False,
           use_polar=True, precond_method="gram_ns")


def _run(model, opt, x, tgt, steps=4):
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        ((model(x) - tgt) ** 2).mean().backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def _max_abs_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a, b))


# ─── the default inherits, it does not override ───────────────────────────────

@pytest.mark.parametrize("diag_metric,expected", [(True, "product"), (False, "factorwise")])
def test_default_precond_inherits_diag_metric(diag_metric, expected):
    """precond=None must READ the spec's diag_metric, not write it.

    Five specs pin diag_metric=False (curvature-whiten-lora,
    curvature-whiten-polar-lora, kl-shampoo-lora, kl-shampoo-polar-lora and the
    two flatout variants). A `precond="product"` default would silently move all
    of them onto the partner-Gram slots.
    """
    m, _, _ = _make()
    opt = CurvatureWhitenLoRA(m, diag_metric=diag_metric, **_KW)
    assert opt.precond == expected
    assert opt.diag_metric is diag_metric
    assert opt.rr_identity is False


@pytest.mark.parametrize("name,diag_metric", [
    ("kl-diag-polar-lora", True),
    ("kl-shampoo-polar-lora", False),
])
def test_explicit_precond_reproduces_the_legacy_path_bitwise(name, diag_metric):
    """`--precond product` on the diag_metric spec, and `--precond factorwise` on
    the non-diag_metric one, must be BIT-IDENTICAL to leaving it unset. This is
    what lets the existing runs of both keep their meaning under the new flag."""
    want = "product" if diag_metric else "factorwise"
    m1, x, tgt = _make(seed=5)
    m2, _, _ = _make(seed=5)
    kw = dict(lr=3e-2, curvature_beta=0.99, muon_ns_steps=8, precond_delta=1e-4,
              cw_nesterov=True, precond_method="gram_ns")
    a = _run(m1, build_optimizer(m1, name, **kw), x, tgt)
    b = _run(m2, build_optimizer(m2, name, precond=want, **kw), x, tgt)
    assert _max_abs_diff(a, b) == 0.0


def test_precond_and_msign_reject_unknown_values():
    m, _, _ = _make()
    with pytest.raises(ValueError, match="precond must be one of"):
        CurvatureWhitenLoRA(m, precond="one_sided", **_KW)   # underscore, not hyphen
    m2, _, _ = _make()
    with pytest.raises(ValueError, match="msign must be one of"):
        CurvatureWhitenLoRA(m2, msign="rownorm", **_KW)


def test_msign_diag_requires_a_matrix_sign_to_approximate():
    """On a spec that applies no matrix sign, msign='diag' would silently do
    nothing — which is the failure mode an arm can't detect from its loss."""
    m, _, _ = _make()
    kw = {**_KW, "use_polar": False}
    with pytest.raises(ValueError, match="use_polar=True"):
        CurvatureWhitenLoRA(m, msign="diag", **kw)


# ─── one-sided is identity in the ESTIMATOR, not just the direction ───────────

def test_one_sided_holds_both_slots_at_exact_identity():
    m, x, tgt = _make(seed=3)
    opt = CurvatureWhitenLoRA(m, precond="one-sided", **_KW)
    assert opt.rr_identity is True
    _run(m, opt, x, tgt, steps=3)
    r = opt.pair_state[0]['P_A'].shape[-1]
    eye = torch.eye(r)
    for st in opt.pair_state.values() if isinstance(opt.pair_state, dict) else opt.pair_state:
        assert torch.equal(st['P_A'], eye), "C_B drifted off the identity"
        assert torch.equal(st['Q_B'], eye), "C_A drifted off the identity"


def test_one_sided_q_update_is_the_unwhitened_gram_diagonal():
    """The p, q updates must whiten by I, i.e.

        qhat = (1/r) diag(G_A^T G_A),   phat = (1/r) diag(G_B G_B^T)

    A scalar slot cI would leave the DIRECTION untouched but scale these, so this
    is the assertion that actually distinguishes exact-I from damped-inverse-of-I.
    """
    m, x, tgt = _make(seed=11)
    opt = CurvatureWhitenLoRA(m, precond="one-sided", **_KW)
    opt.zero_grad(set_to_none=False)
    ((m(x) - tgt) ** 2).mean().backward()
    gA = [A.grad.detach().float().clone() for A, B in opt.pairs]
    gB = [B.grad.detach().float().clone() for A, B in opt.pairs]
    # Compare against the EMA's own prior state rather than reconstructing the
    # cw_metric_init fill — that keeps the assertion exact and independent of
    # which init mode is in force.
    prev = [(opt.pair_state[i]['Q'].clone(), opt.pair_state[i]['P'].clone())
            for i in range(len(opt.pairs))]
    opt.step()
    cb = opt.curvature_beta
    for i, (ga, gb) in enumerate(zip(gA, gB)):
        r = ga.shape[0]
        st = opt.pair_state[i]
        d_in_prev, d_out_prev = prev[i]
        want_in = cb * d_in_prev + (1.0 - cb) / r * (ga * ga).sum(dim=0)
        want_out = cb * d_out_prev + (1.0 - cb) / r * (gb * gb).sum(dim=1)
        assert torch.allclose(st['Q'], want_in, atol=1e-8, rtol=1e-5), \
            "q update did not whiten by the identity"
        assert torch.allclose(st['P'], want_out, atol=1e-8, rtol=1e-5), \
            "p update did not whiten by the identity"


def test_precond_branches_are_mutually_distinct():
    outs = {}
    for p in sorted(PRECOND_CHOICES):
        m, x, tgt = _make(seed=9)
        outs[p] = _run(m, CurvatureWhitenLoRA(m, precond=p, **_KW), x, tgt)
        assert all(torch.isfinite(t).all() for t in outs[p]), f"{p} produced non-finite params"
    names = sorted(outs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert _max_abs_diff(outs[a], outs[b]) > 1e-8, f"{a} and {b} are the same step"


def test_factorwise_diag_keeps_only_the_ema_diagonal():
    """The new arm must fit the diagonal recurrence itself, not form a dense
    r x r EMA and merely use its diagonal later."""
    m, x, tgt = _make(seed=19)
    opt = CurvatureWhitenLoRA(m, precond="factorwise-diag", **_KW)
    opt.zero_grad(set_to_none=False)
    ((m(x) - tgt) ** 2).mean().backward()
    before = [(st["P_A"].clone(), st["Q_B"].clone(), st["Q"].clone(), st["P"].clone())
              for st in opt.pair_state.values()]
    grads = [(A.grad.detach().float().clone(), B.grad.detach().float().clone())
             for A, B in opt.pairs]
    opt.step()
    cb = opt.curvature_beta
    for i, ((gA, gB), (pa0, qb0, q0, p0)) in enumerate(zip(grads, before)):
        q_inv = opt._rdinv(q0, partner_trace=p0.sum()).square()
        p_inv = opt._rdinv(p0, partner_trace=q0.sum()).square()
        want_pa_diag = cb * pa0.diagonal() + (1.0 - cb) / gA.shape[1] * (
            gA.square() * q_inv.unsqueeze(0)).sum(dim=1)
        want_qb_diag = cb * qb0.diagonal() + (1.0 - cb) / gB.shape[0] * (
            gB.square() * p_inv.unsqueeze(1)).sum(dim=0)
        pa = opt.pair_state[i]["P_A"]
        qb = opt.pair_state[i]["Q_B"]
        assert torch.count_nonzero(pa - torch.diag_embed(pa.diagonal())) == 0
        assert torch.count_nonzero(qb - torch.diag_embed(qb.diagonal())) == 0
        assert torch.allclose(pa.diagonal(), want_pa_diag, atol=1e-8, rtol=1e-5)
        assert torch.allclose(qb.diagonal(), want_qb_diag, atol=1e-8, rtol=1e-5)


# ─── msign=diag is row/column normalization, and is orthogonal to precond ─────

def test_msign_diag_is_row_and_column_normalization():
    """`_direction_op` with msign='diag' must return exactly rownorm on the wide
    A side and colnorm on the tall B side — unit row norms / unit column norms."""
    m, _, _ = _make()
    opt = CurvatureWhitenLoRA(m, precond="one-sided", msign="diag", **_KW)
    torch.manual_seed(0)
    zA = torch.randn(3, 4, 16)   # (N, r, d_in) — wide
    zB = torch.randn(3, 12, 4)   # (N, d_out, r) — tall
    uA = opt._direction_op(zA, None, None, 'A')
    uB = opt._direction_op(zB, None, None, 'B')
    assert torch.allclose(uA.pow(2).sum(dim=-1).sqrt(), torch.ones(3, 4), atol=1e-5)
    assert torch.allclose(uB.pow(2).sum(dim=-2).sqrt(), torch.ones(3, 4), atol=1e-5)
    # And it is the stated closed form, not merely unit-normed.
    assert torch.allclose(uA, zA / zA.pow(2).sum(-1, keepdim=True).sqrt(), atol=1e-6)
    assert torch.allclose(uB, zB / zB.pow(2).sum(-2, keepdim=True).sqrt(), atol=1e-6)


def test_msign_diag_survives_a_zero_row():
    """A rank direction with no momentum must not become an unbounded direction."""
    m, _, _ = _make()
    opt = CurvatureWhitenLoRA(m, precond="one-sided", msign="diag", **_KW)
    zA = torch.randn(1, 4, 16)
    zA[0, 2] = 0.0
    out = opt._direction_op(zA, None, None, 'A')
    assert torch.isfinite(out).all()
    assert out[0, 2].abs().max().item() == 0.0


@pytest.mark.parametrize("precond", sorted(PRECOND_CHOICES))
def test_msign_is_orthogonal_to_precond(precond):
    """Both selectors must be live on every combination: switching msign has to
    change the step whichever slot contents are in play, and stay finite."""
    m1, x, tgt = _make(seed=13)
    m2, _, _ = _make(seed=13)
    full = _run(m1, CurvatureWhitenLoRA(m1, precond=precond, msign="full", **_KW), x, tgt)
    diag = _run(m2, CurvatureWhitenLoRA(m2, precond=precond, msign="diag", **_KW), x, tgt)
    assert all(torch.isfinite(t).all() for t in diag)
    assert _max_abs_diff(full, diag) > 1e-8, \
        f"msign=diag was a no-op under precond={precond}"


def _spy_on_expensive_primitives(monkeypatch):
    """Record calls to the O(r^2 d) primitives; returns the growing call list.

    Watches the r x r inverse square root (`gram_ns_inv_sqrt`, a module-level
    function that `_cw_apply_grouped` calls as a bare global, so it is patched on
    the module) and the polar/msign step (`_polar_ns_guarded`, a method). Every
    name is resolved with a bare `getattr` and no None-skip, so a name that stops
    existing fails the test instead of quietly dropping out of the watch set.
    """
    import lora_playground.optim as O

    calls = []

    orig_polar = CurvatureWhitenLoRA._polar_ns_guarded

    def polar_spy(self, *a, **k):
        calls.append("_polar_ns_guarded")
        return orig_polar(self, *a, **k)

    monkeypatch.setattr(CurvatureWhitenLoRA, "_polar_ns_guarded", polar_spy)

    orig_invsqrt = O.gram_ns_inv_sqrt

    def invsqrt_spy(*a, **k):
        calls.append("gram_ns_inv_sqrt")
        return orig_invsqrt(*a, **k)

    monkeypatch.setattr(O, "gram_ns_inv_sqrt", invsqrt_spy)
    return calls


def test_cheap_branch_does_no_rxr_inverse_sqrt(monkeypatch):
    """precond=one-sided + msign=diag is the O(rd) configuration: no r x r matmul
    or inverse square root anywhere in the direction. Assert it by refusing the
    inverse-sqrt primitives, which is what actually costs O(r^2 d).

    The spies are installed with `monkeypatch`, which restores the original
    attributes on the SAME class object. This used to undo itself with
    `importlib.reload(lora_playground.optim)`, which does not restore anything —
    it rebinds every name in the module to a NEW object, so
    `lora_playground.optim.CurvatureWhitenLoRA` stopped being the class that
    other already-imported modules hold. Nothing noticed until a test elsewhere
    asserted `isinstance(build_optimizer(...), CurvatureWhitenLoRA)`, which then
    failed only when this file ran first in the same session.
    """
    calls = _spy_on_expensive_primitives(monkeypatch)
    m, x, tgt = _make(seed=17)
    opt = CurvatureWhitenLoRA(m, precond="one-sided", msign="diag", **_KW)
    _run(m, opt, x, tgt, steps=2)
    assert not calls, f"the cheap branch still called {sorted(set(calls))}"


def test_the_expensive_primitive_spy_is_not_vacuous(monkeypatch):
    """Known-positive control for the spy the test above relies on.

    Without this, that test passes for two uninteresting reasons as easily as
    the interesting one: a watched name that does not resolve, or a call site
    the patch does not reach. Both had happened — it watched
    `_newton_schulz_batched` as a CurvatureWhitenLoRA attribute, but that is a
    module-level function, so `getattr` returned None and the name was silently
    skipped; and it never watched `gram_ns_inv_sqrt`, which is the r x r inverse
    square root the cheap branch is supposed to avoid. So its headline claim,
    "no inverse square root anywhere in the direction", was the one thing it did
    not check.
    """
    calls = _spy_on_expensive_primitives(monkeypatch)
    m, x, tgt = _make(seed=17)
    opt = CurvatureWhitenLoRA(m, precond="product", msign="full", **_KW)
    _run(m, opt, x, tgt, steps=2)
    assert "gram_ns_inv_sqrt" in calls, \
        f"the r x r inverse sqrt spy never fired on the expensive branch: {sorted(set(calls))}"
    assert "_polar_ns_guarded" in calls, \
        f"the polar spy never fired on the expensive branch: {sorted(set(calls))}"


def test_factorwise_diag_skips_dense_slot_inverse_sqrt(monkeypatch):
    calls = _spy_on_expensive_primitives(monkeypatch)
    m, x, tgt = _make(seed=23)
    opt = CurvatureWhitenLoRA(m, precond="factorwise-diag", **_KW)
    _run(m, opt, x, tgt, steps=2)
    assert "gram_ns_inv_sqrt" not in calls
    assert "_polar_ns_guarded" in calls


@pytest.mark.parametrize("precond", sorted(PRECOND_CHOICES))
def test_each_precond_branch_is_companion_independent(precond, companion_independent):
    """Every `precond` branch must give a pair the same update whichever shape
    group it lands in.

    Replaces a grouped-vs-per-pair comparison at `precond="one-sided"`, which
    was VACUOUS: one-sided pins the whitener to the identity in both paths, so
    the comparison never reached the code the two implementations differed on.
    It also ran only at `precond_method="eigh"`. This runs all three branches at
    the production `gram_ns`, bit-exactly.
    """
    companion_independent(lr=3e-2, curvature_beta=0.99, ns_steps=8, delta=1e-4,
                          cw_nesterov=True, kl_coupled=True, soap_v=False,
                          use_polar=True, precond_method="gram_ns",
                          precond=precond)


@pytest.mark.parametrize("msign", sorted(MSIGN_CHOICES))
def test_precond_and_msign_reach_the_optimizer_through_build_optimizer(msign):
    """Both must survive the spec-forwarding layer, not just the constructor:
    `optim_specs` skip-sets decide which kwargs reach the class at all, and a
    dropped one fails silently at the default."""
    m, _, _ = _make()
    opt = build_optimizer(m, "kl-diag-polar-lora", lr=3e-2, curvature_beta=0.99,
                          muon_ns_steps=8, precond_delta=1e-4, cw_nesterov=True,
                          precond_method="gram_ns", precond="one-sided", msign=msign)
    assert opt.precond == "one-sided"
    assert opt.msign == msign
    assert opt.rr_identity is True
