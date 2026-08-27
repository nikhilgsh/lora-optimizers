"""``CurvatureWhitenLoRA`` against Algorithm 1 of ``paper/manuscript/main.tex``.

The manuscript is the specification; this class is the implementation, and the
class docstring says so ("CANONICAL SPEC: paper/manuscript/main.tex Algorithm 1
(`alg:ours`)"). Nothing else in the suite compares the two. The existing tests
pin the code against ITSELF -- branch A reproduces branch B bitwise, the
grouped path agrees with the per-pair path, a flag is not a no-op -- all of
which stay green while both sides drift away from the paper together.

So each test here re-derives one line of Alg 1 (main.tex:413-442) or one
constant of its appendices from the manuscript text and asserts the optimizer
matches. A future edit that changes the update math without changing the paper
fails here rather than silently invalidating a published number.

What is pinned, by manuscript line:

  main.tex:684 / app:init      eps = 1e-12, and it stays SEPARATE from the 1e-8
                               that serves three other roles in this class.
  Alg 1 line 423 (ln:mom)      the Nesterov look-ahead, on by default.
  app:damping (1118-1128)      Chat = C + max(delta*lambda_max(C), eps) I caps
                               cond(Chat) at ~1/delta.
  Alg 1 line 428 (ln:pqnorm)   P, Q are infinity-norm normalised, so the
                               preconditioner is invariant to the scale of p, q.
  Alg 1 lines 435-436          rho = eta/(||A||_2 + ||B||_2), and the applied
    (ln:rho, ln:spectralnorm-  step is rescaled to spectral norm exactly rho.
     update)
  Alg 1 lines 439-440          the p, q updates carry the 1/r normaliser.
    (ln:qmom, ln:pmom)

Everything runs on CPU in fp32 on an r=8, d_in=64, d_out=48 pair -- these are
statements about the code path, not about shapes or hardware.

The arm under test is the paper protagonist: ``kl-diag-polar-lora`` at the
production settings recorded in ``lora_playground/plotting/arms.py:209``
(``CW_PRODUCTION``) and ``lora_playground/bench/config.py:protagonist_config``
-- kl_coupled=True, soap_v=False, diag_metric=True (precond="product"),
use_polar=True, cw_nesterov=True, precond_method="gram_ns",
polar_method="polar_express", delta=1e-4, 8 Gram-NS iterations, 8 polar steps,
beta1=0.9, curvature_beta=0.99 (main.tex:678-685).
"""
import math

import pytest
import torch
import torch.nn as nn

import lora_playground.optim as O
from lora_playground.optim import (
    RR_FREE_INIT_EPS, CurvatureWhitenLoRA, gram_ns_inv_sqrt,
)

R, D_IN, D_OUT = 8, 64, 48

# The protagonist's production configuration. `delta` is `--precond_delta`,
# `ns_steps` is `--muon_ns_steps` and `higham_iters` is `--higham_iters` (the
# Gram-NS iteration count for the r x r inverse square root, main.tex:681).
PROTO = dict(
    lr=3e-2, betas=(0.9, 0.999), curvature_beta=0.99, delta=1e-4,
    kl_coupled=True, soap_v=False, diag_metric=True, use_polar=True,
    cw_nesterov=True, precond_method="gram_ns", polar_method="polar_express",
    ns_steps=8, higham_iters=8,
)


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        # B != 0 on purpose: with the paper's B=0 init, app:init says G_A, Mhat_A
        # and C_B all vanish so D_A = 0 and A does not move (main.tex:1112-1114).
        # That is the correct behaviour but it makes the magnitude rule vacuous on
        # the A side, so these tests start B off zero to exercise both factors.
        nn.init.normal_(self.lora_B["default"].weight, std=0.1)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


def _make(seed=0):
    torch.manual_seed(seed)
    model = _FakeLoRALinear(D_IN, D_OUT, R)
    return model, torch.randn(16, D_IN), torch.randn(16, D_OUT)


def _factors(model):
    return model.lora_A["default"].weight, model.lora_B["default"].weight


def _backward(model, x, tgt, opt):
    opt.zero_grad(set_to_none=False)
    ((model(x) - tgt) ** 2).mean().backward()


def _smax(M):
    return torch.linalg.matrix_norm(M.detach().float(), ord=2).item()


@pytest.fixture(autouse=True)
def _quiet_diagnostics(monkeypatch):
    """`log_basic_diagnostics=True` prints one JSONL line per step. The tests
    below turn it on only to reach `_cw_diag_record`'s arguments, so drop the
    emitter."""
    monkeypatch.setattr(O, "_emit_optim_diagnostics", lambda *a, **k: None)


# ─── main.tex:684 / app:init — eps = 1e-12, and it is not the other 1e-8 ──────

# The attribute holding Alg 1 line 436's clamp. ONE name, deliberately.
#
# The IDENTIFIER is not the paper contract -- the value, and its separateness
# from `eps`, are -- so a deliberate rename is meant to be a one-line edit here
# rather than a test failure. But the list must never carry a name that denotes
# a DIFFERENT quantity. It previously also accepted `sigma_floor`, which is
# spectral.py's data-dependent degenerate-vector floor and is called out at
# optim.py:670 as distinct from this constant. Had `alg1_magnitude_floor` been
# deleted while some `sigma_floor` attribute existed, the test would have gone
# on passing against the wrong number -- exactly the substitution it exists to
# catch. Verified on a live CurvatureWhitenLoRA: `alg1_magnitude_floor` = 1e-12,
# `sigma_floor` absent.
_FLOOR_ATTR = "alg1_magnitude_floor"


def _floor_attr(opt):
    assert hasattr(opt, _FLOOR_ATTR), (
        f"the optimizer has no `{_FLOOR_ATTR}`. Alg 1 line 436's magnitude clamp "
        "must keep its own constant, separate from `eps` -- if it was renamed, "
        "point _FLOOR_ATTR at the new name (one name, and it must denote THIS "
        "constant, not a similarly-spelled floor)."
    )
    return _FLOOR_ATTR


def test_magnitude_clamp_constant_is_1e_12_and_distinct_from_eps():
    """main.tex:684 states the numerical-stability constant is eps = 1e-12, and
    Alg 1 line 436 (ln:spectralnormupdate) is the only place Alg 1 spends it:

        A <- A - rho * D_A / max(||D_A||_2, eps)

    `self.eps` is a DIFFERENT constant. It is 1e-8 and it serves three roles the
    manuscript never assigns to Alg 1's eps: the SOAP Adam denominator, the
    relative floor inside `_direction_op`, and the quadratic clamp in
    `cw_solved_rho`. 1e-8 is right for those and wrong for line 436. One name
    cannot carry both values, which is why the clamp has its own -- so the point
    of this test is that a later cleanup must not re-merge them.
    """
    model, _, _ = _make()
    opt = CurvatureWhitenLoRA(model, **PROTO)
    floor = getattr(opt, _floor_attr(opt))
    assert floor == 1e-12
    assert opt.eps == 1e-8
    assert floor != opt.eps


def test_magnitude_clamp_is_applied_from_the_alg1_floor_not_eps():
    """The constant existing is not the constant being SPENT at Alg 1 line 436.

    Drive ||D_A||_2 and ||D_B||_2 to exactly zero (patch the sigma_max estimator
    at the two rescale call sites), which forces the clamp to be the whole
    denominator: dA = -(c_A * rho / floor) * W_A. Then |dA| is inversely
    proportional to the floor, so shifting the exponent scales the step by a known
    factor. Running the identical seed at floor 1e-12 and floor 1e-8 must give a
    ratio of exactly 1e4; if line 436 read `self.eps` instead, mutating the Alg 1
    floor would change nothing and the ratio would be 1.
    """
    def one_step(floor):
        model, x, tgt = _make(seed=3)
        opt = CurvatureWhitenLoRA(model, **PROTO)
        if floor is not None:
            setattr(opt, _floor_attr(opt), floor)
        inner = opt._smax_warm

        def zero_at_rescale(M, states, key, n_warm=3):
            s = inner(M, states, key, n_warm)   # still caches v_init in state
            return torch.zeros_like(s) if key in ("v_sigma_WA", "v_sigma_WB") else s

        opt._smax_warm = zero_at_rescale
        A, _ = _factors(model)
        A_pre = A.detach().clone()
        _backward(model, x, tgt, opt)
        opt.step()
        return (A.detach() - A_pre).norm().item()

    shipped = one_step(None)          # 1e-12
    mutated = one_step(1e-8)
    assert mutated > 0.0
    # 1e4 exactly in exact arithmetic; measured 9999.99954 (5e-8 relative), which
    # is fp32 rounding on a ~5e10-magnitude step, so 1e-5 relative is ~200x the
    # observed residual and still 1e4x away from the "clamp is inert" answer of 1.
    assert shipped / mutated == pytest.approx(1e4, rel=1e-5)


# ─── Alg 1 line 423 (ln:mom) — the Nesterov look-ahead ───────────────────────

def test_nesterov_lookahead_defaults_on_and_is_refused_on_the_soap_path():
    """Alg 1 line 423 is the look-ahead momentum

        Mhat_A <- beta1 * M_A + (1 - beta1) * G_A

    evaluated AFTER line 422 has already written M_A, i.e. the Muon convention.
    It is part of the published algorithm, so leaving `cw_nesterov` unset must
    resolve to True on the closed-form-Shampoo path the manuscript describes.

    `soap_v=True` is a different arm (a SOAP v-hat EMA with no closed-form
    Shampoo core to feed), where the look-ahead is undefined -- so it resolves
    False there, and an EXPLICIT cw_nesterov=True alongside it is a
    misconfiguration that must raise rather than be silently dropped.
    """
    unset = {k: v for k, v in PROTO.items() if k != "cw_nesterov"}
    model, _, _ = _make()
    assert CurvatureWhitenLoRA(model, **unset).cw_nesterov is True

    # The soap_v arm: SOAP v-hat needs the eigenbasis, so it pins precond_method
    # to eigh and turns the kl/diag/polar identity off.
    soap = {**unset, "soap_v": True, "kl_coupled": False, "diag_metric": False,
            "use_polar": False, "precond_method": "eigh"}
    model2, _, _ = _make()
    assert CurvatureWhitenLoRA(model2, **soap).cw_nesterov is False

    model3, _, _ = _make()
    with pytest.raises(ValueError, match="cw_nesterov=True is only defined"):
        CurvatureWhitenLoRA(model3, cw_nesterov=True, **soap)


# ─── app:damping (main.tex:1118-1128) — relative damping caps the cond ───────

def test_relative_damping_caps_the_condition_number_at_one_over_delta():
    """app:damping states each inverse in Alg 1 is damped relative to the top
    eigenvalue,

        Chat = C + max(delta * lambda_max(C), eps) I,

    "which caps the condition number of Chat at (1+delta)/delta ~ 1/delta"
    (main.tex:1125). The optimizer spends that through
    `gram_ns_inv_sqrt(..., eps=delta, eps_relative=True)` -- the exact call the
    gram_ns branch makes on the r x r slots -- so push a badly conditioned r x r
    matrix through THAT call and read the cap off the result.

    The returned X approximates Chat^{-1/2}, so the damped matrix the optimizer
    effectively inverted is (X X)^{-1} and its condition number is the quantity
    the manuscript bounds.
    """
    delta = 1e-4
    cap = (1.0 + delta) / delta          # main.tex:1125, = 10001.0 at delta=1e-4

    # cond exactly 1e12 and exactly representable in fp32 (gram_ns_inv_sqrt's
    # docstring requires fp32: bf16 blows up at the delta floor). A rotated C
    # cannot be used -- forming it in fp32 rounds the 1e-12 eigenvalue away and
    # the input is no longer the conditioning the test claims to feed in.
    ev = torch.logspace(0.0, -12.0, R, dtype=torch.float32)
    C = torch.diag(ev)
    assert (ev.max() / ev.min()).item() == pytest.approx(1e12, rel=1e-6)

    X = gram_ns_inv_sqrt(C.unsqueeze(0), nsteps=8, eps=delta, eps_relative=True)[0]
    cond_damped = torch.linalg.cond(X.double() @ X.double()).item()

    # Tolerance: the Gram-NS inverse square root is itself approximate (~2e-5
    # relative against an exact eigh of the damped matrix at these settings), and
    # the measured cap here is 10000.999 -- 1e-7 below the analytic 10001. 1%
    # is ~500x that residual, and still 3800x tighter than the undamped answer
    # asserted just below, so the test cannot pass by accident.
    assert cond_damped <= cap * 1.01
    # ...and the damping must not be over-applied either: a floor much larger
    # than delta*lambda_max would crush the cap well below 1/delta and silently
    # change the preconditioner the paper describes.
    assert cond_damped >= cap * 0.5

    # Discrimination: with the damping removed the same input comes back at
    # cond ~ 4e7, i.e. the assertion above is doing work.
    X_undamped = gram_ns_inv_sqrt(C.unsqueeze(0), nsteps=8, eps=1e-30,
                                  eps_relative=True)[0]
    cond_undamped = torch.linalg.cond(X_undamped.double() @ X_undamped.double()).item()
    assert cond_undamped > 100.0 * cap


# ─── Alg 1 line 428 (ln:pqnorm) — P, Q are infinity-norm normalised ──────────

def test_preconditioner_is_invariant_to_the_scale_of_p_and_q():
    """Alg 1 line 428 forms

        P = Diag(p / ||p||_inf),   Q = Diag(q / ||q||_inf)

    and main.tex:409 says why: the Kronecker product is invariant to
    (P, Q) -> (aP, a^-1 Q), so the EMA updates are made invariant to it too by
    normalising each factor to largest entry one.

    `_rdinv` variant "A" -- the shipped/paper variant -- implements the damped
    inverse square root of that normalised diagonal, (x/x_max + delta)^{-1/2}.
    Scale invariance is exactly what the normalisation buys, so scaling q by any
    positive constant must leave the returned factor unchanged.
    """
    model, _, _ = _make()
    opt = CurvatureWhitenLoRA(model, **PROTO)
    assert opt.rdinv_variant == "A"

    q = torch.rand(D_IN, dtype=torch.float32).abs() + 1e-3
    base = opt._rdinv(q)
    for c in (1e-6, 1e-3, 1.0, 7.0, 1e6):
        # fp32 division rounding only; measured max |delta| <= 2.4e-7 over this
        # 12-decade span, against a factor whose own magnitude is ~1.
        assert torch.allclose(opt._rdinv(q * c), base, atol=1e-6, rtol=0), \
            f"the preconditioner moved when q was scaled by {c}"

    # Discrimination: variant "B" is the same damping floor in the RAW gauge,
    # (x + delta*x_max)^{-1/2}, which carries a c^{-1/2}. If the assertion above
    # were vacuous it would pass here too.
    model_b, _, _ = _make()
    opt_b = CurvatureWhitenLoRA(model_b, **{**PROTO, "rdinv_variant": "B"})
    assert not torch.allclose(opt_b._rdinv(q * 100.0), opt_b._rdinv(q),
                              atol=1e-6, rtol=0)


# ─── Alg 1 lines 435-436 (ln:rho, ln:spectralnormupdate) — the magnitude rule ─

def test_applied_step_has_spectral_norm_exactly_rho():
    """Alg 1 line 435 sets rho = eta / (||A||_2 + ||B||_2) and line 436 divides
    the direction by max(||D||_2, eps), so the APPLIED step's spectral norm is
    rho by construction, on both factors. Both halves are asserted:

      (a) sigma_max(dA) / rho == 1 and sigma_max(dB) / rho == 1 (line 436), and
      (b) rho == eta / (sigma_max(A) + sigma_max(B)) computed by an exact SVD on
          the pre-step factors (line 435).

    NOTE the repo's `finite_step_spectral_norm` diagnostic is NOT this quantity
    -- `_finite_step_product_diagnostics` defines it as sigma_max(dB @ dA), the
    second-order term of the product update. Do not substitute one for the other.

    The optimizer estimates every spectral norm by warm-started power iteration
    floored at the max row norm (app:smax, main.tex:1130-1150), so both ratios
    approach 1 from above over the first few steps as the cached start vectors
    converge; this asserts on the settled step.
    """
    model, x, tgt = _make(seed=0)
    opt = CurvatureWhitenLoRA(model, log_basic_diagnostics=True,
                              diagnostics_every=1, **PROTO)
    seen = []
    inner = opt._cw_diag_record

    def capture(**kw):
        seen.append(kw)
        return inner(**kw)

    opt._cw_diag_record = capture

    for _ in range(8):
        _backward(model, x, tgt, opt)
        opt.step()

    rec = seen[-1]
    rho = float(rec["rho"])
    assert rho > 0.0
    # (b) line 435, against an exact SVD rather than the optimizer's estimator.
    rho_paper = PROTO["lr"] / (_smax(rec["A_pre"]) + _smax(rec["B_pre"]))
    assert rho / rho_paper == pytest.approx(1.0, rel=1e-4)
    # (a) line 436. Measured worst case over three seeds at this step is 1.6e-5;
    # 1e-4 leaves ~6x headroom for the power-iteration residual while a dropped
    # rescale would land at sigma_max(W) ~ 13, i.e. off by 1000x.
    assert _smax(rec["dA"]) / rho == pytest.approx(1.0, rel=1e-4)
    assert _smax(rec["dB"]) / rho == pytest.approx(1.0, rel=1e-4)


# ─── Alg 1 lines 439-440 (ln:qmom, ln:pmom) — the 1/r normaliser ─────────────

def test_p_and_q_updates_carry_the_one_over_r_normaliser():
    """Alg 1 lines 439-440 are

        q <- beta2 * q + (1 - beta2) * diag(G_A^T C_B^-1 G_A) / r
        p <- beta2 * p + (1 - beta2) * diag(G_B C_A^-1 G_B^T) / r

    with r the LoRA rank -- the same 1/r in both, on the coupled estimator. This
    mirrors the whole first step in closed form and compares.

    Step one is chosen because it is the only step whose inputs are all known
    from the manuscript: cw_metric_init="1e-12" fills p = q = eps*1 (app:init,
    main.tex:1110), so the infinity-norm normalisation of line 428 maps both to
    the identity metric and every damped diagonal is the single constant
    (1 + delta)^{-1/2}. C_B = B^T P B and C_A = A Q A^T then follow from the
    initial factors, and their inverses come from the same `gram_ns_inv_sqrt`
    call the optimizer makes.
    """
    model, x, tgt = _make(seed=3)
    opt = CurvatureWhitenLoRA(model, **PROTO)
    A, B = _factors(model)
    st = opt.pair_state[0]

    q0, p0 = st["D_in"].clone(), st["D_out"].clone()
    assert torch.allclose(q0, torch.full_like(q0, 1e-12))
    assert torch.allclose(p0, torch.full_like(p0, 1e-12))

    _backward(model, x, tgt, opt)
    gA = A.grad.detach().float().clone()
    gB = B.grad.detach().float().clone()
    Af = A.detach().float().clone()
    Bf = B.detach().float().clone()
    opt.step()

    b2, delta = opt.curvature_beta, opt.delta
    # Line 428: P, Q from the normalised p, q. Their damped inverse squares are
    # what the metric multiplies by, so the metric itself is the reciprocal.
    Qh, Ph = opt._rdinv(q0), opt._rdinv(p0)
    Q_m, P_m = (Qh * Qh).reciprocal(), (Ph * Ph).reciprocal()
    # Line 429: C_B = B^T P B, C_A = A Q A^T.
    C_B = Bf.T @ (P_m.unsqueeze(-1) * Bf)
    C_A = (Af * Q_m.unsqueeze(0)) @ Af.T
    C_B_inv_half = gram_ns_inv_sqrt(0.5 * (C_B + C_B.T), nsteps=PROTO["higham_iters"],
                                    eps=delta, eps_relative=True)
    C_A_inv_half = gram_ns_inv_sqrt(0.5 * (C_A + C_A.T), nsteps=PROTO["higham_iters"],
                                    eps=delta, eps_relative=True)
    C_B_inv = C_B_inv_half @ C_B_inv_half
    C_A_inv = C_A_inv_half @ C_A_inv_half

    want_q = b2 * q0 + (1.0 - b2) / R * (gA * (C_B_inv @ gA)).sum(dim=0)
    want_p = b2 * p0 + (1.0 - b2) / R * (gB * (gB @ C_A_inv)).sum(dim=1)
    # Measured relative residual ~7e-8 (fp32 on a chain of r x r matmuls).
    assert st["D_in"] == pytest.approx(want_q, rel=1e-5, abs=0.0)
    assert st["D_out"] == pytest.approx(want_p, rel=1e-5, abs=0.0)

    # Discrimination: r=8 against d_in=64 is a factor of 8, so a wrong
    # normaliser is nowhere near this tolerance.
    wrong = b2 * q0 + (1.0 - b2) / D_IN * (gA * (C_B_inv @ gA)).sum(dim=0)
    assert not torch.allclose(st["D_in"], wrong, rtol=1e-3, atol=0.0)


def test_factorwise_recursion_normalises_by_d_in_and_d_out():
    """The `precond="factorwise"` branch replaces the product slots with r x r
    EMAs fitted from the factor gradients themselves,

        P_A <- beta2 P_A + (1 - beta2)/d_in  * G_A Q^-1 G_A^T
        U_B <- beta2 U_B + (1 - beta2)/d_out * G_B^T P^-1 G_B

    THIS RECURSION HAS NO COUNTERPART IN THE MANUSCRIPT. Alg 1 has one pair of
    EMAs, the p, q of lines 439-440, and they are normalised by r; there is no
    published equation these two lines implement, and the branch is an internal
    ablation (`kl-shampoo-polar-lora` and `--precond factorwise`). The test
    exists to record that gap explicitly: if this ever becomes a paper arm, an
    equation has to be added to Alg 1 and this normaliser checked against it.
    Until then it pins the /d_in and /d_out that the ablation's recorded runs
    were produced with, so the arm keeps its meaning.

    Step one again: the P_A / Q_B buffers start at `RR_FREE_INIT_EPS * I`, which
    is 1e-12 against a gradient Gram of order 1, so the beta2 * prev term is
    numerically negligible and the normaliser is the only thing left to get
    wrong. It is carried in `want_PA`/`want_QB` regardless rather than assumed
    away.
    """
    model, x, tgt = _make(seed=3)
    opt = CurvatureWhitenLoRA(model, precond="factorwise", **PROTO)
    assert opt.precond == "factorwise" and opt.diag_metric is False
    A, B = _factors(model)
    st = opt.pair_state[0]

    q0, p0 = st["D_in"].clone(), st["D_out"].clone()
    PA0, QB0 = st["P_A"].clone(), st["Q_B"].clone()
    # The free r x r factors start at eps * I, not zero: the lambda_max
    # normalization at consume maps that to exactly the identity metric on step
    # 1 (the analogue of the paper's p = q = eps * 1, `app:init`) while leaving
    # no statistical prior. A raw identity would instead inject an O(1) prior
    # decaying only as beta2^t.
    eye_r = torch.eye(R, dtype=PA0.dtype)
    assert PA0 == pytest.approx(RR_FREE_INIT_EPS * eye_r, rel=0.0, abs=1e-20)
    assert QB0 == pytest.approx(RR_FREE_INIT_EPS * eye_r, rel=0.0, abs=1e-20)

    _backward(model, x, tgt, opt)
    gA = A.grad.detach().float().clone()
    gB = B.grad.detach().float().clone()
    opt.step()

    b2 = opt.curvature_beta
    Q_inv = opt._rdinv(q0) ** 2          # Q^-1 on the d_in side
    P_inv = opt._rdinv(p0) ** 2          # P^-1 on the d_out side
    want_PA = b2 * PA0 + (1.0 - b2) / D_IN * ((gA * Q_inv.unsqueeze(0)) @ gA.T)
    want_QB = b2 * QB0 + (1.0 - b2) / D_OUT * (gB.T @ (gB * P_inv.unsqueeze(-1)))
    assert st["P_A"] == pytest.approx(want_PA, rel=1e-5, abs=0.0)
    assert st["Q_B"] == pytest.approx(want_QB, rel=1e-5, abs=0.0)

    # Discrimination: Alg 1's own /r would be off by d_in/r = 8 here.
    wrong = b2 * PA0 + (1.0 - b2) / R * ((gA * Q_inv.unsqueeze(0)) @ gA.T)
    assert not torch.allclose(st["P_A"], wrong, rtol=1e-3, atol=0.0)
    assert math.isclose(D_IN / R, 8.0)
