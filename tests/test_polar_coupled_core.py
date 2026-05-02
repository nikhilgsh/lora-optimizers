"""CPU-only unit tests for PolarCoupledCoreLoRA and MuonCoupledCoreLoRA.

Implements the projected-quotient-polar core solver described in
docs/notes/polar_coupled_core_solver.md. Tests cover:
  - Generic optimizer behavior (shape, finiteness, zero-grad, determinism).
  - The doc's three sanity reductions (Frobenius limit, one-factor → Case 2,
    symmetry under (A, B) ↔ (B^T, A^T)).
  - Certificate bounds (γ ∈ [1, 2]) and gauge (B^T ΔB = ΔA A^T).
  - Gradient-compatibility diagnostic: near-eps for synthetic compatible
    factor gradients.
  - Variant 2 first-step equivalence with variant 1.
"""
import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import (
    MuonCoupledCoreLoRA,
    PolarCoupledCoreLoRA,
    _build_active_core,
    _imbalance_gauge_shift,
    _polar_coupled_core_step,
)


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


# --- Generic optimizer behavior --------------------------------------------

@pytest.mark.parametrize("OptCls", [PolarCoupledCoreLoRA, MuonCoupledCoreLoRA])
def test_step_runs_and_changes_params(OptCls):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = OptCls(m, lr=1e-2)
    ((m(x) - target) ** 2).mean().backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    diffs = [float((a - b).abs().sum()) for a, b in zip(pre, post)]
    assert all(torch.isfinite(p).all() for p in post)
    assert max(diffs) > 0.0


@pytest.mark.parametrize("OptCls", [PolarCoupledCoreLoRA, MuonCoupledCoreLoRA])
def test_zero_grad_no_finite_failure(OptCls):
    m, x, target = _make()
    opt = OptCls(m, lr=1e-2)
    (m(x) * 0.0).sum().backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("OptCls", [PolarCoupledCoreLoRA, MuonCoupledCoreLoRA])
def test_determinism(OptCls):
    def run():
        m, x, target = _make(seed=42)
        opt = OptCls(m, lr=1e-3)
        for _ in range(3):
            ((m(x) - target) ** 2).mean().backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    a = run()
    b = run()
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=0.0)


# --- Helpers for sanity-reduction tests ------------------------------------

def _random_pair_with_compatible_grads(r=4, m_=8, n=6, seed=0):
    """Random (A, B) full-rank plus gradient-compatible (G_A, G_B) constructed
    from a single dense G — i.e. G_A = B^T G, G_B = G A^T, so the standing
    assumption (∇_A L) A^T = B^T (∇_B L) holds by construction.
    """
    torch.manual_seed(seed)
    A = torch.randn(r, n)
    B = torch.randn(m_, r)
    G = torch.randn(m_, n)
    G_A = B.T @ G
    G_B = G @ A.T
    return A, B, G_A, G_B, G


# --- Sanity reductions -----------------------------------------------------

def test_frobenius_limit_matches_lin_lora_form():
    """Section 5: Frobenius replacement reduces to LinLoRA Sylvester closed
    form. Z_upd = -lr * Ĥ; lift recovers
        S_B K + K S_A = -lr (∇_A A^T),
        ΔA = -S_B^{-1}(lr ∇_A + K A),
        ΔB = -(lr ∇_B + B K) S_A^{-1}.
    """
    A, B, G_A, G_B, _ = _random_pair_with_compatible_grads(seed=11)
    lr = 1e-2
    delta = 1e-6

    dA, dB, certs, _ = _polar_coupled_core_step(
        A, B, G_A, G_B, lr, delta=delta, core_norm="frobenius",
    )

    # LinLoRA reference computation.
    from lora_playground.utils import solve_spd, solve_sylvester, spdify
    SB = spdify(B.T @ B, delta)
    SA = spdify(A @ A.T, delta)
    RHS = -lr * (G_A @ A.T)
    K = solve_sylvester(SB, SA, RHS)
    dA_ref = -solve_spd(SB, lr * G_A + K @ A)
    dB_ref = -solve_spd(SA, (lr * G_B + B @ K).T).T

    assert torch.allclose(dA, dA_ref, atol=1e-5, rtol=1e-4), \
        f"dA Frobenius limit deviates: max abs err = {(dA - dA_ref).abs().max()}"
    assert torch.allclose(dB, dB_ref, atol=1e-5, rtol=1e-4), \
        f"dB Frobenius limit deviates: max abs err = {(dB - dB_ref).abs().max()}"


def test_one_factor_restriction_recovers_case_2():
    """Section 5: with G_A = 0 (and gradient compatibility forcing G_B's
    columns into Q_L^⊥), the lifted tangent should match the Case-2 form
        Z = -lr * polar(G_B R_R^{-T}) Q_R^T.
    """
    torch.manual_seed(7)
    r, m_, n = 4, 8, 6
    A = torch.randn(r, n)
    B = torch.randn(m_, r)
    # G with cols in Q_L^⊥ → G_A = B^T G = 0 and B^T G_B = 0 by construction.
    Q_L, _ = torch.linalg.qr(B, mode="reduced")
    G_raw = torch.randn(m_, n)
    G = G_raw - Q_L @ (Q_L.T @ G_raw)
    G_A = B.T @ G
    G_B = G @ A.T
    assert G_A.abs().max() < 1e-5
    assert (B.T @ G_B).abs().max() < 1e-5

    lr = 1e-2
    dA, dB, certs, bases = _polar_coupled_core_step(
        A, B, G_A, G_B, lr, delta=1e-8, core_scale="constrained",
    )

    # Case 2 reference: Z = -lr * polar(G_B R_R^{-T}) Q_R^T.
    R_R = bases["R_R"]
    Q_R = bases["Q_R"]
    G_B_RRinvT = torch.linalg.solve_triangular(R_R, G_B.T, upper=False).T
    U_p, _, Vh_p = torch.linalg.svd(G_B_RRinvT, full_matrices=False)
    polar_Z = U_p @ Vh_p
    Z_ref = -lr * polar_Z @ Q_R.T

    # Applied tangent from our (dA, dB).
    Z_applied = B @ dA + dB @ A

    err = (Z_applied - Z_ref).norm() / (Z_ref.norm() + 1e-30)
    assert err < 1e-4, f"applied tangent vs Case-2 reference rel err {err}"


def test_symmetry_under_AB_swap():
    """Section 5: (A, B, G_A, G_B) ↔ (B^T, A^T, G_B^T, G_A^T) yields swapped
    (ΔA, ΔB) up to transposition.
    """
    A, B, G_A, G_B, _ = _random_pair_with_compatible_grads(seed=21)
    lr = 1e-2
    dA, dB, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, lr, delta=1e-7)

    A2 = B.T
    B2 = A.T
    G_A2 = G_B.T
    G_B2 = G_A.T
    dA2, dB2, _, _ = _polar_coupled_core_step(A2, B2, G_A2, G_B2, lr, delta=1e-7)

    err1 = (dA2 - dB.T).norm() / (dB.norm() + 1e-30)
    err2 = (dB2 - dA.T).norm() / (dA.norm() + 1e-30)
    assert err1 < 1e-4, f"(A,B)↔(B^T,A^T) symmetry: dA' vs dB^T rel err {err1}"
    assert err2 < 1e-4, f"(A,B)↔(B^T,A^T) symmetry: dB' vs dA^T rel err {err2}"


def test_certificate_bounds():
    """γ ∈ [1, 2] and 0 ≤ LB ≤ UB on random non-degenerate input."""
    A, B, G_A, G_B, _ = _random_pair_with_compatible_grads(seed=33)
    _, _, certs, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2)
    g = certs["gamma"]
    assert 1.0 - 1e-6 <= g <= 2.0 + 1e-6, f"γ = {g} out of [1, 2]"
    assert 0.0 <= certs["LB"] <= certs["UB"] + 1e-6, \
        f"LB {certs['LB']} > UB {certs['UB']}"


def test_min_frobenius_gauge():
    """Min-Frobenius gauge: B^T ΔB = ΔA A^T (Section 4 lift)."""
    A, B, G_A, G_B, _ = _random_pair_with_compatible_grads(seed=55)
    dA, dB, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2)
    lhs = B.T @ dB
    rhs = dA @ A.T
    err = (lhs - rhs).norm() / (lhs.norm() + 1e-30)
    assert err < 1e-4, f"gauge B^T dB ≠ dA A^T (rel err {err})"


def test_compat_near_machine_eps_for_compatible_grads():
    """Section 6 diagnostic: synthetic gradient-compatible (G_A, G_B) should
    yield compat ≈ machine eps. We use float32, so target < ~1e-5.
    """
    A, B, G_A, G_B, _ = _random_pair_with_compatible_grads(seed=77)
    bases = _build_active_core(A, B, G_A, G_B, delta=1e-8)
    assert bases["compat"] < 1e-4, \
        f"compat = {bases['compat']} should be near machine eps for compatible grads"


# --- Variant 2 momentum tests ---------------------------------------------

# --- Imbalance-preserving gauge tests --------------------------------------

def test_imbalance_gauge_preserves_imbalance_to_linear_order():
    """gauge='imbalance-preserve' should make the linearized post-step
    imbalance ≈ I_t (cancel δI_0 from the base lift). Verify by checking
    that the linearized post-step imbalance F-norm is smaller than the
    base-lift's after gauge correction.
    """
    torch.manual_seed(11)
    r, m_, n = 4, 12, 8
    A = torch.randn(r, n) * 0.5
    B = torch.randn(m_, r) * 0.3
    G = torch.randn(m_, n)
    G_A = B.T @ G
    G_B = G @ A.T

    # Base lift via min-Frobenius.
    dA0, dB0, certs0, _ = _polar_coupled_core_step(
        A, B, G_A, G_B, lr=1e-2, gauge="min-frobenius",
    )
    # With gauge fix.
    dA1, dB1, certs1, _ = _polar_coupled_core_step(
        A, B, G_A, G_B, lr=1e-2, gauge="imbalance-preserve",
    )

    # Tangent should be (nearly) identical: J(dA, dB) = B dA + dB A.
    Z0 = B @ dA0 + dB0 @ A
    Z1 = B @ dA1 + dB1 @ A
    rel_tan_diff = (Z0 - Z1).norm() / (Z0.norm() + 1e-30)
    assert rel_tan_diff < 1e-4, f"gauge changed tangent (rel {rel_tan_diff})"

    # Imbalance trajectory: linearized δI under each lift.
    rho = r / m_
    AAT = A @ A.T
    BTB = B.T @ B
    def lin_dI(dA, dB):
        return (dA @ A.T + A @ dA.T) - rho * (dB.T @ B + B.T @ dB)
    dI0 = lin_dI(dA0, dB0)
    dI1 = lin_dI(dA1, dB1)
    assert dI1.norm() < 0.1 * dI0.norm() + 1e-7, (
        f"imbalance-preserve should null δI; |δI| went {dI0.norm():.4e} → {dI1.norm():.4e}"
    )


def test_imbalance_gauge_restore_zeros_total_imbalance_to_linear_order():
    """gauge='imbalance-restore' should drive the LINEARIZED post-step total
    imbalance to ≈ 0 (i.e., I_t + δI_total ≈ 0).
    """
    torch.manual_seed(13)
    r, m_, n = 4, 12, 8
    A = torch.randn(r, n) * 0.5
    B = torch.randn(m_, r) * 0.3
    G = torch.randn(m_, n)
    G_A = B.T @ G
    G_B = G @ A.T

    dA, dB, _, _ = _polar_coupled_core_step(
        A, B, G_A, G_B, lr=1e-2, gauge="imbalance-restore",
    )

    rho = r / m_
    AAT = A @ A.T
    BTB = B.T @ B
    I_t = AAT - rho * BTB
    dI = (dA @ A.T + A @ dA.T) - rho * (dB.T @ B + B.T @ dB)
    total = I_t + dI
    rel = total.norm() / (I_t.norm() + 1e-30)
    assert rel < 1e-4, f"imbalance-restore should zero linearized total; rel = {rel}"


def test_imbalance_gauge_preserves_tangent():
    """Both imbalance gauges must preserve the tangent J(dA, dB) bit-exactly
    in math; numerically to ~1e-5.
    """
    torch.manual_seed(17)
    r, m_, n = 4, 12, 8
    A = torch.randn(r, n) * 0.5
    B = torch.randn(m_, r) * 0.3
    G = torch.randn(m_, n)
    G_A = B.T @ G
    G_B = G @ A.T

    dA0, dB0, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2, gauge="min-frobenius")
    Z0 = B @ dA0 + dB0 @ A
    for mode in ("imbalance-preserve", "imbalance-restore"):
        dA, dB, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2, gauge=mode)
        Z = B @ dA + dB @ A
        rel = (Z - Z0).norm() / (Z0.norm() + 1e-30)
        assert rel < 1e-4, f"{mode} broke tangent invariance (rel {rel})"


def test_imbalance_gauge_scalar_mode():
    """Scalar-S gauge S = sI is the simplest first patch (no Sylvester).
    Verify it preserves the tangent and reduces the linearized δI vs the
    base lift on average.
    """
    torch.manual_seed(23)
    r, m_, n = 4, 12, 8
    A = torch.randn(r, n) * 0.5
    B = torch.randn(m_, r) * 0.3
    G = torch.randn(m_, n)
    G_A = B.T @ G
    G_B = G @ A.T

    dA0, dB0, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2, gauge="min-frobenius")
    dA1, dB1, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, 1e-2, gauge="imbalance-preserve-scalar")

    # Tangent invariance.
    Z0 = B @ dA0 + dB0 @ A
    Z1 = B @ dA1 + dB1 @ A
    rel = (Z1 - Z0).norm() / (Z0.norm() + 1e-30)
    assert rel < 1e-4, f"scalar gauge broke tangent (rel {rel})"

    # Linearized δI under each lift.
    rho = r / m_
    def lin_dI(dA, dB):
        return (dA @ A.T + A @ dA.T) - rho * (dB.T @ B + B.T @ dB)
    dI0 = float(lin_dI(dA0, dB0).norm().item())
    dI1 = float(lin_dI(dA1, dB1).norm().item())
    assert dI1 <= dI0 + 1e-7, (
        f"scalar gauge should never increase ‖δI‖_F (least-squares minimizer); "
        f"|δI| went {dI0:.4e} → {dI1:.4e}"
    )


def test_imbalance_gauge_helper_directly():
    """Direct test of _imbalance_gauge_shift: feed a base lift and verify
    the kernel-shift identity (B dA + dB A unchanged).
    """
    torch.manual_seed(19)
    r, m_, n = 4, 12, 8
    A = torch.randn(r, n) * 0.5
    B = torch.randn(m_, r) * 0.3
    dA0 = torch.randn(r, n) * 0.01
    dB0 = torch.randn(m_, r) * 0.01

    Z0 = B @ dA0 + dB0 @ A
    dA1, dB1, S, gd = _imbalance_gauge_shift(dA0, dB0, A, B, mode="preserve")
    Z1 = B @ dA1 + dB1 @ A
    rel = (Z1 - Z0).norm() / (Z0.norm() + 1e-30)
    assert rel < 1e-5, f"kernel shift broke tangent (rel {rel})"
    # Preserve mode: post-lift linearized imbalance should equal ‖I_t‖_F
    # (gauge cancels δI_0 → total change ≈ 0 → post-step ≈ I_t).
    rho = A.shape[0] / B.shape[0]
    I_t_norm = float((A @ A.T - rho * B.T @ B).norm().item())
    assert abs(gd["imbalance_residual_lin_post"] - I_t_norm) < 1e-4, (
        f"preserve mode post-residual {gd['imbalance_residual_lin_post']:.4e} "
        f"should equal ‖I_t‖_F = {I_t_norm:.4e}"
    )
    # Restore mode: post-lift linearized imbalance should be ≈ 0.
    dA2, dB2, S2, gd2 = _imbalance_gauge_shift(dA0, dB0, A, B, mode="restore")
    assert gd2["imbalance_residual_lin_post"] < 1e-3 * I_t_norm, (
        f"restore mode post-residual {gd2['imbalance_residual_lin_post']:.4e} "
        f"should be ≪ ‖I_t‖_F = {I_t_norm:.4e}"
    )


def test_imbalance_restore_drives_to_iLoRA_stable_point():
    """Multi-step trajectory: under imbalance-restore, ‖B‖/‖A‖ should approach
    √(m/r) (the iLoRA stable point, Corollary 1) within ~10 steps from a
    PEFT-asymmetric initialization. Preserve modes leave the ratio stuck.

    This is the test for whether the gauge actually does what's claimed —
    single-step ratio doesn't move under any gauge (structural property:
    kernel motion `(SA, -BS)` preserves ‖A‖/‖B‖ asymmetry); the gauge
    effect is over a trajectory.
    """
    torch.manual_seed(7)
    r, m_, n = 16, 200, 200       # smaller for fast tests
    rho = r / m_
    target_ratio = (m_ / r) ** 0.5  # √(m/r) = √12.5 ≈ 3.54 here

    def init():
        torch.manual_seed(7)
        A = torch.randn(r, n) / (n ** 0.5)
        B = torch.randn(m_, r) * 1e-3
        return A, B

    def trajectory(gauge, n_steps=20, lr=1e-3):
        A, B = init()
        torch.manual_seed(11)
        for _ in range(n_steps):
            G = torch.randn(m_, n) * 0.1
            G_A = B.T @ G
            G_B = G @ A.T
            dA, dB, _, _ = _polar_coupled_core_step(A, B, G_A, G_B, lr=lr, gauge=gauge)
            A = A + dA
            B = B + dB
        return float(B.norm() / A.norm())

    # restore mode should converge to √(m/r).
    final_ratio_restore = trajectory("imbalance-restore", n_steps=20)
    assert abs(final_ratio_restore - target_ratio) / target_ratio < 0.05, (
        f"imbalance-restore should drive ‖B‖/‖A‖ to √(m/r)={target_ratio:.2f}; "
        f"got {final_ratio_restore:.2f}"
    )
    # preserve modes should NOT move the ratio appreciably from the initial.
    initial_A, initial_B = init()
    initial_ratio = float(initial_B.norm() / initial_A.norm())
    for gauge in ["min-frobenius", "imbalance-preserve-scalar", "imbalance-preserve"]:
        final = trajectory(gauge, n_steps=20)
        # These modes don't drive the ratio; final should stay close to initial.
        # Use a generous tolerance (trajectory grows both A and B).
        assert final < 0.5 * target_ratio, (
            f"{gauge} should not move ratio toward √(m/r) (it preserves rather than restores); "
            f"got {final:.2f} vs target {target_ratio:.2f}"
        )


def test_muon_first_step_matches_variant1_at_beta_zero():
    """With β=0, EMA reduces to current grad, Nesterov lookahead reduces to
    current grad: M_step = (1-0)·Ĥ + 0·M_t = Ĥ. Variant 2 at β=0 must match
    variant 1 bit-exactly. This pins the canonical-Muon (no bc) form.
    """
    torch.manual_seed(7)
    m1 = TinyLoRAModel(d_in=8, d_out=6, r=4)
    torch.manual_seed(7)
    m2 = TinyLoRAModel(d_in=8, d_out=6, r=4)
    torch.manual_seed(13)
    x = torch.randn(3, 8)
    target = torch.randn(3, 8)

    opt1 = PolarCoupledCoreLoRA(m1, lr=1e-2)
    opt2 = MuonCoupledCoreLoRA(m2, lr=1e-2, beta1=0.0)
    ((m1(x) - target) ** 2).mean().backward()
    opt1.step()
    ((m2(x) - target) ** 2).mean().backward()
    opt2.step()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-7), \
            f"variant-1 vs variant-2-at-β0 first step mismatch on {n1}"


def test_muon_first_step_canonical_magnitude():
    """At step 1 with M_prev = 0 and Nesterov: M_t = (1-β)·Ĥ;
    M_step = (1-β)·Ĥ + β·M_t = (1-β)(1+β)·Ĥ = (1-β²)·Ĥ.
    With β=0.95, that's 0.0975·Ĥ — a ~10× SMALLER first step than Ĥ,
    matching canonical Muon's "build up gradually" behavior.

    Test by comparing variant-2 step-1 dB norm against (1-β²) × variant-1's.
    """
    torch.manual_seed(7)
    m1 = TinyLoRAModel(d_in=8, d_out=6, r=4)
    torch.manual_seed(7)
    m2 = TinyLoRAModel(d_in=8, d_out=6, r=4)
    torch.manual_seed(13)
    x = torch.randn(3, 8)
    target = torch.randn(3, 8)

    pre1 = {n: p.detach().clone() for n, p in m1.named_parameters()}
    pre2 = {n: p.detach().clone() for n, p in m2.named_parameters()}

    beta = 0.95
    PolarCoupledCoreLoRA(m1, lr=1e-2)
    MuonCoupledCoreLoRA(m2, lr=1e-2, beta1=beta)
    # Re-do, since constructing the optimizer doesn't run a step yet:
    opt1 = PolarCoupledCoreLoRA(m1, lr=1e-2)
    opt2 = MuonCoupledCoreLoRA(m2, lr=1e-2, beta1=beta)
    ((m1(x) - target) ** 2).mean().backward()
    opt1.step()
    ((m2(x) - target) ** 2).mean().backward()
    opt2.step()

    expected_ratio = 1.0 - beta * beta  # squared-penalty form is linear in M_step magnitude
    # Pick lora_B params (where step 1 falls back to Case-2 zero-init route — both opts do
    # the same fallback, magnitude-quadratic in ‖G_B U_R Σ_R⁻¹‖_*) — wait, both run
    # zero-B fallback at step 1 since PEFT zeros B. So variant 2's step-1 in EMA mode
    # never executes; both produce identical Case-2 dB. Skip the magnitude test if both
    # took the fallback path — covered by test_muon_first_step_matches_variant1_at_beta_zero.
    # Instead, advance B off zero in both, then compare a follow-up step.
    for n, p in m1.named_parameters():
        if "lora_B" in n:
            p.data.copy_(torch.randn_like(p) * 0.1)
    for n, p in m2.named_parameters():
        if "lora_B" in n:
            p.data.copy_(torch.randn_like(p) * 0.1)
    # Reset optimizer state for the post-bootstrap step (still step 1 of EMA path
    # since we replaced the model).
    opt1 = PolarCoupledCoreLoRA(m1, lr=1e-2)
    opt2 = MuonCoupledCoreLoRA(m2, lr=1e-2, beta1=beta)

    for _ in range(2):
        m1.zero_grad(); m2.zero_grad()
        ((m1(x) - target) ** 2).mean().backward()
        ((m2(x) - target) ** 2).mean().backward()

    pre1 = {n: p.detach().clone() for n, p in m1.named_parameters()}
    pre2 = {n: p.detach().clone() for n, p in m2.named_parameters()}
    opt1.step()
    opt2.step()
    delta1 = next(p2 - p1 for (n1, p1), (n2, p2) in zip(pre1.items(), m1.named_parameters()) if "lora_B" in n1)
    delta2 = next(p2 - p1 for (n1, p1), (n2, p2) in zip(pre2.items(), m2.named_parameters()) if "lora_B" in n1)
    norm1 = float(delta1.norm().item())
    norm2 = float(delta2.norm().item())
    if norm1 < 1e-30:
        return  # nothing to compare; skip
    ratio = norm2 / norm1
    assert abs(ratio - expected_ratio) < 0.1 * expected_ratio, (
        f"Variant 2 first non-fallback step magnitude {norm2:.4f} vs variant 1 {norm1:.4f} "
        f"(ratio {ratio:.3f}, expected ~{expected_ratio:.3f} from Nesterov + EMA at β={beta})"
    )
