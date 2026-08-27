"""Tests for CurvatureWhitenLoRA (SOAP in an S⊗D curvature basis).

Covers: step runs and changes params (both polar arms), zero-grad finiteness,
determinism, the SOAP-basis/chord-tight update equation, and factory dispatch.
"""
import torch
import torch.nn as nn
import pytest

from lora_playground.optim import CurvatureWhitenLoRA, build_optimizer, _newton_schulz


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


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar)
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params after step"
    assert max(float((a - b).abs().sum()) for a, b in zip(pre, post)) > 0.0, "No params changed"


@pytest.mark.parametrize("use_polar", [False, True])
def test_zero_grad_no_finite_failure(use_polar):
    m, x, target = _make()
    opt = CurvatureWhitenLoRA(m, lr=1e-2, use_polar=use_polar)
    loss = (m(x) * 0.0).sum()
    loss.backward()
    opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param after zero-grad step"


@pytest.mark.parametrize("use_polar", [False, True])
def test_determinism(use_polar):
    def run():
        m, x, target = _make(seed=42)
        opt = CurvatureWhitenLoRA(m, lr=1e-3, use_polar=use_polar)
        for _ in range(3):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pa, pb in zip(run(), run()):
        assert torch.allclose(pa, pb, atol=0.0)


def _top_power_iter_vector(M):
    U, _, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return (U[:, 0] if M.shape[0] <= M.shape[1] else Vh[0]).contiguous()


def _rdinv_like(x, delta):
    xmax = x.amax(dim=-1, keepdim=True)
    out = (x / xmax.clamp_min(1e-30) + delta).rsqrt()
    return torch.where(xmax < 1e-30, torch.ones_like(out), out)


# Bad-warm-start σ_max recovery is now covered by the blessed library's own
# regressions (tests/test_sigma_max_power_iter.py::
# test_batched_warm_start_recovers_when_vector_enters_nullspace and
# ::test_batched_estimate_has_row_norm_floor_for_bad_warm_start). The bespoke
# CurvatureWhitenLoRA._sigma_max_block_guarded it used to exercise was removed
# in favour of the single blessed spectral.sigma_max_power_iter[_batched] path.


@pytest.mark.parametrize("use_polar", [False, True])
def test_curvature_whiten_matches_soap_sxd_chord_tight_formula(use_polar):
    """Pin the requested update equation.

    No-polar:
        z_A = U_A [(Q_Aᵀ m_A) / sqrt(v_A)]
        W_A = S_A^{-1/2} z_A D_in^{-1/2}
    Polar:
        W_A = S_A^{-1/2} polar(z_A) D_in^{-1/2}

    B uses the transposed orientation. The final applied update is the
    chord-tight spectral rescale, σ_max(Δfactor)=ρ.
    """
    torch.manual_seed(0)
    sub = _FakeLoRALinear(3, 4, 2)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = sub
        def forward(self, x):
            return self.l0(x)

    m = _M()
    A = sub.lora_A["default"].weight
    B = sub.lora_B["default"].weight
    A.data.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    B.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]]))
    gA = torch.tensor([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]])
    gB = torch.tensor([[0.25, -1.0], [0.5, 0.75], [-1.25, 0.5], [1.0, -0.5]])

    lr = 1e-2
    delta = 1e-3
    opt = CurvatureWhitenLoRA(
        m, lr=lr, betas=(0.0, 0.0), delta=delta, use_polar=use_polar,
        ns_steps=12, precond_refresh_every=100,
    )
    st = opt.pair_state[0]
    theta_A = torch.tensor(0.37)
    theta_B = torch.tensor(-0.51)
    UA = torch.tensor([
        [torch.cos(theta_A), -torch.sin(theta_A)],
        [torch.sin(theta_A), torch.cos(theta_A)],
    ])
    UB = torch.tensor([
        [torch.cos(theta_B), -torch.sin(theta_B)],
        [torch.sin(theta_B), torch.cos(theta_B)],
    ])
    lam_A = torch.tensor([4.0, 1.0])
    lam_B = torch.tensor([1.0, 16.0])
    st["U_A"].copy_(UA)
    st["U_B"].copy_(UB)
    st["P_A"].copy_(UA @ torch.diag(lam_A) @ UA.T)
    st["Q_B"].copy_(UB @ torch.diag(lam_B) @ UB.T)
    st["D_in"].copy_(torch.tensor([9.0, 1.0, 4.0]))
    st["D_out"].copy_(torch.tensor([1.0, 4.0, 9.0, 16.0]))
    opt._q_initialized = True

    lamA_is = _rdinv_like(lam_A, delta)
    lamB_is = _rdinv_like(lam_B, delta)
    din_is = _rdinv_like(torch.tensor([9.0, 1.0, 4.0]), delta)
    dout_is = _rdinv_like(torch.tensor([1.0, 4.0, 9.0, 16.0]), delta)

    zA_basis = UA.T @ gA
    zB_basis = gB @ UB
    zA = UA @ (zA_basis / (zA_basis.abs() + opt.eps))
    zB = (zB_basis / (zB_basis.abs() + opt.eps)) @ UB.T
    if use_polar:
        zA = _newton_schulz(zA, nsteps=12, pre_norm="spec")
        zB = _newton_schulz(zB, nsteps=12, pre_norm="spec")
    WA = UA @ ((UA.T @ zA) * lamA_is.unsqueeze(-1))
    WA = WA * din_is.unsqueeze(0)
    WB = ((zB @ UB) * lamB_is.unsqueeze(0)) @ UB.T
    WB = dout_is.unsqueeze(-1) * WB

    # Warm-start power iteration at the exact top singular vectors so the
    # optimizer's spectral rescale is deterministic and comparable here.
    st["v_sigA"] = _top_power_iter_vector(A)
    st["v_sigB"] = _top_power_iter_vector(B)
    st["v_sigWA"] = _top_power_iter_vector(WA)
    st["v_sigWB"] = _top_power_iter_vector(WB)

    rho = lr / (
        torch.linalg.matrix_norm(A.float(), ord=2)
        + torch.linalg.matrix_norm(B.float(), ord=2)
        + opt.eps
    )
    expected_dA = -(rho / torch.linalg.matrix_norm(WA, ord=2)) * WA
    expected_dB = -(rho / torch.linalg.matrix_norm(WB, ord=2)) * WB
    A_before = A.detach().clone()
    B_before = B.detach().clone()

    A.grad = gA.clone()
    B.grad = gB.clone()
    opt.step()

    assert torch.allclose(A.detach() - A_before, expected_dA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(B.detach() - B_before, expected_dB, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name,expect_polar", [
    ("curvature-whiten-lora", False),
    ("curvature-whiten-polar-lora", True),
])
def test_factory_dispatch(name, expect_polar):
    m, _, _ = _make()
    opt = build_optimizer(m, name, lr=1e-3, curvature_beta=0.99, muon_ns_steps=5)
    assert isinstance(opt, CurvatureWhitenLoRA)
    assert opt.use_polar is expect_polar
