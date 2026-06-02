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
def test_batched_matches_per_pair(use_polar):
    """The grouped batched path and the per-pair oracle must agree on the KL
    path too — guards the new coupled-EMA / Shampoo-core orientation."""
    def run(batched):
        m, x, target = _make(seed=3)
        opt = _kl_opt(m, lr=1e-2, use_polar=use_polar, precond_refresh_every=2)
        opt._batched_step = batched
        for _ in range(4):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    for pg, pp in zip(run(True), run(False)):
        assert torch.allclose(pg, pp, atol=1e-5, rtol=1e-4), "batched vs per-pair KL mismatch"


def test_warmup_kl_gram_identity():
    """On step 1 the factors are zero ⇒ the relative-damped inverses fall back to
    identity, so the coupled KL Gram update reduces to its one-sided warm form
    with the 1/d normalizer:
        L_A  = (1-β_c)/d_in * g_A g_Aᵀ
        D_in = (1-β_c)/r    * diag(g_Aᵀ g_A)
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
    expected_LA = (1.0 - cb) / d_in * (gA @ gA.T)
    expected_Din = (1.0 - cb) / r * (gA * gA).sum(dim=0)
    assert torch.allclose(st["L_A"], expected_LA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["D_in"], expected_Din, atol=1e-6, rtol=1e-5)


def test_coupled_kl_gram_update_nontrivial():
    """Pin the coupled KL Gram EMA (Prop 4) against injected NON-identity factors.

    The warmup test only sees identity inverses, and grouped-vs-per-pair
    equivalence can't catch a CONSISTENT orientation/normalizer bug (same error
    in both paths). This recomputes the four coupled EMA targets from the
    pre-update factors using the reference formula and the optimizer's own
    relative-damped inverse, then checks the post-step state matches:
        L_A  ← β L_A  + ((1-β)/d_in)  g_A diag(D_in)^{-1} g_A^T
        D_in ← β D_in + ((1-β)/r)     diag(g_A^T S_curv,A^{-1} g_A)
    (B-side symmetric with 1/d_out and 1/r). precond_refresh_every is set huge so
    the injected eigenbasis Q is not overwritten mid-step.
    """
    cb = 0.9
    m, x, target = _make(seed=5)
    opt = _kl_opt(m, lr=1e-3, curvature_beta=cb, precond_refresh_every=10_000)
    opt._batched_step = False  # exercise the per-pair oracle
    opt._q_initialized = True  # no eigh reseed; keep the injected Q
    A, B = opt.pairs[0]
    r, d_in = A.shape
    d_out = B.shape[0]
    st = opt.pair_state[0]

    torch.manual_seed(1)
    QA = torch.linalg.qr(torch.randn(r, r))[0]
    QB = torch.linalg.qr(torch.randn(r, r))[0]
    lamA_eig = torch.tensor([4.0, 1.0, 0.5, 2.0])[:r]
    lamB_eig = torch.tensor([1.0, 3.0, 0.25, 1.5])[:r]
    st["Q_A"].copy_(QA)
    st["Q_B"].copy_(QB)
    st["L_A"].copy_(QA @ torch.diag(lamA_eig) @ QA.T)
    st["R_B"].copy_(QB @ torch.diag(lamB_eig) @ QB.T)
    st["D_in"].copy_(torch.linspace(0.5, 3.0, d_in))
    st["D_out"].copy_(torch.linspace(0.4, 2.5, d_out))

    LA0 = st["L_A"].clone(); Din0 = st["D_in"].clone()
    RB0 = st["R_B"].clone(); Dout0 = st["D_out"].clone()

    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    gA = A.grad.detach().float().clone()
    gB = B.grad.detach().float().clone()
    opt.step()

    # Reference inverses from the PRE-update factors (matches the code path).
    lamA = opt._rdinv((QA * (LA0 @ QA)).sum(dim=0))      # (-1/2)
    lamB = opt._rdinv((QB * (RB0 @ QB)).sum(dim=0))
    dinA = opt._rdinv(Din0); doutB = opt._rdinv(Dout0)
    Din_inv = dinA * dinA; Dout_inv = doutB * doutB       # (-1)
    SAinv = QA @ (torch.diag(lamA * lamA) @ QA.T)
    RBinv = QB @ (torch.diag(lamB * lamB) @ QB.T)

    exp_LA = cb * LA0 + ((1 - cb) / d_in) * ((gA * Din_inv.unsqueeze(0)) @ gA.T)
    exp_Din = cb * Din0 + ((1 - cb) / r) * (gA * (SAinv @ gA)).sum(dim=0)
    exp_RB = cb * RB0 + ((1 - cb) / d_out) * (gB.T @ (gB * Dout_inv.unsqueeze(-1)))
    exp_Dout = cb * Dout0 + ((1 - cb) / r) * (gB * (gB @ RBinv)).sum(dim=1)

    assert torch.allclose(st["L_A"], exp_LA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["D_in"], exp_Din, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["R_B"], exp_RB, atol=1e-6, rtol=1e-5)
    assert torch.allclose(st["D_out"], exp_Dout, atol=1e-6, rtol=1e-5)


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
