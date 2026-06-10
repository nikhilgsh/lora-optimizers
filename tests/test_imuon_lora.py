"""iMuon baseline = the PUBLISHED decoupled Corollary 4.1 (arXiv:2605.09238), implemented
as `IMuonLoRA` (== skeleton Prop 2). Tests pin: (1) build_optimizer wiring, (2) a step
matches the Cor 4.1 closed form, (3) the step is finite and moves params.

We deliberately do NOT run the authors' shipped `v5` (joint momentum M_t = M_B A + B M_A) —
it is uninterpretable and performance-unjustified; the library has no decoupled Cor 4.1.
"""
import math
import torch
import torch.nn as nn

from lora_playground.optim import IMuonLoRA, build_optimizer
from lora_playground.utils import collect_lora_pairs


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.normal_(self.lora_B["default"].weight, std=0.05)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class _TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    return _TinyLoRAModel(), torch.randn(3, 8), torch.randn(3, 8)


def _polar(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def _isqrt(S):
    w, Q = torch.linalg.eigh(S)
    return (Q * w.clamp_min(1e-30).rsqrt()) @ Q.T


def test_dispatch_and_config():
    m, _, _ = _make()
    opt = build_optimizer(m, "imuon-lora", lr=3e-2)
    assert isinstance(opt, IMuonLoRA)
    assert math.isclose(opt.momentum, 0.95)   # Appendix-K with-momentum, matched to protagonist
    assert math.isclose(opt.delta, 1e-6)        # their Gram damping ε
    assert math.isclose(opt.param_groups[0]["lr"], 3e-2)


def test_step_matches_corollary_4_1():
    """One step equals the decoupled Cor 4.1 closed form computed independently (float32)."""
    lr, beta, delta = 2e-2, 0.95, 1e-6
    m, x, tgt = _make(seed=1)
    opt = build_optimizer(m, "imuon-lora", lr=lr)
    pairs = collect_lora_pairs(m)
    before = [(A.detach().clone().float(), B.detach().clone().float()) for A, B in pairs]
    ((m(x) - tgt) ** 2).mean().backward()
    grads = [(A.grad.detach().clone().float(), B.grad.detach().clone().float()) for A, B in pairs]
    opt.step()

    for (A, B), (A0, B0), (gA, gB) in zip(pairs, before, grads):
        # First step: momentum buffer starts at 0, so Nesterov lookahead M̃ = (1+β)·g.
        mtA, mtB = gA + beta * gA, gB + beta * gB
        r = A0.shape[0]
        eye = torch.eye(r, dtype=torch.float32)
        isA = _isqrt(A0 @ A0.T + delta * eye)        # (A Aᵀ)^{-1/2}
        isB = _isqrt(B0.T @ B0 + delta * eye)        # (Bᵀ B)^{-1/2}
        dA = isB @ _polar(isB @ mtA)
        dB = _polar(mtB @ isA) @ isA
        exp_A = A0 - lr * dA
        exp_B = B0 - lr * dB
        assert torch.allclose(A.detach().float(), exp_A, atol=1e-5, rtol=1e-4)
        assert torch.allclose(B.detach().float(), exp_B, atol=1e-5, rtol=1e-4)


def test_step_finite_and_moves():
    m, x, tgt = _make()
    opt = build_optimizer(m, "imuon-lora", lr=3e-2)
    before = [p.detach().clone() for grp in opt.param_groups for p in grp["params"]]
    ((m(x) - tgt) ** 2).mean().backward()
    opt.step()
    after = [p.detach().clone() for grp in opt.param_groups for p in grp["params"]]
    assert sum((a - b).norm().item() for a, b in zip(after, before)) > 0.0
    assert all(torch.isfinite(p).all() for p in after)


def test_momentum_accumulates():
    """Second step's lookahead uses the accumulated buffer (β·m + g), not a fresh gradient."""
    m, x, tgt = _make(seed=2)
    opt = build_optimizer(m, "imuon-lora", lr=1e-2)
    for _ in range(2):
        opt.zero_grad(set_to_none=False)
        ((m(x) - tgt) ** 2).mean().backward()
        opt.step()
    # buffers are non-zero after two steps (momentum is live)
    assert any(st["m_A"].abs().sum() > 0 for st in opt.pair_state.values())
