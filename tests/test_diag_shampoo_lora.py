"""Tests for diag-shampoo-(polar-)lora — the non-KL ablation of kl-diag.

diag-shampoo is CurvatureWhitenLoRA with the consistent diagonal metric
(``diag_metric=True``, small side M_A = Bᵀ diag(D_out) B) and the closed-form
Shampoo whitening (``soap_v=False``) — IDENTICAL to kl-diag — but with
``kl_coupled=False`` so the diagonals D_in/D_out are textbook grad-energy EMAs
instead of the KL coupled fixed point. It isolates what the KL coupling buys at
fixed diagonal-metric geometry.

Covers: factory dispatch + flags for both names, step finiteness/change (both
arms), the clobber guard (with diag_metric the else-branch must NOT overwrite the
recomputed P_A/Q_B with a Gram EMA), and multistep finiteness.
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


@pytest.mark.parametrize("name,polar", [
    ("diag-shampoo-lora", False),
    ("diag-shampoo-polar-lora", True),
])
def test_factory_dispatch_flags(name, polar):
    m, _, _ = _make()
    opt = build_optimizer(m, name, lr=1e-2, precond_delta=1e-4)
    assert isinstance(opt, CurvatureWhitenLoRA)
    # The defining ablation: diagonal metric + Shampoo core, but NO KL coupling.
    assert opt.diag_metric is True
    assert opt.soap_v is False
    assert opt.kl_coupled is False
    assert opt.use_polar is polar


@pytest.mark.parametrize("use_polar", [False, True])
def test_step_runs_and_changes_params(use_polar):
    m, x, target = _make()
    name = "diag-shampoo-polar-lora" if use_polar else "diag-shampoo-lora"
    pre = [p.detach().clone() for p in m.parameters()]
    opt = build_optimizer(m, name, lr=1e-2, precond_delta=1e-4)
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    assert all(torch.isfinite(p).all() for p in post), "Non-finite params after step"
    assert max(float((a - b).abs().sum()) for a, b in zip(pre, post)) > 0.0, "No params changed"


@pytest.mark.parametrize("use_polar", [False, True])
def test_multistep_finite(use_polar):
    m, x, target = _make(seed=7)
    name = "diag-shampoo-polar-lora" if use_polar else "diag-shampoo-lora"
    opt = build_optimizer(m, name, lr=1e-2, precond_delta=1e-4)
    opt.precond_refresh_every = 2
    for _ in range(6):
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    for p in m.parameters():
        assert torch.isfinite(p).all(), "Non-finite param over multiple diag-shampoo steps"


@pytest.mark.parametrize("batched", [False, True])
def test_diag_metric_LA_recomputed_not_clobbered(batched):
    """With diag_metric=True the small-side P_A/Q_B are recomputed each step as
    M_A = Bᵀ diag(Dout_m) B (resp. M_B = A diag(Din_m) Aᵀ), NOT accumulated as a
    Gram EMA. The else-branch (kl_coupled=False) must skip the P_A/Q_B Gram EMA so
    it doesn't clobber the recompute. On step 1 the diagonals are still zero ⇒
    Dout_m=Din_m=1 ⇒ M_A = B0ᵀ B0 exactly (B0 = pre-step factor). If the clobber
    guard were missing, P_A would instead hold (1-β_c)·gA gAᵀ.
    """
    m, x, target = _make(seed=3)
    opt = build_optimizer(m, "diag-shampoo-polar-lora", lr=1e-2, precond_delta=1e-4)
    opt._batched_step = batched
    A_pre = [A.detach().float().clone() for A, B in opt.pairs]
    B_pre = [B.detach().float().clone() for A, B in opt.pairs]
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    for i in range(len(opt.pairs)):
        st = opt.pair_state[i]
        exp_PA = B_pre[i].T @ B_pre[i]          # r×r, Dout_m=1 at step 1
        exp_QB = A_pre[i] @ A_pre[i].T          # r×r, Din_m=1 at step 1
        assert torch.allclose(st["P_A"], exp_PA, atol=1e-5, rtol=1e-4), \
            f"pair {i}: P_A not the diag-metric recompute (clobbered?)"
        assert torch.allclose(st["Q_B"], exp_QB, atol=1e-5, rtol=1e-4), \
            f"pair {i}: Q_B not the diag-metric recompute (clobbered?)"
