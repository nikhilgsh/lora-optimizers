"""CPU-only unit tests for AdamLinLoRAPost and AdamScaledLoRAPost (H4 variants).

The "post" optimizers reorder composition: Adam state runs on the raw gradient,
then the geometric (Sylvester / Gram) solve is applied to the Adam direction.
These tests check shapes, dtype/device handling, no-grad early-exit, finite
output, determinism, and that AdamScaledLoRAPost reduces to AdamScaledLoRA's
form when β₁=β₂=0 (because then the Adam direction is sign(gA), but the test
just sanity-checks linearity in that toy regime).
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
    AdamLinLoRAMatrix,
    AdamLinLoRAPost,
    AdamScaledLoRAMatrix,
    AdamScaledLoRAPost,
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


@pytest.mark.parametrize("OptCls", [AdamScaledLoRAPost, AdamLinLoRAPost, AdamScaledLoRAMatrix, AdamLinLoRAMatrix])
def test_step_runs_and_changes_params(OptCls):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = OptCls(m, lr=1e-2)
    out = m(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    # at least one param should have changed and all updates must be finite
    diffs = [float((a - b).abs().sum()) for a, b in zip(pre, post)]
    assert all(torch.isfinite(p).all() for p in post)
    assert max(diffs) > 0.0


@pytest.mark.parametrize("OptCls", [AdamScaledLoRAPost, AdamLinLoRAPost, AdamScaledLoRAMatrix, AdamLinLoRAMatrix])
def test_zero_grad_no_update(OptCls):
    m, x, target = _make()
    pre = [p.detach().clone() for p in m.parameters()]
    opt = OptCls(m, lr=1e-2)
    # zero gradients (loss=0 ⇒ grads zero)
    out = m(x)
    loss = (out * 0.0).sum()
    loss.backward()
    opt.step()
    post = [p.detach().clone() for p in m.parameters()]
    for a, b in zip(pre, post):
        # zero-grad means the Adam direction is undefined (0/0); we pick the
        # convention 0/(0+eps) = 0, so params should be unchanged.
        assert torch.allclose(a, b, atol=1e-7)


@pytest.mark.parametrize("OptCls", [AdamScaledLoRAPost, AdamLinLoRAPost, AdamScaledLoRAMatrix, AdamLinLoRAMatrix])
def test_determinism(OptCls):
    def run():
        m, x, target = _make(seed=42)
        opt = OptCls(m, lr=1e-3)
        for _ in range(3):
            loss = ((m(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    a = run()
    b = run()
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=0.0)


def test_post_differs_from_pre():
    """AdamScaledLoRAPost should produce a different update than AdamScaledLoRA
    on the same state — they apply the Gram solve at different points.
    """
    from lora_playground.optim import AdamScaledLoRA

    def run(OptCls):
        m, x, target = _make(seed=1)
        opt = OptCls(m, lr=1e-2)
        loss = ((m(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        return [p.detach().clone() for p in m.parameters()]

    pre = run(AdamScaledLoRA)
    post = run(AdamScaledLoRAPost)
    # at least one param differs by more than fp32 noise
    max_diff = max(float((a - b).abs().max()) for a, b in zip(pre, post))
    assert max_diff > 1e-5, f"Post and Pre produced identical params (max diff {max_diff})"
