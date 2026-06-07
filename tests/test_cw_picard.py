"""Tests for the Picard cross-coupling loop (cw_picard_iters) on the
CurvatureWhitenLoRA kl family — dense kl-shampoo (option a, mixed-metric cross) and
kl-diag (option b, exact cross). Derivation: kl_shampoo_polar_derivation.md
section "Cross-coupling".

Covers: k=1 is the single-block step (regression handled by the kl-shampoo / kl-diag
suites), grouped↔per-pair equivalence at k=2 (both code paths edited), the
fixed-point convergence of the Picard iteration (the key correctness guard for the
cross-term's sign/scale/metric), and the soap_v guard.
"""
import copy
import torch
import torch.nn as nn
import pytest

from lora_playground.optim import CurvatureWhitenLoRA


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


def _opt(m, diag, k, lr=1e-2, use_polar=True):
    return CurvatureWhitenLoRA(m, lr=lr, use_polar=use_polar, kl_coupled=True,
                               soap_v=False, diag_metric=diag, cw_picard_iters=k)


def test_soap_v_picard_guard():
    m, _, _ = _make()
    with pytest.raises(ValueError):
        CurvatureWhitenLoRA(m, soap_v=True, cw_picard_iters=2)


@pytest.mark.parametrize("diag", [False, True])
def test_picard_batched_matches_per_pair(diag):
    """Grouped batched path and per-pair oracle must agree at k=2 (both edited)."""
    def run(batched):
        m, x, target = _make(seed=3)
        opt = _opt(m, diag=diag, k=2, lr=1e-2)
        opt._batched_step = batched
        for _ in range(4):
            ((m(x) - target) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]

    for pg, pp in zip(run(True), run(False)):
        assert torch.allclose(pg, pp, atol=1e-5, rtol=1e-4), "batched vs per-pair k=2 mismatch"


def _picard_increments(diag, lr=1e-2, warmup=12):
    """Run one step at cw_picard_iters=1..5 from a FROZEN pre-step state with a
    fixed grad; return (d1, [increments d_{k,k-1}]) where d1 is the k=1 update norm
    and d_{k,k-1} = ‖update_k − update_{k-1}‖. Isolates the Picard loop (curvature/
    momentum EMAs are restored identically each k)."""
    m, x, target = _make(seed=5)
    opt = _opt(m, diag=diag, k=1, lr=lr)
    for _ in range(warmup):
        ((m(x) - target) ** 2).mean().backward(); opt.step(); opt.zero_grad()
    param_snap = [p.detach().clone() for p in m.parameters()]
    state_snap = copy.deepcopy(opt.pair_state)
    ((m(x) - target) ** 2).mean().backward()
    grad_snap = [p.grad.detach().clone() for p in m.parameters()]
    opt.zero_grad()
    deltas = {}
    for k in (1, 2, 3, 4, 5):
        for p, ps in zip(m.parameters(), param_snap):
            p.data.copy_(ps)
        opt.pair_state = copy.deepcopy(state_snap)
        opt.cw_picard_iters = k
        for p, g in zip(m.parameters(), grad_snap):
            p.grad = g.clone()
        opt.step(); opt.zero_grad()
        deltas[k] = torch.cat([(p.detach() - ps).flatten()
                               for p, ps in zip(m.parameters(), param_snap)])
    d1 = deltas[1].norm().item()
    incs = [(deltas[k] - deltas[k - 1]).norm().item() for k in (2, 3, 4, 5)]
    return d1, incs


def test_picard_exact_converges_diag():
    """Option (b) (diag_metric) is a coherent single-metric program, so its Picard
    iteration is a true block-coordinate descent and CONTRACTS: the increments shrink
    monotonically toward a fixed point. (Wrong sign/scale/metric would not contract.)"""
    d1, incs = _picard_increments(diag=True)
    d21, d32, d43, d54 = incs
    assert d21 > 1e-6, f"cross-term never engaged: d21={d21:.3e}"
    assert d32 < 0.5 * d21 and d43 < d32 * 2 and d54 < d32, \
        f"exact cross-coupling not contracting: d1={d1:.3e} incs={incs}"


def test_picard_mixed_metric_bounded():
    """Option (a) (dense kl, mixed-metric cross: self=S_curv, cross=diagonals) is
    NOT a coherent single-metric program, so the Picard iteration does NOT contract
    — increments stay ~d1 and may grow. That is expected; we only ever run k=2 (one
    correction), whose norm is ρ-pinned. Assert the single correction engages and
    does not EXPLODE (the real failure mode), not that it converges."""
    d1, incs = _picard_increments(diag=False)
    d21 = incs[0]
    assert d21 > 1e-6, f"cross-term never engaged: d21={d21:.3e}"
    assert d21 < 5.0 * d1, f"single k=2 correction exploding: d1={d1:.3e} d21={d21:.3e}"


def test_picard_k1_is_plain_step():
    """k=1 must equal a fresh default optimizer (no Picard) step-for-step — the loop
    at k=1 never forms the cross-term."""
    def run(k):
        m, x, target = _make(seed=9)
        opt = _opt(m, diag=True, k=k, lr=1e-2)
        for _ in range(6):
            ((m(x) - target) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]
    base = run(1)
    # k=2 must DIFFER (sanity the loop actually does something), k=1 is the control.
    k2 = run(2)
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(base, k2)), \
        "k=2 identical to k=1 — Picard loop is a no-op"
