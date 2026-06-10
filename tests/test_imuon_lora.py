"""iMuon baseline = the authors' VENDORED reference (arXiv:2605.09238), called directly with
`variant='v5_warmup'` (their built-in init-stable variant). These tests pin the build_optimizer
wiring so the config can't silently drift.

Why v5_warmup and not our own Corollary 4.1: the decoupled closed form is NOT viable at this
project's B=0 LoRA init — `S_B^{-1/2} ≈ δ^{-1/2}` blows the A-side step up (param_l2 explodes,
loss flat — measured at OLMo opc r256). The authors ship the joint projector form + a warmup
precisely to avoid this. We run their code. Deviations from their default: wd=0 (our protocol)
and adjust_lr=False (scalar lr, not the Muon √d heuristic).
"""
import math
import torch
import torch.nn as nn

from lora_playground.optim import build_optimizer
from lora_playground.third_party.imuon_muon import Muon as _IMuonRef


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


def test_dispatch_and_locked_config():
    m, _, _ = _make()
    opt = build_optimizer(m, "imuon-lora", lr=3e-2)

    assert isinstance(opt, _IMuonRef)
    assert opt.lora_riemannian_variant == "v5_warmup"  # their built-in init-stable variant
    assert opt.lora_riemannian_muon is True
    assert opt.lora_riemannian_adjust_lr is False       # scalar lr (no Muon √d heuristic)
    assert opt.lora_riemannian_ortho_method == "ns"
    assert math.isclose(opt.lora_precond_eps, 1e-6, rel_tol=0, abs_tol=0)

    g = opt.param_groups[0]
    assert math.isclose(g["lr"], 3e-2)
    assert math.isclose(g["momentum"], 0.95)            # Appendix-K with-momentum, matched
    assert g["nesterov"] is True
    assert g["ns_steps"] == 5
    assert g["wd"] == 0.0                                # ENFORCED: our protocol, no wd confound


def test_step_executes_and_is_finite():
    """A real step routes through the Riemannian path on every LoRA pair and stays finite.
    (v5_warmup runs the joint `full` variant during warmup; both update all pairs.)"""
    m, x, tgt = _make()
    opt = build_optimizer(m, "imuon-lora", lr=3e-2)

    before = [p.detach().clone() for grp in opt.param_groups for p in grp["params"]]
    ((m(x) - tgt) ** 2).mean().backward()
    opt._diagnostic_updated_pairs = 0
    opt.step()
    after = [p.detach().clone() for grp in opt.param_groups for p in grp["params"]]

    assert opt._diagnostic_updated_pairs == 2
    assert sum((a - b).norm().item() for a, b in zip(after, before)) > 0.0
    assert all(torch.isfinite(p).all() for p in after)


def test_adapter_matches_direct_construction():
    """build_optimizer passes exactly the flags a direct construction would — guards drift."""
    m1, x, tgt = _make(seed=1)
    m2, _, _ = _make(seed=1)

    opt1 = build_optimizer(m1, "imuon-lora", lr=2e-2)

    from lora_playground.utils import collect_lora_pairs
    pairs2 = collect_lora_pairs(m2)
    muon_params2 = [p for A, B in pairs2 for p in (A, B)]
    opt2 = _IMuonRef(
        lr=2e-2, wd=0.0, muon_params=muon_params2,
        momentum=0.95, nesterov=True, ns_steps=5,
        lora_pairs=pairs2, lora_riemannian_muon=True,
        lora_riemannian_variant="v5_warmup", lora_riemannian_adjust_lr=False,
    )

    for m, opt in ((m1, opt1), (m2, opt2)):
        ((m(x) - tgt) ** 2).mean().backward()
        opt.step()

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), f"adapter diverged from direct at {n1}"
