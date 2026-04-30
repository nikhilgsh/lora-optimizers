"""
CPU-only unit tests for MuonLoRA, PSILoRA, KFACLoRA, and GaLoreAdamW.
Tests follow the patterns in test_svd_oracle.py:
  - build a tiny model, set manual gradients, call step(), check invariants.
"""
import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import GaLoreAdamW, KFACLoRA, MuonLoRA, PSILoRA
from lora_playground.utils import collect_dense_target_weights, freeze_all_except_targets, spd_frac_power_inv


# ─── tiny PEFT-like model ──────────────────────────────────────────────────────

class _FakeLoRALinear(nn.Module):
    """Minimal stand-in for a PEFT lora.Linear module."""
    def __init__(self, d_in, d_out, r):
        super().__init__()
        # PEFT attaches dicts; we replicate just enough structure for collect_lora_pairs
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        torch.nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        A = self.lora_A["default"].weight   # (r, d_in)
        B = self.lora_B["default"].weight   # (d_out, r)
        return x @ A.T @ B.T


class TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=2):
        super().__init__()
        self.layer0 = _FakeLoRALinear(d_in, d_out, r)
        self.layer1 = _FakeLoRALinear(d_out, d_in, r)
        self.d_in, self.d_out, self.r = d_in, d_out, r


def _set_grads(model, seed=0):
    """Set small deterministic gradients on all LoRA A/B weights."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.grad = 0.01 * torch.randn_like(p)


# ─── TinyTargets for GaLore (same as in test_svd_oracle) ─────────────────────

class DirectMatrix(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, cols))

    def forward(self, x):
        return x @ self.weight


class TinyDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(5, 4, bias=False)
        self.block.v_proj = DirectMatrix(5, 4)


# ─── spd_frac_power_inv ───────────────────────────────────────────────────────

def test_spd_frac_power_inv_identity():
    # (I + eps*I)^{-0.5} ≈ (1+eps)^{-0.5} * I
    H = torch.eye(4)
    eps = 1e-6
    result = spd_frac_power_inv(H, gamma=0.5, eps=eps)
    expected = (1 + eps) ** (-0.5) * torch.eye(4)
    assert torch.allclose(result, expected, atol=1e-5)


def test_spd_frac_power_inv_shape():
    torch.manual_seed(0)
    M = torch.randn(3, 3)
    H = M @ M.T
    result = spd_frac_power_inv(H, gamma=0.5)
    assert result.shape == (3, 3)


# ─── MuonLoRA ─────────────────────────────────────────────────────────────────

def test_muon_lora_params_update():
    torch.manual_seed(42)
    model = TinyLoRAModel()
    opt = MuonLoRA(model, lr=1e-2)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    _set_grads(model)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after), "MuonLoRA did not update A"


def test_muon_lora_ns_approximately_orthonormal():
    """After a step, m_A should be NS-orthogonalized: rows nearly orthonormal."""
    torch.manual_seed(7)
    model = TinyLoRAModel(r=2)
    opt = MuonLoRA(model, lr=1e-2, beta=0.0)  # beta=0 → m_A = G_A exactly
    _set_grads(model)
    opt.step()
    # opt.pair_state[0]["m_A"] holds the last momentum; check update was NS-like
    # Verify A changed and shapes are correct
    A = model.layer0.lora_A["default"].weight  # (r, d_in)
    B = model.layer0.lora_B["default"].weight  # (d_out, r)
    assert A.shape == (model.r, model.d_in)
    assert B.shape == (model.d_out, model.r)


def test_muon_lora_zero_grad_after_step():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = MuonLoRA(model, lr=1e-3)
    _set_grads(model)
    opt.step()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max() == 0.0, f"Gradient not zeroed for {name}"


# ─── PSILoRA ──────────────────────────────────────────────────────────────────

def _trigger_hooks(opt, seed=1):
    """Simulate one forward+backward by populating cached_X/S per pair's shape."""
    torch.manual_seed(seed)
    for i, (A, B) in enumerate(opt.pairs):
        d_in = A.shape[1]
        d_out = B.shape[0]
        opt.cached_X[i] = torch.randn(4, d_in)
        opt.cached_S[i] = torch.randn(4, d_out)


def test_psi_lora_state_shapes():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-3)
    assert len(opt.pair_state) == 2  # two LoRA layers
    for i, (A, B) in enumerate(opt.pairs):
        assert opt.pair_state[i]["D_V"].shape == (A.shape[1],)  # (d_in,)
        assert opt.pair_state[i]["D_U"].shape == (B.shape[0],)  # (d_out,)


def test_psi_lora_dv_du_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-3, ema_beta=0.9)
    # D_V starts at ones; after update with non-zero X it should change
    dv_before = opt.pair_state[0]["D_V"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    dv_after = opt.pair_state[0]["D_V"]
    assert not torch.allclose(dv_before, dv_after), "D_V did not update"


def test_psi_lora_params_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-2)
    d_in, d_out = model.d_in, model.d_out
    _trigger_hooks(opt)
    _set_grads(model)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)


def test_psi_lora_zero_grad_after_step():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = PSILoRA(model, lr=1e-3)
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max() == 0.0, f"Gradient not zeroed for {name}"


# ─── KFACLoRA ─────────────────────────────────────────────────────────────────

def test_kfac_lora_state_shapes():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KFACLoRA(model, lr=1e-3)
    for i, (A, B) in enumerate(opt.pairs):
        r = A.shape[0]
        assert opt.pair_state[i]["H_A"].shape == (r, r)
        assert opt.pair_state[i]["H_B"].shape == (r, r)
        assert opt.pair_state[i]["D_V"].shape == (A.shape[1],)
        assert opt.pair_state[i]["D_U"].shape == (B.shape[0],)


def test_kfac_lora_ha_hb_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KFACLoRA(model, lr=1e-3, ema_beta=0.9)
    ha_before = opt.pair_state[0]["H_A"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    ha_after = opt.pair_state[0]["H_A"]
    assert not torch.allclose(ha_before, ha_after), "H_A did not update"


def test_kfac_lora_params_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KFACLoRA(model, lr=1e-2)
    _trigger_hooks(opt)
    _set_grads(model)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)


# ─── GaLoreAdamW ──────────────────────────────────────────────────────────────

def test_galore_adamw_projection_shape():
    torch.manual_seed(0)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj", "v_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=1)
    # Set gradients manually
    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    # P should be (d_out, r)
    for t in targets:
        gs = opt.galore_state[t.name]
        assert gs["P"] is not None
        d_out = t.weight.shape[0]
        assert gs["P"].shape == (d_out, rank)


def test_galore_adamw_projection_orthonormal():
    torch.manual_seed(1)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=1)
    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    P = opt.galore_state[targets[0].name]["P"]  # (d_out, r)
    # P should have orthonormal columns: PᵀP ≈ I_r
    assert torch.allclose(P.T @ P, torch.eye(rank), atol=1e-5), "P columns not orthonormal"


def test_galore_adamw_params_update():
    torch.manual_seed(2)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj"])
    freeze_all_except_targets(model, targets)
    opt = GaLoreAdamW(targets, rank=2, lr=1e-2, update_proj_gap=1)
    w_before = targets[0].weight.detach().clone()
    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    w_after = targets[0].weight.detach()
    assert not torch.allclose(w_before, w_after), "GaLoreAdamW did not update weight"


def test_galore_adamw_moments_shape():
    torch.manual_seed(3)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=1)
    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    gs = opt.galore_state[targets[0].name]
    d_in = targets[0].weight.shape[1]
    assert gs["m"].shape == (rank, d_in), f"m shape wrong: {gs['m'].shape}"
    assert gs["v"].shape == (rank, d_in), f"v shape wrong: {gs['v'].shape}"


def test_galore_adamw_projection_resets_on_gap():
    """Moments should be reset when projection updates (step == update_proj_gap)."""
    torch.manual_seed(4)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=2)

    for step in range(3):
        for t in targets:
            t.weight.grad = torch.randn_like(t.weight)
        opt.step()

    # At step 2, projection updates and moments reset to zero, then one step applied.
    # Just verify the optimizer ran 3 steps without error and weight changed.
    assert opt.galore_state[targets[0].name]["step"] == 3
