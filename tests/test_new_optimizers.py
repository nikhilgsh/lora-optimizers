"""
CPU-only unit tests for MuonLoRA, DiagScaledLoRA, KronGradLoRA, and GaLoreAdamW.
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

from lora_playground.optim import (
    AdamMuonLoRA,
    AdamProductMuonLoRA,
    DiagScaledLoRA,
    GaLoreAdamW,
    KronGradLoRA,
    MuonAdamLoRA,
    MuonLoRA,
    ProductMuonLoRA,
    PSILoRA,
)
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


def test_newton_schulz_short_fat():
    """NS on (r, d) with r ≤ d should produce ≈ orthonormal rows: Y Yᵀ ≈ I_r."""
    from lora_playground.optim import _newton_schulz
    torch.manual_seed(7)
    for (r, d) in [(4, 32), (16, 256), (2, 8)]:
        for scale in [1e-6, 1.0, 1e3]:
            X = torch.randn(r, d) * scale
            Y = _newton_schulz(X, nsteps=5)
            err = (Y @ Y.T - torch.eye(r)).abs().max().item()
            assert err < 0.1, f"NS not orthonormal for ({r},{d}) scale={scale}: err={err}"
            # Frobenius norm independent of input magnitude (canonical Muon)
            assert abs(Y.norm().item() - r ** 0.5) < 0.2


def test_newton_schulz_tall_skinny():
    """NS on (d, r) with d > r should produce ≈ orthonormal cols: Yᵀ Y ≈ I_r."""
    from lora_playground.optim import _newton_schulz
    torch.manual_seed(8)
    for (d, r) in [(32, 4), (256, 16)]:
        X = torch.randn(d, r)
        Y = _newton_schulz(X, nsteps=5)
        err = (Y.T @ Y - torch.eye(r)).abs().max().item()
        assert err < 0.1, f"NS tall not orthonormal for ({d},{r}): err={err}"


def test_newton_schulz_scale_invariant_update():
    """Canonical Muon: NS output magnitude is independent of input magnitude.
    This is the property that makes lr the sole step-size knob."""
    from lora_playground.optim import _newton_schulz
    torch.manual_seed(9)
    X = torch.randn(8, 64)
    Y_small = _newton_schulz(X * 1e-5, nsteps=5)
    Y_large = _newton_schulz(X * 1e3, nsteps=5)
    # Same direction, same norm — regardless of scaling
    assert torch.allclose(Y_small, Y_large, atol=1e-3)


def test_muon_lora_zero_grad_after_step():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = MuonLoRA(model, lr=1e-3)
    _set_grads(model)
    opt.step()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max() == 0.0, f"Gradient not zeroed for {name}"


def test_muon_lora_lr_b_multiplier_scales_b_step():
    """LoRA+ for Muon: with B init at 0 and m=k, ‖δB‖ = k · ‖δB at m=1‖ exactly.
    A's update is unchanged across m values (no coupling in vanilla Muon)."""
    for k in [1.0, 4.0, 16.0]:
        torch.manual_seed(0)
        model = TinyLoRAModel()
        for p in model.parameters():
            if p.requires_grad and p.dim() == 2 and p.shape[0] < p.shape[1]:
                continue  # leave A as Kaiming
        # B was already zeroed in _FakeLoRALinear.__init__
        _set_grads(model, seed=7)
        opt = MuonLoRA(model, lr=1e-2, lr_b_multiplier=k)
        # Snapshot A grads before step (zeroed during step)
        A_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if "lora_A" in name}
        B_before = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_B" in name}
        A_before = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_A" in name}
        opt.step()
        for name, p in model.named_parameters():
            if "lora_B" in name:
                if k == 1.0:
                    ref_dB_norm_1 = (p.detach() - B_before[name]).norm().item()
                    test_muon_lora_lr_b_multiplier_scales_b_step._ref = ref_dB_norm_1
                else:
                    ref = test_muon_lora_lr_b_multiplier_scales_b_step._ref
                    actual = (p.detach() - B_before[name]).norm().item()
                    assert abs(actual - k * ref) / max(k * ref, 1e-9) < 1e-3, \
                        f"k={k}: |δB| should be {k}× ref={ref:.4e}, got {actual:.4e}"


def test_muon_lora_ns_steps_zero_is_momentum_sgd():
    """ns_steps=0 disables Newton-Schulz; update is then plain momentum × lr.
    Tier-2 sanity check — isolates orthogonalization's contribution."""
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = MuonLoRA(model, lr=1e-2, ns_steps=0, beta=0.9)
    _set_grads(model, seed=3)
    # Snapshot grads (zeroed after step)
    grad_snap = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    param_before = {name: p.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    opt.step()
    # Step 1 momentum: m = 0.1 * g; update = -lr * m
    for name, p in model.named_parameters():
        if name not in grad_snap:
            continue
        expected = -1e-2 * 0.1 * grad_snap[name]
        actual = p.detach() - param_before[name]
        assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-6), \
            f"ns_steps=0 should give momentum SGD update; mismatch on {name}"


# ─── ProductMuonLoRA (Tier 3) ────────────────────────────────────────────────

def test_product_muon_lora_step_runs():
    """B=0 init: at step 1 only B updates (min-norm Sylvester sets δA=0 because
    B·δA = 0 contributes nothing to ΔW). At step 2 B is nonzero so A also updates.
    This is the *correct* product-norm behavior — and re-derives the H1 motivation
    for LoRA+ asymmetry."""
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    # Manually init B to nonzero so step 1 already exercises both updates.
    with torch.no_grad():
        model.layer0.lora_B["default"].weight.copy_(0.1 * torch.randn(6, 2))
        model.layer1.lora_B["default"].weight.copy_(0.1 * torch.randn(8, 2))
    opt = ProductMuonLoRA(model, lr=1e-2, alpha=2, rank=2, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    B_before = model.layer0.lora_B["default"].weight.detach().clone()
    _set_grads(model, seed=2)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    B_after = model.layer0.lora_B["default"].weight.detach()
    assert not torch.allclose(A_before, A_after), "ProductMuonLoRA did not update A"
    assert not torch.allclose(B_before, B_after), "ProductMuonLoRA did not update B"
    assert torch.isfinite(A_after).all() and torch.isfinite(B_after).all()


def test_product_muon_lora_b_zero_init_only_b_updates():
    """At B=0 init, the min-Frobenius Sylvester recovery sets δA=0 (any δA
    produces zero merged-W contribution since B·δA = 0). This re-derives
    why LoRA+ asymmetry helps: A is "frozen" until B picks up signal."""
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    # B is already zero from _FakeLoRALinear's init.
    opt = ProductMuonLoRA(model, lr=1e-2, alpha=2, rank=2, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    _set_grads(model, seed=2)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert torch.allclose(A_before, A_after, atol=1e-7), \
        "B=0 init should imply δA=0 at step 1 (min-Frobenius Sylvester)"


def test_product_muon_lora_gauge_invariance():
    """Gauge symmetry: A → R A, B → B R^{-1} leaves the model output unchanged.
    A correctly-formulated Muon for LoRA should produce updates that, after
    composition with R, yield the SAME merged-weight update ΔW = B'·δA' + δB'·A'.

    We check: starting from two gauge-equivalent (A, B) and (A', B') = (R A, B R^{-1})
    with consistent gradients, one step of ProductMuonLoRA produces the SAME ΔW
    in merged-weight space."""
    torch.manual_seed(11)
    r, d_in, d_out = 2, 8, 6
    A = torch.randn(r, d_in) * 0.3
    B = torch.randn(d_out, r) * 0.3   # NB: nonzero B, otherwise gauge orbit collapses
    # Random invertible R ∈ ℝ^{r×r}
    R = torch.eye(r) + 0.3 * torch.randn(r, r)
    R_inv = torch.linalg.inv(R)
    A_p = R @ A           # (r, d_in)
    B_p = B @ R_inv       # (d_out, r)
    # Pick a merged-weight target gradient G ∈ ℝ^{d_out × d_in} consistent across gauges
    G = 0.01 * torch.randn(d_out, d_in)

    def _run(A0, B0, G0):
        m = TinyLoRAModel(d_in=d_in, d_out=d_out, r=r)
        m.layer1 = m.layer0  # Tie layers so we have a single (A, B) to act on; we'll only inspect layer0
        m.layer0.lora_A["default"].weight.data.copy_(A0)
        m.layer0.lora_B["default"].weight.data.copy_(B0)
        # Set grads consistent with chain-rule from G: ∇_A = (α/r) B^T G, ∇_B = (α/r) G A^T
        scale = 1.0  # alpha=rank=r in this test, so α/r = 1
        for name, p in m.named_parameters():
            p.grad = torch.zeros_like(p)
        m.layer0.lora_A["default"].weight.grad = scale * (B0.T @ G0)
        m.layer0.lora_B["default"].weight.grad = scale * (G0 @ A0.T)
        opt = ProductMuonLoRA(m, lr=1e-2, alpha=r, rank=r, ns_steps=5, beta=0.0)
        opt.step()
        A1 = m.layer0.lora_A["default"].weight.detach().clone()
        B1 = m.layer0.lora_B["default"].weight.detach().clone()
        # Linearized merged update: ΔW ≈ (α/r)(B (A1−A0) + (B1−B0) A) — at α/r = 1
        dW = B0 @ (A1 - A0) + (B1 - B0) @ A0
        return dW

    dW = _run(A, B, G)
    dW_p = _run(A_p, B_p, G)
    rel = (dW - dW_p).norm() / max(dW.norm().item(), 1e-9)
    assert rel < 5e-2, f"Gauge invariance broken: relative ΔW mismatch = {rel:.3e}"


def test_product_muon_lora_zero_grad_after_step():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = ProductMuonLoRA(model, lr=1e-3, alpha=2, rank=2)
    _set_grads(model)
    opt.step()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max() == 0.0


# ─── AdamMuonLoRA (Tier 4) ──────────────────────────────────────────────────

def test_adam_muon_lora_step_runs():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = AdamMuonLoRA(model, lr=1e-3, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    _set_grads(model)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)
    assert torch.isfinite(A_after).all()


# ─── AdamProductMuonLoRA (H2 ⊗ H4) ──────────────────────────────────────────

def test_adam_product_muon_lora_step_runs():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    with torch.no_grad():
        model.layer0.lora_B["default"].weight.copy_(0.1 * torch.randn(6, 2))
        model.layer1.lora_B["default"].weight.copy_(0.1 * torch.randn(8, 2))
    opt = AdamProductMuonLoRA(model, lr=1e-2, alpha=2, rank=2, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    B_before = model.layer0.lora_B["default"].weight.detach().clone()
    _set_grads(model, seed=2)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    B_after = model.layer0.lora_B["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)
    assert not torch.allclose(B_before, B_after)
    assert torch.isfinite(A_after).all() and torch.isfinite(B_after).all()


def test_muon_adam_lora_step_runs():
    """MuonAdamLoRA = NS first, then Adam (reverse of AdamMuonLoRA)."""
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = MuonAdamLoRA(model, lr=1e-3, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    _set_grads(model)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)
    assert torch.isfinite(A_after).all()


def test_adam_product_muon_lora_b_zero_init_only_b_updates():
    """Like ProductMuonLoRA: at B=0, Sylvester min-Frobenius sets precond_A=0,
    so m_A stays 0 and δA=0 at step 1. B updates normally."""
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = AdamProductMuonLoRA(model, lr=1e-2, alpha=2, rank=2, ns_steps=5)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    B_before = model.layer0.lora_B["default"].weight.detach().clone()
    _set_grads(model, seed=2)
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    B_after = model.layer0.lora_B["default"].weight.detach()
    assert torch.allclose(A_before, A_after, atol=1e-7), \
        "B=0 init: AdamProductMuonLoRA should leave A unchanged at step 1"
    assert not torch.allclose(B_before, B_after)


# ─── DiagScaledLoRA ───────────────────────────────────────────────────────────

def _trigger_hooks(opt, seed=1):
    """Simulate one forward+backward by populating cached_X/S per pair's shape."""
    torch.manual_seed(seed)
    for i, (A, B) in enumerate(opt.pairs):
        d_in = A.shape[1]
        d_out = B.shape[0]
        opt.cached_X[i] = torch.randn(4, d_in)
        opt.cached_S[i] = torch.randn(4, d_out)


def test_diag_scaled_lora_state_shapes():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = DiagScaledLoRA(model, lr=1e-3)
    assert len(opt.pair_state) == 2  # two LoRA layers
    for i, (A, B) in enumerate(opt.pairs):
        assert opt.pair_state[i]["D_V"].shape == (A.shape[1],)  # (d_in,)
        assert opt.pair_state[i]["D_U"].shape == (B.shape[0],)  # (d_out,)


def test_diag_scaled_lora_dv_du_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = DiagScaledLoRA(model, lr=1e-3, ema_beta=0.9)
    # D_V starts at ones; after update with non-zero X it should change
    dv_before = opt.pair_state[0]["D_V"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    dv_after = opt.pair_state[0]["D_V"]
    assert not torch.allclose(dv_before, dv_after), "D_V did not update"


def test_diag_scaled_lora_params_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = DiagScaledLoRA(model, lr=1e-2)
    d_in, d_out = model.d_in, model.d_out
    _trigger_hooks(opt)
    _set_grads(model)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)


def test_diag_scaled_lora_zero_grad_after_step():
    torch.manual_seed(0)
    model = TinyLoRAModel()
    opt = DiagScaledLoRA(model, lr=1e-3)
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max() == 0.0, f"Gradient not zeroed for {name}"


# ─── KronGradLoRA ─────────────────────────────────────────────────────────────

def test_kron_grad_lora_state_shapes():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KronGradLoRA(model, lr=1e-3)
    for i, (A, B) in enumerate(opt.pairs):
        r = A.shape[0]
        assert opt.pair_state[i]["H_A"].shape == (r, r)
        assert opt.pair_state[i]["H_B"].shape == (r, r)
        assert opt.pair_state[i]["D_V"].shape == (A.shape[1],)
        assert opt.pair_state[i]["D_U"].shape == (B.shape[0],)


def test_kron_grad_lora_ha_hb_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KronGradLoRA(model, lr=1e-3, ema_beta=0.9)
    ha_before = opt.pair_state[0]["H_A"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    ha_after = opt.pair_state[0]["H_A"]
    assert not torch.allclose(ha_before, ha_after), "H_A did not update"


def test_kron_grad_lora_params_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = KronGradLoRA(model, lr=1e-2)
    _trigger_hooks(opt)
    _set_grads(model)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    assert not torch.allclose(A_before, A_after)


# ─── PSILoRA (F-LoRSUM-based, paper Algorithm 3) ─────────────────────────────

def test_psi_lora_state_shapes():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-3)
    for i, (A, B) in enumerate(opt.pairs):
        s = opt.pair_state[i]
        assert s["D_V"].shape == (A.shape[1],)
        assert s["D_U"].shape == (B.shape[0],)
        assert s["M_A"].shape == A.shape       # (r_m=r, d_in)
        assert s["M_B"].shape == B.shape       # (d_out, r_m=r)


def test_psi_lora_step_runs_and_updates_params():
    """With LoRA's PEFT init (B=0), the first step's A-update uses B=0 and the
    proximal prior pulls A back to A₀. The B-update then uses A=A₀ and moves B
    off zero. After a *second* step, both A and B should have changed."""
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-2, ema_beta=0.9, momentum=0.9,
                  inner_iters=1, proximal_rho=0.01)
    A_before = model.layer0.lora_A["default"].weight.detach().clone()
    B_before = model.layer0.lora_B["default"].weight.detach().clone()
    for _ in range(2):
        _trigger_hooks(opt)
        _set_grads(model)
        opt.step()
    A_after = model.layer0.lora_A["default"].weight.detach()
    B_after = model.layer0.lora_B["default"].weight.detach()
    assert torch.isfinite(A_after).all() and torch.isfinite(B_after).all()
    assert not torch.allclose(B_before, B_after), "B did not update"
    assert not torch.allclose(A_before, A_after), "A did not update after 2 steps"


def test_psi_lora_momentum_buffer_updates():
    """Same K=1 ALS asymmetry as the params update test: M_B (init zero) moves on
    step 1, M_A moves on step 2."""
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-3, momentum=0.5, inner_iters=1)
    M_B_before = opt.pair_state[0]["M_B"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    M_B_after = opt.pair_state[0]["M_B"]
    assert not torch.allclose(M_B_before, M_B_after), "M_B did not update via LoRSUM"


def test_psi_lora_dv_du_update():
    torch.manual_seed(0)
    model = TinyLoRAModel(d_in=8, d_out=6, r=2)
    opt = PSILoRA(model, lr=1e-3, ema_beta=0.9)
    dv_before = opt.pair_state[0]["D_V"].clone()
    _trigger_hooks(opt)
    _set_grads(model)
    opt.step()
    dv_after = opt.pair_state[0]["D_V"]
    assert not torch.allclose(dv_before, dv_after), "D_V did not update"


# ─── GaLoreAdamW ──────────────────────────────────────────────────────────────

def test_galore_adamw_projection_shape():
    torch.manual_seed(0)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj", "v_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=1)
    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    # ortho shape depends on weight shape (proj_type="std" picks shorter dim).
    for t in targets:
        gs = opt.galore_state[t.name]
        assert gs["ortho"] is not None
        d_out, d_in = t.weight.shape
        if d_out >= d_in:
            assert gs["side"] == "right"
            assert gs["ortho"].shape == (rank, d_in)
        else:
            assert gs["side"] == "left"
            assert gs["ortho"].shape == (d_out, rank)


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
    gs = opt.galore_state[targets[0].name]
    ortho = gs["ortho"]
    if gs["side"] == "right":
        # rows of ortho orthonormal: ortho @ orthoᵀ ≈ I_r
        assert torch.allclose(ortho @ ortho.T, torch.eye(ortho.shape[0]), atol=1e-5)
    else:
        # cols of ortho orthonormal: orthoᵀ @ ortho ≈ I_r
        assert torch.allclose(ortho.T @ ortho, torch.eye(ortho.shape[1]), atol=1e-5)


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
    d_out, d_in = targets[0].weight.shape
    expected = (d_out, rank) if d_out >= d_in else (rank, d_in)
    assert gs["m"].shape == expected, f"m shape wrong: {gs['m'].shape} vs {expected}"
    assert gs["v"].shape == expected, f"v shape wrong: {gs['v'].shape} vs {expected}"


def test_galore_adamw_moments_persist_across_proj_update():
    """Faithful GaLore: moments must NOT reset when projection refreshes (matches
    official galore_torch.adamw which only initializes m/v once)."""
    torch.manual_seed(4)
    model = TinyDenseModel()
    targets = collect_dense_target_weights(model, ["q_proj"])
    freeze_all_except_targets(model, targets)
    rank = 2
    opt = GaLoreAdamW(targets, rank=rank, lr=1e-3, update_proj_gap=2)

    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()
    m_after_step1 = opt.galore_state[targets[0].name]["m"].clone()
    assert m_after_step1.abs().max() > 0, "m should be non-zero after first step"

    for t in targets:
        t.weight.grad = torch.randn_like(t.weight)
    opt.step()  # step 2 triggers projection refresh (2 % 2 == 0)
    m_after_step2 = opt.galore_state[targets[0].name]["m"]
    assert m_after_step2.abs().max() > 0, "moments wrongly reset on projection update"
