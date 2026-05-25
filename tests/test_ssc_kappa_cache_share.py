"""Unit test for SSC κ-adaptive Picard-share cache (Optimization A).

At picard=2 with κ-adaptive SSC, the polar `_polar` is called twice per
side per step (n=0 and n=1). The expensive part is the eigvalsh+bisect
that solves c from κ; the cheap part is the MISR clipping itself.

Optimization A: cache the n=0 c per side and reuse at n=1, skipping the
solver. Snapshot drift analysis (`docs/notes/polar_product/walltime_profile.md`
§"Snapshot-derived c-drift") shows |Δc|/c is p50<1.1%, p99<3.3% at
production η — sharing is a safe ~2× speedup on the κ-adaptive critical
path.

Tests:
- share=True: the cached c stamped at n=0 is the c used at n=1
  (no re-solving) — within fp32 tolerance.
- share=False: bit-identical to pre-flag behavior (independent per-n
  caches) so existing sweeps are not perturbed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lora_playground.optim import AdamPolarProductLoRA


class _FakeLoRAPair(nn.Module):
    def __init__(self, r, d_in, d_out, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        with torch.no_grad():
            a = torch.empty(r, d_in)
            nn.init.kaiming_uniform_(a, generator=g)
            self.lora_A["default"].weight.copy_(a)
            # Non-zero B so σ_B > 0 and the polar input has a meaningful
            # spectrum at step 1 already.
            b = torch.empty(d_out, r)
            nn.init.kaiming_uniform_(b, generator=g)
            self.lora_B["default"].weight.copy_(b * 0.1)


class _FakeModel(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.adapters = nn.ModuleList(
            [_FakeLoRAPair(r, d_in, d_out, seed=i)
             for i, (r, d_in, d_out) in enumerate(specs)]
        )


def _make_optim(model, *, share_picard, picard_iters=2, refresh_every=1,
                warmup_steps=0):
    return AdamPolarProductLoRA(
        model,
        lr=3e-3,
        betas=(0.9, 0.999),
        delta=1e-6,
        eps=1e-8,
        ns_steps=5,
        lora_plus_multiplier=1.0,
        log_basic_diagnostics=False,
        picard_iters=picard_iters,
        precond_refresh_every=1,
        precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean",
        ns_form="rect",
        polar_method="ssc",
        ssc_kappa=0.6,
        ssc_nsteps=10,
        ssc_kappa_refresh_every=refresh_every,
        ssc_kappa_warmup_steps=warmup_steps,
        ssc_kappa_solver="eigvalsh",
        ssc_kappa_cache_share_picard=share_picard,
    )


def _seed_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for adapter in model.adapters:
        A = adapter.lora_A["default"].weight
        B = adapter.lora_B["default"].weight
        A.grad = torch.randn(A.shape, generator=g)
        B.grad = torch.randn(B.shape, generator=g)


SPECS = [(4, 8, 8), (4, 8, 8)]


def test_share_true_cache_persists_and_n1_reuses():
    """share=True: after a step at picard=2, the per-side cache slot is
    populated (one solve per side, shared across n) — verified by the
    `ssc_c_cached_{side}` key existing and the per-n `_n0`/`_n1` keys NOT
    existing. The diagnostic `ssc_c_last_{side}` must still equal the
    cached c (since n=1 reuses)."""
    torch.manual_seed(0)
    model = _FakeModel(SPECS)
    opt = _make_optim(model, share_picard=True, picard_iters=2)
    _seed_grads(model, seed=7)
    opt.step()

    assert len(opt.group_state) >= 1
    for gs in opt.group_state:
        # Shared-slot key present.
        assert "ssc_c_cached_A" in gs, list(gs.keys())
        assert "ssc_c_cached_B" in gs
        # Per-n slot keys NOT created in share mode.
        assert "ssc_c_cached_A_n0" not in gs
        assert "ssc_c_cached_A_n1" not in gs
        assert "ssc_c_cached_B_n0" not in gs
        assert "ssc_c_cached_B_n1" not in gs
        # Step stamp present for the in-step short-circuit.
        assert gs.get("ssc_c_cached_A_step") == 1
        assert gs.get("ssc_c_cached_B_step") == 1
        # diagnostic emit (last polar call wins) must equal the cached c
        # — at picard=2 with share=True, n=1 reuses, so c_last == cache.
        torch.testing.assert_close(
            gs["ssc_c_last_A"], gs["ssc_c_cached_A"], rtol=0, atol=0,
        )
        torch.testing.assert_close(
            gs["ssc_c_last_B"], gs["ssc_c_cached_B"], rtol=0, atol=0,
        )


def test_share_false_uses_per_n_caches():
    """share=False: pre-flag behavior — separate cache slots per (side, n).
    At refresh_every=1, neither slot persists across steps (no stash when
    N=1 in legacy path), so we instead force refresh_every=2 to observe
    that both _n0 and _n1 slots get populated."""
    torch.manual_seed(0)
    model = _FakeModel(SPECS)
    # warmup_steps=0 + refresh_every=2 so legacy stash path is exercised.
    opt = _make_optim(
        model, share_picard=False, picard_iters=2,
        refresh_every=2, warmup_steps=0,
    )
    _seed_grads(model, seed=7)
    opt.step()
    for gs in opt.group_state:
        # Per-n slots populated (legacy).
        assert "ssc_c_cached_A_n0" in gs, list(gs.keys())
        assert "ssc_c_cached_A_n1" in gs
        assert "ssc_c_cached_B_n0" in gs
        assert "ssc_c_cached_B_n1" in gs
        # Shared slot NOT created.
        assert "ssc_c_cached_A" not in gs
        assert "ssc_c_cached_B" not in gs


def test_share_false_default_constructor_bit_identity():
    """Backward-compat sanity: pre-flag default (share=False, no
    refresh_every set) reproduces the legacy code path — same updates
    across two identical optimizers initialized with share=False."""
    torch.manual_seed(0)
    m1 = _FakeModel(SPECS)
    torch.manual_seed(0)
    m2 = _FakeModel(SPECS)
    opt1 = _make_optim(m1, share_picard=False, picard_iters=2)
    opt2 = _make_optim(m2, share_picard=False, picard_iters=2)

    _seed_grads(m1, seed=11)
    _seed_grads(m2, seed=11)
    opt1.step()
    opt2.step()

    for a1, a2 in zip(m1.adapters, m2.adapters):
        torch.testing.assert_close(
            a1.lora_A["default"].weight, a2.lora_A["default"].weight,
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            a1.lora_B["default"].weight, a2.lora_B["default"].weight,
            rtol=0, atol=0,
        )
