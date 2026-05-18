"""Regression test: --precond_delta_relative must actually reach the
inverse-sqrt kernel, not get silently dropped.

The bug this guards against:
    `AdamPolarProductLoRA.__init__` accepts `precond_delta_relative=True`
    and stores it as `self.precond_delta_relative`, but `_step_batched`
    calls `spd_inv_sqrt_higham_batched(..., eps=self.delta)` WITHOUT
    passing `eps_relative=...`. The kernel defaults to `eps_relative=False`,
    so relative damping is silently downgraded to ABSOLUTE δ.

    Symptom: ε_rel=1e-2 sweep cells behave as if absolute δ=1e-2, which
    happens to look similar when σ_max(SB)≈O(1) but diverges when σ_max
    leaves that range. Caused a real NaN at r=256 k=3 lr=1e-2.

The check: with eps_relative=True and a Gram whose σ_max=1000, the
resulting Z = (H + δ_eff·I)^(-1/2) has λ_max(ZHZ−I) reflecting damping
~ ε·σ_max = ε·1000, NOT damping ~ ε. Equivalently, the smallest
eigenvalue of the regularized H is ε·σ_max + σ_min, not ε + σ_min.

We test directly: call `spd_inv_sqrt_higham_batched` with eps_relative
True vs False on a high-σ_max matrix and verify the outputs DIFFER
substantially. Then verify the optimizer's batched call site actually
selects the True branch when `precond_delta_relative=True`.
"""
import math

import pytest
import torch

from lora_playground.utils import (
    spd_inv_sqrt_higham_batched, collect_lora_pairs_named,
)
from lora_playground.optim import AdamPolarProductLoRA


# ---------------------------------------------------------------------------
# 1. The kernel itself: relative vs absolute branches produce DIFFERENT Z.
# ---------------------------------------------------------------------------
def test_higham_relative_vs_absolute_branches_differ():
    """If σ_max(H) >> 1, relative damping (ε·σ_max) is much larger than
    absolute (ε), so the two Z outputs must differ measurably."""
    torch.manual_seed(0)
    # Construct H with σ_max ≈ 1000: diag of [1, 1, ..., 1, 1000].
    r = 16
    diag = torch.ones(r); diag[-1] = 1000.0
    H = torch.diag(diag).unsqueeze(0)  # batched: (1, r, r)
    Z_abs = spd_inv_sqrt_higham_batched(H, n_iters=10, eps=1e-2,
                                         eps_relative=False)
    Z_rel = spd_inv_sqrt_higham_batched(H, n_iters=10, eps=1e-2,
                                         eps_relative=True)
    # Absolute: H + 1e-2·I, ~no change to existing diag.
    # Relative: H + 1e-2·1000·I = H + 10·I, big change.
    diff = (Z_abs - Z_rel).abs().max().item()
    assert diff > 0.05, (
        f"Z_abs and Z_rel agree (max diff {diff:.4g}) — the eps_relative "
        f"branch was probably ignored. σ_max=1000, ε=1e-2 → absolute "
        f"adds 0.01, relative adds 10."
    )


# ---------------------------------------------------------------------------
# 2. Optimizer call site: precond_delta_relative must flow into the kernel.
#    We monkeypatch spd_inv_sqrt_higham_batched to capture its kwargs.
# ---------------------------------------------------------------------------
def test_optimizer_passes_eps_relative_to_higham_batched(monkeypatch):
    """Construct an optimizer with precond_delta_relative=True and run
    one batched step. Verify spd_inv_sqrt_higham_batched is called with
    eps_relative=True. (Equivalent test for False to confirm both pass
    through.)"""
    import lora_playground.optim as O
    import lora_playground.utils as U

    captured = []

    def fake_higham(H, n_iters=10, eps=1e-6, n_power_iter=4,
                    eps_relative=False, **kwargs):
        captured.append({"eps": eps, "eps_relative": bool(eps_relative)})
        # Return a valid SPD-ish output so the rest of the step doesn't crash.
        n = H.shape[-1]
        return torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)

    # Patch in both modules — optimizer imports it locally with
    # `from .utils import spd_inv_sqrt_higham_batched`, so we need to
    # patch the utils module's binding (which the local import re-fetches).
    monkeypatch.setattr(U, "spd_inv_sqrt_higham_batched", fake_higham)

    # Minimal fake model: 2 LoRA pairs at r=8 (small) to exercise the
    # batched path.
    r = 8
    A1 = torch.nn.Parameter(torch.randn(r, 16) * 0.1)
    B1 = torch.nn.Parameter(torch.zeros(16, r))
    A2 = torch.nn.Parameter(torch.randn(r, 16) * 0.1)
    B2 = torch.nn.Parameter(torch.zeros(16, r))
    monkeypatch.setattr(O, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [
                            (A1, B1, "pair_0"), (A2, B2, "pair_1")])
    monkeypatch.setattr(U, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [
                            (A1, B1, "pair_0"), (A2, B2, "pair_1")])

    opt = AdamPolarProductLoRA(
        model=None, lr=1e-3, delta=1e-2,
        precond_delta_relative=True,
        precond_method="higham", picard_iters=1,
        magnitude_rule="spectral_chord_tight",
    )

    # Fake gradients (needed by the step).
    A1.grad = torch.randn_like(A1) * 1e-3
    B1.grad = torch.randn_like(B1) * 1e-3
    A2.grad = torch.randn_like(A2) * 1e-3
    B2.grad = torch.randn_like(B2) * 1e-3

    # Run one step. Should call spd_inv_sqrt_higham_batched twice (SA, SB).
    opt.step()

    assert captured, (
        "spd_inv_sqrt_higham_batched was not called — test path didn't "
        "actually exercise the batched precond refresh."
    )
    assert all(c["eps_relative"] for c in captured), (
        f"spd_inv_sqrt_higham_batched was called with eps_relative=False "
        f"even though precond_delta_relative=True at construction. "
        f"Captured: {captured}. This is the silent-drop bug."
    )
    assert all(abs(c["eps"] - 1e-2) < 1e-12 for c in captured), (
        f"eps value mismatched: {captured}"
    )


def test_optimizer_default_does_not_set_eps_relative(monkeypatch):
    """Mirror: without precond_delta_relative=True, eps_relative MUST
    flow as False (default). Catches the other regression direction:
    accidentally always enabling relative damping."""
    import lora_playground.optim as O
    import lora_playground.utils as U

    captured = []

    def fake_higham(H, n_iters=10, eps=1e-6, n_power_iter=4,
                    eps_relative=False, **kwargs):
        captured.append({"eps_relative": bool(eps_relative)})
        n = H.shape[-1]
        return torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)

    monkeypatch.setattr(U, "spd_inv_sqrt_higham_batched", fake_higham)

    r = 8
    A = torch.nn.Parameter(torch.randn(r, 16) * 0.1)
    B = torch.nn.Parameter(torch.zeros(16, r))
    monkeypatch.setattr(O, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [(A, B, "pair_0")])
    monkeypatch.setattr(U, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [(A, B, "pair_0")])

    opt = AdamPolarProductLoRA(
        model=None, lr=1e-3, delta=1e-6,
        # precond_delta_relative defaults to False
        precond_method="higham", picard_iters=1,
        magnitude_rule="spectral_chord_tight",
    )

    A.grad = torch.randn_like(A) * 1e-3
    B.grad = torch.randn_like(B) * 1e-3
    opt.step()

    assert captured
    assert all(c["eps_relative"] is False for c in captured), (
        f"Default path leaked eps_relative=True: {captured}"
    )
