"""Regression test for chord-tight whiten + Init[A] dA-collapse bug.

Background. With Init[A] (B initialized to exactly zero, the PEFT default),
the chord-tight whiten batched optimizer was leaving parameter A frozen
forever. Mechanism:
  * Step 1: u_A = 0 because gA = 0 (forward independent of A when B=0).
            Several intermediate quantities are zero (X_A_pre, geo_A).
            The warm-start power-iter vectors (v_sigma_XA, v_sigma_B,
            v_op_geoA) stored in shape-group state are exactly zero.
  * Step 2+: power-iter is called with v_init = 0; the iteration
             v ← M·M^T·v / ||·|| preserves the zero vector, so σ_max
             is returned as 0 regardless of M. Caller does u_A /= σ_max+eps
             → u_A explodes; NS overflows; downstream dA collapses to 0.
  * Per-pair path is unaffected because `_sigma_max_power_iter` doesn't
    accept warm-start in any production call site and re-initializes
    deterministically every step.

The fix in `_sigma_max_power_iter_batched` adds a sticky-zero guard:
when v_init has degenerate (≤ eps) norm in a batch element, fall back
to the deterministic M·ones init for that element.

This test exercises a single LoRA pair through two batched steps with
Init[A]-realistic gradients and asserts that ||dA|| at step 2 is
non-trivial. CPU-only is fine; the bug reproduces identically.
"""
import math

import pytest
import torch
import torch.nn as nn

import lora_playground.optim as O
import lora_playground.utils as U


def _make_optim(A, B, monkeypatch):
    monkeypatch.setattr(
        U, "collect_lora_pairs",
        lambda model, adapter_name=None: [(A, B)],
    )
    monkeypatch.setattr(
        O, "collect_lora_pairs",
        lambda model, adapter_name=None: [(A, B)],
    )
    monkeypatch.setattr(
        U, "collect_lora_pairs_named",
        lambda model, adapter_name=None: [(A, B, "test_pair_0")],
    )
    monkeypatch.setattr(
        O, "collect_lora_pairs_named",
        lambda model, adapter_name=None: [(A, B, "test_pair_0")],
    )
    return O.AdamPolarProductLoRA(
        model=None, lr=1e-2, delta=1e-6, eps=1e-8, betas=(0.9, 0.999),
        ns_steps=5, precond_method="higham", higham_iters=10,
        magnitude_rule="spectral_chord_tight", operator_type="polar",
        polar_method="ns", polar_norm_dir="frob",
        picard_iters=1, lora_plus_multiplier=1.0,
    )


def _make_init_a_pair(r=32, d_in=64, d_out=64, device="cpu", seed=0):
    """A: Kaiming-uniform (matches PEFT lora_A). B: zero (Init[A] / PEFT default)."""
    g = torch.Generator(device=device).manual_seed(seed)
    A_init = torch.empty(r, d_in, device=device)
    nn.init.kaiming_uniform_(A_init, a=math.sqrt(5), generator=g)
    B_init = torch.zeros(d_out, r, device=device)
    return nn.Parameter(A_init), nn.Parameter(B_init)


def _step_pair(opt, A, B, gA, gB, step_fn):
    A.grad = gA.clone()
    B.grad = gB.clone()
    A_pre = A.data.clone()
    B_pre = B.data.clone()
    step_fn()
    return (A.data - A_pre).clone(), (B.data - B_pre).clone()


@pytest.mark.parametrize("step_fn_name", ["_step_batched", "_step_per_pair"])
def test_chord_tight_whiten_init_a_dA_nontrivial(step_fn_name, monkeypatch):
    """Both code paths must produce nonzero dA at step 2 of Init[A].

    Step 1 dA is expected to be zero (gA = 0 since B = 0).
    Step 2 dA must be non-trivial (gA nonzero now that B has been updated).
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r, d_in, d_out = 32, 64, 64

    A, B = _make_init_a_pair(r=r, d_in=d_in, d_out=d_out, device=device)
    opt = _make_optim(A, B, monkeypatch)
    step_fn = getattr(opt, step_fn_name)

    # Step 1 — gA is exactly zero (B=0 → ∂L/∂A = 0). gB nonzero.
    gA_1 = torch.zeros_like(A)
    torch.manual_seed(1)
    gB_1 = torch.randn_like(B)
    dA_1, dB_1 = _step_pair(opt, A, B, gA_1, gB_1, step_fn)

    assert dA_1.norm().item() == 0.0, (
        f"Step 1 dA must be zero when gA=0 (Init[A]). Got ||dA||={dA_1.norm().item():.4e}"
    )
    assert dB_1.norm().item() > 1e-4, (
        f"Step 1 dB must be nonzero (gB≠0). Got ||dB||={dB_1.norm().item():.4e}"
    )

    # Step 2 — B has been updated, so gA flowing through B^T is nonzero.
    torch.manual_seed(2)
    gA_2 = 1e-3 * torch.randn_like(A)
    gB_2 = torch.randn_like(B)
    dA_2, dB_2 = _step_pair(opt, A, B, gA_2, gB_2, step_fn)

    assert torch.isfinite(dA_2).all(), "dA at step 2 must be finite (no NaN/inf)"
    assert dA_2.norm().item() > 1e-6, (
        f"Step 2 dA must be non-trivial on path {step_fn_name!r} when gA≠0. "
        f"Got ||dA||={dA_2.norm().item():.4e}. This is the chord-tight + Init[A] "
        f"sticky-zero warm-start bug — see _sigma_max_power_iter_batched."
    )


def test_sigma_max_opt_vs_exact_logged(monkeypatch):
    """`sigma_{A,B}_relerr` must be present in the diag record and small.
    Regression guard: if a future change re-introduces a degenerate
    warm-start path (or any σ_max estimator that drifts from the exact
    eigvalsh-of-Gram value), this probe surfaces it in logs."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r, d_in, d_out = 32, 64, 64
    A, B = _make_init_a_pair(r=r, d_in=d_in, d_out=d_out, device=device)
    # Use Init[AB]-flavored B (nonzero) so step 1 itself exercises the
    # full chord-tight path (no zero-gradient degeneracy).
    with torch.no_grad():
        B.data.copy_(0.01 * torch.randn_like(B))
    monkeypatch.setattr(U, "collect_lora_pairs",
                        lambda model, adapter_name=None: [(A, B)])
    monkeypatch.setattr(O, "collect_lora_pairs",
                        lambda model, adapter_name=None: [(A, B)])
    monkeypatch.setattr(U, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [(A, B, "test_pair_0")])
    monkeypatch.setattr(O, "collect_lora_pairs_named",
                        lambda model, adapter_name=None: [(A, B, "test_pair_0")])
    opt = O.AdamPolarProductLoRA(
        model=None, lr=1e-2, delta=1e-6, eps=1e-8, betas=(0.9, 0.999),
        ns_steps=5, precond_method="higham", higham_iters=10,
        magnitude_rule="spectral_chord_tight", operator_type="polar",
        polar_method="ns", polar_norm_dir="frob",
        picard_iters=1, lora_plus_multiplier=1.0,
        log_basic_diagnostics=True, diagnostics_every=1,
    )

    # Capture emitted records
    import lora_playground.optim as O_mod
    captured = []
    real_emit = O_mod._emit_optim_diagnostics
    def _capture_emit(step, recs):
        captured.extend(recs)
    monkeypatch.setattr(O_mod, "_emit_optim_diagnostics", _capture_emit)

    torch.manual_seed(1)
    A.grad = 1e-3 * torch.randn_like(A)
    B.grad = torch.randn_like(B)
    opt._step_batched()

    assert captured, "expected at least one diag record"
    rec = captured[0]
    for k in ("sigma_A_opt", "sigma_A_exact", "sigma_A_relerr",
              "sigma_B_opt", "sigma_B_exact", "sigma_B_relerr"):
        assert k in rec, f"missing diag field {k!r}; rec keys = {list(rec.keys())}"
    # The power-iter estimator with the v_init guard should be within ~5% of
    # exact on typical Gauss-like factors (looser than per-pair's tolerance
    # since batched runs in lower precision under bf16 NS).
    assert rec["sigma_A_relerr"] < 0.1, f"sigma_A_relerr={rec['sigma_A_relerr']}"
    assert rec["sigma_B_relerr"] < 0.1, f"sigma_B_relerr={rec['sigma_B_relerr']}"


def test_batched_vs_per_pair_dA_agree_after_init_a_step1(monkeypatch):
    """After the step-1-with-gA=0 trap, batched and per-pair must produce
    dA of the same order of magnitude at step 2 (within a loose tolerance —
    they aren't bit-equivalent because of unit-polar normalization and
    NS-bf16 quantization, but the order of magnitude must match)."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r, d_in, d_out = 32, 64, 64

    dAs = {}
    for step_fn_name in ("_step_batched", "_step_per_pair"):
        A, B = _make_init_a_pair(r=r, d_in=d_in, d_out=d_out, device=device)
        opt = _make_optim(A, B, monkeypatch)
        step_fn = getattr(opt, step_fn_name)
        # Step 1 with gA=0
        torch.manual_seed(1)
        gB_1 = torch.randn_like(B)
        _step_pair(opt, A, B, torch.zeros_like(A), gB_1, step_fn)
        # Step 2 with gA small nonzero
        torch.manual_seed(2)
        gA_2 = 1e-3 * torch.randn_like(A)
        gB_2 = torch.randn_like(B)
        dA_2, _ = _step_pair(opt, A, B, gA_2, gB_2, step_fn)
        dAs[step_fn_name] = dA_2.norm().item()

    ratio = dAs["_step_batched"] / max(dAs["_step_per_pair"], 1e-30)
    assert 0.1 < ratio < 10.0, (
        f"||dA||_batched / ||dA||_per_pair = {ratio:.4e} "
        f"(batched={dAs['_step_batched']:.4e}, per_pair={dAs['_step_per_pair']:.4e}). "
        f"Order of magnitude must match — large discrepancy indicates regression "
        f"in one of the two paths."
    )
