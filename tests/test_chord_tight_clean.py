"""Unit coverage for `_chord_tight_clean_polar_pipeline`
(`spectral_chord_tight_clean` magnitude rule).

The end-to-end smokes in the wall-time and ablation logs catch gross
trajectory regressions, but they cannot localize silent drift in the
cross-coupling formula, the Picard-iter-keyed warm-start, or the
per-step σ-rescale. This file exercises those properties on a tiny
CPU fixture so a regression surfaces in CI instead of in a sweep.

Tests:
- `test_sigma_AB_rho_formula` — pipeline outputs σ_A, σ_B, ρ that
  satisfy ρ = η/(σ_A + σ_B) within power-iter convergence noise.
- `test_post_polar_unit_op_norm` — after pre-rescale (§2.5) and the
  Newton–Schulz polar map (§2.6) on a whitened input with cond ≪ ∞,
  σ_max(P_A) is close to 1 (polar invariant).
- `test_update_op_norm_matches_rho` — after the magnitude rescale,
  σ_max(dA) ≈ ρ — this is the "tight-tangent radius" property that
  the rule is named for.
- `test_determinism` — same inputs, same seed → bit-identical
  updates across two independent optimizers.
- `test_lam_max_hoist_equivalence` — passing `sigma_A` / `sigma_B`
  into the pipeline produces the same Higham-damped preconditioner
  (within Higham-residual tolerance) as letting Higham re-derive
  λ_max internally.
"""
import copy
import math

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import AdamPolarProductLoRA


class FakeLoRALinearPair(nn.Module):
    """Matches the PEFT-orientation `lora_A`/`lora_B` layout that
    `collect_lora_pairs` walks. lora_A: (r, d_in); lora_B: (d_out, r)."""

    def __init__(self, r, d_in, d_out, seed=0, init_B_nonzero=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        with torch.no_grad():
            a = torch.empty(r, d_in)
            nn.init.kaiming_uniform_(a, generator=g)
            self.lora_A["default"].weight.copy_(a)
            if init_B_nonzero:
                # Non-trivial B so σ_max(B) > 0 from step 0. Needed for
                # tests where we want s = σ_A + σ_B both contributing.
                b = torch.empty(d_out, r)
                nn.init.kaiming_uniform_(b, generator=g)
                self.lora_B["default"].weight.copy_(b * 0.1)
            else:
                self.lora_B["default"].weight.zero_()


class FakeLoRAModel(nn.Module):
    def __init__(self, group_specs, init_B_nonzero=False):
        super().__init__()
        self.adapters = nn.ModuleList()
        for i, (r, d_in, d_out) in enumerate(group_specs):
            self.adapters.append(FakeLoRALinearPair(
                r, d_in, d_out, seed=i, init_B_nonzero=init_B_nonzero,
            ))


def _make_optimizer(model, *, picard_iters=2, ns_form="rect",
                    precond_refresh_every=1):
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
        precond_refresh_every=precond_refresh_every,
        precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean",
        ns_form=ns_form,
    )


def _seed_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for adapter in model.adapters:
        A = adapter.lora_A["default"].weight
        B = adapter.lora_B["default"].weight
        A.grad = torch.randn(A.shape, generator=g)
        B.grad = torch.randn(B.shape, generator=g)


def _capture_pipeline(optim):
    """Wrap `_chord_tight_clean_polar_pipeline` so we can inspect the
    locals it returns. Bound-method wrapping is sufficient because the
    method already returns a dict of every interesting tensor."""
    captured = {}
    orig = optim._chord_tight_clean_polar_pipeline.__func__

    def _wrapped(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        captured["result"] = result
        return result

    optim._chord_tight_clean_polar_pipeline = _wrapped.__get__(optim)
    return captured


@pytest.fixture
def tiny_specs():
    # Two LoRA pairs sharing the same shape so they batch into one
    # group. r=4 / d=8 is small enough for fp32 + 8-iter power iter to
    # converge cleanly (cond is moderate after one step of Adam).
    return [(4, 8, 8), (4, 8, 8)]


def test_sigma_AB_rho_formula(tiny_specs):
    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    opt = _make_optimizer(model, picard_iters=2)
    captured = _capture_pipeline(opt)

    _seed_grads(model, seed=7)
    opt.step()

    res = captured["result"]
    sigma_A = res["sigma_A"].detach()
    sigma_B = res["sigma_B"].detach()
    rho = res["rho"].detach()
    lr = opt.param_groups[0]["lr"]

    expected_rho = lr / (sigma_A + sigma_B + 1e-30)
    torch.testing.assert_close(rho, expected_rho, rtol=1e-6, atol=1e-8)


def test_post_polar_unit_op_norm(tiny_specs):
    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    opt = _make_optimizer(model, picard_iters=1, ns_form="rect")
    captured = _capture_pipeline(opt)

    _seed_grads(model, seed=11)
    opt.step()

    P_A = captured["result"]["P_A"].detach().float()
    P_B = captured["result"]["P_B"].detach().float()
    # Polar(X) has σ_max = 1 in exact arithmetic; NS=5 quintic on a
    # whitened, pre-rescaled (σ_max(X)=1) input lands well within 5%.
    sA = torch.linalg.svdvals(P_A)[..., 0]
    sB = torch.linalg.svdvals(P_B)[..., 0]
    assert torch.allclose(sA, torch.ones_like(sA), atol=5e-2), sA
    assert torch.allclose(sB, torch.ones_like(sB), atol=5e-2), sB


def test_update_op_norm_matches_rho(tiny_specs):
    """The tight-tangent radius: after the §2.6 magnitude rescale
    `dA = -ρ · geo_A / σ_max(geo_A)`, σ_max(dA) = ρ exactly (up to
    the residual of the σ_max(geo) power iter)."""
    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    opt = _make_optimizer(model, picard_iters=2)
    captured = _capture_pipeline(opt)

    _seed_grads(model, seed=13)
    opt.step()

    dA = captured["result"]["dA"].detach().float()
    dB = captured["result"]["dB"].detach().float()
    rho = captured["result"]["rho"].detach().float()
    sA = torch.linalg.svdvals(dA)[..., 0]
    sB = torch.linalg.svdvals(dB)[..., 0]
    # σ_max(dA) = ρ · σ_max(geo_A) / σ_max(geo_A) = ρ exactly in exact
    # arithmetic. Power-iter residual + NS sub-orthogonality give a
    # small slack; 5% is the same band as the polar-invariant test.
    torch.testing.assert_close(sA, rho, rtol=5e-2, atol=1e-6)
    torch.testing.assert_close(sB, rho, rtol=5e-2, atol=1e-6)


def test_determinism(tiny_specs):
    torch.manual_seed(0)
    model_a = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    model_b = copy.deepcopy(model_a)
    opt_a = _make_optimizer(model_a, picard_iters=2)
    opt_b = _make_optimizer(model_b, picard_iters=2)

    for step in range(3):
        _seed_grads(model_a, seed=100 + step)
        _seed_grads(model_b, seed=100 + step)
        opt_a.step()
        opt_b.step()

    for ad_a, ad_b in zip(model_a.adapters, model_b.adapters):
        torch.testing.assert_close(
            ad_a.lora_A["default"].weight, ad_b.lora_A["default"].weight,
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            ad_a.lora_B["default"].weight, ad_b.lora_B["default"].weight,
            rtol=0, atol=0,
        )


def test_no_graph_breaks_under_compile(tiny_specs):
    """`_chord_tight_clean_polar_pipeline` must compile under
    `fullgraph=True` — i.e., no graph breaks. Catches regressions
    where a Python-only construct (dict-key f-string mutation, `.item()`
    call, host-side branching on a tensor value, prints) forces dynamo
    to fall back to eager in the middle of the optimizer step.

    Historical note: the slot-list refactor at `optim.py:3558` replaced
    per-Picard-iter f-string dict keys (`f'v_op_geoA_n{n}'`) with
    preallocated lists; without that change this test would fail."""
    import torch._dynamo
    torch._dynamo.reset()

    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    opt = _make_optimizer(model, picard_iters=2)

    raw = AdamPolarProductLoRA._chord_tight_clean_polar_pipeline
    compiled = torch.compile(raw, fullgraph=True, dynamic=False)
    opt._chord_tight_clean_polar_pipeline = compiled.__get__(opt)

    _seed_grads(model, seed=42)
    # `opt.step()` will raise torch._dynamo.exc.Unsupported on any
    # graph break under fullgraph=True. No explicit assertion needed.
    opt.step()


def test_lam_max_hoist_equivalence(tiny_specs):
    """The λ_max hoist must not change Higham's output: passing
    `lam_max=σ_max(A)²` should produce the same `SA_half_inv` (within
    Higham residual tolerance) as letting Higham re-derive it via its
    internal power iter."""
    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    A_stack = torch.stack([
        ad.lora_A["default"].weight.detach().float() for ad in model.adapters
    ])
    SA = A_stack @ A_stack.transpose(-2, -1)

    from lora_playground.utils import spd_inv_sqrt_higham_batched
    inv_internal = spd_inv_sqrt_higham_batched(
        SA, n_iters=10, eps=1e-6, eps_relative=True, lam_max=None,
    )
    sigma_A = torch.linalg.svdvals(A_stack)[..., 0]  # (N,)
    inv_hoisted = spd_inv_sqrt_higham_batched(
        SA, n_iters=10, eps=1e-6, eps_relative=True,
        lam_max=sigma_A.pow(2),
    )
    # Same matrix, same iteration count, only difference is the
    # internal λ_max estimate (power iter on S_A) vs the exact value
    # σ_max(A)² — they should agree to a few power-iter residuals.
    rel = (inv_internal - inv_hoisted).norm() / inv_internal.norm()
    assert rel < 1e-3, f"hoisted Higham differs by rel_err={rel:.2e}"
