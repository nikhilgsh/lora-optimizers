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
                    precond_refresh_every=1, log_non_finite=False,
                    polar_method="ns", ssc_kappa=None,
                    ssc_kappa_solver="eigvalsh"):
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
        polar_method=polar_method,
        ssc_kappa=ssc_kappa,
        ssc_kappa_solver=ssc_kappa_solver,
        log_non_finite=log_non_finite,
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


def test_site_c_rescale_survives_nullspace_warm_start():
    """Site-C σmax warm starts must not over-scale Picard updates.

    The NaN trace first exposed an A-side Picard effective covector blow-up.
    One local way to create that is for the σmax warm start used to rescale
    `geo_A`/`geo_B` to lie in the current matrix's nullspace: power iteration
    then estimates σ=0 for a nonzero `geo`, making `rho / (sigma + 1e-30)`
    enormous and contaminating the next Picard cross-term.

    This directly exercises the clean polar pipeline with Picard=2 and
    adversarial per-Picard warm-start slots. The crafted rank-one inputs make
    the bad vector `[0, 1]` a nullspace vector for both final `geo_*` matrices.
    """
    torch.manual_seed(0)
    model = FakeLoRAModel([(2, 2, 2)], init_B_nonzero=True)
    opt = _make_optimizer(model, picard_iters=2, ns_form="rect")
    gs = opt.group_state[0]

    A_f = torch.eye(2).view(1, 2, 2)
    B_f = torch.eye(2).view(1, 2, 2)
    u_A = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    u_B = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    eye = torch.eye(2).view(1, 2, 2)
    bad_v = torch.tensor([[0.0, 1.0]])
    gs["v_op_geoA_slots"] = [bad_v.clone(), bad_v.clone()]
    gs["v_op_geoB_slots"] = [bad_v.clone(), bad_v.clone()]

    res = opt._chord_tight_clean_polar_pipeline(
        gs, A_f, B_f, u_A, u_B, eye, eye, opt.param_groups[0]["lr"],
        timer=None, sigma_A=torch.tensor([1.0]), sigma_B=torch.tensor([1.0]),
        step_count=1,
    )

    for name in ("u_A_eff", "u_B_eff", "BT_dB_A", "B_dA_AT", "dA", "dB"):
        assert torch.isfinite(res[name]).all(), f"{name} became non-finite"
    torch.testing.assert_close(res["op_geoA_b"], torch.ones(1), rtol=0, atol=0)
    torch.testing.assert_close(res["op_geoB_b"], torch.ones(1), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.svdvals(res["dA"].float())[..., 0],
        res["rho"].float(), rtol=1e-6, atol=1e-8,
    )
    torch.testing.assert_close(
        torch.linalg.svdvals(res["dB"].float())[..., 0],
        res["rho"].float(), rtol=1e-6, atol=1e-8,
    )


def test_site_c_sigma_guard_hit_is_logged(capsys):
    torch.manual_seed(0)
    model = FakeLoRAModel([(2, 2, 2)], init_B_nonzero=True)
    opt = _make_optimizer(
        model, picard_iters=2, ns_form="rect", log_non_finite=True,
    )
    gs = opt.group_state[0]

    A_f = torch.eye(2).view(1, 2, 2)
    B_f = torch.eye(2).view(1, 2, 2)
    u_A = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    u_B = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    eye = torch.eye(2).view(1, 2, 2)
    bad_v = torch.tensor([[0.0, 1.0]])
    gs["v_op_geoA_slots"] = [bad_v.clone(), bad_v.clone()]
    gs["v_op_geoB_slots"] = [bad_v.clone(), bad_v.clone()]

    opt._chord_tight_clean_polar_pipeline(
        gs, A_f, B_f, u_A, u_B, eye, eye, opt.param_groups[0]["lr"],
        timer=None, sigma_A=torch.tensor([1.0]), sigma_B=torch.tensor([1.0]),
        step_count=1,
    )

    out = capsys.readouterr().out
    assert '"event": "sigma_max_guard_hit"' in out
    assert '"site": "site_c_geo"' in out
    assert '"side": "A"' in out


def test_debug_picard_trace_records_first_iter_tensors():
    torch.manual_seed(0)
    model = FakeLoRAModel([(2, 2, 2)], init_B_nonzero=True)
    opt = _make_optimizer(
        model, picard_iters=2, ns_form="rect", log_non_finite=True,
    )
    gs = opt.group_state[0]

    A_f = torch.eye(2).view(1, 2, 2)
    B_f = torch.eye(2).view(1, 2, 2)
    u_A = torch.eye(2).view(1, 2, 2)
    u_B = torch.eye(2).view(1, 2, 2)
    u_B[0, 0, 0] = float("nan")
    eye = torch.eye(2).view(1, 2, 2)

    res = opt._chord_tight_clean_polar_pipeline(
        gs, A_f, B_f, u_A, u_B, eye, eye, opt.param_groups[0]["lr"],
        timer=None, sigma_A=torch.tensor([1.0]), sigma_B=torch.tensor([1.0]),
        step_count=1,
    )

    trace = res["picard_trace"]
    assert "P_B_n0" in trace
    assert "dB_n0" in trace
    assert "dB_n1" in trace
    assert not torch.isfinite(trace["P_B_n0"]).all()
    assert not torch.isfinite(trace["dB_n0"]).all()


def test_debug_picard_trace_records_ssc_c_by_iteration():
    torch.manual_seed(0)
    model = FakeLoRAModel([(2, 2, 2)], init_B_nonzero=True)
    opt = _make_optimizer(
        model, picard_iters=2, ns_form="rect", log_non_finite=True,
        polar_method="ssc", ssc_kappa=0.6,
    )
    gs = opt.group_state[0]

    A_f = torch.eye(2).view(1, 2, 2)
    B_f = torch.eye(2).view(1, 2, 2)
    u_A = torch.eye(2).view(1, 2, 2)
    u_B = torch.eye(2).view(1, 2, 2)
    eye = torch.eye(2).view(1, 2, 2)

    res = opt._chord_tight_clean_polar_pipeline(
        gs, A_f, B_f, u_A, u_B, eye, eye, opt.param_groups[0]["lr"],
        timer=None, sigma_A=torch.tensor([1.0]), sigma_B=torch.tensor([1.0]),
        step_count=1,
    )

    trace = res["picard_trace"]
    for key in ("ssc_c_A_n0", "ssc_c_B_n0", "ssc_c_A_n1", "ssc_c_B_n1"):
        assert key in trace
        assert torch.isfinite(trace[key]).all()


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


def test_first_step_with_lora_B_zero_init_does_not_nan(tiny_specs):
    """Regression test for the LoRA init edge case:
    `B` initializes to zero (PEFT convention), so on the first step
    σ_max(B) = 0, which is passed as `lam_max=0` into Higham via the
    λ_max hoist. The closed-form damping in Higham must mirror the
    `.clamp_min(1e-12)` that the H-damping line applies, otherwise
    `s = lam_max_damped = 0` and `Y = H / 0 = NaN` — which then
    propagates through Adam (m, v ← NaN) and bricks training. Caught
    in a production smoke; this test would have caught it earlier."""
    torch.manual_seed(0)
    # init_B_nonzero=False is the production default — B starts at 0.
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=False)
    opt = _make_optimizer(model, picard_iters=2)
    _seed_grads(model, seed=0)
    opt.step()
    for ad in model.adapters:
        A_post = ad.lora_A["default"].weight
        B_post = ad.lora_B["default"].weight
        assert torch.isfinite(A_post).all(), \
            f"A NaN/Inf after first step with B=0 init"
        assert torch.isfinite(B_post).all(), \
            f"B NaN/Inf after first step with B=0 init"


def test_higham_with_lam_max_zero_returns_finite():
    """Lower-level regression: passing lam_max=zeros into
    `spd_inv_sqrt_higham_batched` (eps_relative=True path) must
    produce a finite output, matching what the non-hoisted path
    returns when called on the same zero-Gram matrix."""
    from lora_playground.utils import spd_inv_sqrt_higham_batched
    N, r = 2, 4
    H_zero = torch.zeros(N, r, r)
    lam_max_zero = torch.zeros(N)
    # Closed-form path with lam_max=0 — the bug case.
    Z_hoisted = spd_inv_sqrt_higham_batched(
        H_zero.clone(), n_iters=10, eps=1e-2, eps_relative=True,
        lam_max=lam_max_zero,
    )
    assert torch.isfinite(Z_hoisted).all(), \
        f"Higham(H=0, lam_max=0) produced non-finite: {Z_hoisted}"
    # And the non-hoisted path with the SAME zero matrix — should
    # also be finite (sanity check that the fix doesn't regress the
    # internal-power-iter path).
    Z_internal = spd_inv_sqrt_higham_batched(
        H_zero.clone(), n_iters=10, eps=1e-2, eps_relative=True,
        lam_max=None,
    )
    assert torch.isfinite(Z_internal).all(), \
        f"Higham(H=0, lam_max=None) produced non-finite: {Z_internal}"


# ---- κ-adaptive SSC (Appendix C.5 state-dependent c) ----

def _kappa_from_X_c(X, c):
    """Reference κ = ‖H_c(X)‖_F² / (r · h_c(1)²) computed via _ssc_svd."""
    from lora_playground.optim import _ssc_svd
    H = _ssc_svd(X, c=c)
    r = X.shape[-2] if X.shape[-2] <= X.shape[-1] else X.shape[-1]
    h1_sq = 1.0 / (1.0 + 1.0 / (c ** 2))
    # Reduce over the last two axes; keep batch leading dims.
    return H.pow(2).sum(dim=(-2, -1)) / (r * h1_sq)


def test_ssc_adaptive_kappa_recovers_target():
    """Adaptive kernel: realized κ on its output matches the target κ to
    ~1e-3 across a range of targets."""
    from lora_playground.optim import _ssc_adaptive_kappa_batched
    torch.manual_seed(0)
    N, r, d = 4, 8, 32
    X = torch.randn(N, r, d) * 0.5
    # Pre-rescale to σ_max=1 like §2.5 of the algorithm doc.
    sigma_max = torch.linalg.svdvals(X).max(dim=-1).values
    X = X / sigma_max.view(-1, 1, 1)
    for kappa in (0.6, 0.7, 0.85, 0.95):
        H, c = _ssc_adaptive_kappa_batched(X, kappa=kappa, nsteps=20)
        h1_sq = 1.0 / (1.0 + 1.0 / c.pow(2))
        realized = H.pow(2).sum(dim=(-2, -1)) / (r * h1_sq)
        # Per-pair realized κ should match target.
        assert torch.allclose(realized, torch.full_like(realized, kappa),
                              atol=2e-3), \
            f"target κ={kappa}: solved c={c.tolist()}, realized={realized.tolist()}"


def test_ssc_adaptive_matches_fixed_c_round_trip():
    """κ→c bisection round-trips: pick a fixed c, compute κ(c), feed κ to
    the adaptive kernel, recover c."""
    from lora_playground.optim import _ssc_adaptive_kappa_batched
    torch.manual_seed(1)
    r, d = 6, 24
    for trial in range(3):
        X = torch.randn(1, r, d) * 0.5
        sigma_max = torch.linalg.svdvals(X).max(dim=-1).values
        X = X / sigma_max.view(-1, 1, 1)
        for c_target in (0.3, 0.5, 0.7, 1.0):
            kappa = _kappa_from_X_c(X, c_target)  # (1,) per-pair κ at c_target
            _, c_solved = _ssc_adaptive_kappa_batched(
                X, kappa=float(kappa), nsteps=20,
            )
            rel_err = float(((c_solved - c_target).abs() / c_target).max())
            assert rel_err < 1e-3, (
                f"trial={trial} c_target={c_target}: solved c={c_solved.tolist()} "
                f"κ(c_target)={kappa.tolist()} rel_err={rel_err}"
            )


def test_stable_rank_c_exact_on_one_spike_flat_tail():
    """The stable-rank closure is exact for the spectrum it assumes."""
    from lora_playground.optim import _stable_rank_c_from_kappa_batched

    r, d = 8, 16
    kappa = 0.75
    m = 0.08
    singulars = torch.tensor([1.0] + [math.sqrt(m)] * (r - 1))
    X = torch.zeros(1, r, d)
    X[0, torch.arange(r), torch.arange(r)] = singulars

    c = _stable_rank_c_from_kappa_batched(X, kappa=kappa)
    kappa_tail = (r * kappa - 1.0) / (r - 1)
    expected = math.sqrt(m * (1.0 - kappa_tail) / (kappa_tail - m))
    assert torch.allclose(c, torch.tensor([expected]), rtol=2e-5, atol=2e-6)


def test_stable_rank_c_is_finite_on_low_rank_inputs():
    """Rank-concentrated inputs should clip to a finite c, not underflow."""
    from lora_playground.optim import _stable_rank_c_from_kappa_batched

    X = torch.zeros(3, 8, 16)
    X[:, 0, 0] = 1.0
    c = _stable_rank_c_from_kappa_batched(X, kappa=0.75)

    assert torch.isfinite(c).all()
    assert torch.all(c >= 1e-3)
    assert torch.all(c <= 1e3)


def test_stable_rank_solver_routes_without_cache_state(tiny_specs):
    """Optimizer dispatch for stable_rank is stateless and records finite c."""
    torch.manual_seed(0)
    model = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    opt = _make_optimizer(
        model,
        picard_iters=2,
        polar_method="ssc",
        ssc_kappa=0.75,
        ssc_kappa_solver="stable_rank",
    )

    _seed_grads(model, seed=7)
    opt.step()

    for gs in opt.group_state:
        assert "ssc_c_last_A" in gs
        assert "ssc_c_last_B" in gs
        assert torch.isfinite(gs["ssc_c_last_A"]).all()
        assert torch.isfinite(gs["ssc_c_last_B"]).all()
        assert not any(k.startswith("ssc_c_cached") for k in gs)
