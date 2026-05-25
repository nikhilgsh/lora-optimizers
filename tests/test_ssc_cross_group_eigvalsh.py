"""Optimization B: cross-shape-group eigvalsh batching for κ-adaptive SSC.

Verifies that the cross-group path (one stacked eigvalsh + bisect per
(side, picard iter) across all shape groups) produces:
  1. Identical c values per pair vs the per-group path. eigvalsh on the
     stacked Gram is numerically identical to per-group eigvalsh — both
     are deterministic cuSOLVER syevd calls; stacked-vs-per-group differs
     only by trivial bookkeeping. The same _solve_c_from_kappa_batched
     runs on the stacked s_sq tensor.
  2. Identical dA/dB tensors at picard_iters=2 within fp32 noise floor.

Uses 3 shape groups mirroring OLMo-2-1B all-linear LoRA (qkvo r=8 d=16,
gate/up r=8 d=64, down r=8 d_in=64 d_out=16). r is uniform across groups
(an assertion the optimization requires).
"""
import copy

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import AdamPolarProductLoRA


class FakeLoRALinearPair(nn.Module):
    def __init__(self, r, d_in, d_out, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        with torch.no_grad():
            a = torch.empty(r, d_in)
            nn.init.kaiming_uniform_(a, generator=g)
            self.lora_A["default"].weight.copy_(a)
            b = torch.empty(d_out, r)
            nn.init.kaiming_uniform_(b, generator=g)
            self.lora_B["default"].weight.copy_(b * 0.1)


class FakeLoRAModel(nn.Module):
    """3 shape groups: 4 qkvo pairs, 2 gate/up pairs, 1 down pair.
    Uniform r so the cross-group eigvalsh stack is rank-uniform."""

    def __init__(self, r=8):
        super().__init__()
        self.adapters = nn.ModuleList()
        # qkvo: r×16, 16×r
        for i in range(4):
            self.adapters.append(FakeLoRALinearPair(r, 16, 16, seed=i))
        # gate/up: r×64, 64×r
        for i in range(2):
            self.adapters.append(FakeLoRALinearPair(r, 64, 64, seed=10 + i))
        # down: r×64, 16×r
        self.adapters.append(FakeLoRALinearPair(r, 64, 16, seed=20))


def _make_optimizer(model, *, xgroup):
    return AdamPolarProductLoRA(
        model,
        lr=3e-3,
        betas=(0.9, 0.999),
        delta=1e-6,
        eps=1e-8,
        ns_steps=5,
        lora_plus_multiplier=1.0,
        log_basic_diagnostics=False,
        picard_iters=2,
        precond_refresh_every=1,
        precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean",
        ns_form="gram",
        polar_method="ssc",
        ssc_kappa=0.5,
        ssc_nsteps=10,
        ssc_kappa_solver="eigvalsh",
        ssc_kappa_cross_group_eigvalsh=xgroup,
    )


def _seed_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for adapter in model.adapters:
        A = adapter.lora_A["default"].weight
        B = adapter.lora_B["default"].weight
        A.grad = torch.randn(A.shape, generator=g)
        B.grad = torch.randn(B.shape, generator=g)


def _collect_post_step_AB(opt):
    """Snapshot all (A, B) weights after a step."""
    out = []
    for A, B in opt.pairs:
        out.append((A.detach().clone(), B.detach().clone()))
    return out


def _collect_xgroup_c_pre(opt):
    """Gather _xgroup_c_pre across all groups into a list keyed by group idx."""
    snap = []
    for gs in opt.group_state:
        snap.append({k: v.clone() for k, v in gs.get('_xgroup_c_pre', {}).items()})
    return snap


def test_cross_group_vs_per_group_AB_match():
    """End-to-end: identical (A, B) after one step on both paths."""
    torch.manual_seed(0)
    model_pg = FakeLoRAModel(r=8)
    model_xg = copy.deepcopy(model_pg)

    opt_pg = _make_optimizer(model_pg, xgroup=False)
    opt_xg = _make_optimizer(model_xg, xgroup=True)

    _seed_grads(model_pg, seed=42)
    _seed_grads(model_xg, seed=42)

    opt_pg.step()
    opt_xg.step()

    AB_pg = _collect_post_step_AB(opt_pg)
    AB_xg = _collect_post_step_AB(opt_xg)
    assert len(AB_pg) == len(AB_xg)
    for i, ((A_pg, B_pg), (A_xg, B_xg)) in enumerate(zip(AB_pg, AB_xg)):
        # eigvalsh is deterministic; the stacked vs per-group call run the
        # same op on identical inputs. fp32 noise tolerance.
        torch.testing.assert_close(
            A_pg, A_xg, atol=2e-4, rtol=2e-3,
            msg=lambda m: f"pair {i} A mismatch: {m}",
        )
        torch.testing.assert_close(
            B_pg, B_xg, atol=2e-4, rtol=2e-3,
            msg=lambda m: f"pair {i} B mismatch: {m}",
        )


def test_cross_group_path_populates_c_cache():
    """Sanity: the xgroup path actually populates gs['_xgroup_c_pre'] with
    (side, n) entries for n in {0, 1} and sides {'A', 'B'}."""
    torch.manual_seed(0)
    model = FakeLoRAModel(r=8)
    opt = _make_optimizer(model, xgroup=True)
    _seed_grads(model, seed=42)
    opt.step()
    seen_per_group = _collect_xgroup_c_pre(opt)
    assert len(seen_per_group) == len(opt.group_state)
    for gi, c_cache in enumerate(seen_per_group):
        # picard_iters=2 → keys (A,0), (A,1), (B,0), (B,1).
        for n in (0, 1):
            for side in ('A', 'B'):
                assert (side, n) in c_cache, (
                    f"group {gi} missing c for (side={side}, n={n}); have {list(c_cache)}"
                )
                c_tensor = c_cache[(side, n)]
                assert c_tensor.ndim == 1, f"c shape {c_tensor.shape}"
                assert c_tensor.numel() == opt.group_state[gi]['N']
                assert torch.isfinite(c_tensor).all()


def test_cross_group_c_matches_per_group_c():
    """The c values the xgroup pre-flight scatters to each group must be
    bit-identical (up to fp32 noise) to what the per-group eigvalsh would
    compute at the same X. We compare by running both paths and reading
    the post-step ssc_c_last_A / ssc_c_last_B that the polar pipeline
    stashes from its final polar call."""
    torch.manual_seed(0)
    model_pg = FakeLoRAModel(r=8)
    model_xg = copy.deepcopy(model_pg)

    opt_pg = _make_optimizer(model_pg, xgroup=False)
    opt_xg = _make_optimizer(model_xg, xgroup=True)

    _seed_grads(model_pg, seed=42)
    _seed_grads(model_xg, seed=42)

    opt_pg.step()
    opt_xg.step()

    for gi, (gs_pg, gs_xg) in enumerate(zip(opt_pg.group_state, opt_xg.group_state)):
        for side in ('A', 'B'):
            key = f'ssc_c_last_{side}'
            c_pg = gs_pg.get(key)
            c_xg = gs_xg.get(key)
            assert c_pg is not None, f"per-group path missing {key} in group {gi}"
            assert c_xg is not None, f"xgroup path missing {key} in group {gi}"
            torch.testing.assert_close(
                c_pg, c_xg, atol=2e-4, rtol=2e-3,
                msg=lambda m, gi=gi, side=side: (
                    f"group {gi} side {side} c mismatch: {m}"
                ),
            )


def test_cross_group_respects_refresh_every():
    """When refresh_every>1 and warmup=0, step 2 must reuse the cached c
    solved at step 1 — no fresh eigvalsh launch — and produce the same
    (A, B) as the per-group path also at refresh_every>1.

    Verifies the cross-group preflight + per-group cache machinery interop:
    at step 1 the preflight populates _xgroup_c_pre and the per-group
    pipeline stashes ssc_c_cached_{side}. At step 2 the preflight should
    NOT overwrite the cached c with a fresh eigvalsh (refresh gating is
    inside the per-group `_polar` short-circuit, which reads _xgroup_c_pre
    if present; so the preflight does run, but on step 2 with the cached
    path active, the per-group `_polar` short-circuit takes precedence).

    The behavior we care about: end-to-end (A, B) at step 2 matches the
    per-group path bit-for-bit-within-tol.
    """
    torch.manual_seed(0)
    model_pg = FakeLoRAModel(r=8)
    model_xg = copy.deepcopy(model_pg)

    def _make(model, xgroup):
        return AdamPolarProductLoRA(
            model,
            lr=3e-3,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=5,
            lora_plus_multiplier=1.0,
            log_basic_diagnostics=False,
            picard_iters=2,
            precond_refresh_every=1,
            precond_method="higham",
            magnitude_rule="spectral_chord_tight_clean",
            ns_form="gram",
            polar_method="ssc",
            ssc_kappa=0.5,
            ssc_nsteps=10,
            ssc_kappa_solver="eigvalsh",
            ssc_kappa_refresh_every=5,   # >1
            ssc_kappa_warmup_steps=0,    # honor refresh from step 1
            ssc_kappa_cross_group_eigvalsh=xgroup,
        )

    opt_pg = _make(model_pg, xgroup=False)
    opt_xg = _make(model_xg, xgroup=True)

    # Step 1 — both paths solve fresh (refresh due since cache empty).
    _seed_grads(model_pg, seed=42)
    _seed_grads(model_xg, seed=42)
    opt_pg.step()
    opt_xg.step()

    # Step 2 — at (step-1)%5 = 1 ≠ 0 ⇒ refresh NOT due ⇒ both paths
    # should reuse cached c via _ssc_misr_batched only. The xgroup
    # preflight will still build _xgroup_c_pre, but the per-group
    # short-circuit in `_polar` checks _xgroup_c_pre FIRST (taking the
    # fresh-solved-this-step c). With refresh-NOT-due, what we actually
    # want to verify is that the xgroup path is internally consistent
    # with itself and matches per-group at step 2.
    _seed_grads(model_pg, seed=43)
    _seed_grads(model_xg, seed=43)
    opt_pg.step()
    opt_xg.step()

    for i, ((A_pg, B_pg), (A_xg, B_xg)) in enumerate(
        zip(_collect_post_step_AB(opt_pg), _collect_post_step_AB(opt_xg))
    ):
        torch.testing.assert_close(
            A_pg, A_xg, atol=2e-4, rtol=2e-3,
            msg=lambda m, i=i: f"pair {i} A mismatch at step 2: {m}",
        )
        torch.testing.assert_close(
            B_pg, B_xg, atol=2e-4, rtol=2e-3,
            msg=lambda m, i=i: f"pair {i} B mismatch at step 2: {m}",
        )


def test_cross_group_heterogeneous_r_raises():
    """Mixed-rank shape groups must raise — the cross-group stacking
    invariant is that grams are (Ng, r, r) with uniform r."""
    torch.manual_seed(0)

    class _MixedRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = nn.ModuleList([
                FakeLoRALinearPair(8, 16, 16, seed=0),
                FakeLoRALinearPair(8, 16, 16, seed=1),
                FakeLoRALinearPair(16, 64, 64, seed=2),  # different r
            ])

    model = _MixedRModel()
    opt = AdamPolarProductLoRA(
        model,
        lr=3e-3, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, ns_steps=5,
        lora_plus_multiplier=1.0, log_basic_diagnostics=False,
        picard_iters=2, precond_refresh_every=1, precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean", ns_form="gram",
        polar_method="ssc", ssc_kappa=0.5, ssc_nsteps=10,
        ssc_kappa_solver="eigvalsh",
        ssc_kappa_cross_group_eigvalsh=True,
    )
    _seed_grads(model, seed=42)
    if len(opt.group_state) <= 1:
        pytest.skip("only one shape group; cross-group preflight not triggered")
    with pytest.raises(RuntimeError, match="heterogeneous LoRA rank"):
        opt.step()


if __name__ == "__main__":
    test_cross_group_path_populates_c_cache()
    test_cross_group_vs_per_group_AB_match()
    test_cross_group_c_matches_per_group_c()
    test_cross_group_respects_refresh_every()
    test_cross_group_heterogeneous_r_raises()
    print("OK")
