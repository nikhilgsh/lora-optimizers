# Batched-step integration plan for AdamPolarProductLoRA

**Status:** primitives validated, integration not yet implemented. Higham dropped from the plan — `batched_eigh + K=5` reaches the same wall-time target without numerical risk.

## Goal

Refactor `AdamPolarProductLoRA.step()` so its hot path:
1. Iterates per shape group (3 groups of 64/32/16 pairs at OLMo-2-1B r=16) instead of per-pair (112 pairs).
2. Calls validated batched primitives for every component except picard cross-coupling (which is compute-bound — keep loop).
3. Uses foreach for adam_direction and apply.
4. Runs NS in bf16 (still validates all earlier convergence checks; same residual band as fp32).

Projected total-step impact at OLMo-2-1B r=16, bf16 fwd/bwd:

| variant | current ms | post-integration | × AdamW |
|---|---:|---:|---:|
| polar k=1 | 1655 | ~1479 | **1.007×** |
| polar k=3 | 1873 | ~1543 | **1.051×** |

For comparison: AdamW total step is 1468 ms.

## Validated primitives (microbench + regression test)

All in repo today:

| primitive | location | speedup | tests |
|---|---|---:|---|
| `_newton_schulz_batched` | `lora_playground/optim.py` | 5.3× fp32 / +3.6× bf16 | `tests/test_newton_schulz_batched.py` |
| `spd_inv_sqrt_higham_batched` | `lora_playground/utils.py` | 40–125× | `tests/test_spd_inv_sqrt_batched.py` |
| `unwhiten_rescale_frob_batched` | `lora_playground/_batched_polar.py` | 19× | `tests/test_unwhiten_rescale_batched.py` |
| `_foreach`-style adam direction | (helper to add to `optim.py`) | 6.6× | bench-only currently |
| `_foreach`-style apply | trivial | est ~5× | bench-only currently |

Also confirmed **NOT to batch**: `picard_cross_coupling` (compute-bound, 0.97× speedup; keep per-pair loop).

## Precond_refresh choice: batched_eigh + K=5 (higham dropped)

`batched_higham` is faster on paper (40-125× vs 1.28-125× for `batched_eigh`) but has empirical failures at r=256 (`polar_K1_r256_rerun_2k`: NaN at step 800; `polar_K_higham_r256_2k`: Δ=0.0033 vs eigh, ~5σ outside noise). Compound error from the 1e-4 per-step approximation drives the failures.

`batched_eigh` is bit-near-exact (1e-7 vs eigh-truth) but only 1.28× faster than loop_eigh at r=64. Without caching, that's a 82 ms / step penalty at r=64.

**The K knob closes the gap.** From the project's prior K-sweep at r=64 (`logs/polar_K_sweep_r64_2k`):

| K | final eval loss | Δ vs K=1 |
|---:|---:|---:|
| 1 | 0.745436 | — |
| 2 | 0.745289 | −0.00015 |
| **5** | **0.745451** | **+0.00002** |
| 10 | 0.745890 | +0.00045 |
| 20 | 0.746203 | +0.00077 |

K=5 is empirically free at r=64 (well within σ_AdamW ≈ 0.0007). Combined with batched_eigh K=5, amortized precond_refresh is 82/5 = 16 ms / step at r=64 — closes ~80% of the gap to batched_higham.

Updated wall-time projection at r=64 with `batched_eigh + K=5`:
- k=1: total step 1492 ms = 1.016× AdamW
- k=3: total step 1554 ms = 1.058× AdamW

vs the batched_higham K=1 projection of 1.051× — 0.7% wall difference, negligible.

**Decision: ship batched_eigh + K=5 default.** Zero algorithmic risk, zero higham integration test needed. The K=5 default for AdamPolarProductLoRA may need a small constructor change (currently defaults to K=1).

Already done independently: deterministic init for higham power iteration (`utils.py` updated) — keeps higham available as an option but not the default. Useful regardless of the precond_refresh choice since the random `randn` init was a reproducibility bug.

## Refactor design

### Pair shape groups (computed once, in `__init__`)

```python
shape_groups: dict[tuple, list[int]]  # (A.shape, B.shape) → list of pair indices
```

For OLMo-2-1B r=16: 3 groups (64, 32, 16 pairs each).

Per-group state stacked once in `__init__` and held alongside the existing `pair_state` dict for diagnostic-path compatibility:

```python
self.group_state: list[dict] = [
    {
        "indices": [global_pair_idx, ...],        # for scatter back
        "A_stack": Tensor (N, r, d_in),           # views into A_i
        "B_stack": Tensor (N, d_out, r),
        "m_A_stack": Tensor (N, r, d_in),
        ...
    },
    ...
]
```

If `A_stack` etc. are **views** into the original parameter tensors (via `torch.stack` of views — actually requires concatenation, not stacking views), we avoid extra memory. Most likely we keep stacks as separate tensors and copy-in / copy-back per step. Memory cost: 2× the LoRA params for the stacks.

### Hot path (`_step_batched`)

```python
def _step_batched(self):
    timer = getattr(self, "_step_timer", None)
    lr = self.param_groups[0]["lr"]

    # 1. Collect grads, refresh stacks (copy from per-pair tensors).
    for group in self.group_state:
        # gA_stack[i] := pairs[indices[i]][0].grad.float()
        # gB_stack[i] := pairs[indices[i]][1].grad.float()
        ...

    # 2. Adam direction, foreach across all stacks.
    with maybe_time(timer, "adam_direction"):
        all_m_A = [g["m_A_stack"] for g in self.group_state]
        all_g_A = [g["gA_stack"] for g in self.group_state]
        torch._foreach_mul_(all_m_A, beta1)
        torch._foreach_add_(all_m_A, all_g_A, alpha=1-beta1)
        ... (v_A, m_B, v_B same pattern)
        # u_A_stack[i] = (m_A_stack[i] / bc1) / (sqrt(v_A_stack[i] / bc2) + eps)
        ...

    # 3. Per-group: precond_refresh, picard, polar, apply.
    for group in self.group_state:
        if step % K == 0:
            with maybe_time(timer, "precond_refresh"):
                SA = group["A_stack"] @ group["A_stack"].transpose(-2,-1)
                SB = group["B_stack"].transpose(-2,-1) @ group["B_stack"]
                group["SA_half_inv"] = spd_inv_sqrt_higham_batched(SA, ...)
                group["SB_half_inv"] = spd_inv_sqrt_higham_batched(SB, ...)
        SA_inv, SB_inv = group["SA_half_inv"], group["SB_half_inv"]
        u_A_eff, u_B_eff = group["u_A"], group["u_B"]
        dA_prev = torch.zeros_like(group["A_stack"])
        dB_prev = torch.zeros_like(group["B_stack"])

        for k_iter in range(self.picard_iters):
            with maybe_time(timer, "picard_cross_coupling"):
                if k_iter > 0:
                    # Per-pair loop INSIDE the group — not bmm. Empirically
                    # bmm is 0.97× here (compute-bound large-d contraction).
                    for j in range(group_size):
                        u_A_eff[j] = u + alpha * (B[j].T @ dB_prev[j] @ A[j]) / lr
                        ...
            with maybe_time(timer, "polar_pipeline"):
                X_A = SB_inv @ u_A_eff
                X_B = u_B_eff @ SA_inv
                P_A = _newton_schulz_batched(X_A, j=5, dtype=bf16).float()
                P_B = _newton_schulz_batched(X_B, j=5, dtype=bf16).float()
                dA, dB = unwhiten_rescale_frob_batched(P_A, P_B, SA_inv, SB_inv, u_A_eff, u_B_eff, lr, lora_plus_mul)
            dA_prev, dB_prev = dA, dB

    # 4. Apply, foreach across all stacks.
    with maybe_time(timer, "apply"):
        # Cast back to A/B dtypes; in-place add into original params.
        ...
```

### Compatibility branches

The current step has several non-default branches that must keep working:
- `core_remix_alpha > 0`: small per-pair compute in u_A, u_B; can be batched within the group iteration above
- `exact_chord`: Picard recompute with refreshed precond. Easy enough; same shape contract
- `anderson_m > 0`: history-flatten + linear solve. Stays per-step (not per-pair)
- `end_rms_align`: alternate magnitude rule. Per-group elementwise scale, easy
- `log_diagnostics`: emits per-pair stats. **Diagnostic path stays per-pair.** Detected via flag at top of step; if set, fall through to existing `_step_per_pair` (the current code, renamed).

Only the production-default path (all flags off, picard_iters ∈ {1, 3}) goes through `_step_batched`.

### Behavioral equivalence test

The **single most important regression test** for the integration. Without it, we can't claim the batched optimizer matches the per-pair optimizer's behavior.

Plan: a unit test that:
1. Builds a tiny LoRA model (4 pairs of 2 distinct shapes).
2. Instantiates the per-pair optimizer and the batched optimizer with identical state.
3. Runs N=5 steps with deterministic gradients (seeded).
4. Compares dA, dB at each step against the per-pair reference. Tolerance: 1e-5 fp32, 1e-2 bf16 NS (since bf16 NS only matches the polar to 1e-3).

Locate at `tests/test_polar_product_batched_equivalence.py`. Run on every commit.

## Implementation order

1. **Add `_batched_eigh_inv_half`** to `utils.py` (one-liner; already prototyped in `bench_precond_refresh.py`)
2. **Add foreach helpers** to `optim.py`:
   - `_adam_direction_foreach(state_lists, ...)` — already prototyped in `bench_foreach_adam.py`
3. **Compute shape groups** in `AdamPolarProductLoRA.__init__`
4. **Change default `precond_refresh_every=1 → 5`** for `AdamPolarProductLoRA` (or set on the registered variants in `OPTIMIZER_CHOICES`). Validated by `polar_K_sweep_r64_2k` data
5. **Implement `_step_batched`** dispatched from `step()` when no diagnostic / exotic flag is set. Leave existing per-pair code as `_step_per_pair` for the diagnostic path
6. **Behavioral equivalence test** — must pass before merge
7. **Re-run `bench_polar_product_components.py`** on A6000 to confirm projected wall times. Update `profiling_a6000_2026_05_04.md` with the post-integration breakdown
8. **(Deferred)** A100 measurement once QOS frees up; re-validate the projection on canonical hardware

## Out of scope

- CUDA graph capture (additional ~1% wall-time win, real implementation cost — discussed and dropped)
- bf16 / fp8 precond_refresh (numerical risk, marginal at low r)
- Warm-start NS (no wall-time win without dropping j; j=5 is settled)
- K eval-loss tolerance sweep (no longer needed; precond_refresh batched at <2 ms / step at any K)
- Full per-pair → group view aliasing (memory optimization; keep separate stacks for v1)

## Estimated lines of code

- `_batched_polar.py` additions (eigh helper if needed): ~30 lines
- `optim.py` foreach helpers + `_step_batched` + `__init__` group computation: ~250 lines
- `tests/test_polar_product_batched_equivalence.py`: ~80 lines
- Total: ~360 lines
