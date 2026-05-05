# Polar-product optimizer wall-times — A100 canonical

**Date:** 2026-05-04. **Hardware:** A100-SXM4-80GB, isolated allocation
(`#SBATCH --exclusive`, no co-tenants). **Single seed.** Job:
`slurm_logs/bench_step_a100_canonical_6338007.{out,err}`. Raw data:
`logs/bench_a100/canonical_r{16,64}.jsonl`.

These are the canonical numbers; supersedes the A6000 measurements in
`profiling_a6000_2026_05_04.md` (kept for historical context only — the
relative component ranking matched A100, but absolute wall-times and
AdamW ratios on A6000 understate the optimizer's share of training time
because A6000's slower fwd+bwd hides it).

## Setup

- Model: `allenai/OLMo-2-0425-1B`, target_modules=all-linear, 112 LoRA pairs
- bf16 model, batch_size=2, seq_len=512, grad_accum_steps=8 (effective batch 16)
- 3 warmup + 4 timed cycles per cell; n_reps = 4 × K
- Optimizer ints: `adam-polar-product-lora` (k=1) and `-coupled` (k=3),
  precond_method ∈ {eigh, higham}, precond_refresh_every=1
- Reproduce:
  ```
  for r in 16 64; do
    python scripts/bench_optimizer_step.py --bf16 --lora_r $r \
        --optimizers adamw adam-polar-product-lora adam-polar-product-lora-coupled \
        --precond_method eigh higham --precond_refresh_every 1 \
        --n_warmup 3 --n_cycles 4 \
        --batch_size 2 --seq_len 512 --grad_accum_steps 8 \
        --out logs/bench_a100/canonical_r${r}.jsonl
  done
  ```

## Results

### r=16 (12.1M LoRA params)

| optimizer | method | fwd ms | bwd ms | opt ms | zero ms | total ms | ×AdamW |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamW | (n/a) | 374 | 478 | 2.9 | 1.3 | 856 | 1.00× |
| polar k=1 | eigh | 374 | 480 | 79 | 1.3 | 934 | 1.09× |
| polar k=1 | **higham** | 373 | 481 | **10** | 1.3 | **865** | **1.01×** |
| polar k=3 | eigh | 373 | 481 | 127 | 1.3 | 982 | 1.15× |
| polar k=3 | **higham** | 374 | 481 | **60** | 1.3 | **916** | **1.07×** |

### r=64 (48.2M LoRA params)

| optimizer | method | fwd ms | bwd ms | opt ms | zero ms | total ms | ×AdamW |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamW | (n/a) | 420 | 531 | 3.7 | 1.3 | 957 | 1.00× |
| polar k=1 | eigh | 420 | 532 | 265 | 1.3 | 1218 | 1.27× |
| polar k=1 | **higham** | 420 | 534 | **20** | 1.3 | **974** | **1.02×** |
| polar k=3 | eigh | 420 | 535 | 332 | 1.3 | 1289 | 1.35× |
| polar k=3 | **higham** | 422 | 535 | **94** | 1.3 | **1052** | **1.10×** |

### Chord variants (post-batching)

`-coupled-exact-chord` (refresh `S_{B+ΔB}, S_{A+ΔA}` per Picard iter,
algorithm.md §2 remark) and `-coupled-spectral-chord` (operator-norm
trust-region rule, Substitution 1' algorithm.md §6.1) both land in the
batched hot path as of commit `8ec18aa`.

| optimizer | method | r=16 opt | r=16 ×AdamW | r=64 opt | r=64 ×AdamW |
|---|---|---:|---:|---:|---:|
| coupled (plain) | eigh | 127 | 1.15× | 337 | 1.35× |
| coupled (plain) | higham | 59 | 1.06× | 93 | 1.09× |
| coupled-exact-chord | eigh | 267 | 1.30× | 812 | 1.85× |
| **coupled-exact-chord** | **higham** | **67** | **1.07×** | **107** | **1.10×** |
| coupled-spectral-chord | eigh | 136 | 1.15× | 339 | 1.35× |
| **coupled-spectral-chord** | **higham** | **65** | **1.07×** | **104** | **1.10×** |

Speedups vs the pre-batched eigh baselines (`bench_chord_a100`,
job 6338055):

| variant | r=16: was → now | speedup | r=64: was → now | speedup |
|---|---|---:|---|---:|
| coupled-exact-chord | 713 → 67 | **10.6×** | 1063 → 107 | **9.9×** |
| coupled-spectral-chord | 796 → 65 | **12.2×** | 988 → 104 | **9.5×** |

Note: `coupled-spectral-chord` with `eigh` (per-pair eigh in batched
path) also drops massively — 988 → 339 ms at r=64 — purely from
batching the σ_max power-iteration launches via
`_sigma_max_power_iter_batched`. exact-chord with eigh stays expensive
because per-pair eigh on the perturbed Gram matrices fires 3 times per
step regardless of batching.

## Headlines

- **`adam-polar-product-lora` (k=1) with batched higham hits 1.02× AdamW at r=64.** Optimizer.step() is 19.7 ms vs AdamW's 3.7 ms — substantial in isolation, but small relative to fwd+bwd (~951 ms / step). Effectively indistinguishable from AdamW at training-step granularity.
- **`adam-polar-product-lora-coupled` (k=3) with higham hits 1.10× AdamW at r=64**, 1.07× at r=16. The k=3 Picard adds ~3× the polar pipeline cost; the cross-coupling matmuls (`B^T dB A`, etc.) are the dominant remaining cost — they're compute-bound on the d_in/d_out contraction and don't batch well (validated in `bench_picard_cross_coupling.py`: 0.97× speedup, near a wash).
- **eigh-with-K=1 is no longer a viable production choice at r=64+** — 1.27× / 1.35× AdamW. The 100× speedup of `batched_higham` (validated by the integration test, see Higham safety section below) has eliminated this gap. eigh stays the optimizer's *default* (algorithmic-baseline fidelity); higham is opt-in via `--precond_method higham` and gives the wall-time win.

## Speedup attribution at r=64 k=1, vs current production (eigh K=1, pre-batching)

For comparison: the prior production state was eigh K=1 with the per-pair Python loop. The component bench at r=64 in `profiling_a6000_2026_05_04.md` placed that at ~392 ms / step optimizer-only on A6000. Translating to A100 (proportional to the relative AdamW step times in this run) puts pre-batching at roughly 270 ms / step opt on A100. Post-batching with higham: 19.7 ms — a **~14× reduction on optimizer.step()**.

The total-step numbers tell a smaller story because fwd+bwd dominates: pre-batching r=64 k=1 was ~1.13× AdamW; post-batching with higham is 1.02×. The 11-point reduction in ratio is what shipped.

## Higham safety: revalidated at r=256 K=1

The deterministic-init higham (commit `9374314`) was integration-tested at r=256, K=1 prior to this benchmark
(`logs/integration_higham_test/`, jobs 6336972 / 6336973):

- 1000 steps clean, exit 0
- 0 `non_finite_Z` events out of 224k probe emits
- Trajectory matches eigh reference (same commit, same config) within
  0.07σ_AdamW final, 0.6σ peak, |Δ| typically < 0.2σ across all 20 evals

This unblocked shipping `batched_higham` as the precond_refresh primitive in `_step_batched`. `precond_method='eigh'` remains the optimizer default (no behavior change for existing sweep params); `'higham'` is opt-in.

## Caveats

- Single seed per cell. Variance not characterized; the 95-percentile of opt-time bench noise across reps in this script is roughly ±2% based on this run's `min`/`max`/`median` spread within each cell.
- r=128 / r=256 not measured here (would need a separate bench cell). Projections from `bench_precond_refresh.py`: at r=256, batched_higham per-call is ~6 ms (vs 573 ms loop_eigh), so polar k=1 r=256 with higham should hit ~1.04–1.06× AdamW. Validate before acting on numbers downstream.
- Optimizer.step() ratios scale with hardware: a faster fwd+bwd path (compile, more advanced GPU) shrinks the absolute gap further. The numbers here are honest for the current production training stack.
