# Newton-Schulz vs Polar Express bench

Source: `scripts/bench/bench_polar_orthog.py` (commit `8b39faddc1`). Snapshot u_A from `chord_tight_r64_k3_snapshot_blackwell/step_2000`.

Variants timed (all in `lora_playground.optim`):

- `_newton_schulz` — cubic, per-matrix, fp32 (canonical Muon).
- `_newton_schulz_gram_batched` — Dao 2026 Algorithm 3 with fp16 iteration + restart at iter 2 (production default).
- `_polar_express` — Amsel 2025, degree-5 with optimal coefficients for σ ∈ [1e-3, 1].

- **a6000** = `NVIDIA RTX A6000` (n_reps=30)

## Wall time (ms / call)

Random fp32 inputs at production LoRA shapes (A-side: `(r, d_in)`). Mean over n_reps CUDA-event samples, after warmup.

| shape | K | NS_rect (a6000) | NS_gram (fp16+restart) (a6000) | PolarExpress (a6000) |
|---|---|---|---|---|
| r16_d2048 | 3 |  0.213 |  0.865 |  0.301 |
| r16_d2048 | 5 |  0.312 |  0.679 |  0.437 |
| r16_d2048 | 6 |  0.354 |  0.755 |  0.504 |
| r16_d2048 | 7 |  0.398 |  0.825 |  0.709 |
| r16_d2048 | 8 |  0.444 |  0.928 |  0.649 |
| r16_d2048 | 10 |  0.536 |  1.044 |  0.791 |
| r256_d2048 | 3 |  0.284 |  0.562 |  0.326 |
| r256_d2048 | 5 |  0.422 |  0.715 |  0.481 |
| r256_d2048 | 6 |  0.492 |  0.787 |  0.564 |
| r256_d2048 | 7 |  0.560 |  0.860 |  0.641 |
| r256_d2048 | 8 |  0.629 |  0.937 |  0.724 |
| r256_d2048 | 10 |  0.764 |  1.085 |  0.927 |
| r64_d2048 | 3 |  0.239 |  0.546 |  0.327 |
| r64_d2048 | 5 |  0.345 |  0.703 |  0.477 |
| r64_d2048 | 6 |  0.388 |  0.767 |  0.553 |
| r64_d2048 | 7 |  0.442 |  0.847 |  0.619 |
| r64_d2048 | 8 |  0.490 |  0.927 |  0.700 |
| r64_d2048 | 10 |  0.590 |  1.060 |  0.971 |

## Accuracy: max|σ_i − 1| after K iterations

Lower = closer to true polar. Two input sources:

- **random**: fp32 Gaussian at the shape (single sample).
- **snapshot u_A**: real bias-corrected Adam direction from the chord-tight r=64 production snapshot, aggregated across the loaded pairs as `[min, median, max]`.

### Random fp32 inputs

| shape | K | NS_rect | NS_gram (fp16+restart) | PolarExpress |
|---|---|---|---|---|
| r16_d2048 | 3 | 3.22e-01 | 3.46e-01 | 3.67e-01 |
| r16_d2048 | 5 | 2.75e-02 | 3.57e-02 | 1.28e-01 |
| r16_d2048 | 6 | 1.12e-03 | 2.42e-03 | 1.61e-03 |
| r16_d2048 | 7 | 3.93e-06 | 1.87e-03 | 3.64e-06 |
| r16_d2048 | 8 | 1.2e-07 | 2.01e-03 | 2.50e-06 |
| r16_d2048 | 10 | 1.2e-07 | 2.01e-03 | 1.2e-07 |
| r256_d2048 | 3 | 8.64e-01 | 8.70e-01 | 8.44e-01 |
| r256_d2048 | 5 | 7.00e-01 | 7.14e-01 | 1.31e-01 |
| r256_d2048 | 6 | 5.63e-01 | 5.82e-01 | 1.83e-03 |
| r256_d2048 | 7 | 3.87e-01 | 4.10e-01 | 5.77e-05 |
| r256_d2048 | 8 | 1.95e-01 | 2.17e-01 | 3.90e-05 |
| r256_d2048 | 10 | 4.20e-03 | 6.30e-03 | 2.4e-07 |
| r64_d2048 | 3 | 6.59e-01 | 6.75e-01 | 8.44e-01 |
| r64_d2048 | 5 | 3.23e-01 | 3.46e-01 | 1.31e-01 |
| r64_d2048 | 6 | 1.39e-01 | 1.59e-01 | 1.77e-03 |
| r64_d2048 | 7 | 2.78e-02 | 3.60e-02 | 1.49e-05 |
| r64_d2048 | 8 | 1.14e-03 | 2.46e-03 | 2.56e-06 |
| r64_d2048 | 10 | 1.2e-07 | 1.66e-03 | 1.2e-07 |

### Snapshot u_A inputs (real production tensors)

| (r, d) | K | n_pairs | NS_rect | NS_gram (fp16+restart) | PolarExpress |
|---|---|---|---|---|---|
| (64, 2048) | 3 | 7 | [8.86e-01, 9.18e-01, 9.53e-01] | [8.91e-01, 9.22e-01, 9.56e-01] | [8.61e-01, 8.63e-01, 8.66e-01] |
| (64, 2048) | 5 | 7 | [7.47e-01, 8.16e-01, 8.95e-01] | [7.59e-01, 8.25e-01, 9.00e-01] | [1.31e-01, 1.31e-01, 1.31e-01] |
| (64, 2048) | 6 | 7 | [6.29e-01, 7.27e-01, 8.44e-01] | [6.45e-01, 7.40e-01, 8.51e-01] | [1.82e-03, 1.87e-03, 1.88e-03] |
| (64, 2048) | 7 | 7 | [4.68e-01, 6.01e-01, 7.68e-01] | [4.90e-01, 6.18e-01, 7.78e-01] | [1.29e-05, 1.47e-05, 1.55e-05] |
| (64, 2048) | 8 | 7 | [2.78e-01, 4.33e-01, 6.58e-01] | [3.01e-01, 4.55e-01, 6.73e-01] | [2.50e-06, 2.56e-06, 2.56e-06] |
| (64, 2048) | 10 | 7 | [1.59e-02, 7.98e-02, 3.20e-01] | [2.16e-02, 9.56e-02, 3.45e-01] | [1.19e-07, 1.79e-07, 2.38e-07] |
| (64, 8192) | 3 | 1 | [9.05e-01, 9.05e-01, 9.05e-01] | [9.09e-01, 9.09e-01, 9.09e-01] | [8.43e-01, 8.43e-01, 8.43e-01] |
| (64, 8192) | 5 | 1 | [7.88e-01, 7.88e-01, 7.88e-01] | [7.98e-01, 7.98e-01, 7.98e-01] | [1.31e-01, 1.31e-01, 1.31e-01] |
| (64, 8192) | 6 | 1 | [6.87e-01, 6.87e-01, 6.87e-01] | [7.01e-01, 7.01e-01, 7.01e-01] | [1.87e-03, 1.87e-03, 1.87e-03] |
| (64, 8192) | 7 | 1 | [5.46e-01, 5.46e-01, 5.46e-01] | [5.65e-01, 5.65e-01, 5.65e-01] | [1.50e-05, 1.50e-05, 1.50e-05] |
| (64, 8192) | 8 | 1 | [3.65e-01, 3.65e-01, 3.65e-01] | [3.89e-01, 3.89e-01, 3.89e-01] | [2.56e-06, 2.56e-06, 2.56e-06] |
| (64, 8192) | 10 | 1 | [4.37e-02, 4.37e-02, 4.37e-02] | [5.56e-02, 5.56e-02, 5.56e-02] | [2.38e-07, 2.38e-07, 2.38e-07] |

## Cost-matched: NS_rect K=10 vs PolarExpress K∈{6,7,8}

Question raised in `notebooks/muon_squared_snapshot_analysis.ipynb`: given that the leaderboard shows NS j=10 > j=5, does PE-j=k do better than NS-j=10 at comparable wall? The Polar Express schedule is fully exhausted by iter 7 (iter 8 onward uses plain NS-deg5), so K∈{6, 7, 8} bracket the candidate replacements.

| shape | hw | NS K=10 ms | PE K=6 ms | PE K=7 ms | PE K=8 ms | NS K=10 resid | PE K=6 resid | PE K=7 resid | PE K=8 resid |
|---|---|---|---|---|---|---|---|---|---|
| r16_d2048 | a6000 |  0.536 |  0.504 |  0.709 |  0.649 | 1.2e-07 | 1.61e-03 | 3.64e-06 | 2.50e-06 |
| r256_d2048 | a6000 |  0.764 |  0.564 |  0.641 |  0.724 | 4.20e-03 | 1.83e-03 | 5.77e-05 | 3.90e-05 |
| r64_d2048 | a6000 |  0.590 |  0.553 |  0.619 |  0.700 | 1.2e-07 | 1.77e-03 | 1.49e-05 | 2.56e-06 |

## Notes on interpretation

- `NS_gram_fp16` residual plateaus at ~1–6e-3 (half-precision noise floor on the Gram iterates) — that's the cost of the production fp16+restart path. Whether this matters downstream is the same empirical question the K=5 vs K=10 leaderboard comparison answered: small but not zero.
- PolarExpress is designed to reach the σ_min=1e-3 region in ~7 iterations. The synthetic-random shapes here have wider cond(X) than typical u_A inputs, so the random-input residual column is a worst-case bound; the snapshot u_A column is what the production optimizer actually sees.
- NS_rect K=10 reaches fp32 noise floor only at small r; at r=256 it still has ~1e-3 residual on random inputs. PE K=10 reaches fp32 noise floor across all r.

## Reproducibility

```
# Blackwell (canonical):
#   submit slurm_pending/bench_polar_orthog_blackwell.sbatch
# A6000 (local):
python scripts/bench/bench_polar_orthog.py \
    --n_warmup 5 --n_reps 30 --n_pairs 8 --Ks 3 5 6 7 8 10 \
    --hardware a6000 --out logs/bench/polar_orthog_a6000.jsonl
# Then re-render:
python scripts/analysis/render_polar_orthog_table.py
```
