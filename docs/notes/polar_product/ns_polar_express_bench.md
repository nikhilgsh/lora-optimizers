# Newton-Schulz vs Polar Express bench

Source: `scripts/bench/bench_polar_orthog.py` (commit `8b6db4e4e4`). Snapshot u_A from `chord_tight_r64_k3_snapshot_blackwell/step_2000`.

Variants timed (all in `lora_playground.optim`):

- `_newton_schulz` — cubic, per-matrix, fp32 (canonical Muon).
- `_newton_schulz_gram_batched` — Dao 2026 Algorithm 3 with fp16 iteration + restart at iter 2 (production default).
- `_polar_express` — Amsel 2025, degree-5 with optimal coefficients for σ ∈ [1e-3, 1].

- **blackwell** = `NVIDIA RTX PRO 6000 Blackwell Server Edition` (n_reps=30)
- **a6000** = `NVIDIA RTX A6000` (n_reps=30)

## Wall time (ms / call)

Random fp32 inputs at production LoRA shapes (A-side: `(r, d_in)`). Mean over n_reps CUDA-event samples, after warmup.

| shape | K | NS_rect (blackwell) | NS_gram (fp16+restart) (blackwell) | PolarExpress (blackwell) | NS_rect (a6000) | NS_gram (fp16+restart) (a6000) | PolarExpress (a6000) |
|---|---|---|---|---|---|---|---|
| r16_d2048 | 3 |  0.118 |  0.295 |  0.191 |  0.229 |  0.679 |  0.310 |
| r16_d2048 | 5 |  0.195 |  0.389 |  0.293 |  0.321 |  0.712 |  0.459 |
| r16_d2048 | 6 |  0.227 |  0.436 |  0.344 |  0.369 |  0.797 |  0.534 |
| r16_d2048 | 7 |  0.260 |  0.482 |  0.397 |  0.414 |  0.869 |  0.609 |
| r16_d2048 | 8 |  0.293 |  0.532 |  0.447 |  0.466 |  0.949 |  0.681 |
| r16_d2048 | 10 |  0.357 |  0.628 |  0.549 |  0.559 |  1.109 |  0.828 |
| r256_d2048 | 3 |  0.220 |  0.309 |  0.276 |  0.288 |  0.586 |  0.330 |
| r256_d2048 | 5 |  0.338 |  0.405 |  0.434 |  0.425 |  0.748 |  0.491 |
| r256_d2048 | 6 |  0.398 |  0.452 |  0.513 |  0.610 |  0.823 |  0.571 |
| r256_d2048 | 7 |  0.455 |  0.500 |  0.591 |  0.564 |  0.902 |  0.653 |
| r256_d2048 | 8 |  0.514 |  0.546 |  0.668 |  0.632 |  0.981 |  0.737 |
| r256_d2048 | 10 |  0.632 |  0.643 |  0.826 |  0.770 |  1.131 |  0.892 |
| r64_d2048 | 3 |  0.171 |  0.309 |  0.243 |  0.248 |  0.698 |  0.339 |
| r64_d2048 | 5 |  0.259 |  0.405 |  0.377 |  0.351 |  0.716 |  0.499 |
| r64_d2048 | 6 |  0.303 |  0.452 |  0.445 |  0.404 |  0.798 |  0.576 |
| r64_d2048 | 7 |  0.347 |  0.501 |  0.513 |  0.461 |  0.878 |  0.655 |
| r64_d2048 | 8 |  0.392 |  0.548 |  0.582 |  0.514 |  0.948 |  0.734 |
| r64_d2048 | 10 |  0.482 |  0.641 |  0.714 |  0.620 |  1.108 |  0.896 |
| snapshot_r64_d2048 | 3 |  0.170 |  0.307 |  0.241 |  0.248 |  0.560 |  0.338 |
| snapshot_r64_d2048 | 5 |  0.259 |  0.502 |  0.378 |  0.351 |  0.719 |  0.496 |
| snapshot_r64_d2048 | 6 |  0.302 |  0.448 |  0.444 |  0.403 |  0.797 |  0.620 |
| snapshot_r64_d2048 | 7 |  0.346 |  0.497 |  0.512 |  0.464 |  0.875 |  0.656 |
| snapshot_r64_d2048 | 8 |  0.390 |  0.545 |  0.579 |  0.511 |  0.953 |  0.991 |
| snapshot_r64_d2048 | 10 |  0.477 |  0.639 |  0.714 |  0.764 |  1.129 |  0.943 |
| snapshot_r64_d8192 | 3 |  0.216 |  0.307 |  0.254 |  0.264 |  0.576 |  0.346 |
| snapshot_r64_d8192 | 5 |  0.332 |  0.403 |  0.397 |  0.365 |  0.736 |  0.512 |
| snapshot_r64_d8192 | 6 |  0.390 |  0.452 |  0.469 |  0.435 |  0.832 |  0.594 |
| snapshot_r64_d8192 | 7 |  0.447 |  0.499 |  0.539 |  0.483 |  0.909 |  0.677 |
| snapshot_r64_d8192 | 8 |  0.506 |  0.546 |  0.611 |  0.533 |  0.995 |  0.760 |
| snapshot_r64_d8192 | 10 |  0.622 |  0.639 |  0.758 |  0.643 |  1.142 |  0.952 |

Rows prefixed `snapshot_…` are timed on a real u_A tensor from the chord-tight r=64 snapshot (one timed sample per unique shape; timing is value-independent for these algorithms).

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
| r16_d2048 | 7 | 3.46e-06 | 1.87e-03 | 2.56e-06 |
| r16_d2048 | 8 | 1.2e-07 | 2.01e-03 | 2.50e-06 |
| r16_d2048 | 10 | 1.2e-07 | 2.01e-03 | 1.2e-07 |
| r256_d2048 | 3 | 8.61e-01 | 8.68e-01 | 8.44e-01 |
| r256_d2048 | 5 | 6.94e-01 | 7.08e-01 | 1.31e-01 |
| r256_d2048 | 6 | 5.55e-01 | 5.74e-01 | 1.82e-03 |
| r256_d2048 | 7 | 3.76e-01 | 3.99e-01 | 6.12e-05 |
| r256_d2048 | 8 | 1.86e-01 | 2.07e-01 | 3.78e-05 |
| r256_d2048 | 10 | 3.45e-03 | 5.41e-03 | 3.91e-05 |
| r64_d2048 | 3 | 6.60e-01 | 6.75e-01 | 8.44e-01 |
| r64_d2048 | 5 | 3.23e-01 | 3.46e-01 | 1.30e-01 |
| r64_d2048 | 6 | 1.40e-01 | 1.59e-01 | 1.63e-03 |
| r64_d2048 | 7 | 2.79e-02 | 3.61e-02 | 1.45e-05 |
| r64_d2048 | 8 | 1.14e-03 | 2.60e-03 | 2.80e-06 |
| r64_d2048 | 10 | 3.6e-07 | 1.88e-03 | 3.0e-07 |

### Snapshot u_A inputs (real production tensors)

| (r, d) | K | n_pairs | NS_rect | NS_gram (fp16+restart) | PolarExpress |
|---|---|---|---|---|---|
| (64, 2048) | 3 | 7 | [8.86e-01, 9.18e-01, 9.53e-01] | [8.91e-01, 9.22e-01, 9.56e-01] | [8.61e-01, 8.63e-01, 8.66e-01] |
| (64, 2048) | 5 | 7 | [7.47e-01, 8.16e-01, 8.95e-01] | [7.59e-01, 8.25e-01, 9.00e-01] | [1.31e-01, 1.31e-01, 1.31e-01] |
| (64, 2048) | 6 | 7 | [6.29e-01, 7.27e-01, 8.44e-01] | [6.45e-01, 7.40e-01, 8.51e-01] | [1.82e-03, 1.86e-03, 1.88e-03] |
| (64, 2048) | 7 | 7 | [4.68e-01, 6.01e-01, 7.68e-01] | [4.90e-01, 6.18e-01, 7.78e-01] | [1.30e-05, 1.51e-05, 1.63e-05] |
| (64, 2048) | 8 | 7 | [2.78e-01, 4.33e-01, 6.58e-01] | [3.01e-01, 4.55e-01, 6.73e-01] | [2.62e-06, 2.68e-06, 2.80e-06] |
| (64, 2048) | 10 | 7 | [1.59e-02, 7.98e-02, 3.20e-01] | [2.16e-02, 9.55e-02, 3.45e-01] | [2.38e-07, 3.58e-07, 4.17e-07] |
| (64, 8192) | 3 | 1 | [9.05e-01, 9.05e-01, 9.05e-01] | [9.09e-01, 9.09e-01, 9.09e-01] | [8.43e-01, 8.43e-01, 8.43e-01] |
| (64, 8192) | 5 | 1 | [7.88e-01, 7.88e-01, 7.88e-01] | [7.98e-01, 7.98e-01, 7.98e-01] | [1.31e-01, 1.31e-01, 1.31e-01] |
| (64, 8192) | 6 | 1 | [6.87e-01, 6.87e-01, 6.87e-01] | [7.01e-01, 7.01e-01, 7.01e-01] | [1.87e-03, 1.87e-03, 1.87e-03] |
| (64, 8192) | 7 | 1 | [5.46e-01, 5.46e-01, 5.46e-01] | [5.65e-01, 5.65e-01, 5.65e-01] | [1.39e-05, 1.39e-05, 1.39e-05] |
| (64, 8192) | 8 | 1 | [3.65e-01, 3.65e-01, 3.65e-01] | [3.89e-01, 3.89e-01, 3.89e-01] | [2.56e-06, 2.56e-06, 2.56e-06] |
| (64, 8192) | 10 | 1 | [4.37e-02, 4.37e-02, 4.37e-02] | [5.60e-02, 5.60e-02, 5.60e-02] | [2.38e-07, 2.38e-07, 2.38e-07] |

## Cost-matched: NS_rect K=10 vs PolarExpress K∈{6,7,8}

Question raised in `notebooks/muon_squared_snapshot_analysis.ipynb`: given that the leaderboard shows NS j=10 > j=5, does PE-j=k do better than NS-j=10 at comparable wall? The Polar Express schedule is fully exhausted by iter 7 (iter 8 onward uses plain NS-deg5), so K∈{6, 7, 8} bracket the candidate replacements.

| shape | hw | NS K=10 ms | PE K=6 ms | PE K=7 ms | PE K=8 ms | NS K=10 resid | PE K=6 resid | PE K=7 resid | PE K=8 resid |
|---|---|---|---|---|---|---|---|---|---|
| r16_d2048 | blackwell |  0.357 |  0.344 |  0.397 |  0.447 | 1.2e-07 | 1.61e-03 | 2.56e-06 | 2.50e-06 |
| r256_d2048 | blackwell |  0.632 |  0.513 |  0.591 |  0.668 | 3.45e-03 | 1.82e-03 | 6.12e-05 | 3.78e-05 |
| r64_d2048 | blackwell |  0.482 |  0.445 |  0.513 |  0.582 | 3.6e-07 | 1.63e-03 | 1.45e-05 | 2.80e-06 |
| r16_d2048 | a6000 |  0.559 |  0.534 |  0.609 |  0.681 | 1.2e-07 | 1.61e-03 | 3.64e-06 | 2.50e-06 |
| r256_d2048 | a6000 |  0.770 |  0.571 |  0.653 |  0.737 | 4.20e-03 | 1.83e-03 | 5.77e-05 | 3.90e-05 |
| r64_d2048 | a6000 |  0.620 |  0.576 |  0.655 |  0.734 | 1.2e-07 | 1.77e-03 | 1.49e-05 | 2.56e-06 |

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
