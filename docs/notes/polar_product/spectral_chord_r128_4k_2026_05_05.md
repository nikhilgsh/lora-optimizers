# Spectral-chord magnitude rule at r=128: extended-horizon comparison

**Date:** 2026-05-05. Single-seed at extended 4k-step horizon (the
canonical comparison horizon is 2k; the r=128 extended horizon was
adopted because the linearized variant exhibited a step-~2200 σ_max(B)
redistribution event in the diagnostic logs of an earlier run, and the
question motivating this experiment was whether the σ-aware magnitude
rule prevents it).

Base model OLMo-2-1B, code-instruction adaptation
(Magicoder-OSS-Instruct-75K, 70k samples, sequence length 512), r=128,
α=128, target = `all-linear`. Sweep group:
`logs/spectral_chord_r128_4k/`.

## 1. The variant

`adam-polar-product-lora-coupled-spectral-chord` is the Picard-coupled
polar–product variant of [algorithm.md](./algorithm.md), with the
per-block magnitude rescale (Substitution 1 in the doc) replaced by the
σ-aware **chord-spectral** rescale (Substitution 1′ — derived from
Spectron's [arXiv:2602.12429] chord bound):

$$\rho \;=\; \frac{\eta}{\sigma_{\max}(A) + \sigma_{\max}(B) + 1},
\qquad \Delta A \leftarrow -\rho \cdot \mathrm{geo}_A,\ 
\Delta B \leftarrow -\rho \cdot \mathrm{geo}_B,$$

after each Picard inner step ($\mathrm{geo}_A, \mathrm{geo}_B$ are the
unit-direction polar factors of the whitened Adam updates). The bound
guarantees $\|\Delta W\|_{\mathrm{op}} \le \eta$ along the chord defined
by $(B + \Delta B)(A + \Delta A) - BA$, including the $\Delta B\,
\Delta A$ cross term.

Implementation: `lora_playground/optim.py`,
`AdamPolarProductLoRA(magnitude_rule="spectral_chord")`.
σ-quantities computed by power iteration with warm-starts persisted in
`pair_state['u_A_top'], pair_state['u_B_top']`; cold n_iters=8 inside
the Picard loop, warm n_iters=3 across steps. See
`tests/test_sigma_max_power_iter.py` for accuracy and warm-start
validation.

## 2. Result

All cells single-seed, 4000 steps, r=128, eval_every=200.

| optimizer | lr | step 200 | 1000 | 2000 | 2200 | 3000 | **4000** |
|---|---|---|---|---|---|---|---|
| spectral_chord | 3e-3 | 0.8533 | 0.7888 | 0.7624 | 0.7590 | 0.7469 | 0.7366 |
| **spectral_chord** | **1e-2** | 0.8222 | 0.7663 | 0.7403 | 0.7375 | 0.7249 | **0.7151** |
| spectral_chord | 3e-2 | 0.8295 | 0.7731 | 0.7437 | 0.7412 | 0.7262 | 0.7154 |
| adam-polar-product-lora | 1e-4 | 0.8290 | 0.7717 | 0.7458 | 0.7426 | 0.7303 | 0.7197 |
| adam-polar-product-lora | 3e-4 | 0.8169 | 0.7681 | 0.7446 | 0.7419 | 0.7293 | 0.7200 |
| adam-polar-product-lora-coupled | 3e-4 | 0.8168 | 0.7610 | 0.7313 | 0.7453 | 0.7265 | 0.7197 |
| adam-polar-product-lora-coupled | 1e-4 | 0.8293 | 0.7726 | 0.7468 | 0.7438 | 0.7315 | 0.7203 |
| adamw | 1e-4 | 0.8280 | 0.7740 | 0.7489 | 0.7459 | 0.7337 | 0.7240 |

Best non-spectral-chord polar-product baseline at this rank/horizon:
0.7197 (gauge or coupled polar-product at lr ∈ {1e-4, 3e-4}). Best
AdamW at this rank/horizon: **0.7240 at lr=1e-4** (other AdamW cells:
3e-5 → 0.7509, 3e-4 → 0.7449, 1e-3 → 1.0239 diverged). Best
spectral_chord cell (lr=1e-2): **0.7151**.

| comparator | best loss | Δ (spectral_chord − comparator) |
|---|---|---|
| AdamW | 0.7240 | **−0.0089** |
| polar-product (gauge / coupled) | 0.7197 | −0.0046 |

The project's noise floor at the canonical 2k horizon is multi-seed
AdamW: σ ≈ 0.0006 at r=16, σ ≈ 0.0007 at r=64. r=128 4k σ has not been
measured, but assuming the same order, **Δ vs AdamW is ≈13σ and Δ vs
the polar-product baseline is ≈6σ**. Multi-seed verification of the
variant itself is deferred; this is a single-seed result.

## 3. Mid-trajectory: the step-2200 region

The original motivation was whether the spectral-chord rule prevents
the σ_max(B) runaway observed in the linearized r=128 lr=3e-4 4k run at
step ~2200. Direct comparison via diagnostic σ-trajectories is **not
possible from this sweep**: optimizer diagnostics were not enabled in
the spectral_chord runs, so per-step σ values are not logged. What is
visible is the eval-loss between steps 2000 and 2200:

| optimizer (lr) | step 2000 → 2200 | Δ |
|---|---|---|
| spectral_chord (1e-2) | 0.7403 → 0.7375 | −0.0028 |
| spectral_chord (3e-2) | 0.7437 → 0.7412 | −0.0025 |
| adam-polar-product-lora-coupled (3e-4) | 0.7313 → 0.7453 | **+0.0140** |
| adam-polar-product-lora (3e-4) | 0.7446 → 0.7419 | −0.0027 |

The coupled gauge variant at lr=3e-4 shows a non-monotone bump at the
runaway step (loss rises by 0.014 between evals, then resumes
descending); the gauge-only variant and both spectral-chord cells are
monotone through this interval. This is consistent with — but not
proof of — the spectral-chord rule preventing the redistribution event.
A follow-up sweep with diagnostics enabled would directly confirm.

## 4. lr robustness

Both lr=1e-2 (0.7151) and lr=3e-2 (0.7154) finish within 0.0003 of each
other — the spectral-chord rule is markedly less lr-sensitive than the
linearized variants, where the optimum sits in a narrow basin around
{1e-4, 3e-4} and the next decade up (1e-3) diverges to ≥0.876. The
chord rule bounds $\|\Delta W\|_{\mathrm{op}} \le \eta$ exactly, so
the effective per-step update magnitude is the user-specified η
regardless of factor scale; this rescaling effect is the natural
explanation for the wider basin.

## 5. Reproducibility

- Sweep config: `params/spectral_chord_r128_4k.json` (lrs 3e-3, 1e-2,
  3e-2; one cell each).
- Submit: `SWEEP_SCOPE=ext_compare,polar_family ./slurm_scripts/submit.sh
  params/spectral_chord_r128_4k.json spectral_chord_r128_4k 3
  scripts/sweep_4k_diag.sh slurm_scripts/sbatch_12h.sh`.
- SLURM job 6338382, 5h35m wall on 3× A100.
- Loader: `load_runs(where={"optimizer":
  "adam-polar-product-lora-coupled-spectral-chord", "lora_r": 128,
  "max_steps": 4000})`.

## 6. Status and next steps

Single-seed, single-rank result: spectral-chord beats the best known
r=128 4k baseline by Δ = −0.0046 (≈ 6× the AdamW r=64 σ). Open items
before this becomes a load-bearing recommendation:

- Re-run with `--log_optim_diagnostics` to verify σ_max(B) trajectory
  is bounded and the step-2200 redistribution is absent.
- Multi-seed spectral_chord at lr=1e-2 r=128 to put a σ on the variant
  itself.
- r-sweep at the **4k horizon** across r ∈ {16, 32, 64, 128, 256} to
  confirm spectral_chord dominates AdamW (and the existing
  polar-product baselines) at every rank — the candidate-headline
  comparison. Existing 2k cells at r=128 in
  `logs/spectral_chord_r128_2k/` and the lo/v2 variants are partial
  precursors and should be analyzed or superseded.
