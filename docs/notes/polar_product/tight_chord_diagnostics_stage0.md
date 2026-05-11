# Stage 0 — chord-tight diagnostics readout (CORRECTED)

**Status:** Pipeline `packed_v1`, optimizer `adam-polar-product-lora-coupled-spectral-chord-tight` at `lr=3e-3`, `lora_alpha=lora_r`, `seed=0`, single-seed. Probe-correctness re-run in `logs/chord_tight_diag_500_r16r64_v3` (job 6382591); 30-step local A6000 smoke results quoted in this writeup until SLURM v3 completes (~25 min).

## ⚠️ Correction (2026-05-11)

An earlier version of this doc reported that the existing chord-tight optimizer was breaching its `‖ΔW‖₂ ≤ η` safety guarantee by up to 2.4× at r=64. **That finding was wrong**. The optimizer was holding the bound throughout; the breach was an artifact in two of the Stage-0 probes (chord_slack and lambda_dir_gain), which used a rank-≤2r `eigvals` shortcut on a non-symmetric Gram product. On real LoRA chord matrices this over-estimated σ_max² by 30–60% (numerical artifact of `.real`-part-of-complex-eigenvalues on polar-map-derived spectra). Replacing the shortcut with a Cholesky+eigvalsh symmetric reduction agrees with direct SVD to fp32 noise floor.

Two consequences:

1. The "F2 — power-iter under-estimate causes bound violations" framing in the prior version of this doc was wrong. The optimizer was correct.
2. The σ_max-via-power-iter → σ_max-via-eigh switch (commits 57a932b, 54311ba) was a *cleanliness* fix (exact, deterministic, ~1% step overhead at r ≤ 256), not a *correctness* fix. It's still worth keeping in tree.

Methodological lesson committed to `~/.claude/CLAUDE.md` ("Suspect the probe before the theorem"): when a diagnostic value contradicts a mathematical bound the code is supposed to enforce, validate the probe against a direct independent computation before chasing the algorithm.

## What the diagnostic ladder still tells us

Probe formulas now in `lora_playground/optim.py:3479–3620`. Observation-only, gated by `--log_optim_diagnostics`, verified byte-identical-to-off.

| Probe | What it measures | Variant it informs |
|---|---|---|
| **A. chord_slack** | `‖ΔW‖₂ / η` via 2r×2r Cholesky+eigvalsh on the rank-≤2r factorization `L = [B+dB, B]`, `R = [A+dA; -A]` | Whether the worst-case ρ leaves spectral-step budget on the table → variants 1 / 2 |
| **B. lambda_dir_gain** | `λ_dir / ρ` where `λ_dir` solves `a λ + b λ² = η` with `a = ‖B P‖_2 + ‖Q A‖_2`, `b = ‖Q P‖_2`, `P, Q` unit-norm factor directions; σ_max's computed via Cholesky+eigvalsh on small Grams | Direct measurement of variant 1's ρ improvement |
| **C. sat_frac_tight, cos_polar_clip_tight** | Saturation fraction of the whitened cost above the §8 threshold `τ_A = ρ/‖D_A‖_2`, and the (whitened-space) cosine between polar and clip-at-τ directions | Variant 3 — at saturating thresholds, polar ≡ clip exactly |
| **D. adam_gauge_residual** | `‖u_A A^T − B^T u_B‖_F / max(‖u_A A^T‖_F, ‖B^T u_B‖_F)`; identically zero for raw factor gradients (Adam preconditioning breaks the identity) | Generic measure of Adam-induced geometric distortion |

## Findings (corrected, r=64, local A6000 30-step smoke)

| Probe | r=64 median | r=64 max-pair |
|---|---|---|
| chord_slack | 0.73–0.78 | ≤ 0.86 |
| lambda_dir_gain | **1.27–1.33** | ≤ 1.42 |
| dir_a_over_s | 0.75–0.79 | — |
| sat_frac_tight_A/B | 1.000 | 1.000 |
| cos_polar_clip_tight_A/B | 1.000 | 1.000 |
| adam_gauge_residual_rel | 0.013–0.028 | — |

(SLURM v3 will give 500-step medians; the qualitative findings won't move.)

### F1 — Variant 3 is a no-op at this scale (unchanged)

`sat_frac_tight = 1.000` universally on both r=16 and r=64. The whitened cost has every singular value above the saturating threshold τ. Clip = polar exactly. Variant 3 (exact clip prox) replaces polar with clip → no-op by construction. **Skip variant 3.**

### F2 — Bound is held (replaces the wrong earlier F2)

`chord_slack ≤ 0.86` for all pairs across all probe steps at r=64. The optimizer satisfies `‖ΔW‖₂ ≤ η`. There is ~14–25% of η budget being left unused by the worst-case ρ — that's the room variant 1 can recover.

### F3 — Variant 1 is more interesting than I initially thought

`lambda_dir_gain` median = 1.30 at r=64 (max 1.42) — direction-aware ρ is **30% larger** than worst-case ρ on average. With chord_slack ≈ 0.78 currently, variant 1 would push `‖ΔW‖₂ / η` toward 0.78 × 1.30 ≈ 1.0 (saturating the budget). At fixed step budget the loss-per-step gain depends on training-regime efficiency; rough projection: **5–10% loss-per-step improvement at the 2k canonical horizon**, well above noise floor (σ_AdamW for r=16/64 at 2k = 0.0006/0.0007 → 1σ ≈ 0.1% of typical loss; expected effect ~50σ_AdamW).

(The prior wrong probe had `lambda_dir_gain ≈ 1.05` and `dir_a_over_s > 1`, both probe artifacts.)

### F4 — Variant 2 marginal value is small (unchanged conclusion)

With variant 1 closing most of the budget gap, variant 2 (exact low-rank chord-norm bisection — no triangle slack on top of the direction-aware bound) recovers at most 2–5% more step size at ~10× the per-step cost of variant 1. Likely net-zero or negative on wall-time-adjusted loss. **Skip variant 2.**

### F5 — Adam preconditioning is mildly geometry-distorting (unchanged)

`adam_gauge_residual_rel ≈ 0.01–0.03` early in training (rises modestly with training progression). For raw factor gradients this identity holds exactly; Adam's coordinate-anisotropy breaks it by a few percent. Useful baseline for cross-optimizer comparisons later; not load-bearing for the variant decisions.

## Decisions

1. **Variant 1 (direction-aware ρ)** — implement. Expected ~5–10% loss-per-step gain, well above noise. Production implementation should use the same Cholesky+eigvalsh form the probe uses (or warm-started power iter for σ_max(BP), σ_max(QA), σ_max(QP) reusing the existing infrastructure — see related discussion in commit notes and `docs/papers/su_spectral_norm_kexue_11736.md` on Krylov-accelerated alternatives).
2. **Skip variant 2** (low marginal value after variant 1).
3. **Skip variant 3** (sat_frac = 1.0 universally; clip ≡ polar at this scale).

## Caveats

- **Single-seed, 30-step local smoke** for the numbers in this doc; SLURM 500-step run (job 6382591) is pending. Qualitative findings won't change but absolute numbers may shift slightly with training progression.
- The previously-reported chord_slack > 1 numbers in `logs/chord_tight_diag_500_r16r64` and `logs/chord_tight_diag_500_r16r64_eigh*` are PROBE ARTIFACTS, not real bound violations. Do not use those numbers in any analysis or paper.
- **No direct power-iter accuracy probe was in Stage-0.** The under-estimate of σ_max via power-iter (real, just not the dominant cause of any visible problem) was indirectly inferred from buggy probes. Future iterative-numerical-method additions should ship with a direct-accuracy probe option (e.g. `sigma_powiter / sigma_eigh` ratio).
- Probe C-tight uses the whitened-space cosine, not the unwhitened applied-direction cosine. At `sat_frac = 1.0` they coincide; if any future setting drives `sat_frac < 1`, the unwhitened version would be needed.

## Reproducing

```bash
# Tokenized data (one-time)
python scripts/data/prepare_data.py \
    --model_name allenai/OLMo-2-0425-1B \
    --max_seq_length 512 --max_train_samples 32000 --max_eval_samples 512 \
    --seed 0 --out_dir data/magicoder_seq512_32k_packed \
    --data_pipeline_version packed_v1

# Sweep submission (2 cells parallel on 2 A100s, ~25 min wall)
SWEEP_SCOPE="diagnostics" \
SWEEP_PURPOSE="..." \
./slurm_scripts/submit.sh \
    params/chord_tight_diag_500_r16r64_v3.json \
    chord_tight_diag_500_r16r64_v3 \
    2 \
    scripts/sweep/sweep_500_r_diag.sh
```

Probe values via the canonical loader:

```python
from lora_playground.loader import load_runs
import json, statistics
runs = load_runs(where={"data_pipeline_version": "packed_v1",
                        "optimizer": "adam-polar-product-lora-coupled-spectral-chord-tight"})
# Read optim_step events; aggregate chord_slack / lambda_dir_gain medians per rank.
```
