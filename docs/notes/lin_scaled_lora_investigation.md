# lin-lora / scaled-lora investigation

Tracking the campaign to make adam-lin-lora and adam-scaled-lora **beat** plain
AdamW (currently essentially tied — final eval losses 0.7564 / 0.7572 / 0.7579
at best η in the 2k-step r=16 sweep).

Plan source: `~/.claude/plans/let-s-do-a-scientific-soft-lampson.md`.

Working hypothesis: composing Adam *after* the geometric (Sylvester / Gram)
solve lets Adam's per-coordinate v̂⁻¹ᐟ² normalize away the cross-coordinate
scale structure the geometric step just installed. Diagnostics confirm /
falsify this; productive code changes either reorder the composition (H4) or
swap per-coord v̂ for a per-pair scalar (H5).

Success bar (per `feedback_beat_dont_match.md`): a productive change wins iff
its best-η final eval loss is **≤ 0.7479** (i.e. ≥ 0.010 below AdamW's
0.7579). Parity is failure.

---

## H1 — Adam's v̂⁻¹ᐟ² wipes out the geometric correction

- **Status:** running (SLURM 6312334, submitted 2026-04-30)
- **Test:** `--log_optim_diagnostics` flag, per-pair cos(Δ_lin, Δ_adamw),
  Frobenius norms, ‖A‖_F/‖B‖_F, σ_min/σ_max(S_A) and (S_B), every 20 steps.
- **Falsifier:** median cos > 0.95 → confirmed; cos < 0.7 throughout but losses
  still match → falsified, look elsewhere.
- **Command:** `./slurm_scripts/submit.sh params/h1_diag_2k.json h1_diag_2k 2 scripts/sweep_2k_diag.sh`
- **Artifacts:** `logs/h1_diag_2k/run_info/logs/log_0.out` (adam-lin-lora η=1e-3),
  `log_1.out` (adam-scaled-lora η=1e-3); `optim_step` JSONL events embedded inline.
- **Smoke preview (5-step run on local A6000, before launch):** at step 20
  across 112 LoRA pairs of OLMo-2-1B, **cos_B_median = 0.98** (geometric B-step
  ≈ AdamW B-step), **cos_A_median = 0.46** (A diverges meaningfully).
  ‖dA_raw‖ / ‖dA_lin‖ ≈ 2× (Sylvester step is half the magnitude of AdamW on A).
- **Status: CONFIRMED (job 6312334 completed in 37 min).** Final values @ step 2000:

  | optimizer        | cos_A | cos_B | ‖dA_lin‖/‖dA_raw‖ | σ_min(S_B) | ‖B‖_F | final eval |
  |------------------|-------|-------|--------------------|------------|--------|------------|
  | adam-lin-lora    | 0.84  | 0.94  | 0.25               | 1.08       | 7.68   | 0.7581     |
  | adam-scaled-lora | 0.88  | 0.97  | 0.27               | 1.14       | 6.52   | 0.7592     |

  Trajectory (adam-lin-lora): cos_A rises 0.46 → 0.84 in first ~500 steps
  then plateaus; cos_B stays in [0.94, 0.99] throughout. σ_min(S_B) climbs
  0.011 → 1.08 driven by ‖B‖² growth — Gram conditioning *improves* over
  training, exactly when most of the loss reduction happens.
- **Decision: H1 confirmed.** Adam's per-coord √v̂ erases the geometric
  correction throughout training (cos_B ≥ 0.94 from step 20). The only
  meaningful direction divergence is on A in the first ~500 steps, before B
  leaves zero. Even there the geometric step is consistently ¼ the magnitude
  of plain AdamW. Net: pre-precondition compositions are ε-perturbed AdamW
  by construction. Productive change must reorder Adam ↔ geometry → H4.

## H2 — Geometric correction matters only early (init scale imbalance)

- **Status:** planned (answered by the same diagnostic run as H1)
- **Test:** trajectory of σ_min(S_B) and ‖B‖_F over training. If S_B
  well-conditioned by step ~200 and cos rises to 1, the correction lives only
  in the early window.
- **Result:** _tbd_
- **Decision:** _tbd_

## H3 — Benefit shows at small r

- **Status:** running (SLURM 6312335, submitted 2026-04-30)
- **Test:** r ∈ {2, 4, 64} × {adamw, adam-lin-lora, adam-scaled-lora} × η ∈
  {3e-4, 1e-3} via disBatch (skip r=16, already in `optim_compare_2k_1ep` /
  `lr_sweep_2k`). 18 runs, 6 GPUs, 4h time limit.
- **Falsifier:** if gap stays < 0.005 at r=2, conditioning isn't the bottleneck
  for this base+dataset.
- **Command:** `./slurm_scripts/submit.sh params/h3_rsweep_2k.json h3_rsweep_2k 6 scripts/sweep_2k_r.sh slurm_scripts/sbatch_4h.sh`
- **Artifacts:** `logs/h3_rsweep_2k/run_info/logs/log_{00..17}.out`
- **Smoke (r=64 on A6000):** peak 7.83 GB → fits A100 80 GB easily.
- **In-flight result (step 2000, η=3e-4):**

  | r  | adamw  | adam-lin-lora | gap (lin − adamw) |
  |----|--------|---------------|--------------------|
  | 2  | 0.7920 | 0.8150        | **+0.023** (worse) |
  | 4  | 0.7807 | 0.8024        | **+0.022** (worse) |
  | 64 | 0.7550 | 0.7527        | −0.002 (≈ tie)     |

- **Decision: H3's premise about small r is falsified, but H3 surfaced a
  bigger result.** Final eval table at η=3e-4, step 2000:

  | r  | adamw  | adam-lin-lora | adam-scaled-lora |
  |----|--------|---------------|-------------------|
  | 2  | 0.7920 | 0.8150        | 0.8134            |
  | 4  | 0.7807 | 0.8024        | 0.8001            |
  | 64 | 0.7550 | **0.7527**    | **0.7506** ← new leaderboard #1 |

  At r=2,4 lin/scaled lose to AdamW by ~0.02 (premise wrong as H1 explains).
  But at **r=64 both lin/scaled beat AdamW**, with adam-scaled-lora at 0.7506
  taking the leaderboard from adam-muon-lora (0.7557 at r=16).
  **Hypothesis why H1 doesn't fully apply at r=64:** at higher r the LoRA
  factor matrices are larger, so per-coord v̂ can't fully wash out the
  cross-coordinate scale structure that S_B⁻¹ installs. We did not run the
  cosine diagnostics at r=64 — a clean follow-up.

## H4 — Productive: Adam on raw grads, geometric solve on Adam step

- **Status:** running (SLURM 6312277, submitted 2026-04-30)
- **Mechanism:** swap composition order. Adam state on raw (∇A, ∇B); compute
  the unitless Adam direction u = m̂/(√v̂+ε); feed u as a synthetic gradient
  through the LinLoRA / ScaledLoRA geometric step, then apply lr afterwards.
  v̂ adapts to natural gradient distribution (its strength), geometry installs
  the (A,B)-coupled rotation post-hoc.
- **New optimizers:** `AdamLinLoRAPost`, `AdamScaledLoRAPost`
  (lora_playground/optim.py); 7 unit tests in `tests/test_optim_post.py`.
- **Sweep:** η ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} × 2 optimizers, r=16, 2k steps
  (10 runs, 4 GPUs, 4h time limit).
- **Falsifier:** best `*-post` final eval ≥ 0.7529 (within 0.005 of AdamW).
  Success bar: ≤ 0.7479.
- **Command:** `./slurm_scripts/submit.sh params/h4_post_2k.json h4_post_2k 4 scripts/sweep_2k.sh slurm_scripts/sbatch_4h.sh`
- **Artifacts:** `logs/h4_post_2k/run_info/logs/log_{00..09}.out`
- **Smoke (5-step on A6000):** adam-lin-lora-post η=3e-3 → eval 1.121;
  adam-scaled-lora-post η=3e-3 → eval 1.197. No NaN/Inf, peak 7.25 GB.
- **In-flight result (step 1400, η=1e-3):** adam-lin-lora-post → **0.7923**;
  adam-scaled-lora-post → **0.8421**. Trending to ~0.78–0.79 / ~0.84 final.
- **Decision: H4 falsifying.** Applying S_B⁻¹ to a sign-like Adam step does
  not produce a useful direction. Compare to `adam-muon-lora` which uses the
  same composition order (Adam → geometric correction) but with NS instead
  of S⁻¹, and *does* beat AdamW (0.7557). Provisional rule: post-Adam
  corrections work iff they're structurally meaningful on a sign-magnitude
  input — NS (spectral cap) qualifies, S⁻¹ (Gram-inverse rescaling) does not.
  Final verdict pending the η=3e-3 runs.

## H5 — Productive: per-pair scalar second moment (matrix-Adam)

- **Status:** running (SLURM 6312354, submitted 2026-04-30 in parallel with H4)
- **Mechanism:** keep AdamLinLoRA / AdamScaledLoRA's flow (geometry-then-Adam
  composition) but replace per-coord v̂ with a single scalar EMA per (A,B)
  pair tracking ‖precond_A‖²_F + ‖precond_B‖²_F. Direction comes from m̂
  per-element; only magnitude is adaptively rescaled per pair.
- **New optimizers:** `AdamLinLoRAMatrix`, `AdamScaledLoRAMatrix`
  (lora_playground/optim.py); 6 additional unit tests in `tests/test_optim_post.py`.
- **Sweep:** η ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} × 2 optimizers, r=16, 2k steps
  (10 runs, 4 GPUs, 4h time limit).
- **Command:** `./slurm_scripts/submit.sh params/h5_matrix_2k.json h5_matrix_2k 4 scripts/sweep_2k.sh slurm_scripts/sbatch_4h.sh`
- **Artifacts:** `logs/h5_matrix_2k/run_info/logs/log_{00..09}.out`
- **Smoke (5-step on A6000):** adam-lin-lora-matrix η=1e-3 → eval 1.186;
  adam-scaled-lora-matrix η=1e-3 → eval 1.186. No NaN/Inf, peak 7.21 GB.
- **First attempt (job 6312354) BROKEN:** at η ∈ {3e-5, 1e-4} step 1000+,
  eval stayed at 1.187 (random init). Bug: per-pair v_pair tracked Σg² (sum)
  instead of mean, giving √v̂ ≈ √N · RMS(g) and effective lr = lr/√N ≈
  lr/700 for typical LoRA shapes — no learning at the standard η range.
- **Fix (commit ac81bba):** divide by N_total = numel(A)+numel(B) so v̂
  tracks mean square. Verified: η=1e-3 step 50 → eval 0.886 (was 1.187 at
  step 1000 broken). Resubmitted as **job 6312759**, currently PENDING.
- **Result:** _tbd_
- **Decision:** _tbd_

---

## Leaderboard (best η per optimizer, r=16, 2k steps)

For the full cross-investigation leaderboard see `docs/notes/optimizer_synthesis.md`.

| rank | optimizer            | best η  | eval loss | source                      | beats AdamW?    |
|------|----------------------|---------|-----------|-----------------------------|-----------------|
| 1    | adam-muon-lora       | 3e-3    | 0.7557    | `adam_muon_2k`              | ✅ Δ=−0.0022    |
| 2    | adam-lin-lora        | 1e-3    | 0.7564    | `optim_compare_high_eta_2k` | ≈ tied          |
| 3    | adam-scaled-lora     | 1e-3    | 0.7572    | `optim_compare_high_eta_2k` | ≈ tied          |
| 4    | adamw                | 3e-4    | 0.7579    | `lr_sweep_2k`               | baseline        |
| ?    | adam-lin-lora-post   | trending 0.79 | tbd | H4 sweep (in flight)        | ❌ falsifying   |
| ?    | adam-scaled-lora-post| trending 0.84 | tbd | H4 sweep (in flight)        | ❌ falsifying   |
| ?    | adam-lin-lora-matrix | tbd     | tbd       | H5 sweep (resubmitted)      | tbd             |
| ?    | adam-scaled-lora-matrix | tbd  | tbd       | H5 sweep (resubmitted)      | tbd             |

Target: any new entry ≤ 0.7479.
