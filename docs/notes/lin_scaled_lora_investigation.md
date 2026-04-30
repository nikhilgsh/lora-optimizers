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
  Strong directional preview: Adam's v̂ is essentially erasing the Sylvester
  rotation on B; A retains some geometric signal but the net step is smaller.
- **Result:** _pending full 2k trajectory (cos rising? ‖B‖ stabilizing?)_
- **Decision:** _tbd_

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
- **Result:** _tbd_
- **Decision:** _tbd_

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
- **Result:** _tbd_
- **Decision:** _tbd_

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
- **Result:** _tbd_
- **Decision:** _tbd_

---

## Leaderboard (best η per optimizer, r=16, 2k steps)

| rank | optimizer            | best η  | eval loss | source |
|------|----------------------|---------|-----------|--------|
| 1    | adam-lin-lora        | 1e-3    | 0.7564    | `optim_compare_high_eta_2k` |
| 2    | adam-scaled-lora     | 1e-3    | 0.7572    | `optim_compare_high_eta_2k` |
| 3    | adamw                | 3e-4    | 0.7579    | `lr_sweep_2k` |
| ?    | adam-lin-lora-post   | tbd     | tbd       | H4 sweep |
| ?    | adam-scaled-lora-post| tbd     | tbd       | H4 sweep |

Target: any new entry ≤ 0.7479.
