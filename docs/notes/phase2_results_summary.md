# Phase 2 results synthesis

Written autonomously while sweeps run. Updated as new data lands. The
running log of timestamped checks is at
`docs/notes/phase2_autonomous_progress.md`; this doc is the "at rest"
view for the user to read on wake.

## TL;DR (status as of last update)

We tested 3 hypothesis branches for closing the 0.06 (r=16) / 0.04
(r=64) eval-loss gap between vanilla `polar-coupled-core-lora`
(variant 1) and hybrid Picard `adam-polar-product-lora-coupled` at the
canonical 2k-step horizon:

1. **State-gauge rebalance** (Phase 1.5 → `state_rebalanced_2k`):
   mechanism works (drives iLoRA invariant to zero, preserves BA),
   but **mid-trajectory eval at step 1400 essentially matches vanilla's
   step-2000 final**. Likely PARTIAL improvement at most — a couple
   hundredths.
2. **Wider lr scan** (Phase 2 A → `polar_core_wide_lr_2k`): tests if
   the gap is just lr-tuning. **Ruled out**: lr=1e-2 plateaus at lr=3e-3's
   level (diff ≈ 0.005); lr=3e-2 is past the optimum.
3. **Per-step elementwise core sign normalization** (Phase 2 B →
   `polar_core_sign_2k`): **smoke at step 5 shows eval = 1.46 vs
   vanilla's 2.45**, a ~1-loss-unit advantage. Sweep currently QUEUED
   (waiting for state_rebalanced GPUs). This is the experiment that
   decides whether per-coord adaptivity in core space (Adam-like, the
   doc's rung-5-lite) is what was missing.

If sign wins: ship `polar-coupled-core-sign-lora` and follow up with
compound experiments (sign + state-rebalance, sign + variant 2 momentum).
If sign doesn't win: the gap is something else (Picard cross-coupling,
spectral equalization mismatch, etc.) and we file a research note.

## What's available for review on wake

### Final eval table

Run `conda run -n ffcv-pl python scripts/phase2_summary.py` for the
canonical comparison: per-(optimizer, r) best eval at step 2000, vs
AdamW + hybrid Picard baselines pulled from canonical loader, with
verdict labels (BIG WIN / PARTIAL / NO HELP).

### Trajectories

Per-cell `eval_loss(step)` is in
`logs/<group>/run_info/logs/log_NN.out` for each of the 4 phase groups:
- `polar_coupled_core_2k` (Phase 1)
- `state_rebalanced_2k` (Phase 1.5)
- `polar_core_wide_lr_2k` (Phase 2 A)
- `polar_core_sign_2k` (Phase 2 B)

Use `lora_playground.loader.load_runs(where={...})` to pull them
programmatically.

### Diagnostics

Each `optim_step` event in the log has `gamma`, `relgap`, `compat`,
`norm_A`, `norm_B`, `imbalance_residual`, `ratio_dA_dB`, plus variant-
specific extras. The state-rebalance sweep verified `imbalance_residual
≈ 0.001` sustained throughout training (confirming mechanism works).

## Code shipped

- `polar-coupled-core-state-rebalanced-lora` (commit `c8482e7`):
  variant 1 + post-step `(B,A) → (BR, R^{-1}A)` rebalance with
  `R R^T = ρ^{-1/2} S_B^{-1/2} (S_B^{1/2} S_A S_B^{1/2})^{1/2} S_B^{-1/2}`,
  ρ = r/d_out (iLoRA invariant). Preserves BA exactly.
- `polar-coupled-core-sign-lora` (commit `1565976`): variant 1 with
  pre-polar elementwise normalization. M̃ = Ĥ / (|Ĥ| + ε). Adam-like
  per-coord adaptivity in core space, no EMA, no basis-rotation
  transport issue.
- `polar-coupled-core-sign-rebalanced-lora`,
  `muon-coupled-core-sign-lora`,
  `muon-coupled-core-sign-rebalanced-lora` (commit `6de1e3a`):
  compound optimizers ready for follow-up if sign wins.
- `scripts/phase2_summary.py` (commit `0b2b689`): autonomous summary
  table + verdict labels.

## Gauge analysis findings

### State-gauge rebalance does what it's designed to

GPU smoke verified: imbalance residual `‖AA^T − ρ B^T B‖_F /
(‖AA^T‖_F + ρ ‖B^T B‖_F + ε)` drops from 1.0 to 0.001 in 2 steps and
stays there throughout 1400+ steps. ‖B‖ grows aggressively (e.g.,
0 → 0.71 in 10 smoke steps vs vanilla's 0.0016). dA/dB ratio drops
from 47-100 to 0.1-0.2 (matches predicted `√(r/d_out)` from iLoRA).

### But the eval gap mostly persists

Vanilla variant 1 r=16 final = 0.8188. State-rebalanced r=16 step 1600
= 0.8158 (Δ = -0.003). The structural fix to factor-state geometry
doesn't translate to substantially better eval. Suggests the gap is
NOT primarily about factor balance / B-growth, despite the dramatic
mechanism difference.

### Sign normalization smoke is dramatically better

Vanilla variant 1 5-step smoke: 2.58 → 2.55 → 2.54 → 2.50 → 2.45.
Sign optimizer 5-step smoke: 2.58 → 2.50 → 1.97 → 1.68 → **1.46**.

The Δ ≈ 1.0 by step 5 is far larger than any gauge-fix produced.
Adam-style per-coord normalization (in core space, not on factors)
appears to be the structurally significant intervention.

## Open compound questions

If sign wins solo, the obvious next moves:

1. **Sign × momentum** (`muon-coupled-core-sign-lora`): does adding
   variant 2's transported core EMA on top of sign-norm help or hurt?
   Smoke at step 3 shows 2.10 vs vanilla sign's 1.97 — slightly
   slower at step 3 but trajectories cross. Need sweep to know.
2. **Sign × state-rebalance**: variant-2 + sign + rebalance smokes
   cleanly (`muon-coupled-core-sign-rebalanced-lora`). Worth trying.
   (Variant-1 + sign + rebalance has a step-2 spike, unstable.
   Excluded from followup.)
3. **EMA over the sign**: rung-5-full with transported V_t. More
   complex, only worth building if sign-without-EMA is competitive.

## Risks / what could go wrong

- **Sign sweep diverges at high lr**: smoke was at lr=2e-4 default; if
  sweep at lr=3e-3 diverges, we'd see eval climb instead of drop.
  Loop monitors for this and would scancel.
- **Sweep fails to start**: queued behind state_rebalanced. If the
  state_rebalanced job hangs past 4h limit, sign sweep stays queued.
  Loop reports if this happens.
- **All cells of sign sweep land in 0.79-0.81 band**: would mean per-
  coord adaptivity also "PARTIAL" not "WIN". That'd be surprising
  given the smoke result, but possible if the smoke advantage washes
  out by step 2000.
