# Phase 2 autonomous run — progress log

**Status**: in progress (overnight run). User asleep; agent updates this
file periodically as sweeps execute.

## Plan summary

After Phase 1 found that variant 1 (`polar-coupled-core-lora`) trails
hybrid Picard by ~0.06 in eval-loss at 2k steps, and post-Phase-1
state-gauge rebalance (`polar-coupled-core-state-rebalanced-lora`,
commit `c8482e7`) drove the iLoRA invariant to zero but didn't appear
to close the eval gap at step 200, Phase 2 tests two hypotheses:

- **(A) wider lr scan** for vanilla variant 1 at `lr ∈ {1e-2, 3e-2}`.
  Vanilla didn't diverge at 3e-3 (where AdamW does); the curve was
  monotonic-down at our previous boundary. Cheapest test of "was the
  gap just lr-tuning?".
- **(B) per-step elementwise core sign normalization**
  (`polar-coupled-core-sign-lora`, commit `1565976`). Core covector
  divided element-wise by `|.|+ε` before polar — gives Adam-like
  per-coord adaptivity in core space without the basis-rotation EMA-
  transport issue rung 5 has.

## Sweeps launched

| Sweep | Group | Job ID | Cells | sbatch | Notes |
|---|---|---|---|---|---|
| state-rebalance (Phase 1.5, was running) | `state_rebalanced_2k` | 6319984 | 16 | sbatch_4h | already running; monitoring |
| (A) wider lr | `polar_core_wide_lr_2k` | 6320044 | 4 | sbatch_4h | RUNNING |
| (B) sign norm | `polar_core_sign_2k` | 6320045 | 8 | sbatch_4h | PENDING (QOSMaxGRESPerUser); starts once state_rebalanced_2k finishes |

All 3 sweeps run on the same model/data/horizon as Phase 1 baselines
(OLMo-2-0425-1B, magicoder, 2k steps, eval every 200, m=1, seed=0).

## Smoke tests (passed)

- `tests/test_polar_coupled_core.py`: 26/26 passing (added 2 sign-specific tests).
- GPU smoke (5 steps, OLMo-1B r=4) for `polar-coupled-core-sign-lora`:
  - eval_loss: 2.58 → 2.50 → 1.97 → 1.68 → **1.46** at step 5
  - vs vanilla variant 1 same smoke: 2.58 → ... → 2.45 at step 5
  - **Sign optimizer is dramatically faster early** (Δ ≈ 1.0 by step 5).
  - Strong positive signal that per-coord adaptivity is the missing piece.

## Reference baselines (from canonical loader, m=1, step 2000)

| optimizer | r=16 | r=64 |
|---|---|---|
| AdamW | 0.7601 (lr=3e-4) | 0.7550 (lr=3e-4) |
| adam-polar-product-coupled (hybrid Picard) | 0.7557 (lr=3e-4) | 0.7382 (lr=3e-4) |
| polar-coupled-core (variant 1, vanilla) | 0.8188 (lr=3e-3) | ~0.78 (lr=3e-3, projected) |

## Verdict criteria (from plan)

| Outcome | Interpretation | Action |
|---|---|---|
| (A) wide-lr lands < 0.78 | gap was lr-tuning | ship variant 1 with extended lr range |
| (B) sign lands < 0.78 | per-coord adaptivity needed | ship `polar-coupled-core-sign-lora` |
| Both ≥ 0.81 | gap is something else | document as research finding |

## Mid-trajectory snapshots

(updates added below as data lands)

### state_rebalanced_2k (job 6319984)

(continues from earlier session — verified at step 200 the rebalance
mechanism works perfectly: imbalance ≈ 0.001, ‖B‖ at 0.5-2.9, ratio
dA/dB ≈ 0.1. But eval at step 200 essentially same as vanilla — gap not
closed early. Final at step 2000 still pending.)

### polar_core_wide_lr_2k (job 6320044)

Just launched. Updates after first eval (step 200, ~10 min from start).

### polar_core_sign_2k (job 6320045)

Pending QOS slot. Updates once started.

---

## Progress updates (most recent on top)

### Update at 1:55 elapsed — state_rebal r=64 trending toward BIG WIN

state_rebal r=64 lr=3e-3 trajectory continuing strongly:
- step 1400: 0.7786
- step 1600: 0.7744
- step 1800: **0.7713**

Drop rate ~0.003 per 200 steps. Final at step 2000 projects ~0.768.
**< 0.78 = BIG WIN at r=64.** Comparable to AdamW 0.7550, Δ to hybrid
Picard 0.7382 = +0.03.

The asymmetry remains stark:
- r=16 final = 0.8104 (NO HELP)
- r=64 projected final ~0.77 (BIG WIN)

State-rebalance is more impactful at higher rank. Possibly because
ρ = r/d_out is less extreme at r=64 (0.016 vs 0.004 at r=16) → the
iLoRA target geometry is less far from initialization → easier to
maintain over training.

### Update at 1:24 elapsed — state_rebal r=64 step 1400 BEATS vanilla

**State-rebal r=64 lr=3e-3 step 1400 = 0.7786** vs vanilla r=64 final
0.7821. State-rebal beats vanilla's *full* trajectory by step 1400.

Trajectory:
- step 800: 0.7942
- step 1000: 0.7884 (Δ = -0.006)
- step 1400: 0.7786 (Δ = -0.010 over 400 steps)

Linear extrapolation to step 2000 lands ~0.764. **That would beat
AdamW (0.7601) and approach hybrid Picard (0.7382).** Δ to hybrid Picard
= +0.026 — closer to "competitive" than "PARTIAL".

So state-rebalance has an asymmetric effect:
- r=16 final: 0.8104 (no help, +0.055 from baseline)
- r=64 projected final: ~0.764 (PARTIAL/competitive, +0.026 from baseline)

Possibly because ρ = r/m is less extreme at r=64 (0.016 vs 0.004 at r=16),
making the iLoRA invariant easier to maintain.

### Update at 1:07 elapsed (state_rebal r=64 still running)

State_rebal r=64 lr=3e-3 step 1000 = **0.7884**. Drop rate from step
800 → 1000 is -0.006 per 200 steps and slowing. Linear extrapolation
to step 2000 lands ~0.77. **Δ vs hybrid Picard 0.7382 = +0.03,
PARTIAL improvement at best for r=64.**

So r=64 is slightly better than r=16 outcome under state-rebalance,
but not the "BIG WIN" we hoped for in either rank.

Wide_lr: lr=1e-2 r=16 step 1400 = 0.8179 ≈ vanilla 0.8188 final.
Confirms (A) ruled out.

Sign sweep still pending. Both other sweeps need to wrap up first.

### state_rebalanced_2k r=16 FINAL (step 2000)

| optimizer | r | best_lr | final eval |
|---|---|---|---|
| polar-coupled-core-state-rebalanced-lora | 16 | 3e-3 | **0.8104** |
| muon-coupled-core-state-rebalanced-lora | 16 | 3e-3 | 0.8641 |

**State-rebalance r=16 verdict: borderline PARTIAL (0.81 boundary).**
- Δ vs hybrid Picard 0.7557 = +0.055.
- Δ vs vanilla 0.8188 = -0.008 (tiny improvement).
- Doesn't move the needle meaningfully. The "rebalance the factor
  state to iLoRA invariant" intervention is structurally correct but
  empirically doesn't translate to substantially better eval at r=16.

r=64 cells still running at step 1000 (eval ≈ 0.7884). Could land
0.76-0.78 final, which would be a clearer win at r=64. Wait and see.

### Update at job-elapsed 58 min (state_rebalanced) + 32 min (wide_lr)

Summary script output:

**State-rebalance r=16 step 1800: 0.8130** (vanilla final 0.8188).
Drop rate ~0.003/200 steps → projected step-2000 final ~0.811. Still
firmly in "PARTIAL" or "NO HELP" band, NOT a big win at r=16.

**State-rebalance r=64 step 800: 0.7942** (still mid-trajectory).
Linear extrapolation to step 2000 lands ~0.76-0.77. **Could match or
beat AdamW (0.7550) and approach hybrid Picard (0.7382)** at r=64.
Worth watching to step 2000.

**Wide-lr** r=16 lr=1e-2 step 1000 = 0.8268 (slightly better than
vanilla's same-step trajectory but trending toward 0.80-0.82 final).
r=64 lr=3e-2 step 400 = 0.8093 (similar). **Hypothesis (A) ruled out**:
no lr in {1e-2, 3e-2} closes the baseline gap.

**Sign sweep**: still PENDING. state_rebalanced needs to finish.

### state_rebalanced_2k r=64 mid-trajectory (step 800)

Pulled r=64 cells specifically (slower per step than r=16):

| lr | optimizer | step 800 eval |
|---|---|---|
| 1e-4 | polar-rebal | 0.8803 |
| 3e-4 | polar-rebal | 0.8479 |
| 1e-3 | polar-rebal | 0.8161 |
| **3e-3** | **polar-rebal** | **0.7942** ⭐ |

**This looks much more promising than r=16!**
- Vanilla r=64 lr=3e-3 step 2000 final = 0.7821.
- State-rebal r=64 lr=3e-3 step 800 = 0.7942 — already close.
- If trajectory continues at the same rate to step 2000, lands maybe
  0.76-0.77, BEATING vanilla and getting close to hybrid Picard 0.7382.

This is a candidate for state-rebalance being a clear win at r=64
(asymmetric d_in vs d_out matters more there).

### Compound optimizer GPU smoke (3 steps, OLMo-1B r=4)

Tested the 3 new compound optimizers (registered for if sign-alone wins
the sweep and we want to try stacking):

| optimizer | s1 | s2 | s3 |
|---|---|---|---|
| polar-coupled-core-sign-rebalanced-lora | 2.5770 | **3.5494** ⚠ | 2.5524 |
| muon-coupled-core-sign-lora | 2.5770 | 2.5019 | 2.1037 |
| muon-coupled-core-sign-rebalanced-lora | 2.5770 | 2.5019 | 1.9671 |

**`polar-sign-rebalanced` has a step-2 spike** (eval grows to 3.55 then
recovers). Suggests sign-norm + state-rebalance interact badly on
variant 1 — sign produces a high-magnitude unit-direction core; rebalance
then changes coords; subsequent step's polar sees a different structure
and overshoots. Worth investigating but NOT for the followup sweep.

`muon-sign` and `muon-sign-rebalanced` look healthy (mirroring sign-only
trajectory at step 3). If sign-only wins the sweep, follow-up should be
muon-sign and muon-sign-rebalanced, NOT polar-sign-rebalanced.

### Mid-run analysis (summary script + corrections)

Built `scripts/phase2_summary.py` to autonomously pull data from all
sweep groups + baselines. Reveals two corrections to earlier framing:

1. **Phase 1 vanilla r=64 final = 0.7821** (Δ to hybrid Picard = +0.044).
   I had been quoting "+0.06 gap" but that's the r=16 gap; r=64 is
   smaller.

2. **State-rebalance not as helpful as the early-step signal suggested.**
   Mid-trajectory comparison at matched step:
   - State-rebal lr=3e-3 r=16 step 1400: 0.8189
   - Vanilla   lr=3e-3 r=16 step 2000:  0.8188 (final)
   At step 1400 state-rebal is roughly tied with vanilla's FINAL. So
   either state-rebal is mostly even with vanilla (and gets a small
   improvement by step 2000) or even slightly behind on per-step
   convergence rate.

If the trajectory continues, state-rebal r=16 final lands ~0.79-0.81
range — borderline "PARTIAL" at best. The big test remains the sign
sweep (Phase 2 B), which the smoke at step 5 strongly suggested would
help.

### Update at job-elapsed 43 min state_rebalanced + 17 min wide_lr

**state_rebalanced_2k** step 1400 r=16, step 600 r=64:
- Best r=16: log_12 polar-rebal lr=3e-3 step 1400 = **0.8189** (vs vanilla
  lr=3e-3 r=16 step 1400 = 0.8242 — Δ = -0.005, marginal).
- Best r=64: log_13 polar-rebal lr=3e-3 step 600 = **0.8021** (vs vanilla
  lr=3e-3 r=64 step 600 = 0.816 — Δ = -0.014).
- Trajectory continues to suggest **PARTIAL** improvement at most. Final
  r=16 will likely land in 0.79-0.80 range.

**polar_core_wide_lr_2k** step 600 (4 cells in flight):
- lr=1e-2 r=16: 0.8740 → 0.8518 → **0.8405** at step 600 (vs vanilla
  lr=3e-3 r=16 step 600 = 0.847 — slightly ahead by 0.006).
- lr=1e-2 r=64: 0.8328 (step 200, vs vanilla lr=3e-3 r=64 step 200 = 0.847).
- lr=3e-2 r=16: 0.9554 → 0.8884 → **0.8596** at step 600 (worse than 1e-2).
- **(A) hypothesis update:** lr=1e-2 may be a marginal best (~0.005 ahead
  of lr=3e-3). NOT a "lr was the gap" win. The eval-loss-vs-lr curve
  is flat at the 3e-3-1e-2 plateau and degrades past 1e-2.

**polar_core_sign_2k**: still PENDING. Will start when state_rebalanced
completes (~15 more min).

### GPU planning note (user feedback)

User flagged that piling 3 sweeps over the QOS limit (16+4+8 = 28 vs ~24
cap) was poor planning — sign sweep is blocked PENDING for ~25 min
behind state_rebalanced. Future sweep submissions should account for
QOS headroom upfront. Saved as memory rule
`feedback_gpu_usage_planning.md`. For this overnight run: not
canceling wide_lr; letting everything finish naturally. Sign starts
when state_rebalanced finishes (~25 more min from this update).

### Update at job-elapsed 34 min state_rebalanced + 8 min wide_lr

**state_rebalanced_2k** step 1000:
- Best r=16: log_12 polar-rebal lr=3e-3 = **0.8275** (vs vanilla extrapolated step 1000 ≈ 0.836; Δ = -0.008).
- Best r=64 (still step 400): 0.8146.
- Trajectory consistent with "PARTIAL improvement" verdict (~0.005-0.015 ahead of vanilla at matched step). Projects to ~0.79-0.80 final.

**polar_core_wide_lr_2k** step 200 (FIRST EVAL):
- lr=1e-2 r=16: 0.8740 (≈ same as lr=3e-3 step 200 = 0.8736)
- lr=3e-2 r=16: 0.9554 (WORSE than 3e-3)
- **(A) HYPOTHESIS LOOKS RULED OUT.** lr=1e-2 plateaus at 3e-3's level; lr=3e-2 is past the optimum. The curve was already at its peak at lr=3e-3. Vanilla's ceiling is ~0.819 at lr=3e-3, NOT lower at higher lr.

**polar_core_sign_2k**: still PENDING. Will start when state_rebalanced_2k frees its 16 GPUs.

### Update at job-elapsed 28 min (state_rebalanced_2k)

**state_rebalanced_2k** (job 6319984, 28 min elapsed):
- r=16 cells at step 800, r=64 at step 400.
- Best r=16 step 800: log_12 polar-rebalanced lr=3e-3 = **0.8330**
- Best r=64 step 400: log_13 polar-rebalanced lr=3e-3 = **0.8146**
- Comparison vs vanilla variant 1 step 800 (lr=3e-3 r=16, from earlier sweep): **0.847**
- Δ = -0.014. State-rebalance is **slightly ahead** of vanilla at step 800.
- Linear extrapolation to step 2000: state-rebalanced lands ~0.79-0.80 r=16. **PARTIAL improvement** band, not "big win" (< 0.78).
- All cells healthy: imbalance still ≈ 0.001, ‖B‖ growing as predicted.

**polar_core_wide_lr_2k** (job 6320044, 1.8 min elapsed): just started, no evals yet.
**polar_core_sign_2k** (job 6320045): still PENDING (QOS limit).

(updated by autonomous agent)
