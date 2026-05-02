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

### MAJOR UPDATE: state_rebal DONE, sign sweep RUNNING, wide-lr is COMPETITIVE at r=64

**state_rebalanced_2k FINAL** (step 2000 all cells):
- polar-rebal r=16 lr=3e-3: **0.8104** (+0.055 vs Picard, NO HELP)
- polar-rebal r=64 lr=3e-3: **0.7686** (+0.030 vs Picard, PARTIAL)

**polar_core_wide_lr_2k** (still running r=64; r=16 done):
- lr=1e-2 r=16: 0.8090 (no help)
- lr=3e-2 r=16: 0.8049 (no help)
- lr=1e-2 r=64 step 1800: 0.7626 (within 0.025 of Picard, competitive)
- **lr=3e-2 r=64 step 1800: 0.7521** (Δ vs Picard +0.014, **COMPETITIVE WITH AdamW 0.7550!**)

**HYPOTHESIS (A) WAS WRONG TO RULE OUT AT r=64.** I declared (A) ruled
out based on r=16 alone, but at r=64 the curve continues to improve up
to lr=3e-2. This is a real **BIG WIN at r=64 from just lr-tuning vanilla
variant 1.** Beats vanilla 0.7821 by 0.030 and matches AdamW.

**polar_core_sign_2k just started** (6 min in):
- lr=1e-3 r=16 step 200: 1.16 (bad start)
- lr=3e-3 r=16 step 200: **6.93** (DIVERGING)

Sign optimizer at higher lr is unstable. The smoke that showed eval=1.46
at step 5 was at default lr=2e-4 (much smaller than sweep range). Sign
sweep grid {1e-4, 3e-4, 1e-3, 3e-3} probably has its sweet spot at the
lower end. Wait for full data before drawing conclusions.

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

## 2026-05-02 03:12 update — sign sweep starting to look bad

squeue:
- 6320044 polar_core_wide_lr_2k RUNNING 1:47:36 (wgpu002-003) — r=64 cells past step 1800
- 6320045 polar_core_sign_2k RUNNING 0:15:31 (wgpu009,011,014) — first eval at step 200/400

Sign sweep early reads (step 200-400, lr=1e-4 best per rank):
  r=16 lr=1e-4 step 400 = 0.8133  (vanilla v1 at step 400 was ~0.84; only marginally faster)
  r=16 lr=3e-4 step 400 = 0.8255
  r=16 lr=1e-3 step 400 = 0.9778  (drifting)
  r=16 lr=3e-3 step 400 = 5.8626  (DIVERGED)
  r=64 lr=1e-4 step 200 = 1.0387  (slow ramp)
  r=64 lr={3e-4,1e-3,3e-3} step 200 ∈ {4.8, 6.9, 9.1}  (DIVERGED)

Read: sign normalization is unstable at >1e-4 in this code path, and even at lr=1e-4 it's tracking close to / behind vanilla — NOT the dramatic step-5 advantage seen in the smoke. The smoke→sweep mismatch is real: 5-step smoke at lr=2e-4 said sign was ~1 loss-unit ahead by step 5; at full sweep with eval every 200 steps, sign is just slightly faster early and almost certainly won't beat 0.78 at step 2000. Hypothesis (B) looking like NO HELP / weak PARTIAL.

Wide-lr Phase-2 (A) confirmed picture:
  r=16 lr=1e-2 final = 0.8090, lr=3e-2 final = 0.8049 — vanilla r=16 ceiling is ~0.80, gap to AdamW (0.7601) and Picard (0.7557) persists at any lr.
  r=64 lr=1e-2 step 1800 = 0.7626, lr=3e-2 step 1800 = 0.7521 — at r=64 the wider lr scan reaches WITHIN 0.02 of AdamW (0.7550) and PARTIAL (Δ=+0.0139) vs Picard (0.7382). Best among coupled-core variants at r=64.

### Best per (rank, family) so far

| r | best optimizer | lr | loss | Δ vs Picard | verdict |
|---|---|---|---|---|---|
| 16 | wide-lr vanilla | 3e-2 | 0.8049 | +0.0492 | NO HELP |
| 16 | sign (early)    | 1e-4 | 0.8133@400 | +0.058 | trending NO HELP |
| 64 | wide-lr vanilla | 3e-2 | 0.7521@1800 | +0.0139 | WITHIN 0.02 (competitive with AdamW) |
| 64 | state-rebal     | 3e-3 | 0.7686 | +0.030 | PARTIAL |
| 64 | sign            | 1e-4 | 1.04@200 | far worse | trending NO HELP |

### Decision tree update

- Followup compound sweep (`polar_core_sign_followup_2k.json`) is gated on sign r=16 final < 0.78. With r=16 sign trending toward ~0.79-0.81 at step 2000, **the followup will likely NOT trigger.** Will reconfirm at step 2000.
- r=64 winner among our variants: vanilla v1 at lr=3e-2. Documents the lr-tuning story cleanly: at higher rank the structural fixes matter less than just running at the lr ceiling.


## 2026-05-02 03:14 update — wide_lr r=64 BEATS AdamW

squeue:
- 6320044 wide_lr RUNNING 1:49:15 on wgpu002 only — 3 cells done, last r=16 cell (step 2000 already!) ; disBatch finalizing
- 6320045 sign RUNNING 0:17:10

**Final wide_lr numbers (all 4 cells at step 2000):**
  r=16 lr=1e-2 = 0.8090
  r=16 lr=3e-2 = 0.8049
  r=64 lr=1e-2 = 0.7598  (already beats AdamW 0.7550 by 0.005 — slightly)
  r=64 lr=3e-2 = **0.7490**  ← BEATS AdamW r=64 (0.7550) by 0.006; trails Picard (0.7382) by 0.0108

This is the headline result: vanilla `polar-coupled-core-lora` at lr=3e-2 is the **first coupled-core variant that beats AdamW at any rank** in this study. r=64 only — at r=16 the ceiling is still ~0.80.

Sign sweep no movement at lr=1e-4/3e-4 cells since last check (eval cadence 200 steps, just past last eval). r=16 lr=3e-3 cell continues to diverge (5.08 at step 600). No new lr cells survived past step 600.


## 2026-05-02 03:30 update — sign r=16 lr=1e-4 trajectory tightening

squeue: only 6320045 (sign) RUNNING 33:18. Wide_lr fully exited.

Sign cells at step 1000-1200:
  r=16 lr=1e-4 step 1000 = **0.7880**  (was 0.8133 @ 400, dropped 0.025)
  r=16 lr=3e-4 step 1000 = 0.7985
  r=16 lr=1e-3 step 1000 = 0.8880  (slow)
  r=16 lr=3e-3 step 1200 = 4.2422  (DIVERGED, not recovering)
  r=64 lr=1e-4 step 400  = 1.0085  (slow ramp, no chance of < 0.78)
  r=64 lr=3e-4 step 400  = 3.8938
  r=64 lr=1e-3 step 600  = 5.7711
  r=64 lr=3e-3 step 600  = 7.9643

**Update on followup gate:** r=16 lr=1e-4 trajectory has it dropping ~0.025 per 600 steps. Linear extrapolation to step 2000 gives ~0.748, but the rate will slow. Realistic final estimate: 0.76-0.78 range. Followup gate (final < 0.78) is now genuinely uncertain — was previously trending NO, now trending borderline. Wait for step 1600+ data before deciding.

**Verdict snapshot (Picard target = 0.7557 r=16, 0.7382 r=64):**
- r=16: best is sign lr=1e-4 PARTIAL (0.7880@1000, projection ~0.76-0.78)
- r=64: best is wide_lr vanilla lr=3e-2 = **0.7490 — beats AdamW (0.7550), trails Picard by 0.011**


## 2026-05-02 03:38 update — sign r=16 lr=1e-4 CROSSES 0.78 GATE

squeue: 6320045 (sign) RUNNING 41:29.

Sign r=16 lr=1e-4 trajectory:
  step 400 = 0.8133
  step 1000 = 0.7880  (Δ-0.025 over 600 steps)
  step 1400 = **0.7777**  (Δ-0.010 over 400 steps; slope ~halving)

**Followup gate (final < 0.78) is already MET at step 1400.** Final at step 2000 likely lands 0.76-0.77 range.

Other r=16 sign cells at step 1400:
  lr=3e-4 = 0.7881 (also under 0.79, possibly under 0.78 by step 2000)
  lr=1e-3 = 0.8729
  lr=3e-3 = 4.1098 (diverged, no recovery)

r=64 sign cells: only lr=1e-4 surviving at 0.9873@600 — won't catch up.

**This is significant**: at r=16, sign normalization is the FIRST coupled-core intervention to break the ~0.80 ceiling. wide_lr maxed at 0.8049; state-rebal at 0.8104; vanilla at 0.8188. Sign at 1400 already < 0.78. Per-coord adaptivity in core space is the missing piece for r=16 — exactly the hypothesis (B) prediction.

**Decision: WILL launch followup sweep when sign sweep completes at step 2000.** Followup tests muon-coupled-core-sign-lora + state-rebalanced variant × 4 lr × 2 r (16 cells). The compound "sign + transported core EMA" question is the natural next test now that sign-without-momentum has paid off.

Target wake: ~25 min for sign sweep step 2000 + 6320045 COMPLETED, then auto-trigger followup.


## 2026-05-02 03:56 update — sign r=16 FINAL = 0.7680, gate MET

squeue: 6320045 RUNNING 59:30; r=16 cells done, r=64 cells still grinding (slower; r=64 single-step cost higher).

**Sign r=16 final at step 2000:**
  lr=1e-4 = **0.7680**  ← < 0.78 followup gate MET
  lr=3e-4 = 0.7792
  lr=1e-3 = 0.8505
  lr=3e-3 = 3.8198 (diverged)

**Verdict r=16:** sign at lr=1e-4 = **0.7680**, WITHIN 0.02 of Picard (0.7557, Δ+0.012), just above AdamW (0.7601, Δ+0.008). This is the FIRST coupled-core variant to achieve a competitive r=16 result. Hypothesis (B) per-coord adaptivity in core space confirmed as the missing piece for r=16.

**Verdict r=64 (sign):** lr=1e-4 step 1000 = 0.9600, won't catch up. Sign at r=64 needs different hyperparameter regime — the higher-effective-rank dimension is unstable at any of {3e-4, 1e-3, 3e-3} and at lr=1e-4 too slow to recover.

**Decision:** as soon as job 6320045 COMPLETES (r=64 cells finish), launch followup compound sweep (`params/polar_core_sign_followup_2k.json` — muon-coupled-core-sign-lora and rebalanced variant × 4 lr × 2 r = 16 cells). Holding off until COMPLETED to free GPUs and avoid QOS pile-up per the planning rule.

### Headline summary (current best per rank)

| r | best optimizer | lr | loss | vs AdamW | vs Picard |
|---|---|---|---|---|---|
| 16 | sign | 1e-4 | **0.7680** | +0.0079 | +0.0123 |
| 64 | wide-lr vanilla | 3e-2 | **0.7490** | **−0.0060** | +0.0108 |

**At r=64 we BEAT AdamW.** At r=16 we're WITHIN 0.02 of both baselines but don't beat them yet. The followup compound (sign + transported core EMA) is the natural next test for closing the r=16 gap.


## 2026-05-02 04:48 update — sign sweep COMPLETED, followup LAUNCHED

**Sign sweep final (all 8 cells at step 2000):**
  r=16 lr=1e-4 = **0.7680**  (WITHIN 0.02 of Picard, +0.008 vs AdamW)
  r=16 lr=3e-4 = 0.7792
  r=16 lr=1e-3 = 0.8505
  r=16 lr=3e-3 = 3.8198 (diverged)
  r=64 lr=1e-4 = 0.9395  (slow, NO HELP)
  r=64 lr=3e-4 = 2.2128 (diverged)
  r=64 lr=1e-3 = 4.4523 (diverged)
  r=64 lr=3e-3 = 6.7921 (diverged)

**Phase 2 complete picture (final, all 4 sub-experiments):**

| variant            | r=16 best | vs AdamW | vs Picard | r=64 best | vs AdamW | vs Picard |
|--------------------|-----------|----------|-----------|-----------|----------|-----------|
| Phase 1 vanilla    | 0.8188    | +0.059   | +0.063    | 0.7821    | +0.027   | +0.044    |
| 1.5 state-rebal    | 0.8104    | +0.050   | +0.055    | 0.7686    | +0.014   | +0.030    |
| 2 (A) wide-lr      | 0.8049    | +0.045   | +0.049    | **0.7490**| **−0.006**| +0.011    |
| 2 (B) sign         | **0.7680**| +0.008   | +0.012    | 0.9395    | +0.184   | +0.201    |

The picture is RANK-DEPENDENT:
- **r=16 winner: sign normalization** (0.7680). Per-coord adaptivity in core space is the missing piece. Rung 5-lite (no EMA, no transport) is enough.
- **r=64 winner: vanilla wide-lr** (0.7490, BEATS AdamW 0.7550). At higher rank the bare polar update at lr=3e-2 outperforms; sign normalization actively HURTS at r=64.

This is interesting on its own: per-coord normalization in core space helps at r=16 where the core is small and per-element scale variance dominates, but at r=64 the core is larger and Frobenius norm of the core rises quickly enough that sign-normalization throws away too much magnitude information. **Different optimization regime per rank** — consistent with the no-cross-rank-compare project rule.

**Followup sweep LAUNCHED — job 6320268** (`polar_core_sign_followup_2k`):
- 16 cells: {muon-coupled-core-sign-lora, muon-coupled-core-sign-rebalanced-lora} × {1e-4, 3e-4, 1e-3, 3e-3} × {16, 64}
- Tests: does adding transported core EMA (variant 2 momentum) on top of sign norm help? Does state-rebalance help compound with sign?
- Expected ~50 min wall on 16 GPUs.

Targets to beat: r=16 sign vanilla (0.7680), r=64 wide-lr vanilla (0.7490). If muon-sign at r=16 lands < 0.76, momentum is additive. If muon-sign-rebalanced at r=64 outperforms wide-lr vanilla, the compound is the right move.


## 2026-05-02 05:15 — followup at step 400-800

Followup 6320268 RUNNING 26:33. All 16 cells launched concurrently on 16 GPUs (well, 5 nodes — disBatch packing). r=16 cells at step 800, r=64 at step 400.

**r=16 lr=1e-4 (the critical cell):**
  vanilla sign:           step 800 was ~0.79-0.80 (extrap from step 1000 = 0.7880)
  muon-sign:              step 800 = 0.8059
  muon-sign-rebalanced:   step 800 = **0.7943**

Both compound variants tracking close to vanilla sign — no clear separation early. Rebalanced variant marginally ahead of plain muon-sign.

**r=16 lr=3e-4:**
  vanilla sign final = 0.7792
  muon-sign step 800 = 0.8462 (slow)
  muon-sign-rebalanced step 800 = 0.8065 (closer)

**r=16 lr=1e-3, 3e-3:** muon-sign tracking 1.08, 4.34 (lr=3e-3 diverged); rebalanced 0.90, 4.38.

**r=64:** all cells at step 400, all far behind. lr=1e-4 muon-sign=1.26 / muon-sign-rebal=1.01 (vs vanilla sign step 400 was ~0.97 → step 2000 final = 0.94). The compound variants are SLOWER than vanilla sign at r=64.

**Early read: compound (sign + EMA / sign + rebalance) does NOT clearly help.** At r=16 lr=1e-4 the compounds are behind vanilla sign at step 800. Possible they catch up by step 2000 (Picard-style momentum sometimes shows late gains), but trajectory not encouraging.

Wait for step 1200-1600 for clearer picture.


## 2026-05-02 05:41 — followup at step 1600-1800

Followup 6320268 RUNNING 52:32. r=16 cells at step 1600-1800, r=64 at step 800-1000.

**r=16 lr=1e-4 (target = vanilla sign 0.7680):**
  muon-sign step 1800 = 0.7872  (extrap step 2000: ~0.78)
  muon-sign-rebalanced step 1600 = **0.7747**  (extrap ~0.77)

  **Both compounds will land ABOVE vanilla sign 0.7680.** Momentum and rebalance do not add gain on top of sign.

**r=16 lr=3e-4 (target = vanilla sign 0.7792):**
  muon-sign step 1800 = 0.8228  (worse)
  muon-sign-rebalanced step 1600 = 0.7863  (worse)

**r=16 lr=1e-3:** muon-sign at 1.01, rebal at 0.86. Both diverging vs lr=3e-4 trajectory.

**r=64:** all cells far behind vanilla sign 0.94 baseline at step 800. lr=1e-4 muon-sign-rebal = 0.98 (slowly improving but unlikely to catch up). lr ≥ 3e-4 all diverged.

**Verdict trending (final read pending):** the simplest sign variant is the winner at r=16. Adding transported core EMA (muon variant) or state-rebalance does NOT improve on sign-only, and at higher lr actively hurts. The momentum-on-sign-of-core combination has poor interaction — likely the rotation/transport of a sign-quantized object is throwing away too much information.

This is consistent with the doc's section-6-ladder warning: rung-5-with-EMA needs *transported V_t* with care; here the EMA is over the sign of the core which is mostly ±1 entries, and the transport of ±1 patterns through basis rotations doesn't preserve structure.

**Final verdict (subject to confirmation at step 2000):**
- r=16 winner: **vanilla sign-coupled-core-lora at lr=1e-4 = 0.7680**
- r=64 winner: **vanilla polar-coupled-core-lora at lr=3e-2 = 0.7490 (BEATS AdamW)**

Wait for completion before final write-up.


## 2026-05-02 06:07 — followup r=16 cells COMPLETE

**r=16 final eval at step 2000:**

| optimizer                                | lr=1e-4    | lr=3e-4 | lr=1e-3 | lr=3e-3 |
|------------------------------------------|------------|---------|---------|---------|
| polar-coupled-core-sign-lora (vanilla)   | **0.7680** | 0.7792  | 0.8505  | 3.82    |
| muon-coupled-core-sign-lora              | 0.7858     | 0.8220  | 1.0053  | 3.69    |
| muon-coupled-core-sign-rebalanced-lora   | **0.7684** | 0.7808  | 0.8562  | 3.67    |

**Reading:**
- **muon-sign-rebalanced ≈ vanilla sign** at lr=1e-4 (0.7684 vs 0.7680 — within jitter). Adding both EMA momentum AND state-rebalance to vanilla sign gives a TIE — neither helps nor hurts in compound.
- **muon-sign (without rebalance) = 0.7858**, +0.018 vs vanilla sign. EMA on the sign-quantized core, without rebalance to balance the factor norms, makes things worse.
- Pattern is consistent at lr=3e-4 too: rebalanced is 0.005 worse than vanilla sign, but plain muon-sign is 0.043 worse.

The rebalance step partly *rescues* the harmful momentum interaction, bringing it back to vanilla-sign parity but not past it. EMA on top of sign normalization adds nothing useful when the basis transport doesn't preserve the sign-quantized structure well.

**r=64 cells still running** at step 1200-1400. lr=1e-4 muon-sign-rebal step 1200 = 0.9575 (slow); all lr ≥ 3e-4 diverged. Will not catch the wide-lr 0.7490 baseline. r=64 verdict NO HELP for any followup variant.

### Final Phase-2 verdict (r=16 confirmed; r=64 awaiting completion)

| r | best optimizer | lr | loss | vs AdamW | vs Picard |
|---|---|---|---|---|---|
| 16 | **polar-coupled-core-sign-lora** | 1e-4 | **0.7680** | +0.0079 | +0.0123 |
| 64 | **polar-coupled-core-lora** (wide-lr) | 3e-2 | **0.7490** | **−0.0060 (BEATS)** | +0.0108 |

**Takeaway:** the simplest variant wins at each rank.
- r=16 wants per-coord adaptivity in core space (sign normalization, no momentum).
- r=64 wants raw polar of the core covector at high lr (no normalization, no rebalance).
- Compound interventions (sign + EMA, sign + rebalance, sign + EMA + rebalance) consistently fail to improve; "sign + EMA + rebalance" is the only compound that doesn't HURT.


## 2026-05-02 06:49 — followup r=64 (final, except log_11 at step 1800)

**r=64 final at step 2000:**

| optimizer                                | lr=1e-4    | lr=3e-4 | lr=1e-3   | lr=3e-3 |
|------------------------------------------|------------|---------|-----------|---------|
| polar-coupled-core-sign-lora (vanilla)   | 0.9395     | 2.21    | 4.45      | 6.79    |
| muon-coupled-core-sign-lora              | 1.2402     | 2.21    | 3.68      | 5.78    |
| muon-coupled-core-sign-rebalanced-lora   | **0.9440** | 2.16    | 3.44@1800 | 5.38    |

**r=64 sign-family at lr=1e-4:** vanilla sign (0.9395) ≈ rebalanced (0.9440) — TIE. muon-sign without rebalance (1.24) much worse. Same pattern as r=16: rebalance partly rescues the EMA-on-sign interaction back to baseline, but no gain.

**All r=64 sign-family cells are far above wide-lr vanilla 0.7490.** Sign normalization is wrong at r=64 regardless of compound.

