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

(updated by autonomous agent)
