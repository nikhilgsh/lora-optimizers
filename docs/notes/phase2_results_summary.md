# Phase 2 results synthesis

Final at-rest view written after all Phase-2 sweeps completed. The running
log of timestamped checks is at `docs/notes/phase2_autonomous_progress.md`.

## TL;DR

The two best coupled-core variants in this study, **at the canonical 2k-step
horizon**:

- **r=16: `polar-coupled-core-sign-lora` at lr=1e-4 → 0.7680**
  - +0.008 vs AdamW r=16 (0.7601)
  - +0.012 vs hybrid Picard r=16 (0.7557)
- **r=64: `polar-coupled-core-lora` (vanilla, wide-lr) at lr=3e-2 → 0.7490**
  - **−0.006 vs AdamW r=64 (0.7550) — first coupled-core variant to BEAT AdamW at any rank**
  - +0.011 vs hybrid Picard r=64 (0.7382)

The picture is **rank-dependent**: per-coord adaptivity in core space (sign
normalization, no momentum) is the right intervention at r=16, while raw polar
of the core covector at high lr (no normalization, no rebalance) is the right
intervention at r=64. Compound interventions (sign + EMA, sign + rebalance,
sign + EMA + rebalance) all consistently fail to improve, with sign + EMA
without rebalance actively hurting at r=16 by +0.018.

## Full final eval table (single-seed, step 2000, m=1)

| variant                                  | r=16 best  | r=64 best  |
|------------------------------------------|------------|------------|
| AdamW (baseline)                         | 0.7601     | 0.7550     |
| Hybrid Picard (`adam-polar-product-lora-coupled`) | 0.7557 | 0.7382 |
| Phase 1 vanilla `polar-coupled-core-lora` | 0.8188 (lr=3e-3) | 0.7821 (lr=3e-3) |
| Phase 1.5 `polar-coupled-core-state-rebalanced-lora` | 0.8104 (lr=3e-3) | 0.7686 (lr=3e-3) |
| Phase 2 (A) wide-lr vanilla              | 0.8049 (lr=3e-2) | **0.7490 (lr=3e-2)** |
| Phase 2 (B) `polar-coupled-core-sign-lora` | **0.7680 (lr=1e-4)** | 0.9395 (lr=1e-4) |
| Followup `muon-coupled-core-sign-lora` | 0.7858 (lr=1e-4) | 1.2402 (lr=1e-4) |
| Followup `muon-coupled-core-sign-rebalanced-lora` | 0.7684 (lr=1e-4) | 0.9440 (lr=1e-4) |

**Bold** = chosen winner at each rank.

## Key findings

1. **Hypothesis (A) — lr ceiling — confirmed at r=64, ruled out at r=16.**
   Vanilla variant 1 at lr=3e-2 reaches 0.7490 at r=64, beating AdamW. At r=16
   the same variant tops out at 0.8049 — gap to baselines persists. The lr
   ceiling story is rank-dependent.

2. **Hypothesis (B) — per-coord adaptivity in core space — confirmed at r=16,
   harmful at r=64.** `polar-coupled-core-sign-lora` (rung 5-lite: per-step
   elementwise normalize the core covector, no EMA, no transport) achieves
   0.7680 at r=16 lr=1e-4 — the first coupled-core variant to break the
   ~0.80 r=16 ceiling. At r=64 it diverges or stalls at any lr.

3. **State-gauge rebalance does what it's designed to but doesn't translate
   to eval gain.** Imbalance residual `‖AA^T − ρ B^T B‖_F / ...` drops from
   1.0 to 0.001 in 2 steps and stays there. Eval gap to vanilla improves only
   by 0.014 at r=64 (PARTIAL) and not at all at r=16. The dA/dB pathology was
   real but is not the primary cause of the eval-loss gap.

4. **Compound interventions don't help.** Adding transported core EMA
   (variant 2 momentum) on top of sign normalization produces no gain and
   often hurts. State-rebalance + sign + EMA is statistically tied with
   plain sign at lr=1e-4 r=16 (0.7684 vs 0.7680). The simplest variant wins
   at each rank.

5. **EMA on a sign-quantized core is harmful without rebalance.** muon-sign
   without rebalance is +0.018 worse than vanilla sign at r=16 lr=1e-4. The
   transport of ±1 patterns through basis rotations does not preserve
   structure well; rebalance partly rescues this back to baseline but adds
   no gain.

## What still trails hybrid Picard

We close 87% of the r=64 gap (vanilla 0.044 → wide-lr 0.011) and 81% of the
r=16 gap (vanilla 0.063 → sign 0.012), but hybrid Picard
(`adam-polar-product-lora-coupled`) still leads by ~0.011 at both ranks.
Picard's specific advantage may be (a) Adam-on-factors providing per-coord
adaptivity that sign-normalization approximates but doesn't exactly match,
or (b) Picard's cross-coupling iteration converging to a different fixed
point than the projected-quotient-polar half-step direction. Neither has
been characterized in this study.

## Optimizer recommendations to ship

- **`polar-coupled-core-sign-lora`** — already shipped (commit `1565976`).
  Best for r=16. Headline hyperparameter: lr=1e-4.
- **`polar-coupled-core-lora`** — already shipped (Phase 1). Best for r=64
  in this study, at higher lr than originally swept. Headline
  hyperparameter: lr=3e-2.

The `state-rebalanced` variants (Phase 1.5, commit `c8482e7`) are correct
and verified but do not justify shipping as a default — they don't help
beyond what wide-lr or sign achieve.

## What's deliberately NOT recommended

- `muon-coupled-core-lora` — variant 2 with transported core EMA. Far
  behind vanilla variant 1 at all ranks (0.9073 / 0.8883 vs 0.8188 / 0.7821).
  The bias correction was verified canonical-Muon style after the fix.
  The basis-rotation EMA-transport is the principled answer to "natural
  Muon-style on a LoRA tangent" but does not pay off in practice.
- All `*-sign-rebalanced-lora` and `muon-*-sign-*` compound variants. Net
  neutral or harmful in every cell tested.

## Reproducibility

All results: `lora_playground.loader.load_runs(where={...})` over groups
`polar_coupled_core_2k`, `state_rebalanced_2k`, `polar_core_wide_lr_2k`,
`polar_core_sign_2k`, `polar_core_sign_followup_2k`. Single-seed, m=1,
canonical 2k-step horizon, seed 0. Diagnostics
(`gamma`, `relgap`, `compat`, `imbalance_residual`, `ratio_dA_dB`, etc.)
attached on every cell via `--log_optim_diagnostics`. Run
`scripts/phase2_summary.py` for the canonical comparison table.
