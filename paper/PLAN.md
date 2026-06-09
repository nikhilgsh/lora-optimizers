# Paper plan — Polar-LoRA (working title)

Status: planning (2026-06-08). Scope: workshop / arXiv preprint. Headline: **we win** —
the cheapest member of the spectral-polar LoRA-optimizer family is the fastest LoRA
optimizer across our settings, and the expensive machinery usually bolted onto it is
unnecessary. Supersedes the chord-tight-centric `docs/plans/paper_plan.md`.

Method name is a placeholder (`Polar-LoRA`); rename via the `\methodname` macro in `main.tex`.
Paper-name → source-slug map: `paper/code_map.md`.

## One-sentence contribution

A LoRA optimizer that applies a polar (spectral-cap) map to **two-sided-whitened plain
momentum** and sizes each step by a **per-factor operator-norm radius** beats AdamW by up
to 1.5–1.7× in steps across 14 (model, dataset, rank) settings, matches the expensive
members of its family (KL-Kronecker curvature coupling, Picard cross-coupling, Adam input),
and — being strictly factor-space — keeps the per-step walltime overhead small enough that
the step speedup survives into walltime.

## Protagonist (the cheap variant)

diag-Shampoo + polar, k=1. Per factor (A-side; B-side symmetric):
1. plain momentum `m_A ← β1 m_A + (1−β1) g_A`
2. two-sided whiten `z_A = S_curv,A^{-1/2} m̂_A D_in^{-1/2}` — partner-side dense `r×r` Gram
   root **plus large-axis diagonal** curvature (the second Shampoo side; this is what
   iMuon omits and what Adam was implicitly providing in chord-tight)
3. polar cap `z_A ← φ(z_A)` (single Newton–Schulz; sigma_max-guarded estimator)
4. unwhiten `W_A = S_curv,A^{-1/2} φ(z_A) D_in^{-1/2}`
5. operator-norm radius `ρ = η/(σ_max(A)+σ_max(B))`, `dA = −ρ W_A/σ_max(W_A)`

Verify exact diagonal form against `lora_playground/optim.py` before writing the §Method
equations. Derivation of record: `docs/notes/polar_product/kl_shampoo_polar_derivation.md`.

## Contributions

- **C1 (operator-norm lr → rank transfer).** Port Spectron's per-factor operator-norm
  radius `ρ = η/(σ_max(A)+σ_max(B))` to LoRA *fine-tuning*; show the stable lr follows the
  muA-predicted **r^{−1/2}** law, so a single lr schedule transfers across rank. iMuon
  deliberately removes this radius (its Gram root bounds the *product* step but leaves the
  raw factor step ∝ 1/σ_min(partner) uncontrolled, and its headline runs are momentum-free
  at r≤16). **Claim is empirical, not a theorem** (idealized theory predicts r^{−1/2}
  shrinkage, not invariance — `init_damping_math.md:586`). Do NOT write "rank-invariant lr."
- **C2 (two-sided curvature on momentum + polar).** vs iMuon: add the large-axis diagonal
  curvature (second Shampoo side). vs AdaPreLoRA (which already does two-sided diagonal
  whitening on momentum): add the **polar cap** and stay **strictly factor-space** (no
  O(d_in·d_out) compute term). vs chord-tight: drop Adam for plain momentum, replacing
  Adam's implicit diagonal preconditioner with explicit two-sided curvature.
- **C3 (ablation — what carries the gain).** The polar cap is load-bearing; the curvature
  *flavor* (KL-Kronecker coupling vs diagonal vs SOAP) and the Picard cross-coupling (k≥2)
  are not. So the cheapest variant is the recommended one. (User's established premises;
  the ablation grid is the evidence figure.)
- **C4 (step → walltime; task/rank dependence).** Report walltime speedup, not just step
  speedup; characterize how speedup grows with task OOD-ness and rank.

## Positioning table (intro + related work spine)

| method | curvature | nonlinearity | per-factor lr pin | input | regime |
|---|---|---|---|---|---|
| iMuon (2605.09238) | partner Gram (1 side) | polar | none (removed ρ) | momentum-free | LoRA r≤16 |
| Spectron (2602.12429) | none | polar | ρ=η/(‖A‖+‖B‖+1) | momentum | native pretraining |
| Tilde CM (compositional SD) | partner Gram (attn products) | polar | isotropic eff-lr | momentum | pretraining |
| AdaPreLoRA (2605.08734) | two-sided diagonal Kron | none | scalar | momentum | LoRA r≤64 |
| muA (2602.06204) | — (theory) | — | scalar (SignSGD) | — | LoRA lr-transfer theory |
| chord-tight (ours, baseline arm) | partner Gram + Adam | polar | ρ | Adam | LoRA |
| **Polar-LoRA (ours)** | partner Gram + large-axis diag | polar | ρ | plain momentum | LoRA, all r |

Citation discipline (from the related-work agents):
- **Spectron** = the operator-norm radius mechanism (their Eq. 16 == our ρ). Credit explicitly.
- **muA** = the rank-transfer theory anchor (η ∝ r^{−(1−γ)/2}); we cite the r^{−1/2} row.
- **iMuon** = partner-Gram-whitened polar on the fixed-rank LoRA manifold; credit it for
  removing runtime rescale, GL(r)-invariance, and the factor-condition-independent rate.
- **Tilde CM** = NOT a repackaging of iMuon — same atomic operator, different composition
  (attention products), no rank axis, no theory. Cite both as partner-Gram-whitened
  spectral updates, distinguish by composition.
- **AdaPreLoRA** = prior art for two-sided-on-momentum LoRA; differentiate sharply (polar +
  factor-space cost), do NOT cite as soft motivation.

## Final experiments

In dependency order. Hardware: Blackwell RTX-PRO-6000 (canonical), `--reservation=rocky9`.

- **E0 — implement iMuon (and Spectron) baselines.** BLOCKS the "we win" claim and C1.
  Add to `optim.py` + `OPTIMIZER_CHOICES`; behavioral-equivalence test vs the paper update.
  Run at our cells, **especially high-rank + with-momentum**, to measure whether iMuon's
  best-η drifts with rank / its raw factor step misbehaves (the C1 evidence).
- **E1 — coverage fill.** Polar-LoRA (diag-Shampoo+polar, k=1) across all 14 workloads, so
  the robustness ranking actually ranks it (currently the protagonist family is ≤4/14;
  chord-tight is 7/14). Gate for the headline figure.
- **E2 — ablation grid** at 2–3 anchor cells: ±polar · {KL-Kronecker | diagonal | SOAP}
  curvature · k=1 vs k=2 · momentum vs Adam input. **The −polar arm is the keystone and is
  nearly unrun today (1/14)** — highest priority. Control lr per arm (the in-house
  curvature-ON-vs-OFF A/B is confounded by an lr-basin shift — `soap_curvature_whitening.md:354`).
- **E3 — lr-transfer figure.** best-η vs rank for Polar-LoRA vs iMuon vs AdamW; overlay the
  muA r^{−1/2} line. Supports C1.
- **E4 — walltime profiling.** Profile Polar-LoRA's per-step (fwd/bwd/opt split) vs AdamW at
  each headline cell, at **global batch ∈ {16, 64}** (opt.step is fixed cost; fwd+bwd scales
  with batch → overhead-fraction shrinks with batch). Publish walltime speedup =
  step-speedup ÷ per-step-ratio alongside step speedup. The protagonist is the cheap variant
  (no dense Kronecker refresh, no Picard) so its overhead should be far below chord-tight's
  `higham` (1.18× @1B-r128, 1.46× @1B-r512 — `walltime_profile.md`). Extend
  `scripts/bench/bench_optimizer_step.py` to the protagonist. **Never profiled the
  protagonist's wall — only chord-tight's.**
- **E5 — task/rank dependence (C4).** Quantify speedup vs task OOD-ness and rank.
  - **OOD axis: well-supported, with one clean controlled pair.** Same model+rank
    (Qwen2.5-1.5B, r256), only dataset changes: opc **1.20×** → Bengali **1.57×** (+0.37×).
    The ≈1.2× regime is the *large in-distribution code* cells (Llama-3-8B opc 1.24×,
    Qwen-1.5B opc 1.20×); OLMo openmath (math, mildly OOD) beats OLMo opc (1.71× vs 1.50×
    at r256). Honest statement: **speedup is smallest on in-distribution code, and within
    that, smallest on the bigger models.**
  - **Rank axis: only testable on OLMo today.** OLMo openmath r64→r256 = 1.50×→1.71×
    (grows); OLMo opc is flat-to-mixed. **Qwen has no r64 cell; Llama-3-8B has only r256.**
    So "speedup grows with rank" rests entirely on OLMo-openmath — suggestive, not
    established. **E5a (new run): add r64 (and ideally r128) cells for Qwen-1.5B and
    Llama-3-8B** so the rank claim has a within-model comparison on the larger models.
  - Caveat for the headline: on several cells the current best variant is *chord-tight*,
    not the diag/KL-Shampoo+polar protagonist (e.g. OLMo opc r64; Bengali, where
    KL-Shampoo+polar only *ties*). E1 coverage-fill must establish the protagonist as
    ≥ chord-tight everywhere, or the "we win with the cheap variant" headline weakens to
    "the cheap variant matches the best."
- **E6 (optional at workshop scope) — multi-seed + pass@1.** σ exists for AdamW r∈{16,64}
  packed_v1; extend to headline cells. HumanEval pass@1 hook (P2) for downstream.

## Tensions to manage in the writing (pre-empt reviewers)

1. C1 vs iMuon: iMuon *chose* to remove ρ. Argue raw-factor-step blowup (1/σ_min) +
   momentum + high rank + the *empirical* flat-η; never claim a rank-invariance theorem.
2. C2 vs AdaPreLoRA: prior art for two-sided-on-momentum. Lead with polar + factor-space cost.
3. "rank-invariant lr" → "r^{−1/2}". muA invariance only at α=r^{−1}/Init[B], not our α=r.
4. Opposite-factor whitening *hurts* early-time at high rank (`chord_tight_whiten_lag_r256.md`)
   — don't claim more preconditioning is monotonically better.
5. KL-Shampoo (no polar) beats SOAP/Shampoo in full-weight pretraining
   (`soap_curvature_whitening.md` Reading B) — i.e. good curvature *might* make polar
   redundant. Our E2 −polar arm must directly refute this in the LoRA setting.

## Open decisions

- Final method name.
- Whether to include Spectron as a run baseline (E0) or argument-only.
- Which 2–3 cells anchor the ablation grid (E2).
- Multi-seed / pass@1 in or out at workshop scope.
