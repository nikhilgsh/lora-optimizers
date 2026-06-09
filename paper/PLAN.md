# Paper plan — Polar-LoRA (working title)

Status: campaign LOCKED (2026-06-09). Scope: arXiv preprint, single-seed. Headline: **we win** —
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

## Protagonist (the cheap variant) — LOCKED 2026-06-09

diag-Shampoo + **full polar (PolarExpress, PE=8)**, k=1, **Nesterov momentum (β1=0.9)**.
Per factor (A-side; B-side symmetric):
1. Nesterov momentum `m_A ← β1 m_A + g_A`, lookahead `g̃_A ← g_A + β1 m_A` (β1=0.9)
2. two-sided whiten `z_A = S_curv,A^{-1/2} m̂_A D_in^{-1/2}` — partner-side dense `r×r` Gram
   root **plus large-axis diagonal** curvature (the second Shampoo side; this is what
   iMuon omits and what Adam was implicitly providing in chord-tight)
3. polar cap `z_A ← φ(z_A)` (**full polar, PE=8** — `--polar_method polar_express
   --muon_ns_steps 8`; sigma_max-guarded estimator)
4. unwhiten `W_A = S_curv,A^{-1/2} φ(z_A) D_in^{-1/2}`
5. operator-norm radius `ρ = η/(σ_max(A)+σ_max(B))`, `dA = −ρ W_A/σ_max(W_A)`

**Exact config string** (the one all coverage/ablation/walltime numbers must use):
`--optimizer diag-shampoo-polar-lora --polar_method polar_express --muon_ns_steps 8
--cw_picard_iters 1 --curvature_beta 0.99 --precond_delta 1e-4 --precond_refresh_every 10
--cw_nesterov` (Nesterov momentum ON, β1=0.9). δ=1e-4 is locked — cross-family δ sweeps
(chord-tight/curvature-whiten) show it's insensitive; no δ pilot needed.

**Protocol (LOCKED):** global batch 16 (`--batch_size 4 --grad_accum_steps 4`),
`--max_seq_length 2048`, `--max_steps 9000`, `--eval_every 250`, `packed_v1.1`, bf16,
`--compile`. Matches the existing OLMo protagonist + AdamW runs, so new cells stay
comparable — no re-run for consistency.

**Nesterov decision:** IN the protagonist (β1=0.9). Step-matched Δ vs plain-EMA
(σ_AdamW = 0.0017): OLMo opc r64 +0.94σ, opc r256 **+1.33σ**, openmath r256 +0.24σ —
consistently positive, and opc r256 exceeds the 1σ floor, so it is NOT "within noise": a
small but real gain. The ±Nesterov ablation row *quantifies* this (it does not justify
exclusion). iMuon baseline uses its own β=0.95 (Appendix K) — intentional asymmetry, stated.

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

## Final experiments — LOCKED arXiv campaign (2026-06-09)

Scope: arXiv preprint. **Single-seed** (multi-seed held off as too expensive; σ_AdamW =
0.0017 is the noise unit — report every Δ in σ-units, reserve "matches/within noise" for
<1σ). Hardware: Blackwell RTX-PRO-6000 (canonical), `--reservation=rocky9`.

### Cell set — minimal but diverse (8 protagonist cells, 1 already in hand)

Design: **code = breadth (across model families + scale), math = depth (one rank ladder).**
Each cell earns its place; this is the whole grid.

| group | cells | illustrates | protagonist coverage |
|---|---|---|---|
| **Code @ r256** (canonical rank) | OLMo-2-1B / Qwen2.5-1.5B / Llama-3.2-1B / **Llama-3-8B** — opc r256 | "we win" × 3 model families + 1 scale point | OLMo ✅ (7 lr PE8); 3 to run |
| **OOD pair** (one rank) | Qwen2.5-1.5B opc r256 ↔ **Qwen2.5-1.5B bengali r256** | task-dependence C4 (only dataset changes) | bengali to run |
| **Rank ladder** (math, popular model) | **Llama-3.2-1B openmath r64 / r128 / r256** | lr-transfer C1 + rank-dependence C4 | all 3 to run |

- **r=256 is the canonical rank**; the Llama-math ladder is the only place rank varies.
- Llama carries the rank-transfer headline (popular model, has r64/r256 registry cells —
  Qwen has no r64 cell so can't ladder cheaply); Qwen carries the OOD headline; OLMo is the
  cheap ablation workhorse.
- **Optional free bonus:** OLMo openmath r256 (in hand) as a 2nd math data point.

### Baselines present at every breadth + ladder cell

- **AdamW** (universal reference + speedup denominator) — already covered at all cells.
- **iMuon** (the main spectral rival) — **always-present** at the breadth + ladder cells.
  E0 implements it; see below.
- **chord-tight** (the expensive family member, for "expensive machinery is dead weight").
  Already covered at 5 of the run-cells (OLMo/Qwen/Llama-3.2 opc r256, Llama-3.2 openmath
  r64/r256). **Fill 2 cells: Llama-3-8B opc r256 + Qwen bengali r256.** (KL-Shampoo
  curvature-whiten is OLMo-only → keep the *KL-Kronecker-flavor* claim scoped to the
  ablation anchors; chord-tight carries the broad claim.)

### Experiments in dependency order

- **E0 — implement iMuon baseline.** Implement the **published closed form, iMuon
  Corollary 4.1** (arXiv 2605.09238) — the *decoupled per-factor* Gram-root-whitened polar:
  `Ȧ = (BᵀB)^{-1/2} Ortho((BᵀB)^{-1/2} G_A)`, `Ḃ = Ortho(G_B (AAᵀ)^{-1/2}) (AAᵀ)^{-1/2}`,
  `G_A = A.grad`, `G_B = B.grad`. This is **bit-exact equal to skeleton Prop 2** (verified
  numerically, rel ≈ 2e-15). With-momentum runs use **per-factor Nesterov β=0.95** (Appendix
  K). Add to `optim.py` + `OPTIMIZER_CHOICES`; behavioral-equivalence test vs a direct
  SVD-based Cor 4.1 (rel < 1e-6). Run at the breadth + ladder cells, **especially the rank
  ladder** (does iMuon's best-η drift with rank — the C1 evidence). **Spectron stays
  argument-only** (native pretraining; our −radius ablation arm already exercises its
  mechanism internally).
  - **Subtlety to note in the paper (one sentence):** the authors' reference code ships a
    *joint-`M_t`* variant (`M_t = M_B A + B M_A`, project-then-orthogonalize — their `v5`,
    used for the Table-1 headline) that does **not** match the proven Cor 4.1 closed form
    (verified: every repo variant rel ≥ 0.15, cos ≤ 0.99) and is structurally the paper's
    *Riemannion* (coupled tangent projection), a baseline iMuon beats. We benchmark the
    canonical decoupled closed form; no evidence the joint form helps.
- **E1 — coverage fill.** Locked protagonist across the 7 to-run cells at an lr grid
  ({0.01, 0.03, 0.1} brackets the OLMo optimum — **verify best-η isn't at a grid edge per
  cell**, widen if it is). Gate for the headline performance profile. Also run the 2
  chord-tight fill cells here.
- **E2 — ablation (leave-one-out)** at **OLMo opc r256 + Qwen-bengali r256**. From the full
  method, remove exactly one component (order-independent — NOT a "peel weakest first" stack):
  - **−radius** → (diag-Shampoo + polar, plain η). Score by the **transfer figure** (E3),
    NOT best-lr loss — the radius is a reparameterization of step magnitude, so at per-cell
    best-lr it is ~neutral; its value is rank transfer.
  - **−Shampoo** (`P,Q→I` ⟹ `C_A = BᵀB = S_B`) → partner-Gram + polar + radius = iMuon +
    radius. Predicted: tanks a lot.
  - **−polar** (`φ→id` ⟹ `dA ∝ C_A^{-1} g_A Q^{-1}`) = AdaPreLoRA. **Keystone arm, nearly
    unrun** — highest priority.
  - plus **±Nesterov** and **PE8-vs-single-NS** rows.
  - **Control lr per arm** (curvature-ON-vs-OFF is confounded by an lr-basin shift —
    `soap_curvature_whitening.md:354`).
  - No separate "incremental ladder" experiment — its rungs (iMuon, −Shampoo arm,
    protagonist, +Nesterov) are all already run for the baseline + LOO; if the LOO bars need
    a cumulative-climb narrative at writing time, re-plot the same arms (a matplotlib call,
    not an experiment).
- **E3 — lr-transfer figure (C1).** Fixed η across the Llama-math ladder (r64/r128/r256) for
  protagonist vs −radius vs iMuon vs AdamW; overlay the muA r^{−1/2} line. **Empirical claim
  only** — no rank-invariance theorem.
- **E4 — walltime.** Profile the protagonist's per-step (fwd/bwd/opt split) vs AdamW at the
  headline cells, **global batch ∈ {16 (comparison horizon), 64 (timing-only bench)}**.
  Publish walltime speedup = step-speedup ÷ per-step-ratio. **Never profiled the
  protagonist's wall — only chord-tight's** (`walltime_profile.md`); extend
  `scripts/bench/bench_optimizer_step.py`. Benchmark the PRODUCTION config (PE8, the exact
  flags above), not `build_optimizer` defaults.
- **E5 — task/rank dependence (C4).** Falls out of E1. OOD: the Qwen opc→bengali pair (same
  model+rank, only dataset). Rank: the Llama-math ladder. **Honest framing:** the
  "speedup *grows* with rank" result is a fresh measurement on Llama — if it doesn't
  replicate, C4 weakens to "lr transfers" (the robust claim). Instrument for both.
- **E6 — pass@1 (minimal downstream).** **HumanEval pass@1** on the **4 code-@-r256 cells**,
  protagonist vs AdamW at the best-lr final adapter. Net-new infra: (a) build a HumanEval
  harness (base + LoRA adapter → generate → exec-test); (b) **retain the final adapter** on
  those 4 cells (override the auto-`rmtree` in `train.py`). Math pass@1 (GSM8K/MATH) →
  future work.

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

## Decisions (resolved 2026-06-09)

- **Protagonist:** PE8 full polar + diag-Shampoo + polar + radius, plain momentum, k=1.
  Nesterov → ablation row.
- **Scope:** arXiv, single-seed (multi-seed held off — too expensive), σ_AdamW as Δ unit.
- **Canonical rank:** r=256; rank varies only on the Llama-math ladder.
- **iMuon:** run as the always-present spectral baseline (E0). **Spectron:** argument-only.
- **Ablation anchors (E2):** OLMo opc r256 + Qwen-bengali r256; leave-one-out (not a peel
  stack); no separate incremental-ladder experiment.
- **pass@1:** HumanEval on the 4 code-@-r256 cells (protagonist vs AdamW); math downstream
  deferred.
- **8B:** one cell (Llama-3-8B opc r256); a 2nd 8B (math) is a stretch goal, not minimal.

### Still open

- Final method name (`\methodname` placeholder = `Polar-LoRA`).

## Limitations to state explicitly in the paper

1. **Single-seed.** Headline speedup and all ablation Δ are single-seed, calibrated against
   σ_AdamW. The arXiv→conference upgrade path is seeds on the headline cells.
2. **Loss-primary.** pass@1 only on the 4 code cells; no math downstream. Lower eval loss
   ≠ better pass@1 in general.
3. **Coverage asymmetry.** Math = essentially the one Llama ladder; OOD = a single Bengali
   cell; 8B = one cell / one task. Each is a clean controlled comparison, but write
   "two task families, one OOD point, one scale point," not broad coverage.
4. **Rank-growth is OLMo-historical / Llama-fresh.** The "speedup grows with rank" claim
   rests on the Llama-math ladder (new) plus OLMo-openmath (historical); the robust claim is
   lr transfer, not growth.
