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

## Protagonist (the cheap variant) — LOCKED 2026-06-11 (pivot: diag-Shampoo → KL-diag)

KL-diag (coupled Kronecker input diagonal) + **full polar (PolarExpress, PE=8)**, k=1,
**Nesterov momentum (β1=0.9)**, inverse-sqrt by **Polar-Express Gram NS** (`gram_ns`).
Per factor (A-side; B-side symmetric):
1. Nesterov momentum `m_A ← β1 m_A + g_A`, lookahead `g̃_A ← g_A + β1 m_A` (β1=0.9)
2. two-sided whiten `z_A = S_curv,A^{-1/2} m̂_A D_in^{-1/2}` — partner-side dense `r×r` Gram
   root **plus large-axis diagonal** curvature, where the diagonal is the **KL-coupled**
   Kronecker diagonal `D_in[i] = EMA[g_A[:,i]ᵀ S_a^{-1} g_A[:,i]]` (each side whitened by the
   OTHER's inverse before forming), NOT raw grad energy. This coupling is the only change vs the
   old diag-Shampoo protagonist and the one that matters: at 8B/r256 the raw-energy diagonal
   underperforms by ~9σ, and the gram_ns sanity confirmed that gap is the **diagonal rule**, not
   a stale-eigenbasis artifact (fresh S_a^{-1/2} gives the raw-energy diag the same uniform ~1–2σ
   edge it gives every cell, and does not close the gap).
3. polar cap `z_A ← φ(z_A)` (**full polar, PE=8** — `--polar_method polar_express
   --muon_ns_steps 8`; sigma_max-guarded estimator)
4. unwhiten `W_A = S_curv,A^{-1/2} φ(z_A) D_in^{-1/2}`
5. operator-norm radius `ρ = η/(σ_max(A)+σ_max(B))`, `dA = −ρ W_A/σ_max(W_A)`

The small-side `S_a^{-1/2}` is computed by **Polar-Express Gram Newton–Schulz** (`gram_ns`, 8
iters, fp32): eigh-free, exact every step (no 10-step-stale eigenbasis), wall-parity with the
amortized QR path. One Gram-NS framework with shared Polar-Express coefficients serves BOTH the
inverse-sqrt and the matrix-sign/polar (`docs/notes/inverse_sqrt_variant_plan.md`).

**Exact config string** (the one all coverage/ablation/walltime numbers must use):
`--optimizer kl-diag-polar-lora --polar_method polar_express --muon_ns_steps 8
--cw_picard_iters 1 --curvature_beta 0.99 --precond_delta 1e-4 --precond_refresh_every 10
--cw_nesterov --beta1 0.9 --precond_method gram_ns --higham_iters 8`. δ=1e-4 is locked —
cross-family δ sweeps (chord-tight/curvature-whiten) show it's insensitive; no δ pilot needed.

**Protocol (LOCKED):** global batch 16 (`--batch_size 4 --grad_accum_steps 4`),
`--max_seq_length 2048`, `--max_steps 9000`, `--eval_every 250`, `packed_v1.1`, bf16,
`--compile`. Matches the existing OLMo protagonist + AdamW runs, so new cells stay
comparable — no re-run for consistency.

**Nesterov decision:** IN the protagonist. Step-matched Δ vs plain-EMA
(σ_AdamW = 0.0017): OLMo opc r64 +0.94σ, opc r256 **+1.33σ**, openmath r256 +0.24σ —
consistently positive, and opc r256 exceeds the 1σ floor, so it is NOT "within noise": a
small but real gain. The step-matched Δ above is the justification; no separate ±Nesterov
ablation row.

**β1 = 0.9.** Scoped β1 sweep (`kl-diag-polar-lora`, OLMo opc r256, **full 9000 horizon**):
best-lr β=0.9 → **0.7357** (lr=0.03) vs β=0.95 → **0.7377** (lr=0.01), Δ = **+0.0020 ≈ 1.2σ**
in favor of 0.9; the lower-lr extension confirmed lr=0.01 IS β=0.95's interior optimum
(lr ∈ {3e-3,1e-3,3e-4} all ≥ 0.754). We adopt **0.9** because (a) it is marginally best on the
data and (b) it is consistent with the existing β=0.9 history, keeping the rerun comparable. (An
earlier draft adopted 0.95 as the Muon-canonical value off a step-matched 5250 pilot; the
full-horizon scoped sweep reversed that — at 9000, 0.9 wins, marginally.)

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
- **C3 (ablation — the expensive machinery is dead weight).** The curvature *flavor*
  (KL-Kronecker coupling vs diagonal vs SOAP) and the Picard cross-coupling (k≥2) do not
  beat the cheap diagonal k=1 protagonist, so the cheapest variant is the recommended one.
  The ablation figure is fed by **existing** logs (`kl_shampoo_polar_*` for flavor,
  `*picard_ablation*` for refinement) — no new ablation runs. The polar cap's necessity is
  **cited from the spectral-method literature** (Muon/iMuon/Spectron), not re-tested by us:
  the non-polar variant is known-weak and removing polar is not novel over iMuon. (User's
  established premises.)
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

### Baselines

- **AdamW** (universal reference + speedup denominator) — at **all** cells; the competitive default.
- **iMuon** (the spectral rival) — **only 2 demonstration cells** (OLMo opc r256 + Qwen bengali
  r256), **lr-tuned**, to show it loses to AdamW across an in-distribution and an OOD setting.
  NOT at all cells, NOT in the rank ladder. Rationale: iMuon is slow (~3.9 s/step, ~10h/run,
  v5_warmup) and a pretraining-oriented method that underperforms AdamW in LoRA finetuning —
  it's the right rival to *cite + beat*, but running it 8× wastes GPU. (Tune its lr so the loss
  is fair; if it surprisingly competes, widen.) E0 implements it; see below.
- These two are the ONLY comparison baselines. The "expensive machinery is dead weight"
  claim is the **E2 leave-one-out ablation** (remove curvature flavor / Picard from the
  protagonist), not a separate expensive-optimizer overlay.

### Experiments in dependency order

- **E0 — iMuon baseline = the authors' vendored `v5_warmup` (done).** We call the authors'
  reference code (`third_party/imuon_muon.py`, imuon @4f1d4b1) directly via `imuon-lora`,
  `variant='v5_warmup'` (their built-in init-stable variant: joint `full` warmup to grow B
  from zero, then v5). Tests pin the wiring (`tests/test_imuon_lora.py`, 3 pass); production
  smoke PASSED (stable, eval decreasing). `momentum=0.95` Nesterov (Appendix K, matched),
  `wd=0` (our protocol), `ε=1e-6`, `ns=5`, `adjust_lr=False` (scalar lr). ~3.9 s/step.
  - **Why NOT the decoupled Corollary 4.1:** we first implemented the paper's *proven* decoupled
    closed form (`Ȧ = (BᵀB)^{-1/2} φ((BᵀB)^{-1/2} M̃_A)` … = skeleton Prop 2). It is **numerically
    non-viable at our B=0 LoRA init** — `(BᵀB)^{-1/2} ≈ δ^{-1/2}` blows the A-side step up
    (param_l2 711→1485 in 15 steps, loss flat, no viable lr — measured). That is exactly why the
    authors ship the **joint** projector form + warmup (the `Bᵀ` prefactor kills the large inverse
    at B=0). So we run their joint v5_warmup, not our decoupled implementation.
  - **Tried + rejected (perf):** the authors' BATCHED `MuonBatched` (v5) was measured *slower*
    here (4.05 vs 3.87 s/step) — heterogeneous all-linear shapes → tiny per-shape groups →
    grouping overhead > batching benefit. Stayed on per-pair `Muon` v5_warmup.
  - **Paper framing:** skeleton Prop 2 (decoupled Cor 4.1) is iMuon's canonical *theory* (and is
    bit-exact correct as an equation), but it is not numerically realizable at our init; the
    optimizer we **run** is the authors' shipped **joint** v5_warmup. One caveat sentence states
    this. **Spectron stays argument-only.**
  - Run at **2 demonstration cells only** (OLMo opc r256 + Qwen bengali r256), **lr-tuned**
    (short pilot first) — show it loses to AdamW. NOT the ladder, NOT all cells (it's slow +
    weak; see Baselines). **Spectron stays argument-only**.
- **E1 — coverage fill.** Locked protagonist (7 cells) + iMuon (**2** demonstration cells,
  lr-tuned) + AdamW (cell 7 only) at an lr grid ({0.01, 0.03, 0.1} brackets the OLMo optimum —
  **verify best-η isn't at a grid edge per cell**, widen if it is). Tracked in
  `paper/e1_coverage_fill.md`. Gate for the headline performance profile.
- **E2 — ablation (leave-one-out)** at **OLMo opc r256 + Qwen-bengali r256**. From the full
  method, remove exactly one component (order-independent — NOT a "peel weakest first" stack):
  - **−radius** → (KL-diag + polar, plain η). Score by the **transfer figure** (E3),
    NOT best-lr loss — the radius is a reparameterization of step magnitude, so at per-cell
    best-lr it is ~neutral; its value is rank transfer.
  - **−Shampoo** (`P,Q→I` ⟹ `C_A = BᵀB = S_B`) → partner-Gram + polar + radius = iMuon +
    radius. Predicted: tanks a lot. This is the novelty-vs-iMuon arm (the diagonal curvature
    is what we add over iMuon).
  - **No −polar arm.** Polar's necessity is cited from the spectral-method literature, not
    ablated: the non-polar variant is known-weak and removing polar is not novel over iMuon.
  - **No ±Nesterov / PE8-vs-NS rows.** Nesterov is a standard Muon trick (`~/Muon`), justified
    by the step-matched Δ already recorded above; full polar (PE8) is the locked default.
  - The C3 "expensive is dead weight" figure (flavor / Picard refinement) is fed by **existing**
    logs, not these two arms — see C3.
  - **Control lr per arm** (curvature-ON-vs-OFF is confounded by an lr-basin shift —
    `soap_curvature_whitening.md:354`).
  - No separate "incremental ladder" experiment — its rungs (iMuon, −Shampoo arm,
    protagonist, +Nesterov) are all already run for the baseline + LOO; if the LOO bars need
    a cumulative-climb narrative at writing time, re-plot the same arms (a matplotlib call,
    not an experiment).
- **E3 — lr-transfer figure (C1).** Fixed η across the Llama-math ladder (r64/r128/r256) for
  protagonist vs −radius vs AdamW; overlay the muA r^{−1/2} line. (iMuon dropped from the
  ladder — too slow to run 3× for a marginal "it also doesn't transfer" strengthening; the
  core claim is ours-transfers-vs-AdamW-doesn't.) **Empirical claim only** — no rank-invariance theorem.
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
   redundant. We do NOT run a −polar arm to refute this (non-polar is known-weak, and
   removing polar is not novel over iMuon); polar's necessity rests on the spectral-method
   literature (Muon/iMuon/Spectron). State this as a cited premise, not an empirical result.

## Decisions (resolved 2026-06-09)

- **Protagonist:** PE8 full polar + KL-diag (coupled diagonal) + polar + radius, Nesterov
  momentum, k=1, β1=0.9, gram_ns inverse-sqrt.
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
