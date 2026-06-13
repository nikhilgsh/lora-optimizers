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
to 1.5–1.7× in steps across 8 (model, dataset, rank) settings, matches the expensive
members of its family (KL-Kronecker curvature coupling, Picard cross-coupling, Adam input),
and — being strictly factor-space — keeps the per-step walltime overhead small enough that
the step speedup survives into walltime.

## Three-pillar framing + anchor cells (2026-06-12)

The method = three things LoRA fine-tuning needs on top of raw factor momentum; each
pillar has a dedicated ablation arm at the anchor cell:

1. **Product structure** — naive per-factor polar (`muon-lora` + PE-8, the protagonist's
   own orthogonalizer so the failure is attributable to structure, not a weak NS) fails;
   the partner-Gram geometry (iMuon family) is the fix.
2. **Magnitude control** — the Spectron-ported operator-norm step dX = −ρ·W/σ_max(W),
   ρ = η/(σ_max A + σ_max B); iMuon/LoRA-Muon have no magnitude pin. Ablated by **−pin**
   (`cw_unpinned`: true-scale Gram roots, no σ_max(W) rescale, dX = −η·W raw = the family
   core at native magnitude). The unpinned core blows up at B=0 (Rem b0 / E0 measured); it
   runs only with `--lora_init_b symmetric` (PiSSA-style, step-0 = pretrained) — and that
   requirement IS the stability evidence (ours trains from standard B=0, the family core
   cannot). **NO lr-transfer claim (CUT 2026-06-12).**
3. **Curvature control** — the KL-coupled diagonal. Evidence: −curvature arm (+4.8σ at
   the anchor), the 8B raw-vs-KL diagonal gap (~9σ).

**The double = −curvature −pin = the iMuon/LoRA-Muon step.** `cw_unpinned` +
`cw_no_diag_curv` + `--lora_init_b symmetric`: bare partner-Gram, no pin, no curvature = the
decoupled sandwich (≡ LoRA-Muon Alg 1 ≡ simplified LoRA-RITE [LoRA-Muon Prop 6] ≡ iMuon
Cor 4.1). This is the **family comparison** (figure label "LoRA-Muon step", not a pin-hybrid).
The ablation is the incremental climb **iMuon-step → +pin → +curvature → ours**:
iMuon-step→(−curvature arm) attributes the pin; (−curvature arm)→ours attributes curvature —
so both controls are attributed WITHOUT a separate `−pin` cell. `−pin` (curvature on,
`cw_unpinned`) is cheap anchor-only insurance for the interaction term; decide at writing
time whether it earns a table row. The old `cw_no_radius` (−adaptive-radius) arm is
**RETIRED** — it kept the pin, so it was never the iMuon step; the flag stays dormant in code,
redirect-commented to `cw_unpinned`.

**Anchor cells.** PRIMARY hero = **Llama-3.2-1B openmath r256** (famous model; carries the
hero figure, the 5-arm pillar panel — AdamW / iMuon / per-factor polar / LoRA-Muon-step /
Polar-LoRA — and the ablation basins). The rank ladder (r64/128/256) shows the method **wins
across r**. SECONDARY hero = **OLMo-2-1B opc r256** (replication cell: iMuon, per-factor polar
already run there). Hard-cell test = Qwen opc r256 (ablation pair; smallest 1B speedup, 1.30×).

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

- **C1 (operator-norm magnitude control).** Port Spectron's per-factor operator-norm
  radius `ρ = η/(σ_max(A)+σ_max(B))` to LoRA *fine-tuning*: it bounds the merged step and is
  stable from the standard **B=0** init, where the unpinned partner-Gram family core is not
  (Rem b0 / E0). iMuon deliberately removes this radius (its Gram root bounds the *product*
  step but leaves the raw factor step ∝ 1/σ_min(partner) uncontrolled; headline runs
  momentum-free at r≤16). The rank ladder shows the method **wins at each r** (r64/128/256) —
  per-rank speedup-vs-AdamW. **lr-TRANSFER is CUT (2026-06-12):** do NOT reintroduce
  "transfers across rank" / "rank-invariant lr" / r^{−1/2} / muA-as-prediction.
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
- **C4 (step → walltime; task/rank coverage).** Report walltime speedup, not just step
  speedup; the method wins across ranks (r64/128/256), and the speedup varies with task
  OOD-ness. Do NOT headline "speedup grows with rank" (single-seed, fragile).

## Positioning table (intro + related work spine)

| method | curvature | nonlinearity | per-factor lr pin | input | regime |
|---|---|---|---|---|---|
| LoRA-RITE (2410.20625, ICLR'25) | partner basis (unmagnified grads) + transported 1-sided matrix 2nd moment | none (polar only in memoryless limit) | $(R_B^\top)^\dagger$ magnitude adjust | 1st+2nd moments | LoRA r=16 |
| LoRA-Muon (2606.12921, concurrent) | partner Gram (1 side) | polar | none (half-split budget) | EMA momentum | from-scratch toy (TinyShakespeare, d=128) |
| iMuon (2605.09238) | partner Gram (1 side) | polar | none (removed ρ) | momentum-free | LoRA r≤16 |
| Spectron (2602.12429) | none | polar | ρ=η/(‖A‖+‖B‖+1) | momentum | native pretraining |
| Tilde CM (compositional SD) | partner Gram (attn products) | polar | isotropic eff-lr | momentum | pretraining |
| AdaPreLoRA (2605.08734) | two-sided diagonal Kron | none | scalar | momentum | LoRA r≤64 |
| muA (2602.06204) | — (theory) | — | scalar (SignSGD) | — | LoRA lr-transfer theory |
| chord-tight (ours, earlier variant — not a paper baseline) | partner Gram + Adam | polar | ρ | Adam | LoRA |
| **Polar-LoRA (ours)** | partner Gram + large-axis diag | polar | ρ | plain momentum | LoRA, all r |

Citation discipline (from the related-work agents):
- **Spectron** = the operator-norm radius mechanism (their Eq. 16 == our ρ). Credit explicitly.
- **muA** = rank-aware lr theory; cite as related work only — we make NO lr-transfer claim, so no muA-as-prediction.
- **iMuon** = partner-Gram-whitened polar on the fixed-rank LoRA manifold; credit it for
  removing runtime rescale, GL(r)-invariance, and the factor-condition-independent rate.
- **Tilde CM** = NOT a repackaging of iMuon — same atomic operator, different composition
  (attention products), no rank axis, no theory. Cite both as partner-Gram-whitened
  spectral updates, distinguish by composition.
- **AdaPreLoRA** = prior art for two-sided-on-momentum LoRA; differentiate sharply (polar +
  factor-space cost), do NOT cite as soft motivation.
- **LoRA-RITE (2410.20625, ICLR'25 — published, NOT concurrent)** = the EARLIEST
  partner-whitened invariant LoRA update (Oct 2024, predates iMuon by ${\sim}1.5$y) and the
  originator of transformation/gauge invariance (their Def 1; Thm 1: invariance ⟹ efficient
  feature learning). Full method = one-sided matrix-Adam in the invariant gauge (transported
  2nd moments + escaped mass); its memoryless core ≡ our iMuon-step arm (the double; lineage
  RITE → iMuon → LoRA-Muon). Real-scale wins (Gemma 2B/7B, r=16, beats Adam/LoRA+/Shampoo/Lamb),
  but no rank axis, no radius, no explicit polar cap. NOT run by us — deferred baseline
  (Limitations item 5). $B{=}0$-safe via pseudo-inverse ($R_B^\dagger{=}0$ ⟹ $\delta A{=}0$).
- **LoRA-Muon (2606.12921, concurrent, posted 2026-06-11)** = decoupled partner-Gram polar
  + split weight decay ($s=\sqrt{1-\lambda\eta}$; algebraic only, never ablated; no-op at our
  $\lambda{=}0$) + gauge-invariance analysis. TinyShakespeare-scale from-scratch, both factors
  random-init — never faces $B{=}0$. Does NOT cite iMuon or Gram-NS/Dao; msign = plain
  PolarExpress (not Gram form), inverse-sqrt = separate NS table. Their Prop 5 (Spectron radius
  fails scalar gauge invariance) transfers to our linearized ρ — pre-empted in tension 6.
- **LoRA-α (2606.12883, concurrent)** = $\alpha^* \approx 256\sqrt{r}$ under AdamW —
  parameterization axis, no optimizer work. Effective adapter scale
  $\alpha/r \propto r^{-1/2}$ corroborates the muA family; cite in one clause alongside muA.
  Optional non-blocking insurance probe: AdamW @ OLMo opc r256 with $\alpha{=}4096$,
  $\eta{=}2{\times}10^{-5}$ vs the tuned baseline — first-order arithmetic predicts parity
  ($(\alpha/r)\,\eta = 3.2{\times}10^{-4} \approx$ tuned $\eta$).

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
| **Rank ladder** (math, popular model) | **Llama-3.2-1B openmath r64 / r128 / r256** | wins across r (C1) + rank coverage (C4) | all 3 to run |

- **r=256 is the canonical rank**; the Llama-math ladder is the only place rank varies.
- Llama carries the across-rank coverage (popular model, has r64/r256 registry cells —
  Qwen has no r64 cell so can't ladder cheaply); Qwen carries the OOD headline; OLMo is the
  cheap ablation workhorse.
- **Optional free bonus:** OLMo openmath r256 (in hand) as a 2nd math data point.

### Baselines

- **AdamW** (universal reference + speedup denominator) — at **all** cells; the competitive default.
- **iMuon** (the spectral rival) — **2 demonstration cells** (OLMo opc r256 + the primary
  anchor Llama openmath r256; the bengali OOD demonstration was dropped 2026-06-12 in favor
  of the famous-model anchor), **lr-tuned** via 1000-step pilot then 3-lr grid. NOT at all
  cells, NOT the full ladder. Rationale: iMuon is slow (measured 3.464 s/step at OLMo r256,
  per-pair v5_warmup) and underperforms AdamW in LoRA finetuning — the right rival to
  *cite + beat*, not to run 8×. E0 implements it; see below.
- These two are the ONLY comparison baselines. The "expensive machinery is dead weight"
  claim is the **E2 leave-one-out ablation** (remove curvature flavor / Picard from the
  protagonist), not a separate expensive-optimizer overlay.

### Experiments in dependency order

- **E0 — iMuon baseline = the authors' vendored `v5_warmup` (done).** Run the authors'
  reference code (`lora_playground/third_party/imuon_muon.py`, imuon @4f1d4b1) directly via
  `imuon-lora`: 50 steps of the `full` variant (an undocumented bootstrap to grow B off zero)
  then `v5`. Tests pin the wiring (`tests/test_imuon_lora.py`, 3 pass); production smoke PASSED
  (stable, eval decreasing). Run at **2 demonstration cells only**, **lr-tuned** (short pilot
  then 3-lr grid) — show it loses to AdamW; NOT the ladder, NOT all cells. ~3.9 s/step;
  momentum=0.95 Nesterov; wd=0; ε=1e-6; ns=5; adjust_lr=False (scalar lr).
  - **v5 IS the paper's published with-momentum algorithm (numerically verified 2026-06-12).**
    Their "Momentum implementation" appendix forms the joint ambient direction
    M̂_t = G̃_B·A + B·G̃_A (Nesterov β=0.95) and applies the metric-scaled spectral LMO (their
    F.3 QR procedure); v5's projector form equals F.3-on-M̂_t to 1e-10 (numpy, exact polar). So
    v5 is NOT a bug and IS in the paper. Only the 50-step `full` warmup is undocumented (it is
    the orthogonalize-then-project form their own Appendix H.1 attributes to the rival
    Riemannion, shipped only for B=0 survival).
  - **KEY DISTINCTION:** with-momentum iMuon (v5, joint ambient momentum) is a DIFFERENT
    algorithm from iMuon's own momentum-free Corollary 4.1 (per-factor decoupled sandwich).
    They differ even at β=0 (rel diff ~0.59, verified) because the joint form carries a cross
    term. LoRA-Muon and simplified-LoRA-RITE correspond to the Cor 4.1 / per-factor lineage,
    NOT to v5. No iMuon code variant implements Cor 4.1 anywhere (v2/v3 are the closest
    per-factor forms but use the full Gram inverse, not the split inverse-sqrt sandwich).
  - **Unused code variants:** v2, v3, v4 are experimental alternates (not any paper algorithm);
    v5_compact is a buggy compact-QR rewrite of v5 (transpose error, rel err ~1.2 on dB,
    verified) — inert, never runs.
  - **Why not Cor 4.1:** iMuon's proven Cor 4.1 (= LoRA-Muon's per-factor form) is B=0-unstable
    from standard LoRA init (measured: param_l2 711→1485 in 15 steps, no viable lr); that is
    exactly why the authors ship the warmup. The form we RUN is their v5_warmup.
- **E1 — coverage fill.** Locked protagonist (7 cells) + iMuon (**2** demonstration cells,
  lr-tuned) + AdamW (cell 7 only) at an lr grid ({0.01, 0.03, 0.1} brackets the OLMo optimum —
  **verify best-η isn't at a grid edge per cell**, widen if it is). Tracked in
  `paper/e1_coverage_fill.md`. Gate for the headline performance profile.
- **E2 — ablation (leave-one-out + the family double)** on the **Llama-3.2-1B openmath
  anchor** (follows `paper/paper_plots.ipynb`). The 2×2 of {curvature, pin} forms the
  incremental climb **iMuon-step → +pin → +curvature → ours**:
  - **−curvature** (`cw_no_diag_curv`: `P,Q→I` ⟹ `C_A = BᵀB = S_B`) → partner-Gram-only
    whitening + polar + **pin kept**. Our momentum (Nesterov β1=0.9), PE-8 polar, δ-damping;
    the novelty-vs-iMuon-family arm. (Already running: `e2_no_diag_curv_*`.)
  - **double = −curvature −pin = the iMuon/LoRA-Muon step** (`cw_unpinned` + `cw_no_diag_curv`
    + `--lora_init_b symmetric`): bare partner-Gram, **no pin** (true-scale roots, no σ_max(W)
    rescale), decoupled sandwich = Cor 4.1 / LoRA-Muon Alg 1. Needs the symmetric (PiSSA-style,
    step-0 = pretrained) init because the unpinned core blows up at B=0 — that requirement is
    itself the stability evidence. The family comparison.
  - **−pin** (`cw_unpinned` + curvature on + `--lora_init_b symmetric`) → cheap insurance at
    the anchor only (the {curv on, pin off} cell; completes the 2×2). Decide at writing time
    whether it earns a table row.
  - The decomposition (iMuon-step → −curvature → ours) attributes both controls **without**
    the `−pin` cell; `−pin` only adds the interaction term. The old **`cw_no_radius`
    (−adaptive-radius)** arm is RETIRED (it kept the pin, so it was never the iMuon step; the
    flag stays dormant in code, redirect-commented to `cw_unpinned`).
  - **No −polar arm.** Polar's necessity is cited from the spectral-method literature, not
    ablated: the non-polar variant is known-weak and removing polar is not novel over iMuon.
  - **No ±Nesterov / PE8-vs-NS rows.** Nesterov is a standard Muon trick (`~/Muon`), justified
    by the step-matched Δ already recorded above; full polar (PE8) is the locked default.
  - The C3 "expensive is dead weight" figure (flavor / Picard refinement) is fed by **existing**
    logs, not these two arms — see C3.
  - **Control lr per arm** (curvature-ON-vs-OFF is confounded by an lr-basin shift —
    `soap_curvature_whitening.md:354`).
  - No separate "incremental ladder" experiment — its rungs (the iMuon step, the −curvature
    arm, the protagonist) are all already run for the baseline + LOO; if the LOO bars need a
    cumulative-climb narrative at writing time, re-plot the same arms (a matplotlib call, not
    an experiment).
- **E3 — speedup-across-rank figure.** Per-rank speedup-vs-AdamW across the Llama-math ladder
  (r64/r128/r256): the protagonist wins at every rank (the ladder shows the method **works
  across r**). NO lr-transfer / r^{−1/2} / rank-invariance overlay (CUT). Report the per-rank
  speedup; do NOT headline "grows with rank" (single-seed, fragile).
- **E4 — walltime.** Profile the protagonist's per-step (fwd/bwd/opt split) vs AdamW at the
  headline cells, **global batch ∈ {16 (comparison horizon), 64 (timing-only bench)}**.
  Publish walltime speedup = step-speedup ÷ per-step-ratio. **Never profiled the
  protagonist's wall — only chord-tight's** (`walltime_profile.md`); extend
  `scripts/bench/bench_optimizer_step.py`. Benchmark the PRODUCTION config (PE8, the exact
  flags above), not `build_optimizer` defaults.
- **E5 — task/rank coverage (C4).** Falls out of E1. OOD: the Qwen opc→bengali pair (same
  model+rank, only dataset). Rank: the Llama-math ladder shows the method **wins at each r**.
  **Robust claim = per-rank win + OOD task-dependence.** Do NOT headline "speedup grows with
  rank" (single-seed Llama-fresh, fragile); let the per-rank numbers stand.
- **E6 — pass@1 (minimal downstream).** **HumanEval pass@1** on the **4 code-@-r256 cells**,
  protagonist vs AdamW at the best-lr final adapter. Net-new infra: (a) build a HumanEval
  harness (base + LoRA adapter → generate → exec-test); (b) **retain the final adapter** on
  those 4 cells (override the auto-`rmtree` in `train.py`). Math pass@1 (GSM8K/MATH) →
  future work.

## Tensions to manage in the writing (pre-empt reviewers)

1. C1 vs iMuon: iMuon *chose* to remove ρ. Argue raw-factor-step blowup (1/σ_min) + B=0
   instability; the radius is magnitude control, NOT lr-transfer (cut).
2. C2 vs AdaPreLoRA: prior art for two-sided-on-momentum. Lead with polar + factor-space cost.
3. **lr-transfer CUT (2026-06-12)** — no "transfers across rank" / rank-invariant / r^{−1/2} /
   muA-as-prediction claim remains to manage.
4. Opposite-factor whitening *hurts* early-time at high rank (`chord_tight_whiten_lag_r256.md`)
   — don't claim more preconditioning is monotonically better.
5. KL-Shampoo (no polar) beats SOAP/Shampoo in full-weight pretraining
   (`soap_curvature_whitening.md` Reading B) — i.e. good curvature *might* make polar
   redundant. We do NOT run a −polar arm to refute this (non-polar is known-weak, and
   removing polar is not novel over iMuon); polar's necessity rests on the spectral-method
   literature (Muon/iMuon/Spectron). State this as a cited premise, not an empirical result.
6. **Radius gauge-sensitivity** (LoRA-Muon Prop 5, on Spectron's quadratic radius; transfers
   to our linearized ρ). Defense, stated as a deliberate trade: per-factor step magnitude is
   itself gauge-dependent, so invariance and per-factor magnitude control are mutually
   exclusive; our merged-step bound ρ(σ_max A + σ_max B) = η holds at every gauge (only the
   factor split varies). Their evidence is an adversarial ×99 rescale with no natural
   occurrence; ours is at the B=0 init fine-tuning actually uses, where the unpinned family
   update leaves the factor step uncapped (‖Ȧ‖₂ ≈ η/σ_min(B); undefined at exactly B=0,
   untested by them). Our stability is the pin: the σ_max(W) rescale makes ‖Ȧ‖₂ = η by
   construction at every conditioning, so the protagonist trains from standard B=0 — E0
   measures the unpinned core diverging there. Credit the invariance concept (RITE's Thm 1)
   to RITE, not LoRA-Muon (RITE itself is B=0-graceful via pseudo-inverse: R_B^†=0 ⟹ δA=0).

## Decisions (resolved 2026-06-09)

- **Protagonist:** PE8 full polar + KL-diag (coupled diagonal) + polar + radius, Nesterov
  momentum, k=1, β1=0.9, gram_ns inverse-sqrt.
- **Scope:** arXiv, single-seed (multi-seed held off — too expensive), σ_AdamW as Δ unit.
- **Canonical rank:** r=256; rank varies only on the Llama-math ladder.
- **iMuon:** run as the always-present spectral baseline (E0). **Spectron:** argument-only.
- **Ablation home (E2):** the Llama-3.2-1B openmath anchor (−curvature at r256; the double =
  iMuon step via `cw_unpinned` + `--lora_init_b symmetric`); leave-one-out + the family double,
  not a peel stack.
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
4. **Rank coverage, not rank-growth.** The ladder shows the method wins at each r
   (r64/128/256); the robust claim is the per-rank win, NOT "speedup grows with rank"
   (single-seed, fragile).
5. **Deferred baseline: full LoRA-RITE** (the published rival a conference reviewer is most
   likely to demand). Preprint coverage: our iMuon-step arm = its memoryless core
   (LoRA-Muon Prop 6); the uncovered residual is RITE's transported one-sided second
   moments. Conference upgrade (with seeds): vendor `~/LoRA-RITE/lora_rite.py` (official
   PyTorch impl, already local), iMuon protocol — 2 demonstration cells, lr-tuned.
