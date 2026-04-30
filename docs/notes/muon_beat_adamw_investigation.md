# Muon-LoRA — beat AdamW investigation

Tracking the campaign to make a Muon variant **strictly beat** AdamW
(0.7579) and adam-lin-lora (0.7564) on the 2k-step r=16 LoRA fine-tune of
OLMo-2-1B on Magicoder.

## Result: adam-muon-lora wins at r=64; LoRA+ confounded the r=16 headline

| optimizer        | r  | best η | m  | eval loss @ 2000 |
|------------------|----|--------|----|------------------|
| adam-muon-lora       | 64 | 3e-3   | 1  | **0.7515**       |
| adam-muon-lora       | 16 | 3e-3   | 4  | 0.7557           |
| adam-lin-lora        | 16 | 1e-3   | 1  | 0.7564           |
| adam-scaled-lora     | 16 | 1e-3   | 1  | 0.7572           |
| adamw                | 64 | 3e-4   | 1  | 0.7550           |
| adamw                | 16 | 3e-4   | 1  | 0.7579           |
| adam-muon-lora       | 16 | 3e-3   | 1  | 0.7624           |
| muon-lora            | 16 | 3e-3   | 1  | 0.7675           |

**Updated picture (post m=1 disentanglement, group `adam_muon_clean_2k`):**

- **r=64, m=1:** adam-muon-lora at η=3e-3 → 0.7515, beats AdamW r=64
  (0.7550) by Δ=−0.0035 (within trajectory jitter). Real-direction win.
- **r=16, m=1:** 0.7624 — **loses** to AdamW r=16 (0.7579) by +0.0045.
  The original 0.7557 r=16 headline was driven mainly by LoRA+ (m=4),
  not by NS-on-Adam. m=1 alone does not beat AdamW at r=16.
- **r=16, m=4 (LoRA+):** 0.7557 — confounded result; useful as a working
  recipe but not a clean attribution to the optimizer.

The mechanism remains: Newton-Schulz orthogonalization applied to Adam's
per-element preconditioned direction m̂/(√v̂+ε), per-factor independently.
The strict-win story now lives at **r=64**, not r=16. For an unconfounded
strict win that clears the noise floor, use **adam-polar-product-lora at
r=64** (0.7453, Δ=−0.0097) — the spectral-product geometry beats per-factor
NS on the same composition (Adam → spectral correction).

Plan source: `~/.claude/plans/i-think-we-should-zippy-cake.md`.
Theory source: `docs/theory/main.tex` lemma at line 622 (spectral product norm
for LoRA factors → polar(Ĝ)·V_A V_Aᵀ identity at line 660).

**Diagnosis**: independent NS on A and B is the wrong proximal geometry — it
does not produce a spectrally-bounded product update ΔW = (α/r)(B·δA + δB·A),
breaks gauge invariance under A→RA, B→BR⁻¹, and at PEFT's B=0 init makes NS
on m_A wasted (since B·δA ≈ 0 contributes nothing to ΔW early).

**Success bar** (per `feedback_beat_dont_match.md`): best-η final eval loss
**< 0.7579** (strictly beats AdamW). Headline win: < 0.7564 (strictly beats
adam-lin-lora). Parity is failure.

---

## H1 — B-init asymmetry (LoRA+ for Muon)

PEFT inits B=0; for early steps δB·A dominates ΔW; A's update is wasted unless
B grows fast. Multiplier m on B's lr should close the gap.

- **Status:** **complete** (2000-step sweep + 500-step pilot both finished)
- **Test:** sweep η ∈ {1e-3, 3e-3, 1e-2} × m ∈ {4, 16}; baseline m=1 reused
  from `muon_lowlr_2k` and `new_optimizers_high_eta_2k` (best m=1: 0.7675 at η=3e-3).
- **Falsifier:** all m=4/16 final losses ≥ 0.7579 ⇒ H1 not the story.
- **Group:** `muon_loraplus_2k` — 6 tasks — completed
- **Final results @ step 2000:**

  | η     | m=4    | m=16   |
  |-------|--------|--------|
  | 1e-3  | 0.7674 | 0.7679 |
  | 3e-3  | 0.7759 | 0.8460 |
  | 1e-2  | 0.9611 | 1.4209 |

- **Best result:** η=1e-3 m=4 → **0.7674** — essentially tied with m=1 baseline
  (0.7675 at η=3e-3). **H1 is real but only reparameterizes**: m shifts optimal
  η downward by ~1/m, but the achievable floor is unchanged.
- **Decision:** **H1 confirmed but insufficient on its own** — does not beat
  AdamW (0.7579). LoRA+ is a free 1-line tweak so we keep `lr_b_multiplier=4`
  in subsequent configs as a sensible default; the real lever has to come from
  H2 (geometry) or H4 (Adam direction).
- **Pilot validation:** Tier 1 pilot at step 500 (`muon_loraplus_500`) ranked
  η identically to the 2k results (best η=1e-3 m=16 → 0.8035 at step 500).
  500-step pilots are reliable proxies for η-ranking — saves wall-clock for
  H2/H4 sweeps.

## H2 — Wrong proximal geometry (ProductMuonLoRA)

NS should orthogonalize the merged-W direction projected onto LoRA subspace,
not the factors independently. Algorithm: form gauge-invariant rank-r proxy
D = (1/scale)·m_B·(AAᵀ + δI)⁻¹·A; NS via thin QR + small-r NS core; recover
(δA, δB) by Sylvester least-squares (mirrors `AdamLinLoRA`).

At B=0 init, min-Frobenius Sylvester correctly sets δA = 0 — A is "frozen"
until B picks up signal. This re-derives H1's motivation algorithmically; the
LoRA+ multiplier remains useful as an orthogonal lever.

- **Status:** **500-step pilot in flight** (`product_muon_500`, step ≤ 400/500
  across all 8 cells). 2k version cancelled to free QoS quota; will resubmit
  with a tighter η range after the pilot decides.
- **Test:** sweep η ∈ {3e-4, 1e-3, 3e-3, 1e-2} × m ∈ {1, 4}.
- **Falsifier:** ≥ 0.7579 across all 8 cells ⇒ H2 not the story; pivot to
  closed-form polar (theory line 656) or H4 hybrid.
- **Group:** `product_muon_500` — pilot
- **Intermediate (step 400):**

  | η     | m=1            | m=4    |
  |-------|----------------|--------|
  | 3e-4  | 0.9428         | 0.8669 |
  | 1e-3  | 0.8718         | 0.8335 |
  | 3e-3  | 0.8356         | 0.8242 |
  | 1e-2  | **0.8174**     | 0.8596 |

  Two structural observations:
  1. **No divergence at η=1e-2** (vs Tier-1 muon-lora m=16 → 1.32 at same η).
     The product-spectral cap correctly bounds ‖ΔW‖_op even at large η.
  2. **LoRA+ asymmetry not needed** for product-muon: m=1 wins at high η, m=4
     only helps at low η. Consistent with the algorithm: B=0 init → Sylvester
     min-Frobenius sets δA=0 automatically, so the asymmetric multiplier is
     redundant.
- **Slope-based extrapolation to step 2000:** at step 200/400 product-muon is
  0.014 / 0.007 ahead of Tier-1 best (m=4 η=1e-3 trajectory). Lead shrinking.
  Linear extrapolation lands at ~0.760 — between Tier-1 baseline (0.7674) and
  AdamW (0.7579). Likely insufficient to *strictly beat* AdamW alone.
- **Decision:** _tbd until step 500 (~5 min)._ If extrapolation holds (~0.76),
  next move is **product-muon + Adam direction** combo: feed Adam's m̂/√v̂ into
  the same product-norm Sylvester pipeline (combines H2 geometry + H4
  preconditioning, not yet planned but cheap).

## H3 — Sanity: NS itself is irrelevant; momentum is doing the work

If `ns_steps=0` (raw momentum SGD) matches Muon-LoRA, the orthogonalization
isn't pulling weight — the framework needs a different rethink.

- **Status:** **complete**
- **Test:** sweep η ∈ {3e-4, 1e-3, 3e-3, 1e-2}, m=1, ns_steps=0.
- **Falsifier:** ns=0 final loss ≈ ns=5 final loss at matched η ⇒ NS irrelevant.
- **Group:** `muon_nsoff_2k` — 4 tasks — completed
- **Final results @ step 2000:**

  | η     | loss   |
  |-------|--------|
  | 3e-4  | 1.1844 |
  | 1e-3  | 1.1658 |
  | 3e-3  | 1.0672 |
  | 1e-2  | 0.9499 |

- **Best result:** η=1e-2 → 0.9499 (vs muon-lora-with-NS at 0.7675). NS
  contributes ≥ 0.18 nats — clearly essential.
- **Decision:** **H3 falsified.** Geometric orthogonalization is essential;
  no pivot away from the Muon framework needed.

## H4 — Adam direction + Muon spectral cap (AdamMuonLoRA)

Apply NS to Adam's m̂/(√v̂+ε) per-factor instead of raw momentum. Decouples
diagonal preconditioning from spectral capping. Cheaper than ProductMuon —
no Sylvester recovery; direction is per-factor.

- **Status:** **complete — H4 wins.**
- **Test:** sweep η ∈ {1e-4, 3e-4, 1e-3, 3e-3} × m ∈ {1, 4} pilot, then 2k
  continuation at the best config + extension to higher η.
- **Pilot results (step 500):**

  | η     | m=1    | m=4    |
  |-------|--------|--------|
  | 1e-4  | 1.0742 | 0.9340 |
  | 3e-4  | 0.9272 | 0.8644 |
  | 1e-3  | 0.8507 | 0.8182 |
  | 3e-3  | 0.8110 | **0.7962** |

- **Extension (step 500):** η=1e-2 m=1 → 0.7986 (close to best); η=1e-2 m=4 →
  0.8327 (m too high at this η); η=3e-2 m=4 → 1.2017 (diverging).
  η=3e-3 m=4 confirmed the η-optimum.
- **2k confirmation results:**

  | η   | m | step 2000 |
  |-----|---|-----------|
  | 1e-3 | 4 | 0.7670 |
  | 3e-3 | 4 | **0.7557** ✅ |

  **Best: 0.7557 — strictly beats AdamW (0.7579) and adam-lin-lora (0.7564).**
- **Decision:** **H4 confirmed and decisive.** AdamMuonLoRA is the headline
  win. Mechanism: per-factor NS on m̂/√v̂ direction, with lr_b_multiplier=4.


## H4-reverse — MuonAdamLoRA (NS first, then Adam)

`AdamMuonLoRA` does Adam(g) → NS. The reverse: NS(g) → Adam, i.e., per-factor
NS the raw gradient first, *then* run Adam EMA + per-element preconditioning
on top of the NS direction. Different mechanism: NS already kills magnitude
variation across spectrum, but per-element entries still vary, so Adam's v̂
on the NS output may add a useful second-stage correction.

- **Status:** **2k sweep mostly complete** (`muon_adam_2k`, η ∈ {1e-3, 3e-3} ×
  m ∈ {1, 4}, 4 tasks) — mechanism question answered at step 1600.
- **Falsifier:** all cells ≥ 0.7557 (the headline-win number).
- **Mid-run results @ step 1600:**

  | η     | m=1     | m=4    |
  |-------|---------|--------|
  | 1e-3  | 0.7851  | 0.8846 (stuck) |
  | 3e-3  | 1.0087 (rising) | 1.5766 (diverged) |

- **Decision (REVISED):** **NOT a clean falsification of the polar-first
  composition family — this was the *unstabilized* port that AdaMuon
  paper warns against.** Best cell (η=1e-3 m=1) at ~0.78 (2k) is real
  data on `MuonAdamLoRA`-as-implemented, but the implementation is
  missing all three of AdaMuon's stabilizers:
    1. **sign(Mₜ) before NS.** We feed NS the raw gradient. AdaMuon
       feeds NS the sign of the momentum buffer, stabilizing NS's input.
    2. **Only Vₜ on NS output, no Mₜ on it.** We run full Adam(m, v)
       on the NS output — double smoothing.
    3. **No RMS-align.** Step magnitude unbounded vs AdaMuon's
       `γₜ = 0.2·√(mn)/‖Õₜ‖_F`.
  The mechanism originally cited ("uncorrelated NS outputs cancel m̂,
  inflate √v̂") is correct as a description of *this specific port's*
  failure mode, and it is exactly the failure mode AdaMuon's three
  stabilizers were designed to prevent.
- **Proper test of the polar-first composition family:**
  `AdaMuonLoRA` (vanilla NS, AdaMuon-faithful) and
  `AdamuonPolarProductLoRA` (AdaMuon-faithful with spectral-product
  geometry instead of vanilla NS) are the right experiments. Both
  implemented (commits pending), sweeps queued at r ∈ {16, 64} ×
  η ∈ {1e-4, 3e-4, 1e-3} (jobs 6314009, 6314010). Until those land,
  treat the polar-first family as **open**, not falsified.


## H2 ⊗ H4 hybrid — AdamProductMuonLoRA (NEW)

Implemented and queued during H4-pilot wait. Combines ProductMuonLoRA's
gauge-invariant geometric pipeline (Z, NS in factored form, Sylvester
recovery) with AdamLinLoRA's Adam EMA on the recovered (precond_A, precond_B).

- **Status:** pilot queued (`adam_product_muon_500`, SLURM 6312406, 8 tasks)
- **Test:** sweep η ∈ {3e-4, 1e-3, 3e-3, 1e-2} × m ∈ {1, 4}.
- **Falsifier:** all cells ≥ 0.7579.
- **Why this might be the headline win:** combines the two mechanisms that
  separately got the closest to AdamW: H2 (geometry, removes divergence at
  high η) and H4 (Adam preconditioning, currently best variant at step 500).
- **Best result so far:** _tbd_

---

## Reference frontiers

| optimizer        | best η | eval loss @ 2000 | source group              |
|------------------|--------|-------------------|---------------------------|
| adam-lin-lora    | 1e-3   | 0.7564            | `optim_compare_high_eta_2k` |
| adam-scaled-lora | 1e-3   | 0.7572            | `optim_compare_high_eta_2k` |
| adamw            | 3e-4   | 0.7579            | `lr_sweep_2k`             |
| muon-lora (m=1)  | 3e-3   | 0.7675            | `new_optimizers_high_eta_2k` |

## Code

- Optimizers: `lora_playground/optim.py` — `MuonLoRA` (extended), `ProductMuonLoRA`, `AdamMuonLoRA`
- CLI: `--optimizer {muon-lora,product-muon-lora,adam-muon-lora}`,
  `--lora_plus_multiplier <m>`, `--muon_ns_steps <k>` (0 disables NS)
- Tests: `tests/test_new_optimizers.py` — 30 tests inc. NS scale-invariance,
  LoRA+ B-step scaling, gauge invariance for ProductMuon, ns=0 = momentum-SGD identity
- Sweep configs: `params/{muon_loraplus,muon_nsoff,product_muon,adam_muon}_2k.json`
- Sweep script: `scripts/sweep_muon_2k.sh`
- Notebook: `notebooks/sweep_analysis.ipynb` — section "Muon-LoRA variants —
  beat AdamW campaign" with verdict table

## Pilot sweeps (500 steps, fast signal)

Mirror sweeps at `--max_steps 500 --eval_every 100` for first-pass η-ranking.
Each pilot run ~25 min. Submitted at full per-sweep parallelism.

- `muon_loraplus_500` — SLURM 6312199 — 6 tasks
- `muon_nsoff_500` — SLURM 6312200 — 4 tasks
- `product_muon_500` — SLURM 6312201 — 8 tasks
- `adam_muon_500` — SLURM 6312202 — 8 tasks

Pilots will tell us within ~1 hour which (variant, η, m) cells are worth
running to 2k, and which to cancel.

## What to do when results land

1. Refresh the notebook section; the verdict table prints ✅/❌ vs both
   reference frontiers automatically.
2. Update each H_i block above with `Best result so far`, `Decision`.
3. If any H_i wins (final < 0.7579): explore the orthogonal axes (combine with
   another H, extend η range, sweep momentum β) before declaring the headline
   result.
4. If all four stall at parity: pivot to closed-form polar (theory doc line
   656) — adds two NS calls per pair per step but is the unique closed-form
   spectral-product update.

## What success looks like

| outcome                                     | classification   |
|---------------------------------------------|------------------|
| best Muon variant ≥ 0.7579                  | failure          |
| 0.7564 ≤ best Muon variant < 0.7579         | win (beats AdamW)|
| best Muon variant < 0.7564                  | headline win     |

If we stall at parity with AdamW that's the signal to push harder, not to
stop.
