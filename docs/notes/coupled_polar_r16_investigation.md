# adam-polar-product-lora: why coupled hurts at r=16

Working doc. Investigation of why `adam-polar-product-lora-coupled` (Picard
iter-2) underperforms the uncoupled (iter-1) variant specifically at r=16,
while winning at r=64/128/256.

Status: **mechanism characterized but not yet causally pinned down**.
H4′ (bilinear feedback in B) is the only mechanism showing differential
signal between coupled and uncoupled at r=16. α-sweep + picard_iters sweep
launched / pending to map the response surface.

## The gap to explain

Final eval losses at η=3e-4, 2k steps, single seed (pulled via
`load_runs(where=…)` from `logs/`):

| r   | uncoupled | coupled  | Δ (coupled − uncoupled) |
|-----|-----------|----------|-------------------------|
| 16  | 0.7546    | 0.7616   | **+0.0070** (worse)     |
| 64  | 0.7454    | 0.7382   | −0.0072                 |
| 128 | 0.7458    | 0.7354   | −0.0104                 |
| 256 | 0.7410    | 0.7396   | −0.0014                 |

At every r except 16, the cross-coupled iter-2 update beats block-diagonal
iter-1. At r=16 it loses by a comparable magnitude. The question is what
makes r=16 special.

## What "coupled" actually does

`adam-polar-product-lora` solves the joint normal equations under the
spectral-product metric via Picard iteration. The equations:

    S_B · ΔA + Bᵀ · ΔB · A = −η · u_A
    ΔB · S_A + B  · ΔA · Aᵀ = −η · u_B

`picard_iters=1` (uncoupled) solves the block-diagonal version: drop the
cross-terms `Bᵀ·ΔB·A` and `B·ΔA·Aᵀ`. `picard_iters=2` (coupled) does one
Picard fixed-point step: feed the iter-1 (ΔA, ΔB) back as `dA_prev`,
`dB_prev` and recompute with the cross-terms in:

    u_A_eff = u_A + α · (Bᵀ · dB_prev · A) / η
    u_B_eff = u_B + α · (B  · dA_prev · Aᵀ) / η

Then through the polar pipeline (Newton-Schulz orthogonalization, RMS-align).
α=1 in the original implementation; α was added to the optimizer for the
sweeps below.

## H1–H4 — what we instrumented and what we found

For probing whether the coupled iter-2 update has any of several catastrophic
mechanisms, the optimizer was instrumented with:

- **γ_A, γ_B** — relative magnitude of the cross-coupling correction:
  γ_A = ‖Bᵀ·dB·A/η‖_F / ‖u_A‖_F, similarly for γ_B. Always-on.
- **nrank_τ(S), stable rank** — soft and hard rank measures of S_A = AAᵀ
  and S_B = BᵀB. Always-on.
- **picard_contract_A_12, _A_23** — successive-iter Picard increment
  ratios: ‖dA² − dA¹‖/‖dA¹‖ and ‖dA³ − dA²‖/‖dA²‖. Probe-step only
  (every `diagnostics_every` steps).
- **polar_cos_A_12, _B_12** — cos between iter-1 and iter-2 polar (NS)
  outputs. Probe-step only.

Code: `lora_playground/optim.py` `AdamPolarProductLoRA.step` (commit
`6ae1ec7`). Per-pair stats; `_emit_optim_diagnostics` aggregates min /
median / max across pairs into one `optim_step` JSONL event.

**Hypotheses:**

- **H1 — cross-term dominance.** γ ≳ 1 and r-dependent (smaller r → larger
  γ because B's top σ are bigger). Predicts: γ explosion at r=16.
- **H2 — Picard non-contracting.** picard_contract_*_23 not ≪ _12;
  successive iterates don't shrink.
- **H3 — polar amplifies perturbations.** polar_cos_*_12 noticeably below
  1 (Newton-Schulz non-Lipschitz at degenerate spectra).
- **H4′ — bilinear feedback into B at small r.** stable_rank_B/r drops
  in coupled relative to uncoupled, more dramatically at r=16.

### What the data says (2×2 sweep at η=3e-4, 2k steps)

Group: `diag_h1234_2x2`. Configs: {coupled, uncoupled} × {r=16, r=128}.
Diagnostics every 20 steps. Final-step trajectories:

**stable_rank_B/r** (out of r):

| step | uncoupled r=16 | coupled r=16 | uncoupled r=128 | coupled r=128 |
|------|---------------|---------------|------------------|----------------|
| 200  | 10.64         | 10.40         | 30.99            | 30.61          |
| 500  | 11.04         | 10.36         | 33.83            | 32.54          |
| 1000 | 11.24         |  9.96         | 35.18            | 32.69          |
| 1500 | 11.20         |  9.50         | 36.46            | 32.82          |
| 1800 | 11.21         |  9.26         | 36.40            | 33.14          |

- r=16: uncoupled drifts up to 11.2/16 (≈70% of available rank);
  coupled drifts down to 9.3/16 (≈58%). Gap = 1.95 (12% of available).
  The two trajectories DIVERGE over training.
- r=128: uncoupled rises to 36.4/128 (≈28%); coupled rises to 33.1/128
  (≈26%). Gap = 3.3 (2.5% of available). Same direction (coupled has
  lower stable rank than uncoupled), but the relative magnitude is much
  smaller.

**γ trajectories:** γ stays small (max ≈ 0.12 in any cell), and γ at r=128
≥ γ at r=16 in every cell — opposite of what H1 predicted. **H1 refuted.**

**picard_contract:** _A_23 / _A_12 ratio ≈ 0.1–0.5 throughout, _A_23
substantially smaller than _A_12. Picard contracts cleanly. **H2 refuted.**

**polar_cos:** ≥ 0.992 everywhere; Newton-Schulz behaves as identity
between iter-1 and iter-2. **H3 refuted.**

**Eval losses at step 1800** (current sweep, reproduces prior data):

| cell           | step 1800 |
|----------------|-----------|
| uncoupled r=16  | 0.7577    |
| coupled r=16    | 0.7643    |
| uncoupled r=128 | 0.7496    |
| coupled r=128   | 0.7388    |

## Mechanism diagnosis (claim, not proof)

The iter-2 update to `dB` is:

    dB ∝ polar_pipeline((u_B + α·B·dA·Aᵀ/η) · S_A^{-1/2}) · S_A^{-1/2}

The cross-term `B · dA · Aᵀ / η` has its column space **strictly inside
col(B)** — B appears as a left factor. The polar (Newton-Schulz)
operator preserves column space; right-multiplication by S_A^{-1/2} acts
on rows. So the iter-2 contribution to dB lives preferentially within
col(B); the iter-1 contribution does not (its column space comes from
u_B, which depends on the loss gradient, not on B's structure).

Repeated application reinforces col(B): each step preferentially
refreshes already-occupied directions of B. At small r, where B's column
space is already near-saturated (sr_B/r ≈ 0.7 at r=16), this concentrates
the limited rank into fewer effective directions. At large r, where
sr_B/r ≈ 0.25, there's spare column space for the reinforcement to
absorb without harm.

This is **inherent to (a) the LoRA factorization ΔW = BA (so ∂ΔW/∂ΔA = B
brings B into the chain rule for any joint update) and (b) joint coupling
on (A, B) variables**. Within LoRA + joint coupling, the bilinear cross-
term is unavoidable; the polar / metric / Picard machinery doesn't change
its column-space property because polar preserves column space and
right-side whitening doesn't act on the left factor.

The mechanism is correlative-supported by the data: coupled-r=16 sr_B
diverges from uncoupled-r=16 sr_B in the direction predicted, while
the divergence at r=128 is small. But the data does NOT establish
causation: that the rank concentration *causes* the loss gap, vs. being
a side effect of some other axis the optimizers differ on.

## Dispersion of S_B predicts the sign of Δ at every existing cell

Pulled all (r, η=3e-4) cells where both coupled and uncoupled were
recorded. Computed κ_B_median = median across pairs of σ_max(S_B) /
σ_min(S_B), late-window (step 1000–2000) of the uncoupled run.

| r   | Δ (cou − unc) | κ_B median (uncoupled, late) |
|-----|---------------|------------------------------|
| 16  | +0.0069 (lose) | 2.45                        |
| 64  | −0.0077 (win)  | 10.96                       |
| 128 | −0.0100 (win)  | 35.85                       |
| 256 | −0.0107 (win)  | 159.87                      |

κ(S_B) is monotone in r and the sign of Δ flips between r=16 (κ ≈ 2.5,
near-flat spectrum) and r=64 (κ ≈ 11). At r=16, the polar pipeline's
S_B^{-1/2} factor is close to a scalar — it doesn't reshape directions
meaningfully; coupled's iter-2 refinement on top has nothing to refine
and the cross-term destabilizes. At r ≥ 64, S_B is peaky enough for the
preconditioner to do meaningful directional work; coupled refines
usefully.

Mechanism details (whether the failure mode at low κ is best described
as "polar near-identity → cross-term is perturbation on Adam direction"
or as "bilinear cross-term concentrates col(B)") may be different
framings of the same phenomenon. Both predict κ-as-predictor.

The α-sweep currently in flight will say whether the loss-vs-α curve
is monotone-in-α at low-κ cells (consistent with both hypotheses) or
has interior structure (would discriminate). Either way, κ(S_B)
appears to be the right variable to characterize regimes.

## What we have NOT pinned down

- **Causation.** Correlation of sr_B drift with loss gap is consistent
  with H4′ but doesn't rule out shared upstream causes.
- **Whether α* (optimal damping on the cross-term) is 0, 1, or interior.**
  Without that, "the cross-term hurts" is ambiguous: maybe a smaller-but-
  nonzero α is best.
- **Whether α* depends on r.** If it does, state-dependent damping is
  justified (in a research-direction sense, not as a deployable HP).
- **Whether α and picard_iters trace the same axis** or are separable
  controls along different paths into the joint-NE solution.

## Open experiments

Two sweeps queued (params + sweep scripts in this branch):

1. **α-sweep at r=16 and r=128, picard_iters=2.** α ∈ {0, 0.25, 0.5,
   0.75, 1.0}. 10 runs. Maps the loss-vs-cross-term-magnitude response.
2. **picard_iters sweep at r=16 and r=128, α=1.** picard_iters ∈ {1, 2,
   3, 4}. 8 runs. Probes whether converging more deeply to the joint-NE
   fixed point worsens (target is wrong) or improves (convergence is
   the issue).

Predicted outcomes if H4′ is correct:
- α-sweep at r=16 monotone-worse from α=0 to α=1; at r=128 monotone-
  better. The α* pattern is r-dependent.
- picard_iters at r=16 monotone-worse (more iters → more cross-term
  application → more rank concentration). At r=128 maybe non-monotone
  (cross-term refines direction while bilinear feedback is mild).

If both match the prediction, H4′ is causally implicated. If either
doesn't, the diagnosis is incomplete.

## Things I got wrong during this investigation (kept for future-self)

- **Cited "AdamW 0.7579 at η=3e-4" from CLAUDE.md as if it were AdamW's
  best at r=16**, then built a hypothesis ("polar barely beats AdamW at
  r=16 so coupling has nothing to refine") on top of it. AdamW's best at
  r=16 is η=1e-4, not η=3e-4. Polar's edge over AdamW peaks at r=64,
  not r=16; the "polar weak at small r" framing is wrong by inspection.
  Memory rule added: feedback_no_baselines_from_memory.md.
- **Claimed "rank collapse" from σ_min(S_B)** before checking whether the
  median pair was anywhere close to rank-collapsing. σ_min on the worst
  layer is not a rank statement; stable rank on the median pair is.
  Walked it back; switched all rank claims to stable rank.
- **Suggested "closed-form solve of the joint NE" as a fix** while
  simultaneously claiming the joint NE was the wrong target. These are
  inconsistent: the closed-form linear-system solve is what
  picard_iters → ∞ converges to (with polar/RMS aside, which makes them
  not literally identical objects), and "more iters hurt" extrapolates
  to "the closed-form fixed point would hurt more, not less." Walked back.
- **Suggested α-damping as a hacky HP and Riemannian-manifold
  reformulations as a "general-purpose mechanism-based fix"** before
  actually deriving anything. The Riemannian reformulation isn't a
  fix to adam-polar-product; it's a different optimizer. The state-
  dependent α(sr_B/r) was hand-waved without a principled derivation
  and required the α-sweep first to even justify. The honest current
  position is: there is no clean mechanism-derived fix within
  LoRA + joint coupling; the bilinear feedback is structural; α* and
  picard_iters* must be characterized empirically before any
  "principled" claim is made about how to set them.

## Logged groups and what they contain

- `diag_h1234_2x2` — 2×2 (coupled/uncoupled × r=16/128) at η=3e-4 with
  H1–H4 probes. Job 6316691.
- (queued) `alpha_sweep_2x2` — α ∈ {0, .25, .5, .75, 1} × r ∈ {16, 128}
  at η=3e-4, picard_iters=2.
- (queued) `picard_iters_sweep_2x2` — picard_iters ∈ {1, 2, 3, 4} × r ∈
  {16, 128} at η=3e-4, α=1.
