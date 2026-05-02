# Coupled-polar LoRA optimizer investigation — unified doc

This is the single coherent record of the investigation into closing the
eval-loss gap to hybrid Picard (`adam-polar-product-lora-coupled`).
Supersedes `phase2_autonomous_progress.md`, `phase2_results_summary.md`,
and `factor_adam_progress.md`.

**Status: Picard still wins by ~0.011 at both ranks despite nine variants
of our principled coupled-polar-core solver. Theory says we should be
beating it. We are not. This doc accounts for what we tried, what the
diagnostics say, and the open hypotheses for what's wrong.**

All numbers single-seed, m=1, OLMo-2-0425-1B + Magicoder-OSS-Instruct,
canonical 2k-step horizon, seed 0.

---

## 1. The problem (math)

LoRA update: `W → W + (α/r) B A`. Frozen base, train factors A ∈ ℝ^(r,n),
B ∈ ℝ^(m,r). Per-step factor gradients

  G_A = (α/r) B^T ∇_W L           shape (r, n)
  G_B = (α/r) ∇_W L A^T           shape (m, r)

By construction these are compatible: B^T G_B = G_A A^T (both are
projections of the same dense gradient ∇_W L).

The joint operator-norm constrained step ("Case 3" of
`docs/notes/polar_coupled_problem.md`):

  min  ⟨G_A, ΔA⟩ + ⟨G_B, ΔB⟩
  s.t. ‖B ΔA + ΔB A‖_op ≤ λ

This is the LoRA-tangent analogue of Muon's spectral-norm-constrained
update, but coupled across the two factors.

---

## 2. Our solver — `polar-coupled-core-lora` (variant 1)

**File:** `lora_playground/optim.py` PolarCoupledCoreLoRA class (line ~3510),
core helper `_polar_coupled_core_step` (line ~3425).

**Math:** at each step, build the **active core** Ĥ in the bases of the
current factors and gradient-residual SVDs:

  B = Q_L R_L           (thin QR, Q_L ∈ ℝ^(m,r))
  A = R_R Q_R^T          (thin QR of A^T, Q_R ∈ ℝ^(n,r))
  L_⊥ = R_L^{-T} G_A − C Q_R^T,   E from SVD of L_⊥
  R_⊥ = G_B R_R^{-T} − Q_L C,     F from SVD of R_⊥
  C   = ½(R_L^{-T} G_A Q_R + Q_L^T G_B R_R^{-T})

  Ĥ = [[ C , E ],
       [ F , 0 ]]                        ((r+t) × (r+s))

Project quotient polar:

  P = polar(Ĥ)                           // SVD-based, U Σ V^T → U V^T
  R = P with R[r:, r:] := 0              // forbid (22) extension
  γ = ‖R‖_op,  Z+ = R / γ                // ½-approximation guarantee
  τ̂ = ‖Ĥ‖_*  / γ                        // squared-penalty scale
  Z_upd = -lr · τ̂ · Z+

Sylvester gauge lift back to factor space:

  S_L = R_L^T R_L + δI,  S_R = R_R R_R^T + δI
  Solve  S_L K + K S_R = R_L^T X R_R^T   (X = Z_upd[:r,:r])
  dA = solve_spd(S_L,  R_L^T X − K R_R) Q_R^T  +  R_L^T Y V^T
  dB = Q_L (X R_R^T − R_L K) solve_spd(S_R^T)  +  U W R_R^T

(See `_polar_coupled_core_lift` for exact form; X, Y, W are blocks of
Z_upd: X is (1,1), Y is (1,2), W is (2,1).)

**Key certificates (logged per step):**
- γ ∈ [1, 2]                   the ½-approximation factor
- compat = ‖C_L − C_R‖ / (‖C_L‖+‖C_R‖+ε)  gradient-compatibility violation
  — for raw gradients ≈ machine eps
- relgap = 1 − 1/γ ∈ [0, 0.5]  how far from rank-1 polar is
- ratio_dA_dB = ‖dA‖/‖dB‖
- imbalance_residual = ‖AA^T − ρ B^T B‖_F / (norm sum)  ρ = r/m

---

## 3. Picard — `adam-polar-product-lora-coupled`

**File:** `lora_playground/optim.py` AdamPolarProductLoRA (line ~1814),
default `picard_iters=2` for the coupled build.

**Math:** factor-Adam, then per-factor polar with spectral preconditioner,
optionally cross-coupled via Picard iteration:

  m_A, v_A = Adam EMA on G_A;  u_A = m̂_A / (√v̂_A + ε)
  m_B, v_B = Adam EMA on G_B;  u_B = m̂_B / (√v̂_B + ε)

  S_A^{-1/2} = (A A^T + δI)^{-1/2}      (m × r side)
  S_B^{-1/2} = (B^T B + δI)^{-1/2}      (n × r side)

  for k in range(picard_iters):
    if k == 0:    u_A_eff, u_B_eff = u_A, u_B
    else:         u_A_eff = u_A + α (B^T dB_prev A) / lr
                  u_B_eff = u_B + α (B dA_prev A^T) / lr
    X_B  = u_B_eff @ S_A^{-1/2}
    P_B  = polar(X_B)                                  // (m × r) polar
    geo_B = P_B @ S_A^{-1/2}
    X_A  = S_B^{-1/2} @ u_A_eff
    P_A  = polar(X_A)                                  // (r × n) polar
    geo_A = S_B^{-1/2} @ P_A
    dA   = -lr · (‖u_A‖/‖geo_A‖) · geo_A               // RMS-align
    dB   = -lr · (‖u_B‖/‖geo_B‖) · geo_B               // RMS-align
    dA_prev, dB_prev = dA, dB
  apply (dA, dB)

**Crucial differences:**

| aspect | our solver | Picard |
|---|---|---|
| polar | one joint, `(r+t)×(r+s)` core | two separate, `(m×r)` and `(r×n)` |
| basis extraction | thin QR of A,B + SVD of residuals | none — direct on factors |
| (22) extension | explicitly zeroed | not constrained |
| symmetrization | C = ½(C_L + C_R), can lose info | no symmetrization |
| step magnitude | lr · τ̂ (depends on ‖Ĥ‖_*) | lr · ‖u_A‖ via RMS-align (preserves Adam scale) |
| coupling | one-shot variational | Picard iterate (2 inner passes by default) |

Picard does NOT solve the joint operator-norm problem. It does
"factor-Adam → per-factor polar → RMS-align → Picard couple". It makes
no variational claim. Empirically it wins.

---

## 4. What we tried (full chronology)

| # | optimizer (commit) | mechanism | r=16 best | r=64 best | result |
|---|---|---|---|---|---|
| 0 | `adam-polar-product-lora-coupled` (Picard) | factor-Adam + per-factor polar + iterate | 0.7557 | 0.7382 | reference target |
| 0' | `adam-polar-product-lora` (uncoupled) | same minus Picard iteration | 0.7546 | — | r=16 BEST |
| 0'' | `adamw` | per-coord Adam | 0.7601 | 0.7550 | reference baseline |
| 1 | `polar-coupled-core-lora` vanilla | one-shot proj-quot-polar, raw factor grads | 0.8188 (lr=3e-3) | 0.7821 (lr=3e-3) | -0.063 / -0.044 |
| 2 | + state-rebalance (commit `c8482e7`) | post-step (B,A)→(BR, R⁻¹A) iLoRA invariant | 0.8104 | 0.7686 | small at r=64, none at r=16 |
| 3 | + wide-lr (no code, just lr) | extend lr scan to 1e-2, 3e-2 | 0.8049 | **0.7490** (lr=3e-2) | beats AdamW r=64; r=16 ceiling |
| 4 | `polar-coupled-core-sign-lora` (commit `1565976`) | + `Ĥ / (\|Ĥ\|+ε)` before polar | **0.7680** (lr=1e-4) | diverges | first to break r=16 ceiling |
| 5 | `muon-coupled-core-lora` (variant 2) | + transported core EMA, Nesterov | 0.9073 | 0.8883 | far worse |
| 6 | followup `muon-coupled-core-sign-rebalanced-lora` | sign + EMA + rebalance | 0.7684 (lr=1e-4) | 0.9440 | tied with #4, r=64 worse |
| 7 | `adam-lin-core-lora` (commit `0b4713c`) | core-Adam in DIFFERENT solver | DIVERGES step 2 | — | cross-check: core-Adam structurally broken |
| 8 | `polar-coupled-core-factor-adam-lora` (commit `f031dce`) | factor-Adam on (G_A,G_B), then our solver | extrap ~0.78 (lr=1e-4) | catastrophic | rung-6 ablation, **falsified** |
| 9 | + state-rebalance | (8) + post-step rebalance | extrap ~0.80 | catastrophic | no help |

**Best so far per rank:**
- r=16: `polar-coupled-core-sign-lora` lr=1e-4 → 0.7680 (still 0.013 behind r=16 best 0.7546)
- r=64: `polar-coupled-core-lora` (vanilla) lr=3e-2 → 0.7490 (beats AdamW; still 0.011 behind Picard 0.7382)

---

## 5. Diagnostic summary

What the per-step logs say across the variants:

**`compat`** — gradient-compatibility violation in core construction.
- Variants 1-6 (raw factor gradients): ≈ machine ε. Compatibility holds
  by construction; the (C_L+C_R)/2 averaging is a no-op.
- Variant 8 (factor-Adam): **0.65–0.88** in r=4 smoke at early steps.
  Factor-Adam genuinely breaks compatibility; the symmetrization is
  doing real work and may be lossy.

**`align_inst` vs `align_mom`** (variant 2 only — measures EMA alignment
with chosen polar direction, comparable to instantaneous-gradient
alignment).
- align_inst median: 0.45–0.50
- align_mom median: 0.30–0.55, **frequently below align_inst**
- Reading: core-space EMA does NOT accumulate constructively in the
  rotating basis. Successive cores point in different directions in the
  Q_L/Q_R frame; averaging dilutes rather than reinforces.

**`transport_residual`** (variant 2 only):
- median 0.04, max 0.08–0.10 → small. Transport is fine. The variant-2
  failure is the EMA itself, not the transport mechanism.

**`imbalance_residual`** (state-rebalance variant):
- 1.0 → 0.001 in 2 steps and stays there. Rebalance does what it claims.
- Eval gain: -0.014 at r=64, ~0 at r=16. Mechanism works but doesn't
  translate to loss reduction. The factor-imbalance pathology is real
  but not the bottleneck.

**`adam-lin-core-lora` cross-check (variant 7):**
- Smoke at lr=1e-3, r=4: eval 2.58 → 12.67 at step 2, Cholesky fails
  step 3. Core-space Adam is structurally broken in a completely
  different solver, by the same mechanism as variant 2: the small core
  matrix lacks heterogeneous coordinate scales, so /√v_M degenerates to
  ≈3·sign(M) at step 1, inflating step magnitude.

**`gamma`, `relgap`** (all our variants):
- γ within [1, 2] always. relgap typically 0.05–0.15. The ½-approximation
  certificate is well-behaved. We're computing the polar correctly.

---

## 6. Why we may still be losing — open hypotheses

**The user's question: it doesn't make sense that we can't beat Picard
when our algorithm is theoretically superior.** The hypotheses below are
ordered by how much we believe each.

### H1. We are solving the wrong problem.

The joint operator-norm problem is variationally clean but may not be
what Picard's recipe is solving. Picard does (i) factor-Adam, (ii) per-
factor polar in the spectral-preconditioned space, (iii) RMS-align step
magnitude back to Adam scale. None of (i-iii) is "solve the joint
operator-norm problem"; it's "do something Muon-shaped per factor with
Adam preconditioning". The fact that it works empirically suggests the
LoRA fine-tuning loss landscape doesn't reward the joint operator-norm
constraint — or rewards a different geometry that Picard happens to
match.

**Test:** strip our solver of the operator-norm objective and replicate
Picard's per-factor polar with our gauge analysis. Does it match Picard?

### H2. The (22) zero projection discards real signal.

Our active-core construction Ĥ = [[C, E], [F, 0]] explicitly zeros the
"extend both A and B simultaneously" mode. Picard's per-factor polar has
no such restriction. At low rank (r=16) where the bottleneck might be
exactly that joint-extension mode, our projection is removing what
Picard preserves.

**Test:** what does a variant of our solver with the (22) block UN-zeroed
do? It violates the variational story but may match Picard.

### H3. Step magnitude is wrong by a factor of τ̂.

Our step magnitude is `lr · τ̂` where τ̂ = ‖Ĥ‖_* / γ. Picard's step
magnitude is `lr · ‖u_A‖` (RMS-aligned to Adam direction norm). At fixed
lr, these are different scales. A variant 1 with lr=3e-2 r=64 = 0.7490
suggests we need bigger effective steps — but the right comparison is
"are we taking the same effective step magnitude as Picard at lr=3e-4?"

**Test:** plot ‖dA‖ trajectories of variant 1 vs Picard at their
respective best lr's. Match magnitudes; rerun.

### H4. Symmetrization (C_L+C_R)/2 is lossy at high compat.

Variant 8 directly hits this: factor-Adam → compat 0.65–0.88. The
averaging projects two genuinely different views into one. Picard avoids
it by never building C; instead it does two separate per-factor polars.
We'd want a variant that processes factor-Adam'd gradients without ever
symmetrizing.

**Test:** "factor-Adam + per-factor polar in core space" — pull factor
gradients, build C_L and C_R separately (no average), do separate polars,
lift separately.

### H5. Picard's iteration is doing something we lack.

Picard's k=2 cross-coupling step adds (B^T dB_prev A)/lr to u_A and (B
dA_prev A^T)/lr to u_B. This is a fixed-point iteration on the joint
problem. Our solver is one-shot. Even if our one-shot direction is
1/2-optimal for the *single-step* objective, Picard's iteration may
converge to a better fixed point of the implicit *training* dynamics.

**Test:** add an outer Picard-style iteration around our solver. Or: try
picard_iters=3, 4, 5 in Picard itself — is performance sensitive to
iteration count?

### H6. We have a bug.

Possibilities worth re-checking:
- α/r LoRA scaling in the gradient: PEFT applies (α/r) at the model
  layer, so G_A, G_B already include it. Our solver doesn't separately
  multiply. Picard also doesn't. Probably fine but worth confirming on
  a tiny case.
- The Sylvester lift formula. The blocks X, Y, W of Z_upd map to (dA,
  dB) via the formulas at top of section 2; the algebra is gauge-
  consistent in unit tests but the *practical* lift may have a sign
  mismatch on some block. Compare a 1-step output of our solver vs a
  hand-derived AdamLinLoRA Sylvester closed-form solution at β=0 and
  no Adam — they should match to ~1e-5 (test 4 in
  test_polar_coupled_core.py asserts this; passes). So this is unlikely
  but cheap to re-verify with a step-by-step trace at a real LoRA pair
  (not a synthetic one).
- The B=0 PEFT-init boundary case. Our solver triggers
  `_zero_B_fallback` at step 1 (B all zero); thereafter regular path.
  Picard's S_B^{-1/2} = (B^T B + δI)^{-1/2} just hits δ^{-1/2}·I at
  step 1, no special-casing. Could the fallback step's magnitude differ
  from what the regular path would compute on an infinitesimally non-
  zero B? Cheap test: initialize B=ε·N(0,1) for small ε at step 1,
  run vanilla path, compare to fallback.

---

## 7. What to try next

In order of expected information per GPU-hour:

1. **Match Picard's recipe with our gauge analysis** (tests H1, H4).
   Build `polar-per-factor-lora`: factor-Adam → per-factor polar (NOT
   joint core polar) → Sylvester gauge lift to enforce
   `B^T dB = dA A^T` post-hoc. If this beats Picard, the gauge analysis
   is the missing piece. If it ties Picard, the gauge analysis is
   irrelevant. If it loses to Picard, something else.

2. **Variant 1 with the (22) block UN-zeroed** (tests H2). Same as
   variant 1 but `R = P` not `P[r:,r:]:=0`. Quickest possible code change.

3. **Step-magnitude diagnostic comparison** (tests H3). Run variant 1
   vs Picard at their best lr's (3e-2 vs 3e-4 at r=64), log ‖dA‖, ‖dB‖
   per step. Plot. If ours is 10× larger or smaller, that's signal.

4. **Re-verify the Sylvester lift on a real LoRA pair, not a synthetic
   one** (tests H6). Single-pair trace, hand-compute, compare. ~30 min.

5. **Sensitivity of Picard to picard_iters** (tests H5). Sweep
   picard_iters ∈ {1, 2, 3, 5, 10} on Picard itself. If the optimum is
   at iters=2, our one-shot solver structurally cannot match. If iters
   doesn't matter, our one-shot story isn't the problem.

(1) and (5) are most informative. (1) is the experiment that would
either confirm we have a real solver advantage or settle that
gauge/coupling don't matter in practice. (5) is the only experiment
that doesn't require new code — a config sweep on existing Picard.

---

## 8. Reproducibility

All sweeps logged in `logs/<group>/run_info/` with manifest. Pull final
results via `lora_playground.loader.load_runs(where={...})`. Diagnostics
on every cell via `--log_optim_diagnostics --optim_diagnostics_every 200`.

Headline analysis script: `scripts/phase2_summary.py`.

Sweeps:
- `polar_coupled_core_2k` — variant 1, variant 2 baseline
- `state_rebalanced_2k` — variant 1.5
- `polar_core_wide_lr_2k` — wide-lr scan
- `polar_core_sign_2k` — sign normalization
- `polar_core_sign_followup_2k` — sign × EMA × rebalance compounds
- `polar_factor_adam_2k` — factor-Adam ablation (rung 6)
