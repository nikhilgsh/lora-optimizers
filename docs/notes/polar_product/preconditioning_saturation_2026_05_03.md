# Preconditioning saturation investigation (2026-05-03)

## TL;DR — corrected after r=16 vs r=64 spectrum check

**Within-matrix preconditioning upstream of polar appears saturated at both
r=16 and r=64** (SOAP-polar tested at both ranks; |Δ| ≤ 0.003 at both). The
"upstream rotation gets absorbed by polar" claim has multi-rank evidence and
the mechanism (Adam already tightens NS-input σ to a convergent regime) is
backed by direct spectrum measurements.

**Within-matrix DOWNSTREAM operators (σ^p / Muon+ / clip) are saturated at
r=16 but plausibly NOT at r=64.** At r=16, X_unc σ_max/σ_min ≈ 2 throughout
training — almost no spread for spectrum-shaping operators to compress. At
r=64, the same X_unc has σ_max/σ_min ≈ 7 — 3× wider spread. The Muon+
and HTMuon negative results from today's r=16 sweeps are likely r=16-specific
artifacts; **we owe r=64 reruns to validate or refute downstream-operator
saturation**.

The implication: claims sourced from r=16-only data are tagged below. The
"polar is doing very little" framing was specifically a r=16 property:
polar's marginal value is **0.0035 nats at r=16 but 0.0074 nats at r=64**
(2.1× larger). r=64 is where downstream-operator differences would matter
most, and we have not yet tested most of today's variants there.

All numbers below are **single-seed at the canonical 2k-step horizon** unless
explicitly noted, pulled via `lora_playground.loader.load_runs(where=…)` and
JSONL parsing of `optim_step` events.

**Significance reference: multi-seed AdamW at η=3e-4 step 2000 gives**

| | n=4 seeds | mean | std | 2σ |
|---|---|---|---|---|
| r=16 | seeds {1,2,3,4} | 0.7600 | 0.0006 | **0.0012** |
| r=64 | seeds {1,2,3,4} | 0.7538 | 0.0007 | **0.0014** |

So variant Δ values below are characterized in σ-units against this noise
floor where useful. Δ within 1σ is consistent with no effect; Δ above 2σ
is a real difference (single-seed → multi-seed extrapolation, but the AdamW
seed variance is a defensible noise floor for the workload).

## Code changes (committed today)

- `AdamSOAPPolarProductLoRA` — v2 corrected from a buggy v1 (momentum-frame
  bug, refresh-order bug, full-eigh discontinuity). Passes equivalence tests
  against the official `nikhilvyas/SOAP` repo and an inline reference of
  paper Algorithm 3.
- `AdaFactorPolarProductLoRA` — Adam with rank-1 v factorization fed into
  polar pipeline. Includes `stable_rank(g²)` diagnostic.
- `LoRAAdafactor` (pure baseline) — wraps HuggingFace's `Adafactor` with
  `scale_parameter=False, relative_step=False, β₁=0.9` so it consumes our
  explicit lr and has Adam-style momentum.
- `polar_norm_dir` knob on `AdamPolarProductLoRA` — Muon+
  (arXiv:2602.21545) row/col ℓ₂ normalization in `_polar_pipeline`.
- `polar_sigma_power` knob — HTMuon (arXiv:2603.10067) σ → σ^p generalized
  polar via SVD.
- `--beta1` `--beta2` CLI knobs for testing instant-Adam (β₂=0).
- New diagnostic emitters: `optim_soap_step` (L_A, R_B PR / top_frac /
  cond), `optim_adafactor_step` (stable_rank(g²)), and `geoA_row/col_norm_cv`
  / `geoB_row/col_norm_cv` fields on the existing `optim_step` events.
- 6 equivalence/correctness tests in `tests/test_polar_product.py`.

## Empirical findings

### Within-matrix preconditioning saturation (UPSTREAM, multi-rank)

All Δ at the best LR are below 1σ (~0.0012-0.0014):

| Optimizer | r=16 best Δ | r=64 best Δ | σ-units |
|---|---|---|---|
| adam-soap-polar-product-lora | ≤ 0.003 (step 2000) | ≤ 0.003 (step 1000) | ~2σ early, settles within 2σ |
| adafactor-polar-product-lora | ≤ 0.001 (step 2000) | −0.0002 (step 400, ongoing) | < 1σ |

Plain LoRA (no polar):

| Optimizer | r=16 step 200 | r=64 step 200 | σ-units |
|---|---|---|---|
| adafactor (HuggingFace) vs adamw | within ±0.005 | within ±0.005 | ≤ 4σ at step 200; converges within 1-2σ later |

**Reading:** at the best LR, AdaFactor and SOAP land within 1-2σ of Adam at
both ranks. The structural reason — LoRA per-example gradients are exactly
rank-1, so v's rank-1 factorization is near-exact — is rank-independent, so
this saturation conclusion is multi-rank robust.

### Mechanistic explanation — but rank-dependent

`xunc_A` and `xunc_B` are the inputs to Newton-Schulz inside the polar
pipeline (Adam direction post-Gram-preconditioner). The existing diagnostic
shows the σ spread is **dramatically rank-dependent**:

| | r=16 (Adam-polar, η=3e-4) | r=64 (Adam-polar, η=3e-4) |
|---|---|---|
| σ_max/σ_min A | 2.2–4.2 throughout | **6.9–11.2 throughout** |
| σ_max/σ_min B | 1.9–2.2 throughout | **6.0–13.6 throughout** |
| stable_rank A | 7.6/16 (0.48r) | 9.5/64 (0.15r) |
| stable_rank B | 9.7/16 (0.61r) | 12.9/64 (0.20r) |

**At r=16** the input to NS is nearly flat (σ_max/σ_min ≈ 2) — well inside
NS's convergent regime, so NS produces a near-perfect polar output and any
upstream variant that yields a similar input lands at the same place.
**At r=64** the input has σ_max/σ_min ≈ 7 — meaningful spread that polar
must compress, and where NS's iteration count + coefficient choice
plausibly matter more.

This rank-dependence directly predicts that **downstream-operator variants
(σ^p, clip, Muon+ normalization) saturate at r=16 by construction but
should differentiate at r=64**. σ^p compression scales as
(σ_max/σ_min)^p:

| p | r=16 post-op ratio | r=64 post-op ratio |
|---|---|---|
| 0 (polar) | 1.0 | 1.0 |
| 0.125 | 1.09 | **1.28** |
| 0.5 | 1.41 | **2.65** |
| 1 (no op) | 2.0 | 7.0 |

At r=16 the post-op ratios for p ∈ [0, 0.5] are all in [1.0, 1.4] — sub-2×
spread, mostly absorbed by RMS-align. At r=64 they're in [1.0, 2.65] —
the σ^p variants plausibly produce visibly different update directions.

### Cross-correlations with the LoRA structure

LoRA gradients are near-rank-1 by construction:

```
gA = (α/r) · Bᵀ · (∂L/∂y) · xᵀ  ⇒  rank ≤ 1 per example
gA² (elementwise) is also rank-1 per example
```

Batch averaging only modestly increases stable rank. The
`stable_rank(g²)` probe on adafactor-polar-product-lora confirms:

| step | sr(gA²) median | sr(gB²) median |
|---|---|---|
| 880 | 1.08 | 1.62 |
| 900 | 1.12 | 2.26 |

This is the *structural* reason rank-1 v ≈ full v: the rank-1
reconstruction is near-exact for LoRA gradients, so AdaFactor's
approximation isn't lossy.

## Marginal value of polar — rank-dependent

Polar's marginal value over `adam-lin-lora` (Adam → Gram precond → no polar)
at step 2000:

| rank | adam-lin-lora | adam-polar | Δ from polar |
|---|---|---|---|
| r=16 | 0.7581 | 0.7546 | **0.0035** |
| r=64 | 0.7527 | 0.7453 | **0.0074** |

**Polar's headroom is 2.1× larger at r=64.** This is a property of the input
σ spread: at r=16 polar compresses ratio 2 → 1 (small change after RMS-align
absorbs); at r=64 polar compresses ratio 7 → 1 (substantive).

Synthesis:

| Mechanism | Owner | r=16 saturated? | r=64 saturated? |
|---|---|---|---|
| Within-block direction (singular VECTORS) | Polar / NS | yes | likely yes |
| Within-block magnitude (singular VALUES) | Polar / NS at convergent input | **yes** (σ ratio ≈ 2) | **partial** (σ ratio ≈ 7, polar still works) |
| Cross-block magnitude (per-block LR) | Adam's 1/√v | yes | yes |
| Spectrum-shaping operator choice (σ^p, clip, polar) | downstream operator | **yes by construction** (input flat) | **NO — UNTESTED** |
| Post-orthogonalization row/col homogenization | Muon+ row/col norm | r=16 says no signal | **UNTESTED at r=64** |
| Effective-rank lifting of momentum | Multi-step lookahead, sketching, sign(m) | running r=16 only | untested |

## Currently running sweeps

| Job | Group | Cells | Tests |
|---|---|---|---|
| 6328137 | `muon_plus_polar_dir_r16_2k` | row, col, row_col, col_row at η=3e-4 r=16 | Muon+ post-orthogonalization normalization |
| 6328169 | `instant_adam_polar_r16_2k` | β₂=0 at η=3e-4 r=16 | Whether v EMA matters upstream of polar |
| 6328170 | `htmuon_polar_r16_2k` | p ∈ {0, 0.125, 0.25, 0.5} at η=3e-4 r=16 | HTMuon σ → σ^p generalized polar |

## Predictions for the running sweeps

These are **prior beliefs**, not derived guarantees. Recorded so we can
calibrate against the actual results.

### Muon+ direction sweep (job 6328137)

**Premise validation first.** The `geoA_row_norm_cv` / `geoB_*_norm_cv`
diagnostic measures std/mean of per-row/col ℓ₂ norms of the orthogonalized
output, *before* any Muon+ normalization. Predictions:

- If CV < 0.1: rows/cols are already balanced; Muon+ is a no-op; expect
  Δ ≈ 0 vs Frobenius baseline.
- If CV ≈ 0.2–0.5: there's real variance to homogenize; Muon+ should help
  by some amount, plausibly Δ ≈ −0.002 to −0.010.
- If CV > 0.5: large variance; Muon+ should help meaningfully, Δ ≈ −0.01
  to −0.03.

**Best-guess prior:** the orthogonalization step (Newton-Schulz) outputs
near-orthogonal matrices with σ ≈ 1, but their rows/cols still have ℓ₂
variance from the underlying U, V structure. CV ≈ 0.2-0.4 is plausible.
The Muon+ pretraining-scale gains were 0.4-2.0 PPL (≈ 0.02-0.05 nats),
not all of which transfers to fine-tuning. **Expected Muon+ best Δ ≈
−0.005 to −0.015** at this scale.

**Direction prior:** Muon+ paper reports col best for some configs, row
for others, composed (col→row, row→col) sometimes wins. No strong prior
on which wins for LoRA — direction is genuinely empirical.

### Instant-Adam β₂=0 (job 6328169)

For LoRA's near-rank-1 g², instant v_t = g²_t is exactly the per-element
square of the current gradient. u = m / (|g| + ε). The momentum m is
EMA-smoothed but the denominator is not.

- The denominator is noisier step-to-step than Adam's EMA-v.
- Polar saturation argument: once σ_max/σ_min of u is in NS-convergent
  regime, the resulting polar output is close to the same.
- BUT: noisier denominator may push σ_max/σ_min wider on individual
  steps, occasionally falling outside the convergent regime.

**Expected:** small loss vs Adam-polar baseline. **Δ ≈ +0.005 to +0.020**
at the best LR. If Δ ≈ 0 or negative, that's strong evidence upstream
EMA is fully saturated. If Δ > +0.02, EMA matters more than the polar
saturation argument suggests.

### HTMuon σ → σ^p (job 6328170)

Four cells, ordered by predicted impact:

**p = 0** (SVD-based exact polar — control vs NS approximation).
HTMuon paper Figure 1a says **Muon_NS beats Muon_SVD** by 1-2 PPL at
LLaMA pretraining. The reason: NS leaves residual σ variation which
incidentally down-weights noise-dominated directions (as discussed in
their §3.2). **Predicted Δ vs NS-polar: +0.005 to +0.015** (worse).

**p = 0.125** (HTMuon paper default).
σ raised to 0.125 concentrates mass into top singular directions while
preserving heavy-tailedness. At LoRA r=16 the underlying matrix has at
most rank 16 and effective rank ~6-10 (per our existing spectrum
diagnostics). σ^0.125 of [1, 0.5, 0.25, ...] gives [1, 0.92, 0.84, ...]
— mild compression, retains most of the hierarchy. **Predicted Δ:
−0.005 to +0.005**, near-noise. Could be slightly positive (heavier tails
on already-low-rank LoRA may help).

**p = 0.25, 0.5** (intermediate).
σ^0.5 = √σ — meaningful retention of magnitude hierarchy. At p=0.5 the
update is closer to (square-root-of-momentum) than (orthogonalized
direction). At LoRA r=16 with already-low effective rank, the larger σ
spread amplifies the dominant direction's contribution and may concentrate
updates into ~1-2 directions per pair. **Predicted Δ: +0.005 to +0.025**
(worse, monotonically increasing with p past 0.125).

**Overall HTMuon prior:** the heavy-tail-preservation thesis is motivated
by HT-SR theory of *weight* spectra in pretrained networks. At LoRA
fine-tuning scale, the base model is already heavy-tailed; LoRA's update
shape may not propagate to weight ESDs the same way. **Expected best
HTMuon p: 0.125 with Δ ≈ 0** (no meaningful improvement). If best Δ <
−0.01, the HT thesis transfers to fine-tuning and warrants follow-up.

## Calibration prior across all 3 sweeps

- **Most likely outcome:** Muon+ best direction wins by 0.005-0.015;
  instant-Adam loses by ≈0.01; HTMuon best p (0.125 or 0) within ±0.005.
- **Lowest-probability surprise:** HTMuon at p=0.125 wins by > 0.02. Would
  motivate an r=64 follow-up where the heavy-tail argument is stronger.

## What this investigation does not test

- Multi-seed reliability of any of the above Δ values. All single-seed.
- r=64 for Muon+, instant-Adam, HTMuon (only r=16 cells queued).
- Stacking: Muon+ × HTMuon, AdaFactor × Muon+, etc.
- Activation-side mechanisms (Davis-Drusvyatskiy's framework only used
  to motivate polar; we do not yet probe activation stable rank
  empirically).
- Effective-rank lifting of the momentum input.

## Sign-momentum result (added after r=16 trajectory + r=64 first eval)

Sign-momentum-polar (LION-style: u = sign(m), drops v entirely) at the best
LR is **+0.003 above Adam-polar** at the best LR, both ranks:

| | best η | step | sign-momentum loss | Adam-polar best | Δ | σ-units |
|---|---|---|---|---|---|---|
| r=16 | 1e-4 | 1200 | 0.7748 | 0.7719 | +0.0029 | ~2.5σ |
| r=64 | 1e-4 | 200 (early) | 0.8230 | 0.8197 | +0.0033 | ~2.4σ |

Gap is stable across the r=16 trajectory (+0.004 at step 400, +0.003 at step
600/1200). 2-2.5σ above the AdamW noise floor — **meaningfully worse than
Adam-polar, not equivalent**. Earlier framing of "near-equivalent" was an
overclaim that violated the project's policy on significance.

The sign(m) → polar variant therefore demonstrates:
- v IS doing real work for the loss (not just stability) — its absence costs
  a measurable ~0.003 nats vs full Adam.
- Bounded-magnitude denominators (sign rather than 1/|g|) avoid the
  divergence that broke instant-Adam β₂=0, but don't recover Adam's full
  performance.
- Sign-momentum is a viable simplification for memory/code reasons IF the
  ~0.003 cost is acceptable; not a drop-in replacement.

## Decision rules for next steps after these sweeps land

**For r=16 sweeps (currently running):** the saturation argument predicts
all variants land within ±0.005 of baseline. That is informative as a
*negative result for r=16*, but is NOT conclusive for the workload as a
whole — see the rank-dependence above.

**Required follow-up regardless of r=16 outcomes:** rerun Muon+, HTMuon,
sign-momentum, and AdaFactor-polar at r=64. The X_unc spectrum data shows
σ ratio ≈ 7 at r=64 vs ≈ 2 at r=16, so downstream-operator variants have
3× the σ-compression headroom to differentiate. The r=16 data does not
generalize.

**If r=64 reruns also show |Δ| < 0.005:** then the saturation conclusion
holds across ranks, and we accept adam-polar-product-lora as near-optimal
in this design space. Pivot to activation-side mechanisms or
effective-rank lifting.

**If any r=64 variant produces |Δ| > 0.01:** that's the unsaturated axis.
Investigate further: multi-seed verification, mechanism diagnostics,
LR sweep, possible stacking with the other variants.
