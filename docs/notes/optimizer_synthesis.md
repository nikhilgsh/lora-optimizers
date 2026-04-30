# Optimizer investigation — synthesis

**Goal:** find a LoRA-aware optimizer that strictly beats AdamW (0.7579 final
eval loss at r=16, 2k steps, OLMo-2-1B + Magicoder code-instruct).

## Methods tried — three-bucket organization

### Bucket 1: clear results, served as intel + launch points (closed)

These are confirmed findings that informed subsequent design but don't
themselves point at a next experiment.

- **H1 cos diagnostics at r=16** (job 6312334, done): cos_post_B ≈ 0.97
  throughout, cos_post_A 0.46→0.84. Established that pre-Adam compositions
  are ε-perturbed AdamW.
- **H1 cos_pre 500-step probe** (job 6313190, done): cos_pre_B ≈ 0.98 (S_A⁻¹
  near-identity on ∇B early); cos_pre_A 0.65→0.85. Established H_weak on
  B early-phase — the geometric correction has nothing to install on B in
  the first 500 steps because A is approximately a random projection.
  *Caveat:* unverified past step 500 (2k probe in flight).
- **H3 r-sweep** (job 6312335, done): final-eval gap (lin/scaled − adamw)
  monotonic in r: +0.023 at r=2, +0.022 at r=4, ≈0 at r=16, −0.005 at r=64.
  Crossover near r=16.
- **H3 robustness side-finding**: at r=64 η=1e-3, plain AdamW *diverges* to
  0.89 while adam-{lin,scaled}-lora hold at 0.77. Geometric preconditioning
  gives lr-headroom at high rank.
- **H4 unfixed *-Post** (job 6312277, done): best 0.7875 at η=1e-3.
  Established the magnitude-drift bug (σ_min(S_B) climbs 0.011→1.08 → step
  magnitude varies 100× over training at fixed lr).
- **H4 RMS-aligned *-Post** (job 6313020, done): adam-scaled-lora-post
  η=3e-4 → 0.7570. After magnitude fix, ties AdamW (within noise floor).
  Confirmed magnitude was the dominant H4 bug.
- **muon-lora baseline** (`new_optimizers_high_eta_2k`): 0.7675 (NS without
  Adam). Establishes how much NS alone gets you.
- **muon ns_steps=0** (`muon_nsoff_2k`): 0.95+ at every η. NS contributes
  ≥ 0.18 nat — geometric orthogonalization is essential, not decorative.
- **diag-scaled-lora, kron-grad-lora**: 0.815, 0.826. Diagonal K-FAC family
  underperforms. Closed branch unless we revisit the K-FAC formulation
  more carefully.

### Bucket 2: promising, worth trying to improve

These have a measured win or measurable trajectory, with concrete next
moves to push further.

- **adam-muon-lora** (0.7557 at r=16, the only confirmed strict Δ=−0.0022).
  Next moves: r=64 sweep (untested combination of two known win-effects);
  AdaMuon-faithful port (sign-stabilization + per-element v̂ on NS-output +
  RMS-align — published precedent at pretraining scale claims +40% efficiency
  but caveats apply for fine-tune regime).
- **adam-scaled-lora at r=64** (0.7506, leaderboard #1, within noise floor).
  Next moves: cos diagnostics at r=64 to confirm mechanism (job 6313087 in
  flight); full η-sweep at r=64 to confirm peak isn't at η=3e-4.
- **AdamPolarProductLoRA** (just implemented; theory's closed-form polar
  update under spectral-product norm). Smoke at r=16 η=3e-4 → eval 1.03 at
  step 5; at r=64 η=1e-3 → eval 0.93 at step 5. Behavioral equivalence
  test passes (reduces to Muon NS at orthogonal init). Sweep pending.
  Mechanistically the cleanest variant we have — uses both the LoRA
  product structure AND a spectrally-meaningful correction (polar) AND
  composes correctly with Adam (matrix-structural so v̂ doesn't erase).
- **H5 matrix-Adam fixed** (job 6312759 in flight). η=1e-3 trending to 0.78
  at step 1200; final pending. If it ties or marginally improves over the
  *-Pre baseline at r=16, the r=64 variant (job 6313316 in flight) might
  win since it preserves direction × extra rank dimensions. Caveat: by
  the user's reframing, this trades Adam's per-coord stability — net is
  uncertain.

### Bucket 3: theoretically promising but empirically weak — open mechanism

Things that *should* have helped per theory or analogous literature but
came in at parity or worse. Each warrants a focused investigation rather
than abandonment.

- **adam-{lin,scaled}-lora pre-Adam** (0.7564 / 0.7572 at r=16, ties
  AdamW). Theory: Sylvester / Frobenius product norm gives optimal
  product-aware update. Empirical: H1 explains the tie (Adam's per-coord
  v̂ erases the rotation). **Open question:** does the same Sylvester
  give a real win under matrix-Adam (H5) with the direction preserved? In
  flight.
- **muon-adam-lora** (NS first, then Adam). Best η=1e-3 step 600: 0.80,
  decisively losing. *Theoretically* a reasonable composition (orthogonalize
  then adapt). Empirical guess for failure: per-coord √v̂ on NS-output is
  unstable because NS already evened out magnitudes. AdaMuon (paper) handles
  this with (1) sign(M) before NS, (2) only v̂ no m̂ on NS-output, (3) RMS-
  align. **Open question:** does our muon-adam-lora's failure trace to
  one of these three missing ingredients, or is it more fundamental?
  Untested.
- **product-muon-lora** (gauge-invariant Sylvester-recovered NS). Pilot
  showed ~0.81 at step 500; 2k extrapolation didn't suggest a clear win.
  Theory says correct product-norm + spectral. Empirical: middling.
  **Open question:** is the failure in the implementation (e.g.,
  Sylvester recovery numerical issues), in the lora_plus_multiplier
  interaction, or in the underlying premise that this gauge-invariant
  form should outperform per-factor independent NS? AdamPolarProductLoRA
  (just implemented) is a cleaner test of "spectral × product" — its
  result will speak to ProductMuon's framing too.
- **PSI-LoRA Algorithm 3** (job 6312780 running): F-LoRSUM proximal ALS +
  low-rank momentum. Paper-faithful port of the published algorithm.
  *Empirical:* pending. *Open question:* whether the paper's claimed wins
  on smaller models transfer to OLMo-2-1B + Magicoder.
- **GaLore** (`galore_fixed_2k`, done — need to load result). ~3× slower
  per step than LoRA-mode (full dense backward). *Open question:* does
  the rank-r projected gradient subspace approach beat plain LoRA at
  matched compute?

### What gets re-examined as in-flight sweeps land

The boundary between Bucket 2 and Bucket 3 will shift over the next ~1
hour as 6 jobs finish:

- 6312759 (H5 r=16): if ties AdamW → confirms matrix-Adam alone isn't
  enough; if marginal win → moves to Bucket 2 with "improve via r=64".
- 6313316 (H5 r=64): the matrix-Adam-direction-preservation × rank
  combination. If wins → Bucket 2 promising path.
- 6313087 (cos at r=4, r=64): mechanism for *why* rank-dependence flips
  sign. Will inform whether AdamPolarProductLoRA is expected to do well.
- 6313261 (2k cos_pre): does S_A stay isotropic? Conditions whether
  H_weak generalizes to full-trajectory training.
- 6313020 (H4 RMS-aligned): already done; *-Post ≈ ties AdamW at r=16.
- AdamPolarProductLoRA sweep (about to submit): the theory-prescribed
  spectral-product test.

## What we currently believe (2026-04-30 ~13:00)

**Confirmed:**
- At r=16, the original `adam-{lin,scaled}-lora` are ε-perturbed AdamW.
  H1: cos_B ≈ 0.97 throughout 2k steps, cos_A → 0.84 over training.
- The *-Post variants without RMS-align (H4 unfixed) lose to AdamW by
  ~0.03 — step magnitude varied by 100× over training (σ_min(S_B) climbs
  from 0.011 → 1.08).
- adam-muon-lora (Muon NS applied to Adam's step direction) is the only
  unambiguous strict win at r=16: 0.7557 vs AdamW 0.7579.
- The geometric correction's effect is **rank-dependent**: gap goes from
  +0.02 at r=2 (lin/scaled lose) to −0.005 at r=64 (lin/scaled win).
  Crossover near r=16.
- B-side cos_post ≈ 0.97 across r ∈ {4, 16, 64} throughout 2k (cos *after*
  Adam, measured for *-Pre variants). Whatever happens upstream of Adam,
  the applied B-direction matches plain AdamW's B-direction.

**Verified through step 500 only (2k probe in flight as job 6313261):**
- cos_pre_B ≈ 0.98: S_A⁻¹ does not rotate ∇B meaningfully in the first 500
  steps. Whether this holds past step 500 is currently being tested. The
  Kaiming-init "random projection" argument that motivated this prediction
  applies at init; it does NOT prove S_A stays isotropic as A trains.

**Likely but not yet verified:**
- The r=64 win is real-direction but its *magnitude* (~0.005) is within
  the single-seed noise floor (mean step-to-step |Δ_loss| ≈ 0.004 in late
  trajectory). Mechanism evidence > confidence intervals.
- The RMS-align fix gets *-Post into AdamW pace at r=16 — final verdict
  pending (step 1000 trajectory matches plain AdamW).
- "Geometry on A only" should perform identically to full geometry-then-
  Adam (since B-side is inert). Untested ablation.

**H_weak vs H_erase — early-phase verdict (job 6313190, 500 steps):**

| factor | cos_pre (step 20) | cos_pre (step 500) | cos_post (step 20) | cos_post (step 500) |
|--------|-------------------|---------------------|---------------------|----------------------|
| B      | 0.98              | 0.98                | 0.98                | 0.97                |
| A      | 0.65              | 0.85                | 0.46                | 0.80                |

**What this verifies (verified):** through step 500 at r=16, S_A⁻¹ does not
meaningfully rotate ∇B (cos_pre_B ≈ 0.98 throughout). On A, S_B⁻¹ does
rotate ∇A meaningfully (cos_pre_A reaches 0.65 at step 20, settles at 0.85
by step 500). Adam's per-coord √v̂ subtracts an additional ~0.2 from cos on
A early, ~0.05 late (cos_post < cos_pre).

**What this does NOT verify (still open, 2k probe in flight as job 6313261):**
whether S_A *stays* near-isotropic past step 500. Possible scenarios:

- (S_A stays I-like throughout 2k): consistent with current evidence;
  would mean the B-side parameter Gram preconditioner has nothing to project
  onto for the full LoRA-fine-tune trajectory. Plausible if A doesn't drift
  far from init in this regime.
- (S_A develops loss-aligned structure late): A receives gradient updates
  ∇A = Bᵀ·∇W which carry loss-aligned structure in B's column space. If
  these updates accumulate enough to dominate the original Kaiming init,
  S_A becomes non-isotropic and S_A⁻¹·∇B becomes a meaningful rotation.
  In this scenario, the B-side correction earns its keep late in training,
  and "drop S_A⁻¹" would be the wrong reframing.

**Reasoning we can do without the late-phase data:**
- The intuition that A is a "random projection" comes from Kaiming init
  variance scaling and concentration of measure for r ≪ d_in. Both apply
  at init. Neither apply unconditionally to a trained A.
- Adam's update on A drives A toward parameter regions where the loss
  gradient (and hence B's gradient) flows. So *some* loss-alignment of A
  is mechanistically expected; the open question is whether it's enough
  to flip cos_pre_B away from 1.

**Why the geometric correction helps more at r=64 than r=16 (open):**
hypothesis is that at higher r, S_B has more dimensions for B to grow into,
so the A-side rotation is richer. The cos diagnostics from h1_rsweep_diag_2k
will speak to this when the sweep finishes.

**One-line mechanistic story (provisional):** Adam's per-coordinate v̂⁻¹ᐟ²
normalization, when applied *to the preconditioned gradient*, erases
whatever cross-coordinate scale structure the preconditioner installed —
*at small rank*. At higher rank (r ≥ 64) the geometric subspace is rich
enough that Adam can't fully flatten it and a small win opens up. The
optimizers that beat AdamW unambiguously at r=16 are the ones that apply
geometry **after** Adam AND choose a correction structurally meaningful on
a sign-like input — **NS (spectral cap)** qualifies, **S⁻¹ (Gram-inverse)**
seemingly does not (H4 *-Post falsifying even after the magnitude fix —
final number pending).

---

## Standing leaderboard (best η, seed=0, 2k steps)

**Cross-rank board:**

| rank | optimizer                       | r  | best η | eval loss  | source                         | vs AdamW r=16 |
|------|---------------------------------|----|--------|------------|--------------------------------|---------------|
| 1    | **adam-scaled-lora**            | 64 | 3e-4   | **0.7506** | `h3_rsweep_2k`                 | ✅ Δ=−0.0073* |
| 2    | adam-lin-lora                   | 64 | 3e-4   | 0.7527     | `h3_rsweep_2k`                 | ✅ Δ=−0.0052* |
| 3    | adamw                           | 64 | 3e-4   | 0.7550     | `h3_rsweep_2k`                 | ✅ Δ=−0.0029  |
| 4    | adam-muon-lora                  | 16 | 3e-3   | 0.7557     | `adam_muon_2k`                 | ✅ Δ=−0.0022  |
| 5    | adam-lin-lora                   | 16 | 1e-3   | 0.7564     | `optim_compare_high_eta_2k`    | ≈ tied        |
| 6    | **adam-scaled-lora-post (RMS-align)** | 16 | 3e-4   | **0.7570** | `h4_post_rmsalign_2k`     | ≈ tied (NEW)  |
| 7    | adam-scaled-lora                | 16 | 1e-3   | 0.7572     | `optim_compare_high_eta_2k`    | ≈ tied        |
| 8    | adamw                           | 16 | 3e-4   | 0.7579     | `lr_sweep_2k`                  | baseline      |
| 9    | adam-scaled-lora-post (RMS-align) | 16 | 1e-3 | 0.7628     | `h4_post_rmsalign_2k`          | +0.005        |
| 10   | adam-lin-lora-post (RMS-align)  | 16 | 3e-4   | 0.7641     | `h4_post_rmsalign_2k`          | +0.006        |
| —    | muon-lora (LoRA+ m=4)           | 16 | 1e-3   | 0.7674     | `muon_loraplus_2k`             | ❌            |
| —    | adam-lin-lora-post (unfixed)    | 16 | 1e-3   | 0.7875     | `h4_post_2k`                   | ❌ (fix saves it) |

*all r=64 wins are within the single-seed noise floor (≈ 0.004); the
*direction* is reliable, the *magnitude* needs mechanism evidence (cos
diagnostics from h1_rsweep_diag_2k) rather than multi-seed.

**Robustness observation from H3:** at r=64 η=1e-3, plain AdamW *diverges*
to 0.89 while adam-{lin,scaled}-lora hold up at 0.77-0.78. Geometric
preconditioning gives the optimizers more lr-headroom at high rank — a
side benefit not visible at r=16.

**r=16-only ranking:** adam-muon-lora wins (0.7557).
**r=64 ranking:** adam-scaled-lora wins (0.7506).
**Overall:** adam-scaled-lora at r=64 is the new headline number.

Two findings tension each other:
- **At r=16, geometry-then-Adam compositions tie AdamW** (H1 confirmed: cos
  ≈ 0.97 throughout, Adam erases geometric correction).
- **At r=64, geometry-then-Adam compositions beat AdamW** (Δ ~ −0.005). H1
  cos diagnostics weren't run at r=64, so the *mechanism* isn't pinned down,
  but the empirical result is clear.

**Hypothesis for the rank dependence:** at higher r, the LoRA factor
matrices are larger, so Adam's per-coord v̂ has more cross-coordinate
structure to erase. Either v̂ can't fully flatten the higher-dimensional
geometric correction, or σ_min(S_B) reaches a regime where the geometric
correction matters more (S_B = BᵀB + δI is r×r — bigger r, more
information). Need to run H1 diagnostics at r=64 to confirm.

---

## What was tried, organized by mechanism

| family                    | optimizer                | preconditioning              | composition order      | result    |
|---------------------------|--------------------------|------------------------------|------------------------|-----------|
| **AdamW family**          | adamw                    | per-coord v̂                  | Adam only              | 0.7579    |
| **Pre-precondition**      | adam-lin-lora            | Sylvester / S_B⁻¹ then v̂     | geometry → Adam        | 0.7564    |
| (geometry feeds Adam)     | adam-scaled-lora         | Gram solve then v̂            | geometry → Adam        | 0.7572    |
| **Post-precondition**     | adam-muon-lora           | NS on Adam direction         | Adam → spectral cap    | **0.7557** |
| (Adam feeds geometry)     | adam-lin-lora-post       | Sylvester on Adam direction  | Adam → Sylvester       | 0.79+ (in flight, falsifying) |
|                           | adam-scaled-lora-post    | Gram solve on Adam direction | Adam → Gram solve      | 0.84+ (in flight, falsifying) |
| **Per-pair scalar v̂**    | adam-lin-lora-matrix     | per-pair scalar Adam on precond | geometry → "matrix-Adam" | (resubmitted, pending) |
|                           | adam-scaled-lora-matrix  | per-pair scalar Adam on precond | geometry → "matrix-Adam" | (resubmitted, pending) |
| **Spectral baseline**     | muon-lora (NS only)      | NS on raw momentum           | NS only                | 0.7675    |
|                           | muon-lora ns_steps=0     | momentum SGD                 | none                   | 0.95+ (NS contributes ≥ 0.18 nats) |
| **Hybrid spectral+geom**  | product-muon-lora        | NS on gauge-invariant proxy  | spectral pipeline       | ~0.81 (500-step extrapolation, not run to 2k) |
|                           | adam-product-muon-lora   | both                         | both                    | tbd (queued) |
| **K-FAC / diag**          | diag-scaled-lora         | per-coord D_V/D_U            | diag preconditioning    | 0.8153    |
|                           | kron-grad-lora           | r×r grad outer-product       | + diag                  | 0.8263    |
|                           | psi-lora (Algorithm 3)   | F-LoRSUM proximal ALS        | + low-rank momentum     | (in `psi_lora_2k`) |
| **Full-weight projected** | galore-adamw             | rank-r SVD projection        | Adam in subspace        | (in `galore_fixed_2k`, ~3× slower) |

---

## H1 — diagnostic: why pre-precondition compositions tie AdamW

**Setup.** `adam-lin-lora` and `adam-scaled-lora` at η=1e-3, 2k steps, with
`--log_optim_diagnostics --optim_diagnostics_every 20`. Each step computes
both the geometric-Adam step Δ_lin and a side-channel plain-AdamW step
Δ_raw on the same gradient (independent m,v state), then logs cosines and
norm ratios across all 112 LoRA pairs (r=16, all-linear).

**Final-step values (step 2000, median across 112 pairs):**

| optimizer        | cos_A | cos_B | ‖dA_lin‖/‖dA_raw‖ | σ_min(S_B) | ‖B‖_F |
|------------------|-------|-------|--------------------|------------|--------|
| adam-lin-lora    | 0.84  | 0.94  | 0.25               | 1.08       | 7.68   |
| adam-scaled-lora | 0.88  | 0.97  | 0.27               | 1.14       | 6.52   |

**Trajectory (adam-lin-lora):** cos_A rises 0.46 → 0.84 in first 500 steps
then plateaus; cos_B starts at 0.98 and stays ≥ 0.94 throughout.
σ_min(S_B) climbs 0.011 → 1.08 driven by ‖B‖² growth.

**Three structural conclusions:**
1. **cos_B ≥ 0.94 from step 20.** The geometric correction on B is
   indistinguishable from a plain-AdamW step direction throughout training.
   There is no early window where the B-side geometry does something special.
2. **cos_A converges to 0.84-0.88, asymptotically tracking AdamW.** The only
   meaningful direction divergence is on A in the first ~500 steps, before B
   leaves zero and S_B becomes well-conditioned.
3. **Geometric step has ¼ the magnitude of AdamW step.** Even where directions
   differ, ‖dA_lin‖/‖dA_raw‖ ≈ 0.25 — the geometric correction is consistently
   damped vs plain AdamW.

**Why:** Adam's per-coord √v̂ in the denominator divides g by its own EMA
RMS coordinate-wise, producing a sign-like update regardless of upstream
scaling. Whatever S_B⁻¹ did to ∇A is a per-coordinate rescaling that v̂
proceeds to *undo* by normalizing each coordinate to ~unit magnitude. The
result is that pre-precondition compositions are ε-perturbed AdamW by
construction.

---

## H4 — productive change attempt: reorder to Adam → S⁻¹

**Motivation.** If the issue is that v̂ erases geometry installed *upstream*,
the obvious fix is to install geometry *downstream*: run Adam on raw ∇,
then apply Sylvester / Gram solve to the Adam step. v̂ adapts to the natural
gradient distribution; geometry installs the (A,B) coupling on top of it.

**Implementation.** `AdamLinLoRAPost`, `AdamScaledLoRAPost`. Reuse
`solve_sylvester` / `solve_spd` / `spdify` from `lora_playground/utils.py`.
13 unit tests pass (shapes, dtype, zero-grad no-update, determinism, post
≠ pre).

**In-flight result (step 1400, η=1e-3, the headline cell):**
- adam-lin-lora-post: **0.7923** (vs AdamW 0.7579 at step 1400 → loss is
  trending to ~0.78–0.79 final, ~0.03 worse than AdamW).
- adam-scaled-lora-post: **0.8421** (decisively worse).

**H4 looks falsified.** Mechanism: applying S_B⁻¹ to a sign-like Adam step
does not produce a useful direction. The Gram inverse only carries useful
information when applied to a vector that lives in the gradient's spectrum
(i.e. has the same per-coord scale structure); on a sign vector,
S_B⁻¹ is a mostly-isotropic rotation that hurts more than it helps.

**Compare to adam-muon-lora.** Same composition order (Adam → spectral
correction), same expected mechanism, but **Newton-Schulz wins where
S_B⁻¹ loses**. The difference: NS is a *spectral cap* — it equalizes
singular values across the rank-r factor matrix, which is meaningful even
on a sign vector. S_B⁻¹ is a *curvature-aware rescaling*, which requires
the input to *have* curvature to rescale.

→ **Provisional rule:** post-Adam corrections work iff they're
*structurally meaningful on a sign-magnitude input*. NS qualifies.
S⁻¹ does not.

---

## H3 — small-r hypothesis falsified

**Premise:** at small r, S_A and S_B are smaller and likelier ill-conditioned,
so the geometric correction should matter more and the gap vs AdamW should
widen.

**In-flight data (step 2000 at η=3e-4):**

| r  | adamw  | adam-lin-lora | gap (lin − adamw) |
|----|--------|---------------|--------------------|
| 2  | 0.7920 | 0.8150        | **+0.023** (worse) |
| 4  | 0.7807 | 0.8024        | **+0.022** (worse) |
| 16 | 0.7795 | 0.7723*       | −0.007 (better)    |
| 64 | 0.7550 | 0.7527        | −0.002 (slightly better) |

*r=16 numbers from `lr_sweep_2k`, η=1e-4 best for adamw, η=1e-3 best for lin.

**Premise falsified.** At small r, lin-lora is *worse* than AdamW by ~0.02 nat.
At large r, the gap narrows (lin slightly better). H1 already explained why:
the conditioning of S_B is dominated by ‖B‖² (init-scale issue), not by r.
Small r doesn't make S_B more ill-conditioned in a way that the geometric
correction can exploit.

**Side benefit:** clean adamw r-scan (0.792 → 0.781 → 0.755) — confirms
"more rank = better" with the standard scaling shape.

---

## H5 — per-pair scalar v̂ (matrix-Adam)

**Motivation.** Replace per-coord v̂ with one scalar EMA per LoRA pair, so
v̂ rescales the Adam step by *magnitude only*, leaving direction untouched.
If H1 is right that per-coord normalization is the eraser, this should
preserve the geometric direction.

**First attempt (job 6312354): broken.** Tracked Σ g² (sum) instead of
mean — √v̂_pair ≈ √N · RMS(g) → effective lr = lr/√N ≈ lr/700 → no learning
at the standard η range. eval stayed at 1.187 (random init) for 1k+ steps.

**Fix (commit ac81bba):** divide by N_total = numel(A) + numel(B) before
EMA so v̂ tracks the mean square. √v̂_pair has units of |g|, matching
per-coord Adam. Verified: η=1e-3 step 50 → eval 0.886 (vs broken: 1.187 at
1000 steps).

**Resubmitted as job 6312759** (4 GPUs, 4h). Result pending.

---

## Mechanistic summary diagram

```
gradient ∇  ──┬──> Adam(m,v) ──> sign-like Δᴬ ──┬──> spectral cap (NS)  ──> 0.7557 ✅
              │                                  │
              │                                  ├──> Sylvester / S⁻¹     ──> 0.79+ ❌
              │                                  │   (Adam → S⁻¹: H4 falsifying)
              │                                  │
              │                                  └──> per-pair scalar v̂   ──> tbd
              │                                      (H5, resubmitted)
              │
              ├──> S⁻¹ ──> precond ──> Adam(m,v) ──> 0.7564–0.7572 ≈ tied ⚠
              │           (geometry → Adam: H1 explains why ≈)
              │
              ├──> NS only ──> 0.7675 ❌ (NS without diagonal preconditioning)
              │
              └──> diag K-FAC + various ──> 0.81–0.83 ❌
```

**Win condition (empirical so far):** Adam-step → spectral correction.
**Loss conditions:** geometry-then-Adam (cos converges to AdamW),
spectral-only (loses Adam's diagonal preconditioning),
Adam-step → Gram-inverse (the inverse rescaling of a sign vector is
structurally meaningless).

---

## Parallel work — relevant arxiv papers (downloaded to docs/papers/)

### AdaMuon (arxiv 2507.11005v3, SJTU + Xiaohongshu, Dec 2025)

The closest existing variant to our `muon-adam-lora` (currently failing on
our LoRA setup) — but with three details we got wrong, any one of which
could explain the gap. Algorithm 1 from the paper:

```
M_t = β·M_{t-1} + G_t                        # plain SGD momentum
O_t = NewtonSchulz(sign(M_t), T)             # NS on SIGN of momentum
V_t = β·V_{t-1} + (1−β)·O_t ⊙ O_t            # element-wise v on NS output
Õ_t = O_t ⊘ (√V_t + ε)                       # variance-adapt
γ_t = 0.2·√(mn) / ‖Õ_t‖_F                    # RMS-align to Adam magnitude
W_{t+1} = W_t − η·(γ_t·Õ_t + λ·W_t)
```

Three differences vs our `muon-adam-lora` (NS → Adam):
1. **sign(M) before NS** — stabilizes NS input so post-NS magnitudes are
   bounded. We feed raw ∇ (or raw M) into NS.
2. **v on O_t only (not full Adam)** — they don't run Adam's m on the NS
   output, just second-moment normalization. We do full Adam(m,v).
3. **RMS-aligned step** with γ_t — sets ‖step‖_F to a target so Adam's lr
   transfers. We don't.

Claim: 40%+ training-efficiency gain over Adam at pretraining scale.

**Implication for us:** "NS → Adam" can work, our implementation is
underpowered. A faithful AdaMuon port is a clear next experiment.

### NorMuon (arxiv 2510.05491v1, Georgia Tech + Microsoft, Oct 2025)

Different diagnosis, same broad mechanism. Observation: after NS,
*singular values* of the update matrix are equalized (low matrix condition
number) but *per-row L² norms* still have high variance — some neurons
dominate. Fix: per-neuron (row-wise) second-order adaptive learning rates
on top of Muon orthogonalization.

This is the third granularity point on the variance-tracking axis we
haven't tried:

| granularity        | example optimizer      | adapts                    |
|--------------------|------------------------|---------------------------|
| per-element        | adam-muon-lora, AdaMuon| each parameter coord      |
| **per-row/neuron** | **NorMuon**            | each output unit          |
| per-pair (matrix)  | adam-lin-lora-matrix   | (A, B) pair (one scalar)  |

For LoRA factors A: (r, d_in), B: (d_out, r) the row dimensions are r and
d_out — so per-row Adam on A is per-LoRA-direction, per-row Adam on B is
per-output-neuron. Cheap to add to our framework.

Claims: 21.74% over Adam, 11.31% over Muon at 1.1B pretraining.

### Caveat on transfer

Both papers measure pretraining efficiency on 1.1B-class models with
hundreds of billions of tokens. We measure final eval-loss after a 2k-step
LoRA fine-tune (effectively ~1M tokens). The *direction* of the gain
(Muon-family ≥ Adam) should transfer; the *magnitude* of the gain almost
certainly does not — fine-tuning regimes are dominated by Adam's variance
adaptation in ways pretraining is not. Useful for design-space mapping,
not for predicting headline numbers.

## A/B asymmetry — refining H1 with rank-sweep diagnostics

H1 ran at r=16 and reported cos(geometric step, plain-AdamW step) saturated
near 0.97 throughout training. The H1 r-sweep (job 6313087, in flight)
extends the cos probe to r ∈ {4, 64} and surfaces a sharper picture:

| r  | cos_A_median (early) | cos_A_median (late) | cos_B_median  | σ_min(S_B) early |
|----|----------------------|----------------------|----------------|--------------------|
| 4  | 0.62 (step 20)       | 0.66 (step 520)     | **0.999** flat | 0.005              |
| 16 | 0.46 (step 20)       | 0.84 (step 1500)    | 0.94–0.98     | 0.011              |
| 64 | 0.36 (step 20)       | 0.72 (step 400)     | 0.97–0.98     | 0.0006 (smaller!)  |

**Two new mechanistic claims:**

1. **cos_B saturates near 1 at every r** (0.97 to 0.999). The B-direction of
   the geometric step is approximately AdamW's B-direction at every rank we
   measured. Whatever advantage the geometric correction provides at large r,
   it does NOT come through B. The geometric work is happening on A only.

2. **cos_A is rank-dependent but non-monotonically informative.** At r=4
   cos_A is *lower* (more different from AdamW) than at r=16 or r=64, yet
   r=4 LOSES to AdamW by +0.02. So "different direction" ≠ "better
   direction" — at small r the geometric step is plausibly randomly-rotated
   noise (S_B near δI for many steps because B is tiny rank, rotation
   informationless). At large r the direction differs *and* helps.

3. **Conditioning is NOT the lever.** σ_min(S_B) is *worse* at r=64 (0.0006)
   than at r=4 (0.005) — because S_B has more eigenvalues at high r and the
   smallest one tends to be smaller. Yet r=64 wins. The "well-conditioned
   geometry helps" intuition is wrong; what helps is *the dimension of the
   subspace the geometric solve has to install structure in*.

## H_weak vs H_erase — open mechanism question

If cos_B ≈ 1 at every r, two failure modes for the B-side correction:

- **H_weak**: S_A⁻¹ is approximately identity-like on ∇B, so the geometric
  rotation barely happens before Adam runs (cos_pre_B ≈ 1).
- **H_erase**: S_A⁻¹ rotates ∇B meaningfully, but Adam's per-coord √v̂
  erases the rotation post-hoc (cos_pre_B < 1, cos_post_B ≈ 1).

Distinguishable with one new probe: cos(precond_B, ∇B) measured *before*
Adam touches it. Added to the diagnostic logger (commit 1c74ba5). Submitted
fast probe at 500 steps × 2 optimizers (job 6313190) — answer at step 20.

If H_weak: B-side geometry adds nothing, *not because Adam is too aggressive
but because the geometry itself doesn't rotate ∇B in this regime*. Implies
the productive change is "geometry on A only, plain Adam on B" — gives the
same result as the full geometry-then-Adam compositions, with half the
preconditioning cost.

If H_erase: Adam's √v̂ is the culprit on B as it is on A. Then the
"preserve direction by avoiding per-coord v̂" fix family (matrix-Adam,
RMS-align) should produce different B-side directions too. We'd predict
H5's matrix-Adam variant (scalar v̂ per pair) to recover B-side geometric
direction.

## Trajectory variance — the r=64 "win" is at noise floor

Pulled the eval_loss trajectory of all three optimizers at r=64 η=3e-4 and
computed late-trajectory step-to-step |Δ|:

- pooled mean step-to-step |Δ_loss|: **0.0038**
- max: 0.0051

Final-step gaps at step 2000:
- adam-scaled-lora vs adamw at r=64: **−0.0044** ← within noise
- adam-lin-lora vs adamw at r=64: −0.0023 ← below noise

The *direction* of the rank-dependence effect is unambiguous (gap goes from
+0.02 at r=2 to −0.005 at r=64), but the *magnitude* of the win at r=64 is
within the noise floor of a single seed. Multi-seed would establish
significance but is the wrong tool — mechanism evidence (H_weak vs H_erase
verdict, cos_pre values, ablations of A-only / B-only geometry) is more
informative per GPU-hour than confidence intervals on a 0.005 gap.

## Open questions

1. **Does H5 (matrix-Adam) save the geometry → Adam family?** With per-pair
   scalar v̂, direction is preserved by Adam — so the cos_A divergence H1
   measured should be more meaningful. But matrix-Adam discards the
   per-coord stability adjustments that make Adam robust; might be
   numerically fragile. Pending.

2. **AdamProductMuonLoRA** (queued, 6312406): combines product-Muon
   geometry (gauge-invariant Sylvester) with Adam preconditioning. If
   product-Muon has the right *direction* and Adam supplies *magnitude*,
   this could compound. Pending.

3. **Higher rank** (r=64) gives 0.7527 / 0.7550 with simple optimizers. Is
   the optimizer-design effort worth it vs just paying for more LoRA
   capacity? 4× more LoRA parameters at r=64; comparable to a rank-16
   train at 2× lr+steps. Worth a budget-matched comparison.

4. **Longer training.** adam-muon-lora at step 2000 was still descending
   (step 1800→2000 dropped 0.002). Is 0.7557 a transient lead or a final
   one? 4k or 8k step rerun would say.

5. **Why is adam-scaled-lora-post much worse than adam-lin-lora-post**
   (0.84 vs 0.79)? Both apply the same composition order swap. The Sylvester
   correction (lin) preserves more information about the (A,B) coupling than
   the simpler S_B⁻¹ Gram solve (scaled). The "post" composition seems to
   *amplify* the lin-vs-scaled gap that was nearly invisible in the
   pre-Adam form.

---

## Cross-references

- **Lin/scaled-lora investigation** (H1–H5, this work): `docs/notes/lin_scaled_lora_investigation.md`
- **Muon-LoRA "beat AdamW" campaign** (H1–H4, completed): `docs/notes/muon_beat_adamw_investigation.md`
- **Theory** (Sylvester preconditioner, spectral product norm): `docs/theory/main.tex`
- **Synthetic motivation** (low-rank matrix recovery): `notebooks/low_rank.ipynb`
- **Sweep analysis notebook** (all baselines): `notebooks/sweep_analysis.ipynb`
- **Lin/scaled investigation notebook** (H1 cosine plots, H4/H5 leaderboard): `notebooks/lin_scaled_investigation.ipynb`
- **Memory:**
  - `~/.claude/projects/-mnt-home-nghosh-lora/memory/project_optimizer_sweep.md`
  - `~/.claude/projects/-mnt-home-nghosh-lora/memory/project_muon_campaign.md`
  - `~/.claude/projects/-mnt-home-nghosh-lora/memory/feedback_beat_dont_match.md`
- **Optimizer code:** `lora_playground/optim.py` (all 17 registered optimizers)

---

## Running experiments (as of 2026-04-30 ~13:00 local)

| job     | group                    | runs   | state         | hypothesis / role                      | verdict so far |
|---------|--------------------------|--------|---------------|----------------------------------------|----------------|
| 6312334 | h1_diag_2k               | 2/2    | DONE          | H1 v̂ erases geometry @ r=16           | **confirmed**: cos_B ≈ 0.97 throughout, cos_A → 0.84 |
| 6312277 | h4_post_2k (unfixed)     | 10/10  | DONE          | H4 *-Post wins (unfixed)               | **falsified**: best 0.7875 at η=1e-3, 0.03 worse than AdamW |
| 6312335 | h3_rsweep_2k             | 12/18  | RUNNING (η=1e-3) | H3 small-r benefit                  | **falsified**: lin loses at r=2,4; **r=64 marginally wins** (within noise floor) |
| 6312759 | h5_matrix_2k (fixed)     | 4/10   | RUNNING       | H5 scalar v̂ preserves direction        | learning now (was 1.187 before fix); high-η runs pending |
| 6313020 | h4_post_rmsalign_2k      | 4/4 step ~1000 | RUNNING | RMS-align fix for *-Post              | **trajectory matches AdamW pace** at step 1000; final pending |
| 6313087 | h1_rsweep_diag_2k        | 4/4 step ~200-400 | RUNNING | cos at r=4, r=64 (mechanism)        | r=4 cos_A=0.62, r=64 cos_A=0.36→0.72; cos_B≈1 at all r |
| 6313190 | h1_pre_probe_500         | 2/2 (just submitted) | PENDING | H_weak vs H_erase                  | answer at step 20 |
| —       | adam_muon_2k             | done   | DONE          | Muon campaign winner                   | **0.7557**, current overall #2 |
| —       | (none — new from H3)     | done   | DONE          | adam-scaled-lora at r=64               | **0.7506**, current overall #1 (within noise) |

---

## What I'd do next, ranked

1. **Wait for H4 to finish.** If best `*-post` final ≥ 0.78 across all η, H4
   is decisively falsified — close the file with one sentence updating the
   doc. If unexpectedly some η hits 0.756 territory, redo with a finer η grid.
2. **Wait for H5 (resubmitted).** If matrix-Adam beats AdamW: that's a real
   second productive variant alongside adam-muon-lora. If not: matrix-Adam
   joins H4 in the "ideas that sounded right but didn't help" pile.
3. **Budget-matched r=16 vs r=64 study.** adamw r=64 → 0.7550 with 1× compute
   beats adam-muon-lora r=16 with custom optimizer. Are we optimizing the
   wrong axis?
4. **adam-muon-lora longer training / lr decay.** The 0.7557 was still
   descending at step 2000. A 4k-step run with cosine decay could push to
   0.74-ish — a real headline.
5. **Closed-form polar update** (theory line 656): the unique closed-form
   spectral-product update. Cheaper than ProductMuon (one extra polar
   per pair per step) and theoretically motivated. Not yet implemented.
