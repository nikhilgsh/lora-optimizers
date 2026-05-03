# Experimental progress log

Live log of optimizer-comparison experiments on OLMo-2-0425-1B + Magicoder, r=16, 2000 steps, seed=0.
Each entry: motivation → action → result → next step. Newest entries on top.

---

## 2026-05-01 — Stale-preconditioner Phase 1 validation (r=64, r=256)

**Motivation.** Plan in `docs/plans/stale_preconditioner_speedup.md` proposes
caching `S_A^{-1/2}, S_B^{-1/2}` for K steps to amortize per-step preconditioner
cost. Validate that final eval loss survives staleness, separately at r=64
(the rank where we already have a non-coupled baseline) and r=256 (where the
speedup matters most).

**Sweeps.** `polar_K_sweep_r64_2k`, `polar_K_sweep_r256_2k`, `polar_K_higham_r256_2k`.
Single-seed, lr=3e-4, optimizer `adam-polar-product-lora` (non-coupled), 2000 steps.

**r=64, eigh, K∈{1,2,5,10,20}:**

| K  | eval_loss |
|----|-----------|
| 1  | 0.7454 |
| 2  | 0.7453 |
| 5  | 0.7455 |
| 10 | 0.7459 |
| 20 | 0.7462 |

Δ(K=1 → K=20) = +0.0008.

**r=256, eigh, K∈{1,5,10,20}:**

| K  | eval_loss | step |
|----|-----------|------|
| 1  | 0.7502 | 1820 (SLURM time-limit, not converged at boundary) |
| 5  | 0.7497 | 2000 |
| 10 | 0.7568 | 2000 |
| 20 | 0.7596 | 2000 |

**r=256, higham (Phase 2 prototype), K∈{1,5}:**

| K | eval_loss | step |
|---|-----------|------|
| 1 | — | crashed at step 800 in diagnostic probe (see below) |
| 5 | 0.7504 | 2000 |

**Findings.**
- At r=64, eval loss is essentially flat across K∈[1,20]; staleness up to
  K=20 is harmless at this rank, lr, and seed.
- At r=256, the picture changes: K=5 matches K=1 within the incomplete-K=1
  data; K=10 is +0.0071 worse than K=5, K=20 is +0.0099 worse. The break is
  between K=5 and K=10 at this rank.
- higham vs eigh at r=256, K=5: 0.7504 vs 0.7497, Δ=+0.0007. Quality
  indistinguishable at this single comparison point. Wall-time speedup not
  measured cleanly here.

**Probe crash on higham K=1.** The optim-diagnostics probe called
`torch.linalg.eigvalsh(BᵀB)` on a 256×256 Gram matrix that became
near-degenerate, raising `_LinAlgError: code 257`. The polar update math
itself was healthy at step 800 (eval_loss=0.7830 mid-trajectory). Patched in
this commit: `_spd_eig_extremes` is now NaN-tolerant on `LinAlgError`, and
the two factor-Gram call sites in `AdamPolarProductLoRA` and
`AdamuonPolarProductLoRA` use `svdvals(X)**2` (numerically more accurate
for `λ_min` of `XᵀX` than `eigvalsh` on the formed Gram).

**Next step.** Rerun K=1 r=256 cells (eigh + higham) with the probe fix and
a 4h time limit — submitted as group `polar_K1_r256_rerun_2k` (SLURM job
6315528). Closes the K=1 data point at r=256.

---

## 2026-04-30 — η-bracketing for scaled-lora, lin-lora, diag-scaled-lora

**Motivation.** Original `lr_sweep_2k` topped out at η=1e-3 with these three optimizers
all at the boundary. Wanted to characterize the actual scale of η they need.

**Sweeps.** boundary_extend_2k (η ∈ {3e-2, 1e-1, 3e-1}), boundary_extend2_2k (η ∈ {1, 3}),
boundary_extend3_2k (η ∈ {10, 30} for lin/scaled only since diag diverged at η=1).

**Final η-vs-loss (step 2000):**

| optimizer        | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 | 1.0 | 3.0 | 10  | 30  |
|------------------|------|------|------|------|------|-----|-----|-----|-----|
| diag-scaled-lora |0.880*|0.870*|0.815 |0.791 |**0.790**|6.14|8.03 | -   | -   |
| lin-lora         |0.886*|0.846*|0.846 |0.836 |0.822 |0.803|**0.778**|0.793|0.824|
| scaled-lora      |0.897*|0.874*|0.856 |0.837 |0.819 |0.795|**0.771**|0.790|1.179|

(* from earlier sweeps with the same optimizer at lower η)

**Findings.**
- All three need η orders of magnitude higher than the AdamW LoRA optimum (3e-4)
- diag-scaled-lora peaks at η=3e-1 → 0.790, diverges at η=1
- lin-lora peaks at η=3.0 → **0.778**, mild interior optimum
- scaled-lora peaks at η=3.0 → **0.771**, interior optimum, only 1.7% behind AdamW

**Headline.** The simple Sylvester-coupled preconditioned methods (`scaled-lora`,
`lin-lora`) — no momentum, no Adam-style √v adaptation, just per-factor (BᵀB + δI)⁻¹
or its linearized variant — get within 1.7% of AdamW LoRA on this task. The (A,B)
coupling does most of the work that Adam's momentum/√v normally does, but the η
needs to be 4 orders of magnitude larger than the Adam regime. This is a strong
"interpretability" point: the optimizer doesn't need to be fancy if the geometric
preconditioning matches the LoRA parametrization.

---

## 2026-04-30 — PSI-LoRA crashes on small η: cholesky escalation insufficient

**Motivation.** Resubmitted PSI-LoRA sweep (job 6312780) with the cholesky escalation
fallback added to `_solve_ridge`. 4 of 6 runs still crashed with the same singular-matrix
error in the lorsum momentum update at `B_t = _solve_ridge(A_t @ A_t.T, ...)`. CPU
repro of the same shapes shows the gram min eigenvalue is ~1500 (well-conditioned), so
something GPU/bf16-specific produces an indefinite gram that doesn't recover even with
ε bumped to 1.0.

**Status.** Open. Need to instrument the failing call to dump the offending matrix and
inspect on GPU. Tabled while we extend lin/scaled-lora — those have higher expected
payoff (now at 0.771-0.778, competitive with AdamW; PSI-LoRA was at 0.892 even when
running, well behind).

---

## 2026-04-30 — PSI-LoRA momentum: code matches paper Algorithm 3 (with the right reading)

**Motivation.** With α₁=0 case behaviorally verified, extended the test to α₁=0.9
(paper's RoBERTa GLUE setting) to upgrade reproduction confidence to "matches reference
in the canonical config used to produce the published numbers."

**Discrepancy investigated.** Paper Algorithm 3 line 5 reads
`Ŵ = U V^T − η Sᵀ X − η α₁ M(G)`  (gradient coeff −η, momentum coeff −η·α₁ — sum form).
Reference code (`~/PSI-LoRA/src/oplora/optimizer.py:1053`) uses
`coefficients=[1.0, -lr*(1.0 - beta1), -lr*beta1]`  (gradient coeff −η·(1-α₁),
momentum coeff −η·α₁ — convex-combination form). These are not the same formula.

**Resolution: follow the code, not the paper box.** The convex-combination form is the
standard "SGD with EMA momentum" identity (`step = η · m_{t+1}` where
`m_{t+1} = α₁·m_t + (1-α₁)·g`). It's also what produces the paper's numbers. The
algorithm box is most plausibly a notational slip. Updated PSILoRA's coefficients
to match the code: `[1.0, -lr·(1-α₁), -lr·α₁]`.

**Other diffs found while extending the test.**
- Reference inits the momentum buffer's "in" side via `torch.randn` (not scaled), zeros
  on "out". Mine had `randn * 0.01`. Now matches the reference (unscaled randn).
- Reference guards the momentum-buffer LoRSUM update with `if beta1 > 0.0` (skip update
  when α₁=0). Mine ran the update unconditionally. Now matches the guard.

**Verification.** Both α₁=0 and α₁=0.9 (paper GLUE default) configs pass:
- α₁=0: step 1 max |Δw| = 3.7e-9; step 5 max = 2.0e-5
- α₁=0.9: step 1 max |Δw| = 3.7e-9; step 5 max = 3.1e-5

Both well under 1e-4 tolerance. Test committed at `scripts/verify_psilora_against_official.py`.

**Resubmitted.** Job 6312401 (`psi_lora_2k`) with all PSI-LoRA fixes: lr-scaled ρ +
proximal stability eps + lmbd clamp + momentum init + convex-combination coefficients.

---

## 2026-04-30 — PSI-LoRA behavioral equivalence test, found two more bugs

**Motivation.** After the GaLore behavioral test caught a real bug, ran the same exercise on PSI-LoRA against `~/PSI-LoRA/src/oplora/optimizer.py:ScaledOPLoraOptimizer` (diagonal K-FAC mode, the canonical Algorithm 3 config per `conf/optimizer/scaled_oplora.yaml`).

**Found two more bugs.** With α₁=0 (momentum off, isolating the F-LoRSUM math), my port produced A_new = A_t exactly (no shrinkage with zero gradient + B=0 init). Reference produced A_new = 0.909·A_t. Tracing the discrepancy:
1. **`_solve` adds an extra cholesky stability eps** (1e-6) on top of the proximal lmbd. So the effective LHS in the proximal solve is `gram + (lmbd + 1e-6)·I`, not just `gram + lmbd·I`. My port used `eps=lmbd` for both purposes, making them cancel in the rank-deficient case.
2. **`lmbd = max(lmbd, 1e-5)` clamp** in the optimizer step (line 973). This makes the proximal regularizer floor at 1e-5 even at small lr where the lr-scaling would have made it tiny.

These produce a small per-step shrinkage of weights when the gram is rank-deficient (early in training while B is small). With B=0 init, the shrinkage is ~9% on A per step — material, not numerical noise.

**Fix.** `_solve_ridge` now takes `eps=lmbd + 1e-6` to match the ref's stability behavior. PSILoRA.step clamps `rho = max(lr * proximal_rho, 1e-5)`.

**Verification.** With α₁=0, single-step weight diff is bit-exact (max |Δw| = 0.0). 5-step run drifts to ~2e-5 max (float32 numerical sensitivity in nested ALS solves). Both well under tolerance.

**Caveat.** Test only validates α₁=0 case. With momentum on, there's a paper-vs-ref-impl discrepancy in the gradient-vs-momentum coefficient weighting (paper: `-η·g, -η·α₁·m`; ref: `-η·(1-α₁)·g, -η·α₁·m` — convex combination instead of sum). My code follows the paper. Resolving this requires reading the paper's appendix carefully or asking the authors. For now, test scope is the F-LoRSUM math + diagonal K-FAC stats, not the momentum convention.

**Resubmitted.** Job 6312353. Test committed as `scripts/verify_psilora_against_official.py`.

---

## 2026-04-30 — GaLore behavioral equivalence verified, found off-by-one bug

**Motivation.** User flagged that the 7% gap between our GaLore (0.811) and AdamW LoRA (0.758) was concerning, and asked whether I'd actually run the official codebase to verify behavioral equivalence (I had only code-reviewed it).

**Action.** Wrote `scripts/verify_galore_against_official.py`: tiny model with both tall and wide linears, deterministic gradients, run N steps with our `GaLoreAdamW` and the official `~/GaLore/galore_torch.AdamW`, compare weight trajectories.

**Found a behavioral divergence.** Steps 1–4 matched (max |Δw| < 1.9e-9, float32 noise). Step 5 diverged sharply (4e-4) and grew to 1.6e-3 by step 12.

**Root cause.** Step counter offset. Official: `state["step"] = 0` initially → project uses `iter=0` (init refresh) → `step += 1` → bias correction uses post-increment step. So `iter` passed to projection is `0,1,2,...` and refresh fires at `iter ∈ {0, gap, 2·gap, ...}`. My port did `step += 1` first, then checked `step % gap == 0` for refresh — fires at `step ∈ {1, gap, 2·gap, ...}`, i.e. **off by one starting at iter=gap**.

**Fix.** Project before incrementing step. After fix, behavioral equivalence to 1.86e-9 across 12 steps spanning 2 refreshes — passes 1e-5 tolerance.

**Resubmitted.** `galore_fixed_2k` job 6312228 (the 0.811 result was from the off-by-one version). Also adds genuine "behavioral match" to the reproduction scorecard.

**Lesson.** Code-review match ≠ behavioral match. The off-by-one was easy to miss visually because both versions look "correct" — it took a numerical comparison to surface it.

---

## Open questions / ongoing

- **PSI-LoRA reproduction performing poorly** (best 1.04 at η=1e-2, lr-pinned) — submitted job 6312167 with the lr-scaled ρ fix; results pending. If still poor, suspect deeper bug in F-LoRSUM port or hyperparameter mismatch (K, ρ, momentum_rank).
- **GaLore underperforms LoRA AdamW by ~7%** (0.811 vs 0.758). Code-reviewed against `~/GaLore` and matches; have NOT run their codebase end-to-end to verify behavioral equivalence. Levers untried: rank > 16, shorter `update_proj_gap`, different `proj_type`. Paper claims competitive on GLUE, not strictly better.
- **Boundary-pinned optimizers** (`diag-scaled-lora`, `lin-lora`, `scaled-lora` all at top η) — extension job 6312193 in flight (η ∈ {3e-2, 1e-1, 3e-1}).

---

## Standings (final eval loss at step 2000, all `r=16`)

| optimizer        | best η | eval loss | gap-to-AdamW | status |
|------------------|--------|-----------|--------------|--------|
| adam-lin-lora    | 1e-3   | **0.7581**| +0.0%        | interior peak |
| adam-scaled-lora | 1e-3   | 0.7572    | -0.1%        | interior peak |
| adamw            | 3e-4   | 0.7579    | (baseline)   | baseline |
| muon-lora        | 3e-3   | 0.7675    | +1.3%        | interior peak (post-NS-fix) |
| scaled-lora      | 3.0    | 0.7706    | +1.7%        | interior peak (η 4 orders above Adam regime) |
| lin-lora         | 3.0    | 0.7776    | +2.6%        | interior peak |
| kron-grad-lora   | 1e-3   | 0.7850    | +3.6%        | interior peak |
| diag-scaled-lora | 3e-1   | 0.7901    | +4.2%        | interior peak |
| galore-adamw     | 3e-4   | 0.8112    | +7.0%        | interior peak (post off-by-one fix) |
| psi-lora         | 3e-3   | 0.8923    | +17.7%       | only 1/6 runs converge; small-η crashes |

Reference: PEFT default initialization (B=0, A=Kaiming). All step-2000 numbers; eval set = 512 samples.

---

## 2026-04-30 — PSI-LoRA lr-scaled-ρ fix

**Motivation.** Initial `psi-lora` sweep showed η-insensitive loss at small η (3e-5..1e-3 all ≈ 1.186, only 1e-2 broke out to 1.045). Same pathology as the original Muon-LoRA NS bug.

**Root cause.** My port of F-LoRSUM hardcoded the proximal regularizer as `lmbd=self.proximal_rho` regardless of η. The reference (`~/PSI-LoRA/src/oplora/optimizer.py:27`, `LR_LMBD=True`) always passes `lr * lmbd` into LoRSUM. When η is small (3e-4) and ρ is fixed (0.01), the proximal pull dominates the gradient term and updates collapse.

**Fix.** `effective_rho = lr * self.proximal_rho` in `PSILoRA.step()`.

**Submitted.** Job 6312167, `psi_lora_2k` resubmit.

---

## 2026-04-30 — Boundary extension for top-pinned optimizers

**Motivation.** Three optimizers' best η is at the top of the swept range:
- `diag-scaled-lora` η=3e-2 → 0.8153
- `lin-lora` η=1e-2 → 0.8457
- `scaled-lora` η=1e-2 → 0.8744

Need to know if these continue improving or peak somewhere in {3e-2 .. 3e-1}.

**Submitted.** Job 6312193, params `boundary_extend_2k.json`: η ∈ {3e-2, 1e-1, 3e-1} × {diag, lin, scaled}-lora, 9 runs.

---

## 2026-04-30 — Honest renames + faithful GaLore + PSI-LoRA Algorithm 3 port

**Motivation.** Two existing optimizers (`psi-lora`, `kfac-lora`) were mislabeled; they didn't implement what the names suggested. Also our GaLore implementation had bugs vs the official `~/GaLore`.

**Renames.**
- `PSILoRA` → `DiagScaledLoRA` — was applying diagonal D_V/D_U scaling to A and B grads independently. Missing the F-LoRSUM proximal subspace iteration, low-rank momentum, and U/V coupling that define the paper's Algorithm 3.
- `KFACLoRA` → `KronGradLoRA` — was a custom variant adding r×r gradient outer product factors on top of the diag scaling. Not from any paper.
- `psi-lora` slot is now the genuine Algorithm 3 port.

**GaLore rewrite (matches `~/GaLore/galore_torch/{adamw,galore_projector}.py`):**
- proj_type="std" axis selection: right projection for tall (d_out ≥ d_in), left for wide (d_out < d_in)
- exact `torch.linalg.svd`, not `torch.svd_lowrank`
- Adam moments **persist** across projection refresh (was being reset every 200 steps — destroyed β₂=0.999 statistics)
- eps default 1e-6, scale=1.0 (the GLUE fine-tuning default; pretraining uses 0.25)

**PSI-LoRA port (paper Algorithm 3 / `~/PSI-LoRA/src/oplora/utils.py`):**
- New utilities: `lorsum` (eq. 10) and `f_lorsum` (eq. 14) in `lora_playground/utils.py`
- New `PSILoRA` class composes dense step proposal as 3 low-rank factors:
  `(A, B)` prox center · 1, `(X, Sᵀ)` gradient · −η, `(M_A, M_B)` momentum · −η·α₁
  then F-LoRSUM K-ALS iterations under K-FAC metrics → new (A, B);
  separate LoRSUM updates the low-rank momentum buffer.
- Tests verify LoRSUM converges to truncated SVD (rel err < 1e-3), F-LoRSUM ≡ LoRSUM with unit metrics.

**Result.** GaLore improved from buggy 0.8417 → fixed 0.8112 (still 7% behind AdamW LoRA). PSI-LoRA initially poor (lr-pinned at η=1e-2) — root cause and fix described above.

**Commits.** 93ae143 (renames + Muon NS + GaLore), 61de8a2 (GaLore sweep config), b5e5bc0 (PSI-LoRA port), 27d05f0 (notebook consolidation).

---

## 2026-04-30 — Notebook consolidation

**Motivation.** 5 floating sections with redundant "all runs (color=opt, alpha=η)" plots had become unreadable as more optimizers were added.

**Action.** Merged "Optimizer comparison" + "New optimizers" + "GaLore" into a single "All optimizers — η sweep" section. Replaced messy line-spaghetti panels with `η vs final eval loss` (log-x) plots — one line per optimizer. Right panel still shows training curves for the best η per optimizer. Same style applied to LoRA+ and SVD oracle sections.

**Result.** 3 sections, each a single 2-panel figure. 10 optimizers on a single axis.

---

## 2026-04-30 — Muon-LoRA NS scale-invariance bug

**Motivation.** Muon-LoRA showed η-insensitivity at all swept lrs (3e-5..1e-3 all gave ≈1.186 final). Same pathology that PSI-LoRA later showed.

**Root cause.** `_newton_schulz` was re-multiplying its output by the input Frobenius norm, breaking Muon's scale-invariance. Combined with PEFT's B=0 init, the gradient `‖G_A‖ ≈ 0` early in training → updates collapsed regardless of lr.

**Fix.** NS output now has Frobenius norm √r independent of input scale (canonical Muon, matches `modded-nanogpt`). Tests added: `test_newton_schulz_{short_fat,tall_skinny,scale_invariant_update}`.

**Result.** η sensitivity restored. Muon-LoRA best at η=3e-3 → 0.7675 (1.5% behind AdamW; competitive but not winning).

**Sweeps.** `muon_lowlr_2k` (η ∈ {3e-5..1e-3}), `new_optimizers_high_eta_2k` (η ∈ {3e-3..3e-2}).

---

## Earlier (pre-log)

Original sweeps: `lr_sweep_2k` (5 LoRA-mode optimizers × 4 lrs), `optim_compare_high_eta_2k` (extension to η ∈ {3e-3, 1e-2}), `loraplus_2k_1ep` (η × m co-sweep for AdamW+LoRA+), `svd_sweep_2k_1ep` (SVD oracle modes), `new_optimizers_2k` (initial buggy PSI/Muon/KFAC), `galore_2k` (initial buggy GaLore).

---

## Honest scorecard on reproductions

| reproduction | code-review match | end-to-end behavioral match | reference |
|--------------|-------------------|------------------------------|-----------|
| Muon NS      | ✓ canonical Muon  | tests verify scale-invariance| modded-nanogpt |
| GaLore       | ✓ vs `~/GaLore`   | ✓ behavioral match to 1.86e-9 over 12 steps + 2 refreshes (`scripts/verify_galore_against_official.py`) | jiaweizzhao/GaLore |
| PSI-LoRA     | ✓ vs `~/PSI-LoRA` | ✓ behavioral match for α₁=0 AND α₁=0.9 (paper GLUE default) — bit-exact single step, ~3e-5 cumulative drift over 5 steps (`scripts/verify_psilora_against_official.py`). | zeligism/PSI-LoRA |

If we want strong claims about reproduction, we need to run the reference codebase on equivalent settings and compare outputs.
