# Experimental progress log

Live log of optimizer-comparison experiments on OLMo-2-0425-1B + Magicoder, r=16, 2000 steps, seed=0.
Each entry: motivation → action → result → next step. Newest entries on top.

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

| optimizer        | best η | eval loss | status |
|------------------|--------|-----------|--------|
| adam-lin-lora    | 1e-3   | 0.7564    | interior peak |
| adam-scaled-lora | 1e-3   | 0.7572    | interior peak |
| adamw            | 3e-4   | 0.7579    | baseline |
| muon-lora        | 3e-3   | 0.7675    | interior peak (post-NS-fix) |
| kron-grad-lora   | 1e-3   | 0.7850    | interior peak |
| galore-adamw     | 3e-4   | 0.8112    | interior peak (post-fix); 7% gap |
| diag-scaled-lora | 3e-2   | 0.8153    | boundary-pinned, ablation only |
| lin-lora         | 1e-2   | 0.8457    | boundary-pinned |
| scaled-lora      | 1e-2   | 0.8744    | boundary-pinned |
| psi-lora         | 1e-2   | 1.0446    | boundary-pinned, lr-scaled-ρ fix in flight |

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
| PSI-LoRA     | ✓ vs `~/PSI-LoRA` | ✓ behavioral match for α₁=0 (`scripts/verify_psilora_against_official.py`) — bit-exact single step, ~2e-5 cumulative drift over 5 steps. Momentum-on case still has a paper-vs-ref discrepancy on gradient/momentum coefficients. | zeligism/PSI-LoRA |

If we want strong claims about reproduction, we need to run the reference codebase on equivalent settings and compare outputs.
