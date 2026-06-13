# Handoff — protagonist switch to `kl-diag-polar-lora`

Status as of 2026-06-11. Read alongside `paper/PLAN.md` (campaign spec) and
`paper/e1_coverage_fill.md` (cell tracker). This doc captures the **decisions, the
in-flight work, and the one open question that gates the rerun**.

---

## 1. Decision: the protagonist is now `kl-diag-polar-lora`

Previously `diag-shampoo-polar-lora`. The locked config (with the 2026-06-11 β/method amendments):
**PolarExpress PE=8, Nesterov, $\beta_1=0.9$, $\delta=10^{-4}$, $k=1$, `curvature_beta=0.99`,
`precond_refresh_every=10`, `precond_method=gram_ns` ($S_a^{-1/2}$ by Polar-Express Gram NS, §4).**

**Why kl-diag (the only change is the input-feature diagonal $D_{in}$):**

- `diag-shampoo`: $D_{in}[i] = \mathrm{EMA}\big[\,\lVert g_A[:,i]\rVert^2\,\big]$ — raw per-feature gradient energy.
- `kl-diag`:      $D_{in}[i] = \mathrm{EMA}\big[\,g_A[:,i]^\top S_a^{-1} g_A[:,i]\,\big]$ — energy after stripping the core-side curvature $S_a = B^\top \mathrm{diag}(D_{out}) B$.

kl-diag keeps the diagonal of the *coupled* KL-Kronecker large factor; diag-shampoo keeps the
diagonal of the *uncoupled* one ($S_a=I$). Same whitened-polar machinery downstream.

**The deciding evidence (8B $\delta$-robustness):** at Llama-3-8B opc r256, the plain diagonal is
$\delta$-fragile — $\delta{=}10^{-4}\to10^{-3}$ swings the loss **11.9$\sigma$** ($\delta{=}10^{-4}$ gives a
~9$\sigma$ blowup, 0.5982). kl-diag's swing is **0.4$\sigma$**. ($\sigma_\text{AdamW}=0.0017$.)
`diag-shampoo` + $\delta{=}10^{-3}$ *also* lands in the winning cluster (0.5840 vs kl-diag 0.5822,
$\approx1\sigma$), but it leaves the $\delta$-cliff standing — kl-diag removes it. **Robustness, not
loss, is the reason.** $\delta=10^{-4}$ stays locked precisely because kl-diag is $\delta$-robust, so the
headline sits at the $\delta$ that breaks the uncoupled diagonal.

**1B equivalence (verified):** at 1B the coupling is inert (dense-Gram cond ≈ 50–190, well below
$1/\delta$), so kl-diag $\equiv$ diag-shampoo. The two-scale story — *coupling inert when the diagonal
is well-conditioned, load-bearing when it is not* — is the appendix $\delta$-sweep figure.

---

## 2. Code change (committed `c838e5c`)

`build_optimizer`'s `kl-diag-polar-lora` branch silently dropped
`cw_no_radius` / `cw_no_diag_curv` / `cw_factor_a` / `cw_factor_b` (same provenance-bug class as
`cw_nesterov`). An ablation sweep would have run the *full* protagonist while logging the flag as set.
Fixed; regression tests added in `tests/test_cw_ablation_flags.py`.

**Ablation-scope consequence — verified in code + test:** `−Shampoo` (`cw_no_diag_curv`) is
**base-independent**. Forcing the large-axis diagonals to $I$ collapses both diag-shampoo and kl-diag
to the same partner-Gram-only update (the coupling's $D_{in}$ EMA is accumulated but never read), bit-identical. So the
`−curvature` arm (`cw_no_diag_curv`) **reuses the existing run** and is *not* part of the rerun.
The magnitude ablation is now the **double (iMuon step)** via `cw_unpinned` + `--lora_init_b symmetric`;
the retired `−radius`/`cw_no_radius` arm kept the pin and is no longer used (dormant in code).

---

## 3. In flight: 1B sanity gate (job `6498391`)

`kl-diag-polar-lora`, PE8 + Nesterov, **$\beta_1=0.9$**, $\delta=10^{-4}$, OLMo opc **r64**,
lr $\in\{10^{-2},3{\times}10^{-2},10^{-1}\}$, 9000 steps. Group
`e1_sanity_kldiag_olmo_opc_r64_blackwell`.

- **$\beta_1=0.9$ on purpose:** OLMo-opc-r64 is **not** a paper cell, so $\beta$ is irrelevant here;
  $\beta_1=0.9$ matches the existing diag-shampoo PE8+Nesterov baseline (best **0.7562** @ lr0.03)
  exactly, isolating *only* the diag→kl coupling. (All paper/rerun cells use $\beta_1=0.9$ — see §5.)
- **Result so far (step-matched @ lr0.03):** kl-diag tracks diag-shampoo within **0.1–0.5$\sigma$** at
  every eval through step 1250 — coupling inert at 1B, as predicted. Final at step 9000 confirms the gate.
- **Notebook:** `paper/paper_plots.ipynb`, the "Sanity gate (pre-rerun)" cell (β-agnostic;
  `polar_express` filter excludes historical ns5 kl-diag runs; partial run shows in the trajectory panel).

**Gate:** kl-diag best-lr within ~1$\sigma$ (0.0007 at r64) of 0.7562 → green-light the rerun.

---

## 4. RESOLVED (2026-06-11): inverse-sqrt = `gram_ns` (Polar-Express Gram NS)

Adopted `precond_method=gram_ns` for the protagonist: Polar-Express Gram Newton–Schulz on the
small-side Gram (`gram_ns_inv_sqrt`, 8 iters, fp32), eigh-free, fresh every step. NOT the
coupled-Iannazzo `spd_inv_sqrt_higham_batched` (under-converged at the default 10 iters; gram_ns
matches eigh accuracy in 8 iters vs Iannazzo's 16). One Gram-NS framework with shared
Polar-Express coefficients now serves BOTH the inverse-sqrt and the matrix-sign/polar.

Verdict from the proper Blackwell methodology (`lora_playground/bench/inverse_sqrt_candidates.py`
+ `step_wall_vs_adamw.py` + `section_breakdown.py`; full writeup
`docs/notes/inverse_sqrt_variant_plan.md`):
- **Accuracy** matches eigh on real snapshot + δ-floor-cond synthetic Grams (rel ~1e-4). **bf16 is
  OUT** (blows up at the δ-floor via the Dao spurious-negative-eigenvalue failure) — fp32 only.
- **End-to-end `optimizer.step()` wall ≈ PARITY** with the amortized QR-eigh path at production
  batch. The earlier "~20% / 94×" microbench numbers were vs a COLD batched eigh, which is NOT the
  production path (QR amortizes a warm refresh 1-in-`precond_refresh_every`). gram_ns is slightly
  faster at r256, slightly slower at r64; both <1% of the full step.
- **Value is exactness, not speed**: no 10-step-stale eigenbasis, eigh-free (no cuSOLVER), drops the
  refresh-cadence knob.
- **Reachability bug fixed**: `build_optimizer(precond_method=…)` was silently dropped for the cw
  protagonist (spec skip + build-wide higham default); now forwards (None → family default).
- **No-degradation confirmed**: the gram_ns sanity (1B-to-9000 + 8B-1000) shows gram_ns ≤ eigh on
  all cells, with a small uniform edge from removing staleness.

---

## 5. Pending (gated on §3 sanity green-light and §4 method decision)

1. **kl-diag rerun**, one batched submission, all at $\beta_1=0.9$ (gram_ns), the locked config of §1:
   - **Protagonist coverage** — the 8 E1 cells (`paper/PLAN.md` §Cell set): OLMo/Qwen/Llama-3.2-1B opc
     r256, Llama-3-8B opc r256, Qwen bengali r256, Llama-3.2-1B openmath r64/r128/r256.
   - **E2 `−radius`** arm (kl-diag) at the ablation anchors. (`−Shampoo` reuses existing — §2.)
   - **iMuon** 2 demonstration cells (OLMo opc r256 + Qwen bengali r256) — still pending, independent.
   - Enumerate the exact completed-vs-needed `(model, data, rank, lr)` tuples via
     `lora_playground.loader.load_runs`, **not** hand-typed lists, before submitting.
2. **Paper updates → kl-diag protagonist:** `paper/PLAN.md` (protagonist string + C3 framing),
   `paper/skeleton.tex`, `paper/paper_plots.ipynb` (re-anchor ablation arms to kl-diag, remove old
   diag-shampoo protag-ablations, add the §1 $\delta$-sweep appendix figure).

## 6. Guardrails (carry forward)

- New protagonist/rerun cells use the **locked config** ($\beta_1=0.9$, gram_ns, PE8, Nesterov, $\delta=10^{-4}$).
  $\beta_1=0.9$ is the locked value (marginally best at the 9000 horizon, §PLAN; the $\beta$ gap is ≤1.2$\sigma$ — see
  `feedback-match-locked-protag-config` memory).
- Timing/profiling **only on Blackwell** (target hardware); use real snapshots, not synthetic, for
  conditioning-sensitive checks; benchmark the production config, not `build_optimizer` defaults.
- Submit via `slurm_pending/` (watcher auto-submits); cancel via `slurm_cancel_pending/`. Commit
  load-bearing changes first (import-closure clean check).
