# Protagonist pivot: diag-Shampoo → KL-diag (+ β₁=0.9, gram_ns) — 2026-06-11

The paper protagonist (`Polar-LoRA`) changed from `diag-shampoo-polar-lora` to
`kl-diag-polar-lora`, with β₁ locked at 0.9 and the small-side inverse-sqrt switched to
`gram_ns`. This note records *why*, so the rationale lives in a doc rather than in the
deprecated `paper/paper_plots_diag_legacy.ipynb` (whose transition cells — the 8B curvature-lever
probe, the kl-diag≡diag sanity gate, the β-select — are the justification record).

## 1. diag-Shampoo → KL-diag: the diagonal RULE, not staleness

The two optimizers differ in ONE thing: the large-axis Kronecker diagonal.
- **diag** (`diag-shampoo-polar-lora`, `kl_coupled=False`): raw grad energy `D_in[i]=EMA[Σ g_A[:,i]²]`.
- **kl-diag** (`kl-diag-polar-lora`, `kl_coupled=True`): the coupled fixed point
  `D_in[i]=EMA[g_A[:,i]ᵀ S_a^{-1} g_A[:,i]]` (each side whitened by the OTHER's inverse before forming).

At **8B/r256 opc** the raw-energy diagonal underperforms kl-diag by ~**9σ** (best-lr eigh @9000:
diag 0.560 / lr3e-4 vs kl-diag 0.552). The candidate explanation "it's the stale QR eigenbasis"
was **tested and rejected**: running diag δ=1e-4 with `gram_ns` (fresh `S_a^{-1/2}` every step, no
10-step-stale basis) moved diag only ~2σ — the **same uniform edge gram_ns gives every cell** —
and did NOT close the 9σ gap toward kl-diag. So the gap is the **diagonal rule** (raw energy can't
match the coupled metric), and kl-diag is the protagonist on a mechanism, not just an observation.
(On 1B the effect is absent — diag δ=1e-4 is fine there — so the lever is 8B-specific.)

## 2. β₁ = 0.9

Scoped β sweep (`kl-diag-polar-lora`, OLMo opc r256, full 9000 horizon): best-lr β=0.9 → **0.7357**
(lr=0.03) vs β=0.95 → **0.7377** (lr=0.01), Δ = **+0.0020 ≈ 1.2σ** for 0.9. A lower-lr extension
confirmed lr=0.01 IS β=0.95's interior optimum (lr ∈ {3e-3,1e-3,3e-4} all ≥ 0.754). Adopted 0.9:
marginally best AND consistent with the existing β=0.9 history (keeps the rerun comparable). An
earlier draft adopted 0.95 (Muon-canonical) off a 5250-step pilot; the full-horizon sweep reversed it.

## 3. precond_method = gram_ns

Adopted Polar-Express Gram Newton–Schulz (`gram_ns_inv_sqrt`, 8 iters, fp32) for `S_a^{-1/2}`, over
the QR eigenbasis (`eigh`) and the coupled-Iannazzo NS (`spd_inv_sqrt_higham_batched`). Full writeup:
`docs/notes/inverse_sqrt_variant_plan.md`. Headlines:
- Accuracy matches eigh (rel ~1e-4); bf16 is OUT (δ-floor blowup); fp32 only.
- **End-to-end wall ≈ parity** with the amortized QR-eigh path (the cold-eigh "94×/20%" microbench is
  NOT the production baseline). Value is **exactness** (no stale eigenbasis), eigh-free (no cuSOLVER),
  one knob fewer — not a speedup.
- Reachability bug fixed: `build_optimizer(precond_method=…)` was silently dropped for the cw
  protagonist; now forwards (build-wide default → `None` = family default).
- No-degradation confirmed by the gram_ns sanity (1B-to-9000 + 8B-1000): gram_ns ≤ eigh everywhere.

## 4. What this changes
- Code/wrapper/bench defaults → kl-diag / β=0.9 / gram_ns (commit "Pivot protagonist defaults…").
- Docs: `paper/{PLAN.md,code_map.md,e1_coverage_fill.md,HANDOFF.md,skeleton.tex}` flipped.
- Notebook: the old diag-Shampoo notebook is frozen as `paper/paper_plots_diag_legacy.ipynb`; the current kl-diag notebook keeps the canonical name `paper/paper_plots.ipynb` (fresh, library-based).
- Reruns: all E1/E2 protagonist cells re-run at the locked config (gated on explicit approval).
- Unchanged: the leaderboard notebooks (`notebooks/leaderboard/*`) auto-filter on labels, no pivot needed.
