# kl-shampoo-polar lr-robustness

**Goal: make kl-shampoo-polar less lr-sensitive while preserving its best-lr performance.**
kl wins at its optimal lr but its held-out loss climbs faster off the optimum than the
canonical full-polar chord baseline — so it needs more lr tuning, against the "one
generally useful optimizer" aim. We want the wide basin *and* the peak.

## Finding: kl is more lr-sensitive than chord, across settings (not low-rank-specific)

Factor-3-band held-out-loss spread (band = opt ±1 grid step; via
`leaderboard.leaderboard_rows` / `workloads.workload_runs`):

| setting | kl δ | kl band-spread | chord ns=8 k=1 band-spread |
|---|---|---|---|
| opc r64 | 1e-4 | **0.040** | 0.021 |
| openmath r256 | 1e-4 | 0.017 | — |
| opc r256 | 1e-3 | 0.008 | 0.011 |

No clean rank-monotone story (tangled with rank, δ, dataset) — so the question is **how to
reduce the sensitivity**, not when it appears. Best test-bed for a fix is where kl is most
sensitive: **opc r64, δ=1e-4 (spread 0.040)**. NB: the sensitivity is **two-sided** (both
too-low and too-high lr hurt); the high-lr side is the one a step-size fix can address (a
too-low lr is just under-training at the fixed horizon).

## Ruled out

- **Balance projection** (BaLoRA, Castin et al. `balora_2605.31484.pdf`): *hurts* — higher
  loss at every lr and a steeper curve (220-step r16 pilot). The balancing-projection code
  was removed (dead lever). The balance drift is an lr-driven symptom, not the cause.
- **More damping** (δ 1e-4→1e-3): ≈ baseline.

## Candidate levers (not yet tested on kl)

1. **Picard k≥2** — kl already supports it via `--cw_picard_iters` (`CurvatureWhitenLoRA`,
   loop at `optim.py:1799`). The cross-term `mhatA_eff = mhatA + (1/η)(Bᵀ dB)A` matches the
   canonical chord-tight-clean coupling (`algorithm_clean_implementation.md` §2.6). On chord
   (regularized/SSC polar) Picard k=2 cuts band-spread ~2–3× with best-loss preserved — but
   that evidence is only **2 clean ns=10 (SSC, *regularized*) pairs**; there is **no
   full-(hard-)polar k=1↔k=2 pair**, and kl uses the **hard** polar, so transfer is unproven.
   **CRUX — §2.5 pre-rescale:** chord spec-norms the polar input to unit σ AND applies the
   same divisor to `u_A` *before* the cross-term, "so the Picard cross-coupling stays
   consistent." kl has **no §2.5 pre-rescale**, so its cross-term relative weight is likely
   mis-normalized. **This must be resolved (derivation or a numerical fixed-point-convergence
   check) before trusting a kl k=2 run** — see `utils.prerescale_unit_op`.
2. **Momo / NGN** (secondary) — loss-aware self-limiting step, `F_*=0` (no new HP; CE≥0;
   robustness from the `min` saturating at high η). Form pinned from `muon_variants_2510.09827.pdf`
   §4 Props 4.1–4.3; F̄_t reuses the existing β. Caps the high-lr side only.

## Status

Not yet tested on kl. Next: resolve the §2.5 normalization for kl's Picard (the load-bearing
correctness question), then the opc-r64 k=1-vs-k=2 measurement (target: spread → ~0.020,
best ≈ 0.757). Diagnostics for this are already logged (`balance_resid`, `stable_rank`,
`sigma_max`, and the product-step `*_over_lr` fields).
