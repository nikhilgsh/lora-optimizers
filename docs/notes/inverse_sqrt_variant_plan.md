# Most-efficient inverse-sqrt variant for `S_a^{-1/2}` — plan

The protagonist (`CurvatureWhitenLoRA`) computes the small-side Shampoo inverse-sqrt
`S_a^{-1/2}` on the **r×r** Gram (r≤256). Current paths: QR eigenbasis (`eigh`,
validated default) and coupled Newton–Schulz (`spd_inv_sqrt_higham_batched`,
Iannazzo/Denman–Beavers, the Phase-1 baseline). This note picks the variant to
adopt, grounded in three sources.

## Sources

1. **Gram Newton–Schulz** (Zhang, Amsel, Chen, Dao — Mar 2026, `docs/papers/Gram…Muon.pdf`).
   Polar(X) = (XXᵀ)^{-1/2} X, and each NS odd polynomial `p(x)=ax+bx³+cx⁵` factors as
   `p(x)=x·h(x²)` with `h(y)=a+by+cy²` — so the **same NS polynomials implicitly compute
   the inverse square root** of the Gram. *Algorithm 2 (Gram NS):* `R₀=XXᵀ, Q₀=I; for t:
   Zₜ=aₜI+bₜR_{t-1}+cₜR²_{t-1}; Qₜ=Q_{t-1}Zₜ; Rₜ=ZₜR_{t-1}Zₜ`; then `Q_T → (XXᵀ)^{-1/2}`.
   All r×r symmetric GEMMs.
2. **kexue.fm matrix sqrt / r-th root** (Su Jianlin — `matrix_sqrt_kexue11158.pdf`,
   `matrix_rth_root_kexue11175.pdf`). Same coefficient family (Polar-Express, /1.01-normalized).
   Gives the **decoupled** `Z`-recursion for `P^{-1/2}` (Eq 12–13) and the **fused
   `G·P^{-1/2}`** recursion (Eq 14–15): because the `YZ` recursion is independent of `Z`,
   folding `G` into `Z₀` yields `G·P^{-1/2}` in ONE recursion with **no explicit `P^{-1/2}`**.
3. **PRISM** (arXiv:2601.22137) — adaptive polynomial via sketched least-squares, "no
   explicit spectral bounds." Alternative to fixed coefficients; the sketch overhead is
   not worth it for a small r×r Gram. Noted, not recommended here.

## Recommendation

**Newton–Schulz inverse-sqrt on the r×r Gram using the protagonist's own Polar-Express
coefficients**, form-once-reuse across the 2–3 apply sites, with the stability recipe below.

Why over the coupled-Iannazzo baseline:
- **Coefficient reuse** — one polynomial family for BOTH the polar map (already
  `polar_method=polar_express`) and the inverse-sqrt. No second hyperparameter surface.
- **r×r symmetric GEMMs** — `Zₜ=aI+bR+cR²`, `Rₜ=ZRZ` are symmetric; GramNS shows ~half
  the work of naive GEMMs (compute lower-triangle, copy).
- **kexue fused `G·P^{-1/2}`** available as a per-site fallback when form-once shows
  instability — strictly more stable than forming `P^{-1/2}` then multiplying.
- Already aligned with our `ns_form="gram"` chord-tight primitive.

## Stability recipe (the load-bearing part — see CLAUDE.md spectral-norm guardrail)

The bf16 failure mode is precise and is exactly our `σ_max`-underestimation/NaN class:
**spurious negative eigenvalues** in the Gram (numerically low-rank → tiny `<0` eigenvalues
in bf16). The update `rₜ = r_{t-1}·hₜ(r_{t-1})²` has `rₜ < (15/8)²·r_{t-1}` for `r<0`, so a
negative eigenvalue **diverges to −∞** ("inverse-sqrt of a negative number") → blowup/Infs.

1. **Normalize** `S_a ← S_a / tr(S_a)` (or `/λ_max`) so eigenvalues sit in [0,1] (the NS
   coefficients are tuned for [0,1]).
2. **Relative-ε floor** `S_a ← S_a + ε·λ_max·I` (we already have `precond_delta_relative`
   + a guarded `λ_max`). Lifts the spectrum off zero/negative — the primary fix.
3. **Do NOT over-iterate in bf16** — the coefficients plateau at `(1.875,-1.25,0.375)`;
   5–6 iters suffice and **more iters accumulate/explode**. Our current default
   `higham_iters=10` is too many for bf16 → change to ~6.
4. **Restart** (GramNS) — for ill-conditioned input, run ~2–5 iters, reconstruct
   `R ← XₜXₜᵀ`, restart; eliminates large-magnitude negative eigenvalues. Likely
   unnecessary for our *damped small* Gram — the snapshot check decides.

## Verification — DONE (`lora_playground/bench/inverse_sqrt_candidates.py`)

Tested on real chord-tight snapshot Grams + synthetic power-law SPD spanning cond ∈
[1e2, 1e4] (the δ=1e-4 floor caps effective cond at ~1/δ), fp32 AND bf16, on the local
A6000 (numerics) and Blackwell workergpu174 (target-hw timing). Candidates: `eigh`
(default + fp64 truth), coupled-Iannazzo `spd_inv_sqrt_higham_batched`, and the
Polar-Express Gram NS `gram_ns_inv_sqrt` (this note's recommendation).

**Verdict — `gram_ns` (Polar-Express Gram NS, 8 iters, fp32) is a wall-parity, eigh-free
alternative; its value is exactness (no stale eigenbasis), not speed.**

### Accuracy + isolated-call cost (micro-bench, the *wrong* production baseline)
Blackwell, production r256 shape (N=224 pairs×sides):

| method                | rel_err p99 | ms/call | vs *cold* eigh |
|-----------------------|-------------|---------|----------------|
| eigh/fp32 (cold)      | 4.9e-5      | 1005    | 1×             |
| coupled_ns16/fp32     | 1.2e-4      | 13.6    | 74×            |
| **gram_ns_pe8/fp32**  | **1.1e-4**  | **10.6**| **94×**        |

- Accuracy matches eigh across the whole regime (rel_p99 ~1e-4 typical, ~7e-3 at the
  cond≈1e4 δ-floor worst case); stable in fp32 (min-eig(Z)>0, 0 nonfinite). Degree-5
  Remez converges in **8 iters** vs coupled-Iannazzo's **16**.
- **bf16 is OUT**: both NS blow up at cond≈1e4 in bf16 (Dao spurious-negative-eigenvalue
  failure). The r×r matrices are tiny; fp32 is cheap and correct.
- This 94× is vs a **cold batched eigh every step** — NOT the production baseline.

### End-to-end `optimizer.step()` wall (the honest number)
The production `eigh` path AMORTIZES a warm QR eigenbasis refresh 1-in-`precond_refresh_every`
(=10) + a cheap per-step Rayleigh/sandwich; gram_ns recomputes fresh every step. Full
`step()` wall on Blackwell (OLMo-2-1B, all-linear, compiled, ns=8), eigh vs gram_ns:

| ×AdamW (per-step wall) | global batch 4 | 16 (prod) | 64 |
|------------------------|----------------|-----------|-----|
| r256 eigh / gram_ns    | 1.227 / 1.212  | 1.058 / 1.055 | 1.014 / 1.013 |
| r64  eigh / gram_ns    | 1.121 / 1.139  | 1.031 / 1.036 | 1.007 / 1.007 |

opt.step: r256 eigh 131→gram_ns 123 ms (~7% faster — eliminating the eigenbasis); r64
eigh 52→gram_ns 60 ms (~15% slower — the 64×64 QR is already cheap). **At production batch
both are <1% of the full step, either sign — effectively wall-PARITY, rank-dependent.**

### What gram_ns actually buys (at parity)
Exact `S^{-1/2}` every step (no 10-step-stale eigenbasis), eigh-free (no cuSOLVER/QR), and
drops the `precond_refresh_every` knob. A cleanliness/robustness win, NOT a speedup — and
NOT yet an eval win (the stale eigenbasis matched exact-fp32 to ~0.2σ in the prior gate;
whether exactness ever matters, e.g. the diag δ=1e-4 question, is an open eval check).

NORMALIZATION (load-bearing): normalize the Gram by **trace** (= Frobenius-of-factor²,
matching `_polar_express_gram_batched` pre_norm="frob"), NOT tight λ_max — else the first
Remez coeff overshoots and `R←MRM` blows up. Trace-norm leaves λ_max(R0) ≤ 1/safety².

### Reachability (silent-drop bug, fixed)
`build_optimizer(precond_method="gram_ns")` for the cw protagonist was SILENTLY DROPPED
(spec `_CW_PRECOND_SKIP` + build-wide `precond_method="higham"` default). Fixed: build-wide
default → `None` (= "use the family default"); spec forwarding omits `None`; cw specs un-skip
precond_method; the 3 polar-product gauge/adamuon classes pinned to their `higham` default;
`train.py --precond_method` default → `None` + `gram_ns` choice. Now `None→eigh` (cw) /
`higham` (pp) unchanged, explicit `gram_ns` reaches the cw protagonist. The cw spec default
stays `eigh` until an eval no-degradation gate clears gram_ns for adoption.
