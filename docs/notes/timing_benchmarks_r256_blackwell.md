# Timing — curvature-whiten-polar optimization (r256, OLMo-2-1B, Blackwell)

Goal: make `curvature-whiten-polar-lora` as fast as possible **without degrading
eval loss**. Model `allenai/OLMo-2-0425-1B`, `lora_r=256`, `all-linear` = **112 LoRA
pairs**, bf16. All `opt.step()` numbers are isolated single-GPU (workergpu176),
**eager fp32**, production refresh cadence, from `scripts/bench/profile_curvature_whiten.py`
and `scripts/bench/section_profile_cw.py`. Single-measurement; the profiler uses a
tiny fwd/bwd (the optimizer step cost is batch/seq-independent), so absolute ms are
the step cost, not end-to-end wall.

---

## 1. Optimization progression — `opt.step()` for curvature-whiten-polar

| stage | opt.step() ms | vs prev | what changed |
|---|---|---|---|
| full SVD in NS spec pre-norm | 7951 | — | `matrix_norm(X, ord=2)` = a full SVD per NS call (224 SVDs/step, 81% of step) |
| → power-iter σ_max | 1788 | 4.4× | spec pre-norm via `spectral.sigma_max_power_iter` (commit 0975d44) |
| → grouped batched step | 137.7 | 13× | per-pair Python loop → one bmm/batched-NS/batched-σ_max per shape group |
| → **gram-NS (spec pre-norm) + warm-start σ_max** | **94.0** | 1.46× | rect `_newton_schulz_batched` (O(r²d)/iter on the 256×2048 factor) → `_newton_schulz_gram_batched` (r×r gram, ~7× fewer FLOPs at r≪d); warm-start the 6 σ_max (`v_init` cached per pair, `n_iters` 8→3) |

**Net: 7951 → 94.0 ms (~85×), with eval loss unchanged (§4).** SVD elimination (4.4×)
and per-pair→batched (13×) are the big wins; gram-NS + warm-start are the targeted
follow-ups closing the two largest remaining sections.

> **Caveat (do not re-try):** an initial gram-NS attempt used `pre_norm="frob"`
> (dropping the σ_max pre-norm to save 2 power-iters → 89 ms). It **degraded eval
> +0.0044 @250**: frob divides by `‖X‖_F`, starting NS at σ_max=1/√(stable_rank),
> which **under-converges the polar at ns=5** (documented in the gram-NS docstring).
> The σ_max ("spec") pre-norm is load-bearing — it starts NS at the σ=1 fixed point.
> Use the power-iter for σ_max + `pre_norm="none"`, never the gram fn's `"spec"`
> branch (that's a full SVD).

---

## 2. Section breakdown — final (gram-NS + warm-start), K=10

From `section_profile_cw.py` (CudaTimer scopes), `logs/bench/section_cw_polar_r256.log`,
median over 30 steps:

| section | ms/step | share | what it is |
|---|---|---|---|
| `cw_polar_ns` | 24.6 | 31% | gram-form NS (fp32) for the polar map + 2 warm σ_max for the spec pre-norm, A+B |
| `cw_basis_proj` | 15.8 | 20% | SOAP basis projections Qᵀg + z reconstruction + Adam m,v EMAs |
| `cw_rescale_sigma` | 14.3 | 18% | 4 warm σ_max (sA,sB,sWA,sWB) for the chord-tight ρ rescale |
| `cw_refresh` | 12.9 | 16% | batched QR eigenbasis refresh (129 ms/refresh ÷ 10) |
| `cw_unwhiten` | 7.0 | 9% | WA/WB reconstruction out of the eigenbasis |
| `cw_curv_grams` | 4.5 | 6% | curvature gram EMAs (L_A, R_B, D_in, D_out) |

The step is now **balanced** — no single dominant bottleneck. The QR refresh (13.1)
is *not* the gap (see §3).

---

## 3. Production-cadence comparison vs chord-tight-clean

`opt.step()` only, isolated r256, **production cadence**: chord-tight-clean at K=1
(`precond_refresh_every=1`, Higham every step — gate `optim.py:5885`); cw-polar at
K=10 (QR refresh every 10 — gate `optim.py:1166`). chord-tight from
`logs/bench/section_chordtight_k1.log` (LORA_PROFILE_OPTIM sections); cw from §2.

| shared stage | chord-tight-clean (K=1) | cw-polar (K=10) | Δ (cw − ct) |
|---|---|---|---|
| Adam dir + input whiten | 12.9 | 15.8 (`basis_proj`) | +3 |
| σ_max power-iters | ~25 (4, **warm**) | ~20 (6, **warm**) | −5 |
| Newton–Schulz polar | 22.7 (2 picard ×ns) | ~19 (gram, A+B) | −4 |
| whitening refresh | 11.5 (Higham/step) | 12.9 (QR/10) | +1 |
| unwhiten + grams (SOAP-only) | 0 | 11.5 | **+12** |
| **wall** | **79.4** | **94.0** | **+15** |

After gram-NS + warm-start, cw-polar's only remaining structural disadvantage vs
chord-tight is the **SOAP eigenbasis transforms** (`unwhiten` + `curv_grams`, ~12 ms)
that chord-tight has no analogue for — it applies one Higham `S^{-1/2}` matmul instead
of projecting in/out of an eigenbasis. The QR refresh (13.1) is matched to chord-tight's
Higham whitener (11.5); it is *not* the gap. The SOAP tax is intrinsic to what cw is
(SOAP-in-curvature-basis) — reducing it changes the algorithm.

---

## 3a. Clean production ×AdamW (uncontended, ns=8 matched, full step)

Back-to-back `bench_optimizer_step.py` (one invocation = shared node state), production
shape seq=2048 bs4 ga4 bf16 compiled packed_v1, **ns=8 for both** (matched; `--muon_ns_steps 8`
routes to cw's `ns_steps` too — `optim.py:10950`). All cells at **251.9 TFLOPS (uncontended)**,
fwd+bwd identical (0.83/1.24 s) — so cross-optimizer comparison is clean. Source
`logs/bench/clean_decomp_ns8_r256.jsonl`. fwd+bwd are optimizer-independent.

| optimizer | K | opt ms | total ms | **×AdamW** | MFU |
|---|---|---|---|---|---|
| adamw | — | 10.4 | 2074 | **1.00** | 42.1% |
| curvature-whiten-polar (prod) | 10 | 95.3 | 2165 | **1.04** | 40.3% |
| chord-tight-clean (prod) | 1 | 100.0 | 2171 | **1.05** | 40.2% |
| curvature-whiten-polar (matched-K) | 1 | 208.1 | 2276 | 1.10 | 38.4% |

**Production per-step overhead vs AdamW: cw-polar +4%, chord-tight-clean +5% — tied.**
fwd+bwd dominate (~95% of the step), so the optimizer choice barely moves the full-step wall.

**cw is NOT cheaper than chord-tight algorithmically.** At **matched K=1**, cw opt = 208 ms vs
chord-tight 100 ms — **cw is 2× more expensive** (it does strictly more: the per-step eigenbasis
QR refresh). cw only edges chord-tight at *production* cadence because cw amortizes its expensive
QR refresh 1-in-10 steps (K=10) while chord-tight refreshes its cheaper Higham whitener every step
(K=1). It's the refresh cadence, not the algebra. (cw production is ns=5, marginally cheaper than
this ns=8-matched number.)

---

## 4. No-degradation gate

gram-NS + warm-start cw-polar at task 09's exact config (lr=1e-2, seed=0, δ=1e-3, ns5,
K=10, packed_v1.1, r256), eval@250 vs the recorded rect-fp32 baseline.

Matched config (lr=1e-2, seed=0, δ=1e-3, ns5, K=10, packed_v1.1, r256, **compiled**,
same eval set `eval_samples=1023`); only the optimizer code differs.

- task 09 (rect-fp32 baseline, compiled): **eval@250 = 0.855240**
  (`logs/curvature_whiten_soap_block_r256_olmo_opc/run_info/logs/log_09.out`)
- gram-NS (spec) + warm-start, compiled: **eval@250 = 0.855117** → **Δ = −0.000123** (~0.2σ_AdamW) ✓
  (`logs/bench/eval_gate_cw_gram_spec_warm_COMPILED.log`)

**No degradation.** Math-equivalence also pinned by unit tests: gram-fp32 NS = rect-fp32
NS to fp32 (`tests/test_ns_gram.py`); grouped = per-pair to 1e-4
(`tests/test_curvature_whiten_batched.py`). Warm-start n=3 is floor-guarded against
under-estimation (`tests/test_sigma_max_power_iter.py::test_batched_estimate_has_row_norm_floor_for_bad_warm_start`).

The earlier `pre_norm="frob"` variant was **rejected** — compiled gate eval@250 = 0.859643
(Δ +0.0044, ~6σ), the ns=5 under-convergence documented above.

---

## Provenance

- Profiler: `scripts/bench/profile_curvature_whiten.py`, `scripts/bench/section_profile_cw.py`
- Logs: `logs/bench/section_cw_polar_r256.log` (cw, gram+warm), `logs/bench/section_chordtight_k1.log`
  (chord-tight K=1), `logs/bench/eval_gate_cw_gram_warmstart.log` (no-degradation gate)
- Code: `_newton_schulz_gram_batched` / `_smax_warm` / `_cw_apply_grouped` in `lora_playground/optim.py`
- Hardware: 1× RTX PRO 6000 Blackwell (workergpu176), `--reservation=rocky9`
