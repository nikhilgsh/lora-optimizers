# Timing benchmarks — r256, OLMo-2-1B, Blackwell

**Status: PRELIMINARY.** The clean, isolated, production-shape numbers come from job
`6465662` (running on workergpu179 as of 2026-06-01); §1 and §3 below are stopgaps from a
tiny-shape microbench and from packing-confounded production logs respectively, and will be
**superseded** by §4 when that job completes. All numbers are single-measurement.

Model `allenai/OLMo-2-0425-1B`, `lora_r=256`, `all-linear` = **112 LoRA pairs**. bf16.

---

## 1. Optimizer-step cost — interactive microbench  ⚠️ DIRECTIONAL ONLY

Source: `scripts/bench/bench_optimizer_step.py` on workergpu176 (1 GPU), **tiny shape**
`seq_len=512, batch=1, grad_accum=2`, **eager (no compile)**, `logs/bench/smokeA.jsonl`,
commit `b3de59e`.

| optimizer | K | fwd ms | bwd ms | **opt ms** | total ms | ×AdamW (total) |
|---|---|---|---|---|---|---|
| `adamw` | 1 | 180 | 229 | **10.8** | 421 | 1.0 |
| `curvature-whiten-lora` (no-polar) | 10 | 181 | 229 | **1397** | 1808 | 4.3 |
| `curvature-whiten-polar-lora` | 10 | 181 | 229 | **7904** | 8315 | 19.8 |

Reading: ordering is `adamw ≪ no-polar ≪ polar`; the polar step is ≈5.6× the no-polar step
(7904/1397) — the Newton–Schulz orthogonalization.

**Why these OVERSTATE production cost (do not quote as production ratios):**
- Tiny fwd/bwd (≈410 ms) lets `optimizer.step` dominate the total artificially; at production
  shape fwd/bwd is ~16–30× larger, so the totals compress hard.
- K=10 with only 10 timed steps means the **one-time eigh init** (expensive at 112×256×256) is
  averaged over those 10 steps instead of amortized over thousands → the `opt ms` is inflated.

---

## 2. Polar vs no-polar at production scale — apples-to-apples  ✅ SOLID

Source: SOAP run `curvature_whiten_soap_block_r256_olmo_opc`, tasks 08/09 — **same node,
same 2-packing, production shape** (`seq2048, bs4, ga4`, 9000 steps). Only the optimizer differs,
so this ratio is clean. Logs `run_info/logs/log_08.out`, `log_09.out`.

| optimizer | s/step (wall-inclusive) |
|---|---|
| `curvature-whiten-lora` (no-polar) | 2.40 |
| `curvature-whiten-polar-lora` | 4.25 |

→ **the polar (Newton–Schulz) variant adds ~77% per-step wall** over no-polar curvature-whitening.

---

## 3. Cross-family production s/step  ⚠️ PACKING-CONFOUNDED — DO NOT COMPARE DIRECTLY

Each from a completed r256 log via `lora_playground.timing.s_per_step_from_log`, but at
**different node-packing levels**, so cross-family ratios are unreliable until §4.

| optimizer (family) | s/step | packing | source group |
|---|---|---|---|
| `adamw` | 1.61 | 5-packed | `adamw_phase_L_lrsweep_r256_blackwell` |
| `adam-polar-product…chord-tight` (ns10) | 1.79 | 5-packed | `chord_tight_polar_express_phase_L_lrsweep_r256_blackwell` |
| `adam-polar-product…chord-tight-clean` (ns8) | 1.74 | 5-packed | `curvature_whitening_ns8_k1_r256_olmo_opc` |
| `curvature-whiten-lora` | 2.40 | 2-packed | SOAP run task 08 |
| `curvature-whiten-polar-lora` | 4.25 | 2-packed | SOAP run task 09 |

Note: even at *lighter* (2-) packing, curvature-whiten-polar is the slowest arm — so it is
genuinely the most expensive family — but the exact cross-family multiples need the isolated job.
Also: `curvature-whiten-{,-polar}-lora` collapse to the **same** registry key
(`…/curvature-whiten/ns5/k1`), so only the slower (polar) value can be stored — record it as the
conservative entry.

---

## 4. PENDING — clean numbers from job `6465662` (running)

Will supersede §1 and §3:
- **Isolated** (single-job-per-node) production-shape intrinsic cost per optimizer → clean
  cross-family comparison.
- **K=1 vs K=10** refresh-cost split (one-time eigh-init vs steady-state QR).
- **Section breakdown** (`LORA_PROFILE_OPTIM=1`): how much of polar's cost is Newton–Schulz vs
  σ_max vs the Gram refresh.
- **Packing curve** N∈{1,2,4,8} + `nvidia-smi` clocks/power → the host-contention penalty and its
  mechanism (throttling vs bandwidth).

Analysis: `python scripts/bench/analyze_packing_curve.py` once the job finishes; registry seeded
via `python -m lora_playground.timing record <isolated-log> --hardware blackwell`.

---

## 5. Cost vs benefit (r256, step 9000, best lr per arm)

Benefit (eval_loss) via `lora_playground.loader.load_runs` over the r256 leaderboard groups —
min final eval_loss per optimizer, **single-seed**. Cost (s/step) from §2–§3 — ⚠ **packing-
confounded across families** (see §3), so the cost column is not yet apples-to-apples.

| optimizer | eval_loss @9000 | s/step | packing | best lr |
|---|---|---|---|---|
| `curvature-whiten-polar-lora` | **0.7387** | 4.25 | 2-pk | 1e-2 |
| `…chord-tight-clean` (ns8) | 0.7394 | 1.74 | 5-pk | 3e-3 |
| `…chord-tight` (ns10) | 0.7414 | 1.79 | 5-pk | 1e-2 |
| `curvature-whiten-lora` (no-polar) | 0.7423 | 2.40 | 2-pk | 1e-2 |
| `adamw` | 0.7524 | 1.61 | 5-pk | 1e-4 |

Reading (provisional — cost packing-confounded, loss single-seed):
- Polar curvature-whiten has the **best loss** but is the **most expensive** arm.
- Edge over no-polar curvature-whiten: **−0.0036 eval_loss for +77% wall** (the +77% is the clean
  §2 ratio).
- Edge over the much cheaper `chord-tight-clean`: only **−0.0007** — within plausible single-seed
  noise (r=64 AdamW σ≈0.0007; an **r256 AdamW σ is not yet measured**). So the expensive arm may
  not buy a real loss win over the cheap one.

**Decision (pending §4 isolated/packing-clean costs + an r256 AdamW σ):** is curvature-whiten-
polar's loss edge real and worth its cost, or does `chord-tight-clean` dominate on cost-for-loss?

---

## 6. Why curvature-whiten-polar costs more than chord-tight-clean (structural, from code)

Both do the same polar map (Newton–Schulz), so **that is not the differentiator**. The cost gap
is three structural things in `CurvatureWhitenLoRA.step` (`optim.py:1200–1317`) that the
chord-tight-clean path (`_chord_tight_clean_polar_pipeline`, `optim.py:4994+`, called from the
**batched** `_step_batched`) does not pay:

1. **Per-pair Python loop vs fully batched.** Curvature-whiten runs a `for` loop over all 112
   pairs (`optim.py:1214`), doing many small `r=256` matmuls + 4 power-iterations *per pair* — 112×
   the kernel-launch overhead and poor GPU utilization on small matrices. Chord-tight-clean runs
   every stage as one **batched** `(N=112, …)` tensor op (`N = A_f.shape[0]`, `optim.py:5017`).
2. **Extra eigenbasis "SOAP" machinery** with no analogue in chord-tight-clean: ~6
   `(r×r)·(r×d)` basis transforms per pair (`gA_basis`, `mA_basis`, `zA=QA@…`, etc.,
   `optim.py:1231–1240`), two curvature-gram EMAs (`L_A += gA@gAᵀ`, `R_B += gBᵀ@gB`,
   `optim.py:1288–1289`), and the periodic batched **eigh/QR refresh** (`optim.py:1301–1317`).
   Chord-tight-clean instead uses a batched Higham whitener (`SA_half_inv`/`SB_half_inv`, refreshed
   periodically) — no eigenbasis transforms, no gram EMAs.
3. **4× σ_max power-iteration per pair** (8 iters each): `sA, sB, sWA, sWB`
   (`_sigma_max_block_guarded`, `optim.py:1271–1279`). Chord-tight-clean hoists/ batches σ_max
   (caller-passed `sigma_A`/`sigma_B`) and does fewer.

So: **batching + the SOAP eigenbasis/gram machinery + 4 per-pair σ_max calls** drive the gap, not
the orthogonalization. The empirical per-section ms split (NS vs σ_max vs eigenbasis vs grams)
comes from §4's `LORA_PROFILE_OPTIM` run; this section is the algorithmic "why".

---

## 7. Profile breakdown of curvature-whiten-polar's step (MEASURED)

`torch.profiler` on an **isolated** Blackwell GPU (workergpu176), r256 production, eager,
`refresh_every=10`. Source: `scripts/bench/profile_curvature_whiten.py`,
`logs/bench/profile_cwpolar.log`. The earlier §6 "per-pair-loop is the bottleneck" hypothesis was
**wrong** — the step is **compute-bound**, and one op dominates:

- **`opt.step()` ≈ 7.95 s/step** (steady 7.88 s, refresh 8.31 s). Self-CUDA per step ≈ 7.40 s.

| op | self CUDA | % CUDA | calls | avg |
|---|---|---|---|---|
| **`aten::_linalg_svd`** | **6.69 s** | **81.6%** | **224** | 29.9 ms |
| `aten::mm` (matmuls) | 0.77 s | 10.5% | 11872 | 65 µs |
| `aten::linalg_qr` | 0.38 s | 4.3% | 4032 | 94 µs |

**Root cause:** the 224 SVDs = 112 pairs × 2 (A and B), one per `_newton_schulz(pre_norm="spec")`
call, which computed σ_max via `torch.linalg.matrix_norm(X, ord=2)` — a **full SVD** — even though
its own docstring says it should cost "one power-iter". So ~80% of the step was spent computing a
scalar (σ_max) the expensive way.

**Fix applied** (this change): `pre_norm="spec"` now calls the library
`spectral.sigma_max_power_iter` (matvec-based power iteration, no SVD/Gram), `n_iters=8`. Safe — NS
only needs σ < √3 and the update magnitude is renormalized downstream (unit test:
`tests/test_curvature_whiten_ns_specnorm.py`).

**Post-fix profile** (`logs/bench/profile_cwpolar_after.log`): **`opt.step()` 7.95 → 1.79 s/step,
~4.4× faster.** The SVD is gone:

| op | self CUDA before | self CUDA after | calls |
|---|---|---|---|
| `aten::_linalg_svd` | 6.69 s (81.6%) | **— (0 calls)** | 0 |
| `aten::mm` | 0.77 s | 0.77 s (54%) | 11872 |
| `aten::linalg_qr` | 0.38 s | 0.38 s (22%) | 4032 |
| `aten::_linalg_eigh` | — | 0.04 s (2.7%) | 448 |
| **Self-CUDA total** | **7.40 s** | **1.42 s** | |

**Next lever (now, not before):** post-fix the step is mildly *launch-bound* (self-CPU 2.05 s >
self-CUDA 1.42 s) — the per-pair Python loop fires ~12k `mm` + 4k `qr` small kernels (§6). Batching
the per-pair step by shape group (reusing the polar-product batched primitives) is the next
increment. SVD elimination was the 4.4× win; batching targets the remaining launch overhead.

### 7a. Measured head-to-head vs chord-tight-clean (post-fix, `opt.step()` only, isolated, r256)

Both via `scripts/bench/profile_curvature_whiten.py` on workergpu176 (`profile_cwpolar_after.log`,
`profile_chordtight.log`):

| | steady step | refresh step | amortized (1-in-10) |
|---|---|---|---|
| `curvature-whiten-polar` (post-SVD-fix) | 1753 ms | 2158 ms | ~1790 ms |
| `chord-tight-clean` | 115 ms | 3629 ms | ~466 ms |

chord-tight is **~3.8× cheaper amortized (~15× on steady steps)**, because it is **batched**
(profile dominated by `aten::bmm`, 390 calls) vs curvature-whiten's per-pair `aten::mm` (11,872
calls). Cost shape differs: chord-tight front-loads into a heavy periodic refresh (3.6 s Higham/eigh
whitener) with trivial steady steps; curvature-whiten is flat (~1.75 s/step, cheap batched-QR
refresh) — so on a *single refresh* step curvature-whiten is actually cheaper. The remaining gap is
the per-pair loop → batching curvature-whiten's step (§6, task #6) is what closes it.
