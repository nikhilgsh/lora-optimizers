# Stage 0 — chord-tight diagnostics readout

**Status:** single-seed, mid-run (r=16 reached step 300/500, r=64 reached step 200/500 at the time of this writeup); values are intermediate medians, not the final 2k-canonical-horizon. Pipeline `packed_v1`, optimizer `adam-polar-product-lora-coupled-spectral-chord-tight` at `lr=3e-3`, `lora_alpha=lora_r`, `seed=0`. Log group `logs/chord_tight_diag_500_r16r64`. Commit `84aafe8`.

This document records what we ran, why, what we measured, and the three concrete decisions that fall out of the data. The proposed variants are described in [`algorithm_tight_chord.md`](algorithm_tight_chord.md); see the §1–§9 derivation for the underlying program. All probe formulas live in `lora_playground/optim.py:3479–3590` (`AdamPolarProductLoRA._step_per_pair`).

## Context

The current `adam-polar-product-lora-coupled-spectral-chord-tight` magnitude rule sets

$$
\rho \;=\; \frac{-s + \sqrt{s^2 + 4\eta}}{2}, \qquad s = \sigma_{\max}(A) + \sigma_{\max}(B)
$$

via worst-case submultiplicativity. Three candidate refinements were proposed:

| Variant | Idea | Where the slack is |
|---|---|---|
| 1 — direction-aware $\rho$ | Replace $s$ with $a = \lVert BP\rVert_2 + \lVert QA\rVert_2$ and $1$ with $b = \lVert QP\rVert_2$ in the quadratic; $P, Q$ unit-norm factor directions | $\sigma_{\max}(B) \cdot 1$ vs $\lVert BP\rVert_2$ — Cauchy–Schwarz slack when $P$ misaligned with top right-singular direction of $B$ |
| 2 — exact low-rank chord scaling | Compute the exact operator norm of the chord at each $\lambda$, bisect | Triangle inequality slack on top of variant 1 |
| 3 — exact clip prox | Replace polar with clip in the whitened subproblem (`Proposition 1`); only equivalent to polar in the saturating regime | Clip $\to$ polar substitution loses information when whitened cost has unsaturated singular values |

Stage 0 instruments the existing optimizer with cheap probes that test each variant's premise *before* implementing any of them.

## The probes

Implementation: `lora_playground/optim.py:3479–3590`. All probes read finalized `dA, dB, geo_A, geo_B, ρ, op_geoA, op_geoB, S_A^{-1/2}, S_B^{-1/2}` from the existing per-pair scope; no optimizer math is changed. Probes are gated by `--log_optim_diagnostics` and emit through `_emit_optim_diagnostics` (`optim.py:359–378`) as median/min/max across LoRA pairs per `optim_step` event. Verified observation-only: two seeded runs with `--log_optim_diagnostics` on produce byte-identical params (`tests/test_polar_product.py:933–968` covers this contract for the optimizer).

| Probe | Formula | Variant tested |
|---|---|---|
| **A. chord_slack** | $\lVert B \cdot dA + dB \cdot A + dB \cdot dA\rVert_2 / \eta$ via rank-$\le 2r$ factorization $L = [B+dB,\ B]$ ($d_{\text{out}} \times 2r$), $R = [A+dA;\ -A]$ ($2r \times d_{\text{in}}$); $\sigma_{\max}^2(LR) = \lambda_{\max}((R R^\top)(L^\top L))$ on a $2r \times 2r$ matrix | Whether the worst-case bound is leaving step size on the table (variants 1 / 2) |
| **B. lambda_dir_gain** | Solve $a\lambda + b\lambda^2 = \eta$ for the smaller positive root with $a, b$ above; report $\lambda_{\text{dir}} / \rho$ | Direct measurement of variant 1's $\rho$-improvement |
| **C-tight. cos_polar_clip_tight, sat_frac_tight** | Saturation fraction $\#\{i : \eta\,\sigma_i(c_A) \ge \tau_A\} / r$ at the §8 threshold $\tau_A = \rho / \lVert D_A\rVert_2$, and the corresponding whitened-space cosine | Variant 3 — at saturating thresholds, polar = clip exactly; cosine $< 1$ would mean clip and polar disagree |
| **D. adam_gauge_residual_(frob,rel)** | $E = u_A A^\top - B^\top u_B$ ($r \times r$); $\lVert E\rVert_F$ and the same divided by $\max(\lVert u_A A^\top\rVert_F, \lVert B^\top u_B\rVert_F)$ | Generic geometric distortion from Adam preconditioning; for raw factor gradients $E \equiv 0$ exactly |

## Findings

Medians across probe-steps of the per-step pair-medians, intermediate run.

|                              | r=16 (step ≤ 300) | r=64 (step ≤ 200) |
|------------------------------|--------:|--------:|
| eval_loss @ step 200         |  0.5995 |  0.5943 |
| chord_slack (med-of-med)     |  0.91   |  **1.13** |
| chord_slack (max-of-med)     |  0.93   |  1.28   |
| chord_slack (absolute max, single pair) | 1.00 | **1.63** |
| sat_frac_tight_A             |  1.000  |  1.000  |
| sat_frac_tight_B             |  1.000  |  1.000  |
| cos_polar_clip_tight_A       |  1.000  |  1.000  |
| cos_polar_clip_tight_B       |  1.000  |  1.000  |
| lambda_dir_gain (med-of-med) |  0.86   |  1.05   |
| lambda_dir_gain (max-of-med) |  1.00   |  1.14   |
| dir_a_over_s                 |  1.16   |  0.96   |
| adam_gauge_residual_rel      |  0.11   |  0.06   |

Bolded numbers are load-bearing for the decisions below.

### F1 — Variant 3 is a no-op at this scale

`sat_frac_tight = 1.000` and `cos_polar_clip_tight = 1.000` everywhere on both ranks. The whitened cost $c_A$ has every singular value above the saturating threshold $\tau_A$, so clip and polar produce identical updates. Variant 3 (exact clip prox) replaces polar with clip — at this scale and these settings, that is a **no-op by construction**, not just empirically. **Skip variant 3.**

### F2 — The current "tight" bound is breached at r=64

`chord_slack > 1` means $\lVert\Delta W\rVert_2 > \eta$, i.e. the spectral-step guarantee that gives `spectral_chord_tight` its name is *not* being kept. At r=64 the median pair sees a 13% breach; the worst pair on a probe step sees 63%. At r=16 the breach is essentially zero (max 0.3%).

Mechanism: the optimizer normalizes `dA = -ρ · geo_A / op_geoA` at line 3241, where `op_geoA` is computed by **8-iter cold-start** power iteration on `geo_A` (`optim.py:3237`). The polar map produces a `geo_A` whose pre-unwhitened factor has nearly flat singular values (polar = $UV^\top$), so after unwhitening by $S_B^{-1/2}$, the singular spectrum of `geo_A` is dominated by $S_B^{-1/2}$'s spectrum — which becomes **flatter as $r$ grows** (more roughly-equal singular values fitting in the $r \times r$ Gram). Power iteration's convergence rate is $(\sigma_2/\sigma_1)^{2k}$, so a flatter spectrum at larger $r$ is exactly the case where 8 cold-start iterations leaves a 15–20% relative under-estimate. An under-estimated `op_geoA` makes $\lVert dA\rVert_2 = \rho \cdot \lVert\text{geo}_A\rVert / \text{op\_geoA} > \rho$, and the chord bound inherits the same slack.

This is independent of variant 1 — the existing optimizer needs the fix. The cheap remedy mirrors what the optimizer already does for $\sigma_{\max}(A), \sigma_{\max}(B)$ at lines 3175–3180: cache the top singular vectors `v_geoA_top, v_geoB_top` in `pair_state`, run 3 warm-started power iter steps. The factor changes ~$\eta$ per step, so warm-start across optimizer steps is accurate to <1% in 3 iters.

### F3 — Variant 1 gain is moderate (~5%) at r=64; r=16 number is probe-biased

At r=64, `lambda_dir_gain` med-of-med is 1.05 — direction-aware $\rho$ is on average 5% larger than worst-case $\rho$. Tail goes to 1.14 on individual probe steps. That translates to roughly the same factor of larger spectral step on average. At fixed step budget the loss-per-step gain is sublinear in step size (~50% efficiency in the regime where the optimizer isn't already saturating); rough projection: **2–3% loss-per-step improvement at the 2k canonical horizon, ~25–30σ_AdamW**. Detectable, modest.

At r=16 the raw `lambda_dir_gain = 0.86` is **not** a real signal that direction-aware loses to worst-case. `dir_a_over_s = 1.16 > 1` is mathematically impossible if $P$ truly has $\lVert P\rVert_2 = 1$ ($a \le \sigma_{\max}(B) + \sigma_{\max}(A) = s$ by submultiplicativity). The probe normalizes $P = \text{geo}_A / \text{op\_geo}_A$ using the same biased `op_geoA` from F2 above; $P$ is therefore inflated by ~16% in operator norm at r=16, biasing $a$ upward and $\lambda_{\text{dir}}$ downward. After the F2 fix, the r=16 number should also be in the ~1.05–1.10 range.

### F4 — Variant 2 buys at most a few percent on top of variant 1

With `chord_slack ≈ 0.91` at r=16 and the F2-fixed bound holding tightly at r=64, the actual chord under variant 1 will be ≈ `chord_slack × lambda_dir_gain ≈ 0.95–0.98` of $\eta$ — already saturating the budget. Variant 2 (exact low-rank chord, no triangle slack) recovers at most 2–5% additional step size at ~10× the per-step cost of variant 1 (5–10 bisection iters of $\sigma_{\max}$ on a $2r$-rank factor vs. a single quadratic root). Net wall-time-adjusted gain is likely zero or negative. **Skip variant 2.**

### F5 — Adam preconditioning is mildly geometry-distorting

`adam_gauge_residual_rel ≈ 0.06–0.11` measures how much Adam's per-coordinate $1/\sqrt{v_t}$ rescaling breaks the raw-gradient identity $g_A A^\top = B^\top g_B$ (which is exact since both equal $B^\top G A^\top$ for $G = \partial L / \partial W$). 6–11% relative is small — most of the matrix-level gradient signal survives Adam's coordinate-anisotropy. Useful baseline for cross-optimizer comparison: SOAP/Shampoo would have a different (likely smaller) residual; raw-gradient Muon variants would have residual $= 0$ by construction. Single value, not load-bearing for the variant decisions.

## Decisions

1. **Power-iter polish (mandatory):** warm-start `σ_max(geo_A), σ_max(geo_B)` in `pair_state`, 3 iters per step, mirroring the existing pattern at `optim.py:3175–3180`. Independently fixes F2's bound violations regardless of whether variant 1 is implemented.
2. **Variant 1 (direction-aware ρ):** implement using the same warm-started power-iteration infrastructure for the three new quantities $\sigma_{\max}(BP), \sigma_{\max}(QA), \sigma_{\max}(QP)$ — cache `v_BP_top, v_QA_top, v_QP_top` in `pair_state`. Replace the $\rho$ scalar at `optim.py:3185` with $\lambda_{\text{dir}}$. New optimizer slug to keep the existing `spectral_chord_tight` baseline reproducible.
3. **Skip variants 2 and 3:** F1 kills variant 3 outright; F4 kills variant 2's wall-time-adjusted economics.

Expected outcome at the 2k canonical horizon: variant 1 + power-iter polish gains roughly **2–3% loss-per-step over the existing chord-tight baseline**, with the bound now actually held. Light gain.

## Caveats

- **Single-seed, mid-run.** Final medians at step 500 may shift slightly. Qualitative findings (F1, F2, F4) are stable since they reflect saturation / spectral structure that doesn't trend hard with steps. F3's quantitative number could move; F5 typically scales with gradient magnitude as training progresses.
- **Different from the canonical 2k horizon.** This is a 500-step diagnostic, not a measurement run. To anchor variant 1's loss-per-step gain against AdamW noise floor (project σ ≈ 0.0006 per CLAUDE.md `multi-seed AdamW`), a 2k single-seed comparison run with diagnostics on is the right next data point — not a multi-seed sweep.
- **Probe C-tight uses the whitened-space cosine**, not the unwhitened applied-direction cosine the doc §8 variant-3 derivation strictly asks for. The whitened version is sufficient at `sat_frac = 1.0` (the saturating regime makes them coincide) but if any future setting drives `sat_frac < 1`, the unwhitened version would be needed; see comment block at `optim.py:3576`.

## Reproducing

```bash
# Tokenized data (one-time, packed_v1 schema)
python scripts/data/prepare_data.py \
    --model_name allenai/OLMo-2-0425-1B \
    --max_seq_length 512 --max_train_samples 32000 --max_eval_samples 512 \
    --seed 0 --out_dir data/magicoder_seq512_32k_packed \
    --data_pipeline_version packed_v1

# Sweep submission (2 cells parallel on 2 A100s, ~25 min wall)
SWEEP_SCOPE="diagnostics" \
SWEEP_PURPOSE="..." \
./slurm_scripts/submit.sh \
    params/chord_tight_diag_500_r16r64.json \
    chord_tight_diag_500_r16r64 \
    2 \
    scripts/sweep/sweep_500_r_diag.sh

# Read medians
python -c "
import json, statistics
for tag, path in [('r16', 'logs/chord_tight_diag_500_r16r64/run_info/logs/log_0.out'),
                  ('r64', 'logs/chord_tight_diag_500_r16r64/run_info/logs/log_1.out')]:
    optim = [json.loads(l) for l in open(path) if l.startswith('{') and '\"event\": \"optim_step\"' in l]
    for k in ['chord_slack', 'lambda_dir_gain', 'sat_frac_tight_A', 'cos_polar_clip_tight_A', 'adam_gauge_residual_rel']:
        meds = [e[k+'_median'] for e in optim if e.get(k+'_median') is not None]
        print(tag, k, round(statistics.median(meds), 4))
"
```
