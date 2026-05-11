# Is the whitening load-bearing? (Preliminary finding, 2026-05-11)

> **Status: provisional, single-seed, in-flight.** Based on partial Blackwell 4k runs (jobs 6382679 chord-tight, 6382741 chord-direction, 6382805 no-whitening). r=16 data through step 800, r=64 data through step 400 at time of writing. Pattern is consistent across all available eval points but the final-horizon picture (step 4000) may shift.

## TL;DR

Removing the `S^{-1/2}` whitening from the chord-tight LoRA optimizer entirely — setting `SA^{-1/2} = SB^{-1/2} = I`, so the polar map operates on the raw Adam direction without any factor-Gram preconditioning — produces **essentially equivalent training loss** to the full chord-tight machinery and **matches variant 1 (direction-aware ρ)** within ~1σ_AdamW. The "tight chord" optimizer's elaborate whitening / unwhitening pipeline is correct in the variational sense (algorithm_tight_chord.md §5 Lemma 2) but practically not load-bearing at this scale.

Surprising consequence: **the chord-tight v0 baseline is the worst of three closely-related optimizers**, and the geometric-machinery cost was apparently not buying us anything in eval loss.

## The three optimizers compared

All three share the same Adam direction, the same polar (Newton-Schulz) map, the same chord-tight ρ scalar, and the same Picard cross-coupling outer loop. They differ only in the per-block direction:

| Optimizer | Per-block direction `dA` | Slug |
|---|---|---|
| **v0 chord-tight** | `−ρ · S_B^{-1/2} · polar(S_B^{-1/2} u_A) / ‖S_B^{-1/2} polar(S_B^{-1/2} u_A)‖_2` | `adam-polar-product-lora-coupled-spectral-chord-tight` |
| **v1 chord-direction** | as v0 but with the worst-case scalar `ρ` replaced by direction-aware `λ_dir` (variant 1 of `algorithm_tight_chord.md`) | `adam-polar-product-lora-coupled-spectral-chord-direction` |
| **no-whitening ablation** | `−ρ · polar(u_A) / ‖polar(u_A)‖_2 = −ρ · polar(u_A)` (Adam direction polar-mapped, no S^{-1/2}) | `adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening` |

All three respect the same `‖ΔW‖_2 ≤ lr` chord-tight magnitude bound (v0, v1 by their direction-aware constructions; no-whitening because dA has `‖dA‖_2 = ρ` from the polar map's normalization).

## Data

Same seed (0), same config (lr=3e-3, lora_r ∈ {16, 64}, lora_alpha=lora_r, packed_v1, sdpa, Blackwell, 4k steps planned). Single seed.

### r=16 evals (σ_AdamW ≈ 0.0006, so 1σ ≈ 0.0006)

| step | v0 chord-tight | v1 chord-direction | no-whitening | Δ(v1−v0) | Δ(nowhite−v0) | Δ(v1−nowhite) |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 0.6022 | 0.6001 | 0.6007 | −0.0021 (−3.5σ) | −0.0015 (−2.5σ) | −0.0006 (−1.0σ) |
| 400 | 0.5851 | 0.5825 | 0.5835 | −0.0026 (−4.3σ) | −0.0016 (−2.7σ) | −0.0010 (−1.7σ) |
| 600 | 0.5756 | 0.5732 | 0.5741 | −0.0024 (−4.0σ) | −0.0015 (−2.5σ) | −0.0009 (−1.5σ) |
| 800 | 0.5693 | 0.5666 | 0.5676 | −0.0027 (−4.5σ) | −0.0017 (−2.8σ) | −0.0010 (−1.7σ) |

### r=64 evals (σ_AdamW ≈ 0.0007)

| step | v0 chord-tight | v1 chord-direction | no-whitening | Δ(v1−v0) | Δ(nowhite−v0) | Δ(v1−nowhite) |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 0.5981 | 0.5962 | 0.5961 | −0.0019 (−2.7σ) | −0.0020 (−2.9σ) | +0.0001 (~0σ) |
| 400 | 0.5815 | 0.5794 | 0.5794 | −0.0021 (−3.0σ) | −0.0021 (−3.0σ) | 0.0000 (~0σ) |

### Two readings of the data

1. **v0 chord-tight (full whitening) is the WORST of the three at every eval point.** Both v1 and no-whitening beat it by 2.5–4.5σ_AdamW. The pattern is consistent across all available eval points.
2. **v1 and no-whitening are tied at r=64, and v1 leads no-whitening by only ~1.5σ at r=16.** The bulk of variant 1's advantage over v0 is *not* coming from the direction-aware bound — it's coming from removing whitening (effectively, by changing what `geo_A / ‖geo_A‖_2` normalizes to).

## What this implies

> *Speculation; intended as hypotheses to test, not conclusions.*

The chord-tight v0 update is:

$$
dA \;=\; -\rho \cdot \frac{S_B^{-1/2}\,\mathrm{polar}(S_B^{-1/2} u_A)}{\lVert S_B^{-1/2}\,\mathrm{polar}(S_B^{-1/2} u_A)\rVert_2}
$$

The no-whitening ablation is:

$$
dA \;=\; -\rho \cdot \mathrm{polar}(u_A)
$$

Both have `‖dA‖_2 = ρ`, so they take the same spectral-step magnitude. The *direction* differs:
- v0's direction is `polar(S_B^{-1/2} u_A)` then unwhitened (i.e., warped back into the original LoRA factor space).
- no-whitening's direction is just `polar(u_A)`.

A few candidate explanations for why these are nearly equivalent on eval loss:

- **Polar map "homogenizes" the direction enough that the whitening pre-step doesn't change the final direction by much.** Polar projects `M = U Σ V^⊤ → U V^⊤`, which preserves only the singular *vectors* and discards all singular-value information. So `polar(S^{-1/2} u_A)` and `polar(u_A)` differ by how much `S^{-1/2}` rotates the top singular vectors of `u_A`. If `S` is well-conditioned (which the in-training higham probe will tell us), the rotation is small.

- **The variational program's *meaning* depends on whitening, but the *optimum it solves to* doesn't differ much from the no-whitening optimum at this scale.** The variational argument says "whitening is the unique linear change of variable that makes the constraint a clean op-norm ball." That's mathematically true. But the resulting update may not differ meaningfully from a simpler update that just polar-projects the raw Adam direction.

- **v0 might actually be *hurt* by whitening when S_B is near-singular** (e.g., early training when B ≈ 0 makes S_B ≈ δI and S_B^{-1/2} ≈ δ^{-1/2} I). Then S_B^{-1/2} u_A is just `δ^{-1/2} u_A`, which gets polar-mapped to the same direction as `u_A` (polar is scale-invariant). So actually for *early* training the v0 and no-whitening should agree, and the gap should *open* later when S_B is non-trivially conditioned. Looking at the data, v0 is consistently 2-3σ behind throughout — no opening gap is visible. So this hypothesis isn't strongly supported.

The cleanest interpretation: **whitening makes the variational program well-posed but doesn't help training**.

## Implications for the broader investigation

1. **(B) Higham accuracy is moot.** If whitening doesn't affect training loss, then the accuracy of the higham-vs-eigh `S^{-1/2}` solver doesn't matter for training quality. The in-training higham accuracy probe is still useful as a sanity check but not load-bearing.

2. **Variant 1's gain over v0 is mostly explained by removing whitening — not by the direction-aware bound.** Variant 1 is doing two things implicitly: (a) using a tighter ρ via the direction-aware bound, and (b) the direction-aware bound is computed on `geo_A` which still goes through whitening. So variant 1 vs v0 mixes both effects. To isolate variant 1's direction-aware contribution, we'd want a **direction-aware + no-whitening** variant — solve `a·λ + b·λ² = lr` with `P = polar(u_A) / ‖polar(u_A)‖_2`. That comparison would isolate the direction-aware bound's true gain.

3. **The chord-tight optimizer's structural premise — that the whitened polar map produces a meaningfully different update than raw polar — is at least partly empirically wrong.** The variational interpretation is correct; the assumption that "the right variational solution gives better training" is what fails.

4. **Skip variant 2** stands: no-whitening already gets most of variant 1's gain at near-zero implementation cost, and variant 2's exact-chord-norm would buy at most ~2-5% on top of variant 1.

## Caveats and pending verification

- **Single seed.** The 1–2σ_AdamW difference between v1 and no-whitening at r=16 could be noise. The 3–4σ_AdamW difference between v0 and the others is more robust (consistent across 4-7 data points and direction-stable). Multi-seed would be a more rigorous test.
- **Mid-training, not 4k.** Final-horizon picture may shift, especially if S becomes ill-conditioned later (the higham accuracy probe will tell). Will update this doc once the three Blackwell runs complete (~2.5 hr from writeup).
- **All three optimizers share the same chord-tight ρ.** This means the loss differences are not from different step magnitudes (they're identical) but from different step *directions* — `polar(u_A)` vs `polar(S^{-1/2} u_A)` (unwhitened). The "no-whitening" ablation is really a "no S^{-1/2}-warping-of-the-direction" ablation, not a "no preconditioning" ablation.
- **The higham accuracy probe (added in commit 82bc6eb) will be in the *next* round of runs, not these three.** When we get data on `cond(S_A), cond(S_B)` across the canonical horizon, we can rule out / confirm whether ill-conditioning of S is even something the optimizer encounters.

## Reproducing

Log paths:
- v0: `logs/chord_tight_diag_4k_r16r64_blackwell` (job 6382679)
- v1: `logs/chord_direction_4k_r16r64_blackwell` (job 6382741)
- no-whitening: `logs/chord_nowhitening_4k_r16r64_blackwell` (job 6382805)

Eval-trajectory readout:
```python
from lora_playground.loader import load_runs
runs = load_runs(where={"optimizer": [
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    "adam-polar-product-lora-coupled-spectral-chord-direction",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
]})
# Group by optimizer × lora_r, plot eval_loss trajectory.
```
