# Chord-tight whiten r=256 early-time lag: mechanism + fix candidates

## The observation

At r=256, chord-tight whiten k=1 (Init[A]) lags AdamW and chord-tight no-whiten
in eval loss for the first ~1000-2000 steps before catching up by step 4000.
The lag is **whiten-specific** (no-whiten doesn't show it) and
**rank-specific** (r=64 doesn't show it).

Final losses at 4k are essentially equal across whiten / no-whiten / direction
once each cell's optimum lr is found; the lag is purely early-time.

## Empirical fingerprint at r=256 step 80, lr-best per cell

| field | whiten (lr=1e-2) | no-whiten (lr=3e-3) | AdamW (lr=1e-4) |
|---|---|---|---|
| eval_loss | 0.6054 | 0.5950 | 0.5926 |
| `cos_A_median` (dA vs plain Adam direction) | **0.538** | 0.977 | — |
| `stable_rank_B_median / r` | **14%** (36/256) | 55% (141/256) | — |
| `sat_frac_tight_A_median` | 0.815 | 0.012 | — |
| `norm_B_median` | 2.04 | 9.25 | — |
| `gamma_A_median` (Picard cross-term) | 1.149 | 0.255 | — |

Most striking: **whiten's dA is rotated 60° from plain Adam direction; no-whiten's is essentially aligned.** This is the load-bearing distortion the loss curve is reading.

## Load-bearing mechanism

`SB^{-1/2}` inverts B's emphasis, and the Adam direction is naturally aligned with B's emphasis. So whitening rotates dA *against* the gradient signal.

1. **Gradient signal direction.** `∂L/∂A = B^T · (∂L/∂output)`. The row-space of `gA` (and `u_A`) is naturally aligned with col(B) — directions where B has support.
2. **Effect of `SB^{-1/2}` on u_A.** `SB^{-1/2} = V_B · D · V_B^T` with `D_i = (σ_i(B)² + δ)^{-1/2}`. The largest entries of D fall on B's smallest singular values — opposite to where the gradient signal sits. So `SB^{-1/2}·u_A` rotates u_A *away* from B's strong modes, *toward* B's weak modes.
3. **NS doesn't undo it.** `polar(SB^{-1/2}·u_A) ≠ SB^{-1/2}·polar(u_A)`. The polar of the rotated input has its own rotated singular vector basis; multiplying by `SB^{-1/2}` again to "unwhiten" gives `geo_A = SB^{-1/2}·polar(SB^{-1/2}·u_A) ≠ polar(u_A)`. The two `SB^{-1/2}`'s don't compose to identity through the non-linear polar. (NS itself produces a near-perfect polar in both whiten and no-whiten — `cos_polar_clip_A_median ≈ 0.92` in both at r=256 step 80, `xunc_A_smax = 1.0` confirming NS input is correctly normalized. NS quality is not the failure mode; what differs is the post-polar `SB^{-1/2}·P_A` reshaping and chord rescaling.)
4. **Net dA direction.** dA ends up at a substantial angle from `polar(u_A)` — from where plain Adam would point.

**Why r=64 doesn't show it, r=256 does:** the anisotropy of `SB^{-1/2}` is bounded by `√(σ_max(B)²+δ) / √(σ_min(B)²+δ)`. At r=64 step 80 with B using ~40% of its rank, the spread ratio is small (~few). At r=256 with B using ~14% of its rank, the spread ratio is ~340 at default damping. Larger spread → more rotation of `u_A` by `SB^{-1/2}` → larger angular displacement of dA from gradient direction.

**Why this is whiten-specific:** no-whiten skips `SB^{-1/2}` entirely. `dA = -ρ · polar(u_A)` has no rotation away from Adam direction.

## dA magnitude is renormalized, not amplified

The chord-tight update is

```
dA = -(ρ / σ_max(geo_A)) · geo_A
```

renormalized so `σ_max(dA) = ρ`. Hence `‖dA‖_F = ρ·√(stable_rank(geo_A)) ≤ ρ·√r`, independent of `trace(SB^{-1})`: the renormalization cancels any magnitude blow-up of `dA` from `SB^{-1/2}` amplification. The whiten-specific effect is on the step *direction* (rotation away from the Adam direction), not its magnitude.

What `SB^{-1/2}` actually does is reshape dA's **direction** (the spectrum/anisotropy of `geo_A` after renormalization), not its overall magnitude. The load-bearing effect is the angular rotation described above, not a magnitude inflation.

## ε_rel damping does not fix the dominant distortion

`ε_rel` (σ_max-relative damping) reduces `SB^{-1/2}`'s eigenvalue spread:

| ε_rel | δ_eff at σ_max(SB)≈0.25 | spread ratio |
|---|---|---|
| 1e-6 (default abs) | 1e-6 | ~340 |
| 1e-3 | ~2.5e-4 | ~34 |
| 1e-2 | ~2.5e-3 | ~11 |
| 1e-1 | ~2.5e-2 | ~3.4 |

But reducing the spread doesn't change the *direction* in which `SB^{-1/2}` rotates — only the magnitude of rotation. And `SB^{-1/2}`'s anti-B-emphasis structure is preserved at any spread.

Empirical: ε_rel = 1e-3 at r=256 whiten lr=1e-2 (partial data through step 800):
| step | whiten default | whiten ε_rel=1e-3 | Δ |
|---|---|---|---|
| 200 | 0.6054 | 0.6036 | −0.0018 |
| 400 | 0.5889 | 0.5875 | −0.0013 |
| 600 | 0.5751 | 0.5747 | −0.0004 |
| 800 | 0.5656 | 0.5654 | −0.0002 |

The improvement is ~0 in the region the lag matters. The no-whiten baseline at step 800 is 0.5601 — ε_rel damping closes 0.0002 of a 0.0055 gap. Not the fix.

## Diagnostic probe (added)

`_emit_basic_diagnostics` now logs two fields per pair per probe step:

```
frac_dA_through_B = ‖B·dA‖_F / (‖dA‖_F + ε)
frac_dB_through_A = ‖dB·A‖_F / (‖dB‖_F + ε)
```

These measure "fraction of the per-factor update that actually reaches the LoRA forward product." Combined with `cos_A` (already logged), they characterize the whitening-induced rotation of A's update.

Useful for future analysis — won't decide the chord-tight whiten lag question on its own.

## Fix candidates (named, not yet tested)

The mechanism points away from "tune ε_rel" and toward "change which preconditioner you whiten with":

- **Rank-deficiency-aware bypass.** Detect `stable_rank_B / r < threshold` and skip `SB^{-1/2}` for that step (fall back to no-whiten). Threshold tuning needed.
- **Co-aligned preconditioner.** Replace `SB^{-1/2}` with `(SB + ε·I)^{1/2}` (co-emphasizes B's strong modes) or with the projector onto col(B). This co-aligns dA with the gradient direction instead of inverting.
- **Hybrid.** Linear interpolation `α·I + (1−α)·SB^{-1/2}` with α set by rank-deficiency. α=1 reduces to no-whiten; α=0 reduces to full whiten.

These have theoretical justifications worth working out before implementation. The current ε_rel sweep cells in v2 will give a clean answer on whether damping helps (it does not, per partial data); the next investigation step would be one of these alternative preconditioners on a small (r=256, whiten k=1, lr=1e-2) ablation.

## Note on bug interaction

The chord-tight whiten r=256 lag predates the sticky-zero warm-start bug
(`766b016`, 2026-05-12 20:52). The observed lag is in pre-bug data at commit
`d2b0ebb`, so it's a real algorithmic property of chord-tight whiten +
rank-deficient B, not a bug artifact.
