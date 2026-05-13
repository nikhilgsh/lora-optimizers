# Chord-tight whiten r=256 early-time lag: mechanism + fix candidates

## The observation

At r=256, chord-tight whiten k=1 (Init[A]) lags AdamW and chord-tight no-whiten
in eval loss for the first ~1000-2000 steps before catching up by step 4000.
The lag is **whiten-specific** (no-whiten doesn't show it) and
**rank-specific** (r=64 doesn't show it).

Final losses at 4k are essentially equal across whiten / no-whiten / direction
once each cell's optimum lr is found; the lag is purely early-time.

## Hypothesized mechanism: wasted dA in B's small-σ subspace

The loss depends on `B·A` (the LoRA product). A change `dA` perturbs the loss
through `B·dA`. SVD B = U_B Σ_B V_B^T: components of dA aligned with
V_B-directions where σ_i(B) is small produce small `B·dA` — those updates are
**invisible to the loss**.

Chord-tight whiten preconditions `dA ∝ SB^{-1/2} · P_A` where
`SB^{-1/2} = V_B · D · V_B^T` and `D_i = (σ_i(B)² + δ)^{-1/2}`. **The largest
entries of D fall on B's smallest singular values** — the whitening operation
preferentially amplifies dA in exactly the directions where B can't transmit
the update to the loss.

**Why r=256 specifically:** at r=256 Init[A] (B=0 at step 0), B is severely
rank-deficient throughout early training. From the existing diagnostic
trajectories (commit `d2b0ebb`, group `r256_chord_whiten_k1_lrsweep_4k_blackwell`,
lr=1e-2):

| step | r=64 whiten stable_rank_B/r | r=256 whiten stable_rank_B/r |
|---|---|---|
| 80   | 0.385 (24.6/64)  | 0.143 (36.5/256) |
| 480  | 0.428            | 0.165            |
| 960  | 0.435            | 0.174            |

At r=64, B uses ~40% of its rank early. At r=256, only ~14-17%. The remaining
~83% of r=256 B-modes have σ_i(B) ≈ 0, regularized only by absolute δ=1e-6 →
`(0² + δ)^{-1/2}` ≈ 1000× amplification factor on dA in those directions.

## Order-of-magnitude estimate

At r=256 Init[A] whiten with default δ=1e-6, σ_max(B) ≈ 0.5, 36 active modes
above √δ, 220 near-null modes below:

- `‖dA‖_F² ≈ ρ²·trace(SB^{-1}) ≈ ρ²·220/δ` (dominated by null modes)
- `‖B·dA‖_F² ≈ ρ²·δ·36` (only active modes survive the suppression by B)
- Predicted ratio `‖B·dA‖_F / ‖dA‖_F ≈ √(36·δ²/220) ≈ δ·√(36/220) ≈ 4·10⁻⁷`

For no-whiten r=256 (`dA = -ρ·P_A`, no preconditioning):

- `‖dA‖_F ≈ ρ·√r = 16ρ`
- `‖B·dA‖_F ≤ σ_max(B)·‖dA‖_F ≈ 0.5·16ρ = 8ρ`
- Predicted ratio `≈ 0.5`

**Predicted ratio gap:** ~10⁻⁶ (whiten) vs ~0.5 (no-whiten) — six orders of
magnitude. If the mechanism is right, the new probe `frac_dA_through_B` will
show this gap directly.

## Why Init[AB] doesn't fix it

Init[AB] sets B ~ N(0, 1/√r), giving B full random rank at step 0. By the
mechanism, this should ELIMINATE the wasted-dA problem (no near-null modes).

Empirically (group `r256_chord_whiten_k1_initAB_4k_blackwell`, lr=1e-2):

| step | Init[A] | Init[AB] | Δ |
|---|---|---|---|
| 200 | 0.6054 | 0.6157 | +0.010 (AB worse) |
| 800 | 0.5656 | 0.5812 | +0.016 |
| 2400 | 0.5285 | 0.5462 | +0.018 |

Init[AB] is uniformly **worse**, not better. Reading `stable_rank_B/r` at the
same step: Init[A] = 0.14-0.17 (rank-deficient as expected), Init[AB] = 0.48-0.55
(rank-rich as expected). So the mechanism's premise holds (Init[AB] has rich B),
but the empirical outcome is opposite. This means Init[AB] has its own,
separate penalty — most likely **random B is misaligned with the loss-optimal
B-subspace**, and the optimizer must un-rotate the random init before progress.
That penalty is persistent (visible at every step) and outweighs the gain from
avoiding wasted dA.

So Init[AB] solves one problem and introduces a worse one. Not a fix candidate.

## Proposed fix: σ_max-relative damping (ε_rel)

Default damping is absolute `δ = 1e-6`. Change to relative:
`δ_eff = ε_rel · σ_max(SB)` (the `--precond_delta_relative` flag, with
`--precond_delta` setting ε_rel).

With σ_max(B) ≈ 0.5 (σ_max(SB) ≈ 0.25), the null-mode amplification becomes
`(σ_max(SB) · ε_rel)^{-1/2}`:

| ε_rel | δ_eff | null-mode amp | active-mode perturbation |
|---|---|---|---|
| 1e-6 (default abs)  | 1e-6  | ~1000× | negligible |
| 1e-4 (weak rel)     | 2.5e-5 | ~200×  | ~0.01% |
| 1e-3                | 2.5e-4 | ~63×   | ~0.1%  |
| **1e-2 (sweet)**    | **2.5e-3** | **~20×** | **~1%** |
| 1e-1 (aggressive)   | 2.5e-2 | ~6×    | ~10%   |

ε_rel = 1e-2 reduces null-mode amplification ~50× while only perturbing the
active spectrum by ~1%. Should substantially shrink the wasted-dA problem.

## Test plan

Cells added to `slurm_pending/contam_rerun_initAB_extend_4k_blackwell.sbatch`:

| cell | r | optimizer | k | lr | init | δ | ε_rel mode |
|---|---|---|---|---|---|---|---|
| 11 | 256 | whiten | 1 | 1e-2 | A (zero) | 1e-3 | on |
| 12 | 256 | whiten | 1 | 1e-2 | A (zero) | 1e-2 | on |
| 13 | 256 | whiten | 1 | 1e-2 | A (zero) | 1e-1 | on |

All at HEAD post-fix. New diagnostic `frac_dA_through_B = ‖B·dA‖_F / ‖dA‖_F`
logged per probe step (every 80 steps); same for `frac_dB_through_A`.

**Predictions:**
1. At ε_rel = 1e-6 (existing baseline at lr=1e-2, group
   `r256_chord_whiten_k1_lrsweep_4k_blackwell` cell 1), `frac_dA_through_B`
   should be ≪ 1e-4 in early steps.
2. At ε_rel = 1e-2, `frac_dA_through_B` should rise toward the no-whiten
   r=256 value (~0.5).
3. Early-time eval loss should improve monotonically with ε_rel within
   `[1e-3, 1e-2]`; at 1e-1 the active-mode perturbation may start to cost
   something.
4. Final loss at 4k should match or beat the default-damping baseline
   (0.5117) for some ε_rel.

## Diagnostic implementation

`_emit_basic_diagnostics` in `lora_playground/optim.py` now logs:

```python
B_dA = B @ dA       # shape (d_out, d_in)
dB_A = dB @ A       # shape (d_out, d_in)
rec["frac_dA_through_B"] = ‖B·dA‖_F / (‖dA‖_F + 1e-30)
rec["frac_dB_through_A"] = ‖dB·A‖_F / (‖dB‖_F + 1e-30)
```

Cost: 2 bmm + 4 Frobenius norms per pair per probe step. Negligible at
default `optim_diagnostics_every=80`.

## Sticky-zero bug not relevant here

The chord-tight whiten r=256 lag predates the warm-start sticky-zero bug
(commit `766b016`, 2026-05-12 20:52). The observed lag is in old data at
commit `d2b0ebb` (2026-05-12 20:03), before warm-start was introduced.
The lag is a real algorithmic property of chord-tight whiten + rank-deficient
B, not a bug artifact.
