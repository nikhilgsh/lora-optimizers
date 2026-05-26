# Kappa-Adaptive SSC Handoff

This note summarizes the adaptive-$\kappa$ SSC campaign around the `kpar`
solver: what was attempted, what is now known, and what should happen next.

Treat older campaign notes and stale sbatches as hypotheses. The current
recommendation is not "run one more kpar validation." It is to keep fixed-$c$ as
the stable baseline, keep exact eigvalsh $\kappa$ as the reference/oracle, and
quarantine the `kpar` fast solver until a simpler adaptive rule is designed.

## Terms

- `SSC`: the shifted-scaled Newton-Schulz polar core used by
  `--polar_method ssc`.
- $c$: the SSC shift/scale parameter. Fixed-$c$ uses `--ssc_c`.
- Adaptive $\kappa$: chooses $c$ to target
  `--ssc_kappa`, with
  $$
  \kappa(c) = {1 \over r}\sum_i \left({h_c(s_i) \over h_c(1)}\right)^2,
  \qquad
  h_c(s) = {s \over \sqrt{1 + (s/c)^2}}.
  $$
- `kpar`: the fast parallel-grid MISR/bisect approximation:
  `--ssc_kappa_solver misr_bisect --ssc_kappa_bisect_mode parallel`.
- Picard: the coupled A/B inner iteration controlled by
  `--picard_iters_override`.

## Why We Tried Adaptive $\kappa$

The motivation was not only best learning rate. The notebook comparison in
`notebooks/packed_v1_leaderboard.ipynb` shows that fixed-$c$ wants different
values by rank, while $\kappa$ looked closer to a rank-invariant knob.

Current packed-v1 4k evidence, recomputed through the canonical loader using
the `_ssc_kappa_vs_fixed_c` filters:

| rank | best fixed-$c$ row | best adaptive-$\kappa$ row | readout |
|---:|---|---|---|
| 16 | `c=0.3`, loss `0.517825` | `kappa=0.75`, loss `0.518301` | fixed-$c$ slightly wins at best LR |
| 64 | `c=0.3`, loss `0.506233` | `kappa=0.6`, loss `0.507153` | fixed-$c$ slightly wins at best LR |
| 256 | `c=0.1`, loss `0.497904` | `kappa=0.6`, loss `0.496682` | adaptive $\kappa$ wins |

At rank 256, $\kappa=0.6$ is not just a cherry-picked best-LR point. Against
fixed `c=0.1`, it is competitive or better across most of the useful LR grid:

| LR | fixed `c=0.1` | `kappa=0.6` |
|---:|---:|---:|
| `3e-4` | `0.543072` | `0.538526` |
| `1e-3` | `0.516910` | `0.515095` |
| `3e-3` | `0.503907` | `0.503905` |
| `1e-2` | `0.498550` | `0.498348` |
| `3e-2` | `0.497904` | `0.496682` |
| `1e-1` | `0.510220` | `0.507663` |

So the fixed-$c$ path is clean enough as a baseline, but stopping there gives
up a plausible rank-invariance/performance benefit, especially at `r=256`.

## What We Accomplished

The main debugging result is that the NaN is no longer an unreproducible
training mystery. We now have enough instrumentation to localize it to an
optimizer-side adaptive-$c$ failure.

Concrete accomplishments:

- Added/repaired replay mechanics so fresh checkpoints can preserve RNG state,
  dataloader alignment, and optimizer transient state such as
  `ssc_c_cached_*`, sigma warm starts, and Picard slots.
- Added non-finite optimizer snapshots with Picard intermediate tensors, so the
  first bad tensor can be inspected instead of inferred from later model NaNs.
- Added a guard for unsafe batched sigma-max power iteration. This fixed a real
  unsafe denominator class, but it was not sufficient to fix the kpar NaN.
- Replayed the failing region from a fresh checkpoint and reproduced the
  failure with enough state to identify the adaptive-$c$ collapse.
- Verified that exact eigvalsh $\kappa$ on the failing tensor does not ask for
  tiny $c$; the tiny $c$ comes from the kpar/cache approximation.

Relevant code/tests touched during this campaign include:

- `lora_playground/checkpoint.py`
- `lora_playground/train.py`
- `lora_playground/optim.py`
- `lora_playground/spectral.py`
- `tests/test_checkpoint.py`
- `tests/test_train_helpers.py`
- `tests/test_chord_tight_clean.py`
- `tests/test_sigma_max_power_iter.py`

There are still local kpar guard edits in the worktree. Those should be treated
as debugging scaffolding, not as the chosen production fix.

## The NaN Issue

The useful failing replay is:

```text
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_picard_trace_replay_1750_1772_6446685
```

The decisive snapshot is:

```text
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_picard_trace_replay_1750_1772_6446685/snapshots/step001771_pair069_non_finite_intermediate_base_model.model.model.layers.9.mlp.down_proj_default_.pt
```

At step 1771, pair 69:

- `X_B_eff_n0` was finite, with norm about `1.08007`.
- `ssc_c_B_n0` was `3.96389e-20`.
- `P_B_n0`, `geo_B_n0`, `op_geoB_b_n0`, and `dB_n0` were NaN.
- A-side Picard n0 was finite: `ssc_c_A_n0=0.0324722`.
- B-side Picard n1 was finite: `ssc_c_B_n1=0.00917727`.

Exact eigvalsh $\kappa$ on that same finite `X_B_eff_n0` gives ordinary values:

| target $\kappa$ | exact $c$ |
|---:|---:|
| `0.6` | `0.009387386` |
| `0.8` | `0.004495496` |
| `0.9` | `0.002638741` |
| `0.95` | `0.001673373` |
| `0.99` | `0.000565130` |

The same spectrum has numerical rank `255/256`, so as $c \to 0$,
`kappa(c)` tends to about `0.99609375`, not to `0.6`. In other words, the true
solver would not keep decreasing $c$ at this point. The downward motion is a
bug in the approximate cached kpar path, not evidence that the exact
$\kappa=0.6$ target mathematically wants $c \approx 0$.

The cached-state audit explains the exact tiny value:

```text
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_sigma_guard_validate_2k/checkpoints/ckpt_step1750/optimizer.pt
```

In group 2, local index 9, the cached value was:

```text
ssc_c_cached_B_n0 = 4.829003384890032e-19
```

Five refreshes later, repeated lower-grid-edge selections multiply by
`exp(-0.5)^5`, predicting:

```text
4.829003384890032e-19 * exp(-0.5)^5 = 3.964e-20
```

That matches the failing `ssc_c_B_n0 = 3.96389e-20`.

## Mechanism Belief

The best current explanation is:

1. The warm-started kpar cache reached an invalid tiny-$c$ region.
2. In that region, the kpar MISR scorer no longer represented the true
   $\kappa(c)$ curve.
3. The scorer computes a ratio of clamped small quantities. At tiny $c$, it
   collapsed to about `1/r = 0.00390625` instead of the true
   rank-normalized limit near `255/256`.
4. That false low score made lower $c$ look attractive. In parallel-grid
   mode, ties selected the lower grid edge.
5. Repeated refreshes ratcheted $c$ down by about `exp(-0.5)` per bad
   refresh until terms involving `1/c^2` overflowed inside `_ssc_misr_batched`.

This answers the apparent mathematical contradiction: if the true solution is
larger, the exact solver should increase $c$. The exact solver would. The
kpar approximation lost the sign of the correction once its scorer entered an
invalid numeric domain.

## What Was Ruled Out

The sigma-max power-iteration path was worth guarding, but it is not the whole
NaN cause. A validation run with the sigma guard still failed:

```text
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_sigma_guard_validate_2k
```

The failure at step 1771 led to the snapshot above. So the sigma guard should
not be described as "the fix." It is one safety improvement that exposed the
deeper kpar cached-$c$ failure.

The old step-1250 replay is also not authoritative. That checkpoint predates
complete persistence of adaptive-$c$ transient state, so it can match token
order and scalar losses while missing the exact optimizer trajectory.

## Timing Context

The reason kpar was tempting is that it appeared to hit the timing target.
Measured on bs=32 200-step logs, using the last 50-step interval:

| run | log | s/step | Adam-normalized |
|---|---|---:|---:|
| Adam | `logs/bench_ssc_drift/v5_bs32_adamw_r256_200.log` | `0.8153` | `1.0000x` |
| fixed-c Picard=2 | `logs/bench_ssc_drift/v5_bs32_fixedc_n10_h10_p2_r256_lr3e-2_200.log` | `0.8974` | `1.1007x` |
| kpar K3/R5/W5 Picard=2 | `logs/bench_ssc_drift/v5_bs32_kpar_split_K3_R5W5_n10eval20_h10_p2_r256_lr3e-2_200.log` | `0.9085` | `1.1143x` |

Adaptive tax over fixed-$c$ was `0.0111` s/step, or `+0.0136x` Adam.

The broader profile in `docs/notes/polar_product/walltime_profile.md` has the
same message: exact eigvalsh $\kappa$ is too expensive for the fast path, while
kpar was near fixed-$c$ cost. That timing success is real, but it does not
outweigh the stability/complexity failure.

## Recommendations

### 1. Keep Fixed-$c$ As The Stable Baseline

Fixed-$c$ is clean and verified enough to use as the immediate NaN-free
baseline. The relevant stable run is:

```text
logs/bench_ssc_drift/v5_bs32_fixedc_n10_h10_p2_r256_lr3e-2_2k.log
```

It completed cleanly to 2k with final `eval_loss=0.503575` in that bs=32 debug
regime.

This does not mean fixed-$c$ is scientifically final. It still needs rank
retuning, and current 4k packed-v1 sweeps show adaptive $\kappa=0.6$ is stronger
at `r=256`.

### 2. Do Not Promote kpar

Do not keep pushing the current warm-started kpar solver as the production fast
adaptive path. It has too many behavior-defining moving parts:

- cached $c$
- refresh cadence
- grid width
- warmup behavior
- approximate scorer validity domain
- edge selection behavior
- hidden floors/clamps

Adding more clamps can make the observed NaN disappear, but that would turn the
algorithm into a collection of secret hyperparameters. The campaign already got
bitten by hidden trust domains such as old `[1e-3, 1e3]` style bounds.

If kpar remains in the codebase, mark it experimental and make invalid-domain
events loud in diagnostics. It should not be the default route for new science
runs.

### 3. Keep Exact eigvalsh $\kappa$ As The Oracle

Exact eigvalsh $\kappa$ is the right reference for debugging and saved-snapshot
analysis. It gave the decisive answer on the failing tensor: the true
`kappa=0.6` solution was around `9e-3`, not `4e-20`.

It is probably too slow for the intended production fast path at `r=256`, but
that is a timing fact, not a reason to discard it as an oracle.

### 4. Design A Simpler Adaptive-$c$ Rule

The next algorithmic direction should be simpler than kpar. The desiderata:

- no warm-started per-pair cache required for correctness
- no parallel grid with edge-ratchet failure mode
- no hidden trust-domain hyperparameters
- exact behavior is clear as $c \to 0$ and as spectra become low-rank
- cheap enough to run at `r=256` without the eigvalsh tax
- instrumented so the selected $c$ can be compared to exact eigvalsh on saved
  snapshots

The most plausible direction is a moment/bulk approximation to $\kappa$, using
stable quantities like Frobenius energy, spectral norm, and rank-normalized
energy. A group-level $c$ by shape/side/Picard is another candidate if it
removes per-pair ratchets, though it may still need exact eigvalsh unless the
group statistic is moment-based.

Do not claim this simpler adaptive rule works yet. The right next step is a
snapshot/offline comparison against exact eigvalsh before any new GPU run.

## Suggested Next Loop

1. Freeze kpar production submissions.
2. Preserve the replay and snapshot instrumentation; it is useful independent
   of the final adaptive rule.
3. Audit the current local kpar guard edits and decide what is pure
   instrumentation versus what should be reverted or isolated behind an
   experimental flag.
4. Write a small offline diagnostic that loads saved finite `X_A_eff` /
   `X_B_eff` tensors and compares:
   fixed $c$, exact eigvalsh $\kappa$, and one or two simple adaptive
   moment-based candidates.
5. Only after the offline diagnostic looks sane, run a bounded GPU smoke
   through `train_lora.py` with non-finite snapshots enabled.
6. Compare against fixed-$c$ and exact eigvalsh reference behavior before
   putting any adaptive candidate back into sweeps.

## Evidence Pointers

Snapshot inspector:

```bash
python scripts/analysis/inspect_optimizer_snapshot.py \
  logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_picard_trace_replay_1750_1772_6446685/snapshots/step001771_pair069_non_finite_intermediate_base_model.model.model.layers.9.mlp.down_proj_default_.pt
```

Notebook motivating adaptive $\kappa$:

```text
notebooks/packed_v1_leaderboard.ipynb
```

Main failing/replay artifacts:

```text
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_nan_fast_2k.log
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_sigma_guard_validate_2k
logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_picard_trace_replay_1750_1772_6446685
```

Timing artifacts:

```text
logs/bench_ssc_drift/v5_bs32_adamw_r256_200.log
logs/bench_ssc_drift/v5_bs32_fixedc_n10_h10_p2_r256_lr3e-2_200.log
logs/bench_ssc_drift/v5_bs32_kpar_split_K3_R5W5_n10eval20_h10_p2_r256_lr3e-2_200.log
docs/notes/polar_product/walltime_profile.md
```

Stable fixed-$c$ debug baseline:

```text
logs/bench_ssc_drift/v5_bs32_fixedc_n10_h10_p2_r256_lr3e-2_2k.log
```

## Open Questions

- Can a no-cache moment/bulk adaptive-$c$ rule recover most of the
  `r=256`, `kappa=0.6` advantage?
- Does group-level $c$ remove enough per-pair sensitivity to be useful, or is
  it just fixed-$c$ with another name?
- Which kpar guard edits should remain as defensive diagnostics, and which
  should be reverted to avoid implying kpar is production-ready?
- After a simpler adaptive rule exists, does it still preserve the low overhead
  that made kpar attractive?
