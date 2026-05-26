# Adaptive-c Optimizer Goal Progress

Objective: implement an adaptive-c SSC optimizer that beats fixed-c performance,
approaches exact $\kappa(c)$ performance, adds at most 2% overhead over fixed-c,
and avoids unstable NaN failure modes.

## Success Criteria

- **Optimizer path exists.** `train_lora.py` can select a cheap adaptive-c mode
  through the normal `--polar_method ssc --ssc_kappa ...` CLI.
- **Quality beats fixed-c.** At the selected comparison regime, best-LR loss and
  useful-LR robustness beat the fixed-c baseline.
- **Quality approaches exact $\kappa(c)$.** The cheap adaptive mode tracks the
  exact eigvalsh $\kappa(c)$ reference closely enough in loss/trajectory to be a
  plausible production substitute.
- **Overhead is $\le 2\%$ over fixed-c.** This must be measured end-to-end, not
  inferred from isolated kernels.
- **No unstable NaNs.** The mode must avoid the known kpar cached-c ratchet and
  pass targeted non-finite smoke/replay checks.

## Current Evidence

- Exact eigvalsh $\kappa(c)$ is the oracle. It avoided the known failing
  `c≈4e-20` kpar value on the failing tensor, where exact `c` was ordinary.
- Raw kpar is not production-ready. It was fast but had a real cached-c failure.
- Offline snapshot diagnostics favor a stateless stable-rank candidate:
  retuning the stable-rank target from `κ=0.6` to roughly `κ_bulk=0.75`
  reduced median c mismatch against exact `κ=0.6` to about `1.12x` on sampled
  snapshots.
- Exact group-median c is also plausible as a robustness policy, but by itself
  it still needs exact per-layer solves unless paired with a cheap estimator.

## Implementation Status

- Added `--ssc_kappa_solver stable_rank` as a stateless adaptive-c candidate.
- The solver computes c from normalized stable rank using a one-spike-plus-flat
  tail approximation, then applies the existing SSC MISR kernel.
- The `stable_rank` solver rejects cross-step refresh caching, Picard cache
  sharing, and cache EMA, so it does not reintroduce the kpar warm-cache ratchet.

## Verification Status

- Done: CPU unit tests for the stable-rank c formula and optimizer routing.
  Relevant subset:
  `python -m pytest tests/test_chord_tight_clean.py::test_stable_rank_c_exact_on_one_spike_flat_tail tests/test_chord_tight_clean.py::test_stable_rank_c_is_finite_on_low_rank_inputs tests/test_chord_tight_clean.py::test_stable_rank_solver_routes_without_cache_state tests/test_chord_tight_clean.py::test_ssc_adaptive_kappa_recovers_target tests/test_chord_tight_clean.py::test_ssc_adaptive_matches_fixed_c_round_trip -q`
  passed (`5 passed`).
- Done: `train_lora.py --help` exposes
  `--ssc_kappa_solver {eigvalsh,misr_bisect,stable_rank}`.
- Done: full local chord-tight test file passed:
  `python -m pytest tests/test_chord_tight_clean.py -x -q` (`17 passed`).
- Done: functional local-GPU smoke through `train_lora.py` completed with
  `--ssc_kappa 0.75 --ssc_kappa_solver stable_rank`, tiny fixtures,
  `max_steps=1`, and `--debug_abort_on_non_finite`; it emitted step-1 eval.
  This is a functional smoke only, not timing evidence.
- Done: Blackwell r=256 compiled timing bench against fixed-c with diagnostics
  off. Config: packed_v1, all-linear, batch `2×8`, seq 512, 50 steps,
  `--compile`, `--ssc_nsteps 20`, `--higham_iters 16`, fixed-c `c=0.3`
  versus stable-rank `κ=0.75`. Steady-state from the last eval interval:
  fixed-c `0.525967 s/step`, stable-rank `0.526484 s/step`, overhead
  `+0.10%`. Logs:
  `logs/bench_ssc_stable_rank/compiled_r256_steps50_fixedc.log` and
  `logs/bench_ssc_stable_rank/compiled_r256_steps50_stable_rank.log`.
- Done: matching-config Blackwell timing bench for the existing 4k comparison
  regime: picard 3, `--ssc_nsteps 10`, `--higham_iters 10`, diagnostics off.
  Fixed-c `c=0.1` was `0.546859 s/step`; stable-rank `κ=0.75` was
  `0.552898 s/step`, overhead `+1.10%`. Logs:
  `logs/bench_ssc_stable_rank/compiled_r256_p3n10h10_steps50_fixedc_c0p1.log`
  and
  `logs/bench_ssc_stable_rank/compiled_r256_p3n10h10_steps50_stable_rank_k0p75.log`.
- Running: 4k r=256 stable-rank LR sweep, job `6446818`, group
  `chord_tight_clean_ssc_stable_rank_r256_k075_lr3sweep_4k_blackwell`.
  Cells are `lr ∈ {1e-2, 3e-2, 1e-1}` with `κ=0.75`, `ssc_nsteps=10`,
  picard 3, `muon_ns_steps=5`, and diagnostics off. Startup verified all
  three logs have `execution_source_dirty=false` and
  `ssc_kappa_solver=stable_rank`. Latest checked progress: all three reached
  step 400. This is now auxiliary evidence only: the fixed-c and exact
  eigvalsh κ baselines selected for the main comparison use
  `muon_ns_steps=10`.
- Pending corrected run: add the same stable-rank LR grid with
  `muon_ns_steps=10`, matching the existing r=256 fixed-c and exact κ
  baselines. The corrected wrapper path has been smoke-tested at r=256,
  `max_steps=1`, `κ=0.75`, `ssc_nsteps=10`; the config command and
  `optimizer_config.ns_steps` both recorded `10`, and the smoke emitted a
  finite eval.
- Pending: training comparison across the useful LR neighborhood, not only
  single best LR.

## Next Gates

1. Submit the corrected `muon_ns_steps=10` stable-rank LR sweep.
2. Wait for the corrected sweep to finish.
3. Analyze stable-rank against existing fixed-c and exact eigvalsh
   $\kappa(c)$ r=256 4k sweeps for best-LR loss and LR robustness.
