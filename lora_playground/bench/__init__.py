"""Standardized benchmark harness for the LoRA optimizer playground.

Single source for (a) full-step wall timing at the *production* config, (b)
inverse-sqrt accuracy on real snapshot Grams, and (c) feeding
``timing_registry.json``. Use ``python -m lora_playground.bench <subcommand>``.

  config.py       BenchConfig — production-config-first; emits build_optimizer
                  kwargs + a resolved-config banner (guards the "benchmark library
                  defaults" footgun).
  harness.py      full-step wall (fwd+bwd+opt+zero); make_batch / time_full_step.
  inverse_sqrt.py Higham-vs-eigh accuracy on real chord-tight snapshot Grams.
  registry.py     record_bench → timing_registry.json (tagged source=bench_step_wall,
                  wall_inclusive=false; never clobbers an authoritative train-log entry).
  cli.py          step-wall | inverse-sqrt | record.

Supersedes (migrate callers here; these will be removed once nothing references them):
  scripts/bench/bench_optimizer_step.py        -> bench.harness + cli step-wall
  scripts/bench/bench_higham_accuracy.py       -> bench.inverse_sqrt
  scripts/bench/higham_convergence_snapshot.py -> bench.inverse_sqrt
  scripts/bench/profile_curvature_whiten.py    -> bench.profile (pending)
  scripts/bench/section_profile_cw.py          -> bench.profile (pending)
The non-timing diagnostic snapshots (ssc_kappa_*, packing-curve, cuda-graphs)
are separate investigations and stay under scripts/bench/.
"""
from .config import BenchConfig, protagonist_config  # noqa: F401
from .registry import record_bench  # noqa: F401

__all__ = ["BenchConfig", "protagonist_config", "record_bench"]
