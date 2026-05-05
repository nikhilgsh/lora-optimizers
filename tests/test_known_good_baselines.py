"""Canary regression tests for known-good headline numbers.

The synthesis (`docs/notes/optimizer_synthesis.md`) cites specific final
eval-loss values as headline references. If a fixture, dedup logic, or
filter is silently changed in a way that drifts these numbers, this test
catches it loudly.

Maintenance rule: when a value here drifts AND the drift is intentional
(e.g., new fixture, longer run, etc.), update both the docs/notes and
the assertion in this file. When the drift is unintentional (silent
filter bug, dedup hijack, fixture corruption), fix the cause — don't
relax the test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.loader import load_runs


# Tolerance for "this is the same number" — eval_loss is logged to 4 decimals,
# so anything within 1e-3 is the same trajectory.
TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def ext_runs() -> list:
    """Load every run for the headline optimizers at the canonical 2k horizon,
    using the loader's deny-list dedup default so any non-runtime cfg field
    (max_steps, train_samples, polar_core_remix_alpha, …) automatically acts
    as a dedup axis — no manual axis enumeration to keep in sync.
    """
    logs_root = str(ROOT / "logs")
    if not (ROOT / "logs").exists():
        pytest.skip("no logs/ directory")
    headline_optimizers = sorted({row[0] for row in HEADLINE_NUMBERS})
    return load_runs(
        where={"optimizer": headline_optimizers, "max_steps": 2000},
        cfg_postprocess=lambda c, g: c.update(_lora_r=int(c.get("lora_r", 16))),
        logs_root=logs_root,
    )


def best_at_m1(runs, optimizer: str, lora_r: int) -> float | None:
    """Best final eval loss for `optimizer` at `lora_r` with m=1, seed=0, step=2000.

    Headline numbers in docs/notes/optimizer_synthesis.md are single-seed
    (seed=0). Without the seed filter, multi-seed AdamW data (seeds 1-4 in
    `adamw_multiseed/`) leaks into a "best across seeds" pick that drifts
    away from the documented baseline.
    """
    best = None
    for cfg, evs in runs:
        if cfg["optimizer"] != optimizer:
            continue
        if cfg["_lora_r"] != lora_r:
            continue
        if float(cfg.get("lora_plus_multiplier", 1.0)) != 1.0:
            continue
        if int(cfg.get("seed", 0)) != 0:
            continue
        if evs[-1]["step"] < 2000:
            continue
        fl = evs[-1]["eval_loss"]
        if best is None or fl < best:
            best = fl
    return best


# Headline numbers (m=1, step=2000, seed=0). The r=16 values reflect the
# current loader's view: newest-wins across every live group, including
# diagnostics reruns of the same configs that produced slightly different
# final losses (~0.002 deltas). The pre-loader-migration synthesis cited
# the lr_sweep_2k values for r=16 (0.7579/0.7564/0.7572/0.7546); under the
# new dedup rule the diagnostics groups (h1_pre_probe_2k etc.) win on
# submitted_at. r=64 values are unchanged.
# Format: (optimizer, lora_r, expected_final, source_section)
HEADLINE_NUMBERS = [
    ("adamw",                     16, 0.7601, "AdamW r=16 baseline"),
    ("adamw",                     64, 0.7550, "AdamW r=64 baseline"),
    ("adam-lin-lora",             16, 0.7581, "pre-Adam Sylvester r=16"),
    ("adam-lin-lora",             64, 0.7527, "pre-Adam Sylvester r=64"),
    ("adam-scaled-lora",          16, 0.7590, "pre-Adam Gram r=16"),
    ("adam-scaled-lora",          64, 0.7506, "pre-Adam Gram r=64"),
    ("adam-polar-product-lora",   16, 0.7546, "AdamPolarProduct r=16"),
    ("adam-polar-product-lora",   64, 0.7454, "AdamPolarProduct r=64 (HEADLINE)"),
]


@pytest.mark.parametrize("optimizer,lora_r,expected,source", HEADLINE_NUMBERS)
def test_canary_headline_baseline(ext_runs, optimizer, lora_r, expected, source):
    """Each known-good headline number must match within ±0.001 of the
    value the synthesis cites. Drift indicates fixture corruption, silent
    filter changes, or dedup bugs.
    """
    actual = best_at_m1(ext_runs, optimizer, lora_r)
    if actual is None:
        pytest.skip(f"no completed m=1 r={lora_r} run for {optimizer}")
    delta = abs(actual - expected)
    assert delta < TOLERANCE, (
        f"Canary drift for {optimizer} at r={lora_r} m=1:\n"
        f"  expected {expected:.4f} ({source})\n"
        f"  actual   {actual:.4f}\n"
        f"  delta    {delta:.4f} (tolerance {TOLERANCE})\n"
        f"If the drift is intentional (new fixture / longer run), update both "
        f"docs/notes/optimizer_synthesis.md and tests/test_known_good_baselines.py. "
        f"If unintentional, fix the cause — don't relax the test."
    )
