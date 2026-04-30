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

from lora_playground.manifest import groups_by_scope, load_manifests
from lora_playground.plot_utils import OPTIM_COLORS, merge_runs


# Tolerance for "this is the same number" — eval_loss is logged to 4 decimals,
# so anything within 1e-3 is the same trajectory.
TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def ext_runs() -> list:
    """Reproduce the leaderboard cell's load: union of all relevant scopes,
    dedup including lora_plus_multiplier so m=1 vs m=4 are kept distinct.
    """
    logs_root = str(ROOT / "logs")
    if not (ROOT / "logs").exists():
        pytest.skip("no logs/ directory")
    mans = load_manifests(logs_root, strict=False)
    groups = []
    for scope in ("ext_compare", "muon_family", "polar_family",
                  "all_optimizers", "loraplus_family"):
        for g in groups_by_scope(scope, manifests=mans, logs_root=logs_root):
            if g not in groups:
                groups.append(g)
    return merge_runs(
        groups,
        key_fn=lambda c: (c["optimizer"], float(c["lr"]),
                          int(c.get("lora_r", 16)),
                          float(c.get("lora_plus_multiplier", 1.0))),
        filter_fn=lambda c: c["optimizer"] in OPTIM_COLORS,
        cfg_postprocess=lambda c, g: c.update(_lora_r=int(c.get("lora_r", 16))),
        logs_root=logs_root,
    )


def best_at_m1(runs, optimizer: str, lora_r: int) -> float | None:
    """Best final eval loss for `optimizer` at `lora_r` with m=1, step=2000."""
    best = None
    for cfg, evs in runs:
        if cfg["optimizer"] != optimizer:
            continue
        if cfg["_lora_r"] != lora_r:
            continue
        if float(cfg.get("lora_plus_multiplier", 1.0)) != 1.0:
            continue
        if evs[-1]["step"] < 2000:
            continue
        fl = evs[-1]["eval_loss"]
        if best is None or fl < best:
            best = fl
    return best


# Headline numbers from synthesis (m=1, step=2000, seed=0).
# Format: (optimizer, lora_r, expected_final, source_section)
HEADLINE_NUMBERS = [
    ("adamw",                     16, 0.7579, "synthesis: AdamW r=16 baseline"),
    ("adamw",                     64, 0.7550, "synthesis: AdamW r=64 baseline"),
    ("adam-lin-lora",             16, 0.7564, "synthesis: pre-Adam Sylvester r=16"),
    ("adam-lin-lora",             64, 0.7527, "synthesis: pre-Adam Sylvester r=64"),
    ("adam-scaled-lora",          16, 0.7572, "synthesis: pre-Adam Gram r=16"),
    ("adam-scaled-lora",          64, 0.7506, "synthesis: pre-Adam Gram r=64"),
    ("adam-polar-product-lora",   16, 0.7546, "synthesis: AdamPolarProduct r=16"),
    ("adam-polar-product-lora",   64, 0.7453, "synthesis: AdamPolarProduct r=64 (HEADLINE)"),
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
