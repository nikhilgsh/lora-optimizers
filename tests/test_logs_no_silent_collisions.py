"""Regression test: walk the project's logs/ tree and assert no two runs
share a deny-list dedup key while differing on any non-runtime cfg field.

This is fix #4 from the 2026-05 reproducibility-infra audit. It closes the
loop on:

  1. ``optimizer_config_dict`` in the config event (fix #1) — the cfg now
     self-describes the running algorithm.
  2. Deny-list dedup in the loader (fix #2) — any cfg field difference
     means "different run".
  3. Symmetric hidden-axis check in merge_runs (fix #3) — missing-vs-present
     counts as a difference.

If a future PR introduces an optimizer hyperparameter that ``config_dict``
doesn't capture (e.g. via the param_groups fallback short-circuiting), or if
two sweeps ship with semantically-different configs that hash equal, this
test fails before merge — instead of silently dropping data downstream.

Note on legacy data: pre-fix-#1 runs lack the resolved ``optimizer_config``
field. If two such runs differ only in algorithm internals not surfaced in
their CLI-derived config event (e.g. ``picard_iters`` default flipping
between commits), they're indistinguishable from the JSONL alone — this
test cannot detect that. New runs after fix #1 are protected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.loader import RUNTIME_FIELDS as LOADER_RUNTIME_FIELDS, _denylist_key
from lora_playground.plotting import RUNTIME_FIELDS as PLOTTING_RUNTIME_FIELDS
from lora_playground.plotting.merge import _hidden_axis_diffs
from lora_playground.run_catalog import RunCatalog
from lora_playground.run_records import run_view


_LOGS_ROOT = ROOT / "logs"


def test_runtime_field_lists_match_across_modules():
    """``loader.RUNTIME_FIELDS`` and ``plotting.RUNTIME_FIELDS`` must match.

    They're duplicated to avoid an import cycle (loader imports from plotting).
    The duplication is fine; silent drift between the two is not — divergence
    would mean the dedup key and the collision-check disagree about what
    counts as runtime metadata.
    """
    assert LOADER_RUNTIME_FIELDS == PLOTTING_RUNTIME_FIELDS, (
        "loader.RUNTIME_FIELDS and plotting.RUNTIME_FIELDS have drifted: "
        f"only-in-loader={LOADER_RUNTIME_FIELDS - PLOTTING_RUNTIME_FIELDS}, "
        f"only-in-plotting={PLOTTING_RUNTIME_FIELDS - LOADER_RUNTIME_FIELDS}. "
        "Update both lists together."
    )


@pytest.mark.skipif(
    not _LOGS_ROOT.exists(),
    reason="logs/ tree not present — test is meaningful only inside the project",
)
def test_logs_have_no_silent_collisions():
    """No two runs in logs/ share a deny-list key while differing on cfg."""
    catalog = RunCatalog.discover(_LOGS_ROOT)

    # key → (cfg, group) of the first run we saw with this key
    seen: dict = {}
    collisions: list[tuple] = []

    for index, record in enumerate(catalog.records):
        view = run_view(record, index)
        cfg = dict(view.semantic_config)
        cfg["log_group"] = view.group
        cfg["_log_filename"] = view.log_filename
        k = _denylist_key(cfg, LOADER_RUNTIME_FIELDS)
        existing = seen.get(k)
        if existing is None:
            seen[k] = (cfg, view.group)
            continue
        ex_cfg, ex_group = existing
        diffs = _hidden_axis_diffs(ex_cfg, cfg)
        if diffs:
            collisions.append((ex_group, view.group, diffs[:5]))

    if collisions:
        msgs = []
        for ex_group, new_group, diffs in collisions[:10]:
            field_summary = ", ".join(
                f"{f}={va!r}↔{vb!r}" for f, va, vb in diffs
            )
            msgs.append(f"  {ex_group!r} vs {new_group!r}: {field_summary}")
        suffix = ""
        if len(collisions) > 10:
            suffix = f"\n  … (+{len(collisions)-10} more)"
        pytest.fail(
            f"Found {len(collisions)} dedup-key collision(s) in logs/ where "
            f"two runs differ on non-runtime cfg field(s) but would be "
            f"silently merged. Either add the differing field(s) to "
            f"RUNTIME_FIELDS (if they're runtime metadata), or delete the "
            f"older log dir if the run is superseded:\n" + "\n".join(msgs) + suffix
        )
