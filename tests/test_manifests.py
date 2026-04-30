"""CPU-only enforcement: every populated log group must have a meta.json
with at least one scope tag (or be explicitly tagged as a pilot).

This test catches drift between submissions — e.g., a sweep submitted
without `SWEEP_SCOPE`, an old log dir restored without metadata, or a
hand-edited `meta.json` with empty scope. Fast (~1 s), no GPU, no
model loads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.plot_utils import has_runs


PILOT_SUFFIXES = ("_500",)


def _is_pilot(group: str) -> bool:
    return any(s in group for s in PILOT_SUFFIXES)


def _populated_groups(logs_root: Path) -> list[str]:
    if not logs_root.exists():
        return []
    return sorted(
        p.parent.name for p in logs_root.glob("*/run_info")
        if has_runs(p.parent.name, str(logs_root))
    )


@pytest.fixture
def logs_root() -> Path:
    return ROOT / "logs"


def test_every_populated_group_has_manifest(logs_root: Path) -> None:
    groups = _populated_groups(logs_root)
    if not groups:
        pytest.skip("no populated log groups in logs/")
    missing = [g for g in groups if not (logs_root / g / "run_info" / "meta.json").exists()]
    assert not missing, (
        f"{len(missing)} populated log group(s) without meta.json:\n"
        + "\n".join(f"  {g}" for g in missing)
        + "\nFix: re-submit via slurm_scripts/submit.sh with SWEEP_SCOPE set, "
        "or write meta.json by hand (schema in lora_playground/manifest.py)."
    )


def test_every_manifest_is_valid_json(logs_root: Path) -> None:
    groups = _populated_groups(logs_root)
    if not groups:
        pytest.skip("no populated log groups in logs/")
    corrupt: list[tuple[str, str]] = []
    for g in groups:
        meta = logs_root / g / "run_info" / "meta.json"
        if not meta.exists():
            continue  # covered by the missing-manifest test
        try:
            json.loads(meta.read_text())
        except json.JSONDecodeError as e:
            corrupt.append((g, str(e)))
    assert not corrupt, "corrupt meta.json:\n" + "\n".join(f"  {g}: {e}" for g, e in corrupt)


def test_optim_colors_are_unique() -> None:
    """OPTIM_COLORS must assign distinct colors to every optimizer.

    Color collisions silently corrupt plots that contain both optimizers
    in the same panel — they appear as a single trace. Catching this in CI
    closes the same drift class as the manifest contract one layer up.
    """
    from collections import Counter
    from lora_playground.plot_utils import OPTIM_COLORS

    counts = Counter(OPTIM_COLORS.values())
    dups = {color: cnt for color, cnt in counts.items() if cnt > 1}
    if dups:
        details = []
        for color, cnt in dups.items():
            opts = [k for k, v in OPTIM_COLORS.items() if v == color]
            details.append(f"  {color} ({cnt}× collision): {opts}")
        msg = (
            f"{len(dups)} color collision(s) in OPTIM_COLORS:\n"
            + "\n".join(details)
            + "\nFix: assign distinct hex colors. See lora_playground/plot_utils.py."
        )
        raise AssertionError(msg)


def test_standard_sweep_figure_appends_rank() -> None:
    """Library-enforced contract: every standard_sweep_figure auto-appends
    `at r={N}` to the suptitle when runs share a single rank, and raises
    when runs span multiple ranks. Prevents per-cell suptitle drift.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lora_playground.plot_utils import standard_sweep_figure

    # Synthetic minimal runs: one adamw + one candidate, all at lora_r=16.
    fake_evs = [{"step": 200 * (i + 1), "eval_loss": 1.0 - 0.05 * i, "lr": 3e-4}
                for i in range(10)]
    runs_r16 = [
        ({"optimizer": "adamw", "lr": 3e-4, "lora_r": 16, "command": ""}, fake_evs),
        ({"optimizer": "adam-lin-lora", "lr": 3e-4, "lora_r": 16, "command": ""}, fake_evs),
    ]

    # Single-rank: rank suffix appended.
    fig, *_ = standard_sweep_figure(
        runs_r16, group_key_fn=lambda c: c["optimizer"],
        color_map={"adamw": "#000", "adam-lin-lora": "#888"},
        reference_runs=runs_r16, suptitle="My panel",
    )
    sup = fig._suptitle.get_text()
    plt.close(fig)
    assert sup == "My panel at r=16", f"expected 'My panel at r=16', got {sup!r}"

    # Title with explicit r= should NOT be doubled.
    fig, *_ = standard_sweep_figure(
        runs_r16, group_key_fn=lambda c: c["optimizer"],
        color_map={"adamw": "#000", "adam-lin-lora": "#888"},
        reference_runs=runs_r16, suptitle="Already r=16 in title",
    )
    sup = fig._suptitle.get_text()
    plt.close(fig)
    assert sup == "Already r=16 in title", f"expected idempotent, got {sup!r}"

    # Multi-rank input: auto-splits, returns a list of (fig, axes, kept, dropped)
    # tuples — one per rank, sorted ascending. Each figure has its own rank
    # in the suptitle.
    runs_mixed = runs_r16 + [
        ({"optimizer": "adamw", "lr": 3e-4, "lora_r": 64, "command": ""}, fake_evs),
        ({"optimizer": "adam-lin-lora", "lr": 3e-4, "lora_r": 64, "command": ""}, fake_evs),
    ]
    results = standard_sweep_figure(
        runs_mixed, group_key_fn=lambda c: c["optimizer"],
        color_map={"adamw": "#000", "adam-lin-lora": "#888"},
        reference_runs=runs_mixed, suptitle="Mixed",
    )
    assert isinstance(results, list) and len(results) == 2, \
        f"expected 2 figures (r=16 and r=64), got {len(results) if isinstance(results, list) else type(results)}"
    titles = sorted(r[0]._suptitle.get_text() for r in results)
    assert titles == ["Mixed at r=16", "Mixed at r=64"], titles
    for fig, *_ in results:
        plt.close(fig)


def test_optim_choices_have_color_entries() -> None:
    """Every optimizer in the registry needs an OPTIM_COLORS entry — otherwise
    the notebook's filter (``c["optimizer"] in OPTIM_COLORS``) silently drops it.
    """
    from lora_playground.optim import OPTIMIZER_CHOICES
    from lora_playground.plot_utils import OPTIM_COLORS

    # SVD oracle modes use training_mode rather than optimizer name and are
    # always 'adamw' — exempt them from the registry-vs-colors check.
    EXEMPT = {"svd-step-adamw", "svd-cumulative-adamw", "sgd", "sgd-m"}
    missing = sorted(o for o in OPTIMIZER_CHOICES if o not in OPTIM_COLORS and o not in EXEMPT)
    assert not missing, (
        f"{len(missing)} optimizer(s) in OPTIMIZER_CHOICES without OPTIM_COLORS entry:\n"
        + "\n".join(f"  {o}" for o in missing)
        + "\nFix: add to OPTIM_COLORS in lora_playground/plot_utils.py."
    )


def test_non_pilot_groups_have_non_empty_scope(logs_root: Path) -> None:
    """Pilots (group name contains _500) may have scope=[] by design.
    Every other populated group must have at least one scope tag.
    """
    groups = _populated_groups(logs_root)
    if not groups:
        pytest.skip("no populated log groups in logs/")
    bad: list[str] = []
    for g in groups:
        if _is_pilot(g):
            continue
        meta = logs_root / g / "run_info" / "meta.json"
        if not meta.exists():
            continue  # covered by missing-manifest test
        try:
            m = json.loads(meta.read_text())
        except json.JSONDecodeError:
            continue  # covered by corrupt-json test
        if not m.get("scope"):
            bad.append(g)
    assert not bad, (
        f"{len(bad)} non-pilot group(s) with empty scope:\n"
        + "\n".join(f"  {g}" for g in bad)
        + "\nFix: edit meta.json scope to a non-empty list of tags. "
        "See lora_playground/manifest.py for known scopes."
    )
