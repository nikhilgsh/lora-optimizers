"""CPU-only tests for optional manifest annotations and strict audits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.plotting import has_runs


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


def test_manifest_audit_enumerates_every_populated_group(logs_root: Path) -> None:
    from lora_playground.manifest import load_manifests

    groups = _populated_groups(logs_root)
    if not groups:
        pytest.skip("no populated log groups in logs/")
    annotations = load_manifests(str(logs_root), strict=False)
    assert {item["group"] for item in annotations} == set(groups)


def test_optim_colors_are_unique() -> None:
    """OPTIM_COLORS must assign distinct colors to every optimizer.

    Color collisions silently corrupt plots that contain both optimizers
    in the same panel — they appear as a single trace. Catching this in CI
    closes the same drift class as the manifest contract one layer up.
    """
    from collections import Counter
    from lora_playground.plotting import OPTIM_COLORS

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
            + "\nFix: assign distinct hex colors. See lora_playground/plotting/colors.py."
        )
        raise AssertionError(msg)


_TAB10_NS_AXIS = ("#1f77b4", "#ff7f0e", "#2ca02c")  # tab10 blue/orange/green
_TAB10_ORANGE = "#ff7f0e"


def test_overlay_palette_does_not_collide_with_tab10() -> None:
    """`overlay_palette` with default Purples cmap must return colors that are
    visually distinct from the tab10 blue/orange/green axis (the conventional
    reserved set for ablation panels). Historical failure was a `Reds`-cmap
    light end landing near tab orange; switching to Purples + a guard prevents
    a recurrence."""
    from lora_playground.plotting import overlay_palette

    # Spot check across the range of n we'd reasonably ask for (1–8 series).
    for n in range(1, 9):
        palette = overlay_palette(n, reserved=_TAB10_NS_AXIS)
        assert len(palette) == n
        # overlay_palette validates internally; the call would have raised
        # ColorCollisionError otherwise.
    assert overlay_palette(0, reserved=_TAB10_NS_AXIS) == []


def test_assert_palette_distinct_catches_reds_orange_collision() -> None:
    """The guard must reject the original failure mode: a `Reds`-cmap light
    shade next to tab orange. If this regression slips through, future overlay
    palettes can silently fuse with a reserved axis color."""
    import matplotlib.pyplot as plt
    from lora_playground.plotting import (
        ColorCollisionError,
        assert_palette_distinct_from_reserved,
    )
    from lora_playground.plotting.colors import _rgb_to_hex

    reds_light = _rgb_to_hex(plt.get_cmap("Reds")(0.45))
    try:
        assert_palette_distinct_from_reserved(
            [reds_light], reserved=_TAB10_NS_AXIS, name="reds-test",
        )
    except ColorCollisionError:
        return  # expected
    raise AssertionError(
        f"Reds(0.45) = {reds_light} should collide with tab orange "
        f"({_TAB10_ORANGE}) but the guard passed."
    )


def test_standard_sweep_figure_appends_rank() -> None:
    """Library-enforced contract: every standard_sweep_figure auto-appends
    `at r={N}` to the suptitle when runs share a single rank, and raises
    when runs span multiple ranks. Prevents per-cell suptitle drift.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lora_playground.plotting import standard_sweep_figure

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
    from lora_playground.plotting import OPTIM_COLORS

    # SVD oracle modes use training_mode rather than optimizer name and are
    # always 'adamw' — exempt them from the registry-vs-colors check.
    EXEMPT = {"svd-step-adamw", "svd-cumulative-adamw", "sgd", "sgd-m"}
    missing = sorted(o for o in OPTIMIZER_CHOICES if o not in OPTIM_COLORS and o not in EXEMPT)
    assert not missing, (
        f"{len(missing)} optimizer(s) in OPTIMIZER_CHOICES without OPTIM_COLORS entry:\n"
        + "\n".join(f"  {o}" for o in missing)
        + "\nFix: add to OPTIM_COLORS in lora_playground/plotting/colors.py."
    )


def _write_manifest_fixture(logs_root: Path, group: str, scope) -> Path:
    run_info = logs_root / group / "run_info"
    log_dir = run_info / "logs"
    log_dir.mkdir(parents=True)
    (run_info / "meta.json").write_text(json.dumps({
        "group": group,
        "submitted_at": "2026-01-01T00:00:00+00:00",
        "scope": scope,
        "purpose": "manifest regression fixture",
    }))
    return log_dir


def test_manifest_cache_notices_first_log_in_existing_group(tmp_path: Path) -> None:
    """An empty cached run_info/ group becomes visible when its first log appears."""
    from lora_playground.manifest import _LOAD_MANIFESTS_CACHE, load_manifests

    logs_root = tmp_path / "logs"
    log_dir = _write_manifest_fixture(logs_root, "starts_empty", ["pilot"])
    _LOAD_MANIFESTS_CACHE.clear()

    assert load_manifests(str(logs_root), strict=True) == []
    (log_dir / "log_0.out").write_text("{}\n")

    loaded = load_manifests(str(logs_root), strict=True)
    assert [m["group"] for m in loaded] == ["starts_empty"]


@pytest.mark.parametrize("scope", [[], "", "   ", [""], [" ", "\t"]])
def test_strict_load_rejects_empty_or_blank_scope(tmp_path: Path, scope) -> None:
    from lora_playground.manifest import (
        _LOAD_MANIFESTS_CACHE,
        UntaggedSweepError,
        live_manifests_newest_first,
        load_manifests,
        warn_untagged,
    )

    logs_root = tmp_path / "logs"
    log_dir = _write_manifest_fixture(logs_root, "blank_scope", scope)
    (log_dir / "log_0.out").write_text("{}\n")
    _LOAD_MANIFESTS_CACHE.clear()

    with pytest.raises(UntaggedSweepError, match="blank_scope: empty scope"):
        load_manifests(str(logs_root), strict=True)

    non_strict = load_manifests(str(logs_root), strict=False)
    assert non_strict[0]["_empty_scope"] is True
    assert warn_untagged(non_strict) == ["blank_scope"]
    assert live_manifests_newest_first(non_strict) == non_strict


def test_default_discovery_keeps_missing_corrupt_and_empty_scope(tmp_path: Path) -> None:
    from lora_playground.manifest import (
        _LOAD_MANIFESTS_CACHE,
        live_manifests_newest_first,
        load_manifests,
        warn_untagged,
    )

    logs_root = tmp_path / "logs"
    missing_logs = _write_manifest_fixture(logs_root, "missing", ["audit"])
    (logs_root / "missing" / "run_info" / "meta.json").unlink()
    corrupt_logs = _write_manifest_fixture(logs_root, "corrupt", ["audit"])
    (logs_root / "corrupt" / "run_info" / "meta.json").write_text("{bad-json")
    empty_logs = _write_manifest_fixture(logs_root, "empty", [])
    for log_dir in (missing_logs, corrupt_logs, empty_logs):
        (log_dir / "log_0.out").write_text("{}\n")
    _LOAD_MANIFESTS_CACHE.clear()

    # strict=False is deliberately the ordinary default.
    annotations = load_manifests(str(logs_root))
    assert [item["group"] for item in annotations] == [
        "corrupt", "empty", "missing"
    ]
    assert warn_untagged(annotations) == ["corrupt", "empty", "missing"]
    assert {item["group"] for item in live_manifests_newest_first(annotations)} == {
        "corrupt", "empty", "missing"
    }


def test_legacy_load_runs_compatibility_does_not_gate_on_manifest(tmp_path: Path) -> None:
    from lora_playground.loader import load_runs
    from lora_playground.manifest import _LOAD_MANIFESTS_CACHE

    logs_root = tmp_path / "logs"
    log_dir = logs_root / "missing" / "run_info" / "logs"
    log_dir.mkdir(parents=True)
    events = [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3,
         "lora_r": 4, "max_steps": 1, "eval_every": 1,
         "data_pipeline_version": "packed_v1.1",
         "git_dirty": True, "git_diff_sha": "unattested",
         "execution_source_dirty": True,
         "execution_source_sha": "not-present-in-git",
         "execution_source_paths": ["train_lora.py"]},
        {"event": "eval", "step": 1, "eval_loss": 0.8, "lr": 1e-3},
    ]
    (log_dir / "log_0.out").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    _LOAD_MANIFESTS_CACHE.clear()

    runs = load_runs(
        where={"optimizer": "adamw", "lora_r": 4},
        logs_root=str(logs_root),
        warn_cross_commit=False,
        quiet=True,
    )

    assert len(runs) == 1
    assert runs[0][0]["log_group"] == "missing"


def test_newest_sort_handles_nullable_and_malformed_timestamps() -> None:
    from lora_playground.manifest import live_manifests_newest_first

    annotations = [
        {"group": "missing", "submitted_at": None, "_untagged": True},
        {"group": "older", "submitted_at": "2026-01-01T00:00:00+00:00"},
        {"group": "malformed", "submitted_at": {"not": "a string"}},
        {"group": "newer", "submitted_at": "2026-01-02T00:00:00Z"},
        {"group": "bad-text", "submitted_at": "yesterday", "_corrupt": True},
    ]

    ordered = live_manifests_newest_first(annotations)
    assert [item["group"] for item in ordered[:2]] == ["newer", "older"]
    assert {item["group"] for item in ordered[2:]} == {
        "missing", "malformed", "bad-text"
    }


def test_atomic_manifest_write_replaces_complete_json(tmp_path: Path) -> None:
    from lora_playground.manifest import write_manifest_atomic

    path = tmp_path / "group" / "run_info" / "meta.json"
    written = write_manifest_atomic(path, {
        "group": "group", "scope": ["audit"], "submitted_at": None,
    })

    assert written == path
    assert json.loads(path.read_text()) == {
        "group": "group", "scope": ["audit"], "submitted_at": None,
    }
    assert path.read_text().endswith("\n")
    assert not list(path.parent.glob(".meta.json.*.tmp"))


def test_submission_writer_owns_the_complete_schema(tmp_path: Path) -> None:
    from lora_playground.manifest import (
        MANIFEST_FIELDS,
        write_submission_manifest,
    )

    path = tmp_path / "meta.json"
    write_submission_manifest(
        path,
        group="group",
        submitted_at="2026-08-27T12:00:00-04:00",
        slurm_job_id="pending",
        n_gpus=4,
        params_file="params.json",
        sweep_script="scripts/sweep/sweep.sh",
        sbatch_script="slurm_scripts/sbatch.sh",
        git_commit="abc",
        git_dirty=False,
        scope="comparison, audit",
        purpose="test",
        data_pipeline_version="packed_v1.1",
    )

    payload = json.loads(path.read_text())
    assert set(payload) == set(MANIFEST_FIELDS)
    assert payload["scope"] == ["comparison", "audit"]
    assert payload["n_gpus"] == 4


def test_atomic_manifest_write_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch
) -> None:
    import lora_playground.manifest as manifest_module

    path = tmp_path / "meta.json"
    path.write_text('{"group": "old"}\n')
    old = path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        manifest_module.write_manifest_atomic(path, {"group": "new"})

    assert path.read_bytes() == old
    assert not list(tmp_path.glob(".meta.json.*.tmp"))


def test_manifest_series_axes_share_neutral_runtime_fields() -> None:
    from lora_playground.manifest import SERIES_AXIS_FIELDS
    from lora_playground.plotting import RUNTIME_FIELDS
    from lora_playground.run_records import RUNTIME_FIELDS as NEUTRAL_RUNTIME_FIELDS

    assert RUNTIME_FIELDS is NEUTRAL_RUNTIME_FIELDS
    assert SERIES_AXIS_FIELDS == RUNTIME_FIELDS | {
        "seed", "lr", "lora_r", "lora_alpha", "max_steps", "eval_every",
    }
