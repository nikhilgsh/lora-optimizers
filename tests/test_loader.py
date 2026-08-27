"""Tests for the predicate-based loader and inventory.

Synthetic-fixture tests — build a temp logs/ tree per test so behavior is
fully isolated from the real project state. Validates:

  - Predicate matching: literal, list, callable.
  - load_runs honors `where` and keeps physical reruns visible.
  - inventory_runs detects orphans, unknown optimizers, and pinning at
    the lr-range boundary.
  - render_inventory smoke (returns non-empty plain-text report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.loader import (
    PINNING_ALL_DIVERGED,
    PINNING_HIGH,
    PINNING_INTERIOR,
    PINNING_LOW,
    PINNING_SINGLE,
    UncontrolledAxisError,
    _matches,
    aggregate_by,
    inventory_runs,
    load_runs,
    render_inventory,
)
from lora_playground import loader as loader_mod


# ─── synthetic-fixture helpers ────────────────────────────────────────────────

def _write_run(log_dir: Path, idx: int, cfg: dict, evals: list[dict]) -> None:
    """Drop one task .out file in log_dir/log_<idx>.out matching the format
    that load_run() in plot_utils.py expects (one JSON per line; first
    'config' event then 'eval' events)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"log_{idx}.out"
    cfg = {"event": "config", **cfg}
    cfg.setdefault("command", f"python train_lora.py --lr {cfg.get('lr','3e-4')}")
    lines = [json.dumps(cfg)]
    for e in evals:
        lines.append(json.dumps({"event": "eval", **e}))
    out.write_text("\n".join(lines) + "\n")


def _write_group(logs_root: Path, group: str, manifest: dict | None,
                 runs: list[tuple[dict, list[dict]]]) -> None:
    """Create logs/<group>/run_info/{meta.json,logs/log_*.out}. If manifest is
    None, no meta.json is written (orphaned-group fixture)."""
    run_info = logs_root / group / "run_info"
    run_info.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        manifest = {"group": group, **manifest}
        manifest.setdefault("submitted_at", "2026-04-30T00:00:00-04:00")
        (run_info / "meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_dir = run_info / "logs"
    for i, (cfg, evals) in enumerate(runs):
        _write_run(log_dir, i, cfg, evals)


def _evs(*pairs: tuple[int, float]) -> list[dict]:
    """Compact eval list builder: ((step, loss), ...)."""
    return [{"step": s, "eval_loss": l, "lr": 3e-4} for s, l in pairs]


def _cfg(optimizer: str, lr: float, lora_r: int = 64,
         lora_plus_multiplier: float = 1.0, seed: int = 0) -> dict:
    """Compact run-config builder — defaults match the canonical sweep."""
    return {
        "optimizer": optimizer,
        "lr": lr,
        "lora_r": lora_r,
        "lora_plus_multiplier": lora_plus_multiplier,
        "seed": seed,
    }


# ─── _matches ─────────────────────────────────────────────────────────────────

def test_matches_literal_equality():
    assert _matches(64, 64)
    assert not _matches(64, 16)
    assert _matches("adamw", "adamw")


def test_matches_collection_membership():
    assert _matches(["adamw", "muon-lora"], "adamw")
    assert not _matches(["adamw", "muon-lora"], "scaled-lora")
    assert _matches({16, 64, 128}, 16)
    assert _matches((1.0, 4.0), 4.0)


def test_matches_callable_predicate():
    assert _matches(lambda x: x > 32, 64)
    assert not _matches(lambda x: x > 32, 8)
    assert _matches(lambda s: s.startswith("adam"), "adamw")


# ─── load_runs ────────────────────────────────────────────────────────────────

def test_load_runs_filters_by_predicate(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g_polar", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 3e-4), _evs((200, 0.8), (2000, 0.74))),
        (_cfg("adam-polar-product-lora", 1e-3), _evs((200, 0.8), (2000, 0.78))),
    ])
    _write_group(logs, "g_adamw", {"scope": ["all_optimizers"]}, [
        (_cfg("adamw", 3e-4), _evs((200, 0.85), (2000, 0.76))),
    ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"}, logs_root=str(logs))
    assert len(runs) == 2
    assert all(c["optimizer"] == "adam-polar-product-lora" for c, _ in runs)

    runs = load_runs(where={"lr": 3e-4}, logs_root=str(logs))
    assert len(runs) == 2
    assert all(float(c["lr"]) == 3e-4 for c, _ in runs)

    runs = load_runs(where={"optimizer": ["adamw", "adam-polar-product-lora"]},
                     logs_root=str(logs))
    assert len(runs) == 3


def test_load_runs_callable_predicate(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["all_optimizers"]}, [
        (_cfg("adamw", 1e-4, lora_r=16),  _evs((2000, 0.78))),
        (_cfg("adamw", 1e-4, lora_r=128), _evs((2000, 0.74))),
        (_cfg("adamw", 1e-4, lora_r=256), _evs((2000, 0.73))),
    ])
    runs = load_runs(where={"lora_r": lambda r: r >= 128}, logs_root=str(logs))
    assert len(runs) == 2
    assert {int(c["lora_r"]) for c, _ in runs} == {128, 256}


def test_load_runs_keeps_colliding_physical_runs_in_filesystem_order(tmp_path: Path):
    """Manifest timestamps neither order nor deduplicate physical records."""
    logs = tmp_path / "logs"
    cfg = _cfg("adam-polar-product-lora", 3e-4)
    _write_group(logs, "a_old",
                 {"scope": ["polar_family"], "submitted_at": "2026-04-01T00:00:00-04:00"},
                 [(cfg, _evs((2000, 0.80)))])
    _write_group(logs, "z_new",
                 {"scope": ["polar_family"], "submitted_at": "2026-04-30T00:00:00-04:00"},
                 [(cfg, _evs((2000, 0.74)))])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"}, logs_root=str(logs))
    assert [(cfg["log_group"], evs[-1]["eval_loss"]) for cfg, evs in runs] == [
        ("a_old", 0.80),
        ("z_new", 0.74),
    ]


def test_load_runs_unique_on_raises_on_uncontrolled_axis(tmp_path: Path):
    """A bucket spanning runs with different htmuon_p values must raise."""
    logs = tmp_path / "logs"
    c1 = _cfg("adam-polar-product-lora", 3e-2)
    c1["htmuon_p"] = None
    c2 = _cfg("adam-polar-product-lora", 3e-2)
    c2["htmuon_p"] = 0.0625
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (c1, _evs((2000, 0.50))),
        (c2, _evs((2000, 0.51))),
    ])
    # Without unique_on: silently returns both.
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"},
                     logs_root=str(logs))
    assert len(runs) == 2

    # With unique_on=(optimizer, lr): bucket has two runs differing on htmuon_p.
    with pytest.raises(UncontrolledAxisError, match="htmuon_p"):
        load_runs(where={"optimizer": "adam-polar-product-lora"},
                  unique_on=("optimizer", "lr"), logs_root=str(logs))

    # allow_axes opt-out: htmuon_p variation acknowledged.
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"},
                     unique_on=("optimizer", "lr"),
                     allow_axes=("htmuon_p",), logs_root=str(logs))
    assert len(runs) == 2

    # Constraining via `where` also clears the violation.
    runs = load_runs(where={"optimizer": "adam-polar-product-lora", "htmuon_p": None},
                     unique_on=("optimizer", "lr"), logs_root=str(logs))
    assert len(runs) == 1


def test_load_runs_unique_on_clean_bucket_passes(tmp_path: Path):
    """Same bucket, runs identical on all cfg axes → no error."""
    logs = tmp_path / "logs"
    cfg = _cfg("adam-polar-product-lora", 3e-2)
    cfg["htmuon_p"] = None
    # Two distinct buckets (different lr) — each bucket has one run.
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (cfg, _evs((2000, 0.50))),
        ({**cfg, "lr": 1e-2}, _evs((2000, 0.51))),
    ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"},
                     unique_on=("optimizer", "lr"), logs_root=str(logs))
    assert len(runs) == 2


def test_aggregate_by_raises_on_hidden_axis(tmp_path: Path):
    """aggregate_by(key=("lr",)) must raise when two runs at the same lr
    differ on an unaccounted-for cfg axis (precond_delta_relative)."""
    logs = tmp_path / "logs"
    c1 = _cfg("adam-polar-product-lora", 3e-2)
    c1["precond_delta_relative"] = False
    c2 = _cfg("adam-polar-product-lora", 3e-2)
    c2["precond_delta_relative"] = True
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (c1, _evs((2000, 0.50))),
        (c2, _evs((2000, 0.51))),
    ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"},
                     logs_root=str(logs))
    assert len(runs) == 2
    with pytest.raises(UncontrolledAxisError, match="precond_delta_relative"):
        aggregate_by(runs, key=("lr",))


def test_aggregate_by_clean_bucket_reduces(tmp_path: Path):
    """No hidden axis variation → reduce runs per bucket."""
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("adamw", 3e-4), _evs((2000, 0.50))),
        (_cfg("adamw", 1e-3), _evs((2000, 0.52))),
    ])
    runs = load_runs(where={"optimizer": "adamw"}, logs_root=str(logs))
    best = aggregate_by(
        runs, key=("lr",),
        reduce=lambda bucket: min(h[-1]["eval_loss"] for _, h in bucket),
    )
    assert best == {(3e-4,): 0.50, (1e-3,): 0.52}


def test_aggregate_by_allow_axes_opt_out(tmp_path: Path):
    """`allow_axes=(...)` lets the caller acknowledge hidden variation."""
    logs = tmp_path / "logs"
    c1 = _cfg("adam-polar-product-lora", 3e-2)
    c1["precond_delta_relative"] = False
    c2 = _cfg("adam-polar-product-lora", 3e-2)
    c2["precond_delta_relative"] = True
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (c1, _evs((2000, 0.50))),
        (c2, _evs((2000, 0.51))),
    ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"},
                     logs_root=str(logs))
    out = aggregate_by(runs, key=("lr",), allow_axes=("precond_delta_relative",))
    assert set(out.keys()) == {(3e-2,)}
    assert len(out[(3e-2,)]) == 2


def test_aggregate_by_seed_is_series_axis(tmp_path: Path):
    """`seed` is in SERIES_AXIS_FIELDS — bucket spanning seeds doesn't raise."""
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("adamw", 3e-4, seed=0), _evs((2000, 0.50))),
        (_cfg("adamw", 3e-4, seed=1), _evs((2000, 0.52))),
    ])
    runs = load_runs(where={"optimizer": "adamw"}, logs_root=str(logs))
    mean_by_lr = aggregate_by(
        runs, key=("lr",),
        reduce=lambda bucket: sum(h[-1]["eval_loss"] for _, h in bucket) / len(bucket),
    )
    assert mean_by_lr == {(3e-4,): 0.51}


def test_load_runs_unique_on_seed_in_series_axis(tmp_path: Path):
    """`seed` is in SERIES_AXIS_FIELDS, so a bucket spanning seeds is fine."""
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("adamw", 3e-4, seed=0), _evs((2000, 0.50))),
        (_cfg("adamw", 3e-4, seed=1), _evs((2000, 0.51))),
    ])
    # seed varies within the (optimizer, lr) bucket but it's a series axis.
    runs = load_runs(where={"optimizer": "adamw"},
                     unique_on=("optimizer", "lr"), logs_root=str(logs))
    assert len(runs) == 2


def test_load_runs_extension_does_not_drop_old_data(tmp_path: Path):
    """The OLD sweep stays loaded; the new extension sweep adds new lr cells
    without touching old ones."""
    logs = tmp_path / "logs"
    _write_group(logs, "g_lowlr",
                 {"scope": ["r_extension"], "submitted_at": "2026-04-15T00:00:00-04:00"},
                 [
                     (_cfg("adam-polar-product-lora", 1e-4, lora_r=256), _evs((2000, 0.741))),
                     (_cfg("adam-polar-product-lora", 3e-4, lora_r=256), _evs((2000, 0.747))),
                     (_cfg("adam-polar-product-lora", 1e-3, lora_r=256), _evs((2000, 0.821))),
                 ])
    _write_group(logs, "g_lowestlr",
                 {"scope": ["r_extension"], "submitted_at": "2026-04-30T00:00:00-04:00"},
                 [
                     (_cfg("adam-polar-product-lora", 3e-5, lora_r=256), _evs((2000, 0.753))),
                 ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"}, logs_root=str(logs))
    lrs = sorted(float(c["lr"]) for c, _ in runs)
    assert lrs == [3e-5, 1e-4, 3e-4, 1e-3], (
        f"extension sweep dropped old data — got lrs {lrs}, expected all four"
    )


# ─── inventory_runs ──────────────────────────────────────────────────────────

def test_inventory_detects_orphaned_group(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g_tagged", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 3e-4), _evs((2000, 0.74))),
    ])
    _write_group(logs, "g_orphan", None, [
        (_cfg("adamw", 3e-4, lora_r=16), _evs((2000, 0.78))),
    ])
    inv = inventory_runs(str(logs))
    assert "g_orphan" in inv.groups_orphaned
    assert "g_tagged" not in inv.groups_orphaned
    assert "g_orphan" in inv.groups_loaded


def test_inventory_detects_unknown_optimizer(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("made-up-optimizer", 3e-4), _evs((2000, 0.74))),
    ])
    inv = inventory_runs(str(logs))
    assert "made-up-optimizer" in inv.optimizers_unknown


def test_inventory_pinning_classification(tmp_path: Path):
    logs = tmp_path / "logs"
    # interior: best lr is in the middle of the swept range
    _write_group(logs, "g_interior", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 1e-4), _evs((2000, 0.78))),
        (_cfg("adam-polar-product-lora", 3e-4), _evs((2000, 0.74))),
        (_cfg("adam-polar-product-lora", 1e-3), _evs((2000, 0.80))),
    ])
    # pinned low: best lr equals lr_min
    _write_group(logs, "g_pinned_low", {"scope": ["r_extension"]}, [
        (_cfg("adam-scaled-lora", 1e-4, lora_r=256), _evs((2000, 0.752))),
        (_cfg("adam-scaled-lora", 3e-4, lora_r=256), _evs((2000, 0.758))),
        (_cfg("adam-scaled-lora", 1e-3, lora_r=256), _evs((2000, 0.872))),
    ])
    # pinned high: best lr equals lr_max
    _write_group(logs, "g_pinned_high", {"scope": ["muon_family"]}, [
        (_cfg("adam-muon-lora", 1e-3, lora_r=16), _evs((2000, 0.79))),
        (_cfg("adam-muon-lora", 3e-3, lora_r=16), _evs((2000, 0.74))),
    ])
    # single: only one lr tried — uninformative
    _write_group(logs, "g_single", {"scope": ["pilot"]}, [
        (_cfg("muon-lora", 3e-4, lora_r=16), _evs((2000, 0.83))),
    ])
    inv = inventory_runs(str(logs))
    by_opt = {(r.optimizer, r.lora_r): r for r in inv.coverage}
    assert by_opt[("adam-polar-product-lora", 64)].pinning == PINNING_INTERIOR
    assert by_opt[("adam-scaled-lora", 256)].pinning == PINNING_LOW
    assert by_opt[("adam-muon-lora", 16)].pinning == PINNING_HIGH
    assert by_opt[("muon-lora", 16)].pinning == PINNING_SINGLE
    pinned_keys = {(r.optimizer, r.lora_r) for r in inv.pinned}
    assert pinned_keys == {("adam-scaled-lora", 256), ("adam-muon-lora", 16)}


def test_inventory_all_diverged(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 1e-2), _evs((200, 5.0), (2000, 8.0))),
    ])
    inv = inventory_runs(str(logs))
    assert len(inv.coverage) == 1
    assert inv.coverage[0].pinning == PINNING_ALL_DIVERGED
    assert inv.coverage[0].best_lr is None


def test_render_inventory_smoke(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 1e-4), _evs((2000, 0.741))),
        (_cfg("adam-polar-product-lora", 3e-4), _evs((2000, 0.745))),
    ])
    inv = inventory_runs(str(logs))
    text = render_inventory(inv)
    assert "Cataloged" in text
    assert "Coverage" in text


def test_inventory_uses_logged_effective_config_without_defaults(tmp_path: Path):
    logs = tmp_path / "logs"
    cfg = {
        "command": "python train_lora.py --optimizer ignored-command-value",
        "_cli_args": {
            "optimizer": "adamw",
            "lr": 3e-4,
            "lora_r": 128,
            "lora_plus_multiplier": 2.0,
        },
    }
    _write_group(logs, "g", {"scope": ["all_optimizers"]}, [
        (cfg, _evs((2000, 0.76))),
    ])

    inv = inventory_runs(str(logs))

    assert len(inv.coverage) == 1
    row = inv.coverage[0]
    assert (row.optimizer, row.lora_r, row.lora_plus_multiplier) == (
        "adamw", 128, 2.0,
    )
    assert not inv.records_incomplete


def test_inventory_keeps_missing_logged_axes_missing(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["all_optimizers"]}, [
        ({"optimizer": "adamw", "lr": 3e-4}, _evs((2000, 0.76))),
    ])

    inv = inventory_runs(str(logs))

    assert len(inv.coverage) == 1
    assert inv.coverage[0].lora_r is None
    assert inv.coverage[0].lora_plus_multiplier is None
    assert inv.records_incomplete == ((
        "g/log_0.out", ("lora_r", "lora_plus_multiplier"),
    ),)


def test_inventory_surfaces_physical_group_without_usable_record(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g_empty", {"scope": ["all_optimizers"]}, [
        (_cfg("adamw", 3e-4), []),
    ])

    inv = inventory_runs(str(logs))

    assert inv.groups_on_disk == ("g_empty",)
    assert inv.groups_loaded == ()
    assert inv.groups_without_records == ("g_empty",)


# ─── cross-commit warning ─────────────────────────────────────────────────────

def test_load_runs_warns_when_runs_span_multiple_commits(tmp_path: Path):
    logs = tmp_path / "logs"
    cfg_a = _cfg("adam-polar-product-lora", 3e-4, lora_r=16)
    cfg_a["git_commit"] = "aaaaaaaa"
    cfg_b = _cfg("adam-polar-product-lora", 3e-4, lora_r=64)
    cfg_b["git_commit"] = "bbbbbbbb"
    _write_group(logs, "g_a", {"scope": ["polar_family"]},
                 [(cfg_a, _evs((2000, 0.74)))])
    _write_group(logs, "g_b", {"scope": ["polar_family"]},
                 [(cfg_b, _evs((2000, 0.74)))])
    with pytest.warns(UserWarning, match=r"runs from 2 recorded commits"):
        load_runs(where={"optimizer": "adam-polar-product-lora"},
                  logs_root=str(logs))


def test_load_runs_no_warning_when_single_commit(tmp_path: Path, recwarn):
    logs = tmp_path / "logs"
    cfg_a = _cfg("adam-polar-product-lora", 3e-4, lora_r=16)
    cfg_a["git_commit"] = "samecommit"
    cfg_b = _cfg("adam-polar-product-lora", 3e-4, lora_r=64)
    cfg_b["git_commit"] = "samecommit"
    _write_group(logs, "g_a", {"scope": ["polar_family"]},
                 [(cfg_a, _evs((2000, 0.74)))])
    _write_group(logs, "g_b", {"scope": ["polar_family"]},
                 [(cfg_b, _evs((2000, 0.74)))])
    load_runs(where={"optimizer": "adam-polar-product-lora"},
              logs_root=str(logs))
    cross_commit_warnings = [
        w for w in recwarn.list
        if issubclass(w.category, UserWarning) and "commits" in str(w.message)
    ]
    assert not cross_commit_warnings


def test_load_runs_warn_cross_commit_can_be_silenced(tmp_path: Path, recwarn):
    logs = tmp_path / "logs"
    cfg_a = _cfg("adam-polar-product-lora", 3e-4, lora_r=16)
    cfg_a["git_commit"] = "aaaaaaaa"
    cfg_b = _cfg("adam-polar-product-lora", 3e-4, lora_r=64)
    cfg_b["git_commit"] = "bbbbbbbb"
    _write_group(logs, "g_a", {"scope": ["polar_family"]},
                 [(cfg_a, _evs((2000, 0.74)))])
    _write_group(logs, "g_b", {"scope": ["polar_family"]},
                 [(cfg_b, _evs((2000, 0.74)))])
    load_runs(where={"optimizer": "adam-polar-product-lora"},
              logs_root=str(logs), warn_cross_commit=False)
    cross_commit_warnings = [
        w for w in recwarn.list
        if issubclass(w.category, UserWarning) and "commits" in str(w.message)
    ]
    assert not cross_commit_warnings


# ─── load_runs end-to-end: logged values only ─────────────────────────────────

def test_load_runs_does_not_reconstruct_command_flags_or_defaults(
    tmp_path: Path, monkeypatch,
):
    """The compatibility facade must not reinterpret old runs with live code."""
    monkeypatch.setattr(
        loader_mod,
        "_argparse_defaults",
        lambda: pytest.fail("load_runs consulted current parser defaults"),
    )
    logs = tmp_path / "logs"
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4, lora_r=64)
    cfg["command"] = (
        "python train_lora.py --lr 3e-4 "
        "--optimizer adam-polar-product-lora-coupled --polar_sigma_power 0.0"
    )
    cfg["git_commit"] = "abc1234"
    _write_group(logs, "g", {"scope": ["polar_family"]},
                 [(cfg, _evs((2000, 0.75)))])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora-coupled"},
                     logs_root=str(logs))
    assert len(runs) == 1
    loaded_cfg, _ = runs[0]
    assert "_derived" not in loaded_cfg
    assert "polar_sigma_power" not in loaded_cfg
    assert "effective_picard_iters" not in loaded_cfg
    assert "data_pipeline_version" not in loaded_cfg


def test_dead_reconstruction_helpers_are_removed():
    retired = (
        "HARDCODED_DEFAULT_HISTORY",
        "HISTORICAL_DEFAULTS_WHEN_MISSING",
        "_backfill_optimizer_config",
        "_derive_effective_inner_polar",
        "_derive_effective_polar_pre_norm",
        "_derive_effective_picard_iters",
        "_derive_precond_by_optimizer",
        "_precond_by_optimizer",
        "_backfill_precond",
        "_enrich_cfg",
    )
    assert not [name for name in retired if hasattr(loader_mod, name)]


# ─── exclusion observability: per-group surfacing ─────────────────────────────
#
# Inventory is a record audit, so legacy admission/exclusion policy must not
# change its filesystem, optimizer, or coverage report.

def test_inventory_treats_recorded_commits_as_audit_only(tmp_path: Path):
    """Recorded commit values do not change physical inventory coverage."""
    logs = tmp_path / "logs"

    cfg = _cfg("adamw", 3e-4)
    cfg["git_commit"] = "badcom1abc"
    _write_group(logs, "g_blanket_excluded", {"scope": ["all_optimizers"]},
                 [(cfg, _evs((2000, 0.76)))])
    cfg_ok = _cfg("adamw", 3e-4)
    cfg_ok["git_commit"] = "goodcomm"
    _write_group(logs, "g_ok", {"scope": ["all_optimizers"]},
                 [(cfg_ok, _evs((2000, 0.76)))])

    inv = inventory_runs(str(logs))
    assert inv.groups_loaded == ("g_blanket_excluded", "g_ok")
    assert {row.source_groups for row in inv.coverage} == {
        ("g_blanket_excluded", "g_ok"),
    }
    assert not hasattr(inv, "groups_all_excluded")
    text = render_inventory(inv)
    assert "ALL RUNS EXCLUDED" not in text


def test_load_runs_treats_recorded_commits_as_audit_only(tmp_path: Path, capsys):
    """Commit values remain visible provenance and never gate discovery."""
    logs = tmp_path / "logs"
    cfg = _cfg("adamw", 3e-4)
    cfg["git_commit"] = "badcom1abc"
    _write_group(logs, "phase_L_test", {"scope": ["all_optimizers"]},
                 [(cfg, _evs((2000, 0.76)))])

    runs = load_runs(where={"optimizer": "adamw"}, logs_root=str(logs), quiet=False)
    out = capsys.readouterr().out
    assert out == ""
    assert len(runs) == 1
    assert runs[0][0]["log_group"] == "phase_L_test"
    assert runs[0][0]["git_commit"] == "badcom1abc"


def test_load_runs_warns_on_unknown_where_key(tmp_path: Path):
    """A typo'd where-key (field absent from every cfg) issues a warning
    instead of silently returning empty results."""
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["all_optimizers"]}, [
        (_cfg("adamw", 3e-4), _evs((2000, 0.76))),
    ])
    import warnings as _warn
    with _warn.catch_warnings(record=True) as caught:
        _warn.simplefilter("always")
        load_runs(where={"datset": "anything"}, logs_root=str(logs),
                  warn_cross_commit=False)
        msgs = [str(w.message) for w in caught]
    assert any("datset" in m for m in msgs), (
        f"expected typo-key warning; got: {msgs}"
    )


def test_load_runs_no_warning_on_legit_value_miss(tmp_path: Path):
    """Filtering on a real cfg key with a value that just doesn't match
    must NOT trigger the typo warning (only key-absence does)."""
    logs = tmp_path / "logs"
    _write_group(logs, "g", {"scope": ["all_optimizers"]}, [
        (_cfg("adamw", 3e-4, lora_r=16), _evs((2000, 0.76))),
    ])
    import warnings as _warn
    with _warn.catch_warnings(record=True) as caught:
        _warn.simplefilter("always")
        runs = load_runs(where={"lora_r": 999}, logs_root=str(logs),
                         warn_cross_commit=False)
        msgs = [str(w.message) for w in caught]
    assert runs == []
    # lora_r is a real cfg field; warning should NOT fire.
    assert not any("lora_r" in m and "typo" in m.lower() for m in msgs), (
        f"unexpected typo warning on legit value miss: {msgs}"
    )
