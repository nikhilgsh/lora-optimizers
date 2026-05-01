"""Tests for the predicate-based loader and inventory.

Synthetic-fixture tests — build a temp logs/ tree per test so behavior is
fully isolated from the real project state. Validates:

  - Predicate matching: literal, list, callable.
  - load_runs honors `where`, dedups via key_axes, excludes deprecated groups.
  - Newest-wins-on-collision (replaces the destructive supersedes mechanism).
  - inventory_runs detects orphans, deprecated, unknown optimizers, and
    pinning at the lr-range boundary.
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
    _matches,
    inventory_runs,
    load_runs,
    render_inventory,
)


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


def test_load_runs_excludes_deprecated_groups(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g_old", {"scope": ["polar_family"], "deprecated": True}, [
        (_cfg("adam-polar-product-lora", 3e-4), _evs((2000, 0.74))),
    ])
    _write_group(logs, "g_new", {"scope": ["polar_family"]}, [
        (_cfg("adam-polar-product-lora", 1e-3), _evs((2000, 0.78))),
    ])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"}, logs_root=str(logs))
    assert len(runs) == 1
    cfg, _ = runs[0]
    assert float(cfg["lr"]) == 1e-3, "deprecated group's run leaked through"


def test_load_runs_newest_wins_on_dedup_collision(tmp_path: Path):
    """When two non-deprecated groups have a colliding key, newer wins.
    Replaces the role the destructive `supersedes` field used to play."""
    logs = tmp_path / "logs"
    cfg = _cfg("adam-polar-product-lora", 3e-4)
    _write_group(logs, "g_old",
                 {"scope": ["polar_family"], "submitted_at": "2026-04-01T00:00:00-04:00"},
                 [(cfg, _evs((2000, 0.80)))])
    _write_group(logs, "g_new",
                 {"scope": ["polar_family"], "submitted_at": "2026-04-30T00:00:00-04:00"},
                 [(cfg, _evs((2000, 0.74)))])
    runs = load_runs(where={"optimizer": "adam-polar-product-lora"}, logs_root=str(logs))
    assert len(runs) == 1, "dedup didn't collapse colliding runs"
    _, evs = runs[0]
    assert evs[-1]["eval_loss"] == 0.74


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


def test_inventory_detects_deprecated(tmp_path: Path):
    logs = tmp_path / "logs"
    _write_group(logs, "g_dep", {"scope": ["polar_family"], "deprecated": True}, [
        (_cfg("adam-polar-product-lora", 3e-4), _evs((2000, 0.74))),
    ])
    inv = inventory_runs(str(logs))
    assert "g_dep" in inv.groups_deprecated
    assert "g_dep" not in inv.groups_loaded


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
    assert "Loaded" in text
    assert "Coverage" in text
