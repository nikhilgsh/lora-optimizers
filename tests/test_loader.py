"""Tests for the predicate-based loader and inventory.

Synthetic-fixture tests — build a temp logs/ tree per test so behavior is
fully isolated from the real project state. Validates:

  - Predicate matching: literal, list, callable.
  - load_runs honors `where`, dedups via key_axes.
  - Newest-wins-on-collision.
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
    HARDCODED_DEFAULT_HISTORY,
    PINNING_ALL_DIVERGED,
    PINNING_HIGH,
    PINNING_INTERIOR,
    PINNING_LOW,
    PINNING_SINGLE,
    _enrich_cfg,
    _matches,
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


# ─── enrichment: effective_inner_polar ────────────────────────────────────────
#
# These tests exercise the single derived field that would have prevented this
# session's polar-method confusion. The raw cfg's polar_method field can be
# "ns" while the effective inner polar is svd_exact (the polar_sigma_power=0.0
# override path); _enrich_cfg must surface the truth.

def _cfg_with_optimizer_config(opt: str, *, polar_method=None,
                                polar_sigma_power=None,
                                picard_iters_override=None,
                                git_commit=None):
    """Mimics post-b0baa4d cfg shape — has explicit optimizer_config dict."""
    cfg = _cfg(opt, 3e-4)
    cfg["optimizer_config"] = {
        "_optim_class": "AdamPolarProductLoRA",
        "polar_method": polar_method,
        "polar_sigma_power": polar_sigma_power,
        "picard_iters_override": picard_iters_override,
    }
    if git_commit is not None:
        cfg["git_commit"] = git_commit
    return cfg


def test_enrich_effective_inner_polar_svd_exact():
    cfg = _cfg_with_optimizer_config(
        "adam-polar-product-lora-coupled",
        polar_method="ns",            # raw says ns ...
        polar_sigma_power=0.0,        # ... but psp=0 overrides to SVD-exact
    )
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_inner_polar"] == "svd_exact"


def test_enrich_effective_inner_polar_sigma_power_nonzero():
    cfg = _cfg_with_optimizer_config(
        "adam-polar-product-lora-coupled",
        polar_method="ns",
        polar_sigma_power=0.125,
    )
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_inner_polar"] == "sigma_power(p=0.125)"


def test_enrich_effective_inner_polar_method_passthrough():
    for pm in ("ns", "ns_hybrid", "polar_express"):
        cfg = _cfg_with_optimizer_config(
            "adam-polar-product-lora-coupled",
            polar_method=pm,
            polar_sigma_power=None,
        )
        _enrich_cfg(cfg)
        assert cfg["_derived"]["effective_inner_polar"] == pm


def test_enrich_effective_inner_polar_none_for_non_polar_optimizer():
    cfg = _cfg("adamw", 3e-4)
    cfg["optimizer_config"] = {"_optim_class": "LoRAPlusAdamW"}
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_inner_polar"] is None


def test_enrich_backfills_log_basic_diagnostics_from_optimizer_config():
    """train.py emits two parallel surfaces for the diagnostics knobs:
    top-level CLI names (``log_basic_diagnostics`` / ``optim_diagnostics_every``)
    and constructor names inside ``optimizer_config`` (same name post-refactor).
    Older cfg events lack the top-level fields entirely. _enrich_cfg backfills
    from optimizer_config so callers reading the top-level field see the
    correct value."""
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4)
    cfg["optimizer_config"] = {
        "_optim_class": "AdamPolarProductLoRA",
        "log_basic_diagnostics": True,
        "diagnostics_every": 80,
    }
    assert cfg.get("log_basic_diagnostics") is None
    assert cfg.get("optim_diagnostics_every") is None
    _enrich_cfg(cfg)
    assert cfg["log_basic_diagnostics"] is True
    assert cfg["optim_diagnostics_every"] == 80


def test_enrich_does_not_override_explicit_log_basic_diagnostics():
    """If the cfg event already has the top-level field set, _enrich_cfg
    leaves it alone — backfill only fills None gaps."""
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4)
    cfg["log_basic_diagnostics"] = False
    cfg["optim_diagnostics_every"] = 200
    cfg["optimizer_config"] = {
        "_optim_class": "AdamPolarProductLoRA",
        "log_basic_diagnostics": True,   # disagrees with top-level
        "diagnostics_every": 20,
    }
    _enrich_cfg(cfg)
    assert cfg["log_basic_diagnostics"] is False
    assert cfg["optim_diagnostics_every"] == 200


def test_enrich_reads_legacy_log_optim_diagnostics_field_names():
    """Pre-2026-05-12-refactor cfgs used `log_optim_diagnostics` (top-level)
    and `log_diagnostics` (in optimizer_config). _enrich_cfg backfills the
    current `log_basic_diagnostics` field from either legacy name so old
    log groups still load with the new field key populated."""
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4)
    cfg["log_optim_diagnostics"] = True   # legacy top-level
    cfg["optimizer_config"] = {
        "_optim_class": "AdamPolarProductLoRA",
        "log_diagnostics": True,   # legacy constructor name
        "diagnostics_every": 80,
    }
    _enrich_cfg(cfg)
    assert cfg["log_basic_diagnostics"] is True
    assert cfg["optim_diagnostics_every"] == 80


def test_enrich_effective_inner_polar_polar_product_pre_feature_falls_back_to_ns():
    """Runs from before commit 4b047f5 (May 3 2026) — when polar_method
    was added — have no polar_method anywhere. Code path was unconditional
    _newton_schulz, so effective polar is 'ns'. picard_k3_r64 is the
    canonical example."""
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4)
    cfg["command"] = "python train_lora.py --optimizer adam-polar-product-lora-coupled"
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_inner_polar"] == "ns"


# ─── enrichment: backfill from command line for old runs ──────────────────────

def test_enrich_backfills_optimizer_config_from_command(tmp_path: Path):
    """Pre-b0baa4d runs lack optimizer_config; _enrich_cfg reconstructs it
    from the command line. The picard_k3_r64-vs-htmuon_polar_k3 confusion
    in this session was caused by manually re-deriving these fields with a
    regex; this test ensures the loader does it once, consistently."""
    cfg = _cfg("adam-polar-product-lora-coupled", 3e-4)
    cfg["command"] = (
        "python train_lora.py --lr 3e-4 "
        "--optimizer adam-polar-product-lora-coupled "
        "--polar_sigma_power 0.0 --lora_r 64"
    )
    # No optimizer_config field — pre-b0baa4d shape.
    _enrich_cfg(cfg)
    opt_cfg = cfg["optimizer_config"]
    assert opt_cfg["_backfilled"] is True
    assert opt_cfg["polar_sigma_power"] == "0.0"  # parse_flag returns string
    # And the derivation correctly identifies SVD-exact:
    assert cfg["_derived"]["effective_inner_polar"] == "svd_exact"


# ─── enrichment: commit-aware effective_picard_iters ──────────────────────────
#
# The build_optimizer default for adam-polar-product-lora-coupled flipped from
# picard_iters=2 to picard_iters=3 in commit dadea5d (May 3 2026). Runs from
# before that commit without --picard_iters_override actually ran k=2; runs
# after ran k=3. Backfilling with the current default would mislabel old runs.

def test_picard_iters_explicit_override_takes_precedence(monkeypatch):
    """If --picard_iters_override is set, no commit lookup is needed."""
    cfg = _cfg_with_optimizer_config(
        "adam-polar-product-lora-coupled",
        picard_iters_override=5,
        git_commit="any_commit_doesnt_matter",
    )
    monkeypatch.setattr(loader_mod, "_is_ancestor",
                        lambda *a, **kw: pytest.fail("should not query git"))
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_picard_iters"] == 5
    assert cfg["_derived"]["effective_picard_iters_certain"] is True


def test_picard_iters_pre_dadea5d_default_is_2(monkeypatch):
    """Run committed before dadea5d → effective k=2 (the historically-correct
    default), not k=3 (the current default)."""
    cfg = _cfg_with_optimizer_config(
        "adam-polar-product-lora-coupled",
        picard_iters_override=None,
        git_commit="d43e04a",  # an actual ancestor of dadea5d in this repo
    )
    # Mock: dadea5d is NOT an ancestor of d43e04a; <initial> always is.
    def fake_is_ancestor(commit, descendant="HEAD"):
        if commit == "<initial>":
            return True
        if commit == "dadea5d" and descendant == "d43e04a":
            return False
        return False
    monkeypatch.setattr(loader_mod, "_is_ancestor", fake_is_ancestor)
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_picard_iters"] == 2
    assert cfg["_derived"]["effective_picard_iters_certain"] is True


def test_picard_iters_post_dadea5d_default_is_3(monkeypatch):
    cfg = _cfg_with_optimizer_config(
        "adam-polar-product-lora-coupled",
        picard_iters_override=None,
        git_commit="3ce7844",  # post-dadea5d
    )
    def fake_is_ancestor(commit, descendant="HEAD"):
        if commit == "<initial>":
            return True
        if commit == "dadea5d" and descendant == "3ce7844":
            return True
        return False
    monkeypatch.setattr(loader_mod, "_is_ancestor", fake_is_ancestor)
    _enrich_cfg(cfg)
    assert cfg["_derived"]["effective_picard_iters"] == 3
    assert cfg["_derived"]["effective_picard_iters_certain"] is True


def test_picard_iters_history_table_contains_known_entry():
    """Regression guard: the registry must contain the documented 2→3 flip.
    If someone deletes this entry, all historical analyses silently shift."""
    history = HARDCODED_DEFAULT_HISTORY[
        ("adam-polar-product-lora-coupled", "picard_iters")
    ]
    commits = {h[0] for h in history}
    values = {h[1] for h in history}
    assert "<initial>" in commits
    assert "dadea5d" in commits
    assert {2, 3} <= values


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
    with pytest.warns(UserWarning, match=r"runs from 2 commits"):
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


# ─── load_runs end-to-end: enrichment is applied ──────────────────────────────

def test_load_runs_enriches_returned_cfgs(tmp_path: Path):
    """End-to-end: a run loaded via load_runs must have _derived populated."""
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
    enriched_cfg, _ = runs[0]
    assert "_derived" in enriched_cfg
    assert enriched_cfg["_derived"]["effective_inner_polar"] == "svd_exact"
    assert "effective_picard_iters" in enriched_cfg["_derived"]


# ─── exclusion observability: per-group surfacing ─────────────────────────────
#
# Regression for the Phase L silent-drop: runs at a blanket-excluded commit
# were dropped with no per-group indication. inventory_runs.groups_all_excluded
# must name the group + dominant reason; the load_runs summary print must
# include a per-reason example list.

def test_inventory_surfaces_blanket_excluded_group(tmp_path: Path, monkeypatch):
    """A group whose every run is on a blanket-excluded commit shows up in
    groups_all_excluded with the dominant exclusion reason."""
    logs = tmp_path / "logs"
    # Inject a synthetic blanket exclusion at runtime so the test doesn't
    # depend on the real commit_exclusions.json contents.
    from lora_playground import commit_exclusions as cx_mod
    monkeypatch.setattr(
        cx_mod, "COMMIT_EXCLUSIONS",
        [("badcom1", "synthetic test exclusion")],
    )

    cfg = _cfg("adamw", 3e-4)
    cfg["git_commit"] = "badcom1abc"
    _write_group(logs, "g_blanket_excluded", {"scope": ["all_optimizers"]},
                 [(cfg, _evs((2000, 0.76)))])
    cfg_ok = _cfg("adamw", 3e-4)
    cfg_ok["git_commit"] = "goodcomm"
    _write_group(logs, "g_ok", {"scope": ["all_optimizers"]},
                 [(cfg_ok, _evs((2000, 0.76)))])

    inv = inventory_runs(str(logs))
    # The blanket-excluded group surfaces with reason.
    excluded_dict = dict(inv.groups_all_excluded)
    assert "g_blanket_excluded" in excluded_dict, (
        f"expected g_blanket_excluded in inventory.groups_all_excluded; "
        f"got {inv.groups_all_excluded}"
    )
    assert "synthetic test exclusion" in excluded_dict["g_blanket_excluded"]
    # The OK group must NOT appear in groups_all_excluded.
    assert "g_ok" not in excluded_dict
    # render_inventory output mentions the group name and reason.
    text = render_inventory(inv)
    assert "g_blanket_excluded" in text
    assert "ALL RUNS EXCLUDED" in text


def test_load_runs_summary_includes_group_examples(tmp_path: Path, monkeypatch, capsys):
    """The exclusion summary print must name an example (group, log_filename)
    so a user can identify which sweep got swept up."""
    logs = tmp_path / "logs"
    from lora_playground import commit_exclusions as cx_mod
    monkeypatch.setattr(
        cx_mod, "COMMIT_EXCLUSIONS",
        [("badcom1", "synthetic blanket exclusion")],
    )
    cfg = _cfg("adamw", 3e-4)
    cfg["git_commit"] = "badcom1abc"
    _write_group(logs, "phase_L_test", {"scope": ["all_optimizers"]},
                 [(cfg, _evs((2000, 0.76)))])

    runs = load_runs(where={"optimizer": "adamw"}, logs_root=str(logs))
    out = capsys.readouterr().out
    assert "phase_L_test" in out, (
        f"summary print missing group name; got: {out!r}"
    )
    # The exclusion blocks user filter from seeing the run.
    assert len(runs) == 0


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
