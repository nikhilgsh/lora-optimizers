"""Deterministic I/O budgets for the notebook's compatibility loader."""
from __future__ import annotations

import pytest


@pytest.fixture
def loader_probes(monkeypatch):
    from lora_playground.plotting import paper_plots_lib as plots

    counts = {"logs_signature": 0, "catalog_discovery": 0, "load_runs": 0}
    catalog = object()
    runs = [({"optimizer": "adamw"}, [{"step": 9000, "eval_loss": 0.8}])]

    def signature(logs_root):
        assert logs_root == str(plots.ROOT / "logs")
        counts["logs_signature"] += 1
        return "unchanged"

    def discover(logs_root):
        assert logs_root == plots.ROOT / "logs"
        counts["catalog_discovery"] += 1
        return catalog

    def load_runs(where, *, catalog: object, warn_cross_commit: bool, quiet: bool):
        assert where
        assert catalog is not None
        assert warn_cross_commit is False
        assert quiet is True
        counts["load_runs"] += 1
        return runs

    monkeypatch.setattr(plots, "logs_signature", signature)
    monkeypatch.setattr(plots.RunCatalog, "discover", discover)
    monkeypatch.setattr(plots, "load_runs", load_runs)
    plots.clear_runs_cache()
    try:
        yield plots, runs, counts
    finally:
        plots.clear_runs_cache()


def test_repeated_cell_read_reuses_parsed_runs(loader_probes):
    plots, runs, counts = loader_probes
    where = {"model_name": "model-a"}
    assert plots.cell_runs(where) is runs
    assert plots.cell_runs(where) is runs
    assert counts == {"logs_signature": 2, "catalog_discovery": 1, "load_runs": 1}


def test_one_panel_checks_the_log_tree_once(loader_probes):
    plots, runs, counts = loader_probes
    where = {"model_name": "model-a"}
    with plots._held_logs_signature():
        assert plots.cell_runs(where) is runs
        assert plots.cell_runs(where) is runs
    assert counts == {"logs_signature": 1, "catalog_discovery": 1, "load_runs": 1}


def test_distinct_cells_share_one_catalog_snapshot(loader_probes):
    plots, runs, counts = loader_probes
    assert plots.cell_runs({"lora_r": 16}) is runs
    assert plots.cell_runs({"lora_r": 64}) is runs
    assert counts == {"logs_signature": 2, "catalog_discovery": 1, "load_runs": 2}


def test_notebook_snapshot_checks_tree_once_across_distinct_cells(loader_probes):
    plots, runs, counts = loader_probes

    plots.begin_notebook_snapshot(refresh=True)
    assert plots.cell_runs({"lora_r": 16}) is runs
    assert plots.cell_runs({"lora_r": 64}) is runs
    # Re-entering without an explicit refresh preserves the same snapshot.
    plots.begin_notebook_snapshot()

    assert counts == {
        "logs_signature": 1,
        "catalog_discovery": 1,
        "load_runs": 2,
    }

    plots.end_notebook_snapshot()
    assert plots.cell_runs({"lora_r": 128}) is runs
    assert counts == {
        "logs_signature": 2,
        "catalog_discovery": 1,
        "load_runs": 3,
    }


def test_rank_panel_reuses_canonical_figure_without_live_io(
    loader_probes, monkeypatch,
):
    plots, _runs, counts = loader_probes
    from lora_playground.plotting import paper_figs

    sentinel = object()
    seen = {}

    def fig3(*, ranks, figsize, save):
        seen.update(ranks=ranks, figsize=figsize, save=save)
        return sentinel

    monkeypatch.setattr(paper_figs, "fig3", fig3)
    assert plots.rank_lr_panel(ranks=(16, 64)) is sentinel
    assert seen == {"ranks": (16, 64), "figsize": (13, 6.0), "save": False}
    assert counts == {
        "logs_signature": 0,
        "catalog_discovery": 0,
        "load_runs": 0,
    }
