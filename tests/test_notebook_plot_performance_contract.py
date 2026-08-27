"""Deterministic performance contracts for paper-notebook plot entry points.

These tests intentionally count expensive boundaries instead of asserting on
wall time.  The latter varies with filesystem cache state and shared-host load;
an extra full logs-tree signature or catalog traversal is the regression that
causes the user-visible delay.
"""
from __future__ import annotations

import pytest


def _install_legacy_panel_probes(monkeypatch):
    """Replace rendering work while preserving the panel's data-access path."""
    from lora_playground.plotting import paper_plots_lib as plots

    counts = {"logs_signature": 0, "load_runs": 0}
    runs = [
        (
            {"label": "AdamW", "optimizer": "adamw", "lr": 1e-4},
            [{"step": plots.HORIZON, "eval_loss": 0.75}],
        )
    ]

    def signature(_logs_root):
        counts["logs_signature"] += 1
        return "unchanged-tree"

    def load_runs(**_kwargs):
        counts["load_runs"] += 1
        return runs

    monkeypatch.setattr(plots, "logs_signature", signature)
    monkeypatch.setattr(plots, "load_runs", load_runs)
    monkeypatch.setattr(
        plots,
        "variant_key_fn",
        lambda _common, _arms: lambda cfg: cfg["label"],
    )
    monkeypatch.setattr(
        plots, "assert_curve_source_coherent", lambda *_args, **_kwargs: None
    )

    def render(*_args, prefetched_runs, **_kwargs):
        # This keyword is load-bearing: omitting it makes the comparison helper
        # issue one catalog query per displayed arm.
        assert prefetched_runs is runs
        return object(), object(), "summary"

    monkeypatch.setattr(plots, "compare_variants_figure", render)
    monkeypatch.setattr(plots.plt, "show", lambda: None)

    def speedup(*_args, runs=None, **_kwargs):
        assert runs is not None
        return "speedup", [], 0.75

    monkeypatch.setattr(plots, "speedup_table", speedup)

    def coverage(_arms, common):
        # Exercise multiple downstream readers.  They must see the same cached
        # snapshot without another filesystem signature or catalog traversal.
        assert plots.cell_runs(common) is runs
        assert plots.cell_runs(common) is runs
        return ""

    monkeypatch.setattr(plots, "coverage_report", coverage)
    plots.clear_runs_cache()
    return plots, counts


def test_panel_n_uses_one_live_catalog_snapshot_per_render(monkeypatch):
    """The compatibility panel may scan/load once, never once per arm/consumer."""
    plots, counts = _install_legacy_panel_probes(monkeypatch)

    assert plots.panel_n(0) == "summary"

    assert counts == {"logs_signature": 1, "load_runs": 1}


def test_repeated_panel_n_reuses_parsed_cell_when_logs_are_unchanged(monkeypatch):
    """A second render rechecks freshness once but does not reparse the catalog."""
    plots, counts = _install_legacy_panel_probes(monkeypatch)

    assert plots.panel_n(0) == "summary"
    assert plots.panel_n(0) == "summary"

    assert counts == {"logs_signature": 2, "load_runs": 1}


def test_archive_panel_never_touches_live_logs_and_parses_archive_once(monkeypatch):
    """Reviewed panels stay independent of the large mutable logs tree."""
    import lora_playground.publication_paper as publication_paper
    from lora_playground.plotting import paper_plots_lib as plots

    archive = publication_paper.load_publication_archive(
        publication_paper.DEFAULT_PUBLICATION_ARCHIVE
    )
    publication_paper.load_paper_publication_archive.cache_clear()
    publication_paper._default_specs_by_label.cache_clear()
    archive_loads = 0

    def load_archive(_path):
        nonlocal archive_loads
        archive_loads += 1
        return archive

    def fail_live_access(*_args, **_kwargs):
        pytest.fail("archive-backed panel touched the live logs tree")

    monkeypatch.setattr(publication_paper, "load_publication_archive", load_archive)
    monkeypatch.setattr(plots, "load_runs", fail_live_access)
    monkeypatch.setattr(plots, "logs_signature", fail_live_access)
    monkeypatch.setattr(
        plots,
        "compare_variants_figure",
        lambda **_kwargs: (object(), object(), "summary"),
    )
    monkeypatch.setattr(plots.plt, "show", lambda: None)

    try:
        assert plots.beta2_panel(256) == "summary"
        assert plots.beta2_panel(256) == "summary"
        assert archive_loads == 1
    finally:
        # Do not leak a monkeypatch-created cached object into later tests.
        publication_paper.load_paper_publication_archive.cache_clear()
        publication_paper._default_specs_by_label.cache_clear()
