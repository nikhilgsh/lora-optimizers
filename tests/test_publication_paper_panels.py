"""Paper consumers use stable IDs from the sealed publication archive."""
from __future__ import annotations

import pytest

from lora_playground.publication_paper import (
    labeled_completed,
    publication_workload_view_panel,
)
from lora_playground.workloads import find_workload


def test_cross_workload_view_keeps_stable_identity():
    workload = find_workload(
        "meta-llama/Llama-3.2-1B", "openmath", 256
    )
    panel = publication_workload_view_panel(
        "paper.adamw_polora.all_workloads.v1", workload,
    )

    assert panel.variant_id("Adam") != "Adam"
    assert panel.variant_id("PoLoRA") != "PoLoRA"
    assert panel.comparison.best_completed[panel.variant_id("Adam")] is not None
    assert panel.comparison.best_completed[panel.variant_id("PoLoRA")] is not None
    assert set(labeled_completed(panel)) == {"Adam", "PoLoRA"}
def test_cross_workload_view_rejects_unknown_view():
    workload = find_workload(
        "meta-llama/Llama-3.2-1B", "openmath", 256
    )
    with pytest.raises(KeyError, match="not declared"):
        publication_workload_view_panel("paper.missing.v1", workload)


@pytest.mark.parametrize(
    ("panel", "has_target"),
    (
        (lambda plots: plots.msign_panel(256), True),
        (lambda plots: plots.magnitude_rule_panel(256), True),
        (lambda plots: plots.beta2_panel(256), False),
        (lambda plots: plots.adamw_beta2_panel(256), False),
    ),
)
def test_archive_backed_notebook_panels_never_call_live_loader(
    monkeypatch, panel, has_target,
):
    from lora_playground.plotting import paper_plots_lib as plots

    monkeypatch.setattr(
        plots,
        "load_runs",
        lambda *args, **kwargs: pytest.fail("archive panel called load_runs"),
    )
    monkeypatch.setattr(
        plots,
        "logs_signature",
        lambda *args, **kwargs: pytest.fail("archive panel scanned logs"),
    )
    seen = {}

    def render(comparison, *, reference_id, target_id, **kwargs):
        seen["comparison"] = comparison
        seen["reference_id"] = reference_id
        seen["target_id"] = target_id
        return object(), object(), "summary"

    monkeypatch.setattr(plots, "render_comparison", render)
    monkeypatch.setattr(plots.plt, "show", lambda: None)
    assert panel(plots) == "summary"
    assert seen["reference_id"] in {
        spec.id for spec in seen["comparison"].variants
    }
    assert (seen["target_id"] is not None) is has_target


def test_archive_backed_panel_fails_closed_for_unknown_view():
    from lora_playground.plotting import paper_plots_lib as plots

    with pytest.raises(KeyError, match="not declared"):
        plots._archive_figure("paper.missing.v1")


def test_archive_backed_paper_figs_pair_never_calls_workload_runs(monkeypatch):
    from lora_playground.plotting import paper_figs

    monkeypatch.setattr(
        paper_figs,
        "workload_runs",
        lambda *args, **kwargs: pytest.fail("archive figure called workload_runs"),
    )
    workload = find_workload(
        "meta-llama/Llama-3.2-1B", "openmath", 256
    )
    panel, labeled = paper_figs._adam_polora_comparison(workload)
    assert set(labeled) == {"Adam", "PoLoRA"}
    assert panel.variant_id("Adam") != "Adam"


def test_fig2_uses_declarative_archive_view(monkeypatch, tmp_path):
    from lora_playground.plotting import paper_figs

    monkeypatch.setattr(
        paper_figs,
        "workload_runs",
        lambda *args, **kwargs: pytest.fail("fig2 called the live workload loader"),
    )
    monkeypatch.setattr(paper_figs, "FIGS", tmp_path)
    fig = paper_figs.fig2()
    assert len(fig.axes) == 2
