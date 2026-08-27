"""Paper consumers use stable IDs from the sealed publication archive."""
from __future__ import annotations

import pytest

from lora_playground.publication_paper import (
    labeled_completed,
    publication_panel,
)
from lora_playground.workloads import find_workload


POLORA = (
    "KL-diag +polar PE=8 (f=10, β_c=0.99, δ=1e-4) "
    "H=8 precond_method=gram_ns"
)


def test_publication_panel_keeps_stable_identity_under_editorial_labels():
    workload = find_workload(
        "meta-llama/Llama-3.2-1B", "openmath", 256
    )
    panel = publication_panel(
        workload,
        {"Adam": "AdamW", "PoLoRA": POLORA},
    )

    assert panel.variant_id("Adam") != "Adam"
    assert panel.variant_id("PoLoRA") != "PoLoRA"
    assert panel.comparison.best_completed[panel.variant_id("Adam")] is not None
    assert panel.comparison.best_completed[panel.variant_id("PoLoRA")] is not None
    assert set(labeled_completed(panel)) == {"Adam", "PoLoRA"}


def test_publication_panel_rejects_unknown_or_duplicate_sealed_views():
    workload = find_workload(
        "meta-llama/Llama-3.2-1B", "openmath", 256
    )
    with pytest.raises(KeyError, match="absent from archive"):
        publication_panel(workload, {"candidate": "not a sealed variant"})
    with pytest.raises(ValueError, match="selected more than once"):
        publication_panel(workload, {"Adam": "AdamW", "baseline": "AdamW"})


def test_archive_backed_notebook_panel_never_calls_live_loader(monkeypatch):
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

    def render(*, comparison, reference_id, target_id, **kwargs):
        seen["comparison"] = comparison
        seen["reference_id"] = reference_id
        seen["target_id"] = target_id
        return object(), object(), "summary"

    monkeypatch.setattr(plots, "compare_variants_figure", render)
    monkeypatch.setattr(plots.plt, "show", lambda: None)
    assert plots.beta2_panel(256) == "summary"
    assert seen["reference_id"] in {
        spec.id for spec in seen["comparison"].variants
    }
    assert seen["target_id"] is None


def test_archive_backed_panel_fails_closed_when_workload_lacks_arm():
    from lora_playground.plotting import paper_plots_lib as plots

    workload = find_workload("allenai/OLMo-2-0425-1B", "opc", 64)
    with pytest.raises(ValueError, match="no OLMo/opc/r64 evidence"):
        plots._archive_figure(
            {"PoLoRA": POLORA},
            workload,
            "PoLoRA",
            "missing archive arm",
            target_label=None,
        )


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
