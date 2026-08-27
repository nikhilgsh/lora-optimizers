"""Regression tests for notebook panel-to-arm wiring."""

import inspect

import matplotlib.pyplot as plt
import pytest


def test_legacy_panels_do_not_manufacture_stable_ids_from_current_labels():
    from lora_playground.plotting import paper_plots_lib as plots

    source = inspect.getsource(plots)
    assert not hasattr(plots, "_records_specs")
    assert "paper.live.v1" not in source


def test_adamw_beta2_panel_reference_is_a_declared_arm(monkeypatch):
    from lora_playground.plotting import paper_plots_lib as plots

    called = []

    def capture(view_id, suptitle=None):
        called.append((view_id, suptitle))
        return "summary"

    monkeypatch.setattr(plots, "_archive_figure", capture)
    assert plots.adamw_beta2_panel(256) == "summary"
    assert called[0][0] == "paper.adamw_beta2.v1"
    with pytest.raises(ValueError, match="sealed only for rank 256"):
        plots.adamw_beta2_panel(64)


def test_matched_precond_panel_excludes_unversioned_adamw(monkeypatch):
    from lora_playground.plotting import paper_plots_lib as plots

    captured = {}

    def capture(arms, _workload, _reference, _title, **kwargs):
        captured["arms"] = arms
        captured["kwargs"] = kwargs

    monkeypatch.setattr(plots, "_records_figure", capture)
    plots.precond_panel(256, matched_revision=True)

    assert set(captured["arms"]) == {
        "product: C_B=B^T P B, C_A=A Q A^T",
        "one-sided: C_B=C_A=I",
        "factorwise: C_B=P_A, C_A=Q_B",
    }
    assert captured["kwargs"] == {
        "target_label": None,
        "semantic_view": "precond_matched",
        "measurement_semantics_revision": plots.MEASUREMENT_SEMANTICS_REVISION,
    }

def test_priority_notebook_panels_execute_against_recorded_evidence(monkeypatch):
    """Exercise the exact Figure 14--16 calls that previously raised KeyError."""
    from lora_playground.plotting import paper_plots_lib as plots

    monkeypatch.setattr(plots.plt, "show", lambda: None)
    try:
        ablation = plots.ablation_panel(256)
        derivation = plots.derivation_ablation_panel(256)
        preconditioner = plots.precond_panel(256)
    finally:
        plt.close("all")

    assert "Polar-LoRA (kl-diag)" in set(ablation["variant"])
    assert "PoLoRA: rxr=B^T P B, shared P,Q" in set(derivation["variant"])
    assert "factorwise: C_B=P_A, C_A=Q_B" in set(preconditioner["variant"])
