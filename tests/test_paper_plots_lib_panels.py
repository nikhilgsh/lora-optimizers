"""Regression tests for notebook panel-to-arm wiring."""

import inspect

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
        plots._arms.PRECOND_PRODUCT_LABEL,
        r"Identity: $C_B=C_A=I$",
        r"Factorwise: $C_B=P_A,\ C_A=Q_B$",
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
        trajectories = {
            line.get_gid(): line for line in plt.gcf().axes[1].get_lines()
        }
        assert plt.gcf().axes[0].xaxis.label.get_fontsize() >= 14
        assert min(
            text.get_fontsize() for text in plt.gcf().legends[0].get_texts()
        ) >= 11
        assert to_rgba(trajectories["trajectory:AdamW"].get_color()) \
            == to_rgba("black")
        assert to_rgba(trajectories[
            f"trajectory:{plots._arms.PRECOND_PRODUCT_LABEL}"
        ].get_color()) != to_rgba("black")
    finally:
        plt.close("all")

    assert "PoLoRA" in set(ablation["variant"])
    assert "PoLoRA" in set(derivation["variant"])
    # Every reported arm resolves to a declared one -- this is the KeyError
    # regression the test exists for. Factorwise is deliberately NOT asserted
    # present: whether it appears is a fact about the runs on disk, not about
    # this code. Every KL-Shampoo run in this cell predates the
    # factorwise-slot fix, so the cohort filter excludes them all and the arm
    # is legitimately empty until post-fix runs land. Asserting its presence
    # is what kept the pre-fix arm rendering.
    declared = set(plots._arms.PRECOND_ARMS)
    reported = set(preconditioner["variant"])
    assert reported <= declared, reported - declared
    assert {"AdamW", plots._arms.PRECOND_PRODUCT_LABEL} <= reported


def test_arm_matching_ignores_the_optimizer_impl_revision():
    """An arm names an algorithm choice, not the code revision that ran it.

    `labels._shared_knobs` suffixes ` impl-rev=N` so that two revisions of one
    config stay two series. Arm predicates come from `arms.arm()`, which pins
    OptimizerConfig fields; `optimizer_impl_revision` is not one (no CLI flag
    sets it -- `run_schema` stamps it from the optimizer class), so the
    predicate side can never carry the suffix. Comparing FULL labels therefore
    matched no arm for any run recording revision 2, which emptied the
    reference arm of the Qwen2.5/openmath/r16 matched panel and raised
    "has no recorded reference arm".
    """
    from lora_playground.plotting.labels import (
        canonical_arm_label, canonical_label,
    )

    cfg = {"optimizer": "kl-diag-polar-lora", "precond": "product",
           "diag_metric": True, "use_polar": True}
    versioned = {**cfg, "optimizer_impl_revision": 2}

    assert canonical_label(versioned) != canonical_label(cfg), (
        "series identity must still see the revision"
    )
    assert canonical_arm_label(versioned) == canonical_arm_label(cfg), (
        "arm identity must not see the revision"
    )
