"""Behavioral contract for the rendering-only comparison adapter."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lora_playground.comparison import VariantSpec, build_comparison
from lora_playground.plotting.render import render_comparison


HORIZON = 1000


def _run(kind, lr, losses, *, aborted=False):
    cfg = {"kind": kind, "lr": lr, "max_steps": HORIZON}
    if aborted:
        cfg["_aborted"] = {"reason": "test"}
    history = [{"step": step, "eval_loss": loss} for step, loss in losses]
    return cfg, history


def _spec(variant_id, label=None, style_key=None):
    return VariantSpec(
        variant_id,
        label or variant_id,
        {"kind": variant_id},
        style_key=style_key,
    )


def _trajectory(ax, variant_id):
    matches = [line for line in ax.get_lines()
               if line.get_gid() == f"trajectory:{variant_id}"]
    assert len(matches) == 1
    return matches[0]


def test_labels_and_styles_do_not_change_selected_data():
    runs = [
        _run("base", 0.1, [(500, 0.70), (1000, 0.60)]),
        _run("candidate", 0.1, [(500, 0.68), (1000, 0.58)]),
        _run("candidate", 0.2, [(500, 0.72), (1000, 0.63)]),
    ]
    result = build_comparison(
        runs,
        (_spec("base", "Base", "old-base"),
         _spec("candidate", "Candidate", "old-candidate")),
        HORIZON,
        completion_slack=0,
    )
    restyled = replace(
        result,
        variants=(
            _spec("base", "Reference renamed", "new-base"),
            _spec("candidate", "Method renamed", "new-candidate"),
        ),
    )

    fig_a, table_a, summary_a = render_comparison(
        result,
        reference_id="base",
        horizon=HORIZON,
        colors={"old-base": "black", "old-candidate": "red"},
    )
    fig_b, table_b, summary_b = render_comparison(
        restyled,
        reference_id="base",
        horizon=HORIZON,
        colors={"new-base": "purple", "new-candidate": "green"},
        markers={"new-base": "D", "new-candidate": "X"},
    )

    np.testing.assert_allclose(table_a.to_numpy(), table_b.to_numpy())
    np.testing.assert_allclose(
        summary_a[["best_lr", "final"]].to_numpy(),
        summary_b[["best_lr", "final"]].to_numpy(),
    )
    for variant_id in ("base", "candidate"):
        line_a = _trajectory(fig_a.axes[1], variant_id)
        line_b = _trajectory(fig_b.axes[1], variant_id)
        np.testing.assert_allclose(line_a.get_xdata(), line_b.get_xdata())
        np.testing.assert_allclose(line_a.get_ydata(), line_b.get_ydata())
    plt.close(fig_a)
    plt.close(fig_b)


def test_completed_table_and_partial_trajectory_precedence():
    runs = [
        _run("complete", 0.2, [(500, 0.65), (1000, 0.60)]),
        _run("complete", 0.1, [(300, 0.59), (600, 0.55)]),
        _run("rescued", 0.2, [(300, 2.0), (500, float("nan"))],
             aborted=True),
        _run("rescued", 0.1, [(300, 0.61), (600, 0.57)]),
        _run("partial-only", 0.1, [(300, 0.63), (600, 0.59)]),
    ]
    result = build_comparison(
        runs,
        (_spec("complete"), _spec("rescued"), _spec("partial-only")),
        HORIZON,
        completion_slack=0,
    )
    fig, table, summary = render_comparison(
        result,
        reference_id="complete",
        horizon=HORIZON,
    )

    assert table.index.tolist() == [0.2]
    assert np.isnan(table.loc[0.2, "partial-only"])
    assert summary["variant"].tolist() == ["complete", "rescued"]

    completed_line = _trajectory(fig.axes[1], "complete")
    rescued_line = _trajectory(fig.axes[1], "rescued")
    partial_line = _trajectory(fig.axes[1], "partial-only")
    assert completed_line.get_xdata()[-1] == HORIZON
    assert "partial" not in completed_line.get_label()
    assert rescued_line.get_xdata()[-1] == 600
    assert r"$\eta$=0.1" in rescued_line.get_label()
    assert "partial @600" in rescued_line.get_label()
    assert partial_line.get_xdata()[-1] == 600
    assert "partial @600" in partial_line.get_label()
    assert fig.axes[1].get_xlim()[1] >= HORIZON
    plt.close(fig)


def test_nonfinite_completed_point_is_a_hollow_marker():
    result = build_comparison(
        [
            _run("base", 0.1, [(1000, 0.55)]),
            _run("base", 0.2, [(500, float("nan"))], aborted=True),
        ],
        (_spec("base"),),
        HORIZON,
        completion_slack=0,
    )
    fig, table, _summary = render_comparison(
        result,
        reference_id="base",
        horizon=HORIZON,
        final_ylim=(0.50, 0.60),
        colors={"base": "red"},
        markers={"base": "o"},
    )

    assert np.isnan(table.loc[0.2, "base"])
    hollow = [
        line for line in fig.axes[0].get_lines()
        if line.get_markerfacecolor() == "none"
        and list(line.get_xdata()) == [0.2]
    ]
    assert len(hollow) == 1
    assert list(hollow[0].get_ydata()) == [0.60]
    plt.close(fig)


def test_best_completed_mapping_drives_summary_and_trajectory():
    result = build_comparison(
        [
            _run("base", 0.1, [(500, 0.60), (1000, 0.50)]),
            _run("base", 0.2, [(500, 0.75), (1000, 0.70)]),
        ],
        (_spec("base"),),
        HORIZON,
        completion_slack=0,
    )
    assert result.best_completed["base"].lr == 0.1
    forced = replace(
        result,
        best_completed=MappingProxyType({
            "base": result.completed["base"][0.2],
        }),
    )

    fig, table, summary = render_comparison(
        forced,
        reference_id="base",
        horizon=HORIZON,
        auto_ylim=False,
    )

    assert table.index.tolist() == [0.1, 0.2]
    assert summary.loc[0, "best_lr"] == 0.2
    assert summary.loc[0, "final"] == 0.70
    selected = _trajectory(fig.axes[1], "base")
    assert r"$\eta$=0.2" in selected.get_label()
    np.testing.assert_allclose(selected.get_ydata(), [0.75, 0.70])
    plt.close(fig)
