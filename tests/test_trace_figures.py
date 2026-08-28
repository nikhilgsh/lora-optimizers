"""Regression tests for the per-step diagnostic trace primitive.

Each test here is a bug the three diagnostics panels carried while each one
hand-rolled its own matplotlib bookkeeping.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from lora_playground.plotting.traces import (
    Trace, TracePanel, trace_figure, two_key_legend,
)


def _arrays(n=5, **fields):
    base = {"step": np.arange(n, dtype=float)}
    base.update({k: np.full(n, v, dtype=float) for k, v in fields.items()})
    return base


def test_a_band_is_requested_only_where_a_band_is_drawn():
    """Asking for an undrawn band silently shortens the trace.

    `loading.load_optim_step_diagnostics` keeps a step only when EVERY
    requested column is present on it (`loading.py:395`), and a step whose
    per-pair values were all NaN records `_median` alone (`optim.py:961`). So
    requesting `_min`/`_max` for a field drawn without a band drops exactly the
    steps where the diagnostic degraded -- the steps most worth seeing.
    """
    from lora_playground.plotting.paper_plots_lib import _diagnostic_fields

    panels = (
        TracePanel("banded", (Trace("wide"),)),
        TracePanel("bare", (Trace("thin", band=False),)),
    )
    fields = _diagnostic_fields(panels)
    assert fields[0] == "step"
    assert {"wide_median", "wide_min", "wide_max"} <= set(fields)
    assert "thin_median" in fields
    assert not {"thin_min", "thin_max"} & set(fields)


def test_a_field_the_run_never_logged_names_itself_and_what_was_logged():
    """The panel author gets the answer, not a bare KeyError from numpy."""
    series = {"arm": _arrays(**{"recorded_median": 1.0})}
    panels = (TracePanel("y", (Trace("absent", band=False),)),)
    with pytest.raises(KeyError) as excinfo:
        trace_figure(series, panels, suptitle="t")
    message = str(excinfo.value)
    assert "absent" in message and "recorded_median" in message
    plt.close("all")


def test_colour_encodes_the_series_and_linestyle_the_field():
    """Two fields on one panel must not consume two series colours.

    Colour has to mean the same thing on every panel of a figure, or the panels
    cannot be read against each other.
    """
    series = {
        "first": _arrays(a_median=1.0, b_median=2.0),
        "second": _arrays(a_median=3.0, b_median=4.0),
    }
    panels = (TracePanel(
        "a solid, b dashed",
        (Trace("a", band=False), Trace("b", ls="--", band=False)),
    ),)
    fig = trace_figure(series, panels, suptitle="t")
    try:
        lines = fig.axes[0].get_lines()
        assert len(lines) == 4
        by_style = {}
        for line in lines:
            by_style.setdefault(line.get_linestyle(), []).append(
                line.get_color())
        assert len(by_style) == 2, "the two fields must differ by linestyle"
        for colours in by_style.values():
            assert len(set(colours)) == 2, "each field spans both series"
        # The same two colours in both linestyles: colour is the series.
        assert set(by_style["-"]) == set(next(
            v for k, v in by_style.items() if k != "-"))
        assert [text.get_text() for text in fig.legends[0].get_texts()] \
            == ["first", "second"]
    finally:
        plt.close("all")


def test_spare_axes_are_hidden_rather_than_drawn_empty():
    series = {"arm": _arrays(a_median=1.0)}
    panels = tuple(TracePanel(f"y{i}", (Trace("a", band=False),))
                   for i in range(3))
    fig = trace_figure(series, panels, suptitle="t")
    try:
        assert len(fig.axes) == 4
        assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    finally:
        plt.close("all")


def test_the_two_legend_keys_anchor_to_the_axes_not_to_picked_fractions():
    """Adding an entry must not walk one key box onto the other.

    The hand-rolled version anchored the two boxes at y=0.66 and y=0.16 of the
    axes, so a seventh learning rate grew the upper box straight into the lower.
    """
    fig, ax = plt.subplots()
    try:
        two_key_legend(
            ax,
            colors={f"eta{i}": f"C{i}" for i in range(7)},
            color_title="Learning rate",
            styles={"Solved": "-", "Bounded": "--"},
            style_title="Magnitude rule",
        )
        legends = [child for child in ax.get_children()
                   if isinstance(child, matplotlib.legend.Legend)]
        assert len(legends) == 2
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = sorted((legend.get_window_extent(renderer)
                        for legend in legends), key=lambda box: box.y0)
        assert boxes[0].y1 <= boxes[1].y0, "the two key boxes overlap"
    finally:
        plt.close("all")


def test_the_rank_panel_plots_the_aggregated_ratio_not_a_ratio_of_medians():
    """`sigma_ratio` is aggregated across LoRA pairs by the optimizer.

    `optim_diagnostics.py:85` says it in as many words -- ratio-of-medians is
    not median-of-ratios -- and this panel used to divide `sigma_max_B_median`
    by `sigma_max_A_median`, which is both the wrong statistic and the
    reciprocal of what the banded panel beside it plots.
    """
    from lora_playground.plotting.paper_plots_lib import (
        _BY_RANK_PANELS, _RESCALE_PANELS,
    )

    ratio_panels = [panel for panels in (_BY_RANK_PANELS, _RESCALE_PANELS)
                    for panel in panels
                    for trace in panel.traces if trace.field == "sigma_ratio"]
    assert len(ratio_panels) == 2, "both rescaling figures plot the ratio"
    for panel in ratio_panels:
        assert panel.ref == 1.0, "the balanced representative is the reference"
        assert r"\|A\|_2/\|B\|_2" in panel.ylabel, panel.ylabel
