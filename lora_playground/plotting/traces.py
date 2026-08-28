"""Multi-panel figures of per-step optimizer diagnostics.

A diagnostics panel is a small amount of information -- which aggregate fields
go on which axes, what reference line each carries -- wrapped in a large amount
of matplotlib bookkeeping: a figure under the notebook rcParams, a colour per
series, a median line and a min-max band per field, a shared x label, one
outside legend, a suptitle. Three panels in ``paper_plots_lib`` each wrote that
bookkeeping out longhand, so adding a fourth meant copying sixty lines and
adding a new place for the palette, the band alpha and the legend placement to
drift apart.

Declare the axes instead::

    trace_figure(
        {"PoLoRA": arrays, "Neither": other},
        (TracePanel(r"$\\|A\\|_2/\\|B\\|_2$", (Trace("sigma_ratio"),), ref=1.0),),
        suptitle="Factor scale over training",
    )

``arrays`` is what ``loading.load_optim_step_diagnostics`` returns for one run:
a mapping from aggregate field name to a numpy array over steps. A ``Trace``
names the field's STEM (``sigma_ratio``), and the figure reads
``sigma_ratio_median`` for the line and ``_min``/``_max`` for the band, because
the aggregation the optimizer already performed across LoRA pairs is the
statistic to plot -- ratio-of-medians is not median-of-ratios
(``optim_diagnostics.py:85``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lora_playground.plotting.labels import canonical_colors
from lora_playground.plotting.style import NOTEBOOK_RCPARAMS

BAND_ALPHA = 0.15
PANEL_SIZE = (6.5, 4.25)


@dataclass(frozen=True, slots=True)
class Trace:
    """One curve on a panel, named by the diagnostic field it draws.

    ``field`` is the stem the optimizer aggregated (``sigma_max_A``), not a
    full column name. ``ls`` distinguishes two fields sharing one panel; the
    panel's ``ylabel`` states that convention, which is why a linestyle needs no
    legend entry of its own.
    """

    field: str
    ls: str = "-"
    band: bool = True


@dataclass(frozen=True, slots=True)
class TracePanel:
    """One axes: its y label, its curves, and its reference decorations."""

    ylabel: str
    traces: tuple[Trace, ...]
    ref: float | None = None
    yscale: str | None = None
    ylim: tuple[float, float] | None = None


def _line_column(arrays, field: str):
    """The median column for ``field``, falling back to the bare field.

    The fallback lets a diagnostic logged as a single scalar per step -- rather
    than aggregated across LoRA pairs -- plot without a second panel type.
    """
    for name in (f"{field}_median", field):
        if name in arrays:
            return arrays[name]
    return None


def _draw(ax, arrays, trace: Trace, colour, x):
    line = _line_column(arrays, trace.field)
    if line is None:
        raise KeyError(
            f"trace field {trace.field!r} is not in this run's diagnostics: "
            f"neither {trace.field}_median nor {trace.field} was recorded. "
            f"Recorded columns: {sorted(arrays)}"
        )
    if trace.band:
        low = arrays.get(f"{trace.field}_min")
        high = arrays.get(f"{trace.field}_max")
        if low is not None and high is not None:
            ax.fill_between(x, low, high, color=colour, alpha=BAND_ALPHA,
                            linewidth=0)
    ax.plot(x, line, color=colour, ls=trace.ls)


def trace_figure(series, panels, *, suptitle, xlabel="Training step",
                 legend_title=None, step_field="step", ncols=2):
    """Draw ``panels`` for every run in ``series``, one colour per series.

    ``series`` maps a display label to one run's diagnostic arrays. Colour
    always encodes the series and never the field, so the panels stay readable
    against each other; a panel showing two fields separates them by linestyle
    and says so in its ``ylabel``.
    """
    series = dict(series)
    if not series:
        raise ValueError("trace_figure needs at least one series")
    colours = canonical_colors(series)
    ncols = min(ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)

    with plt.rc_context(NOTEBOOK_RCPARAMS):
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, constrained_layout=True,
            figsize=(PANEL_SIZE[0] * ncols, PANEL_SIZE[1] * nrows),
        )
        flat = list(axes.flat)
        for ax, panel in zip(flat, panels):
            for label, arrays in series.items():
                for trace in panel.traces:
                    _draw(ax, arrays, trace, colours[label],
                          arrays[step_field])
            if panel.ref is not None:
                ax.axhline(panel.ref, color="black", ls=":", lw=0.8, zorder=0)
            if panel.yscale is not None:
                ax.set_yscale(panel.yscale)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(panel.ylabel)
        for spare in flat[len(panels):]:
            spare.set_visible(False)
        fig.legend(
            handles=[Line2D([], [], color=colours[label], label=label)
                     for label in series],
            title=legend_title, loc="outside lower center",
            ncol=len(series), frameon=False,
        )
        fig.suptitle(suptitle)
    plt.show()
    return fig


def two_key_legend(ax, *, colors, color_title, styles, style_title, x=1.02):
    """Legend colour and linestyle as two keys, never their cross-product.

    ``colors`` maps a label to a colour and ``styles`` a label to a linestyle.
    The two boxes anchor to the top and bottom of the axes rather than to
    hand-picked fractions of its height, so neither can drift onto the other
    when an entry is added.
    """
    upper = ax.legend(
        handles=[Line2D([], [], color=colour, label=label)
                 for label, colour in colors.items()],
        title=color_title, loc="upper left", bbox_to_anchor=(x, 1.0),
        frameon=False,
    )
    ax.add_artist(upper)
    ax.legend(
        handles=[Line2D([], [], color="black", ls=ls, label=label)
                 for label, ls in styles.items()],
        title=style_title, loc="lower left", bbox_to_anchor=(x, 0.0),
        frameon=False,
    )


__all__ = ["BAND_ALPHA", "PANEL_SIZE", "Trace", "TracePanel", "trace_figure",
           "two_key_legend"]
