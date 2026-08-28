"""Shared style constants — fonts, line widths, baseline styling, figure size.

Imported by every plotting submodule and by external scripts that need the
canonical figure dimensions or baseline-styling tuple. No code outside this
file should hard-code these values.
"""
from __future__ import annotations

from cycler import cycler as _cycler

from .colors import SERIES_PALETTE

# Diverge threshold (final eval_loss above this counts as a diverged run).
DIVERGE_THRESHOLD = 1.5

# Default canonical horizon when cfg["max_steps"] is absent (legacy unpacked_v0
# default; packed_v1 sweeps carry max_steps=4000 in cfg, which the inference
# logic prefers).
CANONICAL_HORIZON = 2000

# Legend placement: outside the right edge of the panel.
LEGEND_KW = dict(
    loc="center left", bbox_to_anchor=(1.02, 0.5),
    fontsize=14, frameon=True, borderaxespad=0.0,
    handlelength=2.4, handletextpad=0.7,
    labelspacing=0.55,
)

# Legend style for side-by-side 2-panel figures (compare_variants_figure): a
# SINGLE figure-level legend placed BELOW the whole figure (loc="outside lower
# center" + constrained_layout reserves the space) so a long multi-variant
# legend never covers the panels. Spans the full figure width (not one panel),
# so long entries fit across columns without colliding.
LEGEND_BELOW_KW = dict(
    ncol=3, frameon=False,
    handlelength=2.0, handletextpad=0.6,
    columnspacing=1.4, labelspacing=0.5,
)

# Title / axis-label sizes — bumped throughout for readability.
SUPTITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13

# Candidate-line styling.
MARKER_SIZE = 5.0

# Optimal-hyperparameter star, drawn over that point's series marker. It has to
# be ~3x the series marker to read as a highlight rather than a smudge on it --
# the ratio `_basin` in paper_figs.py uses (ms=4 series, ms=12 star).
STAR_MARKER_SIZE = 13.0
LINE_WIDTH = 1.7

# Reference / overlay styling for non-primary baselines (e.g. adam-lin-lora).
REF_LINE_WIDTH = 1.5

# Primary baseline (AdamW): solid black, circle markers, thicker line. The
# weight + black color + thicker stroke make it visually salient without the
# busy dashed-square convention.
BASELINE_COLOR = "black"
BASELINE_LW_HLINE = 1.5
BASELINE_LS_HLINE = (0, (1, 1.5))   # fine dotted hline — visually distinct from the curve
BASELINE_LW_CURVE = 3.0             # heavier than candidate LINE_WIDTH
BASELINE_LS_CURVE = "-"             # solid (was dashed) per user preference
BASELINE_MARKER = "o"               # circle, matches candidate default
BASELINE_ZORDER = 2                 # behind candidates so it never covers a crossing

# Default figure size. Wide enough that an outside legend with long entries
# (e.g. "adam-polar-product-lora (η=3e-04, final=0.7546)") doesn't squish the
# axes. Constrained layout reserves the right margin for the legend.
DEFAULT_FIGSIZE = (20, 6.0)


# Notebook figures are reviewed on screen rather than reduced into a manuscript
# column. Apply this once in a notebook setup cell; library imports must not mutate
# process-wide Matplotlib state implicitly.
#
# This is the ONLY place a panel's fonts, line weights, grid and legend are set.
# Call sites must not pass `fontsize=` / `lw=` / `ms=` or call `ax.grid()`: a
# constant passed at one call site is a constant the next panel does not get,
# which is how the same quantity ended up drawn at three weights across the
# notebook. Put it here and every panel inherits it.
NOTEBOOK_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": PANEL_TITLE_FONTSIZE,
    "legend.fontsize": 11,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    # Suptitle is the only title a comparison panel carries; the per-panel
    # titles were dropped because they restated the axis labels and were what
    # the off-axis triangle markers collided with.
    "figure.titlesize": SUPTITLE_FONTSIZE,
    "figure.titleweight": "bold",
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    # Makes `C0`..`C7` and any unstyled plot draw from the SAME palette the
    # library panels use. Without it the notebook's hand-written diagnostic
    # cells (which reach for 'C0'/'C3'/f'C{i}') drew in matplotlib's tab10
    # while every panel beside them drew in SERIES_PALETTE.
    "axes.prop_cycle": _cycler("color", list(SERIES_PALETTE)),
    # Round line ENDS. The joint style is left alone: matplotlib already
    # defaults `lines.solid_joinstyle` to "round", so setting it changed
    # nothing, and the faceting between eval samples is inherent to the
    # polyline (splining it would invent losses never measured).
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    # Grid is a background reference, so it sits under the data and stays
    # lighter than the axis frame.
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#c8ccd4",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    # Only the two spines that carry the tick labels; the top and right ones
    # close a box around the data without saying anything.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4f58",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#1a1d21",
    "text.color": "#1a1d21",
    "xtick.color": "#4a4f58",
    "ytick.color": "#4a4f58",
    "xtick.labelcolor": "#1a1d21",
    "ytick.labelcolor": "#1a1d21",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "figure.dpi": 110,
    "savefig.dpi": 200,
}


def apply_notebook_style() -> None:
    """Install the shared readable style for an interactive notebook session."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(NOTEBOOK_RCPARAMS)
