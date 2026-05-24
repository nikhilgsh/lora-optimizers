"""Shared style constants — fonts, line widths, baseline styling, figure size.

Imported by every plotting submodule and by external scripts that need the
canonical figure dimensions or baseline-styling tuple. No code outside this
file should hard-code these values.
"""
from __future__ import annotations

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

# Title / axis-label sizes — bumped throughout for readability.
SUPTITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13

# Candidate-line styling.
MARKER_SIZE = 9
LINE_WIDTH = 2.0

# Reference / overlay styling for non-primary baselines (e.g. adam-lin-lora).
REF_LINE_WIDTH = 1.5

# Primary baseline (AdamW): solid black, circle markers, thicker line. The
# weight + black color + thicker stroke make it visually salient without the
# busy dashed-square convention.
BASELINE_COLOR = "black"
BASELINE_LW_HLINE = 1.5
BASELINE_LS_HLINE = (0, (1, 1.5))   # fine dotted hline — visually distinct from the curve
BASELINE_LW_CURVE = 3.0             # heavier than candidate LINE_WIDTH (2.0)
BASELINE_LS_CURVE = "-"             # solid (was dashed) per user preference
BASELINE_MARKER = "o"               # circle, matches candidate default
BASELINE_ZORDER = 2                 # behind candidates so it never covers a crossing

# Default figure size. Wide enough that an outside legend with long entries
# (e.g. "adam-polar-product-lora (η=3e-04, final=0.7546)") doesn't squish the
# axes. Constrained layout reserves the right margin for the legend.
DEFAULT_FIGSIZE = (20, 6.0)
