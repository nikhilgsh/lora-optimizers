"""Plotting package for the LoRA optimizer-comparison playground.

Submodule layout (one concern per file):
  - :mod:`.style`      style constants (fonts, line widths, baseline styling)
  - :mod:`.colors`     optimizer color/family registries, marker overrides,
                       ablation-axis style constants, overlay-palette guard
  - :mod:`.loading`    per-task log loading + caching
  - :mod:`.dedup`      series-identity, label-collision detection, baseline/
                       variant filtering
  - :mod:`.merge`      sweep merging with hidden-axis collision detection,
                       diverged-run filtering, ``RUNTIME_FIELDS``
  - :mod:`.overlays`   AdamW (and secondary) baseline overlays
  - :mod:`.panels`     single-axis renderers (η-vs-final, best-η curves,
                       leaderboard)
  - :mod:`.figures`    high-level 2-panel entry points
                       (``standard_sweep_figure``, ``two_panel_sweep_figure``)
  - :mod:`.ablations`  canned ablation figures used directly from notebooks

The public API surfaces every name a notebook / test / script needs. Internal
helpers are underscore-prefixed and live in their submodule.
"""
from __future__ import annotations

# Style constants
from .style import (
    AXIS_LABEL_FONTSIZE,
    BASELINE_COLOR,
    BASELINE_LS_CURVE,
    BASELINE_LS_HLINE,
    BASELINE_LW_CURVE,
    BASELINE_LW_HLINE,
    BASELINE_MARKER,
    BASELINE_ZORDER,
    CANONICAL_HORIZON,
    DEFAULT_FIGSIZE,
    DIVERGE_THRESHOLD,
    LEGEND_KW,
    LINE_WIDTH,
    MARKER_SIZE,
    PANEL_TITLE_FONTSIZE,
    REF_LINE_WIDTH,
    SUPTITLE_FONTSIZE,
    TICK_LABEL_FONTSIZE,
)

# Color registries + collision guard
from .colors import (
    ColorCollisionError,
    M_LINESTYLES,
    NS_AXIS_COLORS,
    OPTIM_COLORS,
    OPTIM_FAMILIES,
    OPTIM_MARKERS,
    PICARD_LINESTYLES,
    assert_palette_distinct_from_reserved,
    ssc_overlay_palette,
)

# Loading
from .loading import (
    has_runs,
    load_run,
    load_sweep,
    parse_cli_command,
    parse_flag,
)

# Series identity + dedup
from .dedup import (
    LabelCollisionError,
    assert_label_discriminates,
    detect_group_collisions,
    filter_baseline,
    filter_variants,
    series_id,
)

# Merge + divergence
from .merge import (
    RUNTIME_FIELDS,
    best_run,
    max_loss,
    merge_runs,
    report_diverged,
    split_diverged,
)

# Overlays
from .overlays import baseline_overlay

# Panel renderers
from .panels import (
    plot_best_eta_curves,
    plot_eta_vs_final,
    plot_leaderboard_by_rank,
)

# High-level figure entry points
from .figures import (
    standard_sweep_figure,
    two_panel_sweep_figure,
)

# Ablation helpers
from .ablations import (
    CHORD_TIGHT_CLEAN,
    compare_variants_figure,
    ns_iters_ssc_overlay_figure,
)
