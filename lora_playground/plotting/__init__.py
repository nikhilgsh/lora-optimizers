"""Plotting package for the LoRA optimizer-comparison playground.

Submodule layout (one concern per file, all generic):
  - :mod:`.style`     style constants (fonts, line widths, baseline styling)
  - :mod:`.colors`    optimizer color/family registries, marker overrides,
                      and the overlay-palette collision guard
  - :mod:`.loading`   per-task log loading + caching
  - :mod:`.dedup`     series-identity, label-collision detection, baseline/
                      variant filtering
  - :mod:`.merge`     sweep merging with hidden-axis collision detection,
                      diverged-run filtering, ``RUNTIME_FIELDS``
  - :mod:`.overlays`  AdamW (and secondary) baseline overlays
  - :mod:`.panels`    single-axis renderers + auto-ylim helpers
                      (η-vs-final, best-η curves, leaderboard)
  - :mod:`.figures`   high-level entry points: ``standard_sweep_figure``,
                      ``sweep_figure_with_auto_ylim``,
                      ``compare_variants_figure``

The public API surfaces every name a notebook / test / script needs.
Internal helpers are underscore-prefixed and live in their submodule.
Project-specific composition (slug→short-name dicts, ablation-axis colors,
buggy-commit lists, etc.) belongs in the notebook, not here.
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
    OPTIM_COLORS,
    OPTIM_FAMILIES,
    OPTIM_MARKERS,
    assert_palette_distinct_from_reserved,
    distinct_palette,
    overlay_palette,
)

# Loading
from .loading import (
    clear_run_caches,
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
    dedup_by_canonical,
    detect_group_collisions,
    filter_baseline,
    filter_variants,
    series_id,
)

# Canonical variant labels / colors / order — single source of truth
from .labels import (
    canonical_colors,
    canonical_key,
    canonical_label,
    order_labels,
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

# Panel renderers + auto-ylim helpers
from .panels import (
    auto_ylim_for_final_panel,
    auto_ylim_for_trajectory_panel,
    plot_best_eta_curves,
    plot_eta_vs_final,
    plot_leaderboard_by_rank,
)

# High-level figure entry points
from .figures import (
    compare_variants_figure,
    leaderboard_panel,
    standard_sweep_figure,
    sweep_figure_with_auto_ylim,
    two_panel_sweep_figure,
)
