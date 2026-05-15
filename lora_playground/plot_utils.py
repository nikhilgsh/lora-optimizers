"""Shared utilities for the sweep-analysis notebook.

Loads sweep results from per-task .out logs, filters diverged runs, and
draws the standardized 2-panel figure (η vs final loss + best-η training
curves) that all sections of the notebook share.

Design: every section of the notebook supplies a sequence of (cfg, evs) runs,
plus a callable `group_key(cfg) → str` that maps a run to a color/legend group.
Everything else (filtering, axes, legend placement, training-curve picking,
AdamW baseline overlay, layout, fonts) is handled here.
The intended call site is `standard_sweep_figure(runs, group_key_fn,
color_map, suptitle=..., reference_runs=...)` — every figure produced this
way is uniform by construction.
"""
from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIVERGE_THRESHOLD = 1.5

# ─── per-section configs (kept here so cells are one-line library calls) ──────

# Optimizer color palette shared across the all-optimizer η-sweep and the
# cross-investigation sections. NOTE: pure "black" is reserved for the AdamW
# baseline overlay; no candidate group should use it.
OPTIM_COLORS = {
    # 22 distinct colors — verified no-collision (see test_plot_utils.py).
    # Grouped by family for readability; assignment is unique across the map.
    "adamw":                       "#000000",  # baseline — pure black overlay
    "adafactor":                   "#555555",  # grey baseline — Adafactor (rank-1 v factorization)
    # no-Adam family
    "scaled-lora":                 "#ff7f0e",
    "lin-lora":                    "#2ca02c",
    "muon-lora":                   "#e377c2",
    "polar-product-lora":          "#c49c94",
    # pre-Adam preconditioning (geometry → Adam)
    "adam-scaled-lora":            "#d62728",
    "adam-lin-lora":               "#9467bd",
    "adam-lin-core-lora":          "#b48ad6",  # cross-check: core-space Adam in lin-LoRA solver
    # post-Adam preconditioning (Adam → geometry, falsified family)
    "adam-lin-lora-post":          "#aec7e8",
    "adam-scaled-lora-post":       "#ffbb78",
    "adam-lin-lora-matrix":        "#98df8a",
    "adam-scaled-lora-matrix":     "#c5b0d5",
    # spectral / polar (the headline family)
    "adam-muon-lora":              "#3cb44b",   # vivid green, distinct from lin-lora's tab green
    "muon-adam-lora":              "#dbdb8d",
    "adam-polar-product-lora":     "#8c564b",
    "adam-polar-product-lora-coupled": "#5d342c",   # darker brown — coupled-pair variant of adam-polar
    "adam-polar-product-lora-coupled-endrms": "#a04a3c",  # warm brown — coupled w/ end-of-loop RMS-align
    "adam-polar-product-lora-coupled-exact-chord": "#7d4a2e",  # ochre — coupled w/ exact ΔW chord (not J-tangent)
    "adam-polar-product-lora-coupled-spectral-chord": "#c75c2e",  # burnt orange — coupled w/ spectral chord trust region
    "adam-polar-product-lora-coupled-spectral-chord-tight": "#7a3520",  # darker burnt orange — tight chord (exact root)
    "adam-polar-product-lora-coupled-spectral-chord-tight-exact": "#a8453c",  # red-orange — tight chord + exact-chord direction iter
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening": "#8e6b3c",  # tan — no-whitening ablation
    "adam-polar-product-lora-coupled-spectral-chord-tight-outer": "#c95a35",  # bright orange — outer-rescale (Picard insight)
    "adam-polar-product-lora-coupled-spectral-chord-direction": "#a02a0d",  # red-orange — variant 1 (direction-aware ρ)
    "adam-soap-polar-product-lora": "#c97a3a",  # amber — SOAP eigenbasis Adam before polar
    "adafactor-polar-product-lora": "#d4a04c",  # gold — Adafactor rank-1 v before polar
    "sign-momentum-polar-product-lora": "#7c3aed",  # violet — LION-style sign(m) before polar
    "adam-clip-product-lora":                 "#1f6f8c",  # blue — clip + RMS-align (no gauge), k=1
    "adam-clip-product-lora-coupled":         "#0f4f6f",  # dark blue — clip + RMS-align (no gauge), k=2
    "adam-clip-product-lora-coupled-endrms":  "#3a8fb5",  # light blue — clip + RMS-align k=2 endrms
    "adam-polar-product-lora-gauge":          "#7e3a85",  # purple — Sylvester min-Frob gauge lift, no postscale
    "adam-polar-product-lora-gauge-coupled":  "#5a2d63",  # dark purple — gauge + picard cross-coupling
    "adam-polar-product-lora-clip-gauge":     "#3a857e",  # teal — clip + gauge lift (R-equal τ rule)
    "adam-polar-product-lora-clip-gauge-coupled": "#1e6e63",  # dark teal — clip+gauge with picard cross-coupling
    "polar-coupled-core-lora":     "#7a3a2c",   # variant 1: projected quotient polar, raw factor grads
    "polar-coupled-core-imbalance-scalar-lora":  "#a05030",  # + scalar imbalance-preserving gauge S=sI (recommended primary)
    "polar-coupled-core-balanced-scalar-lora":   "#c87858",  # + scalar balanced gauge (drives imbalance → 0)
    "polar-coupled-core-imbalance-lora":         "#b86a48",  # + full r×r imbalance-preserving gauge
    "polar-coupled-core-imbalance-restore-lora": "#d8a070",  # + iLoRA imbalance-restoring gauge (aggressive; experimental)
    "polar-coupled-core-state-rebalanced-lora":  "#5a2018",  # + post-step state-gauge rebalance (recommended)
    "polar-coupled-core-sign-lora":              "#9a4030",  # + per-step elementwise core sign normalization (rung 5-lite)
    "polar-coupled-core-sign-rebalanced-lora":   "#bb6048",  # + sign + state rebalance (compound)
    "polar-coupled-core-factor-adam-lora":       "#c08040",  # rung 6: factor-Adam + projected-quotient-polar (Picard analog)
    "polar-coupled-core-factor-adam-rebalanced-lora": "#e0a060",  # + state rebalance compound
    "muon-coupled-core-lora":      "#3a5a8a",   # variant 2: + transported core-space momentum
    "muon-coupled-core-imbalance-scalar-lora":   "#4a6aa8",  # variant 2 + scalar imbalance gauge
    "muon-coupled-core-balanced-scalar-lora":    "#7090c8",  # variant 2 + scalar balanced gauge
    "muon-coupled-core-imbalance-lora":          "#5a8ac8",  # variant 2 + full imbalance gauge
    "muon-coupled-core-state-rebalanced-lora":   "#1a3a6a",  # variant 2 + state-gauge rebalance
    "muon-coupled-core-sign-lora":               "#5a7ab8",  # variant 2 + sign norm
    "muon-coupled-core-sign-rebalanced-lora":    "#7a9ae0",  # variant 2 + sign + state rebalance (full stack)
    "adamuon-polar-product-lora":  "#1f77b4",
    "adamuon-lora":                "#ff9896",
    # gauge-invariant variants
    "product-muon-lora":           "#0d3d66",
    "adam-product-muon-lora":      "#9edae5",
    # K-FAC / per-coord / dropped families (plotted only in legacy cells)
    "diag-scaled-lora":            "#17becf",
    "kron-grad-lora":              "#bcbd22",
    "psi-lora":                    "#7f7f7f",
    "galore-adamw":                "#a55194",
    # orthogonal-core LoRA (UCV^T parameterization)
    "adam-ucv-core-lora":          "#2b8c4d",
}


# Family membership for per-cell comparisons. Each entry is "the set of
# optimizers a particular notebook cell wants to compare." Cells reference
# OPTIM_FAMILIES["<family>"] instead of inlining a literal set, so adding a
# new optimizer is a one-file change here (color + family) rather than
# hunting hard-coded sets across the notebook.
#
# When adding a new optimizer:
#   1. add an entry to OPTIM_COLORS above.
#   2. add it to whichever family/families it belongs in below.
# The _validate_family_membership() check at module load surfaces any
# OPTIM_COLORS entry that's missing from every family as a warning.
OPTIM_FAMILIES = {
    # Headline polar/muon spectral family (post-Adam direction-shaping).
    "headline_polar": {
        "adamw",
        "adafactor",
        "adam-polar-product-lora",
        "adam-polar-product-lora-coupled",
        "adam-polar-product-lora-coupled-endrms",
        "adam-polar-product-lora-coupled-exact-chord",
        "adam-polar-product-lora-coupled-spectral-chord",
        "adam-polar-product-lora-coupled-spectral-chord-tight",
        "adam-polar-product-lora-coupled-spectral-chord-tight-exact",
        "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
        "adam-polar-product-lora-coupled-spectral-chord-tight-outer",
        "adam-polar-product-lora-coupled-spectral-chord-direction",
        "adam-soap-polar-product-lora",
        "adafactor-polar-product-lora",
        "sign-momentum-polar-product-lora",
        "adam-clip-product-lora",
        "adam-clip-product-lora-coupled",
        "adam-clip-product-lora-coupled-endrms",
        "adam-polar-product-lora-gauge",
        "adam-polar-product-lora-gauge-coupled",
        "adam-polar-product-lora-clip-gauge",
        "adam-polar-product-lora-clip-gauge-coupled",
        "polar-coupled-core-lora",
        "polar-coupled-core-imbalance-scalar-lora",
        "polar-coupled-core-balanced-scalar-lora",
        "polar-coupled-core-imbalance-lora",
        "polar-coupled-core-imbalance-restore-lora",
        "polar-coupled-core-state-rebalanced-lora",
        "polar-coupled-core-sign-lora",
        "polar-coupled-core-sign-rebalanced-lora",
        "polar-coupled-core-factor-adam-lora",
        "polar-coupled-core-factor-adam-rebalanced-lora",
        "muon-coupled-core-lora",
        "muon-coupled-core-imbalance-scalar-lora",
        "muon-coupled-core-balanced-scalar-lora",
        "muon-coupled-core-imbalance-lora",
        "muon-coupled-core-state-rebalanced-lora",
        "muon-coupled-core-sign-lora",
        "muon-coupled-core-sign-rebalanced-lora",
        "adamuon-lora",
        "adamuon-polar-product-lora",
        "adam-muon-lora",
    },
    # Pre-Adam linear preconditioning (geometry → Adam, H1 found ε-perturbed).
    "pre_adam_lin_scaled": {
        "adamw",
        "adam-lin-lora",
        "adam-lin-core-lora",
        "adam-scaled-lora",
    },
    # No-Adam family (raw momentum / NS / closed-form, no per-coord v̂).
    "no_adam": {
        "adamw",
        "muon-lora",
        "polar-product-lora",
        "lin-lora",
        "scaled-lora",
    },
    # Post-Adam preconditioning + matrix-Adam (H4 falsified family).
    "post_adam_h4": {
        "adamw",
        "adam-lin-lora-post",
        "adam-scaled-lora-post",
        "adam-lin-lora-matrix",
        "adam-scaled-lora-matrix",
        "muon-adam-lora",
    },
    # Orthogonal-core LoRA (UCV^T parameterization, separate from BA).
    "ucv_family": {
        "adamw",
        "adam-polar-product-lora",
        "adam-ucv-core-lora",
    },
    # Bucket-3: theoretically promising, empirically weak.
    "bucket3_weak": {
        "adamw",
        "product-muon-lora",
        "adam-product-muon-lora",
        "diag-scaled-lora",
        "kron-grad-lora",
        "psi-lora",
        "galore-adamw",
    },
}


def _validate_family_membership() -> None:
    """Warn at module load when an OPTIM_COLORS entry is in no OPTIM_FAMILIES
    set. Soft check — some optimizers may be intentionally excluded from every
    cell-level comparison; the warning surfaces the more likely "you added a
    color but forgot a family" failure that silently empties plots."""
    in_some_family: set[str] = set().union(*OPTIM_FAMILIES.values())
    orphans = sorted(set(OPTIM_COLORS) - in_some_family)
    if orphans:
        import warnings
        warnings.warn(
            f"OPTIM_COLORS entries with no OPTIM_FAMILIES membership "
            f"(plots filtering by family will silently drop them): {orphans}. "
            f"Add to a family in plot_utils.py or accept the exclusion.",
            stacklevel=2,
        )


_validate_family_membership()


# Linestyles for LoRA+ multiplier disambiguation. Convention: same color per
# base optimizer across all m, with linestyle carrying the m signal. Used by
# muon-variants and leaderboard cells via M_LINESTYLES (looked up by
# extracting m from the group label "(m=N)" suffix).
M_LINESTYLES = {
    1: "-",         # solid
    4: (0, (5, 2)),    # long-dash
    16: (0, (1, 2)),   # dotted (gappy)
    32: (0, (3, 2, 1, 2)),  # dash-dot
}

# Marker styles to disambiguate near-color pairs. Default is "o" (circle) for
# any optimizer not listed; overrides pick distinct shapes where colors are
# close. Stable to grayscale, colorblind-friendly. Used by plot_leaderboard_by_rank
# and standard_sweep_figure when a marker_map is passed.
OPTIM_MARKERS = {
    # Greens cluster (pure tab green vs vivid green vs olive vs light)
    "lin-lora":                    "o",
    "adam-muon-lora":              "^",   # triangle-up to distinguish from lin-lora
    "muon-adam-lora":              "v",   # triangle-down (yellow-green olive shade)
    "adam-lin-lora-matrix":        "P",   # plus (light green)
    # Reds/pinks cluster
    "adam-scaled-lora":            "o",
    "adamuon-lora":                "X",   # x-filled (light red)
    "muon-lora":                   "*",   # star (medium pink)
    # Blues/cyans cluster
    "adamuon-polar-product-lora":  "o",
    "diag-scaled-lora":            "s",   # square (cyan)
    "adam-product-muon-lora":      "D",   # diamond (light cyan)
    "adam-lin-lora-post":          "h",   # hexagon (light blue)
    # Purples cluster
    "adam-lin-lora":               "o",
    "adam-scaled-lora-matrix":     "p",   # pentagon (light purple)
    "galore-adamw":                "<",   # triangle-left (medium purple)
    # Browns/oranges cluster
    "adam-polar-product-lora":     "o",
    "adam-polar-product-lora-coupled": "P",   # plus (darker brown)
    "adam-polar-product-lora-coupled-endrms": "X",  # x-filled (warm brown)
    "adam-polar-product-lora-coupled-exact-chord": "h",  # hexagon (ochre) — exact-chord variant
    "adam-polar-product-lora-coupled-spectral-chord": "8",  # octagon (burnt orange) — spectral-chord magnitude rule
    "adam-polar-product-lora-coupled-spectral-chord-tight": "H",  # uppercase hexagon — tight chord-spectral (exact root)
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening": "x",  # x — no-whitening ablation
    "adam-polar-product-lora-coupled-spectral-chord-tight-outer": "P",  # plus — outer rescale (Picard insight)
    "adam-polar-product-lora-coupled-spectral-chord-direction": "*",  # star — direction-aware (variant 1)
    "adam-soap-polar-product-lora": "*",   # star (amber) — SOAP eigenbasis preconditioning
    "adam-polar-product-lora-gauge":          "D",  # diamond (purple) — Sylvester gauge variant
    "adam-polar-product-lora-gauge-coupled":  "d",  # thin diamond (dark purple) — gauge + picard
    "adam-polar-product-lora-clip-gauge":     "v",  # down-triangle (teal) — clip + gauge variant
    "adam-polar-product-lora-clip-gauge-coupled": "^",  # up-triangle (dark teal) — clip+gauge + picard
    "polar-product-lora":          "d",   # thin diamond (light brown)
    "scaled-lora":                 ">",   # triangle-right (orange)
    "adam-scaled-lora-post":       "8",   # octagon (light orange)
    # Olive/yellow + dark
    "kron-grad-lora":              "o",
    "product-muon-lora":           "o",
    "psi-lora":                    "o",
    "adamw":                       "s",   # baseline override often passed separately
}


# ─── shared style constants ───────────────────────────────────────────────────

# Legend placement: outside the right edge of the panel.
LEGEND_KW_BASE = dict(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=14, frameon=True, borderaxespad=0.0,
                      handlelength=2.4, handletextpad=0.7,
                      labelspacing=0.55)

# Title / axis-label sizes — bumped throughout for readability.
SUPTITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13

# Backwards-compat alias for any external code that still imports LEGEND_KW.
LEGEND_KW = LEGEND_KW_BASE

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


# ─── data loading ─────────────────────────────────────────────────────────────

def parse_flag(command: str, flag: str) -> str | None:
    """Extract --flag VALUE from a command string."""
    parts = shlex.split(command)
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _coerce_value(v: str):
    """String → int/float/bool when possible; otherwise leave as str.
    Used by parse_cli_command and CLI backfill so analysis code can filter
    on numeric values without per-flag handling."""
    if v in ("True", "true"): return True
    if v in ("False", "false"): return False
    if v in ("None", "none", "null"): return None
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v


def parse_cli_command(command: str) -> dict:
    """Parse a logged ``--flag value`` (and ``--bool-flag``) command into a dict.

    Generic backfill source for the loader: turns the full launcher command
    line into ``{flag_name: typed_value}``. Boolean flags (``store_true`` /
    ``store_false`` argparse actions) become ``True`` when present. Values are
    coerced via :func:`_coerce_value` so numeric flags can be filtered on.

    This is the *systemic* alternative to per-flag setdefault clauses in
    ``_read_run``: every new CLI flag becomes a first-class cfg field
    automatically, no plot_utils patch needed.
    """
    parts = shlex.split(command)
    out: dict = {}
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("--"):
            key = tok[2:]
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt is None or nxt.startswith("--"):
                # Bare boolean flag (store_true) — no value follows
                out[key] = True
                i += 1
            else:
                out[key] = _coerce_value(nxt)
                i += 2
        else:
            i += 1
    return out


class LabelCollisionError(ValueError):
    """Raised when distinct series_ids share one display label.

    The plotting layer averages within a display-label bucket as if all
    runs were seeds. When two distinct series_ids land in one bucket, the
    average silently mixes algorithms — exactly the failure mode this
    contract exists to forbid. The error message names the differing
    cfg field(s); the fix is to either extend the group_key/display_label
    function to discriminate, or (rarely) add the field to
    `manifest.SERIES_AXIS_FIELDS` if it's truly a per-series axis.
    """


def _hashable(v):
    """Recursive conversion to a hashable form (mirrors loader._hashable).
    Dicts → frozenset of items; lists → tuples; everything else → as-is.
    """
    if isinstance(v, dict):
        return frozenset((k, _hashable(vv)) for k, vv in v.items())
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def series_id(cfg: dict, *, axis_fields: frozenset[str] | None = None) -> frozenset:
    """Mechanical series identity = cfg minus SERIES_AXIS_FIELDS, with
    ``None``-valued fields treated as absent.

    Two cfgs with the same series_id are seeds / lr-grid points / horizon
    extensions of the same algorithm at the same model config and may be
    averaged together. Distinct series_ids cannot be averaged regardless
    of what a display-label function returns — see
    `assert_label_discriminates`.

    ``None`` is treated identically to "field not present": an older run
    whose schema didn't include flag X is informationally equivalent to a
    newer run with ``X=None`` (both fall through to argparse default at
    runtime). Treating them as distinct would split every old-vs-new
    cfg pair purely on schema growth. Explicit non-None values (``False``,
    ``0``, ``0.0``, ``""``) ARE series-defining.
    """
    if axis_fields is None:
        from lora_playground.manifest import SERIES_AXIS_FIELDS
        axis_fields = SERIES_AXIS_FIELDS
    # Also exclude:
    #   - underscore-prefixed keys (loader enrichment namespaces:
    #     `_derived`, `_cli_args`, `_optim_steps`, etc.) — these mirror
    #     source-of-truth scalar fields and would double-count.
    #   - dict-valued composites like `optimizer_config` (loader backfill
    #     that mirrors a subset of scalar fields; older runs without it
    #     get backfilled and may not round-trip exactly).
    # Scalar top-level cfg fields ARE the source of truth.
    return frozenset(
        (k, _hashable(v)) for k, v in cfg.items()
        if k not in axis_fields
        and not k.startswith("_")
        and not isinstance(v, dict)
        and v is not None
    )


def _label_collision_report(runs, group_key_fn, *,
                            bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                            axis_fields: frozenset[str] | None = None,
                            ) -> list[dict]:
    """For every (label, *bucket_keys) bucket, return entries where the
    runs split into >1 distinct series_id. Each entry names the bucket,
    the differing cfg field(s), and the per-run values of those fields.
    """
    if axis_fields is None:
        from lora_playground.manifest import SERIES_AXIS_FIELDS
        axis_fields = SERIES_AXIS_FIELDS
    from collections import defaultdict
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for cfg, _evs in runs:
        gk = group_key_fn(cfg)
        bk = tuple(cfg.get(k) for k in bucket_keys)
        buckets[(gk, bk)].append(cfg)

    reports: list[dict] = []
    for (label, bucket_vals), cfgs in buckets.items():
        if len(cfgs) < 2:
            continue
        ids = {series_id(c, axis_fields=axis_fields) for c in cfgs}
        if len(ids) == 1:
            continue
        # Identify the specific cfg fields that disagree across these runs.
        # Mirror series_id's exclusion rule so the reported diff is the
        # actual source-of-truth difference (not a derived/enriched field).
        all_keys: set[str] = set()
        for c in cfgs:
            all_keys.update(k for k, v in c.items()
                            if k not in axis_fields
                            and not k.startswith("_")
                            and not isinstance(v, dict))
        differing: dict[str, set] = {}
        for k in all_keys:
            vals = set()
            for c in cfgs:
                v = c.get(k)
                if v is None:
                    vals.add(None)
                    continue
                try:
                    vals.add(_hashable(v))
                except TypeError:
                    vals.add(repr(v))
            if len(vals) > 1:
                differing[k] = vals
        reports.append({
            "label": label,
            "bucket": dict(zip(bucket_keys, bucket_vals)),
            "n_runs": len(cfgs),
            "n_distinct_series": len(ids),
            "differing_fields": differing,
        })
    return reports


def assert_label_discriminates(runs, group_key_fn, *,
                               bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                               axis_fields: frozenset[str] | None = None,
                               ) -> None:
    """Hard-fail if any (label, *bucket_keys) bucket contains >1 series_id.

    Wire into every plot entry point that averages within a label bucket
    (`standard_sweep_figure`, `plot_eta_vs_final`, `_multi_seed_curve`).
    The failure message names the bucket and the cfg fields that disagree
    — the fix is a one-line extension of the display-label function (or,
    rarely, an addition to `manifest.SERIES_AXIS_FIELDS`).
    """
    reports = _label_collision_report(runs, group_key_fn,
                                      bucket_keys=bucket_keys,
                                      axis_fields=axis_fields)
    if not reports:
        return
    lines = []
    for r in reports:
        diffs = ", ".join(
            f"{k}={sorted(v, key=repr)}" for k, v in r["differing_fields"].items()
        )
        lines.append(
            f"  label={r['label']!r} bucket={r['bucket']} "
            f"n_runs={r['n_runs']} → {r['n_distinct_series']} distinct series_ids; "
            f"differing fields: {diffs}"
        )
    raise LabelCollisionError(
        "display label does not discriminate distinct series_ids; "
        "{} collision(s):\n{}\n"
        "Fix: extend the group_key / display-label function to discriminate "
        "on one of the listed fields, OR add the field to "
        "`lora_playground.manifest.SERIES_AXIS_FIELDS` if it is a true "
        "per-series axis (seed-like)."
        .format(len(reports), "\n".join(lines))
    )


def _baseline_values(runs, *, allowed_to_vary: set) -> dict:
    """Build {field: baseline_value} where baseline_value is the train.py
    argparse default. Only include fields that (a) vary across ``runs``
    AND (b) have their argparse default among the observed values.

    Skips fields that are constant across the run set (those are
    project-wide baselines, not experimental axes), fields without an
    argparse default (e.g. derived), and ones in ``allowed_to_vary``.

    Why argparse default over modal:
      - modal is fragile when an investigation's variant data takes >50%
        of the cell (the variant becomes "modal" → baseline runs get
        dropped instead of variants).
      - argparse default is stable: it's the canonical "ran with the
        default knob" value. Any cell asking "show me the default-knob
        runs" gets the same answer regardless of how many variant runs
        happen to be loaded alongside.
    """
    from collections import defaultdict
    from lora_playground.loader import _argparse_defaults
    defaults = _argparse_defaults()
    observed: dict = defaultdict(set)
    for cfg, _ in runs:
        for k, v in cfg.items():
            if k in allowed_to_vary or k.startswith("_") or isinstance(v, dict):
                continue
            observed[k].add(_hashable(v))
    baseline: dict = {}
    for k, vals in observed.items():
        if len(vals) <= 1:
            continue
        if k not in defaults:
            continue
        default_val = _hashable(defaults[k])
        if default_val in vals:
            baseline[k] = default_val
    return baseline


def filter_baseline(runs, *, varying: tuple[str, ...] = ()):
    """Keep runs that match the argparse-default value on every scalar
    cfg field that VARIES WITHIN their ``varying``-tuple group (and has
    its default observed in the data), excluding fields in ``varying``
    and in ``manifest.SERIES_AXIS_FIELDS``.

    ``varying`` is the analysis cell's INTENDED axis of variation — fields
    whose differences the cell's display_label discriminates on (e.g.
    ``('optimizer', 'effective_picard_iters')`` for an optimizer-x-k
    leaderboard). Within each (varying-tuple) group, other fields that
    take >1 value are research variants from a different investigation
    and get held to the argparse default.

    Per-group (vs global) is essential for co-varying fields: e.g.
    ``polar_norm_dir`` is constant within each optimizer's runs but
    differs across optimizers. A global pass would hold it to the
    argparse default (one optimizer wins, others are excluded). The
    per-group pass sees no variation within an optimizer and doesn't
    filter on it.

    Fields constant within a group don't filter anything in that group
    (single value = no choice to make).

    Pairs with :func:`assert_label_discriminates`: that catches
    *under-discrimination* (label collapses distinct series_ids); this
    one catches *over-inclusion* (run set contains variants from a
    different investigation). Together they bracket the cell's scope.

    Mirror for variant-focused cells: :func:`filter_variants`.

    Future-proof: a new train.py flag added later that some runs turn on
    automatically filters to its argparse default. Cells that should
    include the variant list the field in ``varying``; cells that
    shouldn't include it stay clean with no edit.
    """
    from collections import defaultdict
    from lora_playground.manifest import SERIES_AXIS_FIELDS
    allowed = set(varying) | SERIES_AXIS_FIELDS
    groups: dict = defaultdict(list)
    for cfg, evs in runs:
        gkey = tuple(_hashable(cfg.get(v)) for v in varying)
        groups[gkey].append((cfg, evs))
    kept = []
    for grp in groups.values():
        baseline = _baseline_values(grp, allowed_to_vary=allowed)
        for cfg, evs in grp:
            if all(_hashable(cfg.get(k)) == expected
                   for k, expected in baseline.items()):
                kept.append((cfg, evs))
    return kept


def filter_variants(runs, *, on: tuple[str, ...]):
    """Mirror of :func:`filter_baseline`. Keep runs where AT LEAST ONE of
    the fields in ``on`` is off its argparse default. Use in dedicated
    variant-investigation cells (ε_rel sweep, Init[AB] focused cell) so
    the run set is exactly "runs that explored this variant axis."
    """
    from lora_playground.loader import _argparse_defaults
    defaults = _argparse_defaults()
    kept = []
    for cfg, evs in runs:
        if any(k in defaults and _hashable(cfg.get(k)) != _hashable(defaults[k])
               for k in on):
            kept.append((cfg, evs))
    return kept


def detect_group_collisions(runs, group_key_fn, *,
                            bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                            axis_fields: frozenset[str] | None = None,
                            ) -> list[dict]:
    """Non-raising sibling of `assert_label_discriminates`. Returns the
    same per-bucket collision reports for audit-cell printouts. Most
    callers should prefer the raising version — silent reports are how
    label/series drift accumulates.
    """
    return _label_collision_report(runs, group_key_fn,
                                   bucket_keys=bucket_keys,
                                   axis_fields=axis_fields)


_LOAD_RUN_CACHE: dict[str, tuple[tuple[int, int], dict | None, list[dict]]] = {}


def load_run(log_path: Path) -> tuple[dict | None, list[dict]]:
    """Parse a single task .out file → (config dict, list of eval dicts).

    optim_step diagnostic events (emitted by polar/lin/scaled-LoRA optimizers
    when --log_basic_diagnostics is on) are attached to the config dict as
    ``cfg["_optim_steps"]``. Both ``loader.RUNTIME_FIELDS`` and
    ``_HIDDEN_AXIS_RUNTIME_FIELDS`` (re-exported from this module) list it,
    so dedup ignores it.

    Result is cached by (path, mtime, size); an in-flight file's growing size
    invalidates the entry so re-parses pick up new events. Returned cfg is
    shallow-copied per call because merge_runs mutates ``cfg["log_group"]``
    and downstream enrichment writes ``cfg["_derived"]``; the cached cfg
    must stay clean across callers.
    """
    path_str = str(log_path)
    try:
        st = log_path.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    cached = _LOAD_RUN_CACHE.get(path_str) if sig is not None else None
    if cached is not None and cached[0] == sig:
        cfg, evs = cached[1], cached[2]
        return (None if cfg is None else dict(cfg)), evs

    config, evals, optim_steps = None, [], []
    init_override_mode = None
    for line in Path(log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "config":
            config = obj
        elif obj.get("event") == "eval":
            evals.append(obj)
        elif obj.get("event") == "optim_step":
            optim_steps.append(obj)
        elif obj.get("event") == "lora_init_override":
            init_override_mode = obj.get("mode")
    if config is not None and evals:
        # Loader-assigned: the cfg event itself does not carry a run_id;
        # we synthesize one from the disk path here so downstream RUN_EXCLUSIONS
        # and DIRTY_ATTESTATIONS lookups have a stable per-run key. Note this
        # IS stored on cfg (underscore-prefixed to match the convention for
        # loader-injected fields); _enrich_cfg later promotes a public
        # `run_id` tuple alongside `log_group`.
        config["_log_filename"] = log_path.name
        config.setdefault("lr", evals[0]["lr"])
        cmd = config.get("command", "")
        lp = parse_flag(cmd, "--lora_plus_multiplier")
        config.setdefault("lora_plus_multiplier", float(lp) if lp else 1.0)
        # CLI-only fields commonly varied across runs; surfaced as first-class
        # cfg fields so load_runs(where=...) and key_axes can filter on them.
        rk = parse_flag(cmd, "--precond_refresh_every")
        config.setdefault("precond_refresh_every", int(rk) if rk else 1)
        config.setdefault("precond_method", parse_flag(cmd, "--precond_method"))
        # lora_init_b: explicit cfg field (new runs as of train.py change
        # 2026-05-13), CLI flag fallback, then lora_init_override event
        # (which only fires for non-zero modes), default "zero".
        if config.get("lora_init_b") is None:
            cli_init = parse_flag(cmd, "--lora_init_b")
            config["lora_init_b"] = cli_init or init_override_mode or "zero"

        # Generic CLI backfill — every flag in the launched command line
        # becomes a first-class cfg field via setdefault. Source order:
        #   1. explicit cfg event keys (newest, typed correctly by train.py)
        #   2. `_cli_args` blanket dump in the config event (typed)
        #   3. parsed `command` string (string-coerced fallback)
        # Eliminates per-flag patches whenever a new CLI flag is added.
        cli_blob = config.get("_cli_args")
        if isinstance(cli_blob, dict):
            for k, v in cli_blob.items():
                config.setdefault(k, v)
        if cmd:
            for k, v in parse_cli_command(cmd).items():
                config.setdefault(k, v)
        config["_optim_steps"] = optim_steps
    if sig is not None:
        _LOAD_RUN_CACHE[path_str] = (sig, config, evals)
    return (None if config is None else dict(config)), evals


_LOAD_SWEEP_CACHE: dict[tuple[str, str], tuple[tuple, list[tuple[dict, list[dict]]]]] = {}


_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")


def _task_files(log_dir: Path) -> dict[str, list[Path]]:
    """Group `log_NN.out` and any `log_NN.out.resume_K` siblings by task
    index `NN`. Returned dict is `{task_idx: [files...]}` with files in
    deterministic order (resume_K-suffixed last, sorted by K)."""
    bucket: dict[str, list[Path]] = {}
    for p in log_dir.iterdir():
        m = _TASK_FILE_RE.match(p.name)
        if not m:
            continue
        bucket.setdefault(m.group(1), []).append(p)
    for idx, files in bucket.items():
        # Original `.out` first, then `.resume_K` by K ascending — so the
        # later-run segments are the LAST to feed events through dedup,
        # which means under any "first occurrence wins" rule they get
        # preference for overlapping steps (segments are step-disjoint
        # by design, but tolerate overlap).
        def _sort_key(p: Path):
            if p.name.endswith(".out"):
                return (0, 0)
            return (1, int(p.name.rsplit(".resume_", 1)[1]))
        files.sort(key=_sort_key)
    return bucket


def _merge_segments(segments: list[tuple[dict | None, list[dict]]]) -> tuple[dict | None, list[dict]]:
    """Merge per-file (cfg, evs) into a single (cfg, evs) for one task.

    cfg: take the FIRST non-None cfg (original run's config event is the
      canonical one; resumes re-emit the same config). Later resumes that
      changed `command` would already have different deny-list keys and
      be filtered by the loader's exclusion layer if intentional.
    evs: union sorted by step, deduping by step (latest segment wins).
    """
    cfg = next((c for c, _ in segments if c is not None), None)
    by_step: dict[int, dict] = {}
    for _, evs in segments:
        for ev in evs:
            by_step[int(ev["step"])] = ev
    merged = [by_step[s] for s in sorted(by_step)]
    return cfg, merged


def load_sweep(group: str, logs_root: str = "../logs") -> list[tuple[dict, list[dict]]]:
    """Load all runs for a sweep group. Returns list of (cfg, evs).

    A "run" is one disBatch task index. Per task, events are merged across
    `log_NN.out` and any sibling `log_NN.out.resume_K` files written by
    submit.sh's pre-submit log rotation when a wall-killed run is resubmitted
    with checkpoint resume.

    Cached by the per-file (mtime, size) signature of the group's log files;
    in-flight files invalidate naturally as they grow. Returned cfgs are
    shallow-copied per call so downstream mutation (``merge_runs`` writing
    ``log_group``, ``_enrich_cfg`` writing ``_derived``) doesn't poison
    cached entries.
    """
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    if not log_dir.exists():
        return []
    tasks = _task_files(log_dir)
    # Cache signature: every file's (name, mtime, size). The resume_K
    # suffixes get rotated atomically by submit.sh, which is one renames-event
    # — invalidates the cache by changed name.
    all_files = sorted(
        (f for files in tasks.values() for f in files), key=lambda p: p.name
    )
    sig = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in all_files)
    cache_key = (logs_root, group)
    cached = _LOAD_SWEEP_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return [(dict(cfg), evs) for cfg, evs in cached[1]]

    runs = []
    for task_idx in sorted(tasks):
        segments = [load_run(f) for f in tasks[task_idx]]
        cfg, evs = _merge_segments(segments)
        if cfg is not None and evs:
            # Surface the canonical filename (newest segment) so the
            # exclusion / attestation layers find this task by its current
            # name, not by the first segment's `.out` (which might be a
            # `log_NN.out.resume_0` after rotation).
            cfg["_log_filename"] = tasks[task_idx][-1].name
            runs.append((cfg, evs))
    _LOAD_SWEEP_CACHE[cache_key] = (sig, runs)
    return [(dict(cfg), evs) for cfg, evs in runs]


def has_runs(group: str, logs_root: str = "../logs") -> bool:
    """True if the group has at least one populated `log_NN.out` (or any
    `log_NN.out.resume_K` sibling)."""
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    if not log_dir.exists():
        return False
    return any(
        _TASK_FILE_RE.match(p.name) and p.stat().st_size > 0
        for p in log_dir.iterdir()
    )


# Cfg fields that legitimately differ between otherwise-identical runs (run
# provenance, instrumentation knobs, file paths). Canonical definition;
# ``loader.RUNTIME_FIELDS`` re-exports this name so dedup-key construction
# and the hidden-axis collision check share one source of truth (drift
# between two parallel lists previously caused silent collisions on
# ``_optim_steps`` differences — see git log for 2026-05-06 fix).
RUNTIME_FIELDS: frozenset[str] = frozenset({
    "git_commit", "command", "log_group",
    # Provenance fields (Phase 1 cfg-event enrichment, 2026-05-14): dirty-tree
    # state captured at submission. The values themselves don't affect
    # algorithmic behavior; the loader's invariants/dirty_attestations layers
    # consume them to decide whether a run is includable, but dedup must not
    # split otherwise-identical runs across these axes.
    "git_dirty", "git_diff_sha", "git_untracked_files",
    # Phase 4 (2026-05-14): execution-scope provenance. These drive the
    # loader's exclusion / auto-resolve decisions but don't define the
    # series. Two runs with different execution_source_sha may resolve to
    # the same effective_commit; their algorithmic identity is captured
    # by `optimizer` / `lr` / `seed` / etc. and not by these provenance
    # fields. Exclude from dedup.
    "execution_source_sha", "execution_source_paths",
    "execution_source_dirty", "execution_env", "execution_env_sha",
    # Loader-assigned per-run identifier; see loader._enrich_cfg.
    "run_id", "_log_filename",
    "wandb_project", "wandb_run_name",
    "device", "tf32", "no_tf32",
    # Diagnostic toggles (none affect optimizer math). Both the current
    # names and legacy aliases from before the 2026-05-12 diagnostics
    # refactor are listed so old/new schemas don't split series_id.
    "log_basic_diagnostics", "log_heavy_diagnostics",
    "log_optim_diagnostics", "no-log_optim_diagnostics",
    "optim_diagnostics_every",
    "diagnostics",   # canonical block (Phase 1, 2026-05-14)
    "profile_steps", "profile_dir",
    "_optim_steps",
    "train_file", "eval_file",
    # CLI override flags whose canonical resolved value is promoted by
    # `_enrich_cfg` to a top-level scalar (e.g. `effective_picard_iters`).
    # The raw override is then redundant — the effective field is the
    # source of truth for both loader dedup and series_id.
    "picard_iters_override",
})

# Backward-compatible alias.
_HIDDEN_AXIS_RUNTIME_FIELDS = RUNTIME_FIELDS


def _hidden_axis_diffs(cfg_a: dict, cfg_b: dict) -> list[tuple]:
    """Non-runtime cfg fields where ``cfg_a`` and ``cfg_b`` disagree.

    Treats missing-vs-present as a difference — this is the asymmetric
    failure mode that previously hid collisions. An older cfg without a
    field (e.g. ``picard_iters_override``) coexisting with a newer cfg
    that has it would not trigger the prior fingerprint check, because
    that check only compared a fixed allow-list of axes and silently
    skipped fields outside it.

    Returns a list of ``(field, value_in_a, value_in_b)`` tuples; missing
    values surface as the literal string ``"<missing>"``.
    """
    _MISSING = object()
    # Mirror `_denylist_key` / `series_id` exclusion rules: ignore runtime
    # metadata, underscore-prefixed enrichment namespaces, and dict-valued
    # derived composites. Treat None and absent as equivalent (consistent
    # with the loader's pre-dedup backfill of argparse defaults).
    candidate_keys = set(cfg_a) | set(cfg_b)
    keys = {
        k for k in candidate_keys
        if k not in _HIDDEN_AXIS_RUNTIME_FIELDS
        and not k.startswith("_")
        and not isinstance(cfg_a.get(k), dict)
        and not isinstance(cfg_b.get(k), dict)
    }
    diffs = []
    for k in sorted(keys):
        va = cfg_a.get(k, _MISSING)
        vb = cfg_b.get(k, _MISSING)
        if va is _MISSING:
            va = None
        if vb is _MISSING:
            vb = None
        if va is None and vb is None:
            continue
        if va != vb:
            diffs.append((
                k,
                "<missing>" if va is None else va,
                "<missing>" if vb is None else vb,
            ))
    return diffs


def merge_runs(group_priority: Iterable[str],
               key_fn: Callable[[dict], tuple],
               filter_fn: Callable[[dict], bool] | None = None,
               cfg_postprocess: Callable[[dict, str], None] | None = None,
               logs_root: str = "../logs",
               *, strict_hidden_axes: bool = True) -> list[tuple[dict, list[dict]]]:
    """Merge sweeps, deduplicating by ``key_fn(cfg)``.

    Dedup rule: **longest trajectory wins**, ties broken by group priority
    order (earlier group in ``group_priority`` wins). This handles the
    common case where an in-flight rerun has the same key as a completed
    older run — the completed run keeps its slot until the rerun catches
    up, and only takes over when the rerun reaches the same final step.

    Robustness check (``strict_hidden_axes=True`` default): if two runs
    collide on ``key_fn(cfg)`` but differ on a "hidden" cfg axis (e.g.
    ``lora_plus_multiplier``, ``muon_ns_steps``, ``training_mode``), raise
    a clear error pointing at which axis the dedup key is missing. This
    catches the silent-confounder failure mode where one cfg axis was
    forgotten in the key and runs are arbitrarily collapsed.

    Pass ``strict_hidden_axes=False`` only when you genuinely want axes
    collapsed (e.g., showing the best-of across m∈{1,4} as one curve).
    """
    # priority index for tie-breaking (lower = higher priority)
    prio = {g: i for i, g in enumerate(group_priority)}
    # key → (final_step, group_priority_idx, cfg, evs)
    best: dict[tuple, tuple[int, int, dict, list[dict]]] = {}
    for group, idx in sorted(prio.items(), key=lambda kv: kv[1]):
        if not has_runs(group, logs_root):
            continue
        for cfg, evs in load_sweep(group, logs_root):
            cfg["log_group"] = group
            if cfg_postprocess is not None:
                cfg_postprocess(cfg, group)
            if filter_fn is not None and not filter_fn(cfg):
                continue
            k = key_fn(cfg)
            final_step = evs[-1]["step"] if evs else 0
            existing = best.get(k)
            if existing is None:
                best[k] = (final_step, idx, cfg, evs)
                continue
            ex_step, ex_idx, ex_cfg, ex_evs = existing
            # Hidden-axis robustness: if cfgs share key_fn(cfg) but DIFFER on
            # ANY non-runtime cfg field (including missing-vs-present), that's
            # a missing-axis bug. Raise loudly so the caller adds the field
            # to key_fn — silent collapse is how the rsweep / picard_2x2
            # collision was hidden in 2026-05.
            if strict_hidden_axes:
                diffs = _hidden_axis_diffs(ex_cfg, cfg)
                if diffs:
                    field_summary = ", ".join(
                        f"{f}={va!r}↔{vb!r}" for f, va, vb in diffs[:5]
                    )
                    if len(diffs) > 5:
                        field_summary += f", … (+{len(diffs)-5} more)"
                    raise ValueError(
                        f"merge_runs: dedup key collision on {k!r} between cfgs "
                        f"that differ on non-runtime field(s): {field_summary}. "
                        f"Two distinct runs would be silently collapsed. Either "
                        f"include these fields in key_fn (or use the deny-list "
                        f"default in load_runs), or pass strict_hidden_axes=False "
                        f"if collapsing is intended. "
                        f"Groups: {ex_cfg.get('log_group')!r} vs {cfg.get('log_group')!r}."
                    )
            # Replace iff strictly more steps, OR equal steps but stricter group priority
            if final_step > ex_step or (final_step == ex_step and idx < ex_idx):
                best[k] = (final_step, idx, cfg, evs)
    return [(cfg, evs) for _, _, cfg, evs in best.values()]


# ─── diverged-run filtering ───────────────────────────────────────────────────

def max_loss(evs: list[dict]) -> float:
    # NaN-safe: a NaN eval_loss anywhere counts as inf (diverged).
    return max((float("inf") if (e["eval_loss"] != e["eval_loss"]) else e["eval_loss"])
               for e in evs)


def split_diverged(runs, threshold: float = DIVERGE_THRESHOLD,
                   hard_max: float = float("inf")):
    def _div(cfg, evs):
        final = evs[-1]["eval_loss"]
        # NaN final → diverged. Also any NaN mid-trajectory → diverged.
        if final != final or max_loss(evs) != max_loss(evs):  # NaN propagation guard
            return True
        # Early-termination guard: run that died in the first 10% of training
        # (and didn't get further) counts as diverged. Catches NaN-killed runs
        # whose pre-death eval happened to be finite. Threshold is loose enough
        # not to flag mid-flight partial-horizon runs, which `min_step`
        # filtering handles separately.
        max_steps = cfg.get("max_steps")
        if max_steps is not None and evs[-1]["step"] < max_steps * 0.1:
            return True
        return final >= threshold or max_loss(evs) >= hard_max

    keep = [(c, e) for c, e in runs if not _div(c, e)]
    drop = [(c, e) for c, e in runs if _div(c, e)]
    return keep, drop


def report_diverged(drop, label_fn: Callable[[dict], str]) -> None:
    for cfg, evs in sorted(drop, key=lambda x: label_fn(x[0])):
        print(f"  [filtered diverged] {label_fn(cfg):<24s} "
              f"max={max_loss(evs):.3f} final={evs[-1]['eval_loss']:.3f}")


# ─── baseline / reference helpers ────────────────────────────────────────────

def best_run(runs, filter_fn: Callable[[dict], bool]):
    """Return (cfg, evs) with lowest final eval_loss matching filter_fn, else None."""
    matches = [(c, e) for c, e in runs if filter_fn(c)]
    if not matches:
        return None
    return min(matches, key=lambda x: x[1][-1]["eval_loss"])


# Tuple shapes used by plot_eta_vs_final / plot_best_eta_curves:
#   hline:     (label, y, color, ls, lw)
#   ref_curve: (label, evs, color, ls, lw, marker)
# Older 3-tuple / 4-tuple shapes are accepted for backward compatibility.

def _normalize_hline(entry):
    if len(entry) == 3:
        label, y, color = entry
        return (label, y, color, ":", REF_LINE_WIDTH)
    return entry  # already 5-tuple


def _normalize_ref_curve(entry):
    # Supported tuple shapes (last variant carries multi-seed std band):
    #   4-tuple: (label, evs, color, ls)               — legacy
    #   6-tuple: (label, evs, color, ls, lw, marker)    — standard
    #   7-tuple: (label, mean_evs, color, ls, lw, marker, std_evs) — multi-seed
    if len(entry) == 4:
        label, evs, color, ls = entry
        return (label, evs, color, ls, REF_LINE_WIDTH, None, None)
    if len(entry) == 6:
        label, evs, color, ls, lw, marker = entry
        return (label, evs, color, ls, lw, marker, None)
    return entry  # already 7-tuple with std_evs


def _eta_sweep_points(reference_runs, optimizer: str,
                      *,
                      homogeneous_axes: tuple = ("max_steps", "lora_r"),
                      raise_on_mixed: bool = True):
    """Per-η (η, mean_final, std_final, n) points for `optimizer` in
    `reference_runs`. Multi-seed runs at the same η are aggregated.

    Robustness: ENFORCES that every lr-bucket is homogeneous on the
    `homogeneous_axes` (default: ``max_steps``, ``lora_r``). Mixing runs
    from different experimental regimes (e.g. a `max_steps=9000` phase_L
    sweep into a canonical `max_steps=4000` reference) is a silent
    data-pollution failure — it manifests as huge std on the error bars
    and wildly inflated/zigzag reference curves, which were one of THE
    recurring "where did this number come from?" bugs.

    On a mixed bucket, raise ``ValueError`` with the specific
    (lr, axis, values) detail so callers add the missing filter to
    their ``load_runs(where=...)`` call. Set ``raise_on_mixed=False``
    to downgrade to a stderr warning (only use this when you genuinely
    want cross-regime comparison — almost never the right answer).
    """
    import statistics
    import sys
    from collections import defaultdict
    buckets: dict[float, list[tuple]] = defaultdict(list)
    for c, e in reference_runs:
        if c.get("optimizer") != optimizer:
            continue
        # Store (eval_loss, axis_signature, log_group) so we can both
        # aggregate AND detect cross-regime contamination.
        axis_sig = tuple(_hashable(c.get(k)) for k in homogeneous_axes)
        buckets[float(c["lr"])].append(
            (e[-1]["eval_loss"], axis_sig, c.get("log_group", "?"))
        )

    # Homogeneity audit: each lr must have one and only one axis signature.
    mixed = []
    for lr in sorted(buckets):
        sigs = {entry[1] for entry in buckets[lr]}
        if len(sigs) > 1:
            details = {}
            for sig in sigs:
                groups = {entry[2] for entry in buckets[lr] if entry[1] == sig}
                details[sig] = sorted(groups)
            mixed.append((lr, details))
    if mixed:
        lines = [
            f"_eta_sweep_points: cross-regime pollution detected for "
            f"optimizer={optimizer!r}, homogeneous_axes={list(homogeneous_axes)}:"
        ]
        for lr, det in mixed[:5]:
            lines.append(f"  lr={lr:.0e}:")
            for sig, groups in det.items():
                lines.append(
                    f"    {dict(zip(homogeneous_axes, sig))!r} ← "
                    f"log_groups={groups}"
                )
        lines.append(
            "Fix: add the missing axis to your load_runs(where=...) "
            "filter, e.g. `'max_steps': 4000`. To intentionally mix "
            "regimes, pass raise_on_mixed=False or filter "
            "homogeneous_axes=()."
        )
        msg = "\n".join(lines)
        if raise_on_mixed:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    points = []
    for lr in sorted(buckets):
        ys = [entry[0] for entry in buckets[lr]]
        mean = sum(ys) / len(ys)
        std = statistics.stdev(ys) if len(ys) > 1 else 0.0
        points.append((lr, mean, std, len(ys)))
    return points


def _multi_seed_curve(reference_runs, optimizer: str, best_lr: float):
    """For ``optimizer`` at ``best_lr`` in ``reference_runs``, aggregate
    per-step eval_loss across seeds into a mean-and-std trajectory.

    Returns (mean_evs, std_evs, n_seeds), where ``mean_evs`` and ``std_evs``
    are lists of ``{step, eval_loss}`` dicts sharing the same step axis.
    When only one seed is present, ``std_evs`` has ``eval_loss=0`` at each
    step (renders as a zero-width band, i.e. just the line).

    Steps that don't appear in every seed are dropped (so the band is
    well-defined across the full sweep). Typical eval cadence is identical
    across seeds so this drops nothing in practice.
    """
    import statistics
    from collections import defaultdict
    per_step: dict[int, list[float]] = defaultdict(list)
    seeds_at_lr: set = set()
    n_seeds_at_lr = 0
    averaged_cfgs: list[dict] = []
    for c, evs in reference_runs:
        if c.get("optimizer") != optimizer:
            continue
        if float(c["lr"]) != best_lr:
            continue
        seed = c.get("seed")
        if seed in seeds_at_lr:
            continue
        seeds_at_lr.add(seed)
        n_seeds_at_lr += 1
        averaged_cfgs.append(c)
        for ev in evs:
            per_step[int(ev["step"])].append(float(ev["eval_loss"]))

    # Series-id contract (defensive): the runs we're about to average must
    # all share the same series_id, otherwise we'd silently mix distinct
    # algorithms across "seeds." Reaches here only when the caller bypassed
    # the entry-point assertion (standard_sweep_figure / plot_eta_vs_final).
    if len(averaged_cfgs) > 1:
        ids = {series_id(c) for c in averaged_cfgs}
        if len(ids) > 1:
            raise LabelCollisionError(
                f"_multi_seed_curve({optimizer=!r}, {best_lr=}) would "
                f"average across {len(ids)} distinct series_ids. The caller "
                f"must filter reference_runs to a single algorithm + model "
                f"config before requesting seed-aggregated curves."
            )
    # Keep only steps present in every seed
    common_steps = sorted(s for s, ys in per_step.items() if len(ys) == n_seeds_at_lr)
    mean_evs = []
    std_evs = []
    for s in common_steps:
        ys = per_step[s]
        mean_evs.append({"step": s, "eval_loss": sum(ys) / len(ys)})
        std_evs.append({"step": s,
                        "eval_loss": (statistics.stdev(ys) if len(ys) > 1 else 0.0)})
    return mean_evs, std_evs, n_seeds_at_lr


def baseline_overlay(reference_runs, optimizer: str, *,
                     label: str | None = None,
                     color: str | None = None,
                     is_primary: bool = False,
                     marker_map: dict | None = None,
                     ) -> tuple[list, list, list]:
    """Build (hlines, ref_curves, eta_sweeps) entries for overlaying the
    `optimizer` baseline from `reference_runs`.

    is_primary=True → primary baseline styling (heavier line, long-dash,
    no markers, "(baseline)" label) for the AdamW reference. False →
    lighter dotted reference style for secondary baselines.

    The library guarantees a single visual idiom for the AdamW baseline
    — color, linewidth, linestyle, marker, hline style are all fixed
    constants (BASELINE_*) and not configurable by callers. The floor
    hline uses a different linestyle than the training curve so the two
    don't visually merge.

    Returns:
        hlines:     [(label, y, color, ls, lw)]                                — left-panel floor
        ref_curves: [(label, mean_evs, color, ls, lw, marker, std_evs)]        — right-panel curve;
                    if multi-seed at best η, std_evs is a list of
                    {step, eval_loss} dicts (per-step std) and the renderer
                    fills a ±σ band around mean_evs. Single-seed → std_evs
                    is a list of zeros (no band drawn).
        eta_sweeps: [(label, points, color, ls, lw, marker)]                   — left-panel
                    η-sweep; ``points`` is now ``[(lr, mean, std, n), …]``
                    so the renderer can draw vertical seed-σ error bars.
                    Empty if < 2 η points.
    """
    label = label or optimizer
    sweep_points = _eta_sweep_points(reference_runs, optimizer)
    if not sweep_points:
        return [], [], []
    # Pick the best η by mean final loss (multi-seed aware), NOT by best-seed.
    best_lr, best_mean, best_std, best_n = min(sweep_points, key=lambda p: p[1])
    mean_evs, std_evs, n_seeds = _multi_seed_curve(
        reference_runs, optimizer, best_lr,
    )
    if not mean_evs:
        return [], [], []
    fl = mean_evs[-1]["eval_loss"]

    # Only the PRIMARY baseline (AdamW) gets the distinguished visual
    # register: long-dash, square markers, heavy line, black, floor hline.
    # Secondary references render as ordinary candidate-styled lines (solid,
    # circle markers, normal weight) — they're not baselines, just additional
    # comparisons pulled from reference_runs. This is enforced; callers
    # cannot upgrade a secondary to baseline styling.
    # Always emit the η-sweep entry — even single-η references render as
    # one point-with-error-bar, which is more informative than silently
    # dropping the baseline from the left panel when only one η was tested
    # at this rank (common when an lr-sweep was only run at one rank but
    # the figure is rendered at another).
    if is_primary:
        color = BASELINE_COLOR  # locked black
        hline = (f"{label} floor ({fl:.4f}{' ± ' + format(best_std, '.4f') if n_seeds > 1 else ''})",
                 fl, color, BASELINE_LS_HLINE, BASELINE_LW_HLINE)
        seed_tag = f", n_seed={n_seeds}" if n_seeds > 1 else ""
        curve = (f"{label} (baseline, η={best_lr:.0e}, final={fl:.4f}{seed_tag})",
                 mean_evs, color, BASELINE_LS_CURVE,
                 BASELINE_LW_CURVE, BASELINE_MARKER, std_evs)
        sweep_label = (f"{label} η-sweep ({len(sweep_points)} pts)"
                       if len(sweep_points) >= 2
                       else f"{label} @ η={best_lr:.0e} (only η tested)")
        eta_sweep = (
            sweep_label,
            sweep_points, color, BASELINE_LS_CURVE,
            BASELINE_LW_CURVE, BASELINE_MARKER,
        )
        return [hline], [curve], [eta_sweep]

    # Secondary reference: ordinary candidate styling. No hline.
    color = color or "#1f77b4"
    marker_map = marker_map or {}
    marker = marker_map.get(optimizer, "o")
    curve = (f"{label} (η={best_lr:.0e}, final={fl:.4f})",
             mean_evs, color, "-", LINE_WIDTH, marker, std_evs)
    sweep_label = (f"{label} η-sweep" if len(sweep_points) >= 2
                   else f"{label} @ η={best_lr:.0e}")
    eta_sweep = (
        sweep_label,
        sweep_points, color, "-", LINE_WIDTH, marker,
    )
    return [], [curve], [eta_sweep]


# ─── per-rank leaderboard bar chart ──────────────────────────────────────────

def plot_leaderboard_by_rank(best: dict, baseline_optimizer: str = "adamw",
                              color_map: dict | None = None,
                              marker_map: dict | None = None,
                              linestyle_map: dict | None = None,
                              suptitle: str = "Best eval vs rank"):
    """Single-panel leaderboard: best-η eval loss vs rank, one line per optimizer.

    ``marker_map`` overrides the per-optimizer marker shape (default "o").
    ``linestyle_map`` overrides the per-optimizer linestyle (default "-").
    Use both to disambiguate near-color pairs and per-m variants.
    """
    color_map = color_map or {}
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}
    baseline_floor = {r: best[(baseline_optimizer, r)][2]
                      for (opt, r) in best if opt == baseline_optimizer}
    ranks = sorted(baseline_floor)

    series = {}
    for (opt, r), (cfg, evs, fl) in best.items():
        series.setdefault(opt, []).append((r, fl))
    for opt in series:
        series[opt].sort()

    n_optimizers = len(series)
    # Legend below the plot in N columns so it doesn't squish a wide plot.
    # Plot itself stays at a fixed comfortable size; figure height grows with
    # the legend block underneath.
    legend_ncol = max(1, min(4, (n_optimizers + 7) // 8))
    legend_rows = (n_optimizers + legend_ncol - 1) // legend_ncol
    plot_height = 5.0
    legend_height = 0.30 * legend_rows + 0.5
    fig_height = plot_height + legend_height
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    all_losses = []
    for opt, points in sorted(series.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        is_baseline = opt == baseline_optimizer
        ax.plot(xs, ys,
                color=BASELINE_COLOR if is_baseline else color_map.get(opt, "grey"),
                lw=BASELINE_LW_CURVE if is_baseline else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_baseline else linestyle_map.get(opt, "-"),
                marker=BASELINE_MARKER if is_baseline else marker_map.get(opt, "o"),
                markersize=MARKER_SIZE,
                label=f"{opt} (baseline)" if is_baseline else opt,
                zorder=BASELINE_ZORDER if is_baseline else 5)
        all_losses.extend(ys)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("Best-η final eval loss")
    ax.set_title(suptitle, fontsize=12, fontweight="bold")
    if all_losses:
        lo = min(all_losses) - 0.005
        hi = min(all_losses) + 0.05
        ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25, lw=0.6)
    legend_kw = dict(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_ncol,
        fontsize=11 if n_optimizers > 8 else 13,
        frameon=True,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.4,
        columnspacing=1.5,
    )
    ax.legend(**legend_kw)
    return fig, ax


# ─── standardized 2-panel figure ──────────────────────────────────────────────

def _legend_kw(n_entries: int) -> dict:
    """Single-column legend always — 2-col looked busy and broke the
    visual hierarchy. Figure width carries the room instead."""
    return dict(LEGEND_KW_BASE)


def plot_eta_vs_final(ax, runs, group_key_fn: Callable[[dict], str],
                      color_map: dict, *, hlines: list[tuple] | None = None,
                      ref_eta_sweeps: list[tuple] | None = None,
                      title: str = "Final eval loss vs η, per group",
                      legend: bool = True,
                      adamw_group_keys: set[str] | None = None,
                      marker_map: dict | None = None,
                      linestyle_map: dict | None = None,
                      normalize_x_to_optimum: bool = False,
                      diverged_runs: list | None = None,
                      allow_label_collision: bool = False) -> None:
    """Left panel: η vs final eval loss, one line per group key.

    `ref_eta_sweeps`: list of (label, points, color, ls, lw, marker) tuples
    where `points` is a list of (η, final_loss) — e.g. the AdamW baseline's
    full LR-sweep curve. Drawn before candidates so candidate lines sit on top.

    `normalize_x_to_optimum`: when True, each series is plotted at x = η/η⋆
    where η⋆ is the lr giving the lowest mean eval for that series. Methods
    optimum-align at x=1 so the SHAPE of the lr-response curve is comparable
    across methods with very different absolute optima. When ≥2 lrs per
    series are present.
    """
    adamw_group_keys = adamw_group_keys or set()
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}

    # Defense in depth: enforce the series-id contract here too, in case
    # this is called outside standard_sweep_figure. Idempotent when the
    # caller already asserted.
    if not allow_label_collision and runs:
        assert_label_discriminates(runs, group_key_fn)

    def _maybe_norm(xs, ys):
        """If normalize_x_to_optimum: divide xs by argmin-of-ys's x (= η⋆ for
        this series). Single-point series still normalize (their one lr IS
        the optimum by definition, so x=1).
        """
        if not normalize_x_to_optimum or len(ys) < 1:
            return xs
        best_idx = min(range(len(ys)), key=lambda i: ys[i])
        eta_star = xs[best_idx]
        if eta_star <= 0:
            return xs
        return [x / eta_star for x in xs]

    all_losses = []
    if ref_eta_sweeps:
        for entry in ref_eta_sweeps:
            label, points, color, ls, lw, marker = entry
            if not points:
                continue
            # `points` rows are (lr, mean, std, n) per `_eta_sweep_points`.
            # Legacy (lr, eval) 2-tuples still supported.
            if len(points[0]) == 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                stds = [p[2] for p in points]
                xs = _maybe_norm(xs, ys)
                ax.errorbar(xs, ys, yerr=stds, color=color, ls=ls, lw=lw,
                            marker=marker, markersize=MARKER_SIZE,
                            label=label, zorder=BASELINE_ZORDER,
                            capsize=6, capthick=lw, elinewidth=lw)
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xs = _maybe_norm(xs, ys)
                ax.plot(xs, ys, color=color, ls=ls, lw=lw,
                        marker=marker, markersize=MARKER_SIZE,
                        label=label, zorder=BASELINE_ZORDER)
            all_losses.extend(ys)
    if hlines:
        for entry in hlines:
            label, y, color, ls, lw = _normalize_hline(entry)
            ax.axhline(y, color=color, ls=ls, lw=lw, label=label,
                       zorder=BASELINE_ZORDER)
            all_losses.append(y)

    in_range_losses = []
    groups = sorted({group_key_fn(c) for c, _ in runs})

    # ── PASS 1: aggregate per-group (no clamping yet) ─────────────────────
    # Two-pass design: we need to know `hi` (the visible top of the y-axis,
    # computed from in_range_losses) BEFORE we pick a y_cap for clamping
    # OOR / diverged points. With the old single-pass design y_cap was
    # set to `min(raw_losses) + 0.15` upfront, which could exceed `hi`
    # (for tight in-range clusters), pushing hollow markers off-screen.
    # The two-pass version pins y_cap just below `hi`, guaranteeing
    # markers are visible without inflating the axis.
    import statistics as _stats
    from collections import defaultdict as _dd
    group_agg = []  # [(g, xs, means, stds, is_oor_finite, style_dict)]
    for g in groups:
        # Aggregate per-lr across seeds: (lr → list of eval_losses).
        by_lr: dict[float, list[float]] = _dd(list)
        for c, e in runs:
            if group_key_fn(c) != g:
                continue
            by_lr[float(c["lr"])].append(e[-1]["eval_loss"])
        # Fold diverged runs (NaN-final / max>thresh / early-killed) for
        # THIS group into the same aggregation, sentinel = +∞. They flow
        # through the OOR clamping path → hollow marker on the SAME
        # colored connecting line.
        if diverged_runs:
            for c, e in diverged_runs:
                if group_key_fn(c) != g:
                    continue
                by_lr[float(c["lr"])].append(float("inf"))
        if not by_lr:
            continue
        agg = sorted(
            (lr, sum(ys)/len(ys),
             _stats.stdev(ys) if len(ys) > 1 else 0.0,
             len(ys))
            for lr, ys in by_lr.items()
        )
        xs = [p[0] for p in agg]
        means = [p[1] for p in agg]
        stds = [p[2] for p in agg]
        # Two flavors of "out-of-range":
        #   non-finite (NaN / +∞ sentinel from diverged_runs) — ALWAYS OOR
        #   finite-but-above-y_cap — decided per-group later once y_cap set
        is_nonfinite = [not (math.isfinite(m)) for m in means]
        in_range_losses.extend(m for m, nf in zip(means, is_nonfinite) if not nf)
        is_adamw = g in adamw_group_keys
        style = dict(
            color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
            marker=BASELINE_MARKER if is_adamw else marker_map.get(g, "o"),
            ls=BASELINE_LS_CURVE if is_adamw else linestyle_map.get(g, "-"),
            lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
            zorder=BASELINE_ZORDER if is_adamw else 5,
            is_adamw=is_adamw,
        )
        group_agg.append((g, xs, means, stds, is_nonfinite, style))

    # ── compute y-axis bounds before pass 2 ──────────────────────────────
    if in_range_losses or all_losses:
        lo = min(in_range_losses + all_losses) - 0.005
        hi = (max(in_range_losses) if in_range_losses
               else max(all_losses)) + 0.01
        hi = min(hi + 0.005, lo + 0.16)
    else:
        lo, hi = 0.0, 1.0
    # y_cap = clamping height for OOR / diverged. Just inside the axis top,
    # but never below max(in_range) (else legit in-range points would get
    # clipped). With a comfortable in-range cluster the hollow marker sits
    # ~1 marker-radius above the top of the in-range cloud.
    if in_range_losses:
        max_in_range = max(in_range_losses)
        y_cap = max(max_in_range + 0.003, hi - 0.006)
    else:
        y_cap = hi - 0.006

    # Surface in-range points clipped by y_cap (rare — only when a finite
    # mean is above the cap). Kept as a "where's my data?" sentinel that
    # used to fire often; with the new y_cap≈hi-pad it should be near-zero.
    clipped = []
    for g, xs, means, stds, is_nonfinite, _style in group_agg:
        for lr, m, nf in zip(xs, means, is_nonfinite):
            if not nf and m > y_cap:
                clipped.append((g, lr, m))
    if clipped:
        print(f"  [clipped from left panel y_cap={y_cap:.3f}] {len(clipped)} run(s):")
        for g, lr, fl in sorted(clipped):
            print(f"    {g} η={lr:.0e} final={fl:.4f}")

    # ── PASS 2: render lines + markers ───────────────────────────────────
    for g, xs, means, stds, is_nonfinite, style in group_agg:
        ys_clamped = [y_cap if (nf or m > y_cap) else m
                      for m, nf in zip(means, is_nonfinite)]
        is_oor = [nf or m > y_cap for m, nf in zip(means, is_nonfinite)]
        if normalize_x_to_optimum and any(not o for o in is_oor):
            in_pairs = [(xs[i], means[i]) for i, o in enumerate(is_oor) if not o]
            eta_star = min(in_pairs, key=lambda p: p[1])[0]
            if eta_star > 0:
                xs = [x / eta_star for x in xs]
        color, marker, ls, lw, zorder, is_adamw = (
            style["color"], style["marker"], style["ls"],
            style["lw"], style["zorder"], style["is_adamw"])
        # Connecting line through clamped means.
        ax.plot(xs, ys_clamped, color=color, lw=lw, ls=ls, zorder=zorder,
                label=f"{g} (baseline)" if is_adamw else g)
        # In-range markers + ±σ error bars (zero σ on single-seed cells
        # → caps collapse to the marker, visually identical to a plain marker).
        in_idx = [i for i, o in enumerate(is_oor) if not o]
        if in_idx:
            ax.errorbar(
                [xs[i] for i in in_idx],
                [means[i] for i in in_idx],
                yerr=[stds[i] for i in in_idx],
                color=color, marker=marker, markersize=MARKER_SIZE,
                ls="", zorder=zorder + 1,
                capsize=4, capthick=lw * 0.6, elinewidth=lw * 0.6,
            )
        # Hollow markers for OOR / diverged means (sit at y_cap, inside hi).
        oor_x = [xs[i] for i, o in enumerate(is_oor) if o]
        oor_y = [ys_clamped[i] for i, o in enumerate(is_oor) if o]
        if oor_x:
            ax.plot(oor_x, oor_y, color=color, marker=marker,
                    markersize=MARKER_SIZE + 4, markerfacecolor="none",
                    markeredgewidth=2.2, ls="", zorder=zorder + 2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta / \eta^\star$ (log)" if normalize_x_to_optimum else "η (log)",
                   fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Final eval loss", fontsize=AXIS_LABEL_FONTSIZE)
    if normalize_x_to_optimum:
        ax.axvline(1.0, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.6, which="major")
    ax.grid(True, alpha=0.10, lw=0.5, which="minor")
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if in_range_losses or all_losses:
        lo = min(in_range_losses + all_losses) - 0.005
        hi = (max(in_range_losses) if in_range_losses
               else max(all_losses)) + 0.01
        hi = min(hi + 0.005, lo + 0.16)
        ax.set_ylim(lo, hi)

    # Diverged runs are folded into the per-group aggregation above
    # (sentinel +∞ → clamped to y_cap → hollow marker on the same colored
    # connecting line). Hollow markers sit at y_cap; if y_cap > hi (tight
    # in-range data), they're clipped at the axis top — the colored
    # connecting line still rises off the top of the panel, signalling
    # the diverged points. A follow-up TODO is to compute y_cap from hi
    # in a two-pass aggregate so the markers are guaranteed visible
    # without expanding hi.
    if legend:
        ax.legend(**_legend_kw(len(groups)))


def plot_best_eta_curves(ax, runs, group_key_fn: Callable[[dict], str],
                         color_map: dict, *, ref_curves: list[tuple] | None = None,
                         title: str = "Best η per group — training curves",
                         x_tick_step: int = 200,
                         legend: bool = True,
                         adamw_group_keys: set[str] | None = None,
                         marker_map: dict | None = None,
                         linestyle_map: dict | None = None) -> None:
    """Right panel: training curves for the best (lowest final loss) η per group."""
    adamw_group_keys = adamw_group_keys or set()
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}

    if ref_curves:
        for entry in ref_curves:
            label, evs, color, ls, lw, marker, std_evs = _normalize_ref_curve(entry)
            xs = [e["step"] for e in evs]
            ys = [e["eval_loss"] for e in evs]
            # If multi-seed std present and not all-zero, draw a ±σ shaded
            # band. Otherwise fall through to plain line. Band rendered
            # behind the line so the line stays the visual anchor.
            if std_evs is not None and any(s["eval_loss"] > 0 for s in std_evs):
                stds = [s["eval_loss"] for s in std_evs]
                lo = [y - s for y, s in zip(ys, stds)]
                hi = [y + s for y, s in zip(ys, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18,
                                zorder=BASELINE_ZORDER - 1, linewidth=0)
            ax.plot(xs, ys,
                    color=color, lw=lw, ls=ls, marker=marker,
                    markersize=MARKER_SIZE, label=label,
                    zorder=BASELINE_ZORDER)

    best = {}
    for cfg, evs in runs:
        g = group_key_fn(cfg)
        fl = evs[-1]["eval_loss"]
        if g not in best or fl < best[g][2]:
            best[g] = (cfg, evs, fl)
    n_lines = len(best) + (len(ref_curves) if ref_curves else 0)
    for g, (cfg, evs, fl) in sorted(best.items(), key=lambda kv: kv[1][2]):
        is_adamw = g in adamw_group_keys
        label = (f"{g} (baseline, η={cfg['lr']:.0e}, final={fl:.4f})"
                 if is_adamw else f"{g} (η={cfg['lr']:.0e}, final={fl:.4f})")
        ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
                marker=BASELINE_MARKER if is_adamw else marker_map.get(g, "o"),
                markersize=MARKER_SIZE,
                lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_adamw else linestyle_map.get(g, "-"),
                zorder=BASELINE_ZORDER if is_adamw else 5,
                label=label)
    ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Eval loss", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if legend:
        ax.legend(**_legend_kw(n_lines))


CANONICAL_HORIZON = 2000  # legacy unpacked_v0 default; packed_v1 uses 4000.
                          # `_infer_min_step` reads cfg["max_steps"] from the actual
                          # input runs and falls back to this only when absent.


def _infer_min_step(runs) -> int | None:
    """Return the target horizon by reading `max_steps` from the input cfgs.

    Multiple max_steps in the same `runs` list (mixed-regime) → pick the
    MAX, since the left panel can only report "final" for runs that hit
    the highest target. Returns None if no cfg carries `max_steps`.
    """
    targets = {int(c["max_steps"]) for c, _ in runs if "max_steps" in c}
    return max(targets) if targets else None


# Sentinel: standard_sweep_figure / two_panel_sweep_figure should auto-infer
# min_step from the input runs (cfg["max_steps"]) rather than defaulting to
# CANONICAL_HORIZON=2000, which silently mis-fires under packed_v1's 4000-step
# regime.
_INFER_MIN_STEP: object = object()


def two_panel_sweep_figure(runs, group_key_fn, color_map, *,
                           suptitle: str = "",
                           hlines: list[tuple] | None = None,
                           ref_curves: list[tuple] | None = None,
                           ref_eta_sweeps: list[tuple] | None = None,
                           threshold: float = DIVERGE_THRESHOLD,
                           min_step: int | None | object = _INFER_MIN_STEP,
                           figsize: tuple = DEFAULT_FIGSIZE,
                           x_tick_step: int = 200,
                           left_title: str = "Final eval loss vs η, per group",
                           right_title: str = "Best η per group — training curves",
                           label_fn: Callable[[dict], str] | None = None,
                           adamw_group_keys: set[str] | None = None,
                           marker_map: dict | None = None,
                           linestyle_map: dict | None = None,
                           normalize_x_to_optimum: bool = False):
    """Build the standardized 2-panel sweep figure with diverged-run filtering.
    Returns (fig, axes, n_kept, n_dropped).

    Layout: left panel is η-vs-final-loss with legend suppressed (same colors
    as the right panel which carries descriptive labels). Right panel has
    best-η training curves with descriptive legend outside on the right.
    Constrained layout reserves room for the outside legend.
    """
    keep, drop = split_diverged(runs, threshold)
    if drop:
        if label_fn is None:
            label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
        report_diverged(drop, label_fn)

    # Left panel implicitly assumes the run reached its final loss (it
    # plots evs[-1]['eval_loss'] as "final"). In-flight runs at step 200
    # would bias the comparison. Exclude them from the LEFT panel only —
    # the right panel still shows the partial training curve so progress
    # is visible. Set min_step=None to disable.
    if min_step is _INFER_MIN_STEP:
        min_step = _infer_min_step(keep) or CANONICAL_HORIZON
        print(f"  [auto] min_step inferred from cfg['max_steps']: {min_step}")
    keep_for_right = keep
    keep_for_left = keep
    if min_step is not None:
        partial = [(c, e) for c, e in keep if e[-1]["step"] < min_step]
        keep_for_left = [(c, e) for c, e in keep if e[-1]["step"] >= min_step]
        if partial:
            print(f"  [filtered partial from left panel] {len(partial)} run(s) below min_step={min_step}:")
            if label_fn is None:
                label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
            for cfg, evs in sorted(partial, key=lambda x: label_fn(x[0])):
                print(f"    {label_fn(cfg):<30s} step={evs[-1]['step']} "
                      f"final-so-far={evs[-1]['eval_loss']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    # When normalizing, retitle so the panel header reflects the x-axis.
    effective_left_title = (
        r"Final eval loss vs $\eta/\eta^\star$, per group"
        if normalize_x_to_optimum and left_title == "Final eval loss vs η, per group"
        else left_title
    )
    plot_eta_vs_final(axes[0], keep_for_left, group_key_fn, color_map,
                      hlines=hlines, ref_eta_sweeps=ref_eta_sweeps,
                      title=effective_left_title, legend=False,
                      adamw_group_keys=adamw_group_keys,
                      marker_map=marker_map,
                      linestyle_map=linestyle_map,
                      normalize_x_to_optimum=normalize_x_to_optimum,
                      diverged_runs=drop)
    plot_best_eta_curves(axes[1], keep_for_right, group_key_fn, color_map,
                         ref_curves=ref_curves, title=right_title,
                         x_tick_step=x_tick_step,
                         adamw_group_keys=adamw_group_keys,
                         marker_map=marker_map,
                         linestyle_map=linestyle_map)
    if suptitle:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    return fig, axes, len(keep), len(drop)


def standard_sweep_figure(runs, group_key_fn, color_map, *,
                          reference_runs,
                          suptitle: str = "",
                          extra_baselines: Iterable[tuple[str, str]] = (),
                          baseline_optimizer: str = "adamw",
                          same_value_axes: tuple = ("lora_plus_multiplier",),
                          allow_label_collision: bool = False,
                          **kwargs):
    """High-level uniform sweep figure — every section's single entry point.

    The library guarantees, on every call:
      - The baseline (AdamW) overlay: a single dotted floor hline (left
        panel), the AdamW η-sweep curve (left panel) so the LR grid that
        was tuned is visible, and the AdamW best-η training curve (right
        panel). All in distinguished long-dash black.
      - Identical layout, fonts, legend placement, grid, and figure size.

    Visual styling is locked at the library level — callers cannot override
    color, linestyle, linewidth, marker for the baseline. The only knobs
    are: which optimizer is the primary baseline, and whether to overlay
    secondary references via `extra_baselines`.

    Args:
        runs:            candidate sweeps to compare, list of (cfg, evs).
        group_key_fn:    cfg → legend group string.
        color_map:       group string → matplotlib color.
        reference_runs:  REQUIRED. Sweeps containing the baseline optimizer
                         (typically `all_runs`). Pass `runs` itself when the
                         baseline is among the candidates.
        suptitle:        figure suptitle (run-count tally is appended).
        extra_baselines: extra (optimizer, color) pairs to overlay as light
                         dotted references.
        **kwargs:        forwarded to two_panel_sweep_figure.

    Raises:
        ValueError: if `reference_runs` has no run for `baseline_optimizer`.
    """
    # Multi-rank input: auto-split into per-rank figures. Avoids the
    # silent-collision failure mode (runs at different ranks collapsing under
    # one group_key_fn(cfg)) without forcing callers to pre-filter. Each rank
    # gets its own complete 2-panel figure with the rank in the suptitle.
    ranks = sorted({int(c.get("lora_r", 16)) for c, _ in runs})
    if len(ranks) > 1:
        results = []
        for r in ranks:
            run_slice = [(c, e) for c, e in runs if int(c.get("lora_r", 16)) == r]
            ref_slice = [(c, e) for c, e in reference_runs
                         if int(c.get("lora_r", 16)) == r]
            # Fall back to full reference_runs if the rank-filtered slice has
            # no baseline (older reference data may not be rank-tagged).
            if not any(c.get("optimizer") == baseline_optimizer for c, _ in ref_slice):
                ref_slice = reference_runs
            results.append(standard_sweep_figure(
                run_slice, group_key_fn, color_map,
                reference_runs=ref_slice,
                suptitle=suptitle,
                extra_baselines=extra_baselines,
                baseline_optimizer=baseline_optimizer,
                same_value_axes=same_value_axes,
                allow_label_collision=allow_label_collision,
                **kwargs,
            ))
        return results

    # Series-id discrimination contract: every (label, lora_r, lr) bucket
    # must contain exactly one series_id. Otherwise the per-label averaging
    # downstream silently mixes distinct algorithms. See
    # `assert_label_discriminates`.
    if not allow_label_collision and runs:
        assert_label_discriminates(runs, group_key_fn)

    # Robustness: every run in `runs` must agree on `same_value_axes`. Catches
    # the bug where a panel filter forgets to constrain an axis (e.g. m=1 only)
    # and runs with mixed values silently collapse via group_key_fn into one
    # group, picking "best across the omitted axis" rather than the intended
    # comparison. Pass `same_value_axes=()` to opt out (e.g., muon-variants
    # cell where m is explicit in the group label).
    if same_value_axes and runs:
        for axis in same_value_axes:
            values = {c.get(axis) for c, _ in runs if axis in c}
            if len(values) > 1:
                raise ValueError(
                    f"standard_sweep_figure: runs span multiple values of "
                    f"{axis!r}: {sorted(values)}. group_key_fn would silently "
                    f"collapse these into one group. Filter the runs by "
                    f"{axis} before calling, or pass same_value_axes=() if "
                    f"the multi-value comparison is intentional and the "
                    f"group_key_fn already encodes {axis}."
                )

    # Marker map propagates to baseline overlays so secondary references
    # (extra_baselines) and the primary baseline both honor per-optimizer markers.
    _marker_map = kwargs.get("marker_map") or {}

    # Filter partial-horizon runs out of reference_runs BEFORE the η-sweep is
    # built, so the left-panel AdamW η-sweep doesn't include in-flight runs
    # at intermediate steps. Use the same min_step the candidate-side filter
    # will use (inferred from candidate cfg.max_steps in two_panel_sweep_figure).
    _min_step = kwargs.get("min_step", _INFER_MIN_STEP)
    if _min_step is _INFER_MIN_STEP:
        _min_step = _infer_min_step(runs) or CANONICAL_HORIZON
    if reference_runs is not None and _min_step is not None:
        n_before = len(reference_runs)
        complete = [(c, e) for c, e in reference_runs
                    if e and e[-1]["step"] >= _min_step]
        baseline_complete = [r for r in complete
                             if r[0].get("optimizer") == baseline_optimizer]
        if baseline_complete:
            reference_runs = complete
            n_dropped = n_before - len(reference_runs)
            if n_dropped:
                print(f"  [auto] filtered {n_dropped} partial reference run(s) below min_step={_min_step}")
        else:
            # No complete baseline at this horizon — keep partial baseline runs
            # so the figure still renders with a (possibly in-flight) reference.
            partial_baseline = [r for r in reference_runs
                                if r[0].get("optimizer") == baseline_optimizer]
            if partial_baseline:
                last = max(e[-1]["step"] for _, e in partial_baseline if e)
                print(f"  [auto] no {baseline_optimizer!r} reference reached "
                      f"min_step={_min_step}; using partial baseline (last step={last})")

    if reference_runs is None:
        hlines, ref_curves, eta_sweeps = [], [], []
    else:
        hlines, ref_curves, eta_sweeps = baseline_overlay(
            reference_runs, baseline_optimizer, is_primary=True,
            marker_map=_marker_map,
        )
        if not hlines:
            raise ValueError(
                f"No {baseline_optimizer!r} run found in reference_runs — "
                "every standard sweep figure requires the baseline.")

    # Library-enforced uniform suptitle: append " at r={N}" when single-rank
    # and rank isn't already in the title.
    if ranks and suptitle and "r=" not in suptitle:
        suptitle = f"{suptitle} at r={ranks[0]}"

    for opt, color in extra_baselines:
        if reference_runs is None:
            continue
        _h, r, e = baseline_overlay(
            reference_runs, opt, color=color, is_primary=False,
            marker_map=_marker_map,
        )
        # Secondary baselines contribute the right-panel reference curve
        # AND the left-panel η-sweep (so the LR grid for the secondary
        # comparison is visible too). The hline is suppressed — the
        # primary baseline (AdamW) owns the only floor hline.
        ref_curves.extend(r)
        eta_sweeps.extend(e)

    # If the candidate runs already contain the baseline (e.g. all-optimizers
    # or cross-investigation cells), the candidate's own η-sweep + best-η
    # curve already cover the left + right panel; the overlay would
    # double-draw. Detect overlap, restyle the candidate to baseline style
    # via adamw_group_keys, and drop the overlay's eta_sweep + ref_curve
    # (keep the floor hline — the candidate doesn't draw that).
    candidate_groups = {group_key_fn(c) for c, _ in runs}
    adamw_overlap_groups = candidate_groups & {baseline_optimizer}
    if adamw_overlap_groups:
        ref_curves = []
        eta_sweeps = []

    return two_panel_sweep_figure(
        runs, group_key_fn, color_map,
        suptitle=suptitle,
        hlines=hlines, ref_curves=ref_curves, ref_eta_sweeps=eta_sweeps,
        adamw_group_keys=adamw_overlap_groups,
        **kwargs,
    )
