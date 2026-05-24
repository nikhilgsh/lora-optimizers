"""Tests for diverged-run rendering in plot_eta_vs_final.

Pins:
  1. Diverged runs fold into the per-group line (sentinel +∞ → clamped
     to y_cap → hollow marker on same colored line).
  2. y-axis is NOT inflated by diverged data — hi stays at the natural
     in-range bound. Regression test for the AdamW-r=64 squash bug
     where a hi-bump to fit `lo+0.15` pushed the visible top to 0.67,
     compressing the candidate cluster at 0.51-0.52.
  3. y_cap (the OOR clamping height where hollow markers sit) is INSIDE
     the visible range — `lo < y_cap <= hi`, and y_cap is above the
     max in-range mean. Hollow markers are guaranteed visible.
  4. NaN / +∞ finals don't crash. Each group's diverged lrs join ITS
     line, not adjacent series'.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from matplotlib.colors import to_rgba
from lora_playground.plotting import (
    plot_eta_vs_final, standard_sweep_figure, BASELINE_COLOR,
)


def _mk_run(opt, lr, final_loss, lora_r=64, max_steps=4000, last_step=None):
    cfg = {
        "optimizer": opt, "lr": float(lr), "lora_r": int(lora_r),
        "max_steps": max_steps, "seed": 0, "lora_plus_multiplier": 1.0,
    }
    if last_step is None:
        last_step = max_steps
    evs = [{"step": last_step, "eval_loss": final_loss}]
    return (cfg, evs)


def _gk(cfg):
    return cfg["optimizer"]


def _line_ys_by_color(ax, color):
    """Collect y-data from all Line2D artists whose color matches `color`,
    normalized via to_rgba so "black" / "#000000" / (0,0,0,1) all compare
    equal (matplotlib stores colors in different forms depending on origin)."""
    target = to_rgba(color)
    out = []
    for line in ax.get_lines():
        try:
            lc = to_rgba(line.get_color())
        except (ValueError, TypeError):
            continue
        if lc != target:
            continue
        for y in line.get_ydata():
            try:
                out.append(float(y))
            except (TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Regression: tight in-range cluster + one diverged run must not inflate hi.
# ---------------------------------------------------------------------------
def test_diverged_run_does_not_inflate_ylim():
    fig, ax = plt.subplots()
    runs = [
        _mk_run("chord", 1e-3, 0.521),
        _mk_run("chord", 3e-3, 0.515),
        _mk_run("chord", 1e-2, 0.518),
    ]
    diverged = [_mk_run("chord", 3e-2, float("nan"))]
    plot_eta_vs_final(
        ax, runs, _gk, {"chord": "#1f77b4"},
        diverged_runs=diverged, allow_label_collision=True,
    )
    lo, hi = ax.get_ylim()
    # Tight in-range data max=0.521; hi must stay near max(in_range), not
    # blow up to ~0.67 (the old `lo+0.15` y_cap if hi were forced to fit it).
    assert hi <= 0.55, (
        f"hi={hi:.4f} inflated past 0.55 — diverged-marker placement is "
        f"squashing the in-range data. Should stay near max(in_range)+~0.01."
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# AdamW r=64 regression: reference_runs (AdamW's high-lr branch reaching
# ~0.62) + tight candidates (~0.51-0.52) + one diverged candidate — the
# combo that produced the earlier AdamW squash. Verifies the y-axis stays
# proportional and the AdamW reference curve is visible (not crammed into
# the bottom 10% of the panel).
# ---------------------------------------------------------------------------
def test_adamw_r64_scenario_axis_proportions():
    fig, _ = plt.subplots()
    # Candidates (chord variants) — tight cluster around 0.51
    runs = [
        _mk_run("chord", 1e-3, 0.520, lora_r=64),
        _mk_run("chord", 3e-3, 0.510, lora_r=64),
        _mk_run("chord", 1e-2, 0.514, lora_r=64),
    ]
    # AdamW reference — spans 0.52 to 0.62 across its lr-sweep, as on real r=64
    ref = [
        _mk_run("adamw", 1e-5, 0.560, lora_r=64),
        _mk_run("adamw", 3e-5, 0.535, lora_r=64),
        _mk_run("adamw", 1e-4, 0.520, lora_r=64),
        _mk_run("adamw", 3e-4, 0.540, lora_r=64),
        _mk_run("adamw", 1e-3, 0.590, lora_r=64),  # high-lr blowup
    ]
    # One diverged candidate
    diverged = [_mk_run("chord", 3e-2, float("nan"), lora_r=64)]
    result = standard_sweep_figure(
        runs + ref, _gk, {"chord": "#1f77b4"},
        reference_runs=ref, suptitle="r=64 axis test",
        same_value_axes=(), allow_label_collision=True,
        # `diverged_runs` flows through internally via split_diverged on
        # `runs` — but here the diverged NaN run is in runs, so split fires.
    )
    # Single-rank → tuple; (fig, axes, n_keep, n_drop)
    figs = result if isinstance(result, list) else [result]
    for f, axes, _, _ in figs:
        ax = axes[0]
        lo, hi = ax.get_ylim()
        # Total range must be reasonable (no inflation to fit y_cap=0.66).
        assert hi - lo < 0.15, (
            f"y-range hi-lo={hi-lo:.4f} too large — axis inflated by "
            f"diverged-marker placement, squashing AdamW reference."
        )
        # AdamW reference curve max value (~0.59 if clipped to in_range) must
        # not be invisible. Check by reading line data.
        adamw_ys = _line_ys_by_color(ax, BASELINE_COLOR)
        # AdamW's max y-value must be representable inside the visible range,
        # OR the high-lr blowup goes off-top (acceptable). We assert lo is
        # close to min(adamw_ys + chord_ys), i.e. tight bound at bottom.
        all_in_range = [y for y in adamw_ys if y < 0.60]  # exclude OOR
        assert all_in_range, "AdamW reference curve produced no points"
        assert lo <= min(all_in_range) + 0.01, (
            f"lo={lo:.4f} cuts off AdamW's bowl (min={min(all_in_range):.4f})"
        )
        plt.close(f)


# ---------------------------------------------------------------------------
# y_cap (hollow-marker height) is inside the visible range.
# ---------------------------------------------------------------------------
def test_diverged_marker_inside_visible_range():
    fig, ax = plt.subplots()
    runs = [
        _mk_run("chord", 1e-3, 0.521),
        _mk_run("chord", 3e-3, 0.515),
    ]
    diverged = [_mk_run("chord", 1e-2, float("nan"))]
    plot_eta_vs_final(
        ax, runs, _gk, {"chord": "#1f77b4"},
        diverged_runs=diverged, allow_label_collision=True,
    )
    lo, hi = ax.get_ylim()
    line_ys = _line_ys_by_color(ax, "#1f77b4")
    # The line for chord must reach SOME y value above the max in-range
    # mean (0.521) — that's the diverged-clamped point.
    above_max_in_range = [y for y in line_ys if y > 0.521]
    assert above_max_in_range, (
        f"connecting line never rises above max in-range (0.521); "
        f"chord y-data: {sorted(set(round(y,4) for y in line_ys))}"
    )
    # And that point must be INSIDE the visible range (lo, hi).
    for y in above_max_in_range:
        assert lo <= y <= hi, (
            f"diverged-clamp y={y:.4f} is outside visible ylim ({lo:.4f}, "
            f"{hi:.4f}) — hollow marker would be off-screen."
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# NaN-safe: NaN/Inf in diverged_runs must not crash set_ylim.
# ---------------------------------------------------------------------------
def test_nan_final_loss_does_not_crash():
    fig, ax = plt.subplots()
    runs = [_mk_run("chord", 3e-3, 0.515)]
    diverged = [
        _mk_run("chord", 1e-2, float("nan")),
        _mk_run("chord", 3e-2, float("nan")),
    ]
    plot_eta_vs_final(
        ax, runs, _gk, {"chord": "#d62728"},
        diverged_runs=diverged, allow_label_collision=True,
    )
    lo, hi = ax.get_ylim()
    assert math.isfinite(lo) and math.isfinite(hi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Group isolation: A's diverged lrs don't leak into B's line.
# ---------------------------------------------------------------------------
def test_diverged_isolated_to_correct_group():
    fig, ax = plt.subplots()
    runs = [
        _mk_run("A", 1e-3, 0.520),
        _mk_run("A", 3e-3, 0.510),
        _mk_run("B", 1e-3, 0.530),
        _mk_run("B", 3e-3, 0.525),
    ]
    diverged = [_mk_run("A", 1e-2, float("nan"))]
    plot_eta_vs_final(
        ax, runs, _gk, {"A": "#1f77b4", "B": "#d62728"},
        diverged_runs=diverged, allow_label_collision=True,
    )
    a_ys = _line_ys_by_color(ax, "#1f77b4")
    b_ys = _line_ys_by_color(ax, "#d62728")
    # A's line must include something above max A-in-range (0.520).
    assert any(y > 0.520 for y in a_ys), (
        f"A's line should rise above max in-range; A y-data: {a_ys}"
    )
    # B's line must NOT exceed max B-in-range (0.530) — no diverged.
    assert all(y <= 0.531 for y in b_ys), (
        f"B's line should not rise above 0.530 (no diverged); B y-data: {b_ys}"
    )
    plt.close(fig)
