"""Guardrails for partial (in-flight) leaderboard rendering.

Regression for the class of bug where in-flight runs (only an early eval logged)
were treated as "completed at step N": they leaked onto the final-vs-lr panel,
collapsed the trajectory x-axis to a degenerate window, and got mislabelled as
diverged. The invariants below must hold for any partial-data render.
"""
import matplotlib
matplotlib.use("Agg")

from lora_playground.plotting.figures import compare_variants_figure

HORIZON = 9000


def _run(label, lr, steps_losses, max_steps_cfg=HORIZON):
    cfg = {"label": label, "lr": lr, "max_steps": max_steps_cfg, "optimizer": "adamw"}
    hist = [{"step": s, "eval_loss": l} for s, l in steps_losses]
    return cfg, hist


def _panel(runs):
    return compare_variants_figure(
        variants={"AdamW": {}}, common_where={},
        ref_label="AdamW", target_label="AdamW",
        max_steps=HORIZON, allow_partial=True,
        prefetched_runs=runs, variant_key=lambda c: c["label"],
    )


def test_inflight_runs_are_not_treated_as_final():
    # Two lrs, only the first eval (step 250) logged; one has a much higher loss
    # (would be spuriously "diverged" if treated as final).
    runs = [_run("AdamW", 1e-4, [(250, 0.61)]),
            _run("AdamW", 3e-4, [(250, 0.95)])]
    fig, _tdf, sdf = _panel(runs)
    # Nothing reached the horizon -> final-vs-lr panel + summary are empty.
    assert len(sdf) == 0, "in-flight runs leaked onto the final-vs-lr panel"
    assert len(fig.axes[0].collections) == 0, "final panel drew partial points"


def test_trajectory_xaxis_spans_horizon_for_partial():
    runs = [_run("AdamW", 1e-4, [(250, 0.61)])]
    fig, _tdf, _sdf = _panel(runs)
    x0, x1 = fig.axes[-1].get_xlim()
    # x-axis spans the full horizon (not a degenerate partial window). A small
    # right margin (so a final marker at exactly max_steps isn't clipped by the
    # spine) is fine — assert the upper bound reaches the horizon, within ~5%.
    assert round(x0) == 0 and HORIZON <= x1 <= HORIZON * 1.05, (
        f"trajectory x-axis collapsed to ({x0:.0f}, {x1:.0f}) instead of spanning (0, {HORIZON})")


def test_completed_run_does_appear_as_final():
    # A run that reaches the horizon (within slack) must still be final.
    complete = _run("AdamW", 1e-4, [(s, 0.6) for s in range(250, HORIZON + 1, 250)])
    partial = _run("AdamW", 3e-4, [(250, 0.95)])
    _fig, _tdf, sdf = _panel([complete, partial])
    assert len(sdf) >= 1, "a horizon-reaching run was dropped from the final panel"


def test_one_epoch_run_within_slack_counts_as_final():
    # Tulu-style ~8970-step one-epoch run must count as final (completion slack).
    near = _run("AdamW", 1e-4, [(s, 0.6) for s in range(250, 8971, 250)])
    _fig, _tdf, sdf = _panel([near])
    assert len(sdf) >= 1, "an ~8970-step one-epoch run was wrongly treated as partial"


def test_aborted_highlr_does_not_hide_healthy_partial_trajectory():
    # Regression: when a variant's ONLY completed run is a NaN-aborted high-lr,
    # the "final" bucket is non-empty-but-all-diverged. The trajectory panel must
    # still draw the variant's best HEALTHY in-flight curve (finite partial),
    # not the blown-up aborted one. (Observed live: Bengali AdamW η=1e-3 aborted
    # at step 6500 while η=1e-4 was healthy at step 8000 — the panel plotted the
    # NaN arm and hid the good partial.)
    import math
    healthy = _run("AdamW", 1e-4, [(s, 0.6) for s in range(250, 8001, 250)])
    ab_cfg, ab_hist = _run(
        "AdamW", 1e-3,
        [(s, 2.5) for s in range(250, 6500, 250)] + [(6500, float("nan"))])
    ab_cfg["_aborted"] = {"reason": "nan_eval"}  # marks it completed-but-diverged
    fig, _tdf, _sdf = _panel([healthy, (ab_cfg, ab_hist)])
    curves = [ln for ln in fig.axes[-1].get_lines()
              if not ln.get_label().startswith("_")]
    assert len(curves) == 1, "expected exactly one best-lr trajectory for AdamW"
    lab = curves[0].get_label()
    # Match the lr VALUE, not the prefix: `render_comparison` spells the
    # trajectory legend's learning rate "$\eta$=..." (it used to be "lr=..."),
    # and which arm was plotted is what this regression is about.
    assert "=0.0001" in lab, (
        f"trajectory used the aborted high-lr instead of the healthy partial: {lab}")
    ys = list(curves[0].get_ydata())
    assert ys and all(math.isfinite(y) for y in ys), "healthy trajectory contains NaN"
    assert max(ys) < 1.0, "trajectory plotted the diverged ~2.5 curve, not the healthy one"
