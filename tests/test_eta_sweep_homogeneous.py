"""Tests that _eta_sweep_points raises on cross-regime data pollution.

The specific failure mode this guards against:
    A notebook loads `ref_runs = load_runs(where={'optimizer': 'adamw',
    'data_pipeline_version': 'packed_v1'})` — without pinning max_steps.
    A new phase_L sweep at max_steps=9000 lands in logs/. The notebook's
    `_eta_sweep_points` aggregates all adamw runs at each lr regardless
    of max_steps, mixing the canonical 4k panel reference with
    incompatible 9k-step data. Error bars blow up, the reference curve
    zigzags wildly. Silent data corruption with no traceback.

This is now a HARD error — `_eta_sweep_points` raises ValueError naming
the mismatched axis and log groups, telling the caller exactly which
filter to add.
"""
import pytest

from lora_playground.plotting.overlays import _eta_sweep_points


def _run(optimizer, lr, final, max_steps=4000, lora_r=64, log_group="g"):
    cfg = {
        "optimizer": optimizer, "lr": float(lr), "max_steps": max_steps,
        "lora_r": int(lora_r), "log_group": log_group, "seed": 0,
    }
    return (cfg, [{"step": max_steps, "eval_loss": final}])


def test_uniform_buckets_aggregate_cleanly():
    """Sanity: when all buckets are homogeneous, aggregation works."""
    runs = [
        _run("adamw", 1e-4, 0.520),
        _run("adamw", 1e-4, 0.522),
        _run("adamw", 3e-4, 0.540),
    ]
    points = _eta_sweep_points(runs, "adamw")
    assert len(points) == 2
    lr_to_mean = {lr: m for lr, m, _, _ in points}
    assert abs(lr_to_mean[1e-4] - 0.521) < 1e-6
    assert abs(lr_to_mean[3e-4] - 0.540) < 1e-6


def test_mixed_max_steps_raises():
    """Pollution scenario: a phase_L run at max_steps=9000 lands in the
    same bucket as the canonical max_steps=4000 reference. Must raise."""
    runs = [
        _run("adamw", 1e-4, 0.520, max_steps=4000, log_group="canonical_4k"),
        _run("adamw", 1e-4, 0.795, max_steps=9000, log_group="phase_L"),
    ]
    with pytest.raises(ValueError) as ei:
        _eta_sweep_points(runs, "adamw")
    msg = str(ei.value)
    # Error must name the offending axis, the lr, and the log_groups so
    # the caller can immediately see what to filter on.
    assert "max_steps" in msg
    assert "1e-04" in msg or "1e-4" in msg.lower() or "0.0001" in msg
    assert "phase_L" in msg
    assert "canonical_4k" in msg


def test_mixed_lora_r_raises():
    """Same logic for lora_r — runs at the same lr but different r in
    the same reference set is silent data pollution."""
    runs = [
        _run("adamw", 1e-4, 0.520, lora_r=16, log_group="r16_sweep"),
        _run("adamw", 1e-4, 0.535, lora_r=64, log_group="r64_sweep"),
    ]
    with pytest.raises(ValueError) as ei:
        _eta_sweep_points(runs, "adamw")
    assert "lora_r" in str(ei.value)


def test_raise_on_mixed_can_be_disabled():
    """For intentional cross-regime comparison: raise_on_mixed=False
    downgrades to a stderr warning so the caller can override."""
    runs = [
        _run("adamw", 1e-4, 0.520, max_steps=4000),
        _run("adamw", 1e-4, 0.795, max_steps=9000),
    ]
    # Should not raise.
    points = _eta_sweep_points(runs, "adamw", raise_on_mixed=False)
    assert len(points) == 1


def test_homogeneous_axes_can_be_restricted():
    """`homogeneous_axes=()` reproduces the old behavior (no homogeneity
    check) — escape hatch for legacy notebooks during migration."""
    runs = [
        _run("adamw", 1e-4, 0.520, max_steps=4000),
        _run("adamw", 1e-4, 0.795, max_steps=9000),
    ]
    points = _eta_sweep_points(runs, "adamw", homogeneous_axes=())
    assert len(points) == 1


def test_different_optimizers_dont_count_as_mixed():
    """Sanity: only homogeneity within the SELECTED optimizer matters.
    Other optimizers in the reference set don't pollute the buckets."""
    runs = [
        _run("adamw", 1e-4, 0.520, max_steps=4000),
        _run("muon", 1e-4, 0.799, max_steps=9000),  # different optimizer
    ]
    # Only adamw is selected — muon's max_steps difference shouldn't fire.
    points = _eta_sweep_points(runs, "adamw")
    assert len(points) == 1
