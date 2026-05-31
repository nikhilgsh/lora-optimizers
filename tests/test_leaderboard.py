"""Unit tests for the speed-to-target leaderboard metric."""
import math

from lora_playground.leaderboard import (
    reach_fraction, labeled_completed_runs, merge_labeled, leaderboard_rows,
    performance_profile,
)


def _hist(steps_losses):
    return [{"step": s, "eval_loss": l} for s, l in steps_losses]


def test_reach_fraction_basic():
    h = _hist([(100, 1.0), (200, 0.5), (300, 0.4)])
    # target 0.5 first met at step 200 → 200/400
    assert reach_fraction(h, 0.5, 400) == 200 / 400
    # target 0.45 first met at step 300
    assert reach_fraction(h, 0.45, 400) == 300 / 400


def test_reach_fraction_never():
    h = _hist([(100, 1.0), (200, 0.9)])
    assert math.isnan(reach_fraction(h, 0.5, 400))
    assert math.isnan(reach_fraction(h, 0.5, 400)) and math.isnan(reach_fraction(h, float("nan"), 400))


def test_reach_fraction_unsorted_input():
    h = _hist([(300, 0.4), (100, 1.0), (200, 0.5)])
    assert reach_fraction(h, 0.5, 400) == 200 / 400


def _cfg(opt, lr, **extra):
    c = {"optimizer": opt, "lr": lr}
    c.update(extra)
    return c


def _vk(cfg):
    return "AdamW" if cfg["optimizer"] == "adamw" else "method"


def test_labeled_completes_and_dedups():
    horizon = 1000
    runs = [
        # completed adamw
        (_cfg("adamw", 1e-4), _hist([(500, 0.9), (1000, 0.8)])),
        # incomplete run dropped (last step far below horizon)
        (_cfg("adamw", 1e-4), _hist([(100, 0.95)])),
        # method, two runs same lr → keep longest
        (_cfg("m", 1e-3), _hist([(1000, 0.7)])),
        (_cfg("m", 1e-3), _hist([(500, 0.75), (1000, 0.7), (1000, 0.7)])),
    ]
    lab = labeled_completed_runs(runs, _vk, horizon=horizon)
    assert set(lab) == {"AdamW", "method"}
    assert set(lab["AdamW"]) == {1e-4}            # incomplete dropped
    assert lab["method"][1e-3][0] == 0.7          # final loss


def test_leaderboard_rows_target_and_pinning():
    horizon = 1000
    # AdamW best final = 0.80 (target). method reaches 0.80 at step 500 → 0.5.
    labeled = {
        "AdamW": {1e-4: (0.80, _hist([(500, 0.95), (1000, 0.80)]))},
        "method": {
            3e-4: (0.78, _hist([(500, 0.80), (1000, 0.78)])),  # best lr, reaches at 500
            1e-4: (0.82, _hist([(1000, 0.82)])),               # neighbor never reaches → clamp 1.0
            1e-3: (0.79, _hist([(1000, 0.79)])),               # neighbor reaches at 1000 → 1.0
        },
    }
    rows, target = leaderboard_rows(labeled, horizon=horizon)
    assert target == 0.80
    method = next(r for r in rows if r["variant"] == "method")
    assert method["best_lr"] == 3e-4
    assert method["frac_best_lr"] == 0.5
    # avg over {1e-4 (clamp 1.0), 3e-4 (0.5), 1e-3 (1.0)} = 2.5/3
    assert abs(method["frac_lr_avg"] - (1.0 + 0.5 + 1.0) / 3) < 1e-9


def test_leaderboard_rows_pinned_is_nan():
    horizon = 1000
    labeled = {
        "AdamW": {1e-4: (0.80, _hist([(1000, 0.80)]))},
        # only two lrs, best at the high boundary → no 3x-high → pinned → NaN avg
        "method": {
            3e-4: (0.79, _hist([(1000, 0.79)])),
            1e-3: (0.78, _hist([(500, 0.80), (1000, 0.78)])),  # best, at boundary
        },
    }
    rows, _ = leaderboard_rows(labeled, horizon=horizon)
    method = next(r for r in rows if r["variant"] == "method")
    assert method["best_lr"] == 1e-3
    assert math.isnan(method["frac_lr_avg"])


def test_performance_profile_ranking_and_coverage():
    # 3 workloads. "fast" is fastest on all it ran; "slow" always worse;
    # "narrow" only ran 1 workload (and is fastest there).
    pm = {
        "fast":   {"w1": 0.5, "w2": 0.6, "w3": 0.5},
        "slow":   {"w1": 1.0, "w2": 1.0, "w3": 1.0},
        "narrow": {"w1": 0.4},
        "absent": {"w1": float("nan")},   # NaN dropped → 0 coverage → excluded
    }
    rows = performance_profile(pm, max_tau=4.0)
    by = {r["variant"]: r for r in rows}
    assert "absent" not in by                       # NaN-only excluded
    # best on w1 is narrow (0.4); fast's ratio there is 0.5/0.4 = 1.25
    assert abs(by["fast"]["ratios"]["w1"] - 1.25) < 1e-9
    # coverage-first sort: fast (3) and slow (3) rank above narrow (1)
    assert rows[0]["coverage"] == 3 and rows[-1]["variant"] == "narrow"
    # among full-coverage, fast has the higher robustness score than slow
    assert by["fast"]["robustness_score"] > by["slow"]["robustness_score"]
    # score is a normalised area in [0, 1]
    assert 0.0 <= by["slow"]["robustness_score"] <= 1.0


def test_merge_labeled_keeps_longest():
    a = {"m": {1e-3: (0.7, _hist([(1000, 0.7)]))}}
    b = {"m": {1e-3: (0.7, _hist([(500, 0.75), (1000, 0.7)]))}}
    merged = merge_labeled(a, b)
    assert len(merged["m"][1e-3][1]) == 2
