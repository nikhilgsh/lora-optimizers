"""Leaderboard consumption of the shared comparison reduction."""
from __future__ import annotations

import math

import pytest

from lora_playground.comparison import VariantSpec, build_comparison
from lora_playground.leaderboard import (
    UncertifiedBaselineError,
    leaderboard_rows_from_comparison,
)


def _run(optimizer, lr, losses, *, run_id, seed=0):
    return (
        {"optimizer": optimizer, "lr": lr, "run_id": run_id, "seed": seed},
        [{"step": step, "eval_loss": loss} for step, loss in losses],
    )


def test_rows_consume_replicate_mean_and_core_best_lr_by_stable_id():
    runs = [
        _run("adamw", 3e-5, [(500, 1.00), (1000, 0.90)], run_id="adam-low"),
        _run("adamw", 1e-4, [(500, 0.95), (1000, 0.80)], run_id="adam"),
        _run("adamw", 3e-4, [(500, 1.00), (1000, 0.90)], run_id="adam-high"),
        # The lucky first seed would make 3e-4 look best on its own, but its
        # replicate mean is 0.90, so the comparison core selects 1e-3 at 0.85.
        _run("method", 3e-4, [(500, 0.80), (1000, 0.80)],
             run_id="method-s0", seed=0),
        _run("method", 3e-4, [(500, 1.00), (1000, 1.00)],
             run_id="method-s1", seed=1),
        _run("method", 1e-4, [(500, 0.90), (1000, 0.88)],
             run_id="method-low"),
        _run("method", 1e-3, [(500, 0.80), (1000, 0.85)],
             run_id="method-best"),
        _run("method", 3e-3, [(500, 0.95), (1000, 0.92)],
             run_id="method-high"),
    ]
    result = build_comparison(
        runs,
        [
            VariantSpec("baseline", "AdamW display", {"optimizer": "adamw"}),
            VariantSpec("candidate", "method display", {"optimizer": "method"}),
        ],
        horizon=1000,
        completion_slack=0,
    )

    rows, target = leaderboard_rows_from_comparison(
        result, horizon=1000, baseline_id="baseline"
    )

    assert target == pytest.approx(0.80)
    method = next(row for row in rows if row["variant"] == "method display")
    assert method["best_lr"] == pytest.approx(1e-3)
    assert method["final_at_best"] == pytest.approx(0.85)
    assert method["n_lrs"] == 4
    assert method["frac_best_lr"] == pytest.approx(0.5)
    assert not math.isnan(method["frac_lr_avg"])
    assert result.completed["candidate"][3e-4].n_replicates == 2


def test_missing_baseline_id_fails_closed():
    result = build_comparison(
        [_run("method", 1e-3, [(1000, 0.7)], run_id="method")],
        [VariantSpec("candidate", "method", {"optimizer": "method"})],
        horizon=1000,
        completion_slack=0,
    )

    with pytest.raises(UncertifiedBaselineError, match="no completed finite"):
        leaderboard_rows_from_comparison(
            result, horizon=1000, baseline_id="not-present"
        )


def test_boundary_pinned_baseline_fails_closed():
    result = build_comparison(
        [
            _run("adamw", 1e-4, [(1000, 0.8)], run_id="low"),
            _run("adamw", 3e-4, [(1000, 0.9)], run_id="high"),
        ],
        [VariantSpec("baseline", "AdamW", {"optimizer": "adamw"})],
        horizon=1000,
        completion_slack=0,
    )

    with pytest.raises(UncertifiedBaselineError, match="boundary-pinned"):
        leaderboard_rows_from_comparison(
            result, horizon=1000, baseline_id="baseline"
        )
