"""Pure comparison construction: assignment, replicate means, and partials."""
from __future__ import annotations

import math

import pytest

from lora_playground.comparison import (
    AmbiguousVariantError,
    VariantSpec,
    build_comparison,
)


def _run(
    optimizer: str,
    lr: float,
    losses: list[tuple[int, float]],
    *,
    run_id: str,
    seed: int = 0,
    **cfg_extra,
):
    cfg = {
        "optimizer": optimizer,
        "lr": lr,
        "seed": seed,
        "run_id": run_id,
        **cfg_extra,
    }
    history = [
        {"step": step, "eval_loss": loss, "lr": lr}
        for step, loss in losses
    ]
    return cfg, history


def test_exact_assignment_records_unmatched_and_empty_history():
    runs = [
        _run("adamw", 1e-3, [(1000, 0.8)], run_id="adam"),
        _run("muon", 1e-2, [(1000, 0.7)], run_id="muon"),
        ({"optimizer": "adamw", "lr": 3e-3, "run_id": "empty"}, []),
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {"optimizer": "adamw"})],
        horizon=1000,
        completion_slack=0,
    )

    assert set(result.completed) == {"adam"}
    assert result.completed["adam"][1e-3].run_ids == ("adam",)
    assert result.unmatched_run_ids == ("muon",)
    assert result.empty_history_run_ids == ("empty",)


def test_all_predicates_are_evaluated_and_ambiguity_fails_closed():
    runs = [_run("adamw", 1e-3, [(1000, 0.8)], run_id="r0")]
    variants = [
        VariantSpec("broad", "broad", {"optimizer": "adamw"}),
        VariantSpec("callable", "callable", lambda cfg: cfg["lr"] == 1e-3),
    ]
    with pytest.raises(AmbiguousVariantError) as exc_info:
        build_comparison(runs, variants, horizon=1000, completion_slack=0)

    assert exc_info.value.ambiguities == (("r0", ("broad", "callable")),)
    assert "broad" in str(exc_info.value) and "callable" in str(exc_info.value)


def test_mapping_predicates_are_data_layer_only_and_keep_arm_semantics():
    runs = [
        _run(
            "adamw", 1e-3, [(1000, 0.8)], run_id="r0",
            mode="fast", target_modules=["q_proj", "v_proj"],
        )
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {
            "optimizer": {"adamw", "sgd"},
            "mode": lambda value: value.startswith("f"),
            "target_modules": ["q_proj", "v_proj"],
        })],
        horizon=1000,
        completion_slack=0,
    )

    assert result.completed["adam"][1e-3].run_ids == ("r0",)


def test_completed_lr_uses_replicate_mean_and_best_uses_that_mean():
    runs = [
        _run("adamw", 1e-3, [(500, 1.0), (1000, 0.80)], run_id="s0", seed=0),
        _run("adamw", 1e-3, [(500, 1.2), (1000, 1.00)], run_id="s1", seed=1),
        _run("adamw", 3e-3, [(500, 1.0), (1000, 0.85)], run_id="s2", seed=0),
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {"optimizer": "adamw"})],
        horizon=1000,
        completion_slack=0,
    )

    mean_curve = result.completed["adam"][1e-3]
    assert mean_curve.final_loss == pytest.approx(0.90)
    assert mean_curve.n_replicates == 2
    assert [event["eval_loss"] for event in mean_curve.history] == pytest.approx(
        [1.1, 0.9]
    )
    assert all(event["n_seeds"] == 2 for event in mean_curve.history)
    # The lucky 0.80 seed at 1e-3 must not beat the 0.85 replicate mean at 3e-3.
    assert result.best_completed["adam"].lr == 3e-3
    assert result.best_completed["adam"].final_loss == pytest.approx(0.85)


def test_longer_completed_trajectory_supersedes_shorter_completed_run():
    runs = [
        _run("adamw", 1e-3, [(900, 0.9)], run_id="short"),
        _run("adamw", 1e-3, [(900, 0.9), (1000, 0.8)], run_id="long"),
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {"optimizer": "adamw"})],
        horizon=900,
        completion_slack=0,
    )

    curve = result.completed["adam"][1e-3]
    assert curve.last_step == 1000
    assert curve.final_loss == pytest.approx(0.8)
    assert curve.n_replicates == 1
    assert curve.run_ids == ("long",)


def test_partial_keeps_first_most_progressed_representative_per_lr():
    runs = [
        _run("adamw", 1e-3, [(200, 1.2), (500, 0.9)], run_id="first"),
        _run("adamw", 1e-3, [(200, 1.1), (500, 0.8)], run_id="same-step"),
        _run("adamw", 1e-3, [(200, 1.0), (400, 0.7)], run_id="shorter"),
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {"optimizer": "adamw"})],
        horizon=1000,
        completion_slack=0,
    )

    assert not result.completed["adam"]
    partial = result.partials["adam"][1e-3]
    assert not partial.completed
    assert partial.last_step == 500
    assert partial.final_loss == pytest.approx(0.9)
    assert partial.run_ids == ("first",)
    assert result.best_partial["adam"] is partial


def test_aborted_short_run_is_completed_and_nonfinite_best_is_visible():
    runs = [
        _run(
            "adamw", 1e-3, [(100, math.nan)], run_id="aborted",
            _aborted={"event": "abort_on_nan_eval"},
        )
    ]
    result = build_comparison(
        runs,
        [VariantSpec("adam", "AdamW", {"optimizer": "adamw"})],
        horizon=1000,
        completion_slack=0,
    )

    curve = result.completed["adam"][1e-3]
    assert curve.completed
    assert curve.last_step == 100
    assert math.isnan(curve.final_loss)
    assert result.best_completed["adam"] is curve


def test_label_and_style_changes_do_not_change_identity_or_statistics():
    runs = [_run("adamw", 1e-3, [(1000, 0.8)], run_id="r0")]
    spec_a = VariantSpec(
        "stable-id", "old label", {"optimizer": "adamw"}, "red"
    )
    spec_b = VariantSpec(
        "stable-id", "new label", {"optimizer": "adamw"}, "blue"
    )
    assert spec_a == spec_b
    assert hash(spec_a) == hash(spec_b)
    a = build_comparison(
        runs,
        [spec_a],
        horizon=1000,
        completion_slack=0,
    )
    b = build_comparison(
        runs,
        [spec_b],
        horizon=1000,
        completion_slack=0,
    )

    curve_a = a.completed["stable-id"][1e-3]
    curve_b = b.completed["stable-id"][1e-3]
    assert curve_a.variant_id == curve_b.variant_id == "stable-id"
    assert curve_a.final_loss == curve_b.final_loss == pytest.approx(0.8)
    assert curve_a.run_ids == curve_b.run_ids == ("r0",)


def test_duplicate_variant_ids_and_invalid_bounds_raise():
    duplicate = [
        VariantSpec("x", "one", lambda _cfg: True),
        VariantSpec("x", "two", lambda _cfg: False),
    ]
    with pytest.raises(ValueError, match="duplicate VariantSpec.id"):
        build_comparison([], duplicate, horizon=1000)
    with pytest.raises(ValueError, match="horizon must be positive"):
        build_comparison([], [], horizon=0)
    with pytest.raises(ValueError, match="completion_slack"):
        build_comparison([], [], horizon=1000, completion_slack=-1)
