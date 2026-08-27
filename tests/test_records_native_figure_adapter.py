"""Records-native contract for the high-level comparison figure adapter."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from lora_playground.comparison import (
    SemanticRevisionConflictError,
    VariantSpec,
    build_comparison,
)
from lora_playground.plotting.figures import compare_variants_figure
from lora_playground.run_records import RunRecord


HORIZON = 1000


def _record(
    run_id: str,
    optimizer: str,
    lr: float,
    loss: float,
    *,
    step: int = HORIZON,
    measurement_revision: str = "eval.v1",
    data_pipeline_revision: str = "packed.v1",
) -> RunRecord:
    raw = {
        "run_id": run_id,
        "_log_filename": f"{run_id}.jsonl",
        "run_schema_version": 2,
        "optimizer": optimizer,
        "lr": lr,
        "max_steps": HORIZON,
        "semantic_revisions": {
            "optimizer_impl": f"{optimizer}.v1",
            "measurement": measurement_revision,
            "data_pipeline": data_pipeline_revision,
        },
    }
    return RunRecord.from_parsed(
        raw,
        [{"step": step, "eval_loss": loss}],
        group="records-native-test",
        manifest=None,
    )


def _specs() -> tuple[VariantSpec, ...]:
    return (
        VariantSpec("baseline.v1", "Reference", {"optimizer": "adamw"}),
        VariantSpec("candidate.v1", "Candidate", {"optimizer": "candidate"}),
        VariantSpec("partial.v1", "Partial", {"optimizer": "partial"}),
    )


def _trajectory(fig, variant_id: str):
    matches = [
        line for line in fig.axes[1].get_lines()
        if line.get_gid() == f"trajectory:{variant_id}"
    ]
    assert len(matches) == 1
    return matches[0]


def test_direct_records_use_explicit_variant_ids_without_legacy_loading(monkeypatch):
    import lora_playground.loader as loader

    monkeypatch.setattr(
        loader,
        "load_runs",
        lambda *args, **kwargs: pytest.fail("records path called load_runs"),
    )
    records = (
        _record("base", "adamw", 1e-3, 0.60),
        _record("candidate-best", "candidate", 1e-3, 0.55),
        _record("candidate-other", "candidate", 3e-3, 0.58),
        _record("partial", "partial", 1e-3, 0.53, step=600),
    )

    fig, table, summary = compare_variants_figure(
        records=records,
        variant_specs=_specs(),
        reference_id="baseline.v1",
        target_id="baseline.v1",
        max_steps=HORIZON,
        completion_slack=0,
        allow_partial=True,
        auto_ylim=False,
    )

    assert table.columns.tolist() == ["Reference", "Candidate", "Partial"]
    assert table.loc[1e-3, "Reference"] == pytest.approx(0.60)
    assert table.loc[1e-3, "Candidate"] == pytest.approx(0.55)
    assert summary["variant"].tolist() == ["Reference", "Candidate"]
    assert _trajectory(fig, "candidate.v1").get_ydata()[-1] == pytest.approx(0.55)
    partial = _trajectory(fig, "partial.v1")
    assert partial.get_xdata()[-1] == 600
    assert "partial @600" in partial.get_label()
    plt.close(fig)


def test_prebuilt_comparison_is_rendered_without_reassignment(monkeypatch):
    result = build_comparison(
        (_record("base", "adamw", 1e-3, 0.60),),
        (_specs()[0],),
        HORIZON,
        completion_slack=0,
    )

    import lora_playground.comparison as comparison_module
    import lora_playground.loader as loader

    monkeypatch.setattr(
        comparison_module,
        "build_comparison",
        lambda *args, **kwargs: pytest.fail("comparison path rebuilt result"),
    )
    monkeypatch.setattr(
        loader,
        "load_runs",
        lambda *args, **kwargs: pytest.fail("comparison path called load_runs"),
    )

    fig, table, summary = compare_variants_figure(
        comparison=result,
        reference_id="baseline.v1",
        max_steps=HORIZON,
        auto_ylim=False,
    )

    assert table.loc[1e-3, "Reference"] == pytest.approx(0.60)
    assert summary.loc[0, "best_lr"] == pytest.approx(1e-3)
    assert _trajectory(fig, "baseline.v1").get_ydata()[-1] == pytest.approx(0.60)
    plt.close(fig)


def test_direct_records_fail_closed_on_mixed_measurement_revisions():
    records = (
        _record("base", "adamw", 1e-3, 0.60),
        _record(
            "candidate",
            "candidate",
            1e-3,
            0.55,
            measurement_revision="eval.v2",
        ),
    )

    with pytest.raises(SemanticRevisionConflictError, match="comparison"):
        compare_variants_figure(
            records=records,
            variant_specs=_specs()[:2],
            reference_id="baseline.v1",
            max_steps=HORIZON,
            completion_slack=0,
        )


def test_direct_records_require_explicit_variant_specs():
    with pytest.raises(ValueError, match="requires explicit variant_specs"):
        compare_variants_figure(
            records=(_record("base", "adamw", 1e-3, 0.60),),
            reference_id="baseline.v1",
            max_steps=HORIZON,
        )
