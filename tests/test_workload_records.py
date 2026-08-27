"""Records-native workload discovery without live-log dependencies."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lora_playground.comparison import (
    VariantSpec,
    _comparison_input,
    build_comparison,
)
from lora_playground.plotting.render import render_comparison
from lora_playground.run_lineage import MergedRunLineage
from lora_playground.workloads import Workload, resolve_dataset, workload_records


def _write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def test_resolve_dataset_prefers_logged_data_dir_without_command_parsing():
    assert resolve_dataset({
        "data_dir": "/data/openmath_instruct_2_2m_packed_seq2048",
        "command": "python train.py --data_dir /wrong/opc_sft_stage2",
    }) == "openmath"


def test_workload_records_uses_semantics_and_explicit_cross_group_lineage(
    tmp_path,
):
    logs = tmp_path / "logs"
    common = {
        "event": "config",
        "run_schema_version": 1,
        "checkpoint_identity": "logical/task_0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "adamw",
        "model_name": "model",
        "lora_r": 64,
        "lr": 1e-3,
        "max_steps": 9000,
        "data_pipeline_version": "packed_v1.1",
        "data_dir": "/data/openmath_instruct_2_2m_packed_seq2048",
    }
    _write(logs / "old-group" / "run_info" / "logs" / "log_0.out", [
        {**common, "attempt_id": "attempt-a",
         "resume_parent_attempt_id": None},
        {"event": "eval", "step": 100, "eval_loss": 0.8, "lr": 1e-3},
    ])
    _write(logs / "new-group" / "run_info" / "logs" / "log_7.out", [
        {**common, "attempt_id": "attempt-b",
         "resume_parent_attempt_id": None},
        {"event": "resume", "resume_parent_attempt_id": "attempt-a",
         "checkpoint_identity": "logical/task_0"},
        {"event": "eval", "step": 9000, "eval_loss": 0.7, "lr": 1e-3},
    ])
    workload = Workload(
        "model", "openmath", 64, "Model", "OpenMath", 9000, 0.001, True,
        "packed_v1.1",
    )

    records = workload_records(workload, logs_root=str(logs))

    assert len(records) == 1
    assert isinstance(records[0], MergedRunLineage)
    assert records[0].groups == ("old-group", "new-group")
    cfg, history = _comparison_input(records[0], 0)
    assert cfg["log_group"] == "new-group"
    assert [event["step"] for event in history] == [100, 9000]


def test_workload_records_compose_directly_with_explicit_ids_and_renderer(
    tmp_path,
):
    logs = tmp_path / "logs"
    common = {
        "event": "config",
        "model_name": "model",
        "lora_r": 64,
        "lr": 1e-3,
        "max_steps": 9000,
        "data_pipeline_version": "packed_v1.1",
        "data_dir": "/data/openmath_instruct_2_2m_packed_seq2048",
    }
    for task, optimizer, loss in (
        (0, "adamw", 0.8),
        (1, "method-b", 0.7),
    ):
        _write(logs / "group" / "run_info" / "logs" / f"log_{task}.out", [
            {**common, "optimizer": optimizer},
            {"event": "eval", "step": 9000,
             "eval_loss": loss, "lr": 1e-3},
        ])
    workload = Workload(
        "model", "openmath", 64, "Model", "OpenMath", 9000, 0.001, True,
        "packed_v1.1",
    )
    records = workload_records(workload, logs_root=str(logs))
    specs = (
        VariantSpec(
            "baseline.adamw.v1",
            "AdamW display",
            {"optimizer": "adamw"},
        ),
        VariantSpec(
            "candidate.method_b.v1",
            "Method B display",
            {"optimizer": "method-b"},
        ),
    )

    result = build_comparison(records, specs, horizon=9000)
    figure, table, summary = render_comparison(
        result,
        reference_id="baseline.adamw.v1",
        horizon=9000,
        auto_ylim=False,
    )

    assert set(result.completed) == {
        "baseline.adamw.v1", "candidate.method_b.v1",
    }
    assert result.best_completed["candidate.method_b.v1"].final_loss == 0.7
    assert list(table.columns) == ["AdamW display", "Method B display"]
    assert set(summary["variant"]) == {"AdamW display", "Method B display"}
    plt.close(figure)
