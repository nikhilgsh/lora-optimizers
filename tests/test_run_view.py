"""Public run-view boundary and provenance-sensitive consumers."""
from __future__ import annotations

import json

import pytest

from lora_playground.plotting.paper_view_semantics import (
    FACTORWISE_SLOT_BOUNDARY,
    filter_paper_precond_cohort,
)
from lora_playground.run_records import (
    RunRecord,
    project_run_semantics,
    run_view,
)
from lora_playground.workloads import (
    DatasetProvenanceConflict,
    Workload,
    resolve_record_dataset,
    workload_records,
)


def _record(cfg, *, group="group", history=()):
    return RunRecord.from_parsed(
        {"_log_filename": "log_0.out", **cfg},
        history,
        group=group,
        manifest=None,
    )


def _write_log(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def test_run_view_keeps_semantic_audit_and_raw_surfaces_separate():
    record = _record({
        "optimizer": "method",
        "data_dir": "/data/openmath_instruct_2_2m",
        "command": "python train.py --data_dir /data/openmath_instruct_2_2m",
        "git_commit": "abc123",
        "semantic_revisions": {"optimizer_impl": 2},
    })

    view = run_view(record)

    assert view.semantic_config["optimizer"] == "method"
    assert "command" not in view.semantic_config
    assert "git_commit" not in view.semantic_config
    assert view.audit_config["command"].startswith("python train.py")
    assert view.audit_config["git_commit"] == "abc123"
    assert view.raw_config["data_dir"].endswith("openmath_instruct_2_2m")
    assert view.semantic_revisions == {"optimizer_impl": 2}


def test_semantic_projection_is_narrow_immutable_and_preserves_provenance():
    record = _record({
        "optimizer": "method",
        "git_commit": "abc123",
    })

    projected = project_run_semantics(
        record,
        {"reviewed_view_revision": 2},
        projection_id="paper.reviewed.v1",
    )
    view = run_view(projected)

    assert view.semantic_config["reviewed_view_revision"] == 2
    assert view.audit_config["git_commit"] == "abc123"
    assert view.physical_id == record.physical_id
    with pytest.raises(ValueError, match="audit/private"):
        project_run_semantics(
            record, {"git_commit": "other"}, projection_id="bad"
        )
    with pytest.raises(ValueError, match="overwrite recorded"):
        project_run_semantics(
            record, {"optimizer": "other"}, projection_id="bad"
        )


def test_legacy_command_only_datasets_are_selected_by_workload_records(tmp_path):
    logs = tmp_path / "logs"
    for task, data_dir, dataset in (
        (0, "/data/openmath_instruct_2_2m_packed", "openmath"),
        (1, "/data/opc_sft_stage2_packed", "opc"),
    ):
        _write_log(
            logs / "group" / "run_info" / "logs" / f"log_{task}.out",
            [
                {
                    "event": "config",
                    "optimizer": "adamw",
                    "model_name": "model",
                    "lora_r": 64,
                    "lr": 1e-3,
                    "max_steps": 9000,
                    "data_pipeline_version": "packed_v1.1",
                    "dataset_name": "stale/magicoder",
                    "command": f"python train.py --data_dir {data_dir}",
                },
                {"event": "eval", "step": 9000, "eval_loss": 0.7},
            ],
        )
        workload = Workload(
            "model", dataset, 64, "Model", dataset, 9000, 0.001, True,
            "packed_v1.1",
        )
        selected = workload_records(workload, logs_root=str(logs))
        assert len(selected) == 1
        assert resolve_record_dataset(selected[0]) == dataset


def test_versioned_dataset_is_authoritative_and_legacy_conflict_fails_closed():
    versioned = _record({
        "run_schema_version": 1,
        "data_dir": "/data/openmath_instruct_2_2m_packed",
        "command": "python train.py --data_dir /data/opc_sft_stage2_packed",
        "dataset_name": "stale/magicoder",
    })
    assert resolve_record_dataset(versioned) == "openmath"

    legacy = _record({
        "data_dir": "/data/openmath_instruct_2_2m_packed",
        "command": "python train.py --data_dir /data/opc_sft_stage2_packed",
    })
    with pytest.raises(DatasetProvenanceConflict, match="conflicting datasets"):
        resolve_record_dataset(legacy)

    stale_only = _record({"dataset_name": "opc_sft_stage2"})
    assert resolve_record_dataset(stale_only) is None


def test_paper_policy_reads_audit_commit_from_run_record():
    legacy = _record({
        "optimizer": "method",
        "precond": "factorwise",
        "git_commit": "legacy-post-fix",
    })
    calls = []

    def ancestry(boundary, commit):
        calls.append((boundary, commit))
        return True

    kept, excluded = filter_paper_precond_cohort(
        [legacy], view_id="precond", is_ancestor=ancestry
    )

    assert kept == (legacy,)
    assert excluded == ()
    assert calls == [(FACTORWISE_SLOT_BOUNDARY, "legacy-post-fix")]


def test_paper_policy_recorded_revision_wins_for_run_record():
    versioned = _record({
        "run_schema_version": 1,
        "optimizer": "method",
        "precond": "one-sided",
        "git_commit": "irrelevant",
        "semantic_revisions": {"optimizer_impl": 2},
    })

    def forbidden(*_args):
        raise AssertionError("recorded revision must bypass ancestry")

    kept, excluded = filter_paper_precond_cohort(
        [versioned], view_id="precond_beta2", is_ancestor=forbidden
    )

    assert kept == (versioned,)
    assert excluded == ()
