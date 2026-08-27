from __future__ import annotations

import json

import pytest

from lora_playground.loader import load_records
from lora_playground.run_catalog import RunCatalog
from lora_playground.run_lineage import MergedRunLineage
from lora_playground.run_records import RunRecord


def _write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def test_load_records_is_filesystem_led_and_explicit_lineage_only(tmp_path):
    logs = tmp_path / "logs"
    common = {
        "event": "config",
        "run_schema_version": 1,
        "checkpoint_identity": "group/task_0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "adamw",
        "lr": 1e-3,
        "data_pipeline_version": "packed_v1.1",
    }
    log_dir = logs / "group" / "run_info" / "logs"
    _write(log_dir / "log_0.out.resume_0", [
        {**common, "attempt_id": "attempt-a",
         "resume_parent_attempt_id": None},
        {"event": "eval", "step": 100, "eval_loss": 0.8, "lr": 1e-3},
    ])
    _write(log_dir / "log_0.out", [
        {**common, "attempt_id": "attempt-b",
         "resume_parent_attempt_id": None},
        {"event": "resume", "resume_parent_attempt_id": "attempt-a",
         "checkpoint_identity": "group/task_0"},
        {"event": "eval", "step": 200, "eval_loss": 0.7, "lr": 1e-3},
    ])
    # No manifest: physical discovery must still see both attempts.

    raw = load_records(
        equals={"optimizer": "adamw"},
        logs_root=str(logs),
        resolve_lineages=False,
    )
    resolved = load_records(
        equals={"optimizer": "adamw"},
        logs_root=str(logs),
    )

    assert len(raw) == 2
    assert all(isinstance(run, RunRecord) for run in raw)
    assert len(resolved) == 1
    assert isinstance(resolved[0], MergedRunLineage)
    assert resolved[0].attempt_ids == ("attempt-a", "attempt-b")


def test_load_records_never_stitches_legacy_disjoint_steps(tmp_path):
    logs = tmp_path / "logs"
    log_dir = logs / "legacy" / "run_info" / "logs"
    _write(log_dir / "log_0.out.resume_0", [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3},
        {"event": "eval", "step": 100, "eval_loss": 0.8, "lr": 1e-3},
    ])
    _write(log_dir / "log_0.out", [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3},
        {"event": "eval", "step": 200, "eval_loss": 0.7, "lr": 1e-3},
    ])

    runs = load_records(equals={"optimizer": "adamw"}, logs_root=str(logs))

    assert len(runs) == 2
    assert all(isinstance(run, RunRecord) for run in runs)


def test_load_records_preserves_absence_and_never_gates_source_provenance(
    tmp_path,
):
    logs = tmp_path / "logs"
    log_dir = logs / "group" / "run_info" / "logs"
    _write(log_dir / "log_0.out", [
        {
            "event": "config",
            "optimizer": "adamw",
            "command": "python train_lora.py --optimizer adamw",
            "execution_source_sha": "not-present-in-any-commit",
        },
        {"event": "eval", "step": 10, "eval_loss": 0.9, "lr": 3e-4},
    ])
    # Corrupt annotation remains an issue, never an admission decision.
    meta = logs / "group" / "run_info" / "meta.json"
    meta.write_text("{not-json\n")

    records = load_records(
        equals={"optimizer": "adamw"},
        logs_root=str(logs),
        resolve_lineages=False,
    )

    assert len(records) == 1
    record = records[0]
    assert "lr" not in record.effective_config
    assert "lora_plus_multiplier" not in record.effective_config
    assert "precond_refresh_every" not in record.effective_config
    assert "execution_source_sha" not in record.effective_config
    assert record.audit_provenance.config["execution_source_sha"] == (
        "not-present-in-any-commit"
    )
    assert {issue.code for issue in record.issues} >= {"manifest_corrupt"}


def test_explicit_lineage_crosses_physical_groups_and_query_closes_chain(
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
        "lr": 1e-3,
        "data_pipeline_version": "packed_v1.1",
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
        {"event": "eval", "step": 200, "eval_loss": 0.7, "lr": 1e-3},
    ])

    runs = load_records(equals={"group": "new-group"}, logs_root=str(logs))

    assert len(runs) == 1
    assert isinstance(runs[0], MergedRunLineage)
    assert runs[0].attempt_ids == ("attempt-a", "attempt-b")


def test_load_records_accepts_an_explicit_catalog_snapshot(tmp_path):
    logs = tmp_path / "logs"
    _write(logs / "group" / "run_info" / "logs" / "log_0.out", [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3},
        {"event": "eval", "step": 10, "eval_loss": 0.9, "lr": 1e-3},
    ])
    catalog = RunCatalog.discover(logs)

    records = load_records(
        equals={"optimizer": "adamw"},
        catalog=catalog,
        resolve_lineages=False,
    )

    assert len(records) == 1
    with pytest.raises(ValueError, match="either catalog or logs_root"):
        load_records(catalog=catalog, logs_root=str(logs))
