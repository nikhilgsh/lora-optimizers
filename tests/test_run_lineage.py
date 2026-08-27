"""Explicit continuation validation: never infer resume from step ranges."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from lora_playground.run_lineage import (
    LineageMismatchError,
    build_run_lineages,
)
from lora_playground.run_records import RunRecord


@dataclass
class _RunRecord:
    cfg: dict
    history: list[dict]


def _run(
    attempt_id: str,
    steps: list[int],
    *,
    parent: str | None = None,
    checkpoint: str = "task-7-checkpoints",
    semantic_config: dict | None = None,
    semantic_revisions: dict | None = None,
    provenance: dict | None = None,
):
    cfg = {
        "attempt_id": attempt_id,
        "resume_parent_attempt_id": parent,
        "checkpoint_identity": checkpoint,
        "semantic_config": semantic_config if semantic_config is not None else {
            "optimizer": "adamw",
            "lr": 1e-3,
            "model": "tiny",
        },
        "semantic_revisions": (
            semantic_revisions if semantic_revisions is not None else {
                "optimizer_impl": 1,
                "data_pipeline": "packed_v1",
                "measurement": 1,
            }
        ),
        "provenance": provenance if provenance is not None else {},
    }
    history = [{"step": step, "eval_loss": 1.0 - step / 1000}
               for step in steps]
    return cfg, history


def test_true_resume_merges_by_explicit_parent_and_replaces_replayed_tail():
    root_cfg, root_history = _run("attempt-a", [100, 200, 300])
    child_cfg, child_history = _run(
        "attempt-b", [300, 400], parent="attempt-a",
    )
    child_history[0]["eval_loss"] = 0.61

    # Exercise both accepted shapes: RunRecord-like object and legacy tuple.
    lineages = build_run_lineages([
        _RunRecord(root_cfg, root_history),
        (child_cfg, child_history),
    ])

    assert len(lineages) == 1
    lineage = lineages[0]
    assert lineage.attempt_ids == ("attempt-a", "attempt-b")
    assert [event["step"] for event in lineage.history] == [100, 200, 300, 400]
    assert lineage.history[2]["eval_loss"] == pytest.approx(0.61)
    assert lineage.event_attempt_ids == (
        "attempt-a", "attempt-a", "attempt-b", "attempt-b",
    )


def test_unrelated_disjoint_runs_remain_independent():
    # Matching config/checkpoint and adjacent steps are not lineage evidence.
    lineages = build_run_lineages([
        _run("attempt-a", [100, 200]),
        _run("attempt-b", [300, 400]),
    ])

    assert len(lineages) == 2
    assert [lineage.attempt_ids for lineage in lineages] == [
        ("attempt-a",), ("attempt-b",),
    ]


def test_resume_semantic_config_mismatch_is_structured_error():
    root = _run("attempt-a", [100])
    child = _run(
        "attempt-b", [200], parent="attempt-a",
        semantic_config={"optimizer": "adamw", "lr": 3e-3, "model": "tiny"},
    )

    with pytest.raises(LineageMismatchError) as exc_info:
        build_run_lineages([root, child])

    issue = exc_info.value.issue
    assert issue.code == "semantic_config_mismatch"
    assert issue.attempt_id == "attempt-b"
    assert issue.parent_attempt_id == "attempt-a"
    assert issue.details["differences"]["lr"] == {
        "parent": 1e-3, "child": 3e-3,
    }


def test_resume_semantic_revision_mismatch_is_structured_error():
    root = _run("attempt-a", [100])
    child = _run(
        "attempt-b", [200], parent="attempt-a",
        semantic_revisions={
            "optimizer_impl": 2,
            "data_pipeline": "packed_v1",
            "measurement": 1,
        },
    )

    with pytest.raises(LineageMismatchError) as exc_info:
        build_run_lineages([root, child])

    issue = exc_info.value.issue
    assert issue.code == "semantic_revision_mismatch"
    assert issue.details["differences"]["optimizer_impl"] == {
        "parent": 1, "child": 2,
    }


def test_resume_checkpoint_mismatch_is_structured_error():
    root = _run("attempt-a", [100], checkpoint="task-7-checkpoints")
    child = _run(
        "attempt-b", [200], parent="attempt-a",
        checkpoint="task-8-checkpoints",
    )

    with pytest.raises(LineageMismatchError) as exc_info:
        build_run_lineages([root, child])

    issue = exc_info.value.issue
    assert issue.code == "checkpoint_identity_mismatch"
    assert issue.details == {
        "parent_value": "task-7-checkpoints",
        "child_value": "task-8-checkpoints",
    }


def test_segment_provenance_is_preserved_without_collapsing_to_terminal_cfg():
    root = _run(
        "attempt-a", [100, 200],
        provenance={"git_commit": "aaa", "host": "worker-1", "job_id": "10"},
    )
    child = _run(
        "attempt-b", [300], parent="attempt-a",
        provenance={"git_commit": "bbb", "host": "worker-2", "job_id": "11"},
    )

    lineage = build_run_lineages([root, child])[0]

    assert lineage.cfg["provenance"]["git_commit"] == "bbb"
    assert lineage.segments[0].cfg["provenance"] == {
        "git_commit": "aaa", "host": "worker-1", "job_id": "10",
    }
    assert lineage.segments[1].cfg["provenance"] == {
        "git_commit": "bbb", "host": "worker-2", "job_id": "11",
    }
    assert lineage.segments[0].history == tuple(root[1])
    assert lineage.segments[1].history == tuple(child[1])
    assert lineage.event_attempt_ids == (
        "attempt-a", "attempt-a", "attempt-b",
    )


def test_lineage_accepts_the_catalog_run_record_contract():
    cfg, history = _run("attempt-a", [100, 200])
    cfg["_log_filename"] = "log_0.out"
    record = RunRecord.from_parsed(
        cfg,
        history,
        group="group-a",
        manifest=None,
    )

    lineage = build_run_lineages([record])[0]

    assert lineage.attempt_ids == ("attempt-a",)
    assert [event["step"] for event in lineage.history] == [100, 200]


def test_versioned_catalog_records_use_actual_resume_not_launcher_intent():
    shared = {
        "run_schema_version": 2,
        "checkpoint_identity": "group/task-0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1",
            "measurement": 1,
        },
        "optimizer": "adamw",
        "lr": 1e-3,
    }
    root = RunRecord.from_parsed(
        {**shared, "attempt_id": "attempt-a",
         # A launcher may provide this candidate even when no checkpoint exists.
         "resume_parent_attempt_id": "not-actual"},
        [{"step": 100, "eval_loss": 0.8}],
        group="group-a",
        manifest=None,
    )
    child = RunRecord.from_parsed(
        {**shared, "attempt_id": "attempt-b",
         "resume_parent_attempt_id": "wrong-launcher-candidate",
         "_resume": {
             "resume_parent_attempt_id": "attempt-a",
             "checkpoint_identity": "group/task-0",
         }},
        [{"step": 200, "eval_loss": 0.7}],
        group="group-a",
        manifest=None,
    )

    lineages = build_run_lineages([root, child])

    assert len(lineages) == 1
    assert lineages[0].attempt_ids == ("attempt-a", "attempt-b")
    assert "attempt_id" not in lineages[0].semantic_config
    assert "semantic_revisions" not in lineages[0].semantic_config
