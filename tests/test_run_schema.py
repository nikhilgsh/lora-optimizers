"""Focused contract tests for versioned config-event schema fields."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lora_playground.run_schema import (
    ATTEMPT_ID_ENV,
    CHECKPOINT_IDENTITY_ENV,
    DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION,
    MEASUREMENT_SEMANTICS_REVISION,
    RUN_SCHEMA_VERSION,
    attempt_metadata,
    optimizer_implementation_revision,
    semantic_revisions,
)


class _DefaultOptimizer:
    pass


class _RevisedOptimizer:
    IMPLEMENTATION_REVISION = 3


def test_optimizer_revision_uses_class_attribute_or_stable_default():
    assert optimizer_implementation_revision(_DefaultOptimizer()) == 1
    assert optimizer_implementation_revision(_DefaultOptimizer) == (
        DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION
    )
    assert optimizer_implementation_revision(_RevisedOptimizer()) == 3
    assert optimizer_implementation_revision(_RevisedOptimizer) == 3


def test_curvature_whiten_records_the_reviewed_post_fix_revision():
    from lora_playground.optim import CurvatureWhitenLoRA

    assert optimizer_implementation_revision(CurvatureWhitenLoRA) == 2


@pytest.mark.parametrize("bad_revision", [True, 0, -1, 1.5, "2"])
def test_optimizer_revision_rejects_ambiguous_values(bad_revision):
    class _BadOptimizer:
        IMPLEMENTATION_REVISION = bad_revision

    with pytest.raises(ValueError, match="positive int"):
        optimizer_implementation_revision(_BadOptimizer())


def test_semantic_revisions_are_bounded_and_use_resolved_pipeline():
    resolved = {
        "optimizer": "adamw",
        "lr": 3e-4,
        "data_pipeline_version": "packed_v1.1",
        "large_unrelated_config_block": {"not": "duplicated"},
    }

    revisions = semantic_revisions(_RevisedOptimizer(), resolved)

    assert revisions == {
        "optimizer_impl": 3,
        "data_pipeline": "packed_v1.1",
        "measurement": MEASUREMENT_SEMANTICS_REVISION,
    }
    assert set(revisions) == {"optimizer_impl", "data_pipeline", "measurement"}
    assert "large_unrelated_config_block" not in json.dumps(revisions)


@pytest.mark.parametrize(
    "resolved",
    [{}, {"data_pipeline_version": None}, {"data_pipeline_version": "  "}],
)
def test_semantic_revisions_do_not_reconstruct_missing_pipeline(resolved):
    with pytest.raises(ValueError, match="data_pipeline_version"):
        semantic_revisions(_DefaultOptimizer(), resolved)


def test_attempt_metadata_accepts_dedicated_environment_values():
    env = {
        ATTEMPT_ID_ENV: "attempt-17",
        CHECKPOINT_IDENTITY_ENV: "sweep-a/task-4",
    }

    metadata = attempt_metadata(environ=env)

    assert metadata == {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "attempt_id": "attempt-17",
        "resume_parent_attempt_id": None,
        "checkpoint_identity": "sweep-a/task-4",
    }
    json.dumps(metadata)


def test_explicit_attempt_values_override_environment_and_parent_is_ignored():
    env = {
        ATTEMPT_ID_ENV: "environment-attempt",
        "LORA_RESUME_PARENT_ATTEMPT_ID": "environment-parent",
        CHECKPOINT_IDENTITY_ENV: "environment-checkpoint",
    }

    metadata = attempt_metadata(
        attempt_id="explicit-attempt",
        checkpoint_identity="explicit-checkpoint",
        environ=env,
    )

    assert metadata["attempt_id"] == "explicit-attempt"
    assert metadata["resume_parent_attempt_id"] is None
    assert metadata["checkpoint_identity"] == "explicit-checkpoint"


def test_resume_like_environment_does_not_fabricate_parent_relationship():
    metadata = attempt_metadata(
        environ={
            ATTEMPT_ID_ENV: "attempt-17",
            CHECKPOINT_IDENTITY_ENV: "sweep-a/task-4",
            "SLURM_RESTART_COUNT": "2",
            "RESUME_FROM": "/checkpoints/sweep-a/task-4",
        }
    )

    assert metadata["resume_parent_attempt_id"] is None
    assert set(metadata) == {
        "run_schema_version",
        "attempt_id",
        "resume_parent_attempt_id",
        "checkpoint_identity",
    }
    assert not any("hash" in key or "digest" in key for key in metadata)


@pytest.mark.parametrize(
    "env, missing_field",
    [
        ({CHECKPOINT_IDENTITY_ENV: "checkpoint"}, "attempt_id"),
        ({ATTEMPT_ID_ENV: "attempt"}, "checkpoint_identity"),
    ],
)
def test_attempt_metadata_requires_declared_identity(env, missing_field):
    with pytest.raises(ValueError, match=missing_field):
        attempt_metadata(environ=env)


def test_train_direct_checkpoint_attempt_uses_stable_lineage_namespace(
    tmp_path, monkeypatch
):
    from lora_playground.train import _current_attempt_metadata

    monkeypatch.delenv(ATTEMPT_ID_ENV, raising=False)
    monkeypatch.delenv(CHECKPOINT_IDENTITY_ENV, raising=False)
    args = SimpleNamespace(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        resume_from=str(tmp_path / "checkpoints"),
    )

    first = _current_attempt_metadata(args)
    second = _current_attempt_metadata(args)

    assert first["attempt_id"] != second["attempt_id"]
    assert first["resume_parent_attempt_id"] is None
    assert first["checkpoint_identity"] == second["checkpoint_identity"]


def test_train_launcher_identity_overrides_local_fallback(monkeypatch):
    from lora_playground.train import _current_attempt_metadata

    monkeypatch.setenv(ATTEMPT_ID_ENV, "launch-attempt")
    monkeypatch.setenv(CHECKPOINT_IDENTITY_ENV, "group/task-4")
    args = SimpleNamespace(checkpoint_dir=None, resume_from=None)

    assert _current_attempt_metadata(args) == {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "attempt_id": "launch-attempt",
        "resume_parent_attempt_id": None,
        "checkpoint_identity": "group/task-4",
    }


def test_attempt_fields_are_runtime_but_scalar_revisions_define_series():
    from lora_playground.run_records import RUNTIME_FIELDS

    assert {"run_schema_version", "attempt_id", "resume_parent_attempt_id",
            "checkpoint_identity", "_resume"} <= RUNTIME_FIELDS
    assert "optimizer_impl_revision" not in RUNTIME_FIELDS
    assert "measurement_semantics_revision" not in RUNTIME_FIELDS
