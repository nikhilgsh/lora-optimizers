"""Sealed legacy publication inputs stay explicit and records-native."""
from __future__ import annotations

import pytest

from lora_playground.comparison import build_comparison
from lora_playground.leaderboard_variants import (
    composite_publication_identity,
    publication_variant_specs,
)
from lora_playground.leaderboard import leaderboard_rows_from_comparison
from lora_playground.publication_archive import (
    PublicationArchiveError,
    publication_archive_from_payload,
)
from lora_playground.run_records import run_view


def _payload():
    common = {
        "optimizer": "method",
        "model_name": "model",
        "data_dir": "/data/openmath_instruct_2_2m_packed",
        "lora_r": 64,
        "max_steps": 1000,
        "data_pipeline_version": "packed_v1.1",
    }
    return {
        "schema_version": 2,
        "projection_id": "legacy_publication_v1",
        "variants": [
            {
                "view_key": composite_publication_identity(
                    "baseline.adamw.v1", "zero"
                ),
                "label": "AdamW",
                "style_key": "AdamW",
                "optimizer_semantic_key": "baseline.adamw.v1",
                "lora_init_b": "zero",
                "exact_ids": [composite_publication_identity(
                    "exact.adamw.v1", "zero"
                )],
            },
            {
                "view_key": composite_publication_identity(
                    "candidate.method.v1", "zero"
                ),
                "label": "Method",
                "style_key": "Method",
                "optimizer_semantic_key": "candidate.method.v1",
                "lora_init_b": "zero",
                "exact_ids": [composite_publication_identity(
                    "exact.method.v1", "zero"
                )],
            },
        ],
        "runs": [
            {
                "logical_id": "adamw-lr0",
                "exact_id": composite_publication_identity(
                    "exact.adamw.v1", "zero"
                ),
                "source_segments": [{
                    "physical_id": "old/log_adam_low.out",
                    "contributed_start_step": 1000,
                    "contributed_end_step": 1000,
                }],
                "config": {
                    **common,
                    "optimizer": "adamw",
                    "lr": 3e-4,
                    "lora_init_b": "zero",
                    "measurement_semantics_revision": "measurement.v1",
                },
                "history": [{"step": 1000, "eval_loss": 0.9}],
            },
            {
                "logical_id": "adamw-lr1",
                "exact_id": composite_publication_identity(
                    "exact.adamw.v1", "zero"
                ),
                "source_segments": [{
                    "physical_id": "old/log_0.out",
                    "contributed_start_step": 500,
                    "contributed_end_step": 1000,
                }],
                "config": {
                    **common,
                    "optimizer": "adamw",
                    "lr": 1e-3,
                    "lora_init_b": "zero",
                    "measurement_semantics_revision": "measurement.v1",
                },
                "history": [
                    {"step": 500, "eval_loss": 0.9},
                    {"step": 1000, "eval_loss": 0.8},
                ],
            },
            {
                "logical_id": "adamw-lr2",
                "exact_id": composite_publication_identity(
                    "exact.adamw.v1", "zero"
                ),
                "source_segments": [{
                    "physical_id": "old/log_adam_high.out",
                    "contributed_start_step": 1000,
                    "contributed_end_step": 1000,
                }],
                "config": {
                    **common,
                    "optimizer": "adamw",
                    "lr": 3e-3,
                    "lora_init_b": "zero",
                    "measurement_semantics_revision": "measurement.v1",
                },
                "history": [{"step": 1000, "eval_loss": 0.85}],
            },
            {
                "logical_id": "method-lr1",
                "exact_id": composite_publication_identity(
                    "exact.method.v1", "zero"
                ),
                "source_segments": [
                    {
                        "physical_id": "old/log_1.out",
                        "contributed_start_step": 500,
                        "contributed_end_step": 500,
                    },
                    {
                        "physical_id": "old/log_1.out.resume_1",
                        "contributed_start_step": 1000,
                        "contributed_end_step": 1000,
                    },
                ],
                "config": {
                    **common,
                    "lr": 1e-3,
                    "lora_init_b": "zero",
                    "measurement_semantics_revision": "measurement.v1",
                },
                "history": [
                    {"step": 500, "eval_loss": 0.8},
                    {"step": 1000, "eval_loss": 0.7},
                ],
            },
        ],
    }


def test_archive_runs_feed_comparison_without_legacy_loader_or_defaults():
    archive = publication_archive_from_payload(_payload())
    specs = publication_variant_specs(archive.runs)

    result = build_comparison(archive.runs, specs, horizon=1000)
    rows, target = leaderboard_rows_from_comparison(
        result,
        horizon=1000,
        baseline_id=composite_publication_identity("baseline.adamw.v1", "zero"),
    )

    assert target == 0.8
    assert {row["variant"]: row["final_at_best"] for row in rows} == {
        "AdamW": 0.8,
        "Method": 0.7,
    }
    projected = run_view(next(
        run for run in archive.runs if run.physical_id.endswith("/method-lr1")
    ))
    assert projected.physical_id == "legacy_publication_v1/method-lr1"
    assert projected.semantic_config["_publication_variant_id"] == (
        composite_publication_identity("candidate.method.v1", "zero")
    )
    assert projected.raw_config["source_physical_ids"] == (
        "old/log_1.out", "old/log_1.out.resume_1",
    )


def test_archive_is_immutable_and_orders_runs_by_logical_identity():
    payload = _payload()
    payload["runs"].reverse()
    archive = publication_archive_from_payload(payload)

    assert [run.physical_id for run in archive.runs] == [
        "legacy_publication_v1/adamw-lr0",
        "legacy_publication_v1/adamw-lr1",
        "legacy_publication_v1/adamw-lr2",
        "legacy_publication_v1/method-lr1",
    ]
    with pytest.raises(TypeError):
        archive.runs[0].effective_config["lr"] = 3e-3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["runs"][0]["config"].pop(
            "data_pipeline_version"
        ),
        lambda payload: payload["runs"][0]["config"].pop("lora_init_b"),
        lambda payload: payload["runs"][0]["config"].update(
            lora_init_b="unknown"
        ),
        lambda payload: payload["runs"][0]["config"].update(lora_init_b=[]),
        lambda payload: payload["variants"][0].update(lora_init_b="unknown"),
        lambda payload: payload["variants"][0].update(
            view_key="baseline.adamw.v1"
        ),
    ],
)
def test_archive_requires_known_composite_initialization_identity(mutate):
    payload = _payload()
    mutate(payload)

    with pytest.raises(
        PublicationArchiveError,
        match="data_pipeline_version|lora_init_b|compose",
    ):
        publication_archive_from_payload(payload)


@pytest.mark.parametrize("revision", [None, "", True, 0, {}])
def test_archive_requires_a_valid_measurement_semantics_revision(revision):
    payload = _payload()
    payload["runs"][0]["config"]["measurement_semantics_revision"] = revision

    with pytest.raises(
        PublicationArchiveError, match="measurement_semantics_revision"
    ):
        publication_archive_from_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["variants"].append(
                {
                    "view_key": composite_publication_identity(
                        "duplicate.view", "zero"
                    ),
                    "label": "duplicate",
                    "style_key": "duplicate",
                    "optimizer_semantic_key": "duplicate.view",
                    "lora_init_b": "zero",
                    "exact_ids": [composite_publication_identity(
                        "exact.adamw.v1", "zero"
                    )],
                }
            ),
            "duplicate publication exact id",
        ),
        (
            lambda payload: payload["runs"][0].update(
                exact_id="not-declared"
            ),
            "unknown exact id",
        ),
        (
                lambda payload: payload["runs"][1].update(
                    source_segments=[{
                        "physical_id": "old/log_adam_low.out",
                        "contributed_start_step": 500,
                    "contributed_end_step": 1000,
                }]
            ),
            "multiple archived logical runs",
        ),
        (
            lambda payload: payload["runs"][0].update(history=[
                {"step": 1000, "eval_loss": 0.8},
                {"step": 500, "eval_loss": 0.9},
            ]),
            "strictly increasing",
        ),
    ],
)
def test_archive_rejects_ambiguous_identity_or_trajectory(mutate, message):
    payload = _payload()
    mutate(payload)

    with pytest.raises(PublicationArchiveError, match=message):
        publication_archive_from_payload(payload)
