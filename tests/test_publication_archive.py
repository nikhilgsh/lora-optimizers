"""Sealed legacy publication inputs stay explicit and records-native."""
from __future__ import annotations

import pytest

from lora_playground.comparison import VariantSpec, build_comparison
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
        "schema_version": 1,
        "projection_id": "legacy_publication_v1",
        "variants": [
            {"id": "baseline.adamw.v1", "label": "AdamW"},
            {"id": "candidate.method.v1", "label": "Method"},
        ],
        "runs": [
            {
                "logical_id": "adamw-lr1",
                "variant_id": "baseline.adamw.v1",
                "source_physical_ids": ["old/log_0.out"],
                "config": {**common, "optimizer": "adamw", "lr": 1e-3},
                "history": [
                    {"step": 500, "eval_loss": 0.9},
                    {"step": 1000, "eval_loss": 0.8},
                ],
            },
            {
                "logical_id": "method-lr1",
                "variant_id": "candidate.method.v1",
                "source_physical_ids": [
                    "old/log_1.out", "old/log_1.out.resume_1",
                ],
                "config": {**common, "lr": 1e-3},
                "history": [
                    {"step": 500, "eval_loss": 0.8},
                    {"step": 1000, "eval_loss": 0.7},
                ],
            },
        ],
    }


def test_archive_runs_feed_comparison_without_legacy_loader_or_defaults():
    archive = publication_archive_from_payload(_payload())
    specs = tuple(
        VariantSpec(
            variant.id,
            variant.label,
            {"_publication_variant_id": variant.id},
        )
        for variant in archive.variants
    )

    result = build_comparison(archive.runs, specs, horizon=1000)
    rows, target = leaderboard_rows_from_comparison(
        result, horizon=1000, baseline_id="baseline.adamw.v1"
    )

    assert target == 0.8
    assert {row["variant"]: row["final_at_best"] for row in rows} == {
        "AdamW": 0.8,
        "Method": 0.7,
    }
    projected = run_view(archive.runs[1])
    assert projected.physical_id == "legacy_publication_v1/method-lr1"
    assert projected.semantic_config["_publication_variant_id"] == (
        "candidate.method.v1"
    )
    assert projected.raw_config["source_physical_ids"] == (
        "old/log_1.out", "old/log_1.out.resume_1",
    )


def test_archive_is_immutable_and_orders_runs_by_logical_identity():
    payload = _payload()
    payload["runs"].reverse()
    archive = publication_archive_from_payload(payload)

    assert [run.physical_id for run in archive.runs] == [
        "legacy_publication_v1/adamw-lr1",
        "legacy_publication_v1/method-lr1",
    ]
    with pytest.raises(TypeError):
        archive.runs[0].effective_config["lr"] = 3e-3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["variants"].append(
                {"id": "baseline.adamw.v1", "label": "duplicate"}
            ),
            "duplicate publication variant id",
        ),
        (
            lambda payload: payload["runs"][0].update(
                variant_id="not-declared"
            ),
            "unknown variant",
        ),
        (
            lambda payload: payload["runs"][1].update(
                source_physical_ids=["old/log_0.out"]
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
