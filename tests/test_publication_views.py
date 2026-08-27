from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from lora_playground.leaderboard_variants import (
    PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD,
    PUBLICATION_VARIANT_ID_FIELD,
)
from lora_playground.publication_archive import (
    ArchivedPublicationRun,
    PublicationArchive,
    PublicationVariant,
)
from lora_playground.publication_views import (
    PublicationViewError,
    load_publication_views,
    publication_views_from_payload,
)


def _archive() -> PublicationArchive:
    variants = (
        PublicationVariant("view:adam", "sealed Adam", "sem:adam", "zero", ("exact:adam",), "archive-adam"),
        PublicationVariant("view:ours", "sealed ours", "sem:ours", "zero", ("exact:ours",), "archive-ours"),
    )

    def run(name: str, variant_id: str, semantic_key: str, model: str):
        cfg = MappingProxyType({
            "model_name": model,
            "data_dir": "data/openmath_instruct_fixture",
            "lora_r": 16,
            "data_pipeline_version": "pipeline-v1",
            PUBLICATION_VARIANT_ID_FIELD: variant_id,
            PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD: semantic_key,
        })
        return ArchivedPublicationRun(
            physical_id=name,
            effective_config=cfg,
            history=(MappingProxyType({"step": 10, "eval_loss": 1.0}),),
            raw_config=MappingProxyType({}),
            source_segments=(),
            group="archive:fixture",
        )

    return PublicationArchive(
        projection_id="fixture-v1",
        variants=variants,
        runs=(
            run("adam/a", "view:adam", "sem:adam", "model-a"),
            run("ours/a", "view:ours", "sem:ours", "model-a"),
            run("adam/b", "view:adam", "sem:adam", "model-b"),
        ),
    )


def _payload(*, ours_label: str = "Editorial ours") -> dict:
    return {
        "schema_version": 1,
        "archive_projection_id": "fixture-v1",
        "horizon": 10,
        "workload_selector": {
            "model_name": "model-a",
            "dataset_id": "openmath",
            "lora_r": 16,
            "data_pipeline_version": "pipeline-v1",
        },
        "views": [{
            "id": "figure.main",
            "title": "Main figure",
            "arms": [
                {
                    "variant_id": "view:adam",
                    "label": "Editorial Adam",
                    "roles": ["reference", "target"],
                },
                {
                    "variant_id": "view:ours",
                    "label": ours_label,
                    "style_key": "ours-style",
                    "roles": [],
                },
            ],
        }],
    }


def test_load_resolves_order_roles_workload_and_archive_styles(tmp_path):
    path = tmp_path / "views.json"
    path.write_text(json.dumps(_payload()))
    views = load_publication_views(path, archive=_archive())

    resolved = views.resolve("figure.main", _archive())
    assert [spec.id for spec in resolved.variant_specs] == ["view:adam", "view:ours"]
    assert [spec.label for spec in resolved.variant_specs] == [
        "Editorial Adam", "Editorial ours",
    ]
    assert [spec.style_key for spec in resolved.variant_specs] == [
        "archive-adam", "ours-style",
    ]
    assert resolved.reference_id == "view:adam"
    assert resolved.target_id == "view:adam"
    assert resolved.horizon == 10
    assert [run.physical_id for run in resolved.runs] == ["adam/a", "ours/a"]
    assert resolved.variant_specs[0].predicate == {
        PUBLICATION_VARIANT_ID_FIELD: "view:adam"
    }


def test_editorial_label_change_preserves_ids_assignment_and_order():
    archive = _archive()
    before = publication_views_from_payload(_payload(), archive=archive).resolve(
        "figure.main", archive
    )
    after = publication_views_from_payload(
        _payload(ours_label="Renamed for the paper"), archive=archive
    ).resolve("figure.main", archive)

    assert [spec.id for spec in before.variant_specs] == [
        spec.id for spec in after.variant_specs
    ]
    assert [run.physical_id for run in before.runs] == [
        run.physical_id for run in after.runs
    ]
    assert after.variant_specs[1].label == "Renamed for the paper"


def test_unknown_variant_id_fails_archive_resolution():
    payload = _payload()
    payload["views"][0]["arms"][1]["variant_id"] = "view:missing"
    with pytest.raises(PublicationViewError, match="unknown archive variant"):
        publication_views_from_payload(payload, archive=_archive())


def test_workload_selector_requires_evidence_for_every_arm():
    payload = _payload()
    payload["workload_selector"]["model_name"] = "model-b"
    with pytest.raises(PublicationViewError, match="has no archived run"):
        publication_views_from_payload(payload, archive=_archive())


def test_projection_and_role_ambiguity_fail_closed():
    payload = _payload()
    payload["archive_projection_id"] = "other"
    with pytest.raises(PublicationViewError, match="targets archive"):
        publication_views_from_payload(payload, archive=_archive())

    payload = _payload()
    payload["views"][0]["arms"][0]["roles"] = []
    with pytest.raises(PublicationViewError, match="exactly one reference"):
        publication_views_from_payload(payload)

    payload = _payload()
    payload["horizon"] = 0
    with pytest.raises(PublicationViewError, match="positive integer"):
        publication_views_from_payload(payload)


def test_unknown_schema_fields_and_non_scalar_selectors_are_rejected():
    payload = _payload()
    payload["views"][0]["arms"][0]["sealed_label"] = "identity by text"
    with pytest.raises(PublicationViewError, match="unsupported field"):
        publication_views_from_payload(payload)

    payload = _payload()
    payload["workload_selector"]["lora_r"] = [16, 64]
    with pytest.raises(PublicationViewError, match="JSON scalar"):
        publication_views_from_payload(payload)

    payload = _payload()
    payload["workload_selector"]["data_dir"] = "/physical/path"
    with pytest.raises(PublicationViewError, match="stable 'dataset_id'"):
        publication_views_from_payload(payload)


def test_checked_in_paper_view_resolves_against_checked_in_archive():
    from lora_playground.publication_archive import load_publication_archive

    root = Path(__file__).resolve().parents[1]
    archive = load_publication_archive(
        root / "publication" / "legacy_leaderboard_v1.json"
    )
    views = load_publication_views(
        root / "publication" / "paper_views.json",
        archive=archive,
    )
    payload = json.loads((root / "publication" / "paper_views.json").read_text())
    selector = payload["workload_selector"]
    assert selector["dataset_id"] == "openmath"
    assert "data_dir" not in selector
    assert "openmath_instruct_2_2m_packed_seq2048_llama32" not in json.dumps(payload)
    resolved = views.resolve("paper.hero.adamw_polora.v1", archive)
    assert resolved.horizon == 9000
    assert len(resolved.runs) == 11
    assert resolved.reference_id == resolved.target_id
    assert set(views.views_by_id) == {
        "paper.hero.adamw_polora.v1",
        "paper.msign.v1",
        "paper.magnitude_rule.v1",
        "paper.polora_beta2.v1",
        "paper.adamw_beta2.v1",
        "paper.fig2_ablation.v1",
    }
