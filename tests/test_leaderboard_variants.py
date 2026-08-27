"""Tests for records-native publication variant projection."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from lora_playground.leaderboard_variants import (
    PRODUCER_SEMANTICS_FIELD,
    PUBLICATION_EXACT_ID_FIELD,
    PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD,
    PUBLICATION_STYLE_KEY_FIELD,
    PUBLICATION_VARIANT_ID_FIELD,
    PUBLICATION_VARIANT_LABEL_FIELD,
    PublicationVariantProjectionError,
    project_publication_runs,
    publication_variant_specs,
    stable_publication_variant_id,
)
from lora_playground.publication_semantics import (
    PublicationVariantSemantics,
    normalize_optimizer_variant_fields,
)
from lora_playground.publication_archive import publication_archive_from_payload
from lora_playground.run_records import RunView, freeze_value


def _versioned_run(
    *,
    physical_id: str = "run-1",
    optimizer: str = "toy-optimizer",
    lr: float = 1e-3,
    optimizer_config=None,
    optimizer_effective=None,
    optimizer_impl_revision: int = 3,
) -> RunView:
    optimizer_config = dict(optimizer_config or {
        "_optim_class": "ToyOptimizer",
        "momentum": 0.9,
    })
    optimizer_effective = dict(optimizer_effective or {"mode": "base"})
    variant_config, variant_effective = normalize_optimizer_variant_fields(
        optimizer_config, optimizer_effective
    )
    raw = freeze_value({
        "run_schema_version": 2,
        "run_id": physical_id,
        "optimizer": optimizer,
        "optimizer_config": optimizer_config,
        "optimizer_effective": optimizer_effective,
        PRODUCER_SEMANTICS_FIELD: {
            "schema_version": 1,
            "optimizer": optimizer,
            "config": dict(variant_config),
            "effective": dict(variant_effective),
            "semantic_revision": optimizer_impl_revision,
            "implementation": {
                "class": f"tests.{optimizer_config.get('_optim_class')}",
                "revision": optimizer_impl_revision,
            },
        },
        "semantic_revisions": {
            "optimizer_impl": optimizer_impl_revision,
            "data_pipeline": 1,
            "measurement": 1,
        },
    })
    semantic = freeze_value({
        "optimizer": optimizer,
        "lr": lr,
        "momentum": optimizer_config.get("momentum"),
        "mode": optimizer_effective.get("mode"),
        "_derived": {"stale_reconstruction": True},
    })
    return RunView(
        semantic_config=semantic,
        audit_config=freeze_value({}),
        raw_config=raw,
        history=(freeze_value({"step": 10, "eval_loss": 1.0}),),
        physical_id=physical_id,
        group="records",
        log_filename=f"{physical_id}.log",
        semantic_revisions=raw["semantic_revisions"],
        run_schema_version=2,
    )


def _label(cfg):
    return f"{cfg['optimizer']} [{cfg['_derived']['mode']}]"


def _archive_payload():
    variants = (
        ("archive.zeta", "Zeta"),
        ("archive.adamw", "AdamW"),
        ("archive.alpha", "Alpha"),
    )
    return {
        "schema_version": 1,
        "projection_id": "paper-v1",
        "variants": [
            {
                "view_key": variant_id,
                "label": label,
                "style_key": label,
                "optimizer_semantic_key": variant_id,
                "exact_ids": [f"exact.{variant_id}"],
            }
            for variant_id, label in variants
        ],
        "runs": [
            {
                "logical_id": f"run-{index}",
                "exact_id": f"exact.{variant_id}",
                "source_segments": [{
                    "physical_id": f"logs/source-{index}.log",
                    "contributed_start_step": 10,
                    "contributed_end_step": 10,
                }],
                "config": {
                    "optimizer": "adamw" if label == "AdamW" else "toy",
                    "model_name": "model",
                    "data_dir": "data",
                    "lora_r": 16,
                    "lr": 1e-3,
                    "max_steps": 10,
                    "measurement_semantics_revision": "measurement.v1",
                },
                "history": [{"step": 10, "eval_loss": 1.0 + index}],
            }
            for index, (variant_id, label) in enumerate(variants)
        ],
    }


def test_projection_uses_only_recorded_effective_values_for_derived_label_input():
    seen = []

    def adapter(cfg):
        seen.append(cfg)
        return _label(cfg)

    run = _versioned_run(
        optimizer_config={
            "_optim_class": "ToyOptimizer",
            "momentum": 0.9,
            "picard_iters_override": 2,
        },
        optimizer_effective={"mode": "recorded", "iters": 8},
    )
    projected = project_publication_runs([run], label_adapter=adapter)

    assert seen[0]["_derived"] == {"mode": "recorded", "iters": 8}
    assert "stale_reconstruction" not in seen[0]["_derived"]
    assert seen[0]["picard_iters_override"] == 2
    assert projected[0].effective_config[PUBLICATION_VARIANT_LABEL_FIELD] == (
        "toy-optimizer [recorded]"
    )


def test_lr_and_diagnostic_constructor_fields_do_not_change_identity():
    semantic = {
        "_optim_class": "ToyOptimizer",
        "momentum": 0.9,
        "diag_metric": "learned",
        "lr": 1e-3,
        "log_basic_diagnostics": False,
        "diagnostics_every": 20,
        "dump_pre_polar_every": 0,
        "ssc_kappa_diagnose_eigvalsh": False,
        "ssc_kappa_diag_ema_beta": None,
        "debug_snapshot_limit": 4,
    }
    diagnostics_changed = {
        **semantic,
        "lr": 3e-3,
        "log_basic_diagnostics": True,
        "diagnostics_every": 1,
        "dump_pre_polar_every": 5,
        "ssc_kappa_diagnose_eigvalsh": True,
        "ssc_kappa_diag_ema_beta": 0.99,
        "debug_snapshot_limit": 99,
    }
    first = project_publication_runs(
        [_versioned_run(physical_id="a", lr=1e-3, optimizer_config=semantic)],
        label_adapter=_label,
    )[0]
    second = project_publication_runs(
        [_versioned_run(
            physical_id="b",
            lr=3e-3,
            optimizer_config=diagnostics_changed,
        )],
        label_adapter=_label,
    )[0]

    assert (
        first.effective_config[PUBLICATION_VARIANT_ID_FIELD]
        == second.effective_config[PUBLICATION_VARIANT_ID_FIELD]
    )


def test_new_constructor_field_automatically_forms_a_distinct_identity():
    old = stable_publication_variant_id(PublicationVariantSemantics(
        optimizer="toy-optimizer",
        config={"momentum": 0.9},
        effective={"mode": "base"},
        semantic_revision=3,
        implementation_class="tests.ToyOptimizer",
        implementation_revision=3,
    ))
    new = stable_publication_variant_id(PublicationVariantSemantics(
        optimizer="toy-optimizer",
        config={"momentum": 0.9, "new_semantic_knob": 7},
        effective={"mode": "base"},
        semantic_revision=3,
        implementation_class="tests.ToyOptimizer",
        implementation_revision=3,
    ))

    assert old != new
    assert new.startswith("publication.exact.v1:toy-optimizer:")


@pytest.mark.parametrize(
    ("effective", "revision"),
    [({"mode": "changed"}, 3), ({"mode": "base"}, 4)],
)
def test_effective_semantics_and_impl_revision_change_identity(effective, revision):
    baseline = stable_publication_variant_id(PublicationVariantSemantics(
        optimizer="toy-optimizer",
        config={"momentum": 0.9},
        effective={"mode": "base"},
        semantic_revision=3,
        implementation_class="tests.ToyOptimizer",
        implementation_revision=3,
    ))
    changed = stable_publication_variant_id(PublicationVariantSemantics(
        optimizer="toy-optimizer",
        config={"momentum": 0.9},
        effective=effective,
        semantic_revision=revision,
        implementation_class="tests.ToyOptimizer",
        implementation_revision=revision,
    ))

    assert baseline != changed


def test_label_changes_do_not_change_identity():
    run = _versioned_run()
    first = project_publication_runs([run], label_adapter=lambda _cfg: "First")[0]
    second = project_publication_runs([run], label_adapter=lambda _cfg: "Second")[0]

    assert (
        first.effective_config[PUBLICATION_VARIANT_ID_FIELD]
        == second.effective_config[PUBLICATION_VARIANT_ID_FIELD]
    )
    assert (
        first.effective_config[PUBLICATION_VARIANT_LABEL_FIELD]
        != second.effective_config[PUBLICATION_VARIANT_LABEL_FIELD]
    )


def test_projected_runs_are_immutable():
    projected = project_publication_runs(
        [_versioned_run()], label_adapter=_label
    )[0]

    with pytest.raises(TypeError):
        projected.effective_config["lr"] = 7
    with pytest.raises(TypeError):
        projected.history[0]["eval_loss"] = 7
    with pytest.raises(FrozenInstanceError):
        projected.physical_id = "changed"


def test_archive_publication_fields_are_preserved_without_adapter_reclassification():
    archive = publication_archive_from_payload(_archive_payload())

    def must_not_run(_cfg):
        raise AssertionError("archive labels must not be reconstructed")

    projected = project_publication_runs(
        archive.runs, label_adapter=must_not_run
    )
    specs = publication_variant_specs(projected)

    assert tuple(spec.id for spec in specs) == (
        "archive.adamw",
        "archive.alpha",
        "archive.zeta",
    )
    assert tuple(spec.label for spec in specs) == ("AdamW", "Alpha", "Zeta")
    assert all(spec.style_key == spec.label for spec in specs)
    assert [dict(spec.predicate) for spec in specs] == [
        {PUBLICATION_VARIANT_ID_FIELD: spec.id} for spec in specs
    ]


def _publication_view(
    physical_id: str,
    variant_id: str,
    label: str,
    *,
    exact_id: str | None = None,
) -> RunView:
    cfg = freeze_value({
        PUBLICATION_EXACT_ID_FIELD: exact_id or f"exact.{physical_id}",
        PUBLICATION_VARIANT_ID_FIELD: variant_id,
        PUBLICATION_VARIANT_LABEL_FIELD: label,
        PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD: variant_id,
        PUBLICATION_STYLE_KEY_FIELD: label,
    })
    return RunView(
        semantic_config=cfg,
        audit_config=freeze_value({}),
        raw_config=freeze_value({}),
        history=(freeze_value({"step": 1, "eval_loss": 1.0}),),
        physical_id=physical_id,
        group=None,
        log_filename=None,
        semantic_revisions=freeze_value({}),
    )


@pytest.mark.parametrize(
    "runs",
    [
        [
            _publication_view("a", "same-id", "First"),
            _publication_view("b", "same-id", "Second"),
        ],
        [
            _publication_view("a", "first-id", "Same"),
            _publication_view("b", "second-id", "Same"),
        ],
    ],
)
def test_spec_registry_rejects_non_bijective_id_label_mapping(runs):
    with pytest.raises(
        PublicationVariantProjectionError,
        match="conflicting metadata|maps to both",
    ):
        publication_variant_specs(runs)


def test_spec_registry_allows_multiple_exact_ids_in_one_reviewed_view():
    specs = publication_variant_specs([
        _publication_view("a", "shared-view", "Method", exact_id="exact-a"),
        _publication_view("b", "shared-view", "Method", exact_id="exact-b"),
    ])

    assert len(specs) == 1
    assert specs[0].id == "shared-view"
    assert specs[0].optimizer_semantic_key({
        PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD: "shared-view"
    }) == "shared-view"


@pytest.mark.parametrize("schema_version", [None, 1, True, "2"])
def test_preproducer_live_run_without_archive_fields_fails_closed(schema_version):
    live = _versioned_run()
    raw = {
        key: value
        for key, value in live.raw_config.items()
        if key != "run_schema_version"
    }
    if schema_version is not None:
        raw["run_schema_version"] = schema_version
    preproducer = RunView(
        semantic_config=live.semantic_config,
        audit_config=live.audit_config,
        raw_config=freeze_value(raw),
        history=live.history,
        physical_id=live.physical_id,
        group=live.group,
        log_filename=live.log_filename,
        semantic_revisions=live.semantic_revisions,
        run_schema_version=schema_version,
    )

    with pytest.raises(PublicationVariantProjectionError, match=r"schema 2\+"):
        project_publication_runs([preproducer], label_adapter=_label)


def test_versioned_projection_requires_producer_semantics_block():
    live = _versioned_run()
    raw = dict(live.raw_config)
    raw.pop(PRODUCER_SEMANTICS_FIELD)
    malformed = replace(live, raw_config=freeze_value(raw))

    with pytest.raises(
        PublicationVariantProjectionError,
        match="optimizer_variant_semantics",
    ):
        project_publication_runs([malformed], label_adapter=_label)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 99},
        {
            "schema_version": 1,
            "optimizer": "toy-optimizer",
            "config": {},
            "effective": {},
            "semantic_revision": 1,
            "implementation": {"class": "tests.ToyOptimizer"},
        },
    ],
)
def test_schema2_malformed_producer_semantics_fails_at_publication_boundary(
    payload,
):
    live = _versioned_run()
    raw = dict(live.raw_config)
    raw[PRODUCER_SEMANTICS_FIELD] = payload
    malformed = replace(live, raw_config=freeze_value(raw))

    with pytest.raises(PublicationVariantProjectionError):
        project_publication_runs([malformed], label_adapter=_label)
