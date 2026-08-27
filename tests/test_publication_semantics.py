"""Normalized publication semantics separate provenance, cohorts, and labels."""
from __future__ import annotations

import pytest

from lora_playground.publication_semantics import (
    PublicationSemanticsError,
    PublicationVariantSemantics,
    normalize_legacy_optimizer_variant_fields,
    normalize_optimizer_variant_fields,
    publication_semantics_from_payload,
)


def _semantics(*, implementation_revision="commit-a", semantic_revision=1):
    return PublicationVariantSemantics(
        optimizer="method",
        config={"beta1": 0.9, "precond_delta": 1e-4},
        effective={"mode": "polar"},
        semantic_revision=semantic_revision,
        implementation_class="package.Method",
        implementation_revision=implementation_revision,
    )


def test_exact_source_revision_does_not_change_reviewed_view_key():
    first = _semantics(implementation_revision="commit-a")
    second = _semantics(implementation_revision="commit-b")

    assert first.exact_id != second.exact_id
    assert first.view_key == second.view_key


def test_behavior_revision_changes_exact_and_view_identity():
    first = _semantics(semantic_revision=1)
    second = _semantics(semantic_revision=2)

    assert first.exact_id != second.exact_id
    assert first.view_key != second.view_key


def test_normalizer_uses_one_representation_and_exact_observation_contract():
    config, effective = normalize_optimizer_variant_fields(
        {
            "_optim_class": "RenamedClass",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "delta": 1e-4,
            "ns_steps": 8,
            "diagnostics_every": 1,
            "log_basic_diagnostics": True,
            "new_debug_looking_semantic_field": 7,
        },
        {"effective_inner_polar": "polar_express"},
    )

    assert dict(config) == {
        "beta1": 0.9,
        "beta2": 0.999,
        "precond_delta": 1e-4,
        "muon_ns_steps": 8,
        "new_debug_looking_semantic_field": 7,
    }
    assert dict(effective) == {"effective_inner_polar": "polar_express"}


def test_producer_effective_fields_are_copied_verbatim():
    config, effective = normalize_optimizer_variant_fields(
        {"_optim_class": "LoRAPlusAdamW", "lr": 1e-3},
        {
            "effective_picard_iters": 1,
            "future_recorded_semantic": "keep",
        },
    )

    assert dict(config) == {}
    assert dict(effective) == {
        "effective_picard_iters": 1,
        "future_recorded_semantic": "keep",
    }


def test_legacy_inert_effective_is_removed_without_dropping_unknown():
    config, effective = normalize_legacy_optimizer_variant_fields(
        {"_optim_class": "LoRAPlusAdamW", "lr": 1e-3},
        {
            "effective_picard_iters": 1,
            "future_recorded_semantic": "keep",
        },
        derived_fallback={},
    )

    assert dict(config) == {}
    assert dict(effective) == {"future_recorded_semantic": "keep"}


def test_historical_fallback_only_contributes_known_applicable_fields():
    config, effective = normalize_legacy_optimizer_variant_fields(
        {"polar_method": "polar_express", "ns_steps": 8},
        {},
        derived_fallback={
            "effective_inner_polar": "polar_express",
            "effective_polar_iters": 8,
            "reconstructed_current_default": True,
        },
    )

    assert dict(effective) == {
        "effective_inner_polar": "polar_express",
        "effective_polar_iters": 8,
    }


def test_payload_parser_requires_versioned_normalized_schema():
    semantics = publication_semantics_from_payload({
        "schema_version": 1,
        "optimizer": "method",
        "config": {"momentum": 0.9},
        "effective": {},
        "semantic_revision": 2,
        "implementation": {
            "class": "package.Method",
            "revision": "commit",
        },
    })
    assert semantics.config["momentum"] == 0.9

    with pytest.raises(PublicationSemanticsError, match="schema_version"):
        publication_semantics_from_payload({"schema_version": 2})
