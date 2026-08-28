"""Focused policy tests for paper view semantic cohorts."""
from __future__ import annotations

import copy

import pytest

from lora_playground.comparison import VariantSpec, build_comparison
from lora_playground.plotting.paper_view_semantics import (
    FACTORWISE_SLOT_BOUNDARY,
    ViewSemanticMetadataError,
    factorwise_slot_decision,
    factorwise_slot_semantic_key,
    filter_paper_precond_cohort,
    project_paper_precond_cohort,
)
from lora_playground.run_records import RunRecord, SemanticRunProjection


def test_recorded_revision_wins_without_consulting_ancestry():
    def forbidden(*_args):
        raise AssertionError("recorded revision must not consult Git ancestry")

    decision = factorwise_slot_decision(
        {
            "optimizer_impl_revision": 2,
            "semantic_revisions": {"optimizer_impl": 2},
            "git_commit": "irrelevant",
            "execution_source_sha": None,
            "git_dirty": True,
        },
        is_ancestor=forbidden,
    )

    assert decision.revision == 2
    assert decision.source == "recorded"
    assert decision.eligible


def test_recorded_revision_mismatch_and_future_revision_fail_closed():
    with pytest.raises(ViewSemanticMetadataError, match="disagrees"):
        factorwise_slot_decision({
            "optimizer_impl_revision": 2,
            "semantic_revisions": {"optimizer_impl": 1},
        })

    future = factorwise_slot_decision({"optimizer_impl_revision": 3})
    assert future.revision == 3
    assert not future.eligible


@pytest.mark.parametrize(
    "ancestry, revision, eligible",
    [(True, 2, True), (False, 1, False), (None, None, False)],
)
def test_legacy_commit_uses_tri_state_ancestry(ancestry, revision, eligible):
    calls = []

    def resolver(ancestor, descendant):
        calls.append((ancestor, descendant))
        return ancestry

    decision = factorwise_slot_decision(
        {
            "git_commit": "legacy-commit",
            "execution_source_sha": "arbitrary",
            "execution_source_dirty": True,
        },
        is_ancestor=resolver,
    )

    assert calls == [(FACTORWISE_SLOT_BOUNDARY, "legacy-commit")]
    assert decision.revision == revision
    assert decision.eligible is eligible
    assert decision.source == (
        "unknown" if ancestry is None else "legacy_git_ancestry"
    )


def test_missing_legacy_commit_is_unknown_and_hash_independent():
    a = factorwise_slot_decision({"execution_source_sha": "one"})
    b = factorwise_slot_decision({
        "execution_source_sha": "two", "git_dirty": True,
    })
    assert a == b
    assert not a.eligible
    assert a.source == "unknown"


def test_filter_is_narrow_nonmutating_and_returns_decisions():
    runs = [
        ({"optimizer": "adamw", "precond": "product"}, []),
        ({"optimizer": "method", "precond": "product"}, []),
        ({"optimizer": "legacy-alias", "precond": "factorwise",
          "git_commit": "post"}, []),
        ({"optimizer": "method", "precond": "one-sided",
          "git_commit": "pre"}, []),
    ]
    before = copy.deepcopy(runs)

    kept, excluded = filter_paper_precond_cohort(
        runs,
        view_id="precond_beta2",
        is_ancestor=lambda _boundary, commit: commit == "post",
    )

    assert kept == tuple(runs)
    assert excluded == ()
    assert runs == before

    with pytest.raises(ValueError, match="unknown factorwise-slot view"):
        filter_paper_precond_cohort(runs, view_id="typo")


def test_projection_adds_only_reviewed_view_semantics_to_eligible_records():
    legacy = RunRecord.from_parsed(
        {
            "optimizer": "method",
            "precond": "factorwise",
            "git_commit": "post",
            "_log_filename": "log_0.out",
        },
        [{"step": 1000, "eval_loss": 0.7}],
        group="group",
        manifest=None,
    )
    product = ({"optimizer": "method", "precond": "product"}, [])

    kept, excluded = project_paper_precond_cohort(
        [legacy, product],
        view_id="precond",
        is_ancestor=lambda _boundary, commit: commit == "post",
    )

    assert excluded == ()
    assert isinstance(kept[0], SemanticRunProjection)
    assert kept[0].effective_config["factorwise_slot_revision"] == 2
    assert kept[0].raw_config["git_commit"] == "post"
    assert factorwise_slot_semantic_key(kept[0].effective_config) == 2
    assert kept[1] is product


def test_projection_fails_closed_for_pre_fix_or_unknown_legacy_records():
    runs = [
        ({"optimizer": "method", "precond": "one-sided",
          "git_commit": "pre"}, []),
        ({"optimizer": "method", "precond": "factorwise"}, []),
    ]

    kept, excluded = project_paper_precond_cohort(
        runs,
        view_id="precond_beta2",
        is_ancestor=lambda _boundary, _commit: False,
    )

    assert kept == (runs[0],)
    assert [decision.source for _run, decision in excluded] == ["unknown"]
    assert excluded[0][0] is runs[1]


def test_projected_legacy_and_recorded_revision_share_one_reviewed_curve():
    common = {
        "optimizer": "method",
        "precond": "factorwise",
        "lr": 1e-3,
        "measurement_semantics_revision": 1,
        "data_pipeline_version": "packed_v1.1",
    }
    legacy = ({**common, "git_commit": "post", "seed": 0}, [
        {"step": 1000, "eval_loss": 0.8},
    ])
    recorded = RunRecord.from_parsed(
        {
            **common,
            "seed": 1,
            "run_schema_version": 1,
            "semantic_revisions": {
                "optimizer_impl": 2,
                "measurement": 1,
                "data_pipeline": "packed_v1.1",
            },
            "_log_filename": "log_1.out",
        },
        [{"step": 1000, "eval_loss": 0.7}],
        group="group",
        manifest=None,
    )
    projected, excluded = project_paper_precond_cohort(
        [legacy, recorded],
        view_id="precond",
        is_ancestor=lambda _boundary, commit: commit == "post",
    )
    spec = VariantSpec(
        "factorwise",
        "factorwise",
        {"optimizer": "method"},
        optimizer_semantic_key=factorwise_slot_semantic_key,
    )

    result = build_comparison(projected, [spec], horizon=1000)

    assert excluded == ()
    curve = result.completed["factorwise"][1e-3]
    assert curve.n_replicates == 2
    assert curve.final_loss == pytest.approx(0.75)


def test_matched_view_requires_reviewed_product_but_ordinary_view_does_not():
    old_product = ({
        "optimizer": "method",
        "precond": "product",
        "optimizer_impl_revision": 1,
    }, [])
    reviewed_product = ({
        "optimizer": "method",
        "precond": "product",
        "optimizer_impl_revision": 2,
    }, [])
    old_one_sided = ({
        "optimizer": "method",
        "precond": "one-sided",
        "optimizer_impl_revision": 1,
    }, [])

    ordinary_kept, ordinary_excluded = project_paper_precond_cohort(
        [old_product, reviewed_product, old_one_sided], view_id="precond",
    )
    matched_kept, matched_excluded = project_paper_precond_cohort(
        [old_product, reviewed_product, old_one_sided],
        view_id="precond_matched",
    )

    assert ordinary_kept == (old_product, reviewed_product, old_one_sided)
    assert ordinary_excluded == ()
    assert matched_kept == (reviewed_product,)
    assert [run for run, _decision in matched_excluded] == [
        old_product, old_one_sided,
    ]
