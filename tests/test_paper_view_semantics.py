"""Focused policy tests for paper view semantic cohorts."""
from __future__ import annotations

import copy

import pytest

from lora_playground.plotting.paper_view_semantics import (
    FACTORWISE_SLOT_BOUNDARY,
    ViewSemanticMetadataError,
    factorwise_slot_decision,
    filter_paper_precond_cohort,
)


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

    assert kept == (runs[0], runs[1], runs[2])
    assert excluded[0][0] is runs[3]
    assert excluded[0][1].revision == 1
    assert runs == before

    with pytest.raises(ValueError, match="unknown factorwise-slot view"):
        filter_paper_precond_cohort(runs, view_id="typo")
