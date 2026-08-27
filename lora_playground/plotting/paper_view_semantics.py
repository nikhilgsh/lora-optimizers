"""Narrow semantic-cohort policy for paper preconditioner views.

The factorwise free-slot behavior changed at one recorded Git boundary before
versioned optimizer revisions existed.  This module translates that historical
fact into a transient paper-view decision.  It is deliberately not a loader
admission rule and never consults source hashes, dirty-tree attestations, or
manifest metadata.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Literal, Mapping, Sequence


FACTORWISE_SLOT_BOUNDARY = "7792797b28771e22c58a180fc2b6428ee6e37c8f"
FACTORWISE_SLOT_COHORT = 2
FACTORWISE_SLOT_VIEWS = frozenset({"precond", "precond_beta2"})
FACTORWISE_SLOT_PRECONDS = frozenset({"factorwise", "one-sided"})


class ViewSemanticMetadataError(ValueError):
    """Recorded scalar and structured semantic metadata disagree."""


@dataclass(frozen=True, slots=True)
class ViewSemanticDecision:
    revision: int | None
    source: Literal["recorded", "legacy_git_ancestry", "unknown"]
    eligible: bool
    reason: str


def _recorded_optimizer_revision(cfg: Mapping[str, Any]) -> int | None:
    scalar = cfg.get("optimizer_impl_revision")
    nested_block = cfg.get("semantic_revisions")
    nested = (
        nested_block.get("optimizer_impl")
        if isinstance(nested_block, Mapping)
        else None
    )
    if scalar is not None and nested is not None and scalar != nested:
        raise ViewSemanticMetadataError(
            "optimizer revision metadata disagrees: "
            f"scalar={scalar!r}, nested={nested!r}"
        )
    revision = scalar if scalar is not None else nested
    if revision is None:
        return None
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ViewSemanticMetadataError(
            f"optimizer revision must be a positive int, got {revision!r}"
        )
    return revision


@lru_cache(maxsize=None)
def git_is_ancestor(ancestor: str, descendant: str) -> bool | None:
    """Tri-state ancestry query for the repository containing the caller."""
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def factorwise_slot_decision(
    cfg: Mapping[str, Any],
    *,
    is_ancestor: Callable[[str, str], bool | None] = git_is_ancestor,
) -> ViewSemanticDecision:
    """Decide membership in the exact post-fix factorwise-slot cohort."""
    revision = _recorded_optimizer_revision(cfg)
    if revision is not None:
        eligible = revision == FACTORWISE_SLOT_COHORT
        reason = (
            "recorded optimizer revision matches required factorwise-slot cohort"
            if eligible
            else (
                f"recorded optimizer revision {revision} is not the reviewed "
                f"factorwise-slot cohort {FACTORWISE_SLOT_COHORT}"
            )
        )
        return ViewSemanticDecision(revision, "recorded", eligible, reason)

    commit = cfg.get("git_commit")
    if not isinstance(commit, str) or not commit.strip():
        return ViewSemanticDecision(
            None,
            "unknown",
            False,
            "legacy run has no recorded git_commit",
        )
    try:
        descendant = is_ancestor(FACTORWISE_SLOT_BOUNDARY, commit)
    except Exception:
        descendant = None
    if descendant is None:
        return ViewSemanticDecision(
            None,
            "unknown",
            False,
            f"could not resolve legacy git ancestry for {commit}",
        )
    inferred = 2 if descendant else 1
    return ViewSemanticDecision(
        inferred,
        "legacy_git_ancestry",
        inferred == FACTORWISE_SLOT_COHORT,
        (
            "legacy commit is in the reviewed post-fix factorwise-slot cohort"
            if descendant
            else "legacy commit predates the factorwise-slot fix"
        ),
    )


def filter_paper_precond_cohort(
    runs: Sequence[Any],
    *,
    view_id: str,
    is_ancestor: Callable[[str, str], bool | None] = git_is_ancestor,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ViewSemanticDecision], ...]]:
    """Filter only the factorwise/matched-control arms in two paper views.

    Product and AdamW runs pass unchanged because the known slot change is not
    their view semantic.  Exclusions carry their exact decision so diagnostics
    and missing-arm notes consume one policy result instead of reimplementing
    it.  Input objects are returned unchanged and never mutated.
    """
    if view_id not in FACTORWISE_SLOT_VIEWS:
        raise ValueError(
            f"unknown factorwise-slot view {view_id!r}; expected one of "
            f"{sorted(FACTORWISE_SLOT_VIEWS)!r}"
        )

    from lora_playground.run_records import run_view

    kept = []
    excluded = []
    for index, run in enumerate(runs):
        view = run_view(run, index)
        if view.semantic_config.get("precond") not in FACTORWISE_SLOT_PRECONDS:
            kept.append(run)
            continue
        # The policy needs optimizer semantics and one audit fact.  Construct a
        # transient decision input without merging provenance into the run's
        # semantic identity or exposing source hashes to the policy.
        decision_input = dict(view.semantic_config)
        if view.semantic_revisions:
            decision_input["semantic_revisions"] = view.semantic_revisions
        git_commit = view.audit_config.get("git_commit")
        if git_commit is not None:
            decision_input["git_commit"] = git_commit
        decision = factorwise_slot_decision(
            decision_input, is_ancestor=is_ancestor
        )
        if decision.eligible:
            kept.append(run)
        else:
            excluded.append((run, decision))
    return tuple(kept), tuple(excluded)


__all__ = [
    "FACTORWISE_SLOT_BOUNDARY",
    "FACTORWISE_SLOT_COHORT",
    "FACTORWISE_SLOT_PRECONDS",
    "FACTORWISE_SLOT_VIEWS",
    "ViewSemanticDecision",
    "ViewSemanticMetadataError",
    "factorwise_slot_decision",
    "filter_paper_precond_cohort",
    "git_is_ancestor",
]
