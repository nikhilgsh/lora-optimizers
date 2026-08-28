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
FACTORWISE_SLOT_FIELD = "factorwise_slot_revision"
FACTORWISE_SLOT_PROJECTION_ID = "paper.factorwise_slot.v1"
FACTORWISE_SLOT_PRECONDS = frozenset({"factorwise"})
PRODUCT_PRECOND = "product"
MATCHED_PRECOND_VIEW = "precond_matched"
FACTORWISE_SLOT_VIEWS = frozenset({
    "precond", "precond_beta2", MATCHED_PRECOND_VIEW,
})

def effective_precond(cfg: Mapping[str, Any]) -> str | None:
    """The ``precond`` branch a run actually ran, or None if undetermined.

    Delegates to `optim_specs.resolved_precond`, which reproduces
    `CurvatureWhitenLoRA.__init__`'s own resolution
    (``precond or ("product" if diag_metric else "factorwise")``, optim.py:1713)
    from the spec registry rather than from a list of optimizer names.

    Reading the raw ``precond`` field instead is what let 13 pre-fix
    `kl-shampoo-polar-lora` runs -- which record no ``precond`` but pin
    ``diag_metric=False``, i.e. factorwise -- supply the entire factorwise arm
    of the Llama-3.2-1B/openmath/r256 panel while the slot filter never
    examined them. Kept as a name here because the cohort projection below and
    its tests read it, and because which branch a run ran is a question the
    view asks; the ANSWER belongs to the optimizer that resolved it.
    """
    from ..optim_specs import resolved_precond

    return resolved_precond(cfg)


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


def factorwise_slot_semantic_key(cfg: Mapping[str, Any]) -> Any:
    """Optimizer-semantic key for the reviewed factorwise paper views."""
    return cfg.get(FACTORWISE_SLOT_FIELD, cfg.get("optimizer_impl_revision"))


def _decision_for_run_view(view, is_ancestor) -> ViewSemanticDecision:
    decision_input = dict(view.semantic_config)
    if view.semantic_revisions:
        decision_input["semantic_revisions"] = view.semantic_revisions
    git_commit = view.audit_config.get("git_commit")
    if git_commit is not None:
        decision_input["git_commit"] = git_commit
    return factorwise_slot_decision(decision_input, is_ancestor=is_ancestor)


def project_paper_precond_cohort(
    runs: Sequence[Any],
    *,
    view_id: str,
    is_ancestor: Callable[[str, str], bool | None] = git_is_ancestor,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ViewSemanticDecision], ...]]:
    """Project the reviewed cohort onto an explicit transient semantic field.

    Eligible factorwise runs retain their physical and audit provenance
    while gaining ``factorwise_slot_revision`` for ``VariantSpec``'s
    view-specific optimizer key. Ordinary views leave product unchanged because
    the slot bug did not affect it. The explicit ``precond_matched`` view instead
    requires product, factorwise, and one-sided to share the reviewed optimizer
    revision. Unknown and pre-fix records fail closed into ``excluded`` whenever
    that reviewed cohort is required.  The ordinary view leaves one-sided runs
    unchanged because the factorwise-slot fix did not alter that branch; the
    matched view still requires all three branches to record revision 2.
    """
    if view_id not in FACTORWISE_SLOT_VIEWS:
        raise ValueError(
            f"unknown factorwise-slot view {view_id!r}; expected one of "
            f"{sorted(FACTORWISE_SLOT_VIEWS)!r}"
        )

    from lora_playground.run_records import project_run_semantics, run_view

    reviewed_preconds = (
        FACTORWISE_SLOT_PRECONDS | {PRODUCT_PRECOND, "one-sided"}
        if view_id == MATCHED_PRECOND_VIEW
        else FACTORWISE_SLOT_PRECONDS
    )

    kept = []
    excluded = []
    for index, run in enumerate(runs):
        view = run_view(run, index)
        precond = effective_precond(view.semantic_config)
        if precond not in reviewed_preconds:
            kept.append(run)
            continue
        decision = _decision_for_run_view(view, is_ancestor)
        if not decision.eligible:
            excluded.append((run, decision))
            continue
        if precond == PRODUCT_PRECOND:
            kept.append(run)
            continue
        kept.append(project_run_semantics(
            run,
            {FACTORWISE_SLOT_FIELD: decision.revision},
            projection_id=FACTORWISE_SLOT_PROJECTION_ID,
            index=index,
        ))
    return tuple(kept), tuple(excluded)


def filter_paper_precond_cohort(
    runs: Sequence[Any],
    *,
    view_id: str,
    is_ancestor: Callable[[str, str], bool | None] = git_is_ancestor,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ViewSemanticDecision], ...]]:
    """Filter only the factorwise arm in ordinary paper views.

    Product, one-sided, and AdamW runs pass unchanged because the known slot
    change is not their view semantic.  Exclusions carry their exact decision so
    diagnostics and missing-arm notes consume one policy result instead of
    reimplementing it.  Input objects are returned unchanged and never mutated.
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
        # The policy reads optimizer semantics plus the one audit fact needed
        # for unversioned history; source hashes never enter the decision.
        decision = _decision_for_run_view(view, is_ancestor)
        if decision.eligible:
            kept.append(run)
        else:
            excluded.append((run, decision))
    return tuple(kept), tuple(excluded)


__all__ = [
    "FACTORWISE_SLOT_BOUNDARY",
    "FACTORWISE_SLOT_COHORT",
    "FACTORWISE_SLOT_FIELD",
    "FACTORWISE_SLOT_PRECONDS",
    "FACTORWISE_SLOT_PROJECTION_ID",
    "FACTORWISE_SLOT_VIEWS",
    "MATCHED_PRECOND_VIEW",
    "PRODUCT_PRECOND",
    "ViewSemanticDecision",
    "ViewSemanticMetadataError",
    "factorwise_slot_decision",
    "factorwise_slot_semantic_key",
    "filter_paper_precond_cohort",
    "git_is_ancestor",
    "project_paper_precond_cohort",
]
