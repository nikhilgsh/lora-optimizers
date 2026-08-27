"""Pure construction of plot-ready optimizer comparisons from loaded runs.

This module owns no I/O and imports no rendering code at module import time.
Callers normally provide immutable catalog records returned by
``loader.load_records``; existing ``(cfg, history)`` tuples remain accepted as
a compatibility input.  The builder then performs the one semantic reduction
that every plot needs:

* assign every run to exactly one variant;
* separate completed (including aborted) runs from in-flight runs;
* collapse same-LR completed replicates by their mean, after retaining only
  the longest trajectory cohort; and
* select each variant's best completed LR from those replicate means.

Human labels and style keys are presentation metadata.  All grouping and
selection is keyed exclusively by ``VariantSpec.id``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .leaderboard import is_final, mean_over_seeds


RunConfig = Mapping[str, Any]
History = Sequence[Mapping[str, Any]]
Run = tuple[RunConfig, History]
RunInput = Run | Any
VariantPredicate = Mapping[str, Any] | Callable[[RunConfig], bool]


@dataclass(frozen=True)
class VariantSpec:
    """Stable variant identity plus presentation metadata and a matcher.

    ``predicate`` may be a mapping with the same matching semantics as
    ``plotting.arms.pred_matches`` or a callable returning a boolean.  The
    callable form is the adapter for current label classifiers::

        VariantSpec("adamw", "AdamW", lambda cfg: key_fn(cfg) == "AdamW")

    ``id`` alone defines identity.  Changing ``label`` or ``style_key`` cannot
    change assignment, aggregation, or best-LR selection.
    """

    id: str
    label: str = field(compare=False)
    predicate: VariantPredicate = field(compare=False, repr=False)
    style_key: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("VariantSpec.id must be non-empty")
        if not callable(self.predicate) and not isinstance(self.predicate, Mapping):
            raise TypeError("VariantSpec.predicate must be a mapping or callable")
        if isinstance(self.predicate, Mapping):
            # A frozen dataclass should not change because its caller later
            # mutates the dict used to construct it.
            object.__setattr__(
                self, "predicate", MappingProxyType(dict(self.predicate))
            )


@dataclass(frozen=True)
class AggregatedCurve:
    """One variant/LR trajectory after continuation and replicate reduction."""

    variant_id: str
    lr: float
    final_loss: float
    cfg: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    last_step: int
    completed: bool
    n_replicates: int
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonResult:
    """Completed and partial LR curves, keyed only by stable variant id."""

    variants: tuple[VariantSpec, ...]
    completed: Mapping[str, Mapping[float, AggregatedCurve]]
    partials: Mapping[str, Mapping[float, AggregatedCurve]]
    best_completed: Mapping[str, AggregatedCurve | None]
    best_partial: Mapping[str, AggregatedCurve | None]
    unmatched_run_ids: tuple[str, ...]
    empty_history_run_ids: tuple[str, ...]


class AmbiguousVariantError(ValueError):
    """Raised when one or more runs match multiple variant predicates."""

    def __init__(
        self,
        ambiguities: Sequence[tuple[str, tuple[str, ...]]],
    ) -> None:
        self.ambiguities = tuple(ambiguities)
        details = "; ".join(
            f"{run_id} -> {list(variant_ids)!r}"
            for run_id, variant_ids in self.ambiguities
        )
        super().__init__(
            "each run must match exactly one variant; ambiguous matches: " + details
        )


class SemanticRevisionConflictError(ValueError):
    """One displayed variant contains multiple recorded semantics."""

    def __init__(
        self,
        variant_id: str,
        signatures: Mapping,
        lr: float | None = None,
    ):
        self.variant_id = variant_id
        self.lr = lr
        self.signatures = MappingProxyType(dict(signatures))
        location = (
            f"at lr={lr:g}" if lr is not None else "across selected LRs"
        )
        super().__init__(
            f"variant {variant_id!r} {location} mixes semantic revisions: "
            + "; ".join(
                f"{signature!r} -> {list(run_ids)!r}"
                for signature, run_ids in self.signatures.items()
            )
            + ". Split these into distinct VariantSpec.id values."
        )


@dataclass(frozen=True)
class _AssignedRun:
    cfg: dict[str, Any]
    history: list[dict[str, Any]]
    last: dict[str, Any]
    last_step: int
    run_id: str


def _comparison_input(run: RunInput, index: int) -> tuple[RunConfig, History]:
    """Normalize catalog records/lineages and legacy tuples at one boundary.

    Catalog objects contribute only their logged effective configuration plus
    the few physical/status fields comparison needs.  This deliberately avoids
    reintroducing the raw config as a second source of semantic defaults.
    """
    if isinstance(run, tuple) and len(run) == 2:
        return run

    if hasattr(run, "effective_config") and hasattr(run, "history"):
        cfg = dict(run.effective_config)
        raw = getattr(run, "raw_config", {})
        revisions = raw.get("semantic_revisions", {}) if isinstance(
            raw, Mapping
        ) else {}
        if isinstance(revisions, Mapping):
            cfg.setdefault("optimizer_impl_revision",
                           revisions.get("optimizer_impl"))
            cfg.setdefault("measurement_semantics_revision",
                           revisions.get("measurement"))
            cfg.setdefault("data_pipeline_version",
                           revisions.get("data_pipeline"))
        cfg["run_id"] = str(getattr(run, "physical_id", f"run[{index}]"))
        group = getattr(run, "group", None)
        filename = getattr(run, "log_filename", None)
        if group is not None:
            cfg["log_group"] = group
        if filename is not None:
            cfg["_log_filename"] = filename
        if isinstance(raw, Mapping) and raw.get("_aborted") is not None:
            cfg["_aborted"] = raw["_aborted"]
        return cfg, run.history

    if (hasattr(run, "semantic_config") and hasattr(run, "cfg")
            and hasattr(run, "history")):
        cfg = dict(run.semantic_config)
        raw = run.cfg
        revisions = getattr(run, "semantic_revisions", {})
        if isinstance(revisions, Mapping):
            cfg.setdefault("optimizer_impl_revision",
                           revisions.get("optimizer_impl"))
            cfg.setdefault("measurement_semantics_revision",
                           revisions.get("measurement"))
            cfg.setdefault("data_pipeline_version",
                           revisions.get("data_pipeline"))
        cfg["run_id"] = str(
            getattr(run, "terminal_attempt_id", f"run[{index}]")
        )
        if isinstance(raw, Mapping):
            for field in ("log_group", "_log_filename", "_aborted"):
                if raw.get(field) is not None:
                    cfg[field] = raw[field]
        return cfg, run.history

    raise TypeError(
        "runs must contain (cfg, history) tuples, RunRecord objects, or "
        "MergedRunLineage objects"
    )


def _run_id(cfg: RunConfig, index: int) -> str:
    if cfg.get("run_id") is not None:
        return str(cfg["run_id"])
    group = cfg.get("log_group")
    filename = cfg.get("_log_filename")
    if group is not None or filename is not None:
        return f"{group or '?'}:{filename or '?'}"
    return f"run[{index}]"


def _matches(spec: VariantSpec, cfg: RunConfig) -> bool:
    predicate = spec.predicate
    if callable(predicate):
        return bool(predicate(cfg))
    return all(_field_matches(cfg, field, want)
               for field, want in predicate.items())


def _field_matches(cfg: RunConfig, field: str, want: Any) -> bool:
    """Match one arm-style predicate without depending on plotting code.

    This mirrors the existing public predicate semantics during migration:
    callables test the loaded value, a collection means membership for scalar
    values, and collection-valued configs compare literally.
    """
    if field not in cfg:
        return False
    value = cfg[field]
    if callable(want):
        return bool(want(value))
    collection_types = (list, set, tuple, frozenset)
    if isinstance(want, collection_types):
        if isinstance(value, collection_types):
            return list(value) == list(want)
        return value in want
    return value == want


def _freeze_nested(
    values: Mapping[str, Mapping[float, AggregatedCurve]],
) -> Mapping[str, Mapping[float, AggregatedCurve]]:
    return MappingProxyType({
        variant_id: MappingProxyType(dict(by_lr))
        for variant_id, by_lr in values.items()
    })


def _best_curve(
    by_lr: Mapping[float, AggregatedCurve],
) -> AggregatedCurve | None:
    if not by_lr:
        return None
    finite = [curve for curve in by_lr.values() if math.isfinite(curve.final_loss)]
    if finite:
        return min(finite, key=lambda curve: (curve.final_loss, curve.lr))
    # All values are NaN/inf (e.g. every completed run aborted).  Keep a
    # deterministic representative so divergence remains visible.
    return min(by_lr.values(), key=lambda curve: curve.lr)


def _aggregate_completed(
    variant_id: str,
    lr: float,
    members: Sequence[_AssignedRun],
) -> AggregatedCurve:
    longest = max(member.last_step for member in members)
    cohort = [member for member in members if member.last_step == longest]
    mean_final, merged = mean_over_seeds([
        (member.history, member.last, member.last_step) for member in cohort
    ])
    representative = cohort[0]
    return AggregatedCurve(
        variant_id=variant_id,
        lr=lr,
        final_loss=float(mean_final),
        cfg=MappingProxyType(dict(representative.cfg)),
        history=tuple(MappingProxyType(dict(event)) for event in merged),
        last_step=longest,
        completed=True,
        n_replicates=len(cohort),
        run_ids=tuple(member.run_id for member in cohort),
    )


def _semantic_signature(cfg: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Scalar revision identity visible to both new and compatibility paths."""
    return (
        cfg.get("optimizer_impl_revision"),
        cfg.get("measurement_semantics_revision"),
        cfg.get("data_pipeline_version"),
    )


def _require_one_semantic_signature(
    variant_id: str,
    members: Sequence[_AssignedRun],
    lr: float | None = None,
) -> None:
    signatures: dict[tuple[Any, Any, Any], list[str]] = {}
    for member in members:
        signatures.setdefault(_semantic_signature(member.cfg), []).append(
            member.run_id
        )
    if len(signatures) > 1:
        raise SemanticRevisionConflictError(
            variant_id,
            {signature: tuple(run_ids)
             for signature, run_ids in signatures.items()},
            lr=lr,
        )


def _partial_curve(
    variant_id: str,
    lr: float,
    member: _AssignedRun,
) -> AggregatedCurve:
    return AggregatedCurve(
        variant_id=variant_id,
        lr=lr,
        final_loss=float(member.last["eval_loss"]),
        cfg=MappingProxyType(dict(member.cfg)),
        history=tuple(MappingProxyType(dict(event)) for event in member.history),
        last_step=member.last_step,
        completed=False,
        n_replicates=1,
        run_ids=(member.run_id,),
    )


def build_comparison(
    runs: Sequence[RunInput],
    variants: Sequence[VariantSpec],
    horizon: int,
    completion_slack: int = 300,
) -> ComparisonResult:
    """Assign and reduce loaded runs into a plot-ready comparison.

    Every run's predicate is evaluated against every variant.  Unmatched runs
    are recorded; any multi-match raises :class:`AmbiguousVariantError` after
    the full input has been audited.

    Completed buckets follow ``leaderboard.labeled_completed_runs`` semantics:
    only the greatest ``last_step`` cohort at a (variant, LR) survives, and
    same-step members are replicate-averaged with ``mean_over_seeds``.  A short
    run carrying ``cfg['_aborted']`` is completed-but-diverged, matching the
    current prefetched plotting path.  In-flight runs remain available in
    ``partials``; per (variant, LR), the first most-progressed run is retained.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if completion_slack < 0:
        raise ValueError(
            f"completion_slack must be non-negative, got {completion_slack}"
        )

    specs = tuple(variants)
    ids = [spec.id for spec in specs]
    duplicates = sorted({
        variant_id for variant_id in ids if ids.count(variant_id) > 1
    })
    if duplicates:
        raise ValueError(f"duplicate VariantSpec.id value(s): {duplicates}")

    completed_members: dict[tuple[str, float], list[_AssignedRun]] = {}
    partial_members: dict[tuple[str, float], list[_AssignedRun]] = {}
    unmatched: list[str] = []
    empty_history: list[str] = []
    ambiguities: list[tuple[str, tuple[str, ...]]] = []

    for index, raw_run in enumerate(runs):
        raw_cfg, raw_history = _comparison_input(raw_run, index)
        run_id = _run_id(raw_cfg, index)
        matches = tuple(spec.id for spec in specs if _matches(spec, raw_cfg))
        if not matches:
            unmatched.append(run_id)
            continue
        if len(matches) > 1:
            ambiguities.append((run_id, matches))
            continue
        variant_id = matches[0]

        if not raw_history:
            empty_history.append(run_id)
            continue
        history = [dict(event) for event in raw_history]
        history.sort(key=lambda event: event.get("step", 0) or 0)
        last = max(history, key=lambda event: event.get("step", 0) or 0)
        if "eval_loss" not in last:
            raise ValueError(f"{run_id} last history event has no eval_loss")
        last_step = int(last.get("step", 0) or 0)
        try:
            lr = float(raw_cfg["lr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{run_id} has no numeric lr") from exc

        member = _AssignedRun(
            cfg=dict(raw_cfg),
            history=history,
            last=last,
            last_step=last_step,
            run_id=run_id,
        )
        aborted = raw_cfg.get("_aborted") is not None
        key = (variant_id, lr)
        if is_final(last_step, horizon, completion_slack) or aborted:
            completed_members.setdefault(key, []).append(member)
        else:
            partial_members.setdefault(key, []).append(member)

    if ambiguities:
        raise AmbiguousVariantError(ambiguities)

    # One displayed optimizer curve cannot splice implementation or
    # measurement semantics across learning rates. This is deliberately wider
    # than per-(variant, LR) replicate validation: best-LR selection itself is
    # a cross-LR comparison.
    for variant_id in ids:
        semantic_members = [
            member
            for (member_variant, _lr), members in (
                list(completed_members.items()) + list(partial_members.items())
            )
            if member_variant == variant_id
            for member in members
        ]
        _require_one_semantic_signature(variant_id, semantic_members)

    completed: dict[str, dict[float, AggregatedCurve]] = {
        spec.id: {} for spec in specs
    }
    partials: dict[str, dict[float, AggregatedCurve]] = {
        spec.id: {} for spec in specs
    }
    for (variant_id, lr), members in completed_members.items():
        completed[variant_id][lr] = _aggregate_completed(
            variant_id, lr, members
        )
    for (variant_id, lr), members in partial_members.items():
        member = max(members, key=lambda candidate: candidate.last_step)
        partials[variant_id][lr] = _partial_curve(variant_id, lr, member)

    best_completed = MappingProxyType({
        spec.id: _best_curve(completed[spec.id]) for spec in specs
    })
    best_partial = MappingProxyType({
        spec.id: _best_curve(partials[spec.id]) for spec in specs
    })
    return ComparisonResult(
        variants=specs,
        completed=_freeze_nested(completed),
        partials=_freeze_nested(partials),
        best_completed=best_completed,
        best_partial=best_partial,
        unmatched_run_ids=tuple(unmatched),
        empty_history_run_ids=tuple(empty_history),
    )
