"""Pure validation and merging of explicit run-continuation lineages.

Disjoint step ranges are never lineage evidence.  A child is merged only when
it names its parent and agrees with that parent on the recorded checkpoint
lineage, semantic configuration, and semantic revisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .run_records import freeze_value, thaw_value


@dataclass(frozen=True)
class LineageIssue:
    """Machine-readable explanation for a lineage failure."""

    code: str
    attempt_id: str | None
    parent_attempt_id: str | None
    details: Mapping[str, Any]


class RunLineageError(ValueError):
    """Base error carrying a structured :class:`LineageIssue`."""

    def __init__(self, issue: LineageIssue):
        self.issue = issue
        super().__init__(
            f"run lineage {issue.code}: attempt={issue.attempt_id!r}, "
            f"parent={issue.parent_attempt_id!r}, details={dict(issue.details)!r}"
        )


class LineageStructureError(RunLineageError):
    """The declared attempt graph is incomplete, duplicate, cyclic, or branched."""


class LineageMismatchError(RunLineageError):
    """A declared child disagrees with its parent on resume semantics."""


@dataclass(frozen=True)
class RunSegment:
    """One execution attempt with its original cfg/history provenance."""

    attempt_id: str
    resume_parent_attempt_id: str | None
    checkpoint_identity: str
    semantic_config: Mapping[str, Any]
    semantic_revisions: Mapping[str, Any]
    cfg: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    input_ordinal: int


@dataclass(frozen=True)
class MergedRunLineage:
    """One validated chain plus a plot-compatible merged trajectory."""

    root_attempt_id: str
    terminal_attempt_id: str
    checkpoint_identity: str
    semantic_config: Mapping[str, Any]
    semantic_revisions: Mapping[str, Any]
    cfg: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    event_attempt_ids: tuple[str, ...]
    segments: tuple[RunSegment, ...]

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(segment.attempt_id for segment in self.segments)


def _fail(error_type, code: str, attempt_id=None, parent_attempt_id=None,
          **details):
    raise error_type(LineageIssue(
        code=code,
        attempt_id=attempt_id,
        parent_attempt_id=parent_attempt_id,
        details=freeze_value(details),
    ))


def _unpack(run: Any, ordinal: int):
    if isinstance(run, tuple) and len(run) == 2:
        return run
    if hasattr(run, "cfg") and hasattr(run, "history"):
        return run.cfg, run.history
    if hasattr(run, "raw_config") and hasattr(run, "history"):
        return run.raw_config, run.history
    _fail(LineageStructureError, "invalid_run_shape",
          input_ordinal=ordinal,
          expected=("(cfg, history) or object exposing .cfg/.history or "
                    ".raw_config/.history"))


def _segment(run: Any, ordinal: int) -> RunSegment:
    raw_cfg, raw_history = _unpack(run, ordinal)
    if not isinstance(raw_cfg, Mapping):
        _fail(LineageStructureError, "invalid_config", input_ordinal=ordinal)
    cfg = thaw_value(raw_cfg)
    attempt_hint = cfg.get("attempt_id")
    required = (
        "attempt_id", "resume_parent_attempt_id", "checkpoint_identity",
        "semantic_config", "semantic_revisions",
    )
    missing = [field for field in required if field not in cfg]
    if missing:
        _fail(LineageStructureError, "missing_field", attempt_hint,
              missing_fields=tuple(missing), input_ordinal=ordinal)

    attempt_id = cfg["attempt_id"]
    parent_id = cfg["resume_parent_attempt_id"]
    checkpoint = cfg["checkpoint_identity"]
    if not isinstance(attempt_id, str) or not attempt_id:
        _fail(LineageStructureError, "invalid_attempt_id",
              value=attempt_id, input_ordinal=ordinal)
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        _fail(LineageStructureError, "invalid_parent_attempt_id", attempt_id,
              value=parent_id)
    if not isinstance(checkpoint, str) or not checkpoint:
        _fail(LineageStructureError, "invalid_checkpoint_identity", attempt_id,
              parent_id, value=checkpoint)
    if not isinstance(cfg["semantic_config"], Mapping):
        _fail(LineageStructureError, "invalid_semantic_config", attempt_id,
              parent_id)
    if not isinstance(cfg["semantic_revisions"], Mapping):
        _fail(LineageStructureError, "invalid_semantic_revisions", attempt_id,
              parent_id)
    if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
        _fail(LineageStructureError, "invalid_history", attempt_id, parent_id)

    events = []
    for event_ordinal, raw_event in enumerate(raw_history):
        if (not isinstance(raw_event, Mapping)
                or not isinstance(raw_event.get("step"), int)):
            _fail(LineageStructureError, "invalid_history_event", attempt_id,
                  parent_id, event_ordinal=event_ordinal)
        events.append(freeze_value(raw_event))

    return RunSegment(
        attempt_id=attempt_id,
        resume_parent_attempt_id=parent_id,
        checkpoint_identity=checkpoint,
        semantic_config=freeze_value(cfg["semantic_config"]),
        semantic_revisions=freeze_value(cfg["semantic_revisions"]),
        cfg=freeze_value(cfg),
        history=tuple(events),
        input_ordinal=ordinal,
    )


def _differences(parent: Mapping[str, Any], child: Mapping[str, Any]):
    missing = "<missing>"
    out = {}
    for field in sorted(set(parent) | set(child)):
        parent_value = parent.get(field, missing)
        child_value = child.get(field, missing)
        if parent_value != child_value:
            out[field] = {"parent": parent_value, "child": child_value}
    return out


def _validate_edge(parent: RunSegment, child: RunSegment) -> None:
    common = (child.attempt_id, parent.attempt_id)
    if child.checkpoint_identity != parent.checkpoint_identity:
        _fail(LineageMismatchError, "checkpoint_identity_mismatch", *common,
              parent_value=parent.checkpoint_identity,
              child_value=child.checkpoint_identity)
    config_diff = _differences(parent.semantic_config, child.semantic_config)
    if config_diff:
        _fail(LineageMismatchError, "semantic_config_mismatch", *common,
              differences=config_diff)
    revision_diff = _differences(
        parent.semantic_revisions, child.semantic_revisions,
    )
    if revision_diff:
        _fail(LineageMismatchError, "semantic_revision_mismatch", *common,
              differences=revision_diff)


def _merge(chain: Sequence[RunSegment]) -> MergedRunLineage:
    merged: dict[int, tuple[Mapping[str, Any], str]] = {}
    for index, segment in enumerate(chain):
        local = {event["step"]: event for event in segment.history}
        if index and local:
            replay_start = min(local)
            merged = {step: value for step, value in merged.items()
                      if step < replay_start}
        merged.update({step: (event, segment.attempt_id)
                       for step, event in local.items()})
    ordered = [merged[step] for step in sorted(merged)]
    terminal = chain[-1]
    return MergedRunLineage(
        root_attempt_id=chain[0].attempt_id,
        terminal_attempt_id=terminal.attempt_id,
        checkpoint_identity=terminal.checkpoint_identity,
        semantic_config=terminal.semantic_config,
        semantic_revisions=terminal.semantic_revisions,
        cfg=terminal.cfg,
        history=tuple(event for event, _attempt_id in ordered),
        event_attempt_ids=tuple(attempt_id for _event, attempt_id in ordered),
        segments=tuple(chain),
    )


def build_run_lineages(runs: Sequence[Any]) -> tuple[MergedRunLineage, ...]:
    """Validate explicit links and merge only root-to-leaf resume chains.

    Every cfg must explicitly record ``attempt_id``,
    ``resume_parent_attempt_id``, ``checkpoint_identity``, ``semantic_config``,
    and ``semantic_revisions``.  Independent roots remain separate even when
    their semantic values match and their histories are disjoint.
    """
    segments = tuple(_segment(run, ordinal) for ordinal, run in enumerate(runs))
    by_id: dict[str, RunSegment] = {}
    for segment in segments:
        if segment.attempt_id in by_id:
            _fail(LineageStructureError, "duplicate_attempt_id",
                  segment.attempt_id)
        by_id[segment.attempt_id] = segment

    children: dict[str, list[RunSegment]] = {key: [] for key in by_id}
    for child in segments:
        parent_id = child.resume_parent_attempt_id
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            _fail(LineageStructureError, "missing_parent_attempt",
                  child.attempt_id, parent_id)
        _validate_edge(parent, child)
        children[parent_id].append(child)

    for parent_id, direct_children in children.items():
        if len(direct_children) > 1:
            _fail(LineageStructureError, "branching_lineage", parent_id,
                  child_attempt_ids=tuple(c.attempt_id for c in direct_children))

    # Following parent pointers from every node must reach a root.
    for segment in segments:
        cursor, seen = segment, set()
        while cursor.resume_parent_attempt_id is not None:
            if cursor.attempt_id in seen:
                _fail(LineageStructureError, "lineage_cycle",
                      segment.attempt_id, cursor.resume_parent_attempt_id,
                      cycle_attempt_ids=tuple(sorted(seen)))
            seen.add(cursor.attempt_id)
            cursor = by_id[cursor.resume_parent_attempt_id]

    roots = sorted(
        (segment for segment in segments
         if segment.resume_parent_attempt_id is None),
        key=lambda segment: segment.input_ordinal,
    )
    lineages = []
    for root in roots:
        chain, cursor = [root], root
        while children[cursor.attempt_id]:
            cursor = children[cursor.attempt_id][0]
            chain.append(cursor)
        lineages.append(_merge(chain))
    return tuple(lineages)


__all__ = [
    "LineageIssue", "LineageMismatchError", "LineageStructureError",
    "MergedRunLineage", "RunLineageError", "RunSegment",
    "build_run_lineages",
]
