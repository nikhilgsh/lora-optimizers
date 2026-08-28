"""Immutable run records at the raw-log/catalog boundary.

This module intentionally contains no legacy default reconstruction.  A record
keeps the parser output verbatim (modulo immutable containers), separates audit
metadata from values that describe the executed workload, and only resolves
effective values from schema blocks that were themselves logged by the run.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


# Cfg fields that describe physical execution/provenance rather than the
# algorithm or workload. This neutral boundary is the source of truth for
# catalog records, legacy loader dedup, and plotting collision checks.
RUNTIME_FIELDS: frozenset[str] = frozenset({
    "git_commit", "command", "log_group",
    "git_dirty", "git_status", "git_diff_sha", "git_untracked_files",
    "execution_source_sha", "execution_source_paths",
    "execution_source_dirty", "execution_env", "execution_env_sha",
    "run_id", "_log_filename",
    "run_schema_version", "attempt_id", "resume_parent_attempt_id",
    "checkpoint_identity", "_resume",
    "wandb_project", "wandb_run_name",
    "device", "tf32", "no_tf32",
    "log_basic_diagnostics", "log_heavy_diagnostics",
    "log_optim_diagnostics", "no-log_optim_diagnostics",
    "optim_diagnostics_every", "diagnostics",
    "profile_steps", "profile_dir", "_optim_steps",
    "train_file", "eval_file",
    "checkpoint_dir", "resume_from", "checkpoint_every",
    # `keep_checkpoints` only decides whether train.py rmtree's the checkpoint
    # directory after a clean finish (train.py:2256-2264). It is retention
    # bookkeeping like its four siblings above -- it cannot reach the optimizer
    # or the loss -- so two runs that differ only on it are one series.
    "checkpoint_keep_last", "keep_checkpoints", "picard_iters_override",
})


_AUDIT_FIELDS = frozenset({
    "event", "command", "run_id", "log_group", "_log_filename",
    "run_schema_version", "attempt_id", "resume_parent_attempt_id",
    "checkpoint_identity", "semantic_revisions",
    "git_commit", "git_dirty", "git_status", "git_diff_sha", "git_untracked_files",
    "execution_source_sha", "execution_source_paths",
    "execution_source_dirty", "execution_env", "execution_env_sha",
    "device", "wandb_project", "wandb_run_name",
}) | RUNTIME_FIELDS

_LOGGED_CONFIG_BLOCKS = (
    "_cli_args",
    "optimizer_config",
    "optimizer_variant_semantics",
    "optimizer_effective",
)


def _is_logged_semantic_field(key: Any) -> bool:
    return (
        isinstance(key, str)
        and key not in _AUDIT_FIELDS
        and key not in _LOGGED_CONFIG_BLOCKS
        and not key.startswith("_")
    )


# `logged_effective_value`'s authority order: the logged blocks in
# `_LOGGED_CONFIG_BLOCKS` order, with `None` marking the raw top level.
_LOGGED_VALUE_ORDER = (
    "_cli_args", "optimizer_config", None, "optimizer_effective",
)


def logged_effective_value(
    raw_config: Mapping[str, Any], field: str
) -> tuple[bool, Any]:
    """Resolve one recorded semantic field in canonical authority order."""
    if not _is_logged_semantic_field(field):
        return False, None
    # Later wins. The order is `_LOGGED_CONFIG_BLOCKS` with the raw top level
    # spliced in where `logged_effective_config` places it, so the two cannot
    # disagree about authority if a block is added or reordered. This stays a
    # per-field walk rather than a call to `logged_effective_config`: the run
    # catalog resolves a handful of header fields per run and must not flatten
    # every cfg to do it.
    present = False
    value = None
    for block_name in _LOGGED_VALUE_ORDER:
        block = raw_config if block_name is None else raw_config.get(block_name)
        if isinstance(block, Mapping) and field in block:
            present = True
            value = block[field]
    return present, value


def freeze_value(value: Any) -> Any:
    """Recursively copy JSON-like data into immutable containers."""
    value_type = type(value)
    # Run configs and events are JSON-shaped, so nearly every leaf is one of
    # these exact scalar types.  Return those before the abstract-container
    # checks below: header screening visits millions of scalar leaves while
    # building a cold catalog, and collections.abc instance checks dominated
    # that path without changing any value.
    if value is None or value_type in (str, int, float, bool):
        return value
    if value_type is dict:
        return MappingProxyType({key: freeze_value(item)
                                 for key, item in value.items()})
    if value_type is list or value_type is tuple:
        return tuple(freeze_value(item) for item in value)
    if value_type is set or value_type is frozenset:
        return frozenset(freeze_value(item) for item in value)
    # Preserve support for non-builtin Mapping/sequence containers used by
    # callers outside the JSON parser.
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_value(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


def thaw_value(value: Any) -> Any:
    """Return mutable JSON-style copies suitable for legacy consumers."""
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RunIssue:
    """A machine-readable catalog problem with human-readable context."""

    code: str
    message: str
    source: str


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    """Physical and audit-only metadata, separate from optimizer semantics."""

    group: str
    log_filename: str | None
    config: Mapping[str, Any]
    manifest: Mapping[str, Any] | None


def physical_run_id(
    raw_config: Mapping[str, Any],
    *,
    group: str,
    log_filename: str | None,
    fallback_index: int,
) -> str:
    """Stable physical identity, preferring an explicitly logged run ID."""
    explicit = raw_config.get("run_id")
    if explicit is not None:
        if isinstance(explicit, str):
            return explicit
        try:
            return json.dumps(
                thaw_value(explicit), sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            return repr(explicit)
    if log_filename:
        return f"{group}/{log_filename}"
    # Existing parser output should always carry _log_filename.  Keep a stable
    # deterministic fallback and pair it with a structured issue when it does
    # not, rather than synthesizing semantic provenance.
    return f"{group}/run[{fallback_index}]"


def logged_effective_config(
    raw_config: Mapping[str, Any],
    *,
    source: str,
) -> tuple[Mapping[str, Any], tuple[RunIssue, ...]]:
    """Flatten only values recorded by the run, in authority order.

    ``_cli_args`` records the invoked argument values, ``optimizer_config``
    records the built optimizer attributes, named top-level fields include the
    training/workload schema, and ``optimizer_effective`` records resolved
    short-circuit behavior.  Later sources therefore override earlier ones.
    Missing values stay missing: there is no import of argparse, optimizer
    registries, or current defaults.
    """
    effective: dict[str, Any] = {}
    issues: list[RunIssue] = []

    for block_name in ("_cli_args", "optimizer_config"):
        block = raw_config.get(block_name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            issues.append(RunIssue(
                code="invalid_config_block",
                message=f"{block_name} is not an object and was not flattened",
                source=source,
            ))
            continue
        effective.update({key: value for key, value in block.items()
                          if _is_logged_semantic_field(key)})

    for key, value in raw_config.items():
        if not _is_logged_semantic_field(key):
            continue
        effective[key] = value

    block = raw_config.get("optimizer_effective")
    if block is not None:
        if isinstance(block, Mapping):
            effective.update({key: value for key, value in block.items()
                              if _is_logged_semantic_field(key)})
        else:
            issues.append(RunIssue(
                code="invalid_config_block",
                message=("optimizer_effective is not an object and was not "
                         "used as effective configuration"),
                source=source,
            ))
    return freeze_value(effective), tuple(issues)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One immutable physical run plus its logged semantic interpretation."""

    physical_id: str
    raw_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    audit_provenance: AuditProvenance
    effective_config: Mapping[str, Any]
    issues: tuple[RunIssue, ...] = ()

    @property
    def group(self) -> str:
        return self.audit_provenance.group

    @property
    def log_filename(self) -> str | None:
        return self.audit_provenance.log_filename

    @property
    def semantic_config(self) -> Mapping[str, Any]:
        """Alias emphasizing that ``effective_config`` drives comparison."""
        return self.effective_config

    @classmethod
    def from_parsed(
        cls,
        raw_config: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        *,
        group: str,
        manifest: Mapping[str, Any] | None,
        issues: Sequence[RunIssue] = (),
        fallback_index: int = 0,
    ) -> "RunRecord":
        """Build a record without mutating or default-filling parser output."""
        raw_frozen = freeze_value(raw_config)
        log_filename = raw_frozen.get("_log_filename")
        source = (f"{group}/{log_filename}" if log_filename
                  else f"{group}/run[{fallback_index}]")
        record_issues = list(issues)
        if not log_filename:
            record_issues.append(RunIssue(
                code="missing_log_filename",
                message="parser returned a run without _log_filename",
                source=source,
            ))

        effective, config_issues = logged_effective_config(
            raw_frozen, source=source
        )
        record_issues.extend(config_issues)
        audit_config = {
            key: value for key, value in raw_frozen.items()
            if key in _AUDIT_FIELDS
        }
        provenance = AuditProvenance(
            group=group,
            log_filename=log_filename,
            config=freeze_value(audit_config),
            manifest=(None if manifest is None else freeze_value(manifest)),
        )
        return cls(
            physical_id=physical_run_id(
                raw_frozen,
                group=group,
                log_filename=log_filename,
                fallback_index=fallback_index,
            ),
            raw_config=raw_frozen,
            history=tuple(freeze_value(event) for event in history),
            audit_provenance=provenance,
            effective_config=effective,
            issues=tuple(record_issues),
        )

    def as_legacy_tuple(self) -> tuple[dict, list[dict]]:
        """Mutable ``(cfg, history)`` copies for existing plotting call sites."""
        cfg = thaw_value(self.raw_config)
        cfg.setdefault("log_group", self.group)
        if self.log_filename is not None:
            cfg.setdefault("_log_filename", self.log_filename)
        cfg.setdefault("run_id", self.physical_id)
        return cfg, thaw_value(self.history)


@dataclass(frozen=True, slots=True)
class RunView:
    """Public, read-only view over every supported run representation.

    Semantic inputs, recorded audit metadata, and the raw config deliberately
    remain separate.  Consumers must choose the surface appropriate to their
    decision instead of merging provenance fields into optimizer identity.
    """

    semantic_config: Mapping[str, Any]
    audit_config: Mapping[str, Any]
    raw_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    physical_id: str
    group: str | None
    log_filename: str | None
    semantic_revisions: Mapping[str, Any]
    run_schema_version: Any = None

    @property
    def is_versioned(self) -> bool:
        """Whether the producer recorded an explicit run schema version."""
        return self.run_schema_version is not None


@dataclass(frozen=True, slots=True)
class SemanticRunProjection:
    """A run with a narrow, named view-semantic overlay.

    The underlying raw/audit provenance and physical identity are unchanged.
    This is for reviewed consumer semantics such as a historical paper cohort,
    not for reconstructing missing execution configuration.
    """

    physical_id: str
    effective_config: Mapping[str, Any]
    raw_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    group: str | None
    log_filename: str | None
    semantic_revisions: Mapping[str, Any]
    projection_id: str


def _audit_config(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return freeze_value({
        key: value for key, value in raw_config.items() if key in _AUDIT_FIELDS
    })


def _view_revisions(
    raw_config: Mapping[str, Any],
    explicit: Any = None,
) -> Mapping[str, Any]:
    value = explicit if isinstance(explicit, Mapping) else raw_config.get(
        "semantic_revisions", {}
    )
    return freeze_value(value) if isinstance(value, Mapping) else freeze_value({})


def run_view(run: Any, index: int = 0) -> RunView:
    """Adapt a record, merged lineage, or legacy tuple without conflation.

    New consumers should use this boundary instead of private plotting
    normalizers.  Legacy tuples are interpreted only from values already in
    their config; this function never imports current defaults or registries.
    """
    if isinstance(run, RunView):
        return run

    if isinstance(run, RunRecord):
        return RunView(
            semantic_config=run.semantic_config,
            audit_config=run.audit_provenance.config,
            raw_config=run.raw_config,
            history=run.history,
            physical_id=run.physical_id,
            group=run.group,
            log_filename=run.log_filename,
            semantic_revisions=_view_revisions(run.raw_config),
            run_schema_version=run.raw_config.get("run_schema_version"),
        )

    # Immutable projections (for example a sealed historical publication
    # archive) use the same explicit record contract without pretending to be
    # a physical catalog record.
    if (
        hasattr(run, "effective_config")
        and hasattr(run, "raw_config")
        and hasattr(run, "history")
        and hasattr(run, "physical_id")
    ):
        raw = run.raw_config
        if not isinstance(raw, Mapping):
            raise TypeError("projected run raw_config must be a mapping")
        group = getattr(run, "group", None)
        filename = getattr(run, "log_filename", None)
        return RunView(
            semantic_config=freeze_value(dict(run.effective_config)),
            audit_config=_audit_config(raw),
            raw_config=freeze_value(dict(raw)),
            history=tuple(freeze_value(dict(event)) for event in run.history),
            physical_id=str(run.physical_id),
            group=group,
            log_filename=filename,
            semantic_revisions=_view_revisions(
                raw, getattr(run, "semantic_revisions", None)
            ),
            run_schema_version=raw.get("run_schema_version"),
        )

    # Avoid importing run_lineage here: it already imports this module.  The
    # explicit attribute contract distinguishes a merged lineage from a raw
    # record and from a legacy tuple.
    if (
        hasattr(run, "semantic_config")
        and hasattr(run, "semantic_revisions")
        and hasattr(run, "cfg")
        and hasattr(run, "history")
        and hasattr(run, "terminal_attempt_id")
    ):
        raw = run.cfg
        if not isinstance(raw, Mapping):
            raise TypeError("merged lineage cfg must be a mapping")
        group = getattr(run, "terminal_group", None)
        filename = getattr(run, "terminal_log_filename", None)
        return RunView(
            semantic_config=freeze_value(dict(run.semantic_config)),
            audit_config=_audit_config(raw),
            raw_config=freeze_value(dict(raw)),
            history=tuple(freeze_value(dict(event)) for event in run.history),
            physical_id=str(run.terminal_attempt_id),
            group=group,
            log_filename=filename,
            semantic_revisions=_view_revisions(
                raw, getattr(run, "semantic_revisions", None)
            ),
            run_schema_version=raw.get("run_schema_version"),
        )

    if isinstance(run, tuple) and len(run) == 2:
        raw, history = run
        if not isinstance(raw, Mapping):
            raise TypeError("legacy run config must be a mapping")
        if not isinstance(history, Sequence):
            raise TypeError("legacy run history must be a sequence")
        group_value = raw.get("log_group")
        group = str(group_value) if group_value is not None else None
        filename_value = raw.get("_log_filename")
        filename = str(filename_value) if filename_value is not None else None
        source = filename or f"legacy run[{index}]"
        semantic, _issues = logged_effective_config(raw, source=source)
        fallback_group = group or "legacy"
        return RunView(
            semantic_config=semantic,
            audit_config=_audit_config(raw),
            raw_config=freeze_value(dict(raw)),
            history=tuple(freeze_value(dict(event)) for event in history),
            physical_id=physical_run_id(
                raw,
                group=fallback_group,
                log_filename=filename,
                fallback_index=index,
            ),
            group=group,
            log_filename=filename,
            semantic_revisions=_view_revisions(raw),
            run_schema_version=raw.get("run_schema_version"),
        )

    raise TypeError(
        "run must be a RunRecord, MergedRunLineage, RunView, or "
        "(cfg, history) tuple"
    )


def project_run_semantics(
    run: Any,
    overlay: Mapping[str, Any],
    *,
    projection_id: str,
    index: int = 0,
) -> SemanticRunProjection:
    """Return an immutable semantic overlay without changing run provenance."""
    if not isinstance(overlay, Mapping):
        raise TypeError("overlay must be a mapping")
    if not isinstance(projection_id, str) or not projection_id.strip():
        raise ValueError("projection_id must be a non-empty string")
    invalid = sorted(
        key for key in overlay
        if not isinstance(key, str) or key in _AUDIT_FIELDS or key.startswith("_")
    )
    if invalid:
        raise ValueError(
            "semantic overlays cannot contain audit/private fields: "
            f"{invalid!r}"
        )

    view = run_view(run, index)
    config = dict(view.semantic_config)
    conflicts = {
        key: (config[key], value)
        for key, value in overlay.items()
        if key in config and config[key] != value
    }
    if conflicts:
        raise ValueError(
            f"semantic projection {projection_id!r} would overwrite recorded "
            f"values: {conflicts!r}"
        )
    config.update(overlay)
    return SemanticRunProjection(
        physical_id=view.physical_id,
        effective_config=freeze_value(config),
        raw_config=view.raw_config,
        history=view.history,
        group=view.group,
        log_filename=view.log_filename,
        semantic_revisions=view.semantic_revisions,
        projection_id=projection_id,
    )


__all__ = [
    "AuditProvenance",
    "RUNTIME_FIELDS",
    "RunIssue",
    "RunRecord",
    "SemanticRunProjection",
    "RunView",
    "freeze_value",
    "logged_effective_config",
    "logged_effective_value",
    "physical_run_id",
    "project_run_semantics",
    "run_view",
    "thaw_value",
]
