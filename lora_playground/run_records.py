"""Immutable run records at the raw-log/catalog boundary.

This module intentionally contains no legacy default reconstruction.  A record
keeps the parser output verbatim (modulo immutable containers), separates audit
metadata from values that describe the executed workload, and only resolves
effective values from schema blocks that were themselves logged by the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


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
    "checkpoint_keep_last", "picard_iters_override",
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
    "optimizer_effective",
)


def freeze_value(value: Any) -> Any:
    """Recursively copy JSON-like data into immutable containers."""
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
            return json.dumps(explicit, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return repr(explicit)
    if log_filename:
        return f"{group}/{log_filename}"
    # Existing parser output should always carry _log_filename.  Keep a stable
    # deterministic fallback and pair it with a structured issue when it does
    # not, rather than synthesizing semantic provenance.
    return f"{group}/run[{fallback_index}]"


def _logged_effective_config(
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

    def is_semantic_field(key: Any) -> bool:
        return (
            isinstance(key, str)
            and key not in _AUDIT_FIELDS
            and key not in _LOGGED_CONFIG_BLOCKS
            and not key.startswith("_")
        )

    for block_name in _LOGGED_CONFIG_BLOCKS[:2]:
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
                          if is_semantic_field(key)})

    for key, value in raw_config.items():
        if not is_semantic_field(key):
            continue
        effective[key] = value

    block = raw_config.get("optimizer_effective")
    if block is not None:
        if isinstance(block, Mapping):
            effective.update({key: value for key, value in block.items()
                              if is_semantic_field(key)})
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
        raw_copy = {key: thaw_value(freeze_value(value))
                    for key, value in raw_config.items()}
        log_filename = raw_copy.get("_log_filename")
        source = (f"{group}/{log_filename}" if log_filename
                  else f"{group}/run[{fallback_index}]")
        record_issues = list(issues)
        if not log_filename:
            record_issues.append(RunIssue(
                code="missing_log_filename",
                message="parser returned a run without _log_filename",
                source=source,
            ))

        effective, config_issues = _logged_effective_config(
            raw_copy, source=source
        )
        record_issues.extend(config_issues)
        audit_config = {
            key: value for key, value in raw_copy.items() if key in _AUDIT_FIELDS
        }
        provenance = AuditProvenance(
            group=group,
            log_filename=log_filename,
            config=freeze_value(audit_config),
            manifest=(None if manifest is None else freeze_value(manifest)),
        )
        return cls(
            physical_id=physical_run_id(
                raw_copy,
                group=group,
                log_filename=log_filename,
                fallback_index=fallback_index,
            ),
            raw_config=freeze_value(raw_copy),
            history=tuple(freeze_value(dict(event)) for event in history),
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
