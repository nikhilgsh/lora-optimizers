"""Filesystem-led, lazy catalog of immutable :mod:`run_records`.

Discovery follows populated physical log directories.  ``meta.json`` is read as
an optional annotation and can contribute issues, but it never decides whether
a group or run exists.  A neutral single-file JSONL parser keeps physical
resume segments separate until explicit lineage validates them.  No loader
enrichment, current argparse defaults,
optimizer registry, deduplication, or exclusion policy is applied.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .run_parsing import parse_run_file
from .run_records import RunIssue, RunRecord, freeze_value


__all__ = ["RunCatalog", "load_records"]


_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True, slots=True)
class _GroupDescriptor:
    name: str
    manifest: Mapping[str, Any] | None
    issues: tuple[RunIssue, ...]


def _has_populated_logs(group_dir: Path) -> bool:
    log_dir = group_dir / "run_info" / "logs"
    try:
        entries = os.scandir(log_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False
    with entries:
        for entry in entries:
            if not _TASK_FILE_RE.match(entry.name):
                continue
            try:
                if entry.is_file() and entry.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _physical_log_files(logs_root: Path, group: str) -> tuple[Path, ...]:
    """Return physical task segments without inferring continuation lineage."""
    log_dir = logs_root / group / "run_info" / "logs"
    try:
        entries = [
            Path(entry.path)
            for entry in os.scandir(log_dir)
            if entry.is_file() and _TASK_FILE_RE.match(entry.name)
        ]
    except OSError:
        return ()

    def sort_key(path: Path) -> tuple[int, int]:
        match = _TASK_FILE_RE.match(path.name)
        task = int(match.group(1)) if match is not None else -1
        resume = (0 if ".resume_" not in path.name
                  else int(path.name.rsplit(".resume_", 1)[1]) + 1)
        return task, resume

    return tuple(sorted(entries, key=sort_key))


def _has_scope_annotation(scope: Any) -> bool:
    if isinstance(scope, str):
        return bool(scope.strip())
    if isinstance(scope, (list, tuple, set, frozenset)):
        return any(isinstance(item, str) and item.strip() for item in scope)
    return False


def _manifest_annotation(group_dir: Path) -> tuple[
    Mapping[str, Any] | None, tuple[RunIssue, ...]
]:
    group = group_dir.name
    path = group_dir / "run_info" / "meta.json"
    source = f"{group}/run_info/meta.json"
    try:
        raw = path.read_text()
    except OSError:
        return None, (RunIssue(
            code="manifest_missing",
            message="populated physical log group has no meta.json annotation",
            source=source,
        ),)

    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (RunIssue(
            code="manifest_corrupt",
            message=f"meta.json is not valid JSON: {exc.msg}",
            source=source,
        ),)
    if not isinstance(manifest, dict):
        return None, (RunIssue(
            code="manifest_not_object",
            message="meta.json must be an object to serve as an annotation",
            source=source,
        ),)

    issues: list[RunIssue] = []
    recorded_group = manifest.get("group")
    if recorded_group is not None and recorded_group != group:
        issues.append(RunIssue(
            code="manifest_group_mismatch",
            message=(f"manifest names group {recorded_group!r}; physical group "
                     f"is {group!r}"),
            source=source,
        ))
    if not _has_scope_annotation(manifest.get("scope")):
        issues.append(RunIssue(
            code="manifest_empty_scope",
            message="manifest has no non-blank scope annotation",
            source=source,
        ))
    return freeze_value(manifest), tuple(issues)


def _discover(logs_root: Path) -> tuple[_GroupDescriptor, ...]:
    try:
        children = sorted(
            (Path(entry.path) for entry in os.scandir(logs_root)
             if entry.is_dir()),
            key=lambda path: path.name,
        )
    except OSError:
        return ()
    groups: list[_GroupDescriptor] = []
    for group_dir in children:
        if not _has_populated_logs(group_dir):
            continue
        manifest, issues = _manifest_annotation(group_dir)
        groups.append(_GroupDescriptor(group_dir.name, manifest, issues))
    return tuple(groups)


def _validate_equals(equals: Mapping[str, Any] | None) -> dict[str, Any]:
    if equals is None:
        return {}
    if not isinstance(equals, Mapping):
        raise TypeError("equals must be a field-to-scalar mapping")
    out: dict[str, Any] = {}
    for field, value in equals.items():
        if not isinstance(field, str):
            raise TypeError("equals field names must be strings")
        if callable(value) or not isinstance(value, _SCALAR_TYPES):
            raise TypeError(
                f"equals[{field!r}] must be a scalar; use one_of for choices"
            )
        out[field] = value
    return out


def _validate_one_of(
    one_of: Mapping[str, Sequence[Any]] | None,
) -> dict[str, tuple[Any, ...]]:
    if one_of is None:
        return {}
    if not isinstance(one_of, Mapping):
        raise TypeError("one_of must be a field-to-values mapping")
    out: dict[str, tuple[Any, ...]] = {}
    for field, values in one_of.items():
        if not isinstance(field, str):
            raise TypeError("one_of field names must be strings")
        if (callable(values) or isinstance(values, (str, bytes, Mapping))
                or not isinstance(values, (list, tuple, set, frozenset))):
            raise TypeError(f"one_of[{field!r}] must be an explicit collection")
        candidates = tuple(values)
        if any(callable(value) or not isinstance(value, _SCALAR_TYPES)
               for value in candidates):
            raise TypeError(f"one_of[{field!r}] values must all be scalars")
        out[field] = candidates
    return out


class RunCatalog:
    """A snapshot of physical groups with lazy, per-group record parsing."""

    __slots__ = ("_logs_root", "_groups", "_loaded", "_runtime_issues")

    def __init__(self, logs_root: str | os.PathLike[str]):
        root = Path(logs_root).resolve()
        object.__setattr__(self, "_logs_root", root)
        object.__setattr__(self, "_groups", _discover(root))
        object.__setattr__(self, "_loaded", {})
        object.__setattr__(self, "_runtime_issues", {})

    def __setattr__(self, name, value) -> None:
        raise AttributeError(
            "RunCatalog is an immutable discovery snapshot; construct a new "
            "catalog to rescan"
        )

    @classmethod
    def discover(cls, logs_root: str | os.PathLike[str]) -> "RunCatalog":
        return cls(logs_root)

    @property
    def logs_root(self) -> Path:
        return self._logs_root

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(group.name for group in self._groups)

    @property
    def manifests(self) -> Mapping[str, Mapping[str, Any] | None]:
        return MappingProxyType({group.name: group.manifest
                                 for group in self._groups})

    @property
    def group_issues(self) -> Mapping[str, tuple[RunIssue, ...]]:
        return MappingProxyType({
            group.name: group.issues + self._runtime_issues.get(group.name, ())
            for group in self._groups
        })

    @property
    def issues(self) -> tuple[RunIssue, ...]:
        return tuple(issue for group in self._groups
                     for issue in (group.issues
                                   + self._runtime_issues.get(group.name, ())))

    def _descriptor(self, group_name: str) -> _GroupDescriptor:
        for group in self._groups:
            if group.name == group_name:
                return group
        raise KeyError(f"group {group_name!r} is not in this catalog snapshot")

    def _load_group(self, group_name: str) -> tuple[RunRecord, ...]:
        cached = self._loaded.get(group_name)
        if cached is not None:
            return cached
        descriptor = self._descriptor(group_name)
        records_list: list[RunRecord] = []
        runtime_issues: list[RunIssue] = []
        for index, path in enumerate(_physical_log_files(
            self._logs_root, group_name
        )):
            try:
                parsed = parse_run_file(path)
                cfg = parsed.raw_config()
                history = parsed.mutable_evals()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                runtime_issues.append(RunIssue(
                    code="parser_error",
                    message=f"neutral log parser failed: {exc}",
                    source=f"{group_name}/{path.name}",
                ))
                continue
            if cfg is None or not history:
                runtime_issues.append(RunIssue(
                    code="no_parsed_run",
                    message="physical log has no usable config/eval pair",
                    source=f"{group_name}/{path.name}",
                ))
                continue
            records_list.append(RunRecord.from_parsed(
                cfg,
                history,
                group=group_name,
                manifest=descriptor.manifest,
                issues=descriptor.issues,
                fallback_index=index,
            ))
        records = tuple(records_list)
        if runtime_issues:
            self._runtime_issues[group_name] = tuple(runtime_issues)
        self._loaded[group_name] = records
        return records

    @property
    def records(self) -> tuple[RunRecord, ...]:
        return tuple(record for group in self.groups
                     for record in self._load_group(group))

    def records_for_group(self, group: str) -> tuple[RunRecord, ...]:
        """Load one discovered group without parsing the rest of the tree."""
        return self._load_group(group)

    def query(
        self,
        *,
        equals: Mapping[str, Any] | None = None,
        one_of: Mapping[str, Sequence[Any]] | None = None,
    ) -> tuple[RunRecord, ...]:
        """Filter effective logged config by scalar equality and membership.

        The query is deliberately uncached and accepts no callables.  Missing
        fields do not match.  ``group``, ``log_filename``, and ``physical_id``
        are also available as explicit physical fields.
        """
        eq = _validate_equals(equals)
        choices = _validate_one_of(one_of)
        overlap = sorted(set(eq).intersection(choices))
        if overlap:
            raise ValueError(f"fields cannot appear in equals and one_of: {overlap}")

        def value(record: RunRecord, field: str) -> tuple[bool, Any]:
            if field == "group":
                return True, record.group
            if field == "log_filename":
                return record.log_filename is not None, record.log_filename
            if field == "physical_id":
                return True, record.physical_id
            if field not in record.effective_config:
                return False, None
            return True, record.effective_config[field]

        candidate_groups = self.groups
        if "group" in eq:
            wanted = eq["group"]
            candidate_groups = tuple(
                group for group in candidate_groups if group == wanted
            )
        elif "group" in choices:
            wanted = choices["group"]
            candidate_groups = tuple(
                group for group in candidate_groups if group in wanted
            )
        candidates = (
            record
            for group in candidate_groups
            for record in self._load_group(group)
        )

        selected: list[RunRecord] = []
        for record in candidates:
            if any(not value(record, field)[0]
                   or value(record, field)[1] != expected
                   for field, expected in eq.items()):
                continue
            if any(not value(record, field)[0]
                   or not any(value(record, field)[1] == candidate
                              for candidate in candidates)
                   for field, candidates in choices.items()):
                continue
            selected.append(record)
        return tuple(selected)

    def as_legacy_tuples(
        self,
        records: Iterable[RunRecord] | None = None,
    ) -> list[tuple[dict, list[dict]]]:
        chosen = self.records if records is None else tuple(records)
        return [record.as_legacy_tuple() for record in chosen]

    def resolve_lineages(
        self,
        records: Iterable[RunRecord] | None = None,
    ) -> tuple[Any, ...]:
        """Validate and merge only versioned, explicitly linked attempts.

        Historical records remain independent physical records; this method
        never sends them through filename/step-range resume inference.
        Versioned records are grouped by their declared checkpoint identity,
        not directory or task filename.  When ``records`` is a query subset,
        selecting any segment returns its complete explicitly connected chain.
        """
        from .run_lineage import build_run_lineages

        chosen = self.records if records is None else tuple(records)
        legacy = [record for record in chosen
                  if "run_schema_version" not in record.raw_config]
        selected_attempt_ids = {
            record.raw_config.get("attempt_id")
            for record in chosen
            if "run_schema_version" in record.raw_config
        }
        if not selected_attempt_ids:
            return tuple(legacy)

        buckets: dict[str, list[RunRecord]] = {}
        # Build against the catalog domain, not just the predicate subset, so a
        # child selected by a physical field can still resolve its declared
        # parent and a root selected before a group rename keeps its child.
        seen_attempt_ids: dict[str, str] = {}
        all_records = self.records
        for record in all_records:
            if "run_schema_version" not in record.raw_config:
                continue
            attempt_id = record.raw_config.get("attempt_id")
            checkpoint_identity = record.raw_config.get("checkpoint_identity")
            if isinstance(attempt_id, str) and attempt_id:
                previous = seen_attempt_ids.get(attempt_id)
                if previous is not None and previous != checkpoint_identity:
                    from .run_lineage import LineageStructureError, LineageIssue
                    raise LineageStructureError(LineageIssue(
                        code="duplicate_attempt_id_across_checkpoints",
                        attempt_id=attempt_id,
                        parent_attempt_id=None,
                        details=freeze_value({
                            "checkpoint_identities": (
                                previous, checkpoint_identity,
                            ),
                        }),
                    ))
                seen_attempt_ids[attempt_id] = checkpoint_identity
            bucket = (
                checkpoint_identity
                if isinstance(checkpoint_identity, str)
                else f"<invalid:{record.physical_id}>"
            )
            buckets.setdefault(bucket, []).append(record)

        resolved: list[Any] = list(legacy)
        for key in sorted(buckets):
            for lineage in build_run_lineages(buckets[key]):
                if selected_attempt_ids.intersection(lineage.attempt_ids):
                    resolved.append(lineage)
        return tuple(resolved)

    def resolved_legacy_tuples(
        self,
        records: Iterable[RunRecord] | None = None,
    ) -> list[tuple[dict, list[dict]]]:
        """Plot-compatible copies after explicit-only lineage resolution."""
        from .run_lineage import MergedRunLineage
        from .run_records import thaw_value

        out = []
        for run in self.resolve_lineages(records):
            if isinstance(run, RunRecord):
                out.append(run.as_legacy_tuple())
                continue
            if not isinstance(run, MergedRunLineage):
                raise TypeError(f"unexpected resolved run type: {type(run)!r}")
            cfg = thaw_value(run.cfg)
            cfg.setdefault("run_id", run.terminal_attempt_id)
            out.append((cfg, thaw_value(run.history)))
        return out


def _default_logs_root() -> str:
    """Repo-anchored ``logs/`` path, independent of caller cwd."""
    return str(Path(__file__).resolve().parent.parent / "logs")


def load_records(
    *,
    equals: dict[str, Any] | None = None,
    one_of: dict[str, Iterable[Any]] | None = None,
    logs_root: str | None = None,
    catalog=None,
    resolve_lineages: bool = True,
):
    """Load immutable run records without importing the legacy loader.

    Predicates are intentionally limited to scalar equality and explicit
    membership. The returned objects are immutable ``RunRecord`` instances or,
    when enabled, validated ``MergedRunLineage`` objects. Historical logs stay
    as independent records; only versioned attempts with an actual recorded
    resume edge are merged.

    ``lora_playground.loader.load_records`` is an exact compatibility re-export
    of this function while tuple-based consumers migrate to records.
    """
    if catalog is not None and logs_root is not None:
        raise ValueError("pass either catalog or logs_root, not both")
    if catalog is None:
        catalog = RunCatalog.discover(logs_root or _default_logs_root())
    elif not isinstance(catalog, RunCatalog):
        raise TypeError("catalog must be a RunCatalog")
    records = catalog.query(equals=equals, one_of=one_of)
    return catalog.resolve_lineages(records) if resolve_lineages else records
