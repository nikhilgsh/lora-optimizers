"""Filesystem-led, lazy catalog of immutable :mod:`run_records`.

Discovery follows populated physical log directories.  ``meta.json`` is read as
an optional annotation and can contribute issues, but it never decides whether
a group or run exists.  Parsing temporarily delegates to the existing
single-file JSONL parser; physical resume segments stay separate until explicit
lineage validates them.  No loader enrichment, current argparse defaults,
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

from .run_records import RunIssue, RunRecord, freeze_value


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
        # Temporary dependency: use only the existing per-file parser. Calling
        # load_sweep here would union resume siblings by step before explicit
        # lineage can prove that they belong together.
        from .plotting import loading as legacy_loading

        records_list: list[RunRecord] = []
        runtime_issues: list[RunIssue] = []
        for index, path in enumerate(_physical_log_files(
            self._logs_root, group_name
        )):
            try:
                cfg, history = legacy_loading.load_run(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                runtime_issues.append(RunIssue(
                    code="parser_error",
                    message=f"existing log parser failed: {exc}",
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
