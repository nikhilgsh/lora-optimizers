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
from typing import Any, Callable, Iterable, Mapping, Sequence

from .run_parsing import parse_run_file, parse_run_header
from .run_records import (
    RunIssue,
    RunRecord,
    freeze_value,
    logged_effective_config,
    logged_effective_value,
    physical_run_id,
)


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


def _validate_predicates(
    predicates: Mapping[str, Callable[[Any], Any]] | None,
) -> dict[str, Callable[[Any], Any]]:
    if predicates is None:
        return {}
    if not isinstance(predicates, Mapping):
        raise TypeError("predicates must be a field-to-callable mapping")
    out: dict[str, Callable[[Any], Any]] = {}
    for field, predicate in predicates.items():
        if not isinstance(field, str):
            raise TypeError("predicate field names must be strings")
        if not callable(predicate):
            raise TypeError(f"predicates[{field!r}] must be callable")
        out[field] = predicate
    return out


class RunCatalog:
    """A snapshot of physical groups with lazy, per-group record parsing."""

    __slots__ = (
        "_logs_root",
        "_groups",
        "_loaded",
        "_runtime_issues",
        "_files",
        "_headers",
        "_semantic_headers",
        "_semantic_values",
        "_record_cache",
    )

    def __init__(self, logs_root: str | os.PathLike[str]):
        root = Path(logs_root).resolve()
        object.__setattr__(self, "_logs_root", root)
        object.__setattr__(self, "_groups", _discover(root))
        object.__setattr__(self, "_loaded", {})
        object.__setattr__(self, "_runtime_issues", {})
        object.__setattr__(self, "_files", {})
        object.__setattr__(self, "_headers", {})
        object.__setattr__(self, "_semantic_headers", {})
        object.__setattr__(self, "_semantic_values", {})
        object.__setattr__(self, "_record_cache", {})

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

    def _paths_for_group(self, group_name: str) -> tuple[Path, ...]:
        cached = self._files.get(group_name)
        if cached is None:
            self._descriptor(group_name)
            cached = _physical_log_files(self._logs_root, group_name)
            self._files[group_name] = cached
        return cached

    def _header_config(self, group_name: str, path: Path) -> Mapping[str, Any] | None:
        key = str(path)
        if key in self._headers:
            return self._headers[key]
        try:
            config = parse_run_header(path).frozen_raw_config()
        except (OSError, ValueError, TypeError, KeyError):
            config = None
        self._headers[key] = config
        return config

    def _header_semantic_config(
        self,
        group_name: str,
        path: Path,
    ) -> Mapping[str, Any] | None:
        """Return the flattened startup semantics once per catalog snapshot."""
        key = str(path)
        if key in self._semantic_headers:
            return self._semantic_headers[key]
        raw = self._header_config(group_name, path)
        semantic = None
        if raw is not None:
            semantic, _issues = logged_effective_config(
                raw, source=f"{group_name}/{path.name}"
            )
        self._semantic_headers[key] = semantic
        return semantic

    def _record_issue(self, group_name: str, issue: RunIssue) -> None:
        current = self._runtime_issues.get(group_name, ())
        if issue not in current:
            self._runtime_issues[group_name] = current + (issue,)

    def _load_path(
        self,
        group_name: str,
        path: Path,
        fallback_index: int,
    ) -> RunRecord | None:
        key = str(path)
        if key in self._record_cache:
            return self._record_cache[key]
        descriptor = self._descriptor(group_name)
        try:
            parsed = parse_run_file(path, include_optim_steps=False)
            config = parsed.frozen_raw_config()
            history = parsed.evals
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._record_issue(group_name, RunIssue(
                code="parser_error",
                message=f"neutral log parser failed: {exc}",
                source=f"{group_name}/{path.name}",
            ))
            self._record_cache[key] = None
            return None
        if config is None or not history:
            self._record_issue(group_name, RunIssue(
                code="no_parsed_run",
                message="physical log has no usable config/eval pair",
                source=f"{group_name}/{path.name}",
            ))
            self._record_cache[key] = None
            return None
        record = RunRecord.from_parsed(
            config,
            history,
            group=group_name,
            manifest=descriptor.manifest,
            issues=descriptor.issues,
            fallback_index=fallback_index,
        )
        self._record_cache[key] = record
        return record

    def _load_group(self, group_name: str) -> tuple[RunRecord, ...]:
        cached = self._loaded.get(group_name)
        if cached is not None:
            return cached
        records_list: list[RunRecord] = []
        for index, path in enumerate(self._paths_for_group(group_name)):
            record = self._load_path(group_name, path, index)
            if record is not None:
                records_list.append(record)
        records = tuple(records_list)
        self._loaded[group_name] = records
        return records

    @property
    def records(self) -> tuple[RunRecord, ...]:
        return tuple(record for group in self.groups
                     for record in self._load_group(group))

    def records_for_group(self, group: str) -> tuple[RunRecord, ...]:
        """Load one discovered group without parsing the rest of the tree."""
        return self._load_group(group)

    @property
    def logged_field_names(self) -> frozenset[str]:
        """Fields visible to compatibility queries, from config headers only."""
        fields = {"log_group", "_log_filename", "run_id"}
        for group in self.groups:
            for path in self._paths_for_group(group):
                config = self._header_config(group, path)
                if config is None:
                    continue
                fields.update(config)
                semantic = self._header_semantic_config(group, path)
                if semantic is not None:
                    fields.update(semantic)
        return frozenset(fields)

    @staticmethod
    def _record_value(record: RunRecord, field: str) -> tuple[bool, Any]:
        if field == "group":
            return True, record.group
        if field == "log_filename":
            return record.log_filename is not None, record.log_filename
        if field == "physical_id":
            return True, record.physical_id
        if field not in record.effective_config:
            return False, None
        return True, record.effective_config[field]

    def _candidate_records(
        self,
        equals: Mapping[str, Any],
        one_of: Mapping[str, tuple[Any, ...]],
        *,
        conservative_collections: bool,
        predicates: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> tuple[RunRecord, ...]:
        """Header-screen candidates, then fully parse only survivors."""
        predicates = {} if predicates is None else predicates
        candidate_groups = self.groups
        if "group" in equals:
            wanted = equals["group"]
            candidate_groups = tuple(
                group for group in candidate_groups if group == wanted
            )
        elif "group" in one_of:
            wanted = one_of["group"]
            candidate_groups = tuple(
                group for group in candidate_groups if group in wanted
            )

        selected: list[RunRecord] = []
        for group in candidate_groups:
            for index, path in enumerate(self._paths_for_group(group)):
                raw = self._header_config(group, path)
                # A missing/corrupt header is rare and cannot safely reject a
                # file. Let the full parser diagnose it.
                if raw is None:
                    record = self._load_path(group, path, index)
                    if record is not None:
                        selected.append(record)
                    continue

                def header_value(field: str) -> tuple[bool, Any]:
                    if field == "group":
                        return True, group
                    if field == "log_filename":
                        return True, path.name
                    if field == "physical_id":
                        return True, physical_run_id(
                            raw,
                            group=group,
                            log_filename=path.name,
                            fallback_index=index,
                        )
                    key = (str(path), field)
                    cached = self._semantic_values.get(key)
                    if cached is None:
                        cached = logged_effective_value(raw, field)
                        self._semantic_values[key] = cached
                    return cached

                rejected = False
                for field, expected in equals.items():
                    present, value = header_value(field)
                    if not present or value != expected:
                        rejected = True
                        break
                if rejected:
                    continue
                for field, allowed in one_of.items():
                    present, value = header_value(field)
                    if not present:
                        rejected = True
                        break
                    if conservative_collections and not isinstance(
                        value, _SCALAR_TYPES
                    ):
                        # Legacy `_matches` treats a collection-valued cfg as
                        # literal equality against the predicate collection.
                        # Abstain here and let its exact residual decide.
                        continue
                    if value not in allowed:
                        rejected = True
                        break
                if rejected:
                    continue
                for field, predicate in predicates.items():
                    present, value = header_value(field)
                    if not present or not bool(predicate(value)):
                        rejected = True
                        break
                if rejected:
                    continue
                record = self._load_path(group, path, index)
                if record is not None:
                    selected.append(record)
        return tuple(selected)

    def prefilter(
        self,
        *,
        equals: Mapping[str, Any] | None = None,
        one_of: Mapping[str, Sequence[Any]] | None = None,
        predicates: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> tuple[RunRecord, ...]:
        """Conservatively screen scalar predicates from config headers.

        Returned records are candidates, not a final query result. Collection-
        valued effective fields are retained so compatibility callers can
        apply their historical literal-list matcher after full parsing.
        ``predicates`` is reserved for compatibility callables explicitly
        marked safe for logged-header evaluation; :meth:`query` deliberately
        remains limited to equality and membership.
        """
        eq = _validate_equals(equals)
        choices = _validate_one_of(one_of)
        header_predicates = _validate_predicates(predicates)
        overlap = sorted(
            (set(eq) & set(choices))
            | (set(eq) & set(header_predicates))
            | (set(choices) & set(header_predicates))
        )
        if overlap:
            raise ValueError(
                "fields cannot appear in multiple prefilter maps: "
                f"{overlap}"
            )
        return self._candidate_records(
            eq,
            choices,
            conservative_collections=True,
            predicates=header_predicates,
        )

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

        if not eq and not choices:
            return self.records
        candidates = self._candidate_records(
            eq, choices, conservative_collections=False
        )

        selected: list[RunRecord] = []
        for record in candidates:
            if any(not self._record_value(record, field)[0]
                   or self._record_value(record, field)[1] != expected
                   for field, expected in eq.items()):
                continue
            if any(not self._record_value(record, field)[0]
                   or not any(self._record_value(record, field)[1] == candidate
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

        resolving_whole_catalog = records is None
        chosen = self.records if resolving_whole_catalog else tuple(records)
        legacy = [record for record in chosen
                  if "run_schema_version" not in record.raw_config]
        selected_attempt_ids = {
            record.raw_config.get("attempt_id")
            for record in chosen
            if "run_schema_version" in record.raw_config
        }
        if not selected_attempt_ids:
            return tuple(legacy)

        def checkpoint_identity(config: Mapping[str, Any]) -> Any:
            identity = config.get("checkpoint_identity")
            if identity is not None:
                return identity
            resume = config.get("_resume")
            return (
                resume.get("checkpoint_identity")
                if isinstance(resume, Mapping)
                else None
            )

        selected_checkpoints = {
            checkpoint_identity(record.raw_config)
            for record in chosen
            if "run_schema_version" in record.raw_config
        }
        selected_metadata_valid = (
            all(isinstance(attempt_id, str) and attempt_id
                for attempt_id in selected_attempt_ids)
            and all(
            isinstance(identity, str) and identity
            for identity in selected_checkpoints
            )
        )
        if resolving_whole_catalog:
            all_records = chosen
        elif selected_metadata_valid:
            # Build an undirected graph of explicit resume edges from startup
            # headers. This identifies the complete connected chain around the
            # selected attempts without parsing unrelated histories, including
            # independent roots that happen to reuse a checkpoint identity.
            header_paths: dict[
                tuple[str, str], list[tuple[str, Path, int]]
            ] = {}
            neighbors: dict[tuple[str, str], set[tuple[str, str]]] = {}
            malformed_versioned_header = False
            for group in self.groups:
                for index, path in enumerate(self._paths_for_group(group)):
                    header = self._header_config(group, path)
                    if header is None or "run_schema_version" not in header:
                        continue
                    checkpoint = checkpoint_identity(header)
                    attempt_id = header.get("attempt_id")
                    resume = header.get("_resume")
                    parent_id = (
                        resume.get("resume_parent_attempt_id")
                        if isinstance(resume, Mapping)
                        else None
                    )
                    if (
                        not isinstance(checkpoint, str)
                        or not checkpoint
                        or not isinstance(attempt_id, str)
                        or not attempt_id
                        or (
                            parent_id is not None
                            and (
                                not isinstance(parent_id, str)
                                or not parent_id
                            )
                        )
                    ):
                        malformed_versioned_header = True
                        break
                    if checkpoint not in selected_checkpoints:
                        continue
                    key = (checkpoint, attempt_id)
                    header_paths.setdefault(key, []).append(
                        (group, path, index)
                    )
                    neighbors.setdefault(key, set())
                    if parent_id is not None:
                        parent_key = (checkpoint, parent_id)
                        neighbors[key].add(parent_key)
                        neighbors.setdefault(parent_key, set()).add(key)
                if malformed_versioned_header:
                    break

            if malformed_versioned_header:
                # Without complete startup identity metadata, header closure
                # cannot prove which files belong to the selected chain.
                all_records = self.records
            else:
                connected = {
                    (checkpoint, attempt_id)
                    for checkpoint in selected_checkpoints
                    for attempt_id in selected_attempt_ids
                }
                frontier = list(connected)
                while frontier:
                    key = frontier.pop()
                    for neighbor in neighbors.get(key, ()):
                        if neighbor in connected:
                            continue
                        connected.add(neighbor)
                        frontier.append(neighbor)

                lineage_domain: list[RunRecord] = []
                for key in sorted(connected):
                    for group, path, index in header_paths.get(key, ()):
                        record = self._load_path(group, path, index)
                        if record is not None:
                            lineage_domain.append(record)
                all_records = tuple(lineage_domain)
        else:
            # Old or malformed versioned records may not expose enough startup
            # metadata to prove closure. Fail safe by materializing the catalog
            # rather than returning a partial lineage.
            all_records = self.records

        buckets: dict[str, list[RunRecord]] = {}
        # Build against the selected checkpoint domain, not only the predicate
        # subset, so selecting any segment still returns its complete chain.
        seen_attempt_ids: dict[str, str] = {}
        for record in all_records:
            if "run_schema_version" not in record.raw_config:
                continue
            attempt_id = record.raw_config.get("attempt_id")
            checkpoint = checkpoint_identity(record.raw_config)
            if isinstance(attempt_id, str) and attempt_id:
                previous = seen_attempt_ids.get(attempt_id)
                if previous is not None and previous != checkpoint:
                    from .run_lineage import LineageStructureError, LineageIssue
                    raise LineageStructureError(LineageIssue(
                        code="duplicate_attempt_id_across_checkpoints",
                        attempt_id=attempt_id,
                        parent_attempt_id=None,
                        details=freeze_value({
                            "checkpoint_identities": (
                                previous, checkpoint,
                            ),
                        }),
                    ))
                seen_attempt_ids[attempt_id] = checkpoint
            bucket = (
                checkpoint
                if isinstance(checkpoint, str)
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
