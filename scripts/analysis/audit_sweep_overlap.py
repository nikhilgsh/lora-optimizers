"""Audit a sweep plan against producer-recorded runs.

Usage:
    python scripts/analysis/audit_sweep_overlap.py params/<sweep>.json

The params JSON remains a mapping from argument name to a list of values. Its
cartesian product is compared with immutable records discovered by
``RunCatalog``. Only fields recorded in a run's semantic configuration may
match. This checker does not parse shell launchers or command strings, import
current argparse/optimizer defaults, translate aliases, or maintain a second
table of optimizer equivalences.

``--sweep-script`` is retained because ``slurm_scripts/submit.sh`` passes it.
The path scopes candidates to manifests that recorded the same launcher, but
the launcher contents are never interpreted. A matching record whose launcher
provenance or requested field is missing is UNKNOWN, not silently compatible.

Exit status is non-zero for both proven overlap and uncertainty. The submit
wrapper's existing ``FORCE_OVERLAP=1`` escape hatch therefore remains the
explicit route for a deliberately repeated run or an old, under-recorded run.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lora_playground.publication_semantics import (
    PublicationSemanticsError,
    publication_semantics_from_payload,
)
from lora_playground.run_catalog import RunCatalog
from lora_playground.run_records import RunRecord


PRODUCER_VARIANT_FIELD = "optimizer_variant_semantics"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class PhysicalMatch:
    """One physical record that supports (or weakens) an audit decision."""

    physical_id: str
    group: str
    log_filename: str | None
    variant_view_key: str | None = None
    variant_exact_id: str | None = None
    missing_fields: tuple[str, ...] = ()
    issue: str | None = None


@dataclass(frozen=True, slots=True)
class CellAudit:
    """Result for one cartesian sweep cell."""

    status: str
    cell: Mapping[str, Any]
    evidence: tuple[PhysicalMatch, ...] = ()
    reason: str | None = None


def load_params(path: Path) -> dict[str, list[Any]]:
    """Read and validate the existing list-valued sweep params format."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError("params file must contain a non-empty JSON object")
    params: dict[str, list[Any]] = {}
    for key, choices in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("params keys must be non-empty strings")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"params field {key!r} must be a non-empty list")
        params[key] = choices
    return params


def cartesian(params: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    keys = tuple(params)
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*(params[key] for key in keys))
    ]


def _canonical_param_value(value: Any) -> Any:
    """Interpret JSON-scalar strings the same way argparse records them.

    Sweep files historically quote numeric values because their task generator
    forwards positional strings. Numeric parsing is representation-only: it
    neither supplies a missing value nor changes a field name or algorithm.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if not isinstance(parsed, (dict, list)) else value


def canonical_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _canonical_param_value(value) for key, value in cell.items()}


def _repo_relative_launcher(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _launcher_relation(
    record: RunRecord,
    expected_launcher: str | None,
) -> tuple[bool, str | None]:
    """Return whether launcher provenance agrees, or why it is unknown."""
    if expected_launcher is None:
        return True, None
    manifest = record.audit_provenance.manifest
    if not isinstance(manifest, Mapping):
        return False, "record has no manifest launcher provenance"
    recorded = manifest.get("sweep_script")
    if not isinstance(recorded, str) or not recorded.strip():
        return False, "record manifest has no sweep_script"
    return recorded == expected_launcher, None


def _physical_match(
    record: RunRecord,
    *,
    variant_view_key: str | None = None,
    variant_exact_id: str | None = None,
    missing_fields: Iterable[str] = (),
    issue: str | None = None,
) -> PhysicalMatch:
    return PhysicalMatch(
        physical_id=record.physical_id,
        group=record.group,
        log_filename=record.log_filename,
        variant_view_key=variant_view_key,
        variant_exact_id=variant_exact_id,
        missing_fields=tuple(sorted(missing_fields)),
        issue=issue,
    )


def _recorded_variant(record: RunRecord) -> PhysicalMatch:
    payload = record.raw_config.get(PRODUCER_VARIANT_FIELD)
    if payload is None:
        return _physical_match(
            record,
            issue=f"record has no producer {PRODUCER_VARIANT_FIELD!r} block",
        )
    if not isinstance(payload, Mapping):
        return _physical_match(
            record,
            issue=f"recorded {PRODUCER_VARIANT_FIELD!r} is not an object",
        )
    try:
        semantics = publication_semantics_from_payload(payload)
    except PublicationSemanticsError as exc:
        return _physical_match(record, issue=str(exc))
    return _physical_match(
        record,
        variant_view_key=semantics.view_key,
        variant_exact_id=semantics.exact_id,
    )


def _record_relation(
    record: RunRecord,
    cell: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Classify a record as exact, conflicting, or missing-only."""
    missing: list[str] = []
    for field, expected in cell.items():
        if field not in record.semantic_config:
            missing.append(field)
            continue
        if record.semantic_config[field] != expected:
            return "conflict", ()
    return ("missing", tuple(missing)) if missing else ("exact", ())


def audit_cell(
    cell: Mapping[str, Any],
    catalog: RunCatalog,
    *,
    expected_launcher: str | None = None,
) -> CellAudit:
    """Audit one cell without reconstructing absent execution semantics."""
    normalized = canonical_cell(cell)
    optimizer = normalized.get("optimizer")
    if not isinstance(optimizer, str) or not optimizer:
        return CellAudit(
            "UNKNOWN",
            normalized,
            reason=(
                "candidate does not record optimizer; launcher text is not an "
                "identity source"
            ),
        )

    # Optimizer is the mandatory semantic anchor. Inspecting those records lets
    # a missing newly-added field remain UNKNOWN instead of being filtered out
    # by an exact catalog query and mislabeled NEW.
    candidates = catalog.query(equals={"optimizer": optimizer})
    exact: list[PhysicalMatch] = []
    uncertain: list[PhysicalMatch] = []
    for record in candidates:
        relation, missing = _record_relation(record, normalized)
        if relation == "conflict":
            continue
        launcher_matches, launcher_issue = _launcher_relation(
            record, expected_launcher
        )
        if launcher_issue is not None:
            uncertain.append(_physical_match(record, issue=launcher_issue))
            continue
        if not launcher_matches:
            continue
        if relation == "missing":
            uncertain.append(_physical_match(record, missing_fields=missing))
            continue
        identity = _recorded_variant(record)
        if identity.issue is not None:
            uncertain.append(identity)
        else:
            exact.append(identity)

    if exact:
        view_keys = {item.variant_view_key for item in exact}
        if len(view_keys) != 1:
            return CellAudit(
                "UNKNOWN",
                normalized,
                tuple(exact),
                reason=(
                    "the explicit candidate fields select multiple producer "
                    f"variant identities: {sorted(view_keys)!r}"
                ),
            )
        return CellAudit("EXISTS", normalized, tuple(exact))
    if uncertain:
        return CellAudit(
            "UNKNOWN",
            normalized,
            tuple(uncertain),
            reason=(
                "potential matching physical record(s) lack fields or provenance "
                "needed to prove equivalence"
            ),
        )
    return CellAudit("NEW", normalized)


def audit_sweep(
    params: Mapping[str, list[Any]],
    *,
    catalog: RunCatalog,
    expected_launcher: str | None = None,
) -> tuple[CellAudit, ...]:
    return tuple(
        audit_cell(cell, catalog, expected_launcher=expected_launcher)
        for cell in cartesian(params)
    )


def _format_provenance(match: PhysicalMatch) -> str:
    parts = [
        f"physical_id={match.physical_id!r}",
        f"group={match.group!r}",
        f"log={match.log_filename!r}",
    ]
    if match.variant_view_key is not None:
        parts.append(f"variant_view={match.variant_view_key!r}")
    if match.variant_exact_id is not None:
        parts.append(f"variant_exact={match.variant_exact_id!r}")
    if match.missing_fields:
        parts.append(f"missing_fields={list(match.missing_fields)!r}")
    if match.issue is not None:
        parts.append(f"issue={match.issue!r}")
    return " ".join(parts)


def _print_results(results: tuple[CellAudit, ...]) -> None:
    symbols = {"EXISTS": "✓", "NEW": "☐", "UNKNOWN": "?"}
    for result in results:
        print(f"  {symbols[result.status]} {result.status:<7s} {dict(result.cell)}")
        if result.reason:
            print(f"      {result.reason}")
        for match in result.evidence[:5]:
            print(f"      {_format_provenance(match)}")
        if len(result.evidence) > 5:
            print(f"      ... {len(result.evidence) - 5} more physical record(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params_file")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument(
        "--sweep-script",
        default=None,
        help=(
            "Launcher path used only to scope against the manifest's recorded "
            "sweep_script. Its shell contents are never parsed."
        ),
    )
    args = parser.parse_args(argv)

    params_path = Path(args.params_file)
    params = load_params(params_path)
    launcher = None
    if args.sweep_script is not None:
        launcher_path = Path(args.sweep_script)
        if not launcher_path.is_file():
            parser.error(f"--sweep-script does not exist: {launcher_path}")
        launcher = _repo_relative_launcher(launcher_path)

    catalog = RunCatalog.discover(args.logs_root)
    results = audit_sweep(
        params,
        catalog=catalog,
        expected_launcher=launcher,
    )

    print(f"Sweep params: {params_path}")
    print(f"Cartesian product size: {len(results)} cells")
    if launcher is not None:
        print(f"Launcher provenance scope: {launcher} (contents not parsed)")
    print()
    _print_results(results)
    print()

    overlap = [result for result in results if result.status == "EXISTS"]
    unknown = [result for result in results if result.status == "UNKNOWN"]
    if overlap:
        print(f"OVERLAP: {len(overlap)}/{len(results)} cells already exist.")
    if unknown:
        print(
            f"UNKNOWN: {len(unknown)}/{len(results)} cells cannot be proven new "
            "from recorded fields."
        )
    if overlap or unknown:
        print(
            "Refusing the sweep. Remove proven duplicates; for intentionally "
            "repeated or under-recorded work, use FORCE_OVERLAP=1 and record "
            "the reason in SWEEP_PURPOSE."
        )
        return 1

    print(f"No overlap. All {len(results)} cells are new.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
