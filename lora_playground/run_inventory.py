"""Policy-free inventory over immutable run-catalog records.

This module reports physical discovery, parse completeness, and recorded
coverage.  It deliberately knows nothing about legacy exclusions, argparse
defaults, inferred resume stitching, Git ancestry, or persistent caches.
Consumer registries and thresholds are explicit inputs so importing this leaf
does not pull plotting or training policy into the catalog boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

from .run_catalog import RunCatalog


PINNING_INTERIOR = "interior"
PINNING_LOW = "pinned_low"
PINNING_HIGH = "pinned_high"
PINNING_SINGLE = "single_lr"
PINNING_ALL_DIVERGED = "all_diverged"


@dataclass(frozen=True)
class CoverageRow:
    """One recorded (optimizer, rank, multiplier) coverage cell."""

    optimizer: str
    lora_r: int | None
    lora_plus_multiplier: float | None
    lrs_swept: tuple[float, ...]
    best_lr: float | None
    final_loss_at_best: float | None
    pinning: str
    source_groups: tuple[str, ...]


@dataclass(frozen=True)
class RunInventory:
    """Facts observed in one immutable :class:`RunCatalog` snapshot."""

    groups_on_disk: tuple[str, ...]
    groups_loaded: tuple[str, ...]
    groups_orphaned: tuple[str, ...]
    groups_no_run_info: tuple[str, ...]
    groups_without_records: tuple[str, ...]
    records_incomplete: tuple[tuple[str, tuple[str, ...]], ...]
    optimizers_unknown: tuple[str, ...]
    coverage: tuple[CoverageRow, ...]

    @property
    def pinned(self) -> tuple[CoverageRow, ...]:
        return tuple(
            row for row in self.coverage
            if row.pinning in (PINNING_LOW, PINNING_HIGH)
        )


def _classify_pinning(
    lrs_swept: tuple[float, ...], best_lr: float | None,
) -> str:
    if best_lr is None:
        return PINNING_ALL_DIVERGED
    if len(lrs_swept) <= 1:
        return PINNING_SINGLE
    if best_lr == min(lrs_swept):
        return PINNING_LOW
    if best_lr == max(lrs_swept):
        return PINNING_HIGH
    return PINNING_INTERIOR


def _logged_float(config: Any, field: str) -> float | None:
    value = config.get(field)
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else None
    except (TypeError, ValueError):
        return None


def _logged_int(config: Any, field: str) -> int | None:
    value = config.get(field)
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
        if float(value) != converted:
            return None
        return converted
    except (TypeError, ValueError, OverflowError):
        return None


def audit_run_catalog(
    logs_root: str | Path,
    *,
    known_optimizers: Collection[str],
    diverge_threshold: float,
) -> RunInventory:
    """Audit physical records using only their logged effective config."""
    catalog = RunCatalog.discover(logs_root)
    records = catalog.records
    on_disk = catalog.groups
    contributing_groups = {record.group for record in records}

    manifest_problem_codes = {
        "manifest_missing",
        "manifest_corrupt",
        "manifest_not_object",
        "manifest_group_mismatch",
        "manifest_empty_scope",
    }
    orphaned = tuple(sorted(
        group for group, issues in catalog.group_issues.items()
        if any(issue.code in manifest_problem_codes for issue in issues)
    ))

    rows: dict[tuple[str, int | None, float | None], dict] = {}
    seen_optimizers: set[str] = set()
    incomplete: list[tuple[str, tuple[str, ...]]] = []

    for record in records:
        config = record.semantic_config
        missing: list[str] = []
        optimizer_value = config.get("optimizer")
        optimizer = (
            optimizer_value
            if isinstance(optimizer_value, str) and optimizer_value
            else None
        )
        if optimizer is None:
            missing.append("optimizer")
        else:
            seen_optimizers.add(optimizer)

        lr = _logged_float(config, "lr")
        if lr is None:
            missing.append("lr")
        lora_r = _logged_int(config, "lora_r")
        if lora_r is None:
            missing.append("lora_r")
        multiplier = _logged_float(config, "lora_plus_multiplier")
        if multiplier is None:
            missing.append("lora_plus_multiplier")

        losses: list[float] = []
        invalid_loss = False
        for event in record.history:
            value = event.get("eval_loss")
            if isinstance(value, bool):
                invalid_loss = True
                continue
            try:
                losses.append(float(value))
            except (TypeError, ValueError):
                invalid_loss = True
        if not losses or invalid_loss:
            missing.append("eval_loss")

        if missing:
            incomplete.append((record.physical_id, tuple(missing)))
        if optimizer is None or lr is None or not losses or invalid_loss:
            continue

        final = losses[-1]
        diverged = any(
            loss != loss or loss >= diverge_threshold for loss in losses
        )
        key = (optimizer, lora_r, multiplier)
        row = rows.setdefault(key, {"lrs": {}, "groups": set()})
        row["groups"].add(record.group)
        existing = row["lrs"].get(lr)
        if existing is None or (
            not diverged and (existing[1] or final < existing[0])
        ):
            row["lrs"][lr] = (final, diverged)

    def row_sort_key(item):
        (optimizer, lora_r, multiplier), _info = item
        return (
            optimizer,
            lora_r is None,
            -1 if lora_r is None else lora_r,
            multiplier is None,
            -1.0 if multiplier is None else multiplier,
        )

    coverage: list[CoverageRow] = []
    for (optimizer, lora_r, multiplier), info in sorted(
        rows.items(), key=row_sort_key
    ):
        lrs = tuple(sorted(info["lrs"]))
        non_diverged = [
            (lr, final_loss)
            for lr, (final_loss, diverged) in info["lrs"].items()
            if not diverged
        ]
        if non_diverged:
            best_lr, best_loss = min(non_diverged, key=lambda item: item[1])
        else:
            best_lr, best_loss = None, None
        coverage.append(CoverageRow(
            optimizer=optimizer,
            lora_r=lora_r,
            lora_plus_multiplier=multiplier,
            lrs_swept=lrs,
            best_lr=best_lr,
            final_loss_at_best=best_loss,
            pinning=_classify_pinning(lrs, best_lr),
            source_groups=tuple(sorted(info["groups"])),
        ))

    catalog_groups = set(on_disk)
    no_run_info: list[str] = []
    root = Path(logs_root)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in catalog_groups:
                continue
            if not (child / "run_info").exists() and any(child.iterdir()):
                no_run_info.append(child.name)

    return RunInventory(
        groups_on_disk=on_disk,
        groups_loaded=tuple(sorted(contributing_groups)),
        groups_orphaned=orphaned,
        groups_no_run_info=tuple(no_run_info),
        groups_without_records=tuple(sorted(
            catalog_groups - contributing_groups
        )),
        records_incomplete=tuple(sorted(incomplete)),
        optimizers_unknown=tuple(sorted(
            optimizer for optimizer in seen_optimizers
            if optimizer not in known_optimizers
        )),
        coverage=tuple(coverage),
    )


def render_inventory(inventory: RunInventory) -> str:
    """Render a compact notebook-friendly audit report."""
    lines = [
        f"Cataloged records from {len(inventory.groups_loaded)} of "
        f"{len(inventory.groups_on_disk)} physical log groups."
    ]

    if inventory.groups_orphaned:
        lines.extend((
            "",
            f"MANIFEST ANNOTATION ISSUES ({len(inventory.groups_orphaned)}) "
            "— records remain cataloged:",
        ))
        lines.extend(f"  {group}" for group in inventory.groups_orphaned)

    if inventory.groups_no_run_info:
        lines.extend((
            "",
            f"NO run_info/ ({len(inventory.groups_no_run_info)}) — files "
            "present but outside the RunCatalog layout:",
        ))
        lines.extend(f"  {group}" for group in inventory.groups_no_run_info)

    if inventory.groups_without_records:
        lines.extend((
            "",
            f"NO USABLE RECORDS ({len(inventory.groups_without_records)}) — "
            "physical logs were found but no config/eval record parsed:",
        ))
        lines.extend(
            f"  {group}" for group in inventory.groups_without_records
        )

    if inventory.records_incomplete:
        lines.extend((
            "",
            f"INCOMPLETE COVERAGE CONFIG ({len(inventory.records_incomplete)}) "
            "— logged fields were missing or invalid:",
        ))
        lines.extend(
            f"  {physical_id}: {', '.join(fields)}"
            for physical_id, fields in inventory.records_incomplete
        )

    if inventory.optimizers_unknown:
        lines.extend((
            "",
            f"UNKNOWN OPTIMIZERS ({len(inventory.optimizers_unknown)}) — in "
            "logs but missing from the supplied optimizer registry:",
        ))
        lines.extend(
            f"  {optimizer}" for optimizer in inventory.optimizers_unknown
        )

    lines.extend((
        "",
        f"Coverage: {len(inventory.coverage)} (optimizer, rank, mult) cells",
    ))
    if inventory.pinned:
        lines.append(
            f"PINNED at lr-range boundary ({len(inventory.pinned)}) — "
            "extension sweep recommended:"
        )
        for row in inventory.pinned:
            rank = "?" if row.lora_r is None else str(row.lora_r)
            if row.lora_plus_multiplier is None:
                multiplier = " m=unlogged"
            elif row.lora_plus_multiplier != 1.0:
                multiplier = f" m={row.lora_plus_multiplier:g}"
            else:
                multiplier = ""
            lines.append(
                f"  {row.optimizer:<32}  r={rank:<4}{multiplier:<11}  "
                f"best_lr={row.best_lr:.0e} "
                f"(final={row.final_loss_at_best:.4f}) "
                f"  swept={[f'{lr:.0e}' for lr in row.lrs_swept]} "
                f"→ {row.pinning}"
            )
    else:
        lines.append(
            "No (optimizer, rank, mult) cells pinned at lr-range boundary."
        )
    return "\n".join(lines)


__all__ = [
    "CoverageRow",
    "PINNING_ALL_DIVERGED",
    "PINNING_HIGH",
    "PINNING_INTERIOR",
    "PINNING_LOW",
    "PINNING_SINGLE",
    "RunInventory",
    "audit_run_catalog",
    "render_inventory",
]
