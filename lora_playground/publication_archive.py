"""Sealed historical inputs for records-native publication generation.

Legacy runs predate explicit attempt lineage and complete effective-config
logging.  Reconstructing them with today's defaults makes old results change
when unrelated code moves.  A publication archive instead stores the reviewed
logical trajectory, stable variant identity, and minimal executed workload
fields once.  New versioned runs continue to come from :mod:`run_catalog`.

This module only defines and validates that immutable boundary.  It does not
contain a live-data exporter and never imports the legacy loader.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .run_records import freeze_value


ARCHIVE_SCHEMA_VERSION = 1
_REQUIRED_CONFIG_FIELDS = {
    "optimizer",
    "model_name",
    "data_dir",
    "lora_r",
    "lr",
    "max_steps",
}


class PublicationArchiveError(ValueError):
    """A historical publication archive is malformed or internally ambiguous."""


@dataclass(frozen=True, slots=True)
class PublicationVariant:
    """Stable comparison identity plus mutable presentation metadata."""

    id: str
    label: str
    style_key: str | None = None


@dataclass(frozen=True, slots=True)
class ArchivedPublicationRun:
    """One reviewed logical legacy trajectory accepted by ``build_comparison``."""

    physical_id: str
    effective_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    raw_config: Mapping[str, Any]
    source_physical_ids: tuple[str, ...]
    group: str
    log_filename: None = None
    semantic_revisions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class PublicationArchive:
    """Validated immutable archive payload."""

    projection_id: str
    variants: tuple[PublicationVariant, ...]
    runs: tuple[ArchivedPublicationRun, ...]

    @property
    def variants_by_id(self) -> Mapping[str, PublicationVariant]:
        return MappingProxyType({variant.id: variant for variant in self.variants})


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationArchiveError(f"{context} must be an object")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationArchiveError(f"{context} must be a non-empty string")
    return value


def _parse_variants(payload: Any) -> tuple[PublicationVariant, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise PublicationArchiveError("variants must be an array")
    variants: list[PublicationVariant] = []
    ids: set[str] = set()
    labels: set[str] = set()
    for index, raw_variant in enumerate(payload):
        item = _require_mapping(raw_variant, f"variants[{index}]")
        variant_id = _require_nonempty_string(
            item.get("id"), f"variants[{index}].id"
        )
        label = _require_nonempty_string(
            item.get("label"), f"variants[{index}].label"
        )
        style_key = item.get("style_key")
        if style_key is not None:
            style_key = _require_nonempty_string(
                style_key, f"variants[{index}].style_key"
            )
        if variant_id in ids:
            raise PublicationArchiveError(
                f"duplicate publication variant id {variant_id!r}"
            )
        if label in labels:
            raise PublicationArchiveError(
                f"duplicate publication variant label {label!r}"
            )
        ids.add(variant_id)
        labels.add(label)
        variants.append(PublicationVariant(variant_id, label, style_key))
    if not variants:
        raise PublicationArchiveError("archive must declare at least one variant")
    return tuple(variants)


def _parse_history(payload: Any, context: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise PublicationArchiveError(f"{context} must be an array")
    events: list[Mapping[str, Any]] = []
    steps: list[int | float] = []
    for index, raw_event in enumerate(payload):
        event = _require_mapping(raw_event, f"{context}[{index}]")
        step = event.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise PublicationArchiveError(
                f"{context}[{index}].step must be numeric"
            )
        if not math.isfinite(step) or step < 0:
            raise PublicationArchiveError(
                f"{context}[{index}].step must be finite and non-negative"
            )
        loss = event.get("eval_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise PublicationArchiveError(
                f"{context}[{index}].eval_loss must be numeric"
            )
        steps.append(step)
        events.append(freeze_value(dict(event)))
    if not events:
        raise PublicationArchiveError(f"{context} must not be empty")
    if steps != sorted(set(steps)):
        raise PublicationArchiveError(
            f"{context} steps must be strictly increasing and unique"
        )
    return tuple(events)


def publication_archive_from_payload(payload: Mapping[str, Any]) -> PublicationArchive:
    """Validate a decoded JSON payload and return immutable run objects."""
    root = _require_mapping(payload, "archive")
    if root.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise PublicationArchiveError(
            "unsupported publication archive schema_version "
            f"{root.get('schema_version')!r}; expected {ARCHIVE_SCHEMA_VERSION}"
        )
    projection_id = _require_nonempty_string(
        root.get("projection_id"), "projection_id"
    )
    variants = _parse_variants(root.get("variants"))
    variants_by_id = {variant.id: variant for variant in variants}
    variant_ids = set(variants_by_id)

    raw_runs = root.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise PublicationArchiveError("runs must be an array")

    logical_ids: set[str] = set()
    used_sources: set[str] = set()
    runs: list[ArchivedPublicationRun] = []
    for index, raw_run in enumerate(raw_runs):
        item = _require_mapping(raw_run, f"runs[{index}]")
        logical_id = _require_nonempty_string(
            item.get("logical_id"), f"runs[{index}].logical_id"
        )
        if logical_id in logical_ids:
            raise PublicationArchiveError(
                f"duplicate archived logical_id {logical_id!r}"
            )
        logical_ids.add(logical_id)

        variant_id = _require_nonempty_string(
            item.get("variant_id"), f"runs[{index}].variant_id"
        )
        if variant_id not in variant_ids:
            raise PublicationArchiveError(
                f"runs[{index}] references unknown variant {variant_id!r}"
            )

        raw_sources = item.get("source_physical_ids")
        if not isinstance(raw_sources, Sequence) or isinstance(
            raw_sources, (str, bytes)
        ):
            raise PublicationArchiveError(
                f"runs[{index}].source_physical_ids must be an array"
            )
        sources = tuple(
            _require_nonempty_string(source, f"runs[{index}].source_physical_ids")
            for source in raw_sources
        )
        if not sources or len(set(sources)) != len(sources):
            raise PublicationArchiveError(
                f"runs[{index}].source_physical_ids must be non-empty and unique"
            )
        overlap = used_sources.intersection(sources)
        if overlap:
            raise PublicationArchiveError(
                "physical source assigned to multiple archived logical runs: "
                f"{sorted(overlap)!r}"
            )
        used_sources.update(sources)

        config = dict(_require_mapping(item.get("config"), f"runs[{index}].config"))
        missing = sorted(_REQUIRED_CONFIG_FIELDS - set(config))
        if missing:
            raise PublicationArchiveError(
                f"runs[{index}].config is missing required fields {missing!r}"
            )
        for field_name in ("optimizer", "model_name", "data_dir"):
            _require_nonempty_string(
                config[field_name], f"runs[{index}].config.{field_name}"
            )
        for field_name in ("lora_r", "max_steps"):
            value = config[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PublicationArchiveError(
                    f"runs[{index}].config.{field_name} must be a positive int"
                )
        lr = config["lr"]
        if (isinstance(lr, bool) or not isinstance(lr, (int, float))
                or not math.isfinite(lr) or lr <= 0):
            raise PublicationArchiveError(
                f"runs[{index}].config.lr must be finite and positive"
            )
        config["_publication_variant_id"] = variant_id
        config["_publication_variant_label"] = variants_by_id[variant_id].label
        config["publication_projection_id"] = projection_id
        effective_config = freeze_value(config)
        history = _parse_history(item.get("history"), f"runs[{index}].history")

        physical_id = f"{projection_id}/{logical_id}"
        raw_config = freeze_value({
            "run_id": physical_id,
            "log_group": f"archive:{projection_id}",
            "source_physical_ids": list(sources),
            "publication_projection_id": projection_id,
            **({"_aborted": item["aborted"]} if "aborted" in item else {}),
        })
        runs.append(ArchivedPublicationRun(
            physical_id=physical_id,
            effective_config=effective_config,
            history=history,
            raw_config=raw_config,
            source_physical_ids=sources,
            group=f"archive:{projection_id}",
        ))

    return PublicationArchive(
        projection_id=projection_id,
        variants=variants,
        runs=tuple(sorted(runs, key=lambda run: run.physical_id)),
    )


def load_publication_archive(path: str | Path) -> PublicationArchive:
    """Load one checked-in publication archive without live-log fallback."""
    archive_path = Path(path)
    try:
        payload = json.loads(archive_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationArchiveError(
            f"could not read publication archive {archive_path}: {exc}"
        ) from exc
    return publication_archive_from_payload(payload)


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "ArchivedPublicationRun",
    "PublicationArchive",
    "PublicationArchiveError",
    "PublicationVariant",
    "load_publication_archive",
    "publication_archive_from_payload",
]
