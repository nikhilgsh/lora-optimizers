"""Sealed historical inputs for records-native publication generation.

Legacy runs predate explicit attempt lineage and complete effective-config
logging.  Reconstructing them with today's defaults makes old results change
when unrelated code moves.  A publication archive instead stores the reviewed
logical trajectory, stable variant identity, and minimal executed workload
fields once.  New versioned runs continue to come from :mod:`run_catalog`.

Archive labels and styles are sealed presentation facts for this named
projection. A later publication view may build a new projection; it does not
silently relabel an existing archive. This module never imports the legacy
loader.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .publication_identity import (
    LORA_INIT_B_MODES,
    composite_publication_identity,
    require_lora_init_b,
    split_publication_identity,
)
from .run_records import freeze_value


ARCHIVE_SCHEMA_VERSION = 2
_REQUIRED_CONFIG_FIELDS = {
    "optimizer",
    "model_name",
    "data_dir",
    "lora_r",
    "lr",
    "max_steps",
    "data_pipeline_version",
    "lora_init_b",
    "measurement_semantics_revision",
}


class PublicationArchiveError(ValueError):
    """A historical publication archive is malformed or internally ambiguous."""


@dataclass(frozen=True, slots=True)
class PublicationVariant:
    """One reviewed view cohort and its exact producer identities."""

    view_key: str
    label: str
    optimizer_semantic_key: str
    lora_init_b: str
    exact_ids: tuple[str, ...]
    style_key: str


@dataclass(frozen=True, slots=True)
class ArchivedPublicationRun:
    """One reviewed logical legacy trajectory accepted by ``build_comparison``."""

    physical_id: str
    effective_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    raw_config: Mapping[str, Any]
    source_segments: tuple[Mapping[str, Any], ...]
    group: str
    log_filename: None = None
    semantic_revisions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def source_physical_ids(self) -> tuple[str, ...]:
        return tuple(str(segment["physical_id"]) for segment in self.source_segments)


@dataclass(frozen=True, slots=True)
class PublicationArchive:
    """Validated immutable archive payload."""

    projection_id: str
    variants: tuple[PublicationVariant, ...]
    runs: tuple[ArchivedPublicationRun, ...]

    @property
    def variants_by_id(self) -> Mapping[str, PublicationVariant]:
        return MappingProxyType({
            exact_id: variant
            for variant in self.variants
            for exact_id in variant.exact_ids
        })


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationArchiveError(f"{context} must be an object")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationArchiveError(f"{context} must be a non-empty string")
    return value


def _require_lora_init_b(value: Any, context: str) -> str:
    try:
        return require_lora_init_b(value)
    except ValueError as exc:
        raise PublicationArchiveError(
            f"{context}: {exc}"
        ) from exc


def _require_semantic_revision(value: Any, context: str) -> str | int:
    if isinstance(value, bool):
        raise PublicationArchiveError(
            f"{context} must be a non-empty string or positive integer"
        )
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise PublicationArchiveError(
        f"{context} must be a non-empty string or positive integer"
    )


def _parse_variants(payload: Any) -> tuple[PublicationVariant, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise PublicationArchiveError("variants must be an array")
    variants: list[PublicationVariant] = []
    exact_ids: set[str] = set()
    view_keys: set[str] = set()
    labels: set[str] = set()
    for index, raw_variant in enumerate(payload):
        item = _require_mapping(raw_variant, f"variants[{index}]")
        view_key = _require_nonempty_string(
            item.get("view_key"), f"variants[{index}].view_key"
        )
        label = _require_nonempty_string(
            item.get("label"), f"variants[{index}].label"
        )
        optimizer_semantic_key = _require_nonempty_string(
            item.get("optimizer_semantic_key"),
            f"variants[{index}].optimizer_semantic_key",
        )
        lora_init_b = _require_lora_init_b(
            item.get("lora_init_b"), f"variants[{index}].lora_init_b"
        )
        expected_view_key = composite_publication_identity(
            optimizer_semantic_key, lora_init_b
        )
        if view_key != expected_view_key:
            raise PublicationArchiveError(
                f"variants[{index}].view_key must compose optimizer_semantic_key "
                f"and lora_init_b; expected {expected_view_key!r}, got "
                f"{view_key!r}"
            )
        style_key = _require_nonempty_string(
            item.get("style_key"), f"variants[{index}].style_key"
        )
        raw_exact_ids = item.get("exact_ids")
        if not isinstance(raw_exact_ids, Sequence) or isinstance(
            raw_exact_ids, (str, bytes)
        ):
            raise PublicationArchiveError(
                f"variants[{index}].exact_ids must be an array"
            )
        cohort_exact_ids = tuple(
            _require_nonempty_string(
                exact_id, f"variants[{index}].exact_ids"
            )
            for exact_id in raw_exact_ids
        )
        if not cohort_exact_ids or len(set(cohort_exact_ids)) != len(
            cohort_exact_ids
        ):
            raise PublicationArchiveError(
                f"variants[{index}].exact_ids must be non-empty and unique"
            )
        for exact_id in cohort_exact_ids:
            try:
                _, exact_init = split_publication_identity(exact_id)
            except ValueError as exc:
                raise PublicationArchiveError(
                    f"variants[{index}].exact_ids: {exc}"
                ) from exc
            if exact_init != lora_init_b:
                raise PublicationArchiveError(
                    f"variants[{index}] exact id {exact_id!r} disagrees with "
                    f"lora_init_b={lora_init_b!r}"
                )
        overlap = exact_ids.intersection(cohort_exact_ids)
        if overlap:
            raise PublicationArchiveError(
                f"duplicate publication exact id {sorted(overlap)!r}"
            )
        if view_key in view_keys:
            raise PublicationArchiveError(
                f"duplicate publication view key {view_key!r}"
            )
        if label in labels:
            raise PublicationArchiveError(
                f"duplicate publication variant label {label!r}"
            )
        exact_ids.update(cohort_exact_ids)
        view_keys.add(view_key)
        labels.add(label)
        variants.append(PublicationVariant(
            view_key=view_key,
            label=label,
            optimizer_semantic_key=optimizer_semantic_key,
            lora_init_b=lora_init_b,
            exact_ids=cohort_exact_ids,
            style_key=style_key,
        ))
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
    variants_by_id = {
        exact_id: variant
        for variant in variants
        for exact_id in variant.exact_ids
    }
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

        exact_id = _require_nonempty_string(
            item.get("exact_id"), f"runs[{index}].exact_id"
        )
        if exact_id not in variant_ids:
            raise PublicationArchiveError(
                f"runs[{index}] references unknown exact id {exact_id!r}"
            )

        raw_segments = item.get("source_segments")
        if not isinstance(raw_segments, Sequence) or isinstance(
            raw_segments, (str, bytes)
        ):
            raise PublicationArchiveError(
                f"runs[{index}].source_segments must be an array"
            )
        segments: list[Mapping[str, Any]] = []
        for segment_index, raw_segment in enumerate(raw_segments):
            segment = _require_mapping(
                raw_segment,
                f"runs[{index}].source_segments[{segment_index}]",
            )
            physical_source = _require_nonempty_string(
                segment.get("physical_id"),
                f"runs[{index}].source_segments[{segment_index}].physical_id",
            )
            start = segment.get("contributed_start_step")
            end = segment.get("contributed_end_step")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in (start, end)
            ) or start > end:
                raise PublicationArchiveError(
                    f"runs[{index}].source_segments[{segment_index}] must "
                    "have finite non-negative start/end with start <= end"
                )
            if segments and start <= segments[-1]["contributed_end_step"]:
                raise PublicationArchiveError(
                    f"runs[{index}].source_segments must be strictly ordered "
                    "and non-overlapping"
                )
            segments.append(freeze_value({
                "physical_id": physical_source,
                "contributed_start_step": start,
                "contributed_end_step": end,
            }))
        sources = tuple(str(segment["physical_id"]) for segment in segments)
        if not sources or len(set(sources)) != len(sources):
            raise PublicationArchiveError(
                f"runs[{index}].source_segments must name unique physical sources"
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
        for field_name in (
            "optimizer", "model_name", "data_dir", "data_pipeline_version"
        ):
            _require_nonempty_string(
                config[field_name], f"runs[{index}].config.{field_name}"
            )
        lora_init_b = _require_lora_init_b(
            config["lora_init_b"], f"runs[{index}].config.lora_init_b"
        )
        _require_semantic_revision(
            config["measurement_semantics_revision"],
            f"runs[{index}].config.measurement_semantics_revision",
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
        variant = variants_by_id[exact_id]
        if lora_init_b != variant.lora_init_b:
            raise PublicationArchiveError(
                f"runs[{index}].config.lora_init_b={lora_init_b!r} disagrees "
                f"with variant mode {variant.lora_init_b!r}"
            )
        config["_publication_exact_id"] = exact_id
        config["_publication_variant_id"] = variant.view_key
        config["_publication_variant_label"] = variant.label
        config["_publication_optimizer_semantic_key"] = (
            variant.optimizer_semantic_key
        )
        config["_publication_style_key"] = variant.style_key
        config["publication_projection_id"] = projection_id
        effective_config = freeze_value(config)
        history = _parse_history(item.get("history"), f"runs[{index}].history")
        coverage = [0] * len(segments)
        for event in history:
            matches = [
                segment_index
                for segment_index, segment in enumerate(segments)
                if segment["contributed_start_step"]
                <= event["step"]
                <= segment["contributed_end_step"]
            ]
            if len(matches) != 1:
                raise PublicationArchiveError(
                    f"runs[{index}] history step {event['step']!r} is not "
                    "covered by exactly one source segment"
                )
            coverage[matches[0]] += 1
        if any(count == 0 for count in coverage):
            raise PublicationArchiveError(
                f"runs[{index}] has a source segment with no contributed history"
            )

        physical_id = f"{projection_id}/{logical_id}"
        semantic_revisions = freeze_value({
            "optimizer_impl": variant.optimizer_semantic_key,
            "measurement": config["measurement_semantics_revision"],
            "data_pipeline": config.get("data_pipeline_version"),
        })
        raw_config = freeze_value({
            "run_id": physical_id,
            "log_group": f"archive:{projection_id}",
            "source_physical_ids": list(sources),
            "source_segments": [dict(segment) for segment in segments],
            "publication_projection_id": projection_id,
            "semantic_revisions": semantic_revisions,
            **({"_aborted": item["aborted"]} if "aborted" in item else {}),
        })
        runs.append(ArchivedPublicationRun(
            physical_id=physical_id,
            effective_config=effective_config,
            history=history,
            raw_config=raw_config,
            source_segments=tuple(segments),
            group=f"archive:{projection_id}",
            semantic_revisions=semantic_revisions,
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
