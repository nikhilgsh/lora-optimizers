"""Records-native publication projections and automatic comparison specs.

Versioned producers record one normalized optimizer-semantic snapshot.
Historical archives carry the equivalent reviewed projection explicitly.
Exact provenance IDs may map many-to-one onto a publication view key, but one
view key always has one sealed/display label.  Labels never define identity.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .comparison import VariantSpec
from .publication_semantics import (
    PublicationSemanticsError,
    PublicationVariantSemantics,
    publication_semantics_from_payload,
)
from .run_records import freeze_value, run_view, thaw_value


PUBLICATION_VARIANT_ID_FIELD = "_publication_variant_id"
PUBLICATION_EXACT_ID_FIELD = "_publication_exact_id"
PUBLICATION_VARIANT_LABEL_FIELD = "_publication_variant_label"
PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD = (
    "_publication_optimizer_semantic_key"
)
PUBLICATION_STYLE_KEY_FIELD = "_publication_style_key"
PRODUCER_SEMANTICS_FIELD = "optimizer_variant_semantics"

LabelAdapter = Callable[[Mapping[str, Any]], str | None]


class PublicationVariantProjectionError(ValueError):
    """A run cannot be assigned an unambiguous publication identity."""


@dataclass(frozen=True, slots=True)
class ProjectedPublicationRun:
    """Immutable run with publication identity overlaid on semantic config."""

    physical_id: str
    effective_config: Mapping[str, Any]
    raw_config: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    group: str | None
    log_filename: str | None
    semantic_revisions: Mapping[str, Any]


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationVariantProjectionError(
            f"{context} must be a non-empty string"
        )
    return value


def stable_publication_variant_id(
    semantics: PublicationVariantSemantics,
) -> str:
    """Return the exact ID of one already-normalized semantic snapshot."""
    if not isinstance(semantics, PublicationVariantSemantics):
        raise TypeError("semantics must be PublicationVariantSemantics")
    return semantics.exact_id


def _publication_fields(
    cfg: Mapping[str, Any],
    *,
    physical_id: str,
) -> tuple[str, str, str, str, str] | None:
    values = {
        "exact id": cfg.get(PUBLICATION_EXACT_ID_FIELD),
        "view key": cfg.get(PUBLICATION_VARIANT_ID_FIELD),
        "label": cfg.get(PUBLICATION_VARIANT_LABEL_FIELD),
        "optimizer semantic key": cfg.get(
            PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD
        ),
        "style key": cfg.get(PUBLICATION_STYLE_KEY_FIELD),
    }
    if all(value is None for value in values.values()):
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise PublicationVariantProjectionError(
            f"run {physical_id!r} is missing publication fields {missing!r}"
        )
    return tuple(
        _require_text(value, f"run {physical_id!r} publication {name}")
        for name, value in values.items()
    )


def _recorded_semantics(view: Any) -> PublicationVariantSemantics:
    payload = view.raw_config.get(PRODUCER_SEMANTICS_FIELD)
    if payload is None:
        raise PublicationVariantProjectionError(
            f"run {view.physical_id!r} has no producer-recorded "
            f"{PRODUCER_SEMANTICS_FIELD!r} block"
        )
    try:
        return publication_semantics_from_payload(payload)
    except PublicationSemanticsError as exc:
        raise PublicationVariantProjectionError(
            f"run {view.physical_id!r}: {exc}"
        ) from exc


def project_publication_runs(
    runs: Iterable[Any],
    *,
    label_adapter: LabelAdapter,
) -> tuple[ProjectedPublicationRun, ...]:
    """Project versioned or archived runs onto immutable publication identity.

    A previously projected/archive run carrying the complete publication
    fields is preserved verbatim. Otherwise the run must be versioned and
    contain the producer's normalized ``optimizer_variant_semantics`` block.
    The label adapter receives a deterministic rendering config derived from
    that same snapshot; labels and IDs therefore cannot consult different
    defaults or reconstructed fields.
    """
    if not callable(label_adapter):
        raise TypeError("label_adapter must be callable")

    projected: list[ProjectedPublicationRun] = []
    for index, run in enumerate(runs):
        view = run_view(run, index=index)
        existing = _publication_fields(
            view.semantic_config, physical_id=view.physical_id
        )
        if existing is None:
            if not view.is_versioned:
                raise PublicationVariantProjectionError(
                    f"unversioned run {view.physical_id!r} must come from a "
                    "publication archive with explicit publication fields"
                )
            semantics = _recorded_semantics(view)
            exact_id = semantics.exact_id
            view_key = semantics.view_key
            optimizer_semantic_key = semantics.view_key
            label = _require_text(
                label_adapter(thaw_value(semantics.label_config)),
                f"run {view.physical_id!r} publication variant label",
            )
            style_key = label
        else:
            exact_id, view_key, label, optimizer_semantic_key, style_key = existing

        cfg = dict(view.semantic_config)
        cfg[PUBLICATION_EXACT_ID_FIELD] = exact_id
        cfg[PUBLICATION_VARIANT_ID_FIELD] = view_key
        cfg[PUBLICATION_VARIANT_LABEL_FIELD] = label
        cfg[PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD] = optimizer_semantic_key
        cfg[PUBLICATION_STYLE_KEY_FIELD] = style_key
        projected.append(ProjectedPublicationRun(
            physical_id=view.physical_id,
            effective_config=freeze_value(cfg),
            raw_config=freeze_value(dict(view.raw_config)),
            history=tuple(freeze_value(dict(event)) for event in view.history),
            group=view.group,
            log_filename=view.log_filename,
            semantic_revisions=freeze_value(dict(view.semantic_revisions)),
        ))
    return tuple(projected)


def publication_variant_specs(
    runs: Iterable[Any],
) -> tuple[VariantSpec, ...]:
    """Build view-key specs automatically from projected/archive runs.

    Multiple exact producer IDs may intentionally map to one reviewed view
    key. A view key has exactly one label, style, and optimizer-semantic key;
    one label also names exactly one view key. Ordering is AdamW first.
    """
    exact_to_view: dict[str, str] = {}
    view_metadata: dict[str, tuple[str, str, str]] = {}
    label_to_view: dict[str, str] = {}
    for index, run in enumerate(runs):
        view = run_view(run, index=index)
        fields = _publication_fields(
            view.semantic_config, physical_id=view.physical_id
        )
        if fields is None:
            raise PublicationVariantProjectionError(
                f"run {view.physical_id!r} is not publication-projected"
            )
        exact_id, view_key, label, optimizer_semantic_key, style_key = fields
        prior_view = exact_to_view.get(exact_id)
        if prior_view is not None and prior_view != view_key:
            raise PublicationVariantProjectionError(
                f"publication exact id {exact_id!r} maps to both "
                f"{prior_view!r} and {view_key!r}"
            )
        metadata = (label, style_key, optimizer_semantic_key)
        prior_metadata = view_metadata.get(view_key)
        if prior_metadata is not None and prior_metadata != metadata:
            raise PublicationVariantProjectionError(
                f"publication view key {view_key!r} has conflicting metadata "
                f"{prior_metadata!r} and {metadata!r}"
            )
        prior_view = label_to_view.get(label)
        if prior_view is not None and prior_view != view_key:
            raise PublicationVariantProjectionError(
                f"publication variant label {label!r} maps to both "
                f"{prior_view!r} and {view_key!r}"
            )
        exact_to_view[exact_id] = view_key
        view_metadata[view_key] = metadata
        label_to_view[label] = view_key

    ordered = sorted(
        view_metadata.items(),
        key=lambda item: (item[1][0] != "AdamW", item[1][0], item[0]),
    )
    return tuple(
        VariantSpec(
            id=view_key,
            label=metadata[0],
            predicate={PUBLICATION_VARIANT_ID_FIELD: view_key},
            style_key=metadata[1],
            optimizer_semantic_key=_publication_optimizer_semantic_key,
        )
        for view_key, metadata in ordered
    )


def _publication_optimizer_semantic_key(cfg: Mapping[str, Any]) -> str:
    return _require_text(
        cfg.get(PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD),
        "publication optimizer semantic key",
    )


__all__ = [
    "PRODUCER_SEMANTICS_FIELD",
    "PUBLICATION_EXACT_ID_FIELD",
    "PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD",
    "PUBLICATION_STYLE_KEY_FIELD",
    "PUBLICATION_VARIANT_ID_FIELD",
    "PUBLICATION_VARIANT_LABEL_FIELD",
    "ProjectedPublicationRun",
    "PublicationVariantProjectionError",
    "project_publication_runs",
    "publication_variant_specs",
    "stable_publication_variant_id",
]
