"""Declarative publication views over a sealed publication archive.

This module owns only the schema and its validation.  Workload selectors,
ordered arm membership, editorial labels, styles, and reference/target roles
live in a JSON document.  Variant identity always remains the archive's stable
``PublicationVariant.view_key``; presentation text is never used for lookup.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .comparison import VariantSpec
from .leaderboard_variants import (
    PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD,
    PUBLICATION_VARIANT_ID_FIELD,
)
from .publication_archive import (
    ArchivedPublicationRun,
    PublicationArchive,
    PublicationVariant,
)


PUBLICATION_VIEWS_SCHEMA_VERSION = 1
DATASET_ID_SELECTOR_FIELD = "dataset_id"
_ROLES = frozenset({"reference", "target"})
_SCALAR_TYPES = (str, int, float, bool, type(None))


class PublicationViewError(ValueError):
    """A publication-view document is malformed or cannot resolve exactly."""


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationViewError(f"{context} must be an object")
    return value


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationViewError(f"{context} must be a non-empty string")
    return value


def _require_only(item: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unexpected = sorted(set(item) - allowed)
    if unexpected:
        raise PublicationViewError(
            f"{context} has unsupported field(s) {unexpected!r}"
        )


def _optimizer_semantic_key(cfg: Mapping[str, Any]) -> str:
    value = cfg.get(PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD)
    if not isinstance(value, str) or not value:
        raise PublicationViewError(
            "publication run has no optimizer semantic key"
        )
    return value


@dataclass(frozen=True, slots=True)
class PublicationViewArm:
    """One ordered presentation of an archive-stable variant identity."""

    variant_id: str
    label: str
    roles: tuple[str, ...] = ()
    style_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        _require_text(self.variant_id, "arm.variant_id")
        _require_text(self.label, "arm.label")
        if self.style_key is not None:
            _require_text(self.style_key, "arm.style_key")
        if len(set(self.roles)) != len(self.roles):
            raise PublicationViewError(
                f"arm {self.variant_id!r} has duplicate roles {self.roles!r}"
            )
        unknown = sorted(set(self.roles) - _ROLES)
        if unknown:
            raise PublicationViewError(
                f"arm {self.variant_id!r} has unsupported roles {unknown!r}"
            )


@dataclass(frozen=True, slots=True)
class PublicationView:
    """An ordered figure/report view, independent of archive display labels."""

    id: str
    arms: tuple[PublicationViewArm, ...]
    horizon: int
    title: str | None = None
    workload_selector: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        _require_text(self.id, "view.id")
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon <= 0
        ):
            raise PublicationViewError(
                f"view {self.id!r}.horizon must be a positive integer"
            )
        if self.title is not None:
            _require_text(self.title, f"view {self.id!r}.title")
        if not self.arms:
            raise PublicationViewError(f"view {self.id!r} must declare at least one arm")
        ids = [arm.variant_id for arm in self.arms]
        if len(set(ids)) != len(ids):
            raise PublicationViewError(f"view {self.id!r} has duplicate variant IDs")
        labels = [arm.label for arm in self.arms]
        if len(set(labels)) != len(labels):
            raise PublicationViewError(
                f"view {self.id!r} has duplicate editorial labels"
            )
        references = [arm.variant_id for arm in self.arms if "reference" in arm.roles]
        targets = [arm.variant_id for arm in self.arms if "target" in arm.roles]
        if len(references) != 1:
            raise PublicationViewError(
                f"view {self.id!r} must declare exactly one reference role"
            )
        if len(targets) > 1:
            raise PublicationViewError(
                f"view {self.id!r} may declare at most one target role"
            )
        if self.workload_selector is not None:
            selector = _require_mapping(
                self.workload_selector, f"view {self.id!r}.workload_selector"
            )
            if not selector:
                raise PublicationViewError(
                    f"view {self.id!r}.workload_selector must not be empty"
                )
            if "data_dir" in selector:
                raise PublicationViewError(
                    f"view {self.id!r}.workload_selector must use stable "
                    f"{DATASET_ID_SELECTOR_FIELD!r}, not physical 'data_dir'"
                )
            for field, value in selector.items():
                _require_text(field, f"view {self.id!r} workload field")
                if not isinstance(value, _SCALAR_TYPES):
                    raise PublicationViewError(
                        f"view {self.id!r}.workload_selector[{field!r}] "
                        "must be a JSON scalar"
                    )
                if isinstance(value, float) and not math.isfinite(value):
                    raise PublicationViewError(
                        f"view {self.id!r}.workload_selector[{field!r}] "
                        "must be finite"
                    )
                if (
                    field == DATASET_ID_SELECTOR_FIELD
                    and (not isinstance(value, str) or not value.strip())
                ):
                    raise PublicationViewError(
                        f"view {self.id!r}.workload_selector["
                        f"{DATASET_ID_SELECTOR_FIELD!r}] must be a non-empty string"
                    )
            object.__setattr__(
                self,
                "workload_selector",
                MappingProxyType(dict(selector)),
            )

    @property
    def reference_id(self) -> str:
        return next(
            arm.variant_id for arm in self.arms if "reference" in arm.roles
        )

    @property
    def target_id(self) -> str | None:
        return next(
            (arm.variant_id for arm in self.arms if "target" in arm.roles),
            None,
        )

    def matches_workload(self, run: ArchivedPublicationRun) -> bool:
        selector = self.workload_selector
        if selector is None:
            return True
        for field, value in selector.items():
            if field == DATASET_ID_SELECTOR_FIELD:
                # Historical archives predate an explicit dataset_id. Keep the
                # data_dir/command compatibility translation at this archive
                # query boundary; it never becomes a general loader inference.
                from .workloads import resolve_record_dataset

                if resolve_record_dataset(run) != value:
                    return False
                continue
            if (
                field not in run.effective_config
                or run.effective_config[field] != value
            ):
                return False
        return True


@dataclass(frozen=True, slots=True)
class ResolvedPublicationView:
    """A validated view plus the exact archive records/specs it selected."""

    view: PublicationView
    runs: tuple[ArchivedPublicationRun, ...]
    variant_specs: tuple[VariantSpec, ...]

    @property
    def reference_id(self) -> str:
        return self.view.reference_id

    @property
    def target_id(self) -> str | None:
        return self.view.target_id

    @property
    def horizon(self) -> int:
        return self.view.horizon


@dataclass(frozen=True, slots=True)
class PublicationViews:
    """One immutable view document targeting one archive projection."""

    archive_projection_id: str
    views: tuple[PublicationView, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", tuple(self.views))
        _require_text(self.archive_projection_id, "archive_projection_id")
        if not self.views:
            raise PublicationViewError("publication view document must declare views")
        ids = [view.id for view in self.views]
        if len(set(ids)) != len(ids):
            raise PublicationViewError("publication view document has duplicate view IDs")

    @property
    def views_by_id(self) -> Mapping[str, PublicationView]:
        return MappingProxyType({view.id: view for view in self.views})

    def view(self, view_id: str) -> PublicationView:
        try:
            return self.views_by_id[view_id]
        except KeyError as exc:
            raise KeyError(
                f"publication view {view_id!r} is not declared; "
                f"known views are {list(self.views_by_id)!r}"
            ) from exc

    def validate(self, archive: PublicationArchive) -> None:
        """Fail unless every declared identity resolves in its selected workload."""
        if not isinstance(archive, PublicationArchive):
            raise TypeError("archive must be a PublicationArchive")
        if archive.projection_id != self.archive_projection_id:
            raise PublicationViewError(
                f"view document targets archive {self.archive_projection_id!r}, "
                f"but supplied archive is {archive.projection_id!r}"
            )
        variants_by_view: dict[str, PublicationVariant] = {}
        for variant in archive.variants:
            if variant.view_key in variants_by_view:
                raise PublicationViewError(
                    f"archive has duplicate stable view ID {variant.view_key!r}"
                )
            variants_by_view[variant.view_key] = variant

        for view in self.views:
            selected_runs = tuple(
                run for run in archive.runs if view.matches_workload(run)
            )
            if not selected_runs:
                raise PublicationViewError(
                    f"view {view.id!r} workload selector matches no archived runs"
                )
            selected_ids = {
                run.effective_config.get(PUBLICATION_VARIANT_ID_FIELD)
                for run in selected_runs
            }
            for arm in view.arms:
                if arm.variant_id not in variants_by_view:
                    raise PublicationViewError(
                        f"view {view.id!r} references unknown archive variant "
                        f"{arm.variant_id!r}"
                    )
                if arm.variant_id not in selected_ids:
                    raise PublicationViewError(
                        f"view {view.id!r} arm {arm.variant_id!r} has no archived "
                        "run matching its workload selector"
                    )

    def resolve(
        self,
        view_id: str,
        archive: PublicationArchive,
    ) -> ResolvedPublicationView:
        """Return exact selected records and stable-ID specs for one view."""
        self.validate(archive)
        view = self.view(view_id)
        variants_by_view = {
            variant.view_key: variant for variant in archive.variants
        }
        arm_ids = {arm.variant_id for arm in view.arms}
        runs = tuple(
            run
            for run in archive.runs
            if view.matches_workload(run)
            and run.effective_config.get(PUBLICATION_VARIANT_ID_FIELD) in arm_ids
        )
        specs = tuple(
            VariantSpec(
                id=arm.variant_id,
                label=arm.label,
                predicate={PUBLICATION_VARIANT_ID_FIELD: arm.variant_id},
                style_key=(
                    arm.style_key
                    if arm.style_key is not None
                    else variants_by_view[arm.variant_id].style_key
                ),
                optimizer_semantic_key=_optimizer_semantic_key,
            )
            for arm in view.arms
        )
        return ResolvedPublicationView(view=view, runs=runs, variant_specs=specs)


def _parse_arm(payload: Any, context: str) -> PublicationViewArm:
    item = _require_mapping(payload, context)
    _require_only(item, {"variant_id", "label", "style_key", "roles"}, context)
    raw_roles = item.get("roles", ())
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes)):
        raise PublicationViewError(f"{context}.roles must be an array")
    roles = tuple(_require_text(role, f"{context}.roles") for role in raw_roles)
    style_key = item.get("style_key")
    return PublicationViewArm(
        variant_id=_require_text(item.get("variant_id"), f"{context}.variant_id"),
        label=_require_text(item.get("label"), f"{context}.label"),
        roles=roles,
        style_key=(
            None
            if style_key is None
            else _require_text(style_key, f"{context}.style_key")
        ),
    )


def _parse_view(payload: Any, context: str) -> PublicationView:
    item = _require_mapping(payload, context)
    _require_only(
        item,
        {"id", "title", "horizon", "workload_selector", "arms"},
        context,
    )
    raw_arms = item.get("arms")
    if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes)):
        raise PublicationViewError(f"{context}.arms must be an array")
    title = item.get("title")
    return PublicationView(
        id=_require_text(item.get("id"), f"{context}.id"),
        horizon=item.get("horizon"),
        title=None if title is None else _require_text(title, f"{context}.title"),
        workload_selector=item.get("workload_selector"),
        arms=tuple(
            _parse_arm(arm, f"{context}.arms[{index}]")
            for index, arm in enumerate(raw_arms)
        ),
    )


def publication_views_from_payload(
    payload: Mapping[str, Any],
    *,
    archive: PublicationArchive | None = None,
) -> PublicationViews:
    """Validate decoded JSON, optionally resolving it against an archive."""
    root = _require_mapping(payload, "publication views")
    _require_only(
        root,
        {"schema_version", "archive_projection_id", "views"},
        "publication views",
    )
    if root.get("schema_version") != PUBLICATION_VIEWS_SCHEMA_VERSION:
        raise PublicationViewError(
            "unsupported publication views schema_version "
            f"{root.get('schema_version')!r}; expected "
            f"{PUBLICATION_VIEWS_SCHEMA_VERSION}"
        )
    raw_views = root.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise PublicationViewError("publication views.views must be an array")
    result = PublicationViews(
        archive_projection_id=_require_text(
            root.get("archive_projection_id"), "archive_projection_id"
        ),
        views=tuple(
            _parse_view(view, f"views[{index}]")
            for index, view in enumerate(raw_views)
        ),
    )
    if archive is not None:
        result.validate(archive)
    return result


def load_publication_views(
    path: str | Path,
    *,
    archive: PublicationArchive | None = None,
) -> PublicationViews:
    """Load a declarative publication-view document from JSON."""
    view_path = Path(path)
    try:
        payload = json.loads(view_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationViewError(
            f"could not read publication views {view_path}: {exc}"
        ) from exc
    return publication_views_from_payload(payload, archive=archive)


__all__ = [
    "DATASET_ID_SELECTOR_FIELD",
    "PUBLICATION_VIEWS_SCHEMA_VERSION",
    "PublicationView",
    "PublicationViewArm",
    "PublicationViewError",
    "PublicationViews",
    "ResolvedPublicationView",
    "load_publication_views",
    "publication_views_from_payload",
]
