"""Load small, declarative publication views over a sealed archive."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .comparison import VariantSpec
from .leaderboard_variants import (
    PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD,
    PUBLICATION_VARIANT_ID_FIELD,
)
from .publication_archive import ArchivedPublicationRun, PublicationArchive


PUBLICATION_VIEWS_SCHEMA_VERSION = 1
_ROOT_FIELDS = {
    "schema_version", "archive_projection_id", "horizon",
    "workload_selector", "views",
}
_VIEW_FIELDS = {"id", "title", "arms"}
_ARM_FIELDS = {"variant_id", "label", "style_key", "roles"}
_ROLES = {"reference", "target"}


class PublicationViewError(ValueError):
    """The view file is malformed or does not match its archive."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationViewError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationViewError(f"{name} must be a non-empty string")
    return value


def _fields(item: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extra = sorted(set(item) - allowed)
    if extra:
        raise PublicationViewError(f"{name} has unsupported field(s) {extra!r}")


@dataclass(frozen=True, slots=True)
class PublicationViewArm:
    variant_id: str
    label: str
    roles: tuple[str, ...] = ()
    style_key: str | None = None


@dataclass(frozen=True, slots=True)
class PublicationView:
    id: str
    arms: tuple[PublicationViewArm, ...]
    title: str | None = None

    @property
    def reference_id(self) -> str:
        return next(arm.variant_id for arm in self.arms if "reference" in arm.roles)

    @property
    def target_id(self) -> str | None:
        return next(
            (arm.variant_id for arm in self.arms if "target" in arm.roles), None
        )


@dataclass(frozen=True, slots=True)
class ResolvedPublicationView:
    view: PublicationView
    runs: tuple[ArchivedPublicationRun, ...]
    variant_specs: tuple[VariantSpec, ...]
    horizon: int

    @property
    def reference_id(self) -> str:
        return self.view.reference_id

    @property
    def target_id(self) -> str | None:
        return self.view.target_id


@dataclass(frozen=True, slots=True)
class PublicationViews:
    archive_projection_id: str
    horizon: int
    workload_selector: Mapping[str, Any]
    views: tuple[PublicationView, ...]

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

    def matches_workload(self, run: ArchivedPublicationRun) -> bool:
        from .workloads import resolve_record_dataset

        return all(
            (resolve_record_dataset(run) if field == "dataset_id"
             else run.effective_config.get(field)) == value
            for field, value in self.workload_selector.items()
        )

    def validate(self, archive: PublicationArchive) -> None:
        if archive.projection_id != self.archive_projection_id:
            raise PublicationViewError(
                f"view document targets archive {self.archive_projection_id!r}, "
                f"but supplied archive is {archive.projection_id!r}"
            )
        known = {variant.view_key for variant in archive.variants}
        present = {
            run.effective_config.get(PUBLICATION_VARIANT_ID_FIELD)
            for run in archive.runs if self.matches_workload(run)
        }
        for view in self.views:
            for arm in view.arms:
                if arm.variant_id not in known:
                    raise PublicationViewError(
                        f"view {view.id!r} references unknown archive variant "
                        f"{arm.variant_id!r}"
                    )
                if arm.variant_id not in present:
                    raise PublicationViewError(
                        f"view {view.id!r} arm {arm.variant_id!r} has no archived "
                        "run matching its workload selector"
                    )

    def resolve(
        self, view_id: str, archive: PublicationArchive
    ) -> ResolvedPublicationView:
        self.validate(archive)
        view = self.view(view_id)
        arm_ids = {arm.variant_id for arm in view.arms}
        styles = {variant.view_key: variant.style_key for variant in archive.variants}

        def semantic_key(cfg: Mapping[str, Any]) -> str:
            value = cfg.get(PUBLICATION_OPTIMIZER_SEMANTIC_KEY_FIELD)
            if not isinstance(value, str) or not value:
                raise PublicationViewError("publication run has no optimizer semantic key")
            return value

        return ResolvedPublicationView(
            view=view,
            runs=tuple(
                run for run in archive.runs
                if self.matches_workload(run)
                and run.effective_config.get(PUBLICATION_VARIANT_ID_FIELD) in arm_ids
            ),
            variant_specs=tuple(
                VariantSpec(
                    id=arm.variant_id,
                    label=arm.label,
                    predicate={PUBLICATION_VARIANT_ID_FIELD: arm.variant_id},
                    style_key=arm.style_key or styles[arm.variant_id],
                    optimizer_semantic_key=semantic_key,
                )
                for arm in view.arms
            ),
            horizon=self.horizon,
        )


def _arm(raw: Any, name: str) -> PublicationViewArm:
    item = _object(raw, name)
    _fields(item, _ARM_FIELDS, name)
    roles = item.get("roles", [])
    if not isinstance(roles, list) or any(role not in _ROLES for role in roles):
        raise PublicationViewError(f"{name}.roles must contain only {sorted(_ROLES)!r}")
    style = item.get("style_key")
    return PublicationViewArm(
        variant_id=_text(item.get("variant_id"), f"{name}.variant_id"),
        label=_text(item.get("label"), f"{name}.label"),
        roles=tuple(roles),
        style_key=None if style is None else _text(style, f"{name}.style_key"),
    )


def _view(raw: Any, name: str) -> PublicationView:
    item = _object(raw, name)
    _fields(item, _VIEW_FIELDS, name)
    arms_raw = item.get("arms")
    if not isinstance(arms_raw, list) or not arms_raw:
        raise PublicationViewError(f"{name}.arms must be a non-empty array")
    arms = tuple(
        _arm(raw_arm, f"{name}.arms[{i}]")
        for i, raw_arm in enumerate(arms_raw)
    )
    ids, labels = [arm.variant_id for arm in arms], [arm.label for arm in arms]
    references = sum("reference" in arm.roles for arm in arms)
    targets = sum("target" in arm.roles for arm in arms)
    if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
        raise PublicationViewError(f"{name} has duplicate variant IDs or labels")
    if references != 1 or targets > 1:
        raise PublicationViewError(
            f"{name} must declare exactly one reference and at most one target role"
        )
    title = item.get("title")
    return PublicationView(
        id=_text(item.get("id"), f"{name}.id"),
        title=None if title is None else _text(title, f"{name}.title"),
        arms=arms,
    )


def publication_views_from_payload(
    payload: Mapping[str, Any], *, archive: PublicationArchive | None = None
) -> PublicationViews:
    root = _object(payload, "publication views")
    _fields(root, _ROOT_FIELDS, "publication views")
    if root.get("schema_version") != PUBLICATION_VIEWS_SCHEMA_VERSION:
        raise PublicationViewError("unsupported publication views schema_version")
    horizon = root.get("horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise PublicationViewError("horizon must be a positive integer")
    selector = _object(root.get("workload_selector"), "workload_selector")
    if not selector or "data_dir" in selector:
        raise PublicationViewError(
            "workload_selector must use stable 'dataset_id', not physical 'data_dir'"
        )
    if any(
        not isinstance(value, (str, int, float, bool, type(None)))
        for value in selector.values()
    ):
        raise PublicationViewError("workload_selector values must be JSON scalar")
    raw_views = root.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise PublicationViewError("publication views.views must be a non-empty array")
    views = tuple(_view(raw, f"views[{i}]") for i, raw in enumerate(raw_views))
    if len({view.id for view in views}) != len(views):
        raise PublicationViewError("publication view document has duplicate view IDs")
    result = PublicationViews(
        archive_projection_id=_text(
            root.get("archive_projection_id"), "archive_projection_id"
        ),
        horizon=horizon,
        workload_selector=MappingProxyType(dict(selector)),
        views=views,
    )
    if archive is not None:
        result.validate(archive)
    return result


def load_publication_views(
    path: str | Path, *, archive: PublicationArchive | None = None
) -> PublicationViews:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationViewError(f"could not read publication views {path}: {exc}") from exc
    return publication_views_from_payload(payload, archive=archive)
