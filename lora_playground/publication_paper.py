"""Small records-native boundary for paper panels backed by the sealed archive.

The checked-in publication archive owns reviewed historical semantics and
cohort membership.  Paper modules supply only an explicit mapping from their
editorial labels to the archive's sealed labels; comparison identity remains
the archive ``view_key`` throughout assignment and aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .comparison import ComparisonResult, build_comparison
from .leaderboard_variants import publication_variant_specs
from .publication_archive import PublicationArchive, load_publication_archive
from .publication_queries import publication_workload_runs
from .publication_views import PublicationViews, load_publication_views
from .workloads import Workload


ROOT = Path(__file__).absolute().parents[1]
DEFAULT_PUBLICATION_ARCHIVE = ROOT / "publication" / "legacy_leaderboard_v1.json"
DEFAULT_PUBLICATION_VIEWS = ROOT / "publication" / "paper_views.json"
LEGACY_ADAMW_VARIANT_LABEL = "AdamW"
LEGACY_POLORA_VARIANT_LABEL = (
    "KL-diag +polar PE=8 (f=10, β_c=0.99, δ=1e-4) "
    "H=8 precond_method=gram_ns"
)


@dataclass(frozen=True, slots=True)
class PublicationPanel:
    """One workload comparison plus its editorial-label to stable-ID map."""

    comparison: ComparisonResult
    variant_ids: Mapping[str, str]
    reference_id: str | None = None
    target_id: str | None = None
    horizon: int | None = None
    title: str | None = None

    def variant_id(self, label: str) -> str:
        try:
            return self.variant_ids[label]
        except KeyError as exc:
            raise KeyError(
                f"publication panel has no editorial label {label!r}; "
                f"known labels are {list(self.variant_ids)!r}"
            ) from exc


@lru_cache(maxsize=1)
def load_paper_publication_archive() -> PublicationArchive:
    """Load the repository's sealed publication evidence once per process."""
    return load_publication_archive(DEFAULT_PUBLICATION_ARCHIVE)


@lru_cache(maxsize=1)
def load_paper_publication_views() -> PublicationViews:
    """Load and validate the checked-in publication views once per process."""
    archive = load_paper_publication_archive()
    return load_publication_views(DEFAULT_PUBLICATION_VIEWS, archive=archive)


@lru_cache(maxsize=1)
def _default_specs_by_label():
    archive = load_paper_publication_archive()
    return MappingProxyType({
        spec.label: spec for spec in publication_variant_specs(archive.runs)
    })


def publication_panel(
    workload: Workload,
    variants: Mapping[str, str],
    *,
    horizon: int | None = None,
    completion_slack: int = 300,
    archive: PublicationArchive | None = None,
) -> PublicationPanel:
    """Build a paper comparison from sealed archive variant labels.

    ``variants`` maps the paper's editorial label to the exact label stored in
    the named archive projection.  Labels are used only to resolve that sealed
    projection once.  Each resulting :class:`VariantSpec` retains the archive's
    stable view ID, predicate, and optimizer-semantic key.
    """
    if not variants:
        raise ValueError("publication panel needs at least one variant")
    use_default_archive = archive is None
    if use_default_archive:
        archive = load_paper_publication_archive()
    assert archive is not None
    specs_by_label = (
        _default_specs_by_label()
        if use_default_archive
        else {
            spec.label: spec
            for spec in publication_variant_specs(archive.runs)
        }
    )
    if len(specs_by_label) != len(archive.variants):
        raise ValueError("publication archive labels do not resolve one-to-one")

    selected = []
    variant_ids: dict[str, str] = {}
    for editorial_label, sealed_label in variants.items():
        if not isinstance(editorial_label, str) or not editorial_label:
            raise ValueError("editorial publication labels must be non-empty strings")
        try:
            source = specs_by_label[sealed_label]
        except KeyError as exc:
            raise KeyError(
                f"sealed publication variant {sealed_label!r} is absent from "
                f"archive {archive.projection_id!r}"
            ) from exc
        if source.id in variant_ids.values():
            raise ValueError(
                f"publication view {source.id!r} was selected more than once"
            )
        selected.append(replace(
            source,
            label=editorial_label,
            style_key=editorial_label,
        ))
        variant_ids[editorial_label] = source.id

    records = publication_workload_runs(archive, workload)
    comparison = build_comparison(
        records,
        selected,
        horizon=workload.horizon if horizon is None else horizon,
        completion_slack=completion_slack,
    )
    return PublicationPanel(
        comparison=comparison,
        variant_ids=MappingProxyType(variant_ids),
        horizon=workload.horizon if horizon is None else horizon,
    )


def publication_view_panel(
    view_id: str,
    *,
    completion_slack: int = 300,
    archive: PublicationArchive | None = None,
    views: PublicationViews | None = None,
) -> PublicationPanel:
    """Build one comparison from a checked-in stable-ID publication view."""
    use_defaults = archive is None and views is None
    if archive is None:
        archive = load_paper_publication_archive()
    if views is None:
        views = (
            load_paper_publication_views()
            if use_defaults
            else load_publication_views(DEFAULT_PUBLICATION_VIEWS, archive=archive)
        )
    resolved = views.resolve(view_id, archive)
    comparison = build_comparison(
        resolved.runs,
        resolved.variant_specs,
        horizon=resolved.horizon,
        completion_slack=completion_slack,
    )
    return PublicationPanel(
        comparison=comparison,
        variant_ids=MappingProxyType({
            spec.label: spec.id for spec in resolved.variant_specs
        }),
        reference_id=resolved.reference_id,
        target_id=resolved.target_id,
        horizon=resolved.horizon,
        title=resolved.view.title,
    )


def labeled_completed(panel: PublicationPanel) -> dict[str, dict[float, tuple]]:
    """Return the established labeled-curve shape from one comparison result."""
    labels = {spec.id: spec.label for spec in panel.comparison.variants}
    return {
        labels[variant_id]: {
            lr: (curve.final_loss, [dict(event) for event in curve.history])
            for lr, curve in curves.items()
        }
        for variant_id, curves in panel.comparison.completed.items()
        if curves
    }


__all__ = [
    "DEFAULT_PUBLICATION_ARCHIVE",
    "DEFAULT_PUBLICATION_VIEWS",
    "LEGACY_ADAMW_VARIANT_LABEL",
    "LEGACY_POLORA_VARIANT_LABEL",
    "PublicationPanel",
    "labeled_completed",
    "load_paper_publication_archive",
    "load_paper_publication_views",
    "publication_panel",
    "publication_view_panel",
]
