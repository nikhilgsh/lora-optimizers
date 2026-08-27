"""Small records-native boundary for paper panels backed by the sealed archive.

The checked-in archive owns reviewed historical semantics and cohort
membership. Declarative view files own presentation labels and roles;
comparison identity remains the archive ``view_key`` throughout aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .comparison import ComparisonResult, build_comparison
from .leaderboard_variants import PUBLICATION_VARIANT_ID_FIELD
from .publication_archive import PublicationArchive, load_publication_archive
from .publication_queries import publication_workload_runs
from .publication_views import PublicationViews, load_publication_views
from .workloads import Workload


ROOT = Path(__file__).absolute().parents[1]
DEFAULT_PUBLICATION_ARCHIVE = ROOT / "publication" / "legacy_leaderboard_v1.json"
DEFAULT_PUBLICATION_VIEWS = ROOT / "publication" / "paper_views.json"
DEFAULT_LEADERBOARD_VIEWS = ROOT / "publication" / "leaderboard_view.json"


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
def load_leaderboard_publication_views() -> PublicationViews:
    """Load cross-workload paper/report views once per process."""
    archive = load_paper_publication_archive()
    return load_publication_views(DEFAULT_LEADERBOARD_VIEWS, archive=archive)


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


def publication_workload_view_panel(
    view_id: str,
    workload: Workload,
    *,
    completion_slack: int = 300,
) -> PublicationPanel:
    """Apply one declarative stable-ID view to one archived workload."""
    archive = load_paper_publication_archive()
    resolved = load_leaderboard_publication_views().resolve(view_id, archive)
    arm_ids = {spec.id for spec in resolved.variant_specs}
    records = tuple(
        run for run in publication_workload_runs(archive, workload)
        if run.effective_config.get(PUBLICATION_VARIANT_ID_FIELD) in arm_ids
    )
    comparison = build_comparison(
        records,
        resolved.variant_specs,
        horizon=workload.horizon,
        completion_slack=completion_slack,
    )
    return PublicationPanel(
        comparison=comparison,
        variant_ids=MappingProxyType({
            spec.label: spec.id for spec in resolved.variant_specs
        }),
        reference_id=resolved.reference_id,
        target_id=resolved.target_id,
        horizon=workload.horizon,
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
    "DEFAULT_LEADERBOARD_VIEWS",
    "DEFAULT_PUBLICATION_VIEWS",
    "PublicationPanel",
    "labeled_completed",
    "load_leaderboard_publication_views",
    "load_paper_publication_archive",
    "load_paper_publication_views",
    "publication_view_panel",
    "publication_workload_view_panel",
]
