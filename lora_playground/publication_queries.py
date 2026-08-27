"""Shared, records-native queries over a sealed publication archive.

Publication consumers should select data through these functions rather than
reconstructing historical optimizer defaults or maintaining log-group lists.
The archive owns reviewed legacy semantics; :mod:`workloads` owns comparison
cell identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .publication_archive import PublicationArchive, PublicationArchiveError
from .workloads import Workload, resolve_record_dataset

if TYPE_CHECKING:
    from .publication_archive import ArchivedPublicationRun


@dataclass(frozen=True, slots=True)
class PublicationWorkload:
    """One workload identity derived from recorded archive semantics."""

    model_name: str
    dataset_id: str
    rank: int
    data_pipeline_version: str
    horizon: int

    @property
    def label(self) -> str:
        return (
            f"{self.model_name}|{self.dataset_id}|r={self.rank}|"
            f"{self.data_pipeline_version}"
        )

    @property
    def title(self) -> str:
        return f"{self.model_name} × {self.dataset_id} × r={self.rank}"


def publication_workloads(
    runs: Iterable["ArchivedPublicationRun"],
    *,
    horizon: int,
    completion_slack: int = 300,
) -> tuple[PublicationWorkload, ...]:
    """Derive completed workload cells from archive records, never a registry."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if (
        isinstance(completion_slack, bool)
        or not isinstance(completion_slack, int)
        or completion_slack < 0
    ):
        raise ValueError("completion_slack must be a non-negative integer")
    cells = set()
    for index, run in enumerate(runs):
        cfg = run.effective_config
        last_step = max(
            (event.get("step", 0) for event in run.history),
            default=0,
        )
        if last_step < horizon - completion_slack:
            continue
        model = cfg.get("model_name")
        dataset = resolve_record_dataset(run, index=index)
        rank = cfg.get("lora_r")
        pipeline = cfg.get("data_pipeline_version")
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(dataset, str)
            or not dataset
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or not isinstance(pipeline, str)
            or not pipeline
        ):
            raise PublicationArchiveError(
                f"archived run {run.physical_id!r} lacks a complete workload identity"
            )
        cells.add((model, dataset, rank, pipeline))
    return tuple(
        PublicationWorkload(model, dataset, rank, pipeline, horizon)
        for model, dataset, rank, pipeline in sorted(cells)
    )


def publication_runs_for_workload(
    runs: Iterable["ArchivedPublicationRun"],
    workload: PublicationWorkload,
) -> tuple["ArchivedPublicationRun", ...]:
    """Select one archive-derived workload from an already-scoped run set."""
    return tuple(
        run
        for index, run in enumerate(runs)
        if run.effective_config.get("model_name") == workload.model_name
        and run.effective_config.get("lora_r") == workload.rank
        and run.effective_config.get("data_pipeline_version")
        == workload.data_pipeline_version
        and resolve_record_dataset(run, index=index) == workload.dataset_id
    )


def publication_workload_runs(
    archive: PublicationArchive,
    workload: Workload,
) -> tuple["ArchivedPublicationRun", ...]:
    """Select one declared workload from immutable archived records.

    Model, rank, dataset, minimum horizon, and data-pipeline identity are all
    required to agree.  A missing pipeline is malformed publication evidence,
    not a cue to infer the current default.
    """
    if not isinstance(archive, PublicationArchive):
        raise TypeError("archive must be a PublicationArchive")
    if not isinstance(workload, Workload):
        raise TypeError("workload must be a Workload")

    selected = []
    for index, run in enumerate(archive.runs):
        cfg = run.effective_config
        max_steps = cfg.get("max_steps")
        if (
            cfg.get("model_name") != workload.model_name
            or cfg.get("lora_r") != workload.rank
        ):
            continue
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < workload.min_completed_steps
        ):
            continue
        if resolve_record_dataset(run, index=index) != workload.dataset:
            continue

        pipeline = cfg.get("data_pipeline_version")
        if not isinstance(pipeline, str) or not pipeline:
            raise PublicationArchiveError(
                f"archived run {run.physical_id!r} has no recorded "
                "config.data_pipeline_version"
            )
        if pipeline != workload.data_pipeline_version:
            continue
        selected.append(run)
    return tuple(selected)


__all__ = [
    "PublicationWorkload",
    "publication_runs_for_workload",
    "publication_workload_runs",
    "publication_workloads",
]
