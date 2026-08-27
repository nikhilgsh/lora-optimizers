"""Shared, records-native queries over a sealed publication archive.

Publication consumers should select data through these functions rather than
reconstructing historical optimizer defaults or maintaining log-group lists.
The archive owns reviewed legacy semantics; :mod:`workloads` owns comparison
cell identity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .publication_archive import PublicationArchive, PublicationArchiveError
from .workloads import Workload, resolve_record_dataset

if TYPE_CHECKING:
    from .publication_archive import ArchivedPublicationRun


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


__all__ = ["publication_workload_runs"]
