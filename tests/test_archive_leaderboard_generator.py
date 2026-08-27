"""Focused coverage for the archive-backed leaderboard generator."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lora_playground.publication_archive import (
    PublicationArchiveError,
    load_publication_archive,
)
from lora_playground.publication_queries import (
    publication_runs_for_workload,
    publication_workload_runs,
    publication_workloads,
)
from lora_playground.publication_views import load_publication_views
from lora_playground.workloads import (
    find_workload,
    resolve_record_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "publication" / "legacy_leaderboard_v1.json"
GENERATOR = ROOT / "scripts" / "analysis" / "build_leaderboard_doc.py"


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "archive_leaderboard_generator", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_generator_renders_archive_derived_workloads():
    generator = _generator_module()

    rendered = generator.render_doc(ARCHIVE)

    assert "### allenai/OLMo-2-0425-1B × opc × r=256" in rendered
    assert "checked-in records-native publication archive" in rendered
    assert "## Cross-setting robustness ranking" in rendered
    # The historical beta1 request was not executed and must not split the
    # archived publication view reconstructed by today's labeling defaults.
    assert all(
        "β1=0.95" not in line
        for line in rendered.splitlines()
        if line.startswith("| diag-Shampoo")
    )


def test_checked_in_archive_preserves_reviewed_evidence_counts():
    payload = json.loads(ARCHIVE.read_text())

    assert len(payload["runs"]) == 717
    assert sum(len(run["source_segments"]) for run in payload["runs"]) == 719
    assert sum(len(run["history"]) for run in payload["runs"]) == 25_812
    assert len(payload["variants"]) == 73
    assert sum(len(variant["exact_ids"]) for variant in payload["variants"]) == 170


def test_olmo_opc_r64_pins_recorded_publication_pipeline():
    archive = load_publication_archive(ARCHIVE)
    views = load_publication_views(
        ROOT / "publication" / "leaderboard_view.json",
        archive=archive,
    )
    report_runs = tuple(run for run in archive.runs if views.matches_workload(run))
    workloads = publication_workloads(report_runs, horizon=views.horizon)
    workload = next(
        workload for workload in workloads
        if workload.model_name == "allenai/OLMo-2-0425-1B"
        and workload.dataset_id == "opc"
        and workload.rank == 64
    )
    dimension_records = [
        run
        for run in archive.runs
        if run.effective_config["model_name"] == workload.model_name
        and run.effective_config["lora_r"] == workload.rank
        and resolve_record_dataset(run) == workload.dataset_id
    ]

    assert Counter(
        run.effective_config["data_pipeline_version"]
        for run in dimension_records
    ) == {"packed_v1": 6, "packed_v1.1": 83}
    selected = publication_runs_for_workload(report_runs, workload)
    assert len(selected) == 83
    assert workload.data_pipeline_version == "packed_v1.1"
    assert {
        run.effective_config["data_pipeline_version"] for run in selected
    } == {"packed_v1.1"}


def test_archive_generator_renders_all_pipeline_scoped_workloads():
    generator = _generator_module()

    rendered = generator.render_doc(ARCHIVE)

    assert rendered.count("\n### ") == 19
    assert "across the 18 pipeline-scoped" in rendered
    assert (
        "_Not scored: best LR 0.001 is boundary-pinned in [0.001, 0.003]"
        in rendered
    )
    cell = rendered.split(
        "### allenai/OLMo-2-0425-1B × opc × r=64", 1
    )[1].split("\n### ", 1)[0]
    assert "36.00×" not in cell


def test_archive_workload_selection_requires_recorded_pipeline():
    archive = load_publication_archive(ARCHIVE)
    workload = find_workload("OLMo-2-1B", "opc", 256)
    source = next(
        run
        for run in archive.runs
        if run.effective_config["model_name"] == workload.model_name
        and run.effective_config["lora_r"] == workload.rank
        and resolve_record_dataset(run) == workload.dataset
    )
    config = dict(source.effective_config)
    config.pop("data_pipeline_version")
    malformed = replace(source, effective_config=MappingProxyType(config))
    malformed_archive = replace(archive, runs=(malformed,))

    with pytest.raises(PublicationArchiveError, match="data_pipeline_version"):
        publication_workload_runs(malformed_archive, workload)


def test_missing_archive_does_not_create_output(tmp_path):
    generator = _generator_module()
    output = tmp_path / "leaderboard.md"
    missing = tmp_path / "missing.json"

    assert generator.main(["--archive", str(missing), "--output", str(output)]) == 0
    assert not output.exists()
    assert generator.main([
        "--archive", str(missing),
        "--output", str(output),
        "--require-archive",
    ]) == 2
    assert not output.exists()


def test_generator_does_not_import_the_manual_workload_registry():
    source = GENERATOR.read_text()
    assert "iter_workloads" not in source
    assert "WORKLOADS" not in source
    assert "find_workload" not in source
