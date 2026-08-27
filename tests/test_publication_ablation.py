"""Focused coverage for archive-backed derivation-ablation reports."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lora_playground.publication_ablation import (
    ABLATION_ARMS,
    ABLATION_HORIZON,
    ADAMW_ID,
    DEFAULT_PUBLICATION_ARCHIVE,
    KL_SHAMPOO_ID,
    PROTAGONIST_ID,
    PublicationAblationError,
    build_ablation_comparison,
    load_ablation_evidence,
    seed_trajectories,
)
from scripts.ablation_speedup import render_speedup
from scripts.ablation_table import render_table
from scripts.cell_sigma import render_sigma


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return load_ablation_evidence(DEFAULT_PUBLICATION_ARCHIVE)


@pytest.fixture(scope="module")
def comparison(evidence):
    return build_ablation_comparison(evidence, horizon=ABLATION_HORIZON)


def test_ablation_reports_have_no_legacy_loader_boundary():
    for relative_path in (
        "scripts/ablation_table.py",
        "scripts/ablation_speedup.py",
        "scripts/cell_sigma.py",
        "lora_playground/publication_ablation.py",
    ):
        source = (ROOT / relative_path).read_text()
        assert "lora_playground.loader" not in source
        assert "from lora_playground.loader import" not in source


def test_archive_selects_one_explicit_pipeline_scoped_workload(evidence):
    assert evidence.workload.model_name == "meta-llama/Llama-3.2-1B"
    assert evidence.workload.dataset == "openmath"
    assert evidence.workload.rank == 256
    assert evidence.workload.data_pipeline_version == "packed_v1.1"
    assert len(evidence.runs) == 100
    assert {
        run.effective_config["data_pipeline_version"]
        for run in evidence.runs
    } == {"packed_v1.1"}

    declared_ids = {
        arm.variant_id for arm in ABLATION_ARMS if arm.variant_id is not None
    }
    assert {spec.id for spec in evidence.selected_specs} == declared_ids
    missing = [arm.label for arm in ABLATION_ARMS if arm.variant_id is None]
    assert missing == ["w/o msign (metric^-1/2)"]


def test_comparison_uses_explicit_variants_and_replicate_mean(comparison):
    assert set(comparison.completed) == {
        arm.variant_id for arm in ABLATION_ARMS if arm.variant_id is not None
    }
    protagonist = comparison.completed[PROTAGONIST_ID][0.01]
    assert protagonist.n_replicates == 4
    assert protagonist.final_loss == pytest.approx(0.36495, abs=5e-5)
    assert comparison.best_completed[ADAMW_ID].lr == pytest.approx(1e-4)


def test_seed_spread_selects_exact_variant_and_lr(evidence):
    protagonist = seed_trajectories(
        evidence.runs,
        variant_id=PROTAGONIST_ID,
        lr=0.01,
    )
    shampoo = seed_trajectories(
        evidence.runs,
        variant_id=KL_SHAMPOO_ID,
        lr=0.01,
    )

    assert set(protagonist) == {0, 1, 2, 3}
    assert set(shampoo) == {0, 1, 2, 3}
    assert all(ABLATION_HORIZON in trajectory for trajectory in protagonist.values())


def test_seed_spread_rejects_equal_depth_duplicate(evidence):
    source = next(
        run
        for run in evidence.runs
        if run.effective_config["_publication_variant_id"] == PROTAGONIST_ID
        and run.effective_config["lr"] == 0.01
        and run.effective_config["seed"] == 0
    )
    duplicate = replace(source, physical_id=f"{source.physical_id}/duplicate")

    with pytest.raises(PublicationAblationError, match="equal-depth"):
        seed_trajectories(
            (source, duplicate),
            variant_id=PROTAGONIST_ID,
            lr=0.01,
        )


def test_reports_preserve_rows_and_expose_unexecuted_arm(evidence, comparison):
    table = render_table(comparison, step=None, all_steps=False)
    speedup = render_speedup(comparison, horizon=ABLATION_HORIZON)
    sigma = render_sigma(evidence, requested_step=None)

    assert "step-matched at 9000/9000" in table
    assert "PoLoRA (protagonist)             0.01" in table
    assert "w/o msign (metric^-1/2)" in table
    assert "no data yet" in table
    assert "target = tuned Adam final loss at step 9000" in speedup
    assert "w/o msign (metric^-1/2)" in speedup
    assert "[0, 1, 2, 3]   9000" in sigma
