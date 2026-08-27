"""Execution-source coherence at the LR-curve join boundary."""
from __future__ import annotations

import pytest

from lora_playground.plotting.dedup import (
    SourceCoherenceError,
    assert_curve_source_coherent,
    assert_label_discriminates,
    dedup_by_canonical,
    filter_curve_sources,
)


OLD_SOURCE = "2894c0df9118aea64b744b7a1add3195dbe7b9fd07132b1bb74bf1429ff58a2a"
NEW_SOURCE = "801a80d25a8e489a9d67e782df4221752811f9d2dbc228c8b01862e90fe62ee5"


def _run(lr, source, *, label="factorwise", commit=None, step=9000):
    cfg = {
        "optimizer": "adamw",
        "label": label,
        "model_name": "model",
        "data_dir": "dataset",
        "data_pipeline_version": "packed_v1",
        "lora_r": 16,
        "lr": lr,
        "seed": 0,
        "execution_source_sha": source,
        "git_commit": commit or source[:7],
    }
    return cfg, [{"step": step, "eval_loss": 1.0}]


def _label(cfg):
    return cfg["label"]


def test_disjoint_lr_source_splice_is_known_positive():
    """The ordinary series-id guard misses exactly the observed failure."""
    runs = [
        _run(1e-3, "pre-fix", commit="old"),
        _run(3e-3, "pre-fix", commit="old"),
        _run(1e-2, "post-fix", commit="new"),
        _run(3e-2, "post-fix", commit="new"),
    ]
    assert_label_discriminates(runs, _label)
    with pytest.raises(SourceCoherenceError) as exc_info:
        assert_curve_source_coherent(runs, _label)
    message = str(exc_info.value)
    assert "pre-fix" in message
    assert "post-fix" in message
    assert "0.001" in message
    assert "0.03" in message


def test_exact_source_filter_excludes_legacy_and_labeler_fails_closed():
    old = _run(1e-3, OLD_SOURCE)
    current = _run(1e-2, NEW_SOURCE)
    kept, excluded, checked_key = filter_curve_sources(
        [old, current],
        _label,
        {"factorwise": {NEW_SOURCE}},
    )
    assert kept == [current]
    assert [run for run, _reason in excluded] == [old]
    assert checked_key(current[0]) == "factorwise"
    with pytest.raises(SourceCoherenceError, match="not in the allowed set"):
        checked_key(old[0])


def test_exact_equivalence_override_is_scoped_to_displayed_label():
    runs = [
        _run(1e-3, "old", label="one-sided"),
        _run(1e-2, "new", label="one-sided"),
    ]
    assert_curve_source_coherent(
        runs,
        _label,
        equivalent_source_groups={"one-sided": [{"old", "new"}]},
    )

    factorwise = [
        _run(1e-3, "old", label="factorwise"),
        _run(1e-2, "new", label="factorwise"),
    ]
    with pytest.raises(SourceCoherenceError):
        assert_curve_source_coherent(
            factorwise,
            _label,
            equivalent_source_groups={"one-sided": [{"old", "new"}]},
        )


def test_same_execution_source_across_git_commits_is_coherent():
    runs = [
        _run(1e-3, "same-content", commit="commit-a"),
        _run(1e-2, "same-content", commit="commit-b"),
    ]
    assert_curve_source_coherent(runs, _label)


def test_missing_source_falls_back_to_git_commit_and_missing_all_fails():
    old = _run(1e-3, "unused", commit="commit-a")
    new = _run(1e-2, "unused", commit="commit-b")
    old[0]["execution_source_sha"] = None
    new[0]["execution_source_sha"] = None
    with pytest.raises(SourceCoherenceError):
        assert_curve_source_coherent([old, new], _label)

    new[0]["git_commit"] = None
    with pytest.raises(SourceCoherenceError, match="missing both"):
        assert_curve_source_coherent([old, new], _label)


def test_same_cell_runtime_source_difference_still_deduplicates_first():
    """Source remains runtime metadata; it must not split same-cell reruns."""
    old = _run(1e-3, "old", step=1000)
    new = _run(1e-3, "new", step=9000)
    deduped = dedup_by_canonical([old, new])
    assert len(deduped) == 1
    assert deduped[0][0]["execution_source_sha"] == "new"
    assert_curve_source_coherent(deduped, _label)
