"""Tests for the notebook-derived paper-plot profiler."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_profiler_module():
    import importlib.util

    path = ROOT / "scripts" / "bench" / "profile_paper_plots.py"
    spec = importlib.util.spec_from_file_location("profile_paper_plots", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_harness_discovers_current_notebook_entrypoints():
    profiler = _load_profiler_module()
    expressions = profiler.discover_entrypoints(ROOT / "paper" / "paper_plots.ipynb")

    assert len(expressions) == 28
    assert expressions[0].startswith("P.panel_n(0)")
    assert "P.precond_panel(256)" in expressions
    assert "P.beta2_panel(256)" in expressions
    assert any(expression.endswith("P.rank_lr_panel()") for expression in expressions)


def test_profile_comparison_detects_regression_for_same_entrypoints():
    profiler = _load_profiler_module()
    baseline = {
        "entrypoints": ["P.panel_n(0)"],
        "elapsed_sec": 10.0,
    }
    current = {**baseline, "elapsed_sec": 13.0}

    comparison = profiler.compare_results(
        current, baseline, max_regression_fraction=0.2
    )

    assert comparison["ratio"] == 1.3
    assert comparison["limit_sec"] == 12.0
    assert comparison["passed"] is False


def test_profile_comparison_rejects_different_entrypoints():
    profiler = _load_profiler_module()
    baseline = {
        "entrypoints": ["P.panel_n(0)"],
        "elapsed_sec": 10.0,
    }
    current = {
        **baseline,
        "entrypoints": ["P.panel_n(1)"],
    }

    with pytest.raises(ValueError, match="entrypoints do not match"):
        profiler.compare_results(
            current, baseline, max_regression_fraction=0.2
        )
