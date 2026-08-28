"""Tests for the notebook-derived paper-plot profiler."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "bench" / "baselines" / "paper_plots.json"


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

    # A FLOOR, not an equality. The exact count is a property of how many
    # figures the notebook currently draws, and pinning it here meant every
    # added figure failed a test about discovery rather than about the figure --
    # it read 28 while the notebook drew 33. The count that must stay in step is
    # the one in the checked baseline, which the test below compares and
    # `profile_paper_plots.py --no-baseline` regenerates deliberately. This
    # floor still catches discovery returning nothing or collapsing.
    assert len(expressions) >= 28
    assert expressions[0].startswith("P.panel_n(0)")
    assert "P.precond_panel(256)" in expressions
    assert "P.beta2_panel(256)" in expressions
    assert any(expression.endswith("P.rank_lr_panel()") for expression in expressions)


def test_checked_profile_baseline_matches_current_notebook_entrypoints():
    profiler = _load_profiler_module()
    baseline = json.loads(BASELINE.read_text())

    assert baseline["entrypoints"] == profiler.discover_entrypoints(
        ROOT / "paper" / "paper_plots.ipynb"
    )
    assert baseline["elapsed_sec"] > 0


def test_render_open_figures_encodes_png_bytes():
    profiler = _load_profiler_module()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(2, 1))
    plt.plot([0, 1], [0, 1])
    elapsed, png_bytes, figure_count = profiler.render_open_figures(plt)
    plt.close("all")

    assert elapsed > 0
    assert png_bytes > 0
    assert figure_count == 1


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


@pytest.mark.skipif(
    os.environ.get("RUN_PAPER_PLOT_PROFILE") != "1"
    or not (ROOT / "logs").is_dir(),
    reason="set RUN_PAPER_PLOT_PROFILE=1 with populated logs/ for the wall-time gate",
)
def test_paper_plot_profile_stays_within_checked_budget(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bench" / "profile_paper_plots.py"),
            "--out",
            str(tmp_path / "paper_plots_profile.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"paper plot profile regressed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
