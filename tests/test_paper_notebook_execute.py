"""Fast real-kernel smoke for the paper plotting notebook.

The unit tests around ``paper_plots_lib`` deliberately replace rendering and
data access.  This test instead copies a small set of exact cells from the real
notebook and executes them through nbconvert.  It therefore covers IPython
magics, kernel imports, cross-cell state, the live-panel adapter, the
records-native panel adapter, and inline figure serialization without paying
the cost of all paper figures on every focused test run.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER_NOTEBOOK = ROOT / "paper" / "paper_plots.ipynb"
LOGS_ROOT = ROOT / "logs"


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", [])).strip()


def _one_cell(cells: list[dict], *, description: str, predicate) -> dict:
    matches = [cell for cell in cells if predicate(cell)]
    assert len(matches) == 1, (
        f"paper notebook must contain exactly one {description} cell; "
        f"found {len(matches)}"
    )
    return matches[0]


def _execution_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_path = [str(ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update({
        "IPYTHONDIR": str(tmp_path / "ipython"),
        "JUPYTER_RUNTIME_DIR": str(tmp_path / "jupyter-runtime"),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "PYTHONPATH": os.pathsep.join(python_path),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    })
    return env


def _assert_executed(cells: list[dict]) -> None:
    errors = [
        (cell.get("id"), output.get("ename"), output.get("evalue"))
        for cell in cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, f"paper notebook smoke produced error output(s): {errors}"

    unexecuted = [
        cell.get("id")
        for cell in cells
        if cell.get("cell_type") == "code"
        and _cell_source(cell)
        and cell.get("execution_count") is None
    ]
    assert not unexecuted, f"paper notebook smoke skipped code cell(s): {unexecuted}"

    for expression in (
        "P.panel_n(0)",
        "P.ablation_panel(256)",
        "P.rank_lr_panel()",
    ):
        cell = _one_cell(
            cells,
            description=repr(expression),
            predicate=lambda candidate, expression=expression: (
                any(
                    line.strip().startswith(expression)
                    for line in _cell_source(candidate).splitlines()
                )
            ),
        )
        png_outputs = [
            output
            for output in cell.get("outputs", [])
            if output.get("data", {}).get("image/png")
        ]
        assert png_outputs, f"{expression} executed but did not render an inline PNG"


@pytest.mark.skipif(
    not (LOGS_ROOT.exists() and any(LOGS_ROOT.iterdir())),
    reason="paper notebook smoke requires a populated logs/ tree",
)
def test_paper_notebook_priority_cells_execute_in_real_kernel(tmp_path):
    payload = json.loads(PAPER_NOTEBOOK.read_text())
    cells = payload["cells"]
    selected = [
        _one_cell(
            cells,
            description="setup",
            predicate=lambda cell: cell.get("id") == "setup",
        ),
        _one_cell(
            cells,
            description="P.panel_n(0)",
            predicate=lambda cell: _cell_source(cell).startswith("P.panel_n(0)"),
        ),
        _one_cell(
            cells,
            description="P.ablation_panel(256)",
            predicate=lambda cell: _cell_source(cell).startswith(
                "P.ablation_panel(256)"
            ),
        ),
        _one_cell(
            cells,
            description="P.rank_lr_panel()",
            predicate=lambda cell: _cell_source(cell).endswith(
                "P.rank_lr_panel()"
            ),
        ),
    ]
    smoke_cells = deepcopy(selected)
    for cell in smoke_cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    smoke = {
        "cells": smoke_cells,
        "metadata": deepcopy(payload.get("metadata", {})),
        "nbformat": payload["nbformat"],
        "nbformat_minor": payload["nbformat_minor"],
    }
    source = tmp_path / "paper_plots.smoke.ipynb"
    output = tmp_path / "paper_plots.smoke.executed.ipynb"
    source.write_text(json.dumps(smoke))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(source),
            "--output",
            output.name,
            "--output-dir",
            str(tmp_path),
            "--ExecutePreprocessor.timeout=90",
        ],
        capture_output=True,
        cwd=ROOT,
        env=_execution_env(tmp_path),
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "paper notebook real-kernel smoke failed\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert output.exists(), "nbconvert succeeded without writing its executed notebook"
    _assert_executed(json.loads(output.read_text())["cells"])
