"""Opt-in CI gate: publication notebooks execute end-to-end without errors.

Catches label drift, collision-guard failures, stateful-cell breakage, or any
other execution error before a notebook can render a misleading figure.

Opt-in (it executes full notebooks against logs/ — minutes, not seconds), so it
doesn't slow routine `pytest`. Run it in CI / pre-release with:

    RUN_NB_TESTS=1 python -m pytest tests/test_notebooks_execute.py -q

Skips when logs/ is empty (bare checkout) or the flag is unset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_DIR = ROOT / "notebooks" / "leaderboard"
PAPER_NOTEBOOK = ROOT / "paper" / "paper_plots.ipynb"

LEADERBOARD_NOTEBOOKS = tuple(sorted(LEADERBOARD_DIR.glob("*.ipynb")))
PUBLICATION_NOTEBOOKS = (*LEADERBOARD_NOTEBOOKS, PAPER_NOTEBOOK)

_logs = ROOT / "logs"
requires_notebook_execution = pytest.mark.skipif(
    os.environ.get("RUN_NB_TESTS") != "1"
    or not (_logs.exists() and any(_logs.iterdir())),
    reason="set RUN_NB_TESTS=1 (and have a populated logs/) to run the notebook-execution gate",
)


def test_active_leaderboard_notebooks_are_discovered():
    """The execution gate follows the active directory instead of a stale list."""
    assert LEADERBOARD_NOTEBOOKS, f"no notebooks discovered under {LEADERBOARD_DIR}"
    assert all(path.parent == LEADERBOARD_DIR for path in LEADERBOARD_NOTEBOOKS)
    assert all(path.is_file() for path in PUBLICATION_NOTEBOOKS)


@requires_notebook_execution
@pytest.mark.parametrize("src", PUBLICATION_NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_executes_without_error(src, tmp_path):
    name = src.stem
    out = tmp_path / f"{name}.executed.ipynb"
    is_paper_notebook = src == PAPER_NOTEBOOK
    per_cell_timeout = 180 if is_paper_notebook else 420
    process_timeout = 900 if is_paper_notebook else 600
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
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(src),
            "--output",
            out.name,
            "--output-dir",
            str(tmp_path),
            f"--ExecutePreprocessor.timeout={per_cell_timeout}",
        ],
        capture_output=True, cwd=ROOT, env=env, text=True, timeout=process_timeout,
    )
    assert r.returncode == 0, f"{name}: nbconvert failed\n{r.stderr[-4000:]}"
    assert out.exists(), f"{name}: nbconvert did not write {out}"
    cells = json.loads(out.read_text())["cells"]
    errors = [
        (c.get("id"), o.get("ename"), o.get("evalue"))
        for c in cells for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name}: produced error output(s) {errors}"
    unexecuted = [
        c.get("id") for c in cells
        if c.get("cell_type") == "code"
        and "".join(c.get("source", [])).strip()
        and c.get("execution_count") is None
    ]
    assert not unexecuted, f"{name}: skipped code cell(s) {unexecuted}"
    png_cells = [
        c.get("id") for c in cells
        if any(o.get("data", {}).get("image/png") for o in c.get("outputs", []))
    ]
    assert png_cells, f"{name}: execution produced no inline PNG output"
