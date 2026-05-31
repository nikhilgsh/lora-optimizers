"""CI gate: the leaderboard notebooks must execute end-to-end with zero error
outputs. Catches label drift, a collision-guard raise, or any broken cell
*before* it can render a misleading figure.

Opt-in (it executes full notebooks against logs/ — minutes, not seconds), so it
doesn't slow routine `pytest`. Run it in CI / pre-release with:

    RUN_NB_TESTS=1 python -m pytest tests/test_notebooks_execute.py -q

Skips when logs/ is empty (bare checkout) or the flag is unset.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

LEADERBOARD_NOTEBOOKS = [
    "opc_1b_leaderboard",
    "llama32_opc_1b_leaderboard",
    "llama32_openmath_1b_leaderboard",
    "openmath_1b_leaderboard",
    "tulu3_1b_leaderboard",
    "qwen25_opc_leaderboard",
    "damping_fullpolar_r256_leaderboard",
]

_logs = ROOT / "logs"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NB_TESTS") != "1"
    or not (_logs.exists() and any(_logs.iterdir())),
    reason="set RUN_NB_TESTS=1 (and have a populated logs/) to run the notebook-execution gate",
)


@pytest.mark.parametrize("nb", LEADERBOARD_NOTEBOOKS)
def test_notebook_executes_without_error(nb, tmp_path):
    src = NB_DIR / f"{nb}.ipynb"
    if not src.exists():
        pytest.skip(f"{nb}.ipynb not present")
    out = tmp_path / f"{nb}.executed.ipynb"
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", str(src),
         "--output", str(out), "--ExecutePreprocessor.timeout=420"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"{nb}: nbconvert failed\n{r.stderr[-1000:]}"
    cells = json.loads(out.read_text())["cells"]
    errors = [o.get("ename") for c in cells for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, f"{nb}: produced error output(s) {errors} (label drift / collision guard?)"
