#!/usr/bin/env python3
"""Verify a notebook cell executes without error.

Usage:
  verify_notebook_cell.py <notebook.ipynb> --cell-id <id>
  verify_notebook_cell.py <notebook.ipynb> --all-unrun

Builds a Python script by concatenating all code cells preceding the
target (in notebook order) plus the target itself, then executes it in a
fresh subprocess with matplotlib's Agg backend. Reports OK or the full
traceback. Exit 0 on success, 1 on cell error, 2 on usage error.

Notes on the concat model
-------------------------
Notebook cells implicitly depend on prior cells (imports, module-level
helpers, color maps, etc.). To verify a single cell honestly we must
re-execute every cell before it. This is slower than the live kernel —
typically 10s-2min depending on what the imports load — but it's the
only way to catch "I edited the cell in isolation but it relies on a
helper from cell 3 that I forgot to update."

Magics (`%load_ext`, `%autoreload`, `!` shell calls) are stripped: they
don't parse as Python, and the autoreload semantics don't matter in a
fresh process. If a cell genuinely needs IPython runtime behavior beyond
`display()`, this verifier won't catch a bug in that code path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else (src or "")


def _strip_magics(src: str) -> str:
    """Drop IPython magics + shell calls. Keep everything else verbatim."""
    out = []
    for ln in src.split("\n"):
        stripped = ln.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        out.append(ln)
    return "\n".join(out)


def build_script(nb_path: str, target_cell_id: str) -> str | None:
    """Concatenate code cells up to and including target_cell_id."""
    nb = nbformat.read(nb_path, as_version=4)
    parts = [
        "import sys, os",
        # display() is used in notebooks but not built into Python.
        "try:",
        "    from IPython.display import display",
        "except ImportError:",
        "    def display(x): print(x)",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import warnings",
        "warnings.filterwarnings('ignore')",
    ]
    found = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = _strip_magics(_cell_source(cell))
        cid = cell.get("id", "?")
        parts.append(f"\n# === cell {cid} ===")
        parts.append(src)
        if cid == target_cell_id:
            found = True
            break
    if not found:
        return None
    return "\n".join(parts)


def run(nb_path: str, cell_id: str, cwd: str | None = None, timeout: int = 600) -> int:
    script = build_script(nb_path, cell_id)
    if script is None:
        print(f"ERROR: cell id {cell_id!r} not found in {nb_path}", file=sys.stderr)
        return 2

    if cwd is None:
        cwd = str(Path(nb_path).parent.resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        os.unlink(script_path)
        print(f"VERIFY TIMEOUT for cell {cell_id} after {timeout}s", file=sys.stderr)
        return 1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"VERIFY FAILED for cell {cell_id}")
        if result.stdout:
            print("--- stdout (tail) ---")
            print(result.stdout[-3000:])
        if result.stderr:
            print("--- stderr (tail) ---")
            print(result.stderr[-3000:])
        return 1
    print(f"VERIFY OK for cell {cell_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook")
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--cwd", default=None,
                    help="Working directory for the verification subprocess "
                         "(default: notebook's parent dir).")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Subprocess timeout in seconds (default 600).")
    args = ap.parse_args()
    return run(args.notebook, args.cell_id, args.cwd, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
