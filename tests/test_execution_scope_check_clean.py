"""Tests for `python -m lora_playground.execution_scope check-clean`.

The CLI is the submission-side guard that must agree with the loader's
Phase-4 dirty-tree resolution: any working tree the CLI accepts must
produce a cfg event with execution_source_dirty=False so the loader
loads the resulting runs without an attestation.

Six cases:
  A — clean working tree → exit 0.
  B — edit a closure python file → exit 1; stdout names the file.
  C — edit a load-bearing shell script (scripts/sweep/*.sh) → exit 1.
  D — edit a non-load-bearing file (notebooks/) → exit 0.
  E — untracked .py file imported by the entry → exit 1.
  F — FORCE_DIRTY=1 → exit 0 even when dirty.

Each test builds a throwaway git repo in `tmp_path` whose layout mirrors
the project skeleton (train_lora.py + lora_playground package + a
scripts/sweep/ shell wrapper). Tests invoke the CLI as a subprocess so
they exercise the `__main__` path, exit code, and stderr emission.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(repo: Path, env_extra: dict[str, str] | None = None,
             ) -> tuple[int, str, str]:
    """Invoke `python -m lora_playground.execution_scope check-clean` with
    `repo` as cwd, using the production source tree (imports resolve from
    ROOT, not the fixture). We pass --root explicitly so the CLI scans
    the fixture, not the project.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if env_extra:
        env.update(env_extra)
    # cwd MUST be outside `repo` — Python prepends cwd to sys.path on
    # `-m`, and the fixture's empty lora_playground/__init__.py would
    # otherwise shadow the project module.
    proc = subprocess.run(
        [sys.executable, "-m", "lora_playground.execution_scope",
         "check-clean", "--root", str(repo), "--entry", "train_lora.py"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30.0,
    )
    return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=10.0,
    )
    return proc.stdout.decode()


def _build_fixture(tmp_path: Path) -> Path:
    """Build a minimal repo. The python import-closure is non-empty
    (train_lora.py → lora_playground.train → lora_playground.optim) and
    a load-bearing shell script lives at scripts/sweep/run.sh.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "train_lora.py").write_text(textwrap.dedent("""
        from lora_playground.train import main
        if __name__ == "__main__":
            main()
    """).strip() + "\n")

    pkg = repo / "lora_playground"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "train.py").write_text(textwrap.dedent("""
        from lora_playground.optim import build_optimizer
        def main():
            build_optimizer()
    """).strip() + "\n")
    (pkg / "optim.py").write_text(textwrap.dedent("""
        def build_optimizer():
            return None
    """).strip() + "\n")

    sweep_dir = repo / "scripts" / "sweep"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "run.sh").write_text("#!/bin/bash\necho stub\n")

    nb_dir = repo / "notebooks"
    nb_dir.mkdir()
    (nb_dir / "demo.ipynb").write_text('{"cells": [], "nbformat": 4}\n')

    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_a_clean_tree_exits_zero(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    code, _out, _err = _run_cli(repo)
    assert code == 0


def test_b_edit_closure_module_exits_one(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    (repo / "lora_playground" / "optim.py").write_text(
        "def build_optimizer():\n    return 'edited'\n"
    )
    code, _out, err = _run_cli(repo)
    assert code == 1
    assert "lora_playground/optim.py" in err


def test_c_edit_sweep_shell_exits_one(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    (repo / "scripts" / "sweep" / "run.sh").write_text("#!/bin/bash\necho EDITED\n")
    code, _out, err = _run_cli(repo)
    assert code == 1
    assert "scripts/sweep/run.sh" in err


def test_d_edit_notebook_is_clean(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    (repo / "notebooks" / "demo.ipynb").write_text(
        '{"cells": [{"cell_type": "code", "source": ["x"], "metadata": {}, '
        '"outputs": []}], "nbformat": 4}\n'
    )
    code, _out, _err = _run_cli(repo)
    assert code == 0


def test_e_untracked_imported_module_exits_one(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    # Add an untracked module that train.py imports — must block.
    (repo / "lora_playground" / "checkpoint.py").write_text(
        "def save():\n    pass\n"
    )
    (repo / "lora_playground" / "train.py").write_text(textwrap.dedent("""
        from lora_playground.optim import build_optimizer
        from lora_playground.checkpoint import save  # untracked
        def main():
            build_optimizer(); save()
    """).strip() + "\n")
    code, _out, err = _run_cli(repo)
    assert code == 1
    # Either the "missing from commit" branch (RuntimeError -> exit 1) or
    # the diff branch should surface checkpoint.py.
    assert "checkpoint.py" in err or "checkpoint" in err


def test_f_force_dirty_overrides(tmp_path: Path):
    repo = _build_fixture(tmp_path)
    (repo / "lora_playground" / "optim.py").write_text(
        "def build_optimizer():\n    return 'edited'\n"
    )
    code, _out, err = _run_cli(repo, env_extra={"FORCE_DIRTY": "1"})
    assert code == 0
    assert "FORCE_DIRTY" in err
