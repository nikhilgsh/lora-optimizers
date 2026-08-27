"""The leaderboard hook reminds; explicit commands generate and stage."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(repo: Path, *args: str, check: bool = True, env=None):
    return subprocess.run(
        args,
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
        env=env,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _init_repo(repo: Path) -> None:
    _run(repo, "git", "init", "-q")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")


def _commit_all(repo: Path) -> None:
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-qm", "fixture")


@pytest.mark.parametrize(
    "staged_path",
    [
        "publication/legacy_leaderboard_v1.json",
        "lora_playground/leaderboard_variants.py",
    ],
)
def test_hook_warns_without_invoking_generator_for_relevant_staged_paths(
    tmp_path: Path, staged_path: str,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    shutil.copy2(ROOT / "githooks" / "pre-commit", repo / "pre-commit")
    _write(
        repo / "scripts/analysis/update_leaderboard.sh",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$LEADERBOARD_HOOK_MARKER\"\n",
    )
    os.chmod(repo / "scripts/analysis/update_leaderboard.sh", 0o755)
    _write(repo / "README.md", "base\n")
    _write(repo / "lora_playground/leaderboard_variants.py", "base\n")
    _write(repo / "publication/legacy_leaderboard_v1.json", "{}\n")
    _write(repo / "docs/notes/leaderboard.md", "base\n")
    _commit_all(repo)

    target = repo / staged_path
    target.write_text("changed\n")
    _run(repo, "git", "add", staged_path)
    marker = repo / "hook-called"
    env = {**os.environ, "LEADERBOARD_HOOK_MARKER": str(marker)}
    result = _run(repo, "bash", "pre-commit", env=env)
    assert not marker.exists()
    assert "leaderboard inputs changed" in result.stderr
    assert "regenerate and review" in result.stderr


@pytest.mark.parametrize(
    "staged_path", ["README.md", "lora_playground/optim.py"]
)
def test_hook_is_noop_for_unrelated_staged_path(
    tmp_path: Path, staged_path: str
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    shutil.copy2(ROOT / "githooks" / "pre-commit", repo / "pre-commit")
    _write(
        repo / "scripts/analysis/update_leaderboard.sh",
        "#!/usr/bin/env bash\nprintf called >> \"$LEADERBOARD_HOOK_MARKER\"\n",
    )
    os.chmod(repo / "scripts/analysis/update_leaderboard.sh", 0o755)
    _write(repo / "README.md", "base\n")
    _write(repo / "lora_playground/optim.py", "base\n")
    _commit_all(repo)

    _write(repo / staged_path, "unrelated\n")
    _run(repo, "git", "add", staged_path)
    marker = repo / "hook-called"
    env = {**os.environ, "LEADERBOARD_HOOK_MARKER": str(marker)}
    _run(repo, "bash", "pre-commit", env=env)

    assert not marker.exists()


def test_hook_is_silent_when_reviewed_doc_is_staged_with_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    shutil.copy2(ROOT / "githooks" / "pre-commit", repo / "pre-commit")
    _write(repo / "lora_playground/leaderboard.py", "base\n")
    _write(repo / "docs/notes/leaderboard.md", "base\n")
    _commit_all(repo)

    _write(repo / "lora_playground/leaderboard.py", "changed\n")
    _write(repo / "docs/notes/leaderboard.md", "reviewed\n")
    _run(
        repo,
        "git",
        "add",
        "lora_playground/leaderboard.py",
        "docs/notes/leaderboard.md",
    )

    result = _run(repo, "bash", "pre-commit")
    assert result.stderr == ""


def _workflow_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "githooks").mkdir()
    (repo / "scripts/analysis").mkdir(parents=True)
    shutil.copy2(ROOT / "githooks" / "pre-commit", repo / "githooks/pre-commit")
    shutil.copy2(
        ROOT / "scripts/analysis/update_leaderboard.sh",
        repo / "scripts/analysis/update_leaderboard.sh",
    )
    os.chmod(repo / "githooks/pre-commit", 0o755)
    os.chmod(repo / "scripts/analysis/update_leaderboard.sh", 0o755)
    _write(repo / "lora_playground/__init__.py", "")
    _write(repo / "lora_playground/leaderboard.py", "committed-source\n")
    _write(repo / "publication/legacy_leaderboard_v1.json", "{}\n")
    _write(repo / "docs/notes/leaderboard.md", "committed-doc\n")
    _write(repo / "logs/group/run_info/logs/log_0.out", "fixture\n")
    _write(
        repo / "scripts/analysis/build_leaderboard_doc.py",
        """from __future__ import annotations
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--archive", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--require-archive", action="store_true")
args = parser.parse_args()
if args.require_archive and not Path(args.archive).is_file():
    raise SystemExit(2)
root = Path(__file__).resolve().parents[2]
Path(args.output).write_text((root / "lora_playground/leaderboard.py").read_text())
""",
    )
    _commit_all(repo)
    return repo


def test_explicit_stage_command_generates_and_stages_output(tmp_path: Path):
    repo = _workflow_repo(tmp_path)
    source = repo / "lora_playground/leaderboard.py"
    source.write_text("updated-source\n")

    _run(repo, "bash", "scripts/analysis/update_leaderboard.sh", "--stage")

    doc = repo / "docs/notes/leaderboard.md"
    assert doc.read_text() == "updated-source\n"
    assert _run(repo, "git", "show", ":docs/notes/leaderboard.md").stdout == (
        "updated-source\n"
    )
    assert source.read_text() == "updated-source\n"


def test_explicit_stage_refuses_to_overwrite_conflicting_unstaged_doc(tmp_path: Path):
    repo = _workflow_repo(tmp_path)
    source = repo / "lora_playground/leaderboard.py"
    source.write_text("staged-source\n")
    _run(repo, "git", "add", "lora_playground/leaderboard.py")
    doc = repo / "docs/notes/leaderboard.md"
    doc.write_text("user-uncommitted-doc\n")

    result = _run(
        repo,
        "bash",
        "scripts/analysis/update_leaderboard.sh",
        "--stage",
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to overwrite unstaged" in result.stderr
    assert "update_leaderboard.sh --stage" in result.stderr
    assert doc.read_text() == "user-uncommitted-doc\n"
    assert _run(repo, "git", "show", ":docs/notes/leaderboard.md").stdout == (
        "committed-doc\n"
    )
