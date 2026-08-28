"""Entry point. Process-start source snapshot runs BEFORE any lora_playground
import so the snapshot is race-free with respect to mid-import edits.
See docs/execution_scope.md / ~/.claude/plans/the-data-loading-in-wobbly-pelican.md.
"""
# IMPORTANT: keep this module's imports to stdlib only. Anything that imports
# from `lora_playground` here would fire the package's __init__ BEFORE we
# snapshot — defeating the race protection. The bootstrap is the very first
# code that runs in the process; only after it completes do we import main.
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories to skip when snapshotting project source. The closure walker
# (lora_playground.execution_scope.compute_closure) raises if any path it
# discovers via AST walk is missing from the snapshot — so the right shape of
# this list is "everything that training code is GUARANTEED never to import."
# Tests/scripts/notebooks/docs/data dirs are off-limits to training code by
# project convention.
_SNAPSHOT_EXCLUDED_DIRS = frozenset({
    "tests", "scripts", "notebooks", "docs",
    "logs", "slurm_pending", "slurm_submitted",
    "slurm_cancel_pending", "slurm_cancel_submitted",
    "disbatch_logs", "slurm_logs", "backfill_records",
    "data", "params", "paper",
    "__pycache__", ".git",
})

SOURCE_SNAPSHOT: dict[str, bytes] = {}
SOURCE_SNAPSHOT_SHA: dict[str, str] = {}


def _iter_snapshot_python_paths(root: Path):
    """Yield snapshot-eligible Python paths, tolerating vanished subtrees.

    ``Path.rglob`` propagates an ``OSError`` raised while descending into a
    directory that disappears after its parent was scanned.  ``os.walk``
    treats that race as an unreadable subtree and continues with its siblings;
    top-down pruning preserves the directory exclusions used by the snapshot.
    """
    def handle_walk_error(error: OSError) -> None:
        # Match pathlib's existing tolerance for inaccessible directories and
        # add only the transient-disappearance case. Other traversal failures
        # remain fatal rather than silently weakening the source snapshot.
        if not isinstance(error, (FileNotFoundError, PermissionError)):
            raise error

    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, onerror=handle_walk_error
    ):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in _SNAPSHOT_EXCLUDED_DIRS
        ]
        base = Path(dirpath)
        for filename in filenames:
            if filename.endswith(".py"):
                yield base / filename


def _bootstrap_source_snapshot() -> None:
    """Walk all .py under ROOT (minus excluded dirs) and stash bytes + sha256.
    Called immediately at module load time, before any lora_playground import.
    """
    for p in _iter_snapshot_python_paths(ROOT):
        rel = p.relative_to(ROOT)
        # Skip ourselves — train_lora.py is the entry, not project library code.
        # (We still record train_lora.py because compute_closure starts here.)
        try:
            content = p.read_bytes()
        except OSError:
            continue
        key = str(rel)
        SOURCE_SNAPSHOT[key] = content
        SOURCE_SNAPSHOT_SHA[key] = hashlib.sha256(content).hexdigest()


_bootstrap_source_snapshot()

# Now safe to import project code — the snapshot's bytes match whatever
# Python is about to read for each module.
from lora_playground.train import main  # noqa: E402


if __name__ == "__main__":
    main()
