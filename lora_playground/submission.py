"""Small producer-side helpers shared by sweep submission paths."""
from __future__ import annotations

import argparse
import os
import re
import shlex
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path


_TASK_LOG_RE = re.compile(r"(?:^|[/\\])log_(\d+)\.out(?:\s|$)")


def inject_task_attempt_metadata(
    tasks_path: str | os.PathLike[str],
    *,
    checkpoint_root: str | os.PathLike[str],
    group: str,
    token_factory: Callable[[], str] | None = None,
) -> tuple[dict[str, str], ...]:
    """Atomically prefix generated task lines with explicit run identity.

    Every nonblank task must expose its ``log_NN.out`` redirect. A missing or
    duplicate task number is an input error because it would make checkpoint
    ownership ambiguous. The checkpoint identity is stable for ``group/task``;
    the attempt ID receives a fresh opaque token on every submission.
    """
    if not isinstance(group, str) or not group.strip():
        raise ValueError("group must be a non-empty string")
    token_factory = token_factory or (lambda: uuid.uuid4().hex)
    tasks = Path(tasks_path)
    checkpoint_dir = Path(checkpoint_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rewritten: list[str] = []
    metadata: list[dict[str, str]] = []
    seen: set[str] = set()

    for ordinal, line in enumerate(tasks.read_text().splitlines(), start=1):
        if not line.strip():
            rewritten.append(line)
            continue
        match = _TASK_LOG_RE.search(line)
        if match is None:
            raise ValueError(
                f"task line {ordinal} has no log_NN.out redirect: {line!r}"
            )
        task_number = match.group(1)
        if task_number in seen:
            raise ValueError(f"duplicate task number {task_number} in {tasks}")
        seen.add(task_number)
        token = token_factory()
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token_factory must return a non-empty string")

        task_checkpoint_dir = checkpoint_dir / f"task_{task_number}"
        attempt_id = f"{group}:task_{task_number}:{token}"
        checkpoint_identity = f"{group}/task_{task_number}"
        fields = {
            "CHECKPOINT_DIR": str(task_checkpoint_dir),
            "LORA_ATTEMPT_ID": attempt_id,
            "LORA_CHECKPOINT_IDENTITY": checkpoint_identity,
        }
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in fields.items()
        )
        rewritten.append(f"{prefix} {line}")
        metadata.append(fields)

    if not metadata:
        raise ValueError(f"task file has no runnable lines: {tasks}")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{tasks.name}.", suffix=".tmp", dir=str(tasks.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write("\n".join(rewritten) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, tasks)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return tuple(metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inject = subparsers.add_parser("inject-task-metadata")
    inject.add_argument("--tasks", required=True)
    inject.add_argument("--checkpoint-root", required=True)
    inject.add_argument("--group", required=True)
    args = parser.parse_args(argv)
    if args.command == "inject-task-metadata":
        inject_task_attempt_metadata(
            args.tasks,
            checkpoint_root=args.checkpoint_root,
            group=args.group,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
