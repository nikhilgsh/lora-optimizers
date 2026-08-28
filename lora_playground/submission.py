"""Small producer-side helpers shared by sweep submission paths."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path


_TASK_LOG_RE = re.compile(r"(?:^|[/\\])log_(\d+)\.out(?:\s|$)")
_CHECKPOINT_DIR_RE = re.compile(r"ckpt_step(\d+)")


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


def _latest_checkpoint(root: Path) -> Path | None:
    if (root / "meta.json").is_file():
        return root
    if not root.is_dir():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir()
        and _CHECKPOINT_DIR_RE.fullmatch(path.name)
        and (path / "meta.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(
        _CHECKPOINT_DIR_RE.fullmatch(path.name).group(1)
    ))


def _command_options(command: str) -> tuple[dict[str, str], set[str]]:
    tokens = shlex.split(command)
    values: dict[str, str] = {}
    flags: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                values[token] = tokens[i + 1]
                i += 1
            else:
                flags.add(token)
        i += 1
    return values, flags


def _validate_freeze_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int | None,
    expected_identity: str,
    expected_options: dict[str, str],
    frozen: bool,
) -> int:
    meta_path = checkpoint / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"checkpoint has no meta.json: {checkpoint}")
    meta = json.loads(meta_path.read_text())
    step = int(meta["step"])
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"checkpoint step mismatch: expected {expected_step}, got {step} "
            f"at {checkpoint}"
        )
    if meta.get("checkpoint_identity") != expected_identity:
        raise ValueError(
            "checkpoint identity mismatch: expected "
            f"{expected_identity!r}, got {meta.get('checkpoint_identity')!r}"
        )
    if not meta.get("attempt_id"):
        raise ValueError(f"checkpoint has no explicit attempt_id: {checkpoint}")

    command = meta.get("cfg_snapshot", {}).get("command", "")
    values, flags = _command_options(command)
    for key, expected in expected_options.items():
        flag = f"--{key}"
        actual = values.get(flag)
        if actual != str(expected):
            raise ValueError(
                f"checkpoint option mismatch for {flag}: expected "
                f"{str(expected)!r}, got {actual!r} at {checkpoint}"
            )
    has_freeze = "--freeze_factorwise_slots" in flags
    if has_freeze != frozen:
        state = "frozen" if frozen else "dynamic"
        raise ValueError(
            f"expected a {state} checkpoint, but command was {command!r}"
        )
    return step


def resolve_factorwise_freeze_resume(
    *,
    base_checkpoint: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    source_identity: str,
    destination_identity: str,
    expected_options: dict[str, str],
    final_step: int,
) -> Path:
    """Resolve a frozen continuation's source without conflating destinations.

    The first attempt must fork from the exact dynamic step-2000 checkpoint.
    A later attempt resumes the latest checkpoint in its own frozen destination.
    """
    base = Path(base_checkpoint).resolve()
    destination = Path(destination_root).resolve()
    if base == destination or destination in base.parents:
        raise ValueError("base checkpoint and frozen destination must be separate")

    latest = _latest_checkpoint(destination)
    if latest is not None:
        step = _validate_freeze_checkpoint(
            latest,
            expected_step=None,
            expected_identity=destination_identity,
            expected_options=expected_options,
            frozen=True,
        )
        if step <= 2000:
            raise ValueError(
                f"frozen destination checkpoint must be after step 2000, got {step}"
            )
        if step >= final_step:
            raise ValueError(
                f"frozen continuation is already complete at step {step}"
            )
        return latest

    if base.name != "ckpt_step2000":
        raise ValueError(
            f"base checkpoint must be the exact ckpt_step2000, got {base}"
        )
    _validate_freeze_checkpoint(
        base,
        expected_step=2000,
        expected_identity=source_identity,
        expected_options=expected_options,
        frozen=False,
    )
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inject = subparsers.add_parser("inject-task-metadata")
    inject.add_argument("--tasks", required=True)
    inject.add_argument("--checkpoint-root", required=True)
    inject.add_argument("--group", required=True)
    resolve = subparsers.add_parser("resolve-factorwise-freeze-resume")
    resolve.add_argument("--base-checkpoint", required=True)
    resolve.add_argument("--destination-root", required=True)
    resolve.add_argument("--source-identity", required=True)
    resolve.add_argument("--destination-identity", required=True)
    resolve.add_argument("--lr", required=True)
    resolve.add_argument("--optimizer", required=True)
    resolve.add_argument("--model-name", required=True)
    resolve.add_argument("--data-dir", required=True)
    resolve.add_argument("--lora-r", required=True)
    resolve.add_argument("--precond-delta", required=True)
    resolve.add_argument("--beta1", required=True)
    resolve.add_argument("--data-pipeline-version", required=True)
    resolve.add_argument("--final-step", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "inject-task-metadata":
        inject_task_attempt_metadata(
            args.tasks,
            checkpoint_root=args.checkpoint_root,
            group=args.group,
        )
    elif args.command == "resolve-factorwise-freeze-resume":
        expected_options = {
            "lr": args.lr,
            "optimizer": args.optimizer,
            "model_name": args.model_name,
            "data_dir": args.data_dir,
            "lora_r": args.lora_r,
            "precond_delta": args.precond_delta,
            "beta1": args.beta1,
            "data_pipeline_version": args.data_pipeline_version,
            "max_steps": str(args.final_step),
            "precond": "factorwise",
            "msign": "full",
        }
        print(resolve_factorwise_freeze_resume(
            base_checkpoint=args.base_checkpoint,
            destination_root=args.destination_root,
            source_identity=args.source_identity,
            destination_identity=args.destination_identity,
            expected_options=expected_options,
            final_step=args.final_step,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
