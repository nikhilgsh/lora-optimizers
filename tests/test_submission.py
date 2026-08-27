from __future__ import annotations

import shlex

import pytest

from lora_playground.submission import inject_task_attempt_metadata


def test_inject_task_attempt_metadata_is_explicit_unique_and_stable(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.write_text(
        "sweep.sh a > /logs/log_0.out 2> /logs/log_0.err\n"
        "sweep.sh b > /logs/log_12.out 2> /logs/log_12.err\n"
    )
    tokens = iter(["token-a", "token-b"])

    emitted = inject_task_attempt_metadata(
        tasks,
        checkpoint_root=tmp_path / "checkpoints",
        group="group name",
        token_factory=lambda: next(tokens),
    )

    lines = tasks.read_text().splitlines()
    first = dict(
        token.split("=", 1)
        for token in shlex.split(lines[0])
        if "=" in token and not token.startswith("/")
    )
    second = dict(
        token.split("=", 1)
        for token in shlex.split(lines[1])
        if "=" in token and not token.startswith("/")
    )
    assert first["LORA_ATTEMPT_ID"] == "group name:task_0:token-a"
    assert second["LORA_ATTEMPT_ID"] == "group name:task_12:token-b"
    assert first["LORA_CHECKPOINT_IDENTITY"] == "group name/task_0"
    assert second["LORA_CHECKPOINT_IDENTITY"] == "group name/task_12"
    assert emitted[0]["CHECKPOINT_DIR"].endswith("checkpoints/task_0")
    assert emitted[1]["CHECKPOINT_DIR"].endswith("checkpoints/task_12")


def test_inject_task_attempt_metadata_fails_before_replacing_bad_input(tmp_path):
    tasks = tmp_path / "tasks"
    original = "sweep.sh without-a-log-redirect\n"
    tasks.write_text(original)

    with pytest.raises(ValueError, match="no log_NN.out"):
        inject_task_attempt_metadata(
            tasks,
            checkpoint_root=tmp_path / "checkpoints",
            group="group",
        )

    assert tasks.read_text() == original


def test_inject_task_attempt_metadata_rejects_checkpoint_collisions(tmp_path):
    tasks = tmp_path / "tasks"
    original = (
        "sweep.sh a > /logs/log_2.out\n"
        "sweep.sh b > /other/log_2.out\n"
    )
    tasks.write_text(original)

    with pytest.raises(ValueError, match="duplicate task number 2"):
        inject_task_attempt_metadata(
            tasks,
            checkpoint_root=tmp_path / "checkpoints",
            group="group",
        )

    assert tasks.read_text() == original
