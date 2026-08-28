from __future__ import annotations

import json
import shlex

import pytest

from lora_playground.submission import (
    inject_task_attempt_metadata,
    resolve_factorwise_freeze_resume,
)


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


def _write_checkpoint_meta(
    path,
    *,
    step,
    identity,
    attempt="attempt-a",
    frozen=False,
):
    path.mkdir(parents=True)
    freeze = " --freeze_factorwise_slots" if frozen else ""
    command = (
        "train_lora.py --lr 1e-2 --optimizer kl-diag-polar-lora "
        "--model_name Qwen/Qwen2.5-1.5B --data_dir data/openmath "
        "--lora_r 16 --precond_delta 1e-4 --beta1 0.9 "
        "--data_pipeline_version packed_v1.1 --max_steps 9000 "
        f"--precond factorwise --msign full{freeze}"
    )
    (path / "meta.json").write_text(json.dumps({
        "step": step,
        "attempt_id": attempt,
        "checkpoint_identity": identity,
        "cfg_snapshot": {"command": command},
    }))


def _expected_freeze_options():
    return {
        "lr": "1e-2",
        "optimizer": "kl-diag-polar-lora",
        "model_name": "Qwen/Qwen2.5-1.5B",
        "data_dir": "data/openmath",
        "lora_r": "16",
        "precond_delta": "1e-4",
        "beta1": "0.9",
        "data_pipeline_version": "packed_v1.1",
        "max_steps": "9000",
        "precond": "factorwise",
        "msign": "full",
    }


def test_freeze_resume_starts_from_exact_dynamic_step_2000(tmp_path):
    base = tmp_path / "base" / "ckpt_step2000"
    destination = tmp_path / "frozen"
    _write_checkpoint_meta(base, step=2000, identity="base/task_6")

    resolved = resolve_factorwise_freeze_resume(
        base_checkpoint=base,
        destination_root=destination,
        source_identity="base/task_6",
        destination_identity="frozen/task_2",
        expected_options=_expected_freeze_options(),
        final_step=9000,
    )

    assert resolved == base.resolve()


def test_freeze_retry_uses_its_own_latest_checkpoint(tmp_path):
    base = tmp_path / "base" / "ckpt_step2000"
    destination = tmp_path / "frozen"
    _write_checkpoint_meta(base, step=2000, identity="base/task_6")
    _write_checkpoint_meta(
        destination / "ckpt_step3000",
        step=3000,
        identity="frozen/task_2",
        frozen=True,
    )
    _write_checkpoint_meta(
        destination / "ckpt_step4000",
        step=4000,
        identity="frozen/task_2",
        frozen=True,
    )

    resolved = resolve_factorwise_freeze_resume(
        base_checkpoint=base,
        destination_root=destination,
        source_identity="base/task_6",
        destination_identity="frozen/task_2",
        expected_options=_expected_freeze_options(),
        final_step=9000,
    )

    assert resolved == (destination / "ckpt_step4000").resolve()


@pytest.mark.parametrize(
    "step,identity,frozen,match",
    [
        (1999, "base/task_6", False, "step mismatch"),
        (2000, "wrong/task", False, "identity mismatch"),
        (2000, "base/task_6", True, "dynamic checkpoint"),
    ],
)
def test_freeze_resume_rejects_wrong_source(
    tmp_path, step, identity, frozen, match
):
    base = tmp_path / "base" / "ckpt_step2000"
    _write_checkpoint_meta(
        base,
        step=step,
        identity=identity,
        frozen=frozen,
    )

    with pytest.raises(ValueError, match=match):
        resolve_factorwise_freeze_resume(
            base_checkpoint=base,
            destination_root=tmp_path / "frozen",
            source_identity="base/task_6",
            destination_identity="frozen/task_2",
            expected_options=_expected_freeze_options(),
            final_step=9000,
        )
