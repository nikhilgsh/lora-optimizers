from __future__ import annotations

import itertools
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/sweep/sweep_factorwise_slot_freeze.sh"
LR_TASKS = {
    "1e-3": "task_00",
    "3e-3": "task_03",
    "1e-2": "task_06",
    "1.7e-2": "task_09",
    "3e-2": "task_12",
}


def _cells(params_path: Path):
    params = json.loads(params_path.read_text())
    return [
        dict(zip(params, values))
        for values in itertools.product(*(params[key] for key in params))
    ]


def _write_source_checkpoint(path, cell, identity):
    path.mkdir(parents=True)
    command = " ".join([
        "train_lora.py",
        "--model_name", cell["model"],
        "--data_dir", cell["data_dir"],
        "--data_pipeline_version", "packed_v1.1",
        "--max_steps", "9000",
        "--lr", cell["lr"],
        "--optimizer", cell["optimizer"],
        "--lora_r", cell["lora_r"],
        "--beta1", cell["beta1"],
        "--precond_delta", cell["precond_delta"],
        "--precond", cell["precond"],
        "--msign", cell["msign"],
    ])
    (path / "meta.json").write_text(json.dumps({
        "step": 2000,
        "attempt_id": "dynamic-attempt",
        "checkpoint_identity": identity,
        "cfg_snapshot": {"command": command},
    }))


@pytest.mark.parametrize("rank", [16, 256])
def test_freeze_params_are_exactly_five_factorwise_forks(rank, tmp_path):
    params_path = ROOT / (
        "params/e2_precond_qwen25_openmath_"
        f"r{rank}_factorwise_frozen_step2000.json"
    )
    cells = _cells(params_path)

    assert len(cells) == 5
    assert {cell["lr"] for cell in cells} == set(LR_TASKS)
    assert {cell["lora_r"] for cell in cells} == {str(rank)}
    assert {cell["precond"] for cell in cells} == {"factorwise"}
    assert {cell["msign"] for cell in cells} == {"full"}
    assert {cell["freeze_factorwise_slots"] for cell in cells} == {"1"}

    source_root = tmp_path / f"dynamic-r{rank}"
    source_prefix = f"dynamic-r{rank}"
    for index, cell in enumerate(cells):
        source_task = LR_TASKS[cell["lr"]]
        source = source_root / source_task / "ckpt_step2000"
        _write_source_checkpoint(
            source,
            cell,
            f"{source_prefix}/{source_task}",
        )
        destination = tmp_path / f"frozen-r{rank}" / f"task_{index:02d}"
        args = [
            cell[key] for key in (
                "lr", "optimizer", "seed", "precond_delta", "beta1",
                "model", "data_dir", "lora_r", "precond", "msign",
            )
        ] + [
            str(source_root),
            source_prefix,
            cell["freeze_factorwise_slots"],
        ]
        env = {
            **os.environ,
            "CHECKPOINT_DIR": str(destination),
            "LORA_CHECKPOINT_IDENTITY": f"frozen-r{rank}/task_{index:02d}",
            "LORA_ATTEMPT_ID": f"frozen-r{rank}:task_{index:02d}:test",
            "DRY_RUN": "1",
            "COMPILE": "0",
        }
        result = subprocess.run(
            ["bash", str(WRAPPER), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        command = shlex.split(result.stdout)

        assert command[command.index("--resume_from") + 1] == str(source)
        assert command[command.index("--checkpoint_dir") + 1] == str(destination)
        assert command[command.index("--max_steps") + 1] == "9000"
        assert command[command.index("--lora_r") + 1] == str(rank)
        assert "--freeze_factorwise_slots" in command
        assert "--resume_debug_replay" in command
        assert "--keep_checkpoints" in command
