"""The DataFrame convenience API stays on immutable recorded semantics."""
from __future__ import annotations

import json

import pytest

from lora_playground.loader_df import load_runs_df


def _write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_load_runs_df_uses_recorded_semantics_and_physical_group(tmp_path):
    logs = tmp_path / "logs"
    _write(logs / "actual-group" / "run_info" / "logs" / "log_0.out", [
        {
            "event": "config",
            "optimizer": "adamw",
            "optimizer_config": {"lr": 1e-3, "beta1": 0.9},
            "command": (
                "python train_lora.py --checkpoint_dir "
                "logs/wrong-command-group/run_info/checkpoints"
            ),
            "execution_source_sha": "audit-only",
        },
        {"event": "eval", "step": 10, "eval_loss": 0.9},
        {"event": "eval", "step": 20, "eval_loss": 0.8},
    ])
    _write(logs / "other-group" / "run_info" / "logs" / "log_1.out", [
        {"event": "config", "optimizer": "adamw", "lr": 3e-3},
        {"event": "eval", "step": 10, "eval_loss": 1.1},
    ])

    frame = load_runs_df(
        where={
            "group": "actual-group",
            "lr": lambda value: value < 2e-3,
        },
        logs_root=str(logs),
        resolve_lineages=False,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["group"] == "actual-group"
    assert row["log_filename"] == "log_0.out"
    assert row["physical_id"] == "actual-group/log_0.out"
    assert row["lr"] == pytest.approx(1e-3)
    assert row["beta1"] == pytest.approx(0.9)
    assert row["final_loss"] == pytest.approx(0.8)
    assert row["min_loss"] == pytest.approx(0.8)
    assert row["n_evals"] == 2
    assert row["max_step"] == 20
    assert "command" not in frame.columns
    assert "execution_source_sha" not in frame.columns


def test_load_runs_df_rejects_non_mapping_where(tmp_path):
    with pytest.raises(TypeError, match="where must"):
        load_runs_df(where=[("optimizer", "adamw")], logs_root=str(tmp_path))
