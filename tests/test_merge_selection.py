"""Legacy merge compatibility must not infer cross-run lineage."""

import json

from lora_playground.plotting import load_sweep
from lora_playground.plotting.merge import _select_run, merge_runs


def _evs(*steps):
    return [{"step": step, "eval_loss": 1.0 / step} for step in steps]


def _run(final_step, priority, evs, **cfg):
    cfg.setdefault("lr", 0.01)
    return final_step, priority, cfg, evs


def test_resume_metadata_does_not_stitch_physical_runs():
    original = _run(
        8000,
        0,
        _evs(250, 4000, 8000),
        log_group="original",
        _log_filename="log_0.out",
    )
    continuation = _run(
        9000,
        1,
        _evs(8250, 9000),
        log_group="continuation",
        _log_filename="log_0.out",
        resume_from="/logs/original/run_info/checkpoints/task_0",
    )

    cfg, evs = _select_run([continuation, original])

    assert cfg["log_group"] == "continuation"
    assert [event["step"] for event in evs] == [8250, 9000]
    assert "_legacy_source_physical_ids" not in cfg
    assert "_legacy_source_segments" not in cfg


def test_longest_physical_run_wins_an_overlap():
    complete = _run(9000, 1, _evs(250, 4000, 9000), log_group="complete")
    inflight = _run(3000, 0, _evs(250, 3000), log_group="inflight")

    cfg, evs = _select_run([inflight, complete])

    assert cfg["log_group"] == "complete"
    assert [event["step"] for event in evs] == [250, 4000, 9000]


def test_equal_horizons_use_group_priority():
    preferred = _run(9000, 0, _evs(250, 9000), log_group="preferred")
    fallback = _run(9000, 1, _evs(250, 9000), log_group="fallback")

    cfg, evs = _select_run([fallback, preferred])

    assert cfg["log_group"] == "preferred"
    assert [event["step"] for event in evs] == [250, 9000]


def test_merge_runs_does_not_stitch_a_recorded_cross_group_resume(tmp_path):
    def write_run(group, cfg, steps):
        log_dir = tmp_path / group / "run_info" / "logs"
        log_dir.mkdir(parents=True)
        rows = [
            {"event": "config", **cfg},
            *[
                {
                    "event": "eval",
                    "step": step,
                    "eval_loss": 1.0 / step,
                    "lr": cfg["lr"],
                }
                for step in steps
            ],
        ]
        (log_dir / "log_0.out").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    common = {
        "optimizer": "adamw",
        "lr": 1e-3,
        "command": "train_lora.py --optimizer adamw --lr 0.001",
    }
    write_run("original", common, [250, 4000, 8000])
    write_run(
        "continuation",
        {
            **common,
            "resume_from": "/logs/original/run_info/checkpoints/task_0",
        },
        [8250, 9000],
    )

    runs = merge_runs(
        ["original", "continuation"],
        key_fn=lambda cfg: (cfg["optimizer"], cfg["lr"]),
        logs_root=str(tmp_path),
    )

    assert len(runs) == 1
    cfg, evs = runs[0]
    assert cfg["log_group"] == "continuation"
    assert [event["step"] for event in evs] == [8250, 9000]


def _write_physical_segment(path, steps, source):
    rows = [
        {
            "event": "config",
            "optimizer": "adamw",
            "lr": 1e-3,
            "source": source,
        },
        *[
            {
                "event": "eval",
                "step": step,
                "eval_loss": 1.0 / step,
                "lr": 1e-3,
                "source": source,
            }
            for step in steps
        ],
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_sweep_uses_canonical_file_as_a_late_tiebreak(tmp_path):
    log_dir = tmp_path / "group" / "run_info" / "logs"
    log_dir.mkdir(parents=True)
    _write_physical_segment(
        log_dir / "log_0.out.resume_0", [250, 1000], "rotated"
    )
    _write_physical_segment(
        log_dir / "log_0.out", [500, 1000], "canonical"
    )

    runs = load_sweep("group", logs_root=str(tmp_path))

    assert len(runs) == 1
    cfg, events = runs[0]
    assert cfg["source"] == "canonical"
    assert [event["step"] for event in events] == [500, 1000]


def test_load_sweep_prefers_eval_count_before_canonical_filename(tmp_path):
    log_dir = tmp_path / "group" / "run_info" / "logs"
    log_dir.mkdir(parents=True)
    _write_physical_segment(
        log_dir / "log_0.out.resume_0", [250, 750, 1000], "longer"
    )
    _write_physical_segment(
        log_dir / "log_0.out", [500, 1000], "canonical"
    )

    runs = load_sweep("group", logs_root=str(tmp_path))

    assert len(runs) == 1
    cfg, events = runs[0]
    assert cfg["source"] == "longer"
    assert [event["step"] for event in events] == [250, 750, 1000]


def test_load_sweep_does_not_reconstruct_fields_from_command(tmp_path):
    log_dir = tmp_path / "group" / "run_info" / "logs"
    log_dir.mkdir(parents=True)
    rows = [
        {
            "event": "config",
            "optimizer": "adamw",
            "command": (
                "train_lora.py --optimizer adamw "
                "--lora_plus_multiplier 4 --precond_refresh_every 7"
            ),
        },
        {"event": "eval", "step": 100, "eval_loss": 0.8, "lr": 1e-3},
    ]
    (log_dir / "log_0.out").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    [(cfg, _events)] = load_sweep("group", logs_root=str(tmp_path))

    assert "lr" not in cfg
    assert "lora_plus_multiplier" not in cfg
    assert "precond_refresh_every" not in cfg
