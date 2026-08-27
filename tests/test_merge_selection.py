"""Legacy merge compatibility must not infer cross-run lineage."""

import json

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
