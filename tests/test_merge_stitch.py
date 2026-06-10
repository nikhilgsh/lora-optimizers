"""Resume-segment stitching in merge_runs (_stitch_runs).

A `--resume_from` continuation logs only its post-resume evals. Without
stitching, merge_runs' "longest trajectory wins" picked the resume run and
DROPPED the pre-resume segment, so the trajectory panel showed only the last
few steps. _stitch_runs concatenates same-key segments into one monotonic
trajectory; for overlapping ranges it must reduce to the old longest-wins.
"""
from lora_playground.plotting.merge import _stitch_runs


def _evs(*steps):
    # eval_loss decreasing with step so it's a plausible trajectory
    return [{"step": s, "eval_loss": 1.0 / s} for s in steps]


def _run(final_step, idx, evs, **cfg):
    cfg.setdefault("lr", 0.01)
    return (final_step, idx, cfg, evs)


def test_resume_continuation_stitches_full_trajectory():
    original = _run(8000, 0, _evs(250, 4000, 8000), log_group="orig")
    resume = _run(9000, 1, _evs(8250, 9000), log_group="resume")
    cfg, evs = _stitch_runs([resume, original])  # order-independent
    steps = [e["step"] for e in evs]
    assert steps == [250, 4000, 8000, 8250, 9000]
    # Representative cfg = most-progressed run (the resume).
    assert cfg["log_group"] == "resume"


def test_overlapping_rerun_reduces_to_longest_wins():
    inflight = _run(3000, 1, _evs(250, 3000), log_group="new")
    complete = _run(9000, 0, _evs(250, 4000, 9000), log_group="old")
    cfg, evs = _stitch_runs([inflight, complete])
    steps = [e["step"] for e in evs]
    # No duplicate 250; shorter run contributes nothing new.
    assert steps == [250, 4000, 9000]
    assert cfg["log_group"] == "old"


def test_identical_reruns_tiebreak_by_group_priority():
    a = _run(9000, 0, _evs(250, 9000), log_group="prio0")
    b = _run(9000, 1, _evs(250, 9000), log_group="prio1")
    cfg, evs = _stitch_runs([b, a])
    assert [e["step"] for e in evs] == [250, 9000]
    # Tie on final_step → lower idx (higher priority) is representative.
    assert cfg["log_group"] == "prio0"


def test_single_run_unchanged():
    only = _run(9000, 0, _evs(250, 4000, 9000), log_group="solo")
    cfg, evs = _stitch_runs([only])
    assert [e["step"] for e in evs] == [250, 4000, 9000]
    assert cfg["log_group"] == "solo"


def test_chained_resume_three_segments():
    s0 = _run(3000, 0, _evs(250, 3000), log_group="s0")
    s1 = _run(6000, 1, _evs(3250, 6000), log_group="s1")
    s2 = _run(9000, 2, _evs(6250, 9000), log_group="s2")
    cfg, evs = _stitch_runs([s2, s0, s1])
    assert [e["step"] for e in evs] == [250, 3000, 3250, 6000, 6250, 9000]
    assert cfg["log_group"] == "s2"
