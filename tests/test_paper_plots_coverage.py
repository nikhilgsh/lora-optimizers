"""`paper_plots_lib.coverage_report`: name the runs no arm predicate claimed.

Every arm predicate fails the same way — by matching FEWER runs, silently. In a
rendered panel an arm that pins a field the runs disagree on is indistinguishable
from an arm that legitimately has no data yet. This module's job is to make that
difference visible, so the tests here are mostly KNOWN-POSITIVE tests: they
reintroduce a bug that actually shipped and assert the report names it.

Runs are synthetic and `cell_runs` is monkeypatched, so these do not depend on
what happens to be in `logs/` — a real-tree fixture would change its verdict
every time a sweep lands.
"""
import lora_playground.plotting.paper_plots_lib as P


def _cfg(**over):
    """A config carrying just the fields these tests pin."""
    base = dict(optimizer="adamw", lr=3e-4, lora_r=16, max_steps=9000,
                cw_nesterov=False, precond=None, curvature_beta=0.99)
    base.update(over)
    return base


def _hist(step=9000, loss=0.42):
    return [{"step": step, "eval_loss": loss}]


def _patch_runs(monkeypatch, runs):
    monkeypatch.setattr(P, "cell_runs", lambda where, refresh=False: runs)


def test_clean_cell_reports_nothing(monkeypatch):
    """A cell whose every run is claimed prints nothing at all."""
    runs = [(_cfg(), _hist()), (_cfg(lr=1e-3), _hist())]
    _patch_runs(monkeypatch, runs)
    arms = {"AdamW": {"optimizer": "adamw"}}
    assert P.coverage_report(arms, {}) == ""


def test_names_the_pinned_field_that_excluded_the_run(monkeypatch):
    """The shipped arms.ADAMW bug: it pinned `cw_nesterov=True`, a flag
    LoRAPlusAdamW never reads, while every adamw run at 5 of the 13 CELLS logs
    False. Those cells rendered with no baseline and leaderboard_rows returned a
    NaN speed target. The report has to name the field, not just the count —
    "4 runs unclaimed" sends you looking at the data, "arm wants
    cw_nesterov=True, run has False" sends you to the one-line fix.
    """
    _patch_runs(monkeypatch, [(_cfg(), _hist())])
    arms = {"AdamW": {"optimizer": "adamw", "cw_nesterov": True}}
    out = P.coverage_report(arms, {})
    assert "UNCLAIMED: 1 of 1" in out
    assert "cw_nesterov" in out
    assert "arm wants True" in out and "run has False" in out
    assert "'AdamW'" in out, "must name which arm was closest"


def test_membership_predicate_is_honoured(monkeypatch):
    """A tuple value is a membership test, matching the loader's `where`
    semantics — the fix actually applied to arms.ADAMW was
    `cw_nesterov=(False, True)`. The report must agree with the loader about
    what counts as a match, or it will cry wolf on every admitted run.
    """
    _patch_runs(monkeypatch, [(_cfg(cw_nesterov=False), _hist()),
                              (_cfg(cw_nesterov=True), _hist())])
    arms = {"AdamW": {"optimizer": "adamw", "cw_nesterov": (False, True)}}
    assert P.coverage_report(arms, {}) == ""


def test_absent_field_is_reported_as_absent(monkeypatch):
    """A run missing a referenced field does not match (loader._matches
    semantics), and the reason shown must say the field was absent rather than
    printing a confusing value comparison."""
    cfg = _cfg()
    del cfg["curvature_beta"]
    _patch_runs(monkeypatch, [(cfg, _hist())])
    arms = {"AdamW": {"optimizer": "adamw", "curvature_beta": 0.99}}
    out = P.coverage_report(arms, {})
    assert "curvature_beta" in out and "<absent>" in out


def test_closest_arm_is_the_one_with_fewest_mismatches(monkeypatch):
    """With several arms declared, the diagnosis must point at the arm the run
    nearly matched, not whichever happens to be first."""
    _patch_runs(monkeypatch, [(_cfg(optimizer="kl-diag-polar-lora",
                                    precond="factorwise", lr=0.03), _hist())])
    arms = {
        "far": {"optimizer": "adamw", "lr": 3e-4, "lora_r": 64, "precond": None},
        "near": {"optimizer": "kl-diag-polar-lora", "precond": "factorwise",
                 "curvature_beta": 0.999},
    }
    out = P.coverage_report(arms, {})
    assert "'near'" in out and "'far'" not in out
    assert "curvature_beta" in out


def test_row_cap_is_disclosed_not_silent(monkeypatch):
    """Truncating the list without saying so would read as "only N unclaimed",
    which is the same silent-undercount failure this report exists to end."""
    n = P._COVERAGE_MAX_ROWS + 3
    _patch_runs(monkeypatch, [(_cfg(lr=1e-4 * (i + 1)), _hist()) for i in range(n)])
    arms = {"AdamW": {"optimizer": "adamw", "cw_nesterov": True}}
    out = P.coverage_report(arms, {})
    assert f"UNCLAIMED: {n} of {n}" in out
    assert f"and {n - P._COVERAGE_MAX_ROWS} more unclaimed runs" in out


def test_in_flight_run_shows_its_step(monkeypatch):
    """An unclaimed run that is merely still running should be recognisable as
    such from the report, so it is not chased as a predicate bug."""
    _patch_runs(monkeypatch, [(_cfg(max_steps=2000, curvature_beta=0.999),
                               _hist(step=750))])
    arms = {"AdamW": {"optimizer": "adamw", "curvature_beta": 0.99}}
    out = P.coverage_report(arms, {})
    assert "step 750" in out and "max_steps=2000" in out
