"""`arm_from_runs` derives a working arm from the runs, so a sweep needs no upkeep.

Why this exists
---------------
`arms.arm()` fails closed field by field, but nothing made the READ path track
the WRITE path. `scripts/sweep/*.sh` plus its params say what ran; `arms.py`
separately re-declares what to look for; nothing links them. When they drift the
symptom is MISSING DATA, not an error, because an arm claiming nothing renders
as an empty series.

Measured on this repo: 361 of 453 manifested groups in `logs/` are claimed by no
arm in `ALL_ARM_DICTS` -- 80% of the experiments on disk invisible to every
figure. Three instances were found by hand this week, each after the figure had
already been read (`NOPRODUCT` missing the newer optimizer name, `ADAMW` pinning
a flag `LoRAPlusAdamW` never reads, `PROTO` pinning `curvature_beta=0.99` so a
0.999 sweep matched nothing).

`arm_from_runs` removes the declaration rather than documenting it. These tests
pin the two properties that make that safe: the derived predicate must claim its
own runs, and it must still DISCRIMINATE -- deriving `{}` would trivially claim
everything and "fix" the coverage number while destroying every panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_membership_pin_still_works_for_scalar_cfg_values():
    """`ADAMW` pins `precond_method=(None, "higham")` — a set of allowed values
    against a scalar cfg field. That must keep meaning membership."""
    from lora_playground.plotting.arms import pred_matches
    assert pred_matches({"precond_method": "higham"},
                        {"precond_method": (None, "higham")})
    assert not pred_matches({"precond_method": "eigh"},
                            {"precond_method": (None, "higham")})


def test_a_list_valued_field_is_pinned_by_equality_not_membership():
    """The regression this fixes: a cfg field that genuinely HOLDS a list.

    `target_module_names=[]` was read as "match nothing", because any list-like
    pin meant membership. A derived predicate carrying it therefore matched 0 of
    its own 4 runs.
    """
    from lora_playground.plotting.arms import pred_matches
    assert pred_matches({"target_module_names": []}, {"target_module_names": []})
    assert pred_matches({"target_module_names": ["q_proj", "v_proj"]},
                        {"target_module_names": ["q_proj", "v_proj"]})
    assert not pred_matches({"target_module_names": ["q_proj"]},
                            {"target_module_names": ["q_proj", "v_proj"]})


def test_derives_the_constant_fields_and_drops_the_series_axes():
    """Synthetic, so the contract is pinned without needing logs/."""
    from lora_playground.manifest import SERIES_AXIS_FIELDS
    from lora_playground.plotting.arms import arm_from_runs, pred_matches
    cfgs = [
        {"optimizer": "kl-diag-polar-lora", "precond": "factorwise",
         "lr": 0.01, "seed": 0, "curvature_beta": 0.99},
        {"optimizer": "kl-diag-polar-lora", "precond": "factorwise",
         "lr": 0.03, "seed": 1, "curvature_beta": 0.99},
    ]
    pred = arm_from_runs(cfgs)
    assert pred["optimizer"] == "kl-diag-polar-lora"
    assert pred["precond"] == "factorwise"
    assert pred["curvature_beta"] == 0.99
    # lr and seed VARY and are per-series axes: pinning either would split a
    # series that `leaderboard.mean_over_seeds` is supposed to average.
    assert "lr" not in pred and "seed" not in pred
    assert SERIES_AXIS_FIELDS & set(pred) == set()
    assert all(pred_matches(c, pred) for c in cfgs)


def test_a_differing_field_is_not_pinned():
    from lora_playground.plotting.arms import arm_from_runs
    pred = arm_from_runs([{"optimizer": "a", "precond": "factorwise"},
                          {"optimizer": "a", "precond": "one-sided"}])
    assert pred == {"optimizer": "a"}, "a field the runs disagree on cannot be pinned"


def test_empty_input_raises():
    from lora_playground.plotting.arms import arm_from_runs
    with pytest.raises(ValueError):
        arm_from_runs([])


@pytest.mark.skipif(not (ROOT / "logs").is_dir(), reason="no logs/ tree")
def test_every_group_on_disk_derives_a_predicate_that_claims_its_own_runs():
    """The corpus-wide claim, checked rather than asserted: 409/409 groups."""
    import collections
    import warnings
    from lora_playground.loader import load_runs
    from lora_playground.plotting.arms import arm_from_runs, pred_matches
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [(c, h) for c, h in load_runs(warn_cross_commit=False, quiet=True) if h]
    by = collections.defaultdict(list)
    for cfg, _h in runs:
        if cfg.get("log_group"):
            by[cfg["log_group"]].append(cfg)
    multi = {g: c for g, c in by.items() if len(c) > 1}
    if not multi:
        pytest.skip("no multi-run groups on disk")
    bad = []
    for group, cfgs in multi.items():
        pred = arm_from_runs(cfgs)
        missed = [c for c in cfgs if not pred_matches(c, pred)]
        if missed:
            bad.append(f"  {group}: {len(missed)}/{len(cfgs)} of its own runs unmatched")
    assert not bad, (
        f"{len(bad)} of {len(multi)} groups derive a predicate that does not claim "
        f"their own runs — the derivation is unsound for those shapes:\n"
        + "\n".join(bad[:10]))


@pytest.mark.skipif(not (ROOT / "logs").is_dir(), reason="no logs/ tree")
def test_the_derived_predicate_still_discriminates():
    """Guard against the degenerate fix.

    Deriving `{}` would claim every run on disk and drive the coverage number to
    100% while destroying every panel — the arms would all collapse into one
    label. So a derived arm must reject runs that differ on a real knob.
    """
    import collections
    import warnings
    from lora_playground.loader import load_runs
    from lora_playground.plotting.arms import arm_from_runs, pred_matches
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [(c, h) for c, h in load_runs(warn_cross_commit=False, quiet=True) if h]
    by = collections.defaultdict(list)
    for cfg, _h in runs:
        if cfg.get("log_group"):
            by[cfg["log_group"]].append(cfg)
    g = "e2_precond_r16_postfix_xl"
    if g not in by:
        pytest.skip(f"{g} not on disk")
    pred = arm_from_runs(by[g])
    assert len(pred) > 20, f"derived only {len(pred)} pins — suspiciously permissive"
    # Its own runs mix precond=factorwise and one-sided, so `precond` VARIES and
    # is correctly not pinned. `curvature_beta` does not vary and must be.
    assert "curvature_beta" in pred
    others = [c for c, _h in runs if c.get("log_group") != g]
    assert any(not pred_matches(c, pred) for c in others), (
        "the derived predicate matches every run outside its group — it pins "
        "nothing discriminating")
