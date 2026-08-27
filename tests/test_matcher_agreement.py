"""`loader._matches` and `arms.field_matches` decide the same thing.

Why this exists
---------------
There are two paths from an arm dict to a set of runs, and they use different
matchers:

  - `figures.compare_variants_figure`'s per-variant path calls
    `load_runs(where={**common_where, **extra})` -> `_build_filter` ->
    `loader._matches`;
  - its prefetched path, and every `paper_plots_lib` panel, goes through
    `arms.variant_key_fn` -> `arms.pred_matches` -> `arms.field_matches`.

So a rule added to one and not the other means the SAME arm dict selects
different runs depending on which loading path the caller happened to take —
silently, as a figure with missing curves rather than an error.

That is not hypothetical. A list-vs-list equality branch was added to
`arms.pred_matches` (so a cfg field genuinely holding a list, like
`target_module_names=[]`, could be pinned at all) and not to `loader._matches`.
Measured on `logs/e2_precond_r16_postfix_xl`: a 132-pin predicate derived from
those runs matched **4 of its own 4** through `arms` and **0 of 4** through the
loader.

This test is what keeps the two definitions in step. If it fails, do not "fix"
one side to match today's data — decide what the pin MEANS and change both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (spec, value) pairs spanning every branch both matchers claim to implement.
CASES = [
    # literal equality
    ("higham", "higham", True),
    ("higham", "eigh", False),
    (0.99, 0.99, True),
    (0.99, 0.999, False),
    (None, None, True),
    (True, False, False),
    # membership: a SET OF ALLOWED VALUES against a scalar
    ((None, "higham"), "higham", True),
    ((None, "higham"), None, True),
    ((None, "higham"), "eigh", False),
    (["kl-diag-polar-lora", "kl-shampoo-polar-lora"], "kl-shampoo-polar-lora", True),
    (["kl-diag-polar-lora", "kl-shampoo-polar-lora"], "adamw", False),
    (frozenset({1, 2}), 2, True),
    (frozenset({1, 2}), 3, False),
    # equality: a LITERAL list against a list-valued field
    ([], [], True),
    (["q_proj", "v_proj"], ["q_proj", "v_proj"], True),
    (["q_proj", "v_proj"], ["q_proj"], False),
    ([], ["q_proj"], False),
    # callable
    ((lambda v: v is not None), "anything", True),
    ((lambda v: v is not None), None, False),
]


@pytest.mark.parametrize("spec,value,expected", CASES)
def test_both_matchers_agree_and_are_correct(spec, value, expected):
    from lora_playground.loader import _matches
    from lora_playground.plotting.arms import field_matches
    loader_says = _matches(spec, value)
    arms_says = field_matches({"f": value}, "f", spec)
    assert loader_says == arms_says, (
        f"matchers disagree on spec={spec!r} value={value!r}: "
        f"loader._matches={loader_says}, arms.field_matches={arms_says}. "
        f"The same arm dict now selects different runs on different loading "
        f"paths. Decide what the pin means and change BOTH.")
    assert loader_says == expected


def test_arms_reports_an_absent_field_as_no_match():
    """`arms.field_matches` takes the whole cfg, so it also owns the
    absent-field rule that `loader._matches` never sees (the loader's
    `_build_filter` checks presence before calling it)."""
    from lora_playground.plotting.arms import field_matches
    assert not field_matches({}, "missing", "anything")
    assert not field_matches({"other": 1}, "missing", None), (
        "an absent field must not match a None pin — otherwise every arm "
        "pinning a flag at its None default would claim every older run that "
        "predates the flag")


@pytest.mark.skipif(not (ROOT / "logs").is_dir(), reason="no logs/ tree")
def test_the_two_paths_agree_on_a_real_derived_predicate():
    """The regression, end to end on disk: the case that measured 4/4 vs 0/4."""
    import warnings
    from lora_playground.loader import _matches, load_runs
    from lora_playground.plotting.arms import arm_from_runs, pred_matches
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [(c, h) for c, h in load_runs(warn_cross_commit=False, quiet=True) if h]
    cfgs = [c for c, _ in runs if c.get("log_group") == "e2_precond_r16_postfix_xl"]
    if not cfgs:
        pytest.skip("e2_precond_r16_postfix_xl not on disk")
    pred = arm_from_runs(cfgs)
    via_arms = sum(pred_matches(c, pred) for c in cfgs)
    via_loader = sum(all(k in c and _matches(v, c[k]) for k, v in pred.items())
                     for c in cfgs)
    assert via_arms == via_loader == len(cfgs), (
        f"{via_arms} of {len(cfgs)} matched through arms and {via_loader} "
        f"through the loader — the two selection paths disagree")
