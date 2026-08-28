"""Every arm dict must declare every value its swept axis actually takes on disk.

The recurring defect in this repo is not a wrong number — it is a hand-typed set
of values that stops covering the runs. An arm whose predicate matches nothing
renders as an empty series, which in a figure is indistinguishable from an arm
with no data yet. Three instances, all real:

  - `arms.ADAMW` pinned `cw_nesterov=True`, a flag `LoRAPlusAdamW` never reads,
    while every adamw run at 5 of the 13 panel cells logs False. Those cells
    rendered with NO baseline and `leaderboard_rows` returned a NaN speed target.
  - `arms.ADAMW_BETA2_ARMS` could not inherit `ADAMW`'s pins, so — per the
    comment at `arms.py:380-389` — "all five non-0.999 arms rendered empty and
    `adamw_beta2_panel(256)` showed 1 of 6, with the grid runs present on disk
    the whole time."
  - `adam-polar-product-lora-coupled-spectral-chord-tight-outer` stayed in
    `OPTIM_COLORS` and `OPTIM_FAMILIES["headline_polar"]` after `89f9ccd`
    retired it from `OPTIMIZER_CHOICES`.

Each was found by hand, after the fact, by someone who happened to look. This
test is the general form of that look, applied to every arm dict in
`arms.ALL_ARM_DICTS` at once.

The rule: for an arm dict whose arms differ on exactly ONE field (a value grid —
`PROTO_BETA2_ARMS` over `curvature_beta`, `ADAMW_BETA2_ARMS` over `beta2`), take
the predicate the arms SHARE, find the runs on disk matching it, and require that
every value of the varied field among those runs has an arm. A new sweep point
then fails here with its value named, instead of quietly missing from a panel.

Arm dicts that vary more than one field (`PRECOND_ARMS`, `ABLATION_ARMS`, …) are
not value grids and are skipped — see `test_multi_axis_dicts_are_skipped_loudly`,
which reports what was skipped so the exemption cannot grow silently.
"""
from __future__ import annotations

import collections

import pytest

from lora_playground.loader import load_runs
from lora_playground.plotting import arms as A
from lora_playground.plotting.arms import pred_matches

MIN_STEPS = 8000


@pytest.fixture(scope="module")
def disk_runs():
    runs = load_runs(
        where={"max_steps": lambda s: isinstance(s, int) and s >= MIN_STEPS},
        warn_cross_commit=False, quiet=True)
    if not runs:
        pytest.skip("no long-horizon runs in logs/")
    return [cfg for cfg, _hist in runs]


def _split(arm_dict):
    """``(shared_predicate, [varied field names])`` for one arm dict.

    A field is SHARED when every arm carries it at the same value; anything else
    is an axis the dict varies. Compared by `repr` because predicate values
    include tuples and callables, which are not reliably ``==``-comparable.
    """
    preds = list(arm_dict.values())
    keys = set().union(*(set(p) for p in preds))
    shared = {k: preds[0][k] for k in keys
              if all(k in p and repr(p[k]) == repr(preds[0][k]) for p in preds)}
    return shared, sorted(keys - set(shared))


def value_grids():
    """``{name: (arm_dict, field)}`` for every arm dict that varies one field."""
    out = {}
    for name, d in A.ALL_ARM_DICTS.items():
        if len(d) < 2:
            continue
        _shared, varied = _split(d)
        if len(varied) == 1:
            out[name] = (d, varied[0])
    return out


@pytest.mark.parametrize("name", sorted(value_grids()))
def test_every_on_disk_value_of_the_swept_axis_has_an_arm(name, disk_runs):
    """The failure this exists for: a sweep adds a grid point and the panel
    silently omits it. The message names the value, so the fix is one entry."""
    arm_dict, field = value_grids()[name]
    shared, _ = _split(arm_dict)
    declared = {repr(p.get(field)) for p in arm_dict.values()}
    matched = [c for c in disk_runs if pred_matches(c, shared)]
    if not matched:
        pytest.skip(f"{name}: no run on disk matches the shared predicate")
    on_disk = collections.Counter(repr(c.get(field)) for c in matched)
    undeclared = {v: n for v, n in on_disk.items() if v not in declared}
    assert not undeclared, (
        f"{name} varies {field!r} and declares {sorted(declared)}, but runs "
        f"matching its shared predicate also carry:\n"
        + "\n".join(f"  {field}={v}  ({n} run(s))"
                    for v, n in sorted(undeclared.items(), key=lambda kv: -kv[1]))
        + f"\nThose runs are absent from every panel built on {name}. Fix: add an "
        f"arm for each value, or -- if the value is deliberately excluded -- "
        f"tighten the shared predicate so the runs do not match it either."
    )


@pytest.mark.parametrize("name", sorted(value_grids()))
def test_every_declared_arm_matches_at_least_one_run(name, disk_runs):
    """The other direction, and the one that actually shipped: a declared arm
    with no runs renders an empty series, which reads as a measured absence.
    `adamw_beta2_panel(256)` showed 1 of 6 arms this way."""
    arm_dict, field = value_grids()[name]
    empty = [label for label, pred in arm_dict.items()
             if not any(pred_matches(c, pred) for c in disk_runs)]
    assert not empty, (
        f"{name}: {len(empty)} arm(s) match no run on disk, so they render as "
        f"empty series rather than as missing data:\n"
        + "\n".join(f"  {label}" for label in empty)
        + f"\nFix: drop the arm, or correct the pin that excludes its runs "
        f"(a field the optimizer never reads is the usual culprit -- see "
        f"arms.ADAMW's cw_nesterov)."
    )


def test_multi_axis_dicts_are_skipped_loudly():
    """The skip list must stay visible, or the exemption grows silently.

    An arm dict varying several fields at once is not a value grid, so the
    coverage rule above does not apply to it. Naming them here means adding a
    new one is a visible diff rather than an invisible exemption.
    """
    grids = set(value_grids())
    multi = {}
    for name, d in A.ALL_ARM_DICTS.items():
        if len(d) < 2 or name in grids:
            continue
        _shared, varied = _split(d)
        multi[name] = varied
    # Not an assertion about WHICH dicts are multi-axis -- that changes as panels
    # are added -- but every one must vary at least two fields, or it is a grid
    # the parametrized tests should have picked up.
    for name, varied in multi.items():
        assert len(varied) >= 2, (
            f"{name} varies {varied}, which is a value grid the coverage test "
            f"above should cover but did not -- check `_split`."
        )
    print(f"skipped {len(multi)} multi-axis arm dict(s): "
          + ", ".join(f"{n}({len(v)} axes)" for n, v in sorted(multi.items())))


# ── the near-miss rule: the exact shape of the bugs that shipped ────────────

# Fields that runs are SWEPT over on purpose, so a run differing from an arm on
# one of these is a separate experiment rather than a run the arm should have
# claimed. Every entry needs a reason: this list is the only thing standing
# between the near-miss rule and a silent exemption.
# Fields that IDENTIFY which experiment a run is, rather than pinning how it was
# configured. A run differing here is a different experiment by definition, so it
# is not a near-miss at all -- treating `optimizer` as a near-miss field reported
# every chord-tight run as "one field from the AdamW arm", which is true and
# meaningless. Workload identity (model_name / lora_r / data_dir) is supplied per
# panel by `common_where`, not by the arm, for the same reason.
IDENTITY_FIELDS = frozenset({
    "optimizer", "model_name", "lora_r", "lora_alpha", "data_dir",
    "dataset_name", "max_steps", "seed", "lr",
})

SWEPT_SEPARATELY = {
    "cw_metric_init":   "the diagonal-metric init ablation (9 runs at >=8000)",
    "rdinv_variant":    "the damping-gauge ablation A/B/VN",
    "rdinv_delta":      "the decoupled diagonal floor, swept with rdinv_variant",
    "cw_solved_rho":    "the solved-magnitude arm",
    "cw_factor_a":      "per-factor shape scaling (a, b) grid",
    "cw_factor_b":      "per-factor shape scaling (a, b) grid",
    "curvature_beta":   "swept by PROTO_BETA2_ARMS / PRECOND_BETA2_ARMS",
    "beta2":            "swept by ADAMW_BETA2_ARMS",
    "precond_delta":    "the damping sweep",
    "precond_method":   "eigh / gram_ns / higham, an implementation axis",
    "higham_iters":     "moves with precond_method",
    "global_batch_size": "the small-batch arm",
    "lora_init_b":      "zero vs symmetric, tied to the magnitude ablations",
    "msign":            "swept by MSIGN_ARMS",
    "precond":          "swept by PRECOND_ARMS",
    "freeze_factorwise_slots":
        "the frozen-slot ablation is its own arm (arms.NOPRODUCT_FROZEN); a "
        "frozen run is a different experiment from the live one it forked",
}


def test_no_run_is_one_unswept_field_away_from_an_arm(disk_runs):
    """A run excluded from an arm by ONE field it was never meant to differ on.

    This is the shape of both bugs that actually shipped. `arms.ADAMW` pinned
    `cw_nesterov=True` — one field, which `LoRAPlusAdamW` does not even read —
    and every adamw run at 5 of 13 cells logs False, so those cells rendered
    with no baseline. `ADAMW_BETA2_ARMS` was the same defect at a different pin.

    Neither was detectable from the figure: an arm matching nothing draws
    nothing, exactly like an arm awaiting data. The rule here is that a
    one-field near-miss must be on a field the project sweeps ON PURPOSE
    (`SWEPT_SEPARATELY`, each with a reason) — otherwise the pin is excluding
    runs it was not meant to.
    """
    all_preds = [(f"{dn}/{ln}", p)
                 for dn, d in A.ALL_ARM_DICTS.items() for ln, p in d.items()]
    offenders = collections.Counter()
    for cfg in disk_runs:
        if any(pred_matches(cfg, p) for _n, p in all_preds):
            continue                      # claimed; nothing to say
        for name, pred in all_preds:
            diffs = [k for k, want in pred.items()
                     if not pred_matches(cfg, {k: want})]
            if (len(diffs) == 1 and diffs[0] not in SWEPT_SEPARATELY
                    and diffs[0] not in IDENTITY_FIELDS):
                offenders[(name, diffs[0], repr(cfg.get(diffs[0])))] += 1
    assert not offenders, (
        f"{sum(offenders.values())} run(s) are excluded from an arm by a single "
        f"field that is not in SWEPT_SEPARATELY, so the pin is probably wrong "
        f"rather than the run being a different experiment:\n"
        + "\n".join(f"  {n} run(s): arm {arm!r} excludes them on {field}={val}"
                    for (arm, field, val), n in offenders.most_common(15))
        + "\nFix: admit the value in the arm (arms.ADAMW's "
        "`cw_nesterov=(False, True)` is the worked example), or add the field to "
        "SWEPT_SEPARATELY with the reason it is a separate experiment."
    )
