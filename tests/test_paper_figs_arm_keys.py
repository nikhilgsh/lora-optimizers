"""The `paper_figs.py` arm-key functions discriminate against the runs on disk.

Why this test exists
--------------------
`paper_figs.py` builds the PAPER's figures (fig1, table1, fig2, fig3,
figA_curves, fig_hero_with_tuning). Each one maps a run to a display label
through `paper_variant_key` / `ablation_variant_key` / `arm_key`, and
`leaderboard.labeled_completed_runs` raises `LabelCollisionError` when one label
covers two runs that differ outside `manifest.SERIES_AXIS_FIELDS`.

Nothing tested those three functions against `logs/`. They were hand-written
allowlists — `_is_proto` checked 8 fields and never mentioned `cw_solved_rho`,
`curvature_beta`, `beta2`, `cw_no_radius` or `precond_method` — so every sweep
that set one of those was silently mislabeled until the guard happened to fire.
Measured on Llama-3.2-1B/openmath/r256 before the fix: `paper_variant_key`
raised with 4 collisions, `ablation_variant_key` with 6, `arm_key` with 6, i.e.
the paper's figures could not render at all.

They are now built by `arms.variant_key_fn` over `arms.arm()` predicates, which
pin every `OptimizerConfig` field by default. This test is what keeps that true:
it is the `paper_figs` counterpart of
`tests/test_arm_predicates.py::test_arm_dict_discriminates`, which covers only
`arms.ALL_ARM_DICTS`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(
    not (ROOT / "logs").is_dir(), reason="no logs/ tree")


def _key_fns():
    from lora_playground.plotting.paper_figs import (
        ablation_variant_key, arm_key, paper_variant_key,
    )
    return [("paper_variant_key", paper_variant_key),
            ("ablation_variant_key", ablation_variant_key),
            ("arm_key", arm_key)]


@pytest.fixture(scope="module")
def hero_runs():
    """The cell every one of these functions is exercised on by fig1/fig2/fig3."""
    from lora_playground.workloads import find_workload, workload_runs
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)
    runs = workload_runs(wl)
    if not runs:
        pytest.skip("no runs for the hero workload")
    return wl, runs


@pytest.mark.parametrize("name", [n for n, _ in _key_fns()])
def test_arm_key_discriminates_on_disk(name, hero_runs):
    """No label may cover two runs that differ outside SERIES_AXIS_FIELDS.

    A failure names the differing cfg field: give that value its own arm in
    `paper_figs`' arm dict, or pin it on the existing arm. Do NOT widen the
    label — that is what silently merged two sweeps before.
    """
    from lora_playground.leaderboard import labeled_completed_runs
    wl, runs = hero_runs
    fn = dict(_key_fns())[name]
    labeled_completed_runs(runs, fn, horizon=wl.horizon)


@pytest.mark.parametrize("name", [n for n, _ in _key_fns()])
def test_arm_key_is_not_dead(name, hero_runs):
    """An over-pinned key labels nothing and would "discriminate" perfectly, so
    the emptiness has to be its own failure."""
    wl, runs = hero_runs
    fn = dict(_key_fns())[name]
    labels = {fn(cfg) for cfg, _ in runs}
    labels.discard(None)
    assert labels, (
        f"{name} labels no run on the hero workload: every predicate is dead. "
        f"Most likely a field was pinned to a value no run carries."
    )


def test_key_fns_are_built_from_arms_predicates():
    """The mechanism, not just today's output: these must be `variant_key_fn`
    closures over `arm()`-built dicts, so a new OptimizerConfig field is pinned
    automatically. A hand-written replacement would pass the tests above on
    today's data and rot the next time a flag is added."""
    from lora_playground.plotting import arms as A
    from lora_playground.plotting import paper_figs as F
    for dict_name in ("_PAPER_ARMS", "_ABLATION_ARMS", "_ARM_KEY_ARMS"):
        arms = getattr(F, dict_name)
        assert arms, f"{dict_name} is empty"
        for label, pred in arms.items():
            # An arm pins the pinnable fields its OPTIMIZER can read, not all of
            # them. This loop asserted `PINNED_FIELDS() - set(pred) == set()`
            # until `arm()` started deriving the set: pinning a field the arm's
            # own optimizer cannot receive is what silently dropped runs (5 of 13
            # panel cells lost their AdamW baseline to a pinned `cw_nesterov`
            # that `LoRAPlusAdamW` never reads). The mechanism claim survives —
            # a NEW OptimizerConfig field is still pinned automatically in every
            # arm whose optimizer receives it, which is where it can
            # discriminate.
            opt = pred.get("optimizer")
            assert opt, f"{dict_name}[{label!r}] does not pin `optimizer`"
            names = opt if isinstance(opt, (tuple, list, set, frozenset)) else [opt]
            expected = set.intersection(*(
                set(A.PINNED_FIELDS()) - set(A._inert_fields(o)) for o in names))
            missing = expected - set(pred)
            assert not missing, (
                f"{dict_name}[{label!r}] does not pin {sorted(missing)[:6]}"
                f"{'...' if len(missing) > 6 else ''}, which {opt} DOES receive "
                f"— it was not built by arms.arm(), so a new config field will "
                f"not be pinned in it."
            )
