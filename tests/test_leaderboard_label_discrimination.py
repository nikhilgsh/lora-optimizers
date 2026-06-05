"""Leaderboard label-discrimination CI + ``dedup_by_canonical`` unit tests.

Leaderboard notebooks define flat python lists of log-group-name string
literals (``GROUPS``/``GROUPS_R256``/``EPSREL_GROUPS``/…) and feed them to
``load_runs(where={'log_group': groups})``. The risk is that a hand-rolled
dedup key drifts from ``canonical_label`` and silently collapses distinct
variants at the same lr. This test loads each such group list and asserts the
canonical labeler discriminates every loaded config (no silent merge), plus a
direct unit test of ``dedup_by_canonical``.
"""
from __future__ import annotations

import glob
import json
import os
import re

import pytest

from lora_playground.loader import load_runs
from lora_playground.plotting.dedup import (
    assert_label_discriminates,
    dedup_by_canonical,
)
from lora_playground.plotting.labels import canonical_label

NOTEBOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"
)
LOGS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

OPT_CT = "adam-polar-product-lora-coupled-spectral-chord-tight"
OPT_CLEAN = OPT_CT + "-clean"

# A `<NAME> = [ ... ]` assignment whose NAME contains GROUPS.
_GROUPS_ASSIGN = re.compile(r"\b(\w*GROUPS\w*)\s*=\s*\[(.*?)\]", re.DOTALL)
# Quoted log-group-name string literals (names contain uppercase like phase_L).
_GROUP_NAME = re.compile(r"""['"]([A-Za-z0-9_\-]+)['"]""")


def _extract_group_lists(nb_path):
    """Return [(var_name, [group_names])] statically parsed from a notebook's
    code cells. Each `<NAME>GROUPS<NAME> = [ ... ]` list of string literals."""
    nb = json.load(open(nb_path))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for m in _GROUPS_ASSIGN.finditer(src):
            var = m.group(1)
            names = _GROUP_NAME.findall(m.group(2))
            if names:
                out.append((var, names))
    return out


def _all_group_lists():
    """Flatten every (notebook, var, groups) across leaderboard notebooks."""
    cases = []
    for nb_path in sorted(glob.glob(os.path.join(NOTEBOOKS_DIR, "*.ipynb"))):
        for var, groups in _extract_group_lists(nb_path):
            cases.append((os.path.basename(nb_path), var, groups))
    return cases


_CASES = _all_group_lists()


@pytest.mark.skipif(not _CASES, reason="no GROUPS-style lists found in notebooks/")
@pytest.mark.parametrize(
    "nb_name,var,groups",
    _CASES,
    ids=[f"{nb}:{var}" for nb, var, _ in _CASES],
)
def test_canonical_label_discriminates_leaderboard_groups(nb_name, var, groups):
    """For each notebook group list, the canonical labeler must discriminate
    every loaded config (no two distinct series collapse to one label+bucket).
    Skips when the groups load zero runs (machine without those logs)."""
    runs = load_runs(
        where={"log_group": groups},
        logs_root=LOGS_ROOT,
        warn_cross_commit=False,
        quiet=True,
    )
    if not runs:
        pytest.skip(f"{nb_name}:{var} loaded 0 runs (logs absent)")
    # Restrict to the labeled family — unlabeled runs are out of scope for the
    # canonical labeler's discrimination contract.
    labeled = [(cfg, evs) for cfg, evs in runs if canonical_label(cfg) is not None]
    if not labeled:
        pytest.skip(f"{nb_name}:{var} loaded no labeled-family runs")
    # Dedup the way the notebook does before plotting: dedup_by_canonical
    # collapses same-label+lr runs (e.g. a run resumed across two log groups,
    # which differ only on resume_from/checkpoint_dir paths). The contract is
    # that the SET THE NOTEBOOK PLOTS has a discriminating label per series.
    deduped = dedup_by_canonical(labeled)
    assert_label_discriminates(
        deduped, canonical_label, bucket_keys=("lora_r", "lr")
    )


# ── direct unit tests of dedup_by_canonical ───────────────────────────────


def _cfg(opt, **kw):
    return {"optimizer": opt, "lr": 1e-3, **kw}


def test_dedup_keeps_longest_trajectory_for_same_label_and_lr():
    """Two runs with identical canonical_label + lr but different trajectory
    length collapse to one — the longer trajectory wins."""
    base = dict(muon_ns_steps=5, polar_method="ns", lr=1e-2)
    short = (_cfg(OPT_CT, **base), [{"step": 0}, {"step": 200}])
    long = (_cfg(OPT_CT, **base), [{"step": 0}, {"step": 200}, {"step": 4000}])
    assert canonical_label(short[0]) == canonical_label(long[0])
    out = dedup_by_canonical([short, long])
    assert len(out) == 1
    assert out[0][1] is long[1]

    # Order-independent: longer still wins when it comes first.
    out2 = dedup_by_canonical([long, short])
    assert len(out2) == 1
    assert out2[0][1] is long[1]


def test_dedup_empty_evals_counts_as_step_minus_one():
    empty = (_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns"), [])
    nonempty = (_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns"), [{"step": 0}])
    out = dedup_by_canonical([empty, nonempty])
    assert len(out) == 1
    assert out[0][1] is nonempty[1]


def test_dedup_keeps_both_when_curvature_whitening_differs():
    """canonical_label distinguishes curvature_whitening True/False → both kept."""
    a = (_cfg(OPT_CLEAN, muon_ns_steps=8, polar_method="ns", curvature_whitening=False), [])
    b = (_cfg(OPT_CLEAN, muon_ns_steps=8, polar_method="ns", curvature_whitening=True), [])
    assert canonical_label(a[0]) != canonical_label(b[0])
    out = dedup_by_canonical([a, b])
    assert len(out) == 2


def test_dedup_keeps_both_when_ns_steps_differ():
    """ns=5 vs ns=8 at the same lr must not collapse (the original bug)."""
    a = (_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns", lr=1e-2), [])
    b = (_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns", lr=1e-2), [])
    assert canonical_label(a[0]) != canonical_label(b[0])
    out = dedup_by_canonical([a, b])
    assert len(out) == 2


def test_dedup_unlabeled_runs_not_collapsed_together():
    """canonical_label None → key falls back to optimizer; distinct unlabeled
    optimizers stay separate."""
    a = (_cfg("adam-lin-lora"), [{"step": 100}])
    b = (_cfg("adamw"), [{"step": 100}])
    assert canonical_label(a[0]) is None
    out = dedup_by_canonical([a, b])
    assert len(out) == 2


def test_dedup_preserves_input_order():
    a = (_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns", lr=1e-2), [])
    b = (_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns", lr=1e-2), [])
    c = (_cfg("adamw"), [])
    out = dedup_by_canonical([a, b, c])
    assert [canonical_label(cfg) or cfg["optimizer"] for cfg, _ in out] == [
        canonical_label(a[0]),
        canonical_label(b[0]),
        canonical_label(c[0]),  # "AdamW" — adamw IS in the labeled family
    ]
