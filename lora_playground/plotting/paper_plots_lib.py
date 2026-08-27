"""Definitions behind ``paper/paper_plots.ipynb``.

Everything the notebook needs -- arm predicates, the per-cell run cache, and the panel
functions -- lives here rather than in notebook cells, for one concrete reason: a
definition in a notebook cell only exists in the kernel that executed THAT cell. Editing
it on disk does nothing until the right cell is re-run, and the definitions used to be
spread across cells 1, 33, 35 and 47, so "which cell do I re-run" had no obvious answer.
That cost real time: a fix to ``NOPRODUCT`` sat on disk while the notebook kept raising
the error it fixed.

Import it as a MODULE and reach through it::

    %load_ext autoreload
    %autoreload 2
    import lora_playground.plotting.paper_plots_lib as P

    P.precond_panel(256)

``import ... as P`` and not ``from ... import *``: autoreload re-executes this module on
edit, so ``P.PROTO`` is always current, whereas a name star-imported into the notebook
namespace stays bound to the old object and goes stale exactly the way the cells did.

Arm predicates and the per-figure arm dicts both come from ``arms.py``; nothing here is
hand-typed. See that module for why an arm pins every ``OptimizerConfig`` field rather
than a remembered subset.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from lora_playground.leaderboard import (
    labeled_completed_runs, leaderboard_rows, speedup_from_frac,
)
from lora_playground.loader import load_runs, logs_signature
from lora_playground.plotting import compare_variants_figure

# File-relative, matching paper_figs.py:38 (same depth: plotting/ -> lora_playground/
# -> repo root). This used to walk up from Path.cwd() with a bare `next()` and no
# default, so importing from outside the repo raised StopIteration with an EMPTY
# message -- which is how executing a copy of paper_plots.ipynb from /tmp turned into
# 27 cascading NameErrors with nothing naming the cause. ROOT is a property of where
# this file lives, not of where the kernel was started.
ROOT = Path(__file__).resolve().parents[2]

# AdamW noise floor at packed_v1, r=16/r=64 (see the repo CLAUDE.md anchors). Deltas in
# the summary tables are quoted in units of this.
SIGMA = 0.0017

# The canonical run length every paper cell is read at. Named because three places
# ask "did this run finish": _figure's max_steps, the in-flight notice, and _max_step.
HORIZON = 9000

# ROOT must be the checkout, not wherever the package happens to be installed. Under a
# non-editable `pip install` __file__ lands in site-packages and this would silently
# resolve to <site-packages>/logs -- load_runs would then return nothing and EVERY
# panel would render empty with no exception raised. Fail at import instead: this
# module is only meaningful against a working tree.
assert (ROOT / "logs").is_dir(), (
    f"paper_plots_lib: ROOT={ROOT} has no logs/ -- this module needs the repo "
    f"checkout (an editable install), not an installed copy."
)

# --------------------------------------------------------------------------------------
# Arm predicates
# --------------------------------------------------------------------------------------
# Arm predicates come from `arms.py`, which derives them from `OptimizerConfig`:
# `arm()` pins EVERY config field to its default and takes overrides only for the
# fields an arm genuinely differs on. The hand-typed dicts that used to live here
# were allowlists of 4-14 fields, so any field they did not mention was
# unconstrained -- which is how the `e2_beta2_nomsign` sweep (5 curvature_beta
# values on kl-diag-lora) joined AVGLOSS and raised LabelCollisionError out of
# derivation_ablation_panel. arms.AVGLOSS pins 76 fields including
# curvature_beta, so a new sweep that sets a new flag falls OUT of the old arm
# and renders empty (loud) instead of merging into it (silent).
#
# This is the consolidation this module's docstring asked for. Names are
# re-exported so every caller and notebook cell keeps working unchanged.
from . import arms as _arms  # noqa: E402
from .arms import (  # noqa: F401,E402
    field_matches, pred_matches, variant_key_fn,
    ADAMW, AVGLOSS, DOUBLE, FLATOUT, HALFPOW, IMUON, LORARITE, MUON, NAIVEMAG,
    NOMAG, NOPRODUCT, NOSHAMPOO, ONESIDED, ONESIDED_DIAG, PROTO, PROTO_DIAG,
)

# The E1 cell set (paper/e1_coverage_fill.md), as an ORDER over the workload
# registry rather than a second list of experiments.
#
# `lora_playground.workloads` calls itself the single source of truth for
# (model, dataset, rank) cells and discovers them by predicate. This list used to
# retype 13 of them, and the two had already drifted in BOTH directions: r16 was
# here but absent from the registry (while r32/64/128/256 were all declared), and
# the registry carried OLMo openmath r64 and Qwen3-0.6B openwebmath r64 that
# never appear in a panel. So the leaderboard and the paper panels disagreed
# about which experiments exist, with no way to tell a deliberate omission from
# an oversight.
#
# Now only the ORDER and the panel captions live here, because `panel_n(i)`
# indexes this list and every notebook cell is written as `P.panel_n(3)` -- a
# reordering would silently repoint every figure. Existence comes from the
# registry: `_cell_from_registry` raises on a key the registry does not declare,
# so adding a panel for an undeclared cell fails loudly instead of rendering a
# figure the leaderboard does not know about. Cells the registry declares but
# that no panel shows are listed in `CELLS_NOT_PANELLED` with a reason.
_CELLS_ORDER = [
    ("OLMo-2-1B",      "opc",      256),
    ("Qwen2.5-1.5B",   "opc",      256),
    ("Llama-3.2-1B",   "opc",      256),
    ("Meta-Llama-3-8B", "opc",     256),
    ("Qwen2.5-1.5B",   "bengali",  256),
    ("OLMo-2-1B",      "openmath", 256),
    ("Qwen2.5-1.5B",   "openmath", 256),
    ("Meta-Llama-3-8B", "openmath", 256),
    ("Llama-3.2-1B",   "openmath",  16),
    ("Llama-3.2-1B",   "openmath",  32),
    ("Llama-3.2-1B",   "openmath",  64),
    ("Llama-3.2-1B",   "openmath", 128),
    ("Llama-3.2-1B",   "openmath", 256),
]

# Registry cells with no panel, and why. A key here that the registry does not
# declare, or a registry cell in neither this dict nor `_CELLS_ORDER`, is a test
# failure -- that is what keeps "not shown" distinguishable from "forgotten".
# Reasons bound once: three cells shared one string and two shared another, so
# editing one copy could silently leave its siblings saying something else.
_ONE_RANK_LADDER = "rank ladder is run on Llama-3.2-1B/openmath only"
_NOT_E1_CORPUS = "tulu3 is not an E1 corpus"
CELLS_NOT_PANELLED = {
    ("OLMo-2-1B", "opc", 64): _ONE_RANK_LADDER,
    ("OLMo-2-1B", "openmath", 64): _ONE_RANK_LADDER,
    ("Llama-3.2-1B", "opc", 64): _ONE_RANK_LADDER,
    ("OLMo-2-1B", "tulu3", 64): _NOT_E1_CORPUS,
    ("OLMo-2-1B", "tulu3", 256): _NOT_E1_CORPUS,
    ("Qwen3-0.6B", "openwebmath", 64):
        "continued pretraining (all-token loss), not instruction tuning",
}


# Panel caption overrides, where the registry's `model_display` is not the name
# the paper uses. The registry spells the 8B model "Meta-Llama-3-8B" so its
# `label` property yields "Meta/..." and does not collide with Llama-3.2-1B's
# "Llama/..." cross-setting key; the manuscript and this project's CLAUDE.md both
# say "Llama-3-8B". Keep the registry's disambiguation and the paper's name.
_CAPTION_MODEL = {"Meta-Llama-3-8B": "Llama-3-8B"}


def _cell_from_registry(model_display, dataset, rank):
    """One `CELLS` entry, with existence checked against the workload registry."""
    from lora_playground.workloads import find_workload
    wl = find_workload(model_display, dataset, rank)   # raises KeyError on a miss
    name = _CAPTION_MODEL.get(wl.model_display, wl.model_display)
    return (f"{name} {wl.dataset} r{wl.rank}", wl.model_name, wl.dataset, wl.rank)


CELLS = [_cell_from_registry(*key) for key in _CELLS_ORDER]

# --------------------------------------------------------------------------------------
# Per-cell run cache
# --------------------------------------------------------------------------------------
# One narrow load_runs per cell, memoised. A full-tree load_runs costs ~21s; a
# load_runs(where=<one cell>) is far cheaper because the narrow `where` skips parsing the
# runs that cannot match. So a single up-front snapshot is the wrong trade -- it is fixed
# at kernel start and cannot see a sweep that is still running.
_RUNS_CACHE: dict[str, tuple] = {}   # where-fingerprint -> (logs signature, runs)


def _fingerprint(v):
    """Identify a `where` value for the memo key.

    A callable's qualname alone does NOT identify it: panel()'s data_dir filter is an
    inline ``lambda d, k=key: k in str(d)``, so every panel's lambda has qualname
    '<lambda>' and they all collided -- OLMo-openmath silently reused OLMo-opc's runs and
    drew an empty panel. The bound value lives in __defaults__ (or __closure__ for a
    closed-over variable), so include both.
    """
    if callable(v):
        return ("call", getattr(v, "__qualname__", ""),
                repr(getattr(v, "__defaults__", None)),
                repr(tuple(c.cell_contents for c in (getattr(v, "__closure__", None) or ()))))
    return repr(v)


def cell_runs(where, refresh=False):
    """Runs for one cell, memoised per distinct `where` AND per logs/ signature.

    The memo self-invalidates: `loader.logs_signature` fingerprints every
    manifest and log file's (mtime, size), so a sweep that has advanced since
    the last call produces a different key and the cell re-reads. Nothing has to
    be remembered by the reader.

    This was previously memoised on `where` alone, for the kernel's life. That is
    correct for a finished tree and silently wrong for a live one: a panel run
    while a sweep was at step 8750 kept reporting 8750 after the sweep hit 9000,
    so a completed arm went on being announced as in flight and stayed out of the
    loss-vs-lr panel and the summary table.

    The reason it was not wired this way -- "an unconditional refresh makes every
    panel pay the full query again" -- does not apply to a signature check.
    Measured on this tree: logs_signature 0.21 s warm (0.86 s cold) against
    5.26 s for one cell's load_runs, i.e. 4%. The full query is still paid only
    when the tree has actually moved.

    `refresh=True` forces a re-read regardless; `clear_runs_cache()` drops the
    memo entirely. Neither should now be needed in normal use.
    """
    sig = _logs_signature_now()
    key = repr(sorted((k, _fingerprint(v)) for k, v in where.items()))
    cached = _RUNS_CACHE.get(key)
    if cached is not None and cached[0] == sig and not refresh:
        return cached[1]
    runs = load_runs(where=where, warn_cross_commit=False, quiet=True)
    _RUNS_CACHE[key] = (sig, runs)
    return runs


def clear_runs_cache():
    """Drop the memo so the next panel re-reads from disk."""
    _RUNS_CACHE.clear()
    _SIG_HOLD.clear()


# One `logs/` signature per panel, not one per `cell_runs` call.
#
# `cell_runs` stats the whole tree BEFORE its memo lookup, so even a memo HIT
# paid a full scan -- measured 197.8 ms warm over 2497 log files, and ~4x that
# cold. `_figure` calls `cell_runs` 9-11 times (once per arm in `has`, again per
# arm in `_max_step`, plus `prefetched_runs`, `speedup_table` and
# `coverage_report`), so the same unchanged tree was rescanned ten times per
# figure: `precond_panel(256)` measured 2.274 s, of which 0.046 s was the work.
# Across the notebook's 23 `_figure` calls that is seconds of pure stat().
#
# Nine snapshots inside one figure buy no freshness either -- if the tree really
# did move mid-panel, mixing snapshots across arms is WORSE than holding one,
# because the loss-vs-lr panel and the speedup table would then disagree about
# which runs exist. So the hold is a correctness improvement as well as a speed
# one. It lasts exactly one `_figure`; anything outside that scope re-stats.
_SIG_HOLD: dict[str, str] = {}


def _logs_signature_now() -> str:
    if "sig" not in _SIG_HOLD:
        return logs_signature(str(ROOT / "logs"))
    return _SIG_HOLD["sig"]


@contextmanager
def _held_logs_signature():
    """Freeze the tree signature for the duration of one panel."""
    outer = "sig" in _SIG_HOLD
    if not outer:
        _SIG_HOLD["sig"] = logs_signature(str(ROOT / "logs"))
    try:
        yield
    finally:
        if not outer:
            _SIG_HOLD.pop("sig", None)


def om(rank):
    """common_where for Llama-3.2-1B / openmath at a rank -- the ablation cell."""
    return dict(model_name="meta-llama/Llama-3.2-1B", lora_r=rank,
                data_dir=(lambda d: "openmath" in str(d)))


def cell(model, data_key, rank):
    """common_where for one (model, corpus, rank) cell."""
    return dict(model_name=model, lora_r=rank,
                data_dir=(lambda d, k=data_key: k in str(d)))


def has(where, common):
    """Is there any run for this arm in this cell? Used to drop empty arms."""
    pred = {**where, **common}
    return any(pred_matches(cfg, pred) for cfg, _h in cell_runs(common))


def _max_step(where, common):
    """Furthest step any run of this arm reached, or None if the arm has no runs.

    `has()` answers "does a run match"; this answers "did one FINISH", which is
    the question the loss-vs-lr panel and the summary table actually ask.
    """
    pred = {**where, **common}
    steps = [h[-1]["step"] for cfg, h in cell_runs(common)
             if h and pred_matches(cfg, pred)]
    return max(steps) if steps else None


def _truncate(runs, horizon, keep_full=None):
    """Cut every history at `horizon` steps, dropping runs that never reach it.

    A panel comparing arms run to DIFFERENT horizons has to read them at a
    matched step or it compares one arm's step-9000 loss against another's
    step-2000 loss and calls the difference an effect. `train.py` defaults to
    `lr_scheduler_type=constant` with `warmup_steps=0` (`train.py:452-453`) and
    the sweep wrappers override neither, so the first N steps of a longer run
    ARE an N-step run and this truncation is exact rather than approximate.
    Re-check that if a schedule is ever introduced.

    `keep_full` is a predicate on cfg naming runs to leave UNCUT. It exists for
    the reference arm: AdamW is not one of the things being compared, it is the
    scale the reader reads the others against, and cutting it at 2000 throws away
    the part of its curve that says where the run ends up. Its full trajectory is
    drawn while every COMPARED arm stays matched at `horizon`.

    The cost of that is real and is why this is opt-in: the reference's own
    `final` in the summary table is then its step-9000 loss while the compared
    arms report step-`horizon`, so the speed target is computed against a longer
    run. `_figure` therefore says so in the panel rather than leaving the reader
    to infer it from the legend.
    """
    out = []
    for cfg, hist in runs:
        if keep_full is not None and keep_full(cfg):
            out.append((cfg, hist))
            continue
        cut = [e for e in hist if e.get("step") is not None and e["step"] <= horizon]
        if cut and max(e["step"] for e in cut) >= horizon:
            out.append((cfg, cut))
    return out


def _figure(arms, common, ref_label, suptitle, target_label="AdamW", drop_empty=True,
            horizon=None, trajectory_only=False):
    """Every panel in this module funnels through here.

    Passing prefetched_runs AND variant_key is what arms the ``assert_label_discriminates``
    guard inside compare_variants_figure -- the per-variant loading path does not run it,
    and that is how two arms silently merged for weeks. Do not add a panel that skips it.
    """
    with _held_logs_signature():
        return _figure_inner(arms, common, ref_label, suptitle, target_label,
                             drop_empty, horizon, trajectory_only)


def _figure_inner(arms, common, ref_label, suptitle, target_label, drop_empty, horizon,
                  trajectory_only=False):
    declared_arms = dict(arms)
    if drop_empty:
        missing = [k for k, v in arms.items() if not has(v, common)]
        if missing:
            print("no data yet (omitted):", ", ".join(missing))
        arms = {k: v for k, v in arms.items() if k not in missing}
    # An arm whose runs all stopped short of max_steps is the SILENT case, and it
    # is not covered by `missing` above: `has()` only asks whether any run matches
    # the predicate, so an arm with five in-flight runs looks present. But
    # `allow_partial` files those into compare_variants_figure's
    # `per_variant_partial`, which reaches the trajectory panel ONLY -- the
    # final-loss-vs-lr panel and the returned summary_df are built from complete
    # runs alone. Measured before this print existed: precond_panel declared 4 arms
    # and returned 3 rows, msign_panel declared 5 and returned 2, with no output at
    # all. A reader could not tell "not run here" from "still running".
    h = HORIZON if horizon is None else horizon
    in_flight = {k: n for k, v in arms.items()
                 if (n := _max_step(v, common)) is not None and n < h}
    if in_flight:
        print(f"in flight (trajectory panel only, absent from the loss-vs-lr panel "
              f"and the summary table until {h} steps): "
              + ", ".join(f"{k} @{n}" for k, n in in_flight.items()))
    runs = cell_runs(common)
    if horizon is not None:
        # The reference arm keeps its full trajectory. It is the scale the other
        # arms are read against, not one of the things being compared, and
        # cutting it at `horizon` discards exactly the part that says where the
        # workload ends up. Say so in the panel: its `final` in the table below
        # is then a longer run's than the compared arms'.
        ref_pred = arms.get(target_label)
        keep_full = ((lambda c: pred_matches(c, {**common, **ref_pred}))
                     if ref_pred is not None else None)
        runs = _truncate(runs, horizon, keep_full=keep_full)
        if ref_pred is not None:
            print(f"read at step {horizon}; {target_label!r} is the reference and "
                  f"keeps its full {HORIZON}-step curve, so its 'final' below is "
                  f"not step-matched to the other arms.")
        else:
            print(f"read at step {horizon} (every arm truncated).")
    fig, _t, sdf = compare_variants_figure(
        arms, common_where=common, ref_label=ref_label,
        logs_root=str(ROOT / "logs"), sigma_ref=SIGMA, max_steps=h,
        allow_partial=True, allow_custom_labels=True, target_label=target_label,
        suptitle=suptitle,
        prefetched_runs=runs, variant_key=variant_key_fn(common, arms),
        trajectory_only=trajectory_only)
    plt.show()
    if target_label in arms:
        print(speedup_table(arms, common, baseline_label=target_label, horizon=h)[0])
    else:
        print(f"no speedup table: '{target_label}' is not one of this panel's arms, "
              f"so there is no speed target. Pass target_label=<an arm> to get one.")
    # Coverage is reported against the arms the CALLER declared, before
    # `drop_empty` pruned the empty ones, so an arm that matched nothing is still
    # a candidate for "closest arm" in the diagnosis.
    cov = coverage_report(declared_arms, common)
    if cov:
        print(cov)
    return sdf


# --------------------------------------------------------------------------------------
# Coverage: runs that exist in a cell but that no arm claimed
# --------------------------------------------------------------------------------------
# Every arm predicate fails the same way -- by matching FEWER runs, silently. An
# arm that pins a field the runs disagree on is indistinguishable, in a rendered
# panel, from an arm that legitimately has no data yet. Three instances of that
# in one session:
#
#   arms.ADAMW pinned `cw_nesterov=True`, a flag LoRAPlusAdamW never reads, and
#   every adamw run at 5 of the 13 CELLS logs False -- so those cells rendered
#   with NO baseline and leaderboard_rows returned a NaN speed target.
#
#   arms.NOPRODUCT has to admit two optimizer names for one branch
#   (kl-diag-polar-lora and kl-shampoo-polar-lora); pinning one dropped half the
#   arm, and the r16 factorwise cells matched nothing while the figure showed the
#   arm as absent.
#
#   logs/e2_precond_r16_postfix_xl had no run_info/meta.json, so its 4 completed
#   runs were dropped by load_manifests(strict=False) before any predicate ran.
#
# The first two are invisible to a manifest check and the third is invisible to a
# predicate check, so the panel reports both: what exists in this cell, and which
# pinned field kept each unclaimed run out.
_COVERAGE_MAX_ROWS = 10
# Fields worth naming in the report. A run differs from an arm on dozens of
# fields it was never meant to match, so the diagnostic is only useful if it
# names the arm that is CLOSEST and only the fields that separate them.
_COVERAGE_IDENT = ("optimizer", "precond", "msign", "lr", "lora_r", "max_steps")


def _closest_arm(cfg, arms):
    """``(arm_label, [mismatching field descriptions])`` for the nearest arm.

    "Nearest" is fewest mismatching pinned fields. That is what turns "this run
    matched nothing" into "this run matched nothing because ADAMW pins
    cw_nesterov=True and the run logs False", which is the actionable form.
    """
    best = None
    for label, pred in arms.items():
        diffs = []
        for k, want in pred.items():
            # `arms.field_matches` is THE definition of what a pin means. This
            # used to re-implement the three branches, and the two drifted: a
            # list-vs-list equality branch was added to the matcher and not to
            # the copy here, so this reported `arm wants [], run has []` as a
            # mismatch on a field that matches -- a diagnostic contradicting the
            # predicate it exists to explain.
            # No blanket try/except here. A bare `except Exception: ok = False`
            # turned a NameError on `field_matches` into "every pinned field
            # mismatches", i.e. the diagnostic blamed all ~130 pins instead of
            # failing. Only a CALLABLE pin can legitimately raise on a value it
            # was not written for; everything else must surface.
            if callable(want):
                try:
                    ok = field_matches(cfg, k, want)
                except Exception:
                    ok = False
            else:
                ok = field_matches(cfg, k, want)
            if not ok:
                diffs.append(f"{k}: arm wants {want!r}, run has {cfg.get(k, '<absent>')!r}")
        if best is None or len(diffs) < len(best[1]):
            best = (label, diffs)
    return best if best else ("<no arms>", [])


def coverage_report(arms, common, *, horizon=HORIZON, detail=False):
    """Text naming the runs in this cell that no arm's predicate claimed.

    Returns "" when every run is claimed, so a clean cell prints nothing.

    TERSE BY DEFAULT — one line — because most unclaimed runs are not a bug.
    A paper panel is SELECTIVE: `Llama-3.2-1B/openmath/r256` holds 139 completed
    runs spanning about 35 distinct configurations (every ablation ever run at
    that rank), and `PANEL_ARMS` deliberately plots four of them. Printing ten
    rows plus "and 114 more" under every such panel buries the cells where the
    omission IS a bug, which is the whole point of the check: the signal is a
    cell whose count jumps after a sweep lands, not a cell with a large count.

    `detail=True` restores the per-run diagnosis — the closest arm and the
    fields separating it — which is what to reach for once a count looks wrong.
    """
    runs = cell_runs(common)
    unclaimed = [(cfg, hist) for cfg, hist in runs
                 if not any(pred_matches(cfg, {**common, **pred})
                            for pred in arms.values())]
    if not unclaimed:
        return ""
    head = (f"{len(unclaimed)} of {len(runs)} runs in this cell are outside the "
            f"{len(arms)} plotted arm(s)")
    if not detail:
        return (f"{head} — `coverage_report(arms, common, detail=True)` names them "
                f"and the field that excluded each.")
    lines = [f"UNCLAIMED: {head}. Each is absent from the figure and the table above."]
    for cfg, hist in unclaimed[:_COVERAGE_MAX_ROWS]:
        ident = " ".join(f"{k}={cfg.get(k)!r}" for k in _COVERAGE_IDENT if k in cfg)
        last = max(hist, key=lambda e: e.get("step", 0)).get("step", 0) if hist else 0
        label, diffs = _closest_arm(cfg, arms)
        why = ("; ".join(diffs[:3]) + (f"; +{len(diffs) - 3} more" if len(diffs) > 3 else "")
               if diffs else "no field mismatch -- check the dedup, not the predicate")
        lines.append(f"  step {last}  {ident}")
        lines.append(f"     closest arm {label!r}: {why}")
    if len(unclaimed) > _COVERAGE_MAX_ROWS:
        lines.append(f"  ... and {len(unclaimed) - _COVERAGE_MAX_ROWS} more unclaimed runs "
                     f"(raise _COVERAGE_MAX_ROWS to see them)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Speed-to-target table
# --------------------------------------------------------------------------------------
def speedup_table(arms, common, *, baseline_label="AdamW", horizon=HORIZON):
    """``(text, rows, target)`` for one cell's arms, keyed on speedup-vs-baseline.

    This is the metric optimizer decisions are made on -- HORIZON / (steps to
    reach the baseline arm's best final loss), per ``docs/notes/leaderboard.md``
    -- and `_figure` prints it under every panel so a panel is never read off
    final loss alone. Reading loss instead understates the effect by a lot: at
    the Llama-3.2-1B / openmath / r16 cell the `factorwise` precond arm is
    0.0034 above `one-sided` in final loss, which is 1.10x against 1.26x in
    speedup -- a 2.6x cut in the gain over AdamW.

    Every arm, the baseline included, must come from ``arms.py``: it pins all of
    `OptimizerConfig`, whereas a hand-typed ``{"optimizer": "adamw"}`` admits the
    deliberate `beta2` grid alongside the shipped baseline (6 distinct AdamW
    `series_id`s at one lr) and `labeled_completed_runs` raises
    `LabelCollisionError`. Loading goes through `cell_runs`, so the panel above
    and this table always read the same snapshot.
    """
    if baseline_label not in arms:
        raise ValueError(
            f"baseline_label={baseline_label!r} is not in arms {sorted(arms)}; "
            f"the speed target is the baseline arm's best final loss, so the "
            f"baseline has to be one of the arms being loaded.")
    # Truncated to `horizon` for the same reason `_figure` truncates: a table
    # comparing arms run to different lengths must read them at a matched step,
    # or it quotes one arm's step-9000 loss beside another's step-2000 loss.
    runs = cell_runs(common)
    if horizon != HORIZON:
        runs = _truncate(runs, horizon)
    labeled = labeled_completed_runs(
        runs, variant_key_fn(common, arms), horizon=horizon)
    rows, target = leaderboard_rows(
        labeled, horizon=horizon, baseline_label=baseline_label)
    for r in rows:
        r["speedup"] = speedup_from_frac(r["frac_best_lr"])
        r["speedup_lr_avg"] = speedup_from_frac(r["frac_lr_avg"])
    # NaN speedup means "never reached the target", which sorts last, not first.
    rows.sort(key=lambda r: math.inf if math.isnan(r["speedup"]) else -r["speedup"])
    return _speedup_text(rows, target, baseline_label, horizon), rows, target


def _fmt_x(v):
    return "—" if v is None or math.isnan(v) else f"{v:.2f}x"


def _speedup_text(rows, target, baseline_label, horizon=HORIZON):
    if math.isnan(target):
        return (f"speed target is NaN: no completed {baseline_label} run in this "
                f"cell. Check that the {baseline_label} arm's pinned fields admit "
                f"the runs that exist (`arms.ADAMW` pinning `cw_nesterov=True` "
                f"left 5 of the 13 CELLS with no baseline at all).")
    w = max(len(r["variant"]) for r in rows)
    head = (f"speedup vs {baseline_label} (target {target:.4f}, horizon {horizon})\n"
            f"  {'arm':<{w}}  {'best eta':>9}  {'final':>7}  {'speedup':>8}"
            f"  {'lr-avg':>7}  n_lr")
    body = [f"  {r['variant']:<{w}}  {r['best_lr']:>9g}  {r['final_at_best']:>7.4f}"
            f"  {_fmt_x(r['speedup']):>8}  {_fmt_x(r['speedup_lr_avg']):>7}"
            f"  {r['n_lrs']}" for r in rows]
    return "\n".join([head, *body])


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------
def panel(name, model, key, rank):
    """AdamW vs PoLoRA vs the baselines at one cell; Delta vs AdamW in sigma units."""
    common = dict(model_name=model, lora_r=rank, data_dir=(lambda d, k=key: k in str(d)))
    arms = {"AdamW": ADAMW, "Polar-LoRA (kl-diag)": PROTO, "iMuon": IMUON,
            "Muon (naive)": MUON, "LoRA-RITE": LORARITE,
            "w/o curvature+magnitude (LoRA-Muon step)": DOUBLE}
    return _figure(arms, common, "AdamW", name, drop_empty=False)


def panel_n(i):
    """Panel for CELLS[i] -- so a notebook cell is one short line."""
    return panel(*CELLS[i])


def ablation_panel(rank=256):
    """E2 leave-one-out at one rank."""
    arms = {"Polar-LoRA (kl-diag)": PROTO, "w/o curvature control": NOSHAMPOO,
            "w/o magnitude control": NOMAG,
            "w/o curvature+magnitude (LoRA-Muon step)": DOUBLE}
    return _figure(arms, om(rank), "Polar-LoRA (kl-diag)",
                   f"E2 ablation - Llama-3.2-1B openmath r{rank}", target_label=None)


def derivation_ablation_panel(rank=256):
    """Which derivation premise carries the method: orthogonalization, or metric power."""
    arms = {"AdamW": ADAMW,
            "PoLoRA: rxr=B^T P B, shared P,Q": PROTO,
            "no msign, metric^-1 (averaged loss)": AVGLOSS,
            "no msign, metric^-1/2": HALFPOW,
            "no outer un-whiten: msign only": FLATOUT}
    return _figure(arms, om(rank), "PoLoRA: rxr=B^T P B, shared P,Q",
                   f"Derivation: orthogonalization and metric power - "
                   f"Llama-3.2-1B openmath r{rank}")


def precond_panel(rank=256, model="meta-llama/Llama-3.2-1B",
                  data_key="openmath", model_label="Llama-3.2-1B"):
    """The three `precond` branches: what fills (C_B, C_A). All three share one
    (P, Q), the same p, q updates and the same rho rule.

    Parameterized by cell so the same comparison can be read at another rank
    (C_B and C_A are r x r, so the slot has less to offer as r falls) or on
    another architecture, without a second copy of the figure.
    """
    return _figure(_arms.PRECOND_ARMS, cell(model, data_key, rank),
                   "product: C_B=B^T P B, C_A=A Q A^T",
                   f"The r x r metric slot - {model_label} {data_key} r{rank}")


def msign_panel(rank=256):
    """The `msign` axis at both ends of `precond`: can the matrix sign be replaced
    by its diagonal (rownorm / colnorm) with the slot present, and with it gone?"""
    arms = {"AdamW": ADAMW,
            "product, msign": PROTO,
            "product, diagonal msign": PROTO_DIAG,
            "one-sided, msign": ONESIDED,
            "one-sided, diagonal msign": ONESIDED_DIAG}
    return _figure(arms, om(rank), "product, msign",
                   f"Diagonal msign - Llama-3.2-1B openmath r{rank}")


def magnitude_rule_panel(rank=256):
    """Naive rho = eta against the PoLoRA rule rho = eta/(smax(A)+smax(B))."""
    return _figure(_arms.MAGNITUDE_RULE_ARMS, om(rank),
                   "PoLoRA: rho = eta/(smax A + smax B)",
                   f"Magnitude rule: naive vs PoLoRA - Llama-3.2-1B openmath r{rank}")


def beta2_panel(rank=256):
    """Protagonist curvature_beta grid: the EMA horizon of the P, Q metric."""
    return _figure(_arms.PROTO_BETA2_ARMS, om(rank),
                   "Polar-LoRA (shipped, b2=0.99)",
                   f"Protagonist beta2 sweep - Llama-3.2-1B openmath r{rank}",
                   target_label=None)


def precond_beta2_panel(rank=16):
    """Is factorwise's deficit at small r the cost of whitening by a NOISY estimate?

    `factorwise` fills the r x r slot with `P_A`, an EMA of the factor's own
    gradients; `one-sided` fills it with `I_r`. An EMA over
    n_eff = 1/(1-curvature_beta) samples is anisotropic EVEN WHEN THE TRUE
    CURVATURE IS ISOTROPIC, so whitening by it perturbs the update along
    directions the problem does not have. `I_r` has zero estimation variance and
    cannot make that error; it only forgoes whatever real anisotropy exists.

    Anisotropy here means 1 - stable_rank(M)/r, where stable_rank = sum(lambda) /
    max(lambda) of the r x r Gram: 0 is perfectly isotropic. Feeding the EMA
    gradients whose TRUE second moment is exactly I measures the noise floor --
    0.098 / 0.126 / 0.125 at r = 16 / 64 / 256, flat in rank because the EMA
    window does not depend on r -- against the real anisotropy of the factor's
    own Gram, 0.195 / 0.338 / 0.447, which grows. At r=16 there is only 2.0x as
    much real structure as the estimator manufactures from noise, against 3.6x at
    r=256.

    TRAJECTORY ONLY, and NOT truncated. The beta2=0.999 arms ran 2000 steps
    against 9000-step beta2=0.99 comparands, so the two are on different grids.
    The final-loss-vs-lr panel is dropped rather than shown, because its job is
    to display each arm's minimum sitting INSIDE its lr grid, and a 2-point arm
    drawn beside a 7-point one invites reading a line segment as a resolved
    curve. The trajectory panel is honest about the mismatch by construction: a
    curve that stops at 2000 visibly stops at 2000.

    The one-sided arms are the CONTROL, not padding: `curvature_beta` drives four
    EMAs, not one -- P_A/Q_B (factorwise only) at optim.py:2184-2186, 2200-2201,
    and D_in/D_out (BOTH arms) at 2191-2195, 2202-2203. Raising it helps
    everything, so only a gap that shrinks MORE than the one-sided control moves
    isolates the r x r slot.
    """
    return _figure(_arms.PRECOND_BETA2_ARMS, om(rank), "one-sided, b2=0.99",
                   f"Estimation noise in the r x r slot: curvature_beta x precond "
                   f"- Llama-3.2-1B openmath r{rank}",
                   target_label="AdamW", trajectory_only=True)


def adamw_beta2_panel(rank=256):
    """AdamW beta2 control -- the negative control for the protagonist beta2 grid."""
    return _figure(_arms.ADAMW_BETA2_ARMS, om(rank), "AdamW (beta2=0.999)",
                   f"AdamW beta2 control - Llama-3.2-1B openmath r{rank}",
                   target_label=None)
