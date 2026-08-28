"""Definitions behind ``paper/paper_plots.ipynb``.

Everything the notebook needs -- arm predicates, the per-cell run cache, and the panel
functions -- lives here rather than in notebook cells, for one concrete reason: a
definition in a notebook cell only exists in the kernel that executed THAT cell. Editing
it on disk does nothing until the right cell is re-run, and the definitions used to be
spread across cells 1, 33, 35 and 47, so "which cell do I re-run" had no obvious answer.
That cost real time: a fix to ``NOPRODUCT`` sat on disk while the notebook kept raising
the error it fixed.

Import it as a MODULE and reach through it::

    import lora_playground.plotting.paper_plots_lib as P

    P.precond_panel(256)

Keep one module generation for the kernel lifetime. Restart the kernel after
editing plotting code so cached records and their classes cannot come from
different imports.

Reviewed panels resolve checked-in stable-ID views. Panels whose historical
semantics are not yet sealed retain the existing ``arms.py`` compatibility path;
that path is isolated and does not manufacture publication identities.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from lora_playground.leaderboard import (
    leaderboard_rows_from_comparison,
    speedup_from_frac,
)
from lora_playground.comparison import VariantSpec, build_comparison
from lora_playground.loader import (
    load_runs,
    logged_field_predicate,
    logs_signature,
)
from lora_playground.plotting.labels import (
    canonical_arm_label,
    canonical_colors,
)
from lora_playground.plotting.paper_style import resolve_paper_styles
from lora_playground.plotting.render import render_comparison
from lora_playground.plotting.style import NOTEBOOK_RCPARAMS
from lora_playground.publication_paper import (
    publication_view_panel,
    publication_workload_view_panel,
)
from lora_playground.plotting.paper_view_semantics import project_paper_precond_cohort
from lora_playground.run_catalog import RunCatalog
from lora_playground.run_records import run_view
from lora_playground.run_schema import MEASUREMENT_SEMANTICS_REVISION
from lora_playground.workloads import resolve_dataset

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
_CATALOG_SNAPSHOT: tuple[str, RunCatalog] | None = None
_NOTEBOOK_SIGNATURE: str | None = None


def _fingerprint(v):
    """Identify a `where` value for the memo key.

    A callable's qualname alone does NOT identify it: panel()'s data_dir filter is an
    inline ``lambda d, k=key: k in str(d)``, so every panel's lambda has qualname
    '<lambda>' and they all collided -- OLMo-openmath silently reused OLMo-opc's runs and
    drew an empty panel. The bound value lives in __defaults__ (or __closure__ for a
    closed-over variable), so include both.
    """
    if callable(v):
        cache_key = getattr(v, "cache_key", None)
        if cache_key is not None:
            return ("logged-field-predicate", cache_key)
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
    if not (ROOT / "logs").is_dir():
        raise RuntimeError(
            "paper panels need the live logs/ tree"
        )
    global _CATALOG_SNAPSHOT
    sig = _logs_signature_now()
    key = repr(sorted((k, _fingerprint(v)) for k, v in where.items()))
    cached = _RUNS_CACHE.get(key)
    if cached is not None and cached[0] == sig and not refresh:
        return cached[1]
    catalog = _catalog_for_signature(sig)
    runs = load_runs(
        where=where,
        catalog=catalog,
        warn_cross_commit=False,
        quiet=True,
    )
    _RUNS_CACHE[key] = (sig, runs)
    return runs


def clear_runs_cache():
    """Drop the memo so the next panel re-reads from disk."""
    global _CATALOG_SNAPSHOT, _NOTEBOOK_SIGNATURE
    _RUNS_CACHE.clear()
    _CATALOG_SNAPSHOT = None
    _NOTEBOOK_SIGNATURE = None
    _SIG_HOLD.clear()


def _catalog_for_signature(signature: str) -> RunCatalog:
    """Reuse one parsed catalog while the recorded log tree is unchanged."""
    global _CATALOG_SNAPSHOT
    if _CATALOG_SNAPSHOT is None or _CATALOG_SNAPSHOT[0] != signature:
        _CATALOG_SNAPSHOT = (
            signature, RunCatalog.discover(ROOT / "logs")
        )
    return _CATALOG_SNAPSHOT[1]


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
    if _NOTEBOOK_SIGNATURE is not None:
        return _NOTEBOOK_SIGNATURE
    if "sig" not in _SIG_HOLD:
        return logs_signature(str(ROOT / "logs"))
    return _SIG_HOLD["sig"]


@contextmanager
def _held_logs_signature():
    """Freeze the tree signature for the duration of one panel."""
    if _NOTEBOOK_SIGNATURE is not None:
        yield
        return
    outer = "sig" in _SIG_HOLD
    if not outer:
        _SIG_HOLD["sig"] = logs_signature(str(ROOT / "logs"))
    try:
        yield
    finally:
        if not outer:
            _SIG_HOLD.pop("sig", None)


def begin_notebook_snapshot(*, refresh: bool = False) -> str:
    """Hold one physical log-tree snapshot across all notebook cells.

    A paper render must not mix cells from different moments merely because an
    unrelated sweep appended to ``logs/`` between cells.  Calling this from the
    setup cell also avoids rescanning and rebuilding the entire catalog on each
    such append.  Pass ``refresh=True`` when rerunning the setup cell to pick up
    newly completed runs deliberately.
    """
    global _CATALOG_SNAPSHOT, _NOTEBOOK_SIGNATURE
    if _NOTEBOOK_SIGNATURE is None or refresh:
        signature = logs_signature(str(ROOT / "logs"))
        if signature != _NOTEBOOK_SIGNATURE:
            _RUNS_CACHE.clear()
            if (
                _CATALOG_SNAPSHOT is not None
                and _CATALOG_SNAPSHOT[0] != signature
            ):
                _CATALOG_SNAPSHOT = None
        _NOTEBOOK_SIGNATURE = signature
    return _NOTEBOOK_SIGNATURE


def end_notebook_snapshot() -> None:
    """Return panel calls to per-call live-tree freshness checks."""
    global _NOTEBOOK_SIGNATURE
    _NOTEBOOK_SIGNATURE = None


def _dataset_predicate(key):
    return logged_field_predicate(
        lambda data_dir, dataset=key: (
            resolve_dataset({"data_dir": data_dir}) == dataset
        ),
        cache_key=f"dataset={key}",
    )


def om(rank):
    """common_where for Llama-3.2-1B / openmath at a rank -- the ablation cell."""
    return dict(
        model_name="meta-llama/Llama-3.2-1B",
        lora_r=rank,
        data_dir=_dataset_predicate("openmath"),
    )


def cell(model, data_key, rank):
    """common_where for one (model, corpus, rank) cell."""
    return dict(
        model_name=model,
        lora_r=rank,
        data_dir=_dataset_predicate(data_key),
    )


def _canonical_variant_key(common, arms):
    """Map recorded configs to editorial labels through the canonical labeler.

    Historical logs legitimately omit flags added after they ran.  The old
    arm predicates pin today's full CLI surface and therefore cannot classify
    those records.  Canonical labels already define the repository-wide
    compatibility semantics for absent default-valued fields; derive the
    editorial lookup from the arm declarations once instead of reconstructing
    config data.
    """
    by_canonical = {}
    for editorial_label, predicate in arms.items():
        optimizer = predicate["optimizer"]
        optimizers = (
            optimizer
            if isinstance(optimizer, (list, set, tuple, frozenset))
            else (optimizer,)
        )
        for optimizer_name in optimizers:
            label = canonical_arm_label({**predicate, "optimizer": optimizer_name})
            if label is None:
                continue
            previous = by_canonical.setdefault(label, editorial_label)
            if previous != editorial_label:
                raise ValueError(
                    f"canonical variant {label!r} maps to both "
                    f"{previous!r} and {editorial_label!r}"
                )

    def variant_key(cfg):
        if not pred_matches(cfg, common):
            return None
        return by_canonical.get(canonical_arm_label(cfg))

    return variant_key


_UNPINNED_MEASUREMENT_REVISION = object()


def _render_panel_comparison(
    comparison,
    *,
    reference_id,
    target_id,
    target_label,
    suptitle,
    horizon,
):
    """Render an already-built comparison without crossing a loader adapter."""
    specs = tuple(comparison.variants)
    style_tokens = {
        spec.id: spec.style_key or spec.label
        for spec in specs
    }
    styles_by_token = resolve_paper_styles(style_tokens.values())
    colors = {
        spec.id: styles_by_token[style_tokens[spec.id]]["color"]
        for spec in specs
    }
    markers = {
        spec.id: styles_by_token[style_tokens[spec.id]]["marker"]
        for spec in specs
    }
    # Standard comparison panels have their own stable visual contract. Import
    # order and long-lived notebook state must not restyle later panels.
    with plt.rc_context(NOTEBOOK_RCPARAMS):
        _fig, _table, summary = render_comparison(
            comparison,
            reference_id=reference_id,
            target_id=target_id,
            colors=colors,
            markers=markers,
            sigma_ref=SIGMA,
            horizon=horizon,
            show_partials=True,
            suptitle=suptitle,
        )
    plt.show()

    if target_id is None:
        print("no speedup table: this panel declares no speed target.")
        return summary

    rows, target = leaderboard_rows_from_comparison(
        comparison,
        horizon=horizon,
        baseline_id=target_id,
    )
    for row in rows:
        row["speedup"] = speedup_from_frac(row["frac_best_lr"])
        row["speedup_lr_avg"] = speedup_from_frac(row["frac_lr_avg"])
    rows.sort(key=lambda row: (
        math.inf if math.isnan(row["speedup"]) else -row["speedup"]
    ))
    print(_speedup_text(rows, target, target_label, horizon))
    return summary


def _archive_figure(view_id, suptitle=None):
    """Render one reviewed stable-ID view from the sealed archive."""
    panel = publication_view_panel(view_id)
    comparison = panel.comparison
    if panel.reference_id is None or panel.horizon is None:
        raise ValueError(f"publication view {view_id!r} has no rendering roles")
    labels_by_id = {spec.id: spec.label for spec in comparison.variants}
    target_label = (
        labels_by_id[panel.target_id] if panel.target_id is not None else None
    )
    summary = _render_panel_comparison(
        comparison,
        reference_id=panel.reference_id,
        target_id=panel.target_id,
        target_label=target_label,
        suptitle=suptitle or panel.title,
        horizon=panel.horizon,
    )
    if comparison.unmatched_run_ids:
        print(
            f"{len(comparison.unmatched_run_ids)} archived runs in this cell "
            "belong to other sealed publication variants."
        )
    return summary


def _records_figure(arms, workload, ref_label, suptitle, *,
                    target_label="AdamW", semantic_view=None,
                    measurement_semantics_revision=(
                        _UNPINNED_MEASUREMENT_REVISION
                    )):
    """Render one live workload without converting records back to tuples."""
    with _held_logs_signature():
        from lora_playground.workloads import workload_records

        records = workload_records(
            workload,
            catalog=_catalog_for_signature(_logs_signature_now()),
        )
        excluded = ()
        if semantic_view is not None:
            records, excluded = project_paper_precond_cohort(
                records,
                view_id=semantic_view,
            )
        if measurement_semantics_revision is not _UNPINNED_MEASUREMENT_REVISION:
            # The pin keeps the COMPARED branches on one measurement revision.
            # A run with no `precond` branch is not one of them -- it is the
            # scale the branches are read against -- so pinning it buys nothing
            # and only removes it: every AdamW run in the Qwen matched cells
            # predates the field, so the baseline vanished from panels whose
            # question is which branch fills the r x r slots. Exemption is
            # derived from `effective_precond` returning None (outside the
            # CurvatureWhitenLoRA family), not from the optimizer's name.
            from lora_playground.plotting.paper_view_semantics import (
                effective_precond,
            )

            def _revision_ok(record, index):
                cfg = run_view(record, index).semantic_config
                if effective_precond(cfg) is None:
                    return True
                return cfg.get("measurement_semantics_revision") == \
                    measurement_semantics_revision

            records = tuple(
                record for index, record in enumerate(records)
                if _revision_ok(record, index)
            )
        variant_key = _canonical_variant_key({}, arms)
        specs = tuple(
            VariantSpec(
                id=label,
                label=label,
                style_key=label,
                predicate=(
                    lambda cfg, expected=label: variant_key(cfg) == expected
                ),
            )
            for label in arms
        )
        comparison = build_comparison(records, specs, horizon=workload.horizon)
        missing = [
            spec.label for spec in specs
            if comparison.best_completed[spec.id] is None
            and comparison.best_partial[spec.id] is None
        ]
        if missing:
            print("no data yet (omitted):", ", ".join(missing))
        if ref_label in missing:
            raise ValueError(
                f"{workload.label} has no recorded reference arm {ref_label!r}"
            )
        # A speed target only exists if its arm resolved a completed run. The
        # panel used to demand one whenever the arm was DECLARED, so a cell
        # that simply has no baseline on disk raised UncertifiedBaselineError
        # out of `leaderboard_rows_from_comparison` instead of drawing the
        # comparison it does have -- Qwen2.5/openmath/r16 records no AdamW run
        # at any revision. Absence is a note, not an error.
        has_target = (
            target_label in arms
            and target_label not in missing
            and comparison.best_completed.get(target_label) is not None
        )
        if target_label in arms and not has_target:
            print(f"no speed target: {target_label} has no completed run in "
                  f"{workload.label}.")
        summary = _render_panel_comparison(
            comparison,
            reference_id=ref_label,
            target_id=(target_label if has_target else None),
            target_label=target_label,
            suptitle=suptitle,
            horizon=workload.horizon,
        )
        if excluded:
            reasons = sorted({decision.reason for _run, decision in excluded})
            print(
                f"{len(excluded)} run(s) excluded by {semantic_view!r} "
                f"semantic cohort: {'; '.join(reasons)}"
            )
        return summary


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
    """Archived primary optimizer comparison at one declared workload."""
    from lora_playground.workloads import find_workload
    workload = find_workload(model, key, rank)
    archived = publication_workload_view_panel(
        "paper.e1_comparison.all_workloads.v1",
        workload,
    )
    return _render_panel_comparison(
        archived.comparison,
        reference_id=archived.reference_id,
        target_id=archived.target_id,
        target_label="AdamW",
        suptitle=name,
        horizon=workload.horizon,
    )


def panel_n(i):
    """Panel for CELLS[i] -- so a notebook cell is one short line."""
    return panel(*CELLS[i])


def rank_lr_panel(
    ranks=(16, 32, 64, 128, 256),
    *,
    model="meta-llama/Llama-3.2-1B",
    data_key="openmath",
    model_label="Llama-3.2-1B",
):
    """Compatibility name for the canonical publication rank figure."""
    if model != "meta-llama/Llama-3.2-1B" or data_key != "openmath":
        raise ValueError("the publication rank figure is defined for Llama/openmath")
    from lora_playground.plotting.paper_figs import fig3
    from lora_playground.plotting.style import apply_notebook_style

    # paper_figs installs manuscript-sized rcParams when imported. Restore the
    # intentional notebook style before drawing, and do not overwrite the
    # manuscript's checked-in figure as a side effect of viewing this panel.
    apply_notebook_style()
    fig = fig3(ranks=tuple(ranks), figsize=(13, 6.0), save=False)
    plt.show()
    return fig


def ablation_panel(rank=256):
    """E2 leave-one-out at one rank."""
    from lora_playground.workloads import find_workload
    workload = find_workload("meta-llama/Llama-3.2-1B", "openmath", rank)
    return _records_figure(
        _arms.ABLATION_ARMS,
        workload,
        _arms.POLORA_LABEL,
        f"Component ablation — Llama-3.2-1B openmath, r={rank}",
        target_label=None,
    )


def derivation_ablation_panel(rank=256):
    """Which derivation premise carries the method: the matrix sign, or the
    exponent the metric is applied at."""
    from lora_playground.workloads import find_workload
    workload = find_workload("meta-llama/Llama-3.2-1B", "openmath", rank)
    return _records_figure(
        _arms.DERIVATION_ARMS,
        workload,
        _arms.POLORA_LABEL,
        "Matrix sign and metric exponent — "
        f"Llama-3.2-1B openmath, r={rank}",
    )


def precond_panel(rank=256, model="meta-llama/Llama-3.2-1B",
                  data_key="openmath", model_label="Llama-3.2-1B",
                  trusted_only=False, matched_revision=False):
    """The three `precond` branches: what fills (C_B, C_A). All three share one
    (P, Q), the same p, q updates and the same rho rule.

    Parameterized by cell so the same comparison can be read at another rank
    (C_B and C_A are r x r, so the slot has less to offer as r falls) or on
    another architecture, without a second copy of the figure.

    Cohort membership comes from recorded run semantics. The shared paper-view
    projection excludes pre-fix or unknown factorwise-slot implementations.
    """
    from lora_playground.workloads import find_workload
    workload = find_workload(model, data_key, rank)
    return _records_figure(
        # AdamW stays in the matched view. It was dropped here on the grounds
        # that its unversioned historical runs should not be mixed into a
        # matched cohort, but the cohort exists to make the three `precond`
        # branches comparable and the factorwise-slot fix cannot touch an
        # optimizer with no such branch -- so dropping it removed the scale the
        # branches are read against and left the panel with no baseline.
        dict(_arms.PRECOND_ARMS),
        workload,
        _arms.PRECOND_PRODUCT_LABEL,
        rf"What fills $C_B$ and $C_A$ — {model_label} {data_key}, $r={rank}$",
        target_label="AdamW",
        semantic_view=("precond_matched" if matched_revision else "precond"),
        measurement_semantics_revision=(
            MEASUREMENT_SEMANTICS_REVISION
            if matched_revision
            else _UNPINNED_MEASUREMENT_REVISION
        ),
    )


def msign_panel(rank=256):
    """The `msign` axis at both ends of `precond`: can the matrix sign be replaced
    by its diagonal (rownorm / colnorm) with the slot present, and with it gone?"""
    if rank != 256:
        raise ValueError("paper.msign.v1 is sealed only for rank 256")
    return _archive_figure(
        "paper.msign.v1",
        "Diagonal matrix sign — Llama-3.2-1B openmath, $r=256$",
    )


def magnitude_rule_panel(rank=256):
    """Naive rho = eta against the PoLoRA rule rho = eta/(smax(A)+smax(B))."""
    if rank != 256:
        raise ValueError("paper.magnitude_rule.v1 is sealed only for rank 256")
    return _archive_figure(
        "paper.magnitude_rule.v1",
        "Magnitude rule — naive vs. PoLoRA — "
        "Llama-3.2-1B openmath, $r=256$",
    )


def beta2_panel(rank=256):
    """Protagonist curvature_beta grid: the EMA horizon of the P, Q metric."""
    if rank != 256:
        raise ValueError("paper.polora_beta2.v1 is sealed only for rank 256")
    return _archive_figure(
        "paper.polora_beta2.v1",
        "PoLoRA $\\beta_2$ sweep — Llama-3.2-1B openmath, $r=256$",
    )


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

    NOT truncated, and AdamW is on the RIGHT panel only. The beta2=0.999 arms
    ran 2000 steps against 9000-step beta2=0.99 comparands, so every arm draws
    its own length and the legend carries each one's loss at step 2000 -- the
    largest step all five reached -- so the comparison can be read step-matched
    without the curves being cut. AdamW is dropped from the final-loss-vs-lr
    panel because its grid runs 1e-4..1e-3 against the polar arms' 1e-2..1e-1;
    including it stretches the log-x axis over three decades and squeezes every
    compared arm's minimum into one corner. It stays on the trajectory panel,
    where it is the scale the others are read against.

    The one-sided arms are the CONTROL, not padding: `curvature_beta` drives four
    EMAs, not one -- P_A/Q_B (factorwise only) at optim.py:2184-2186, 2200-2201,
    and Q/P (BOTH arms) at 2191-2195, 2202-2203. Raising it helps
    everything, so only a gap that shrinks MORE than the one-sided control moves
    isolates the r x r slot.
    """
    from lora_playground.workloads import find_workload
    workload = find_workload("meta-llama/Llama-3.2-1B", "openmath", rank)
    return _records_figure(
        _arms.PRECOND_BETA2_ARMS,
        workload,
        r"Identity, $\beta_2=0.99$",
        rf"Estimation noise in $C_B$, $C_A$ — "
        rf"$\beta_2$ by what fills them — Llama-3.2-1B openmath, $r={rank}$",
        target_label="AdamW",
        semantic_view="precond_beta2",
    )


def adamw_beta2_panel(rank=256):
    """AdamW beta2 control -- the negative control for the protagonist beta2 grid."""
    if rank != 256:
        raise ValueError("paper.adamw_beta2.v1 is sealed only for rank 256")
    return _archive_figure(
        "paper.adamw_beta2.v1",
        "AdamW $\\beta_2$ control — Llama-3.2-1B openmath, $r=256$",
    )


# --------------------------------------------------------------------------------------
# Per-step factor-rescaling diagnostics
# --------------------------------------------------------------------------------------
# These read `optim_step` rows, which the leaderboard path deliberately skips
# (`run_catalog.py:337` parses with include_optim_steps=False), so they go
# through `loading.load_optim_step_diagnostics` rather than `cell_runs`. Runs are
# selected by PREDICATE; the notebook cells these replaced listed group names and
# re-implemented log discovery and JSON decoding by hand.

_RESCALE_FIELDS = (
    "step", "sigma_max_A_median", "sigma_max_B_median", "balance_resid_median",
)

# The protagonist cell these diagnostics were recorded in.
_RESCALE_CELL = dict(
    model_name="meta-llama/Llama-3.2-1B",
    optimizer="kl-diag-polar-lora",
    data_dir=_dataset_predicate("openmath"),
)


def _longest_diagnostic_run(where, fields=_RESCALE_FIELDS):
    """The longest per-step diagnostic trace matching `where`, or None."""
    from lora_playground.plotting.loading import load_optim_step_diagnostics

    found = load_optim_step_diagnostics(where, list(fields))
    if not found:
        return None
    return max(found.values(), key=lambda arrays: len(arrays["step"]))


def factor_rescaling_by_rank_panel(ranks=(16, 256), lr=0.01):
    """Does the factors' operator-norm balance weaken at small rank?

    Left: the median ratio sigma_max(B)/sigma_max(A) per step, against 1.
    Right: the median BaLoRA balance residual. One curve per rank at a matched
    learning rate, so rank is the only thing that varies.
    """
    colours = canonical_colors([f"r={r}" for r in ranks])
    with plt.rc_context(NOTEBOOK_RCPARAMS):
        fig, (ax_ratio, ax_resid) = plt.subplots(
            1, 2, figsize=(13, 5.5), constrained_layout=True
        )
        ax_ratio.axhline(1, color="black", ls="--", lw=0.8, zorder=0)
        for rank in ranks:
            arrays = _longest_diagnostic_run({
                **_RESCALE_CELL, "lora_r": rank, "lr": lr,
                "cw_no_diag_curv": lambda v: not v,
                "cw_unpinned": lambda v: not v,
            })
            if arrays is None:
                print(f"no per-step diagnostics recorded at r={rank}, lr={lr:g}")
                continue
            colour = colours[f"r={rank}"]
            ratio = arrays["sigma_max_B_median"] / arrays["sigma_max_A_median"]
            ax_ratio.plot(arrays["step"], ratio, color=colour, label=f"$r={rank}$")
            ax_resid.plot(arrays["step"], arrays["balance_resid_median"],
                          color=colour, label=f"$r={rank}$")
        ax_ratio.set_xlabel("Training step")
        ax_ratio.set_ylabel(r"median $\|B\|_2/\|A\|_2$")
        ax_resid.set_xlabel("Training step")
        ax_resid.set_ylabel("median balance residual")
        ax_resid.set_ylim(0, 1.05)
        handles, labels = ax_ratio.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=r"LoRA rank $r$",
                       loc="outside lower center", ncol=len(labels),
                       frameon=False)
        fig.suptitle(
            rf"Factor rescaling by rank — Llama-3.2-1B openmath, $\eta={lr:g}$"
        )
    plt.show()


_RESCALE_BAND_FIELDS = (
    "step",
    "sigma_ratio_median", "sigma_ratio_min", "sigma_ratio_max",
    "sigma_max_A_median", "sigma_max_A_min", "sigma_max_A_max",
    "sigma_max_B_median", "sigma_max_B_min", "sigma_max_B_max",
    "balance_resid_median", "balance_resid_min", "balance_resid_max",
    "norm_A_median", "norm_B_median",
)

# The two arms these diagnostics contrast, by the fields that define them
# rather than by sweep name: PoLoRA against the arm with both the curvature
# metric and the magnitude rule removed (`arms.DOUBLE`).
_RESCALE_ARMS = {
    "PoLoRA": dict(cw_no_diag_curv=lambda v: not v,
                   cw_unpinned=lambda v: not v),
    r"Neither: $P=Q=I$, $\Delta A=-\eta W_A$": dict(
        cw_no_diag_curv=True, cw_unpinned=True),
}


def factor_rescaling_panel(rank=256):
    """How the two factors' scales evolve, with and without the two controls.

    Four panels, all per-pair medians with min-max bands over the LoRA pairs:
    the operator-norm ratio against 1, PoLoRA's two operator norms, the BaLoRA
    balance residual, and the Frobenius norms. Exploratory -- these fields
    describe the factors' scale, not the loss, so they characterize the
    mechanism rather than testing it.
    """
    traces = {}
    for label, extra in _RESCALE_ARMS.items():
        arrays = _longest_diagnostic_run(
            {**_RESCALE_CELL, "lora_r": rank, **extra}, _RESCALE_BAND_FIELDS,
        )
        if arrays is None:
            print(f"no per-step diagnostics recorded for {label!r} at r={rank}")
            continue
        traces[label] = arrays
    if not traces:
        print("no per-step diagnostics for either arm; nothing to draw.")
        return
    colours = canonical_colors(traces)

    def _band(ax, arrays, colour, stem, label=None):
        ax.fill_between(arrays["step"], arrays[f"{stem}_min"],
                        arrays[f"{stem}_max"], color=colour, alpha=0.15,
                        linewidth=0)
        ax.plot(arrays["step"], arrays[f"{stem}_median"], color=colour,
                label=label)

    with plt.rc_context(NOTEBOOK_RCPARAMS):
        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
        for label, arrays in traces.items():
            _band(ax[0, 0], arrays, colours[label], "sigma_ratio", label)
            _band(ax[1, 0], arrays, colours[label], "balance_resid", label)
            ax[1, 1].plot(arrays["step"], arrays["norm_A_median"],
                          color=colours[label], label=label)
            ax[1, 1].plot(arrays["step"], arrays["norm_B_median"],
                          color=colours[label], ls="--", label="_nolegend_")
        arm_handles, arm_labels = ax[0, 0].get_legend_handles_labels()
        ax[0, 0].axhline(1, color="black", ls=":", lw=0.8, zorder=0)
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_ylabel(r"$\|A\|_2/\|B\|_2$ per pair")

        protagonist = traces.get("PoLoRA")
        if protagonist is not None:
            factor_colours = canonical_colors([r"$\|A\|_2$", r"$\|B\|_2$"])
            for stem, name in (("sigma_max_A", r"$\|A\|_2$"),
                               ("sigma_max_B", r"$\|B\|_2$")):
                _band(ax[0, 1], protagonist, factor_colours[name], stem, name)
            ax[0, 1].legend(frameon=False)
        ax[0, 1].set_ylabel(r"PoLoRA $\|A\|_2$, $\|B\|_2$")

        ax[1, 0].axhline(0, color="black", ls=":", lw=0.8, zorder=0)
        ax[1, 0].set_ylim(0, 1.05)
        ax[1, 0].set_ylabel("balance residual")
        ax[1, 1].set_ylabel(r"$\|A\|_F$ solid, $\|B\|_F$ dashed")
        for axis in ax.flat:
            axis.set_xlabel("Training step")
        if arm_handles:
            fig.legend(arm_handles, arm_labels, loc="outside lower center",
                       ncol=len(arm_labels), frameon=False)
        fig.suptitle(
            rf"Factor scale over training — Llama-3.2-1B openmath, $r={rank}$"
        )
    plt.show()


def magnitude_rule_tracking_panel(rank=256):
    """The solved magnitude rule against the bounded one, per learning rate.

    Colour encodes the learning rate and linestyle the rule, legended as two
    separate keys rather than the cross-product. Both arms are selected on
    `cw_solved_rho`, the field that distinguishes them, so a renamed or added
    sweep is picked up; the notebook cell this replaces named three log groups
    and re-implemented log discovery and JSON decoding, and its bound arm was
    missing four of the seven learning rates that exist.
    """
    from matplotlib.lines import Line2D

    cell = {**_RESCALE_CELL, "lora_r": rank}
    rules = (
        ("Solved", {**cell, "cw_solved_rho": True}, "-"),
        ("Bounded", {**cell, "cw_solved_rho": lambda v: v in (None, False)}, "--"),
    )
    traces = {}
    for name, where, linestyle in rules:
        by_lr = {}
        for cfg, history in cell_runs(where):
            lr = cfg.get("lr")
            points = [(event["step"], event["eval_loss"]) for event in history
                      if event.get("eval_loss") is not None]
            if lr is None or not points:
                continue
            if lr not in by_lr or len(points) > len(by_lr[lr]):
                by_lr[lr] = points
        traces[name] = (by_lr, linestyle)

    learning_rates = sorted({lr for by_lr, _ in traces.values() for lr in by_lr})
    if not learning_rates:
        print("no runs recorded for either magnitude rule; nothing to draw.")
        return
    colours = canonical_colors([f"eta={lr:g}" for lr in learning_rates])

    with plt.rc_context(NOTEBOOK_RCPARAMS):
        fig, ax = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
        for name, (by_lr, linestyle) in traces.items():
            for lr, points in sorted(by_lr.items()):
                ax.plot([s for s, _ in points], [v for _, v in points],
                        ls=linestyle, color=colours[f"eta={lr:g}"])
        eta_key = ax.legend(
            handles=[Line2D([], [], color=colours[f"eta={lr:g}"],
                            label=rf"$\eta={lr:g}$") for lr in learning_rates],
            title=r"Learning rate $\eta$", loc="center left",
            bbox_to_anchor=(1.02, 0.66), frameon=False,
        )
        ax.add_artist(eta_key)
        ax.legend(
            handles=[Line2D([], [], color="black", ls=linestyle, label=name)
                     for name, (_by_lr, linestyle) in traces.items()],
            title="Magnitude rule", loc="center left",
            bbox_to_anchor=(1.02, 0.16), frameon=False,
        )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Evaluation loss")
        fig.suptitle(
            "Solved vs. bounded magnitude rule — "
            rf"Llama-3.2-1B openmath, $r={rank}$"
        )
    plt.show()

    for name, (by_lr, _linestyle) in traces.items():
        if not by_lr:
            print(f"{name}: no runs recorded.")
            continue
        best_lr, points = min(by_lr.items(), key=lambda kv: kv[1][-1][1])
        print(f"{name}: best eta={best_lr:g} final={points[-1][1]:.4f} "
              f"({len(by_lr)} learning rates)")
