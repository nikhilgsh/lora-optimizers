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

from pathlib import Path

import matplotlib.pyplot as plt

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
    pred_matches, variant_key_fn,
    ADAMW, AVGLOSS, DOUBLE, FLATOUT, HALFPOW, IMUON, LORARITE, MUON, NAIVEMAG,
    NOMAG, NOPRODUCT, NOSHAMPOO, ONESIDED, ONESIDED_DIAG, PROTO, PROTO_DIAG,
)

# The E1 cell set (paper/e1_coverage_fill.md). data_key is substring-matched on data_dir.
CELLS = [
    ("OLMo-2-1B opc r256",        "allenai/OLMo-2-0425-1B",     "opc",      256),
    ("Qwen2.5-1.5B opc r256",     "Qwen/Qwen2.5-1.5B",          "opc",      256),
    ("Llama-3.2-1B opc r256",     "meta-llama/Llama-3.2-1B",    "opc",      256),
    ("Llama-3-8B opc r256",       "meta-llama/Meta-Llama-3-8B", "opc",      256),
    ("Qwen2.5-1.5B bengali r256", "Qwen/Qwen2.5-1.5B",          "bengali",  256),
    ("OLMo-2-1B openmath r256",    "allenai/OLMo-2-0425-1B",     "openmath", 256),
    ("Qwen2.5-1.5B openmath r256", "Qwen/Qwen2.5-1.5B",          "openmath", 256),
    ("Llama-3-8B openmath r256",   "meta-llama/Meta-Llama-3-8B", "openmath", 256),
    ("Llama-3.2-1B openmath r16",  "meta-llama/Llama-3.2-1B",    "openmath",  16),
    ("Llama-3.2-1B openmath r32",  "meta-llama/Llama-3.2-1B",    "openmath",  32),
    ("Llama-3.2-1B openmath r64",  "meta-llama/Llama-3.2-1B",    "openmath",  64),
    ("Llama-3.2-1B openmath r128", "meta-llama/Llama-3.2-1B",    "openmath", 128),
    ("Llama-3.2-1B openmath r256", "meta-llama/Llama-3.2-1B",    "openmath", 256),
]

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
    sig = logs_signature(str(ROOT / "logs"))
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


def _figure(arms, common, ref_label, suptitle, target_label="AdamW", drop_empty=True):
    """Every panel in this module funnels through here.

    Passing prefetched_runs AND variant_key is what arms the ``assert_label_discriminates``
    guard inside compare_variants_figure -- the per-variant loading path does not run it,
    and that is how two arms silently merged for weeks. Do not add a panel that skips it.
    """
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
    in_flight = {k: n for k, v in arms.items()
                 if (n := _max_step(v, common)) is not None and n < HORIZON}
    if in_flight:
        print(f"in flight (trajectory panel only, absent from the loss-vs-lr panel "
              f"and the summary table until {HORIZON} steps): "
              + ", ".join(f"{k} @{n}" for k, n in in_flight.items()))
    fig, _t, sdf = compare_variants_figure(
        arms, common_where=common, ref_label=ref_label,
        logs_root=str(ROOT / "logs"), sigma_ref=SIGMA, max_steps=HORIZON,
        allow_partial=True, allow_custom_labels=True, target_label=target_label,
        suptitle=suptitle,
        prefetched_runs=cell_runs(common), variant_key=variant_key_fn(common, arms))
    plt.show()
    return sdf


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


def adamw_beta2_panel(rank=256):
    """AdamW beta2 control -- the negative control for the protagonist beta2 grid."""
    return _figure(_arms.ADAMW_BETA2_ARMS, om(rank), "AdamW (beta2=0.999)",
                   f"AdamW beta2 control - Llama-3.2-1B openmath r{rank}",
                   target_label=None)
