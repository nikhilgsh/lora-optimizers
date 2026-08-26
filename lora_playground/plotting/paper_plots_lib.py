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

    P.rr_slot_panel(256)

``import ... as P`` and not ``from ... import *``: autoreload re-executes this module on
edit, so ``P.PROTO`` is always current, whereas a name star-imported into the notebook
namespace stays bound to the old object and goes stale exactly the way the cells did.

ARM PREDICATES AND WHY THEY PIN SO MUCH
---------------------------------------
An arm is a ``where`` dict handed to ``compare_variants_figure``. Every field it does NOT
mention is a field two different sweeps may disagree on while both still match, in which
case the figure silently keeps whichever run has the lower loss. That has bitten three
times: ``global_batch_size`` (the 2048-token sweep matching the 32768-token protagonist),
``cw_no_rr_precond`` (the r x r = I corner matching the arm it is supposed to contrast
with), and ``beta2`` (the AdamW grid matching the AdamW baseline). Each time the fix was
to pin one more field by hand.

Pinning by hand fails open -- a NEW ablation flag is absent from every existing arm, so
every existing arm silently stops discriminating the moment a sweep sets it. The
durable fix derives the pinned set from ``OptimizerConfig``'s fields and defaults so a
new flag is pinned everywhere automatically (then a sweep that sets it renders an EMPTY
arm, which is loud, instead of merging two arms, which is silent). That lives in
``lora_playground/plotting/arms.py``; when it lands, the dicts below become calls into
it and this docstring section goes away.

Until then the guard is ``assert_label_discriminates``, which
``compare_variants_figure`` runs whenever ``prefetched_runs``/``variant_key`` are passed
-- which is why every panel here passes them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from lora_playground.loader import load_runs
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
from .arms import (  # noqa: F401,E402
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
_RUNS_CACHE: dict[str, list] = {}


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
    """Runs for one cell, memoised per distinct `where` within a kernel.

    `refresh=True` forces a re-read. Do NOT wire it on by default for live sweeps: an
    unconditional refresh makes every panel pay the full query again, which is what made
    the live-tracking figures the slowest cells in the notebook. Call
    ``clear_runs_cache()`` once when a sweep has advanced and you want fresh numbers.
    """
    key = repr(sorted((k, _fingerprint(v)) for k, v in where.items()))
    if refresh or key not in _RUNS_CACHE:
        _RUNS_CACHE[key] = load_runs(where=where, warn_cross_commit=False, quiet=True)
    return _RUNS_CACHE[key]


def clear_runs_cache():
    """Drop the memo so the next panel re-reads from disk."""
    _RUNS_CACHE.clear()


def pred_matches(cfg, pred):
    """In-memory predicate check mirroring ``loader._matches``: literal equality,
    list-like membership, or callable truthiness per field. A run missing a referenced
    field does not match."""
    for k, v in pred.items():
        if k not in cfg:
            return False
        c = cfg[k]
        if callable(v):
            if not v(c):
                return False
        elif isinstance(v, (list, set, tuple, frozenset)):
            if c not in v:
                return False
        elif c != v:
            return False
    return True


def variant_key_fn(common, arms):
    """``compare_variants_figure(variant_key=...)`` selecting among `arms`
    (label -> extra-where dict) for cfgs matching `common`. Equivalent to a per-arm
    ``load_runs(where={**common, **extra})``, applied in memory to one narrow query."""
    def variant_key(cfg):
        if not pred_matches(cfg, common):
            return None
        for label, extra in arms.items():
            if pred_matches(cfg, extra):
                return label
        return None
    return variant_key


def om(rank):
    """common_where for Llama-3.2-1B / openmath at a rank -- the ablation cell."""
    return dict(model_name="meta-llama/Llama-3.2-1B", lora_r=rank,
                data_dir=(lambda d: "openmath" in str(d)))


def has(where, rank):
    """Is there any run for this arm at this rank? Used to drop empty arms."""
    pred = {**where, **om(rank)}
    return any(pred_matches(cfg, pred) for cfg, _h in cell_runs(om(rank)))


def _figure(arms, common, ref_label, suptitle, target_label="AdamW", drop_empty=True):
    """Every panel in this module funnels through here.

    Passing prefetched_runs AND variant_key is what arms the ``assert_label_discriminates``
    guard inside compare_variants_figure -- the per-variant loading path does not run it,
    and that is how two arms silently merged for weeks. Do not add a panel that skips it.
    """
    if drop_empty:
        rank = common.get("lora_r")
        missing = [k for k, v in arms.items() if not has(v, rank)]
        if missing:
            print("no data yet (omitted):", ", ".join(missing))
        arms = {k: v for k, v in arms.items() if k not in missing}
    fig, _t, sdf = compare_variants_figure(
        arms, common_where=common, ref_label=ref_label,
        logs_root=str(ROOT / "logs"), sigma_ref=SIGMA, max_steps=9000,
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


def precond_panel(rank=256):
    """The three `precond` branches: what fills (C_B, C_A). All three share one
    (P, Q), the same p, q updates and the same rho rule."""
    arms = {"AdamW": ADAMW,
            "product: C_B=B^T P B, C_A=A Q A^T": PROTO,
            "one-sided: C_B=C_A=I": ONESIDED,
            "factorwise: C_B=P_A, C_A=Q_B": NOPRODUCT}
    return _figure(arms, om(rank), "product: C_B=B^T P B, C_A=A Q A^T",
                   f"The r x r metric slot - Llama-3.2-1B openmath r{rank}")


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
    arms = {"AdamW": ADAMW,
            "PoLoRA: rho = eta/(smax A + smax B)": PROTO,
            "naive: rho = eta": NAIVEMAG}
    return _figure(arms, om(rank), "PoLoRA: rho = eta/(smax A + smax B)",
                   f"Magnitude rule: naive vs PoLoRA - Llama-3.2-1B openmath r{rank}")


def b2_arms(base, key, values, ref_value, ref_label):
    """{label: predicate} for a beta2 grid, with the shipped value as the reference."""
    return {(ref_label if v == ref_value else f"{key}={v}"): {**base, key: v}
            for v in values}


B2_GRID = [0.81, 0.9090, 0.9564, 0.9791, 0.99]


def beta2_panel(rank=256):
    """Protagonist curvature_beta grid: the EMA horizon of the P, Q metric."""
    arms = b2_arms(PROTO, "curvature_beta", B2_GRID, 0.99,
                   "PoLoRA (curvature_beta=0.99)")
    return _figure(arms, om(rank), "PoLoRA (curvature_beta=0.99)",
                   f"Protagonist beta2 sweep - Llama-3.2-1B openmath r{rank}",
                   target_label=None)


def adamw_beta2_panel(rank=256):
    """AdamW beta2 control -- the negative control for the protagonist beta2 grid."""
    arms = b2_arms({"optimizer": "adamw"}, "beta2", B2_GRID + [0.999], 0.999,
                   "AdamW (beta2=0.999)")
    return _figure(arms, om(rank), "AdamW (beta2=0.999)",
                   f"AdamW beta2 control - Llama-3.2-1B openmath r{rank}",
                   target_label=None)
