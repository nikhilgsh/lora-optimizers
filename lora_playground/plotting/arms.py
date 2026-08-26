"""Fail-closed arm predicates for figure code.

The problem this exists to fix
------------------------------
A hand-typed arm predicate is an ALLOWLIST of the fields it pins::

    PROTO = {'optimizer': 'kl-diag-polar-lora', 'cw_nesterov': True, ...}

Every field it does not mention is unconstrained. So the day a new ablation
flag is added to ``OptimizerConfig`` and a sweep sets it, every pre-existing
predicate silently stops discriminating: runs with the flag ON and OFF both
match, they land in one display-label bucket, and the figure keeps whichever
has the lowest loss. The predicate fails OPEN. This is not hypothetical — it
happened three times in one day, on ``global_batch_size`` (the small-batch
sweep joining the protagonist series), on ``cw_no_rr_precond`` (the
``e2_noprod_norr`` sweep joining the factorwise arm of the r x r 2x2), and on
``beta2`` (the AdamW beta2 grid joining the AdamW baseline).

The inversion
-------------
`arm` defines an arm by the fields it DIFFERS on. Everything else is pinned
automatically to its default, derived from ``OptimizerConfig`` rather than
hand-typed::

    PROTO = arm('kl-diag-polar-lora', cw_nesterov=True, ...)   # pins all 70

Adding a field to ``OptimizerConfig`` then pins it in every arm on the next
import. A sweep that sets the new flag no longer matches the old arms, so the
failure mode becomes "the arm renders with no data" — loud and obvious —
instead of "two arms silently merge". It fails CLOSED.

What is pinned, and what is deliberately not
--------------------------------------------
``PINNED_FIELDS`` = ``OptimizerConfig`` fields, minus
``manifest.SERIES_AXIS_FIELDS`` (``lr``/``seed``/``max_steps``/the diagnostic
toggles — runs are ALLOWED to differ on these within one series, and pinning
them would split series that should be averaged), minus the two fields
train.py exposes no CLI flag for (``muon_alpha``, ``muon_rank``: constructor
knobs only, so no run cfg ever carries them and pinning them would match
nothing). See `check_config_fields_pinnable` — a test asserts that exclusion
list stays exactly those two, so a NEW config field without a CLI flag trips
CI rather than silently falling out of the pin set.

Workload identity — ``model_name`` / ``data_dir`` / ``lora_r`` — is not an
optimizer hyperparameter and stays in the caller's ``common_where``. The
non-``OptimizerConfig`` train.py flags an arm needs (``global_batch_size``,
``lora_init_b``, ``max_steps``) are passed as explicit overrides, and are
checked against train.py's parser so a typo raises instead of silently
pinning nothing.

Why "absent" is not a case to handle
------------------------------------
``_pred_matches`` / ``loader._matches`` treat a missing key as no-match, so
pinning a field the loader does not backfill would silently drop old runs.
It does backfill: ``loader._wrapped_postprocess`` and ``loader._enrich_cfg``
both run ``for k, v in _argparse_defaults().items(): if cfg.get(k) is None:
cfg[k] = ...``, so every train.py CLI flag is present on every cfg the loader
returns, at the value the run actually executed with. Measured over the runs
matched by all 25 arm dicts across the 19 figure calls in
``paper/paper_plots.ipynb``: all 70 pinned fields present on every run; the
only two absent anywhere were ``muon_alpha``/``muon_rank``, which is why they
are excluded. `check_pinned_defaults_agree_with_cli` additionally asserts the
dataclass default and the argparse default agree for every pinned field —
they must, or "pin to the dataclass default" and "what the loader backfilled"
would name different values.
"""
from __future__ import annotations

from dataclasses import fields as _dataclass_fields
from functools import lru_cache

from ..optim_config import OptimizerConfig

# `manifest` and `loader` both import this package, so they are imported
# lazily inside the helpers below rather than at module scope.


def _config_defaults() -> dict:
    return {f.name: f.default for f in _dataclass_fields(OptimizerConfig)}


@lru_cache(maxsize=1)
def _cli_defaults() -> dict:
    from ..loader import _argparse_defaults
    return dict(_argparse_defaults())


@lru_cache(maxsize=1)
def _pinned() -> dict:
    """``{field: default}`` for every field `arm` pins."""
    from ..manifest import SERIES_AXIS_FIELDS
    cli = _cli_defaults()
    return {k: v for k, v in _config_defaults().items()
            if k not in SERIES_AXIS_FIELDS and k in cli}


def PINNED_FIELDS() -> frozenset:
    """The field names `arm` pins (call, don't cache — it tracks the dataclass)."""
    return frozenset(_pinned())


def check_config_fields_pinnable() -> dict:
    """Report which ``OptimizerConfig`` fields fall out of the pin set and why.

    ``{"series_axis": [...], "no_cli_flag": [...]}``. ``no_cli_flag`` is the
    dangerous bucket: such a field is invisible in every run cfg, so an arm
    cannot discriminate on it at all.
    """
    from ..manifest import SERIES_AXIS_FIELDS
    cli = _cli_defaults()
    defaults = _config_defaults()
    return {
        "series_axis": sorted(k for k in defaults if k in SERIES_AXIS_FIELDS),
        "no_cli_flag": sorted(k for k in defaults
                              if k not in SERIES_AXIS_FIELDS and k not in cli),
    }


def check_pinned_defaults_agree_with_cli() -> dict:
    """``{field: (dataclass_default, argparse_default)}`` for every pinned
    field whose two defaults disagree. Must be empty: `arm` pins the dataclass
    default, the loader backfills the argparse default, and a disagreement
    means an arm silently stops matching the runs that took the default."""
    cli = _cli_defaults()
    return {k: (v, cli[k]) for k, v in _pinned().items() if cli[k] != v}


def arm(optimizer: str, **overrides) -> dict:
    """A ``where``-predicate for one arm: ``optimizer`` plus every pinned
    ``OptimizerConfig`` field at its default, with ``overrides`` applied last.

    An override may name a pinned field (the arm differs there), a
    ``SERIES_AXIS_FIELDS`` member (a deliberate restriction — ``max_steps=9000``
    to exclude a short lr pilot), or any other train.py CLI flag
    (``global_batch_size``, ``lora_init_b``). A name that is none of those is a
    typo and raises: silently pinning nothing is the failure this module
    exists to prevent.

    A tuple/list/set value is a membership predicate, matching the loader's
    ``where`` semantics. Use it only for a field the optimizer genuinely
    ignores — see ``ADAMW``'s ``precond_method``.
    """
    known = set(_cli_defaults()) | set(_config_defaults())
    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        raise ValueError(
            f"arm({optimizer!r}) got override(s) {unknown} that are neither "
            f"OptimizerConfig fields nor train.py CLI flags — a predicate on "
            f"an unknown field pins nothing. Check the spelling against "
            f"lora_playground/optim_config.py or train.py's parser."
        )
    return {"optimizer": optimizer, **_pinned(), **overrides}


# ─────────────────────────────────────────────────────────────────────────────
# The paper's arms (paper/paper_plots.ipynb, Llama-3.2-1B openmath + the E1 set)
# ─────────────────────────────────────────────────────────────────────────────

# Fields every curvature-whiten production sweep moves OFF the dataclass
# default. Not guessed — read off a protagonist run's logged cfg
# (logs/e1_kldiag_llama32_openmath_r256_bw) and diffed against
# OptimizerConfig: precond_delta 1e-6 -> 1e-4, precond_method None ->
# 'gram_ns', precond_refresh_every 1 -> 10, higham_iters 10 -> 8,
# muon_ns_steps 5 -> 8, polar_method 'ns' -> 'polar_express', cw_nesterov
# False -> True. Every other pinned field is at its default on that run.
CW_PRODUCTION = dict(
    cw_nesterov=True,
    polar_method="polar_express",
    precond_method="gram_ns",
    precond_delta=1e-4,
    precond_refresh_every=10,
    higham_iters=8,
    muon_ns_steps=8,
)

# Baselines. `precond_method` is inert for AdamW — LoRAPlusAdamW.__init__ takes
# (model, lr, lora_plus_multiplier, betas, eps, weight_decay, adapter_name) and
# no preconditioner — but six older adamw groups inherited
# `--precond_method higham` from a shared sweep wrapper while the newer ones
# log None, so the baseline must admit both or it drops half its runs.
ADAMW = arm("adamw", beta2=0.999, precond_method=(None, "higham"))
# max_steps pins exclude the 1000-step lr pilots (ranking-only, never measured).
IMUON = arm("imuon-lora", max_steps=9000)
MUON = arm("muon-lora", max_steps=9000,
           muon_ns_steps=8, polar_method="polar_express")
LORARITE = arm("lora-rite", max_steps=9000)

# Protagonist. global_batch_size is a train.py flag, not an OptimizerConfig
# field, so it is pinned explicitly: e2_beta2_smallbatch runs this SAME
# optimizer at the same (lr, r, corpus) with 1x1 = 2048 tokens/step instead of
# 4x4 = 32768. Pin global_batch_size, NOT batch_size — every 9000-step paper
# run is global 16, but the composition differs (Llama-3-8B is 2x8).
PROTO = arm("kl-diag-polar-lora", **CW_PRODUCTION, global_batch_size=16)

# E2 leave-one-out arms: the protagonist with one control removed.
NOSHAMPOO = {**PROTO, "cw_no_diag_curv": True}      # w/o curvature control
NOMAG = {**PROTO, "cw_unpinned": True, "lora_init_b": "symmetric"}  # w/o magnitude control
DOUBLE = {**PROTO, "cw_no_diag_curv": True, "cw_unpinned": True,
          "lora_init_b": "symmetric", "max_steps": 9000}  # both removed = the LoRA-Muon step

# Derivation ablations: each is its own registered optimizer, so the optimizer
# string selects it; CW_PRODUCTION holds the rest of the config at the
# protagonist's so the only difference is the derivation premise removed.
AVGLOSS = arm("kl-diag-lora", **CW_PRODUCTION)               # -per-sample: no msign, metric^-1
HALFPOW = arm("kl-diag-flatout-lora", **CW_PRODUCTION)       # -msign at half metric power
FLATOUT = arm("kl-diag-polar-flatout-lora", **CW_PRODUCTION)  # -outer un-whiten: msign only
NOPRODUCT = arm("kl-shampoo-polar-lora", **CW_PRODUCTION)    # -product: per-factor metric

# The r x r metric slot 2x2: {slot = B^T P B or I} x {d-side diagonals shared
# or per-factor}. NORR and NOPRODUCT_NORR are the two `cw_no_rr_precond=True`
# corners; PROTO and NOPRODUCT are the False corners, pinned automatically.
NORR = arm("kl-diag-polar-lora", **CW_PRODUCTION, cw_no_rr_precond=True)
NOPRODUCT_NORR = {**NOPRODUCT, "cw_no_rr_precond": True}

# Magnitude rule: rho = eta flat instead of rho = eta/(smax A + smax B).
NAIVEMAG = {**PROTO, "cw_no_radius": True}


def beta2_arms(base: dict, key: str, values, ref_value, ref_label: str) -> dict:
    """``{label: predicate}`` for a second-moment-decay grid over ``key``,
    with the shipped value carrying ``ref_label``."""
    return {(ref_label if v == ref_value else f"{key}={v}"): {**base, key: v}
            for v in values}


_BETA2_GRID = [0.81, 0.9090, 0.9564, 0.9791, 0.99]
PROTO_BETA2_ARMS = beta2_arms(PROTO, "curvature_beta", _BETA2_GRID, 0.99,
                              "Polar-LoRA (shipped, b2=0.99)")
# The AdamW control sweeps `beta2` itself, so the grid cannot inherit ADAMW's
# beta2 pin; `precond_method` still admits both values (see ADAMW).
ADAMW_BETA2_ARMS = beta2_arms(
    arm("adamw", precond_method=(None, "higham")), "beta2",
    _BETA2_GRID + [0.999], 0.999, "AdamW (shipped, b2=0.999)")


# Per-figure arm dicts (label -> predicate), in legend order.
PANEL_ARMS = {
    "AdamW": ADAMW,
    "Polar-LoRA (kl-diag)": PROTO,
    "iMuon": IMUON,
    "Muon (naive)": MUON,
    "LoRA-RITE": LORARITE,
    "w/o curvature+magnitude (LoRA-Muon step)": DOUBLE,
}
ABLATION_ARMS = {
    "Polar-LoRA (kl-diag)": PROTO,
    "w/o curvature control": NOSHAMPOO,
    "w/o magnitude control": NOMAG,
    "w/o curvature+magnitude (LoRA-Muon step)": DOUBLE,
}
DERIVATION_ARMS = {
    "AdamW": ADAMW,
    "PoLoRA: rxr=B^T P B, shared P,Q": PROTO,
    "no msign, metric^-1 (averaged loss)": AVGLOSS,
    "no msign, metric^-1/2": HALFPOW,
    "no outer un-whiten: msign only": FLATOUT,
}
RR_SLOT_ARMS = {
    "AdamW": ADAMW,
    "PoLoRA: rxr=B^T P B, shared P,Q": PROTO,
    "rxr = I, shared P,Q": NORR,
    "factorwise: own P_A,Q_A / P_B,Q_B": NOPRODUCT,
    "factorwise + rxr = I": NOPRODUCT_NORR,
}
MAGNITUDE_RULE_ARMS = {
    "PoLoRA: rho = eta/(smax A + smax B)": PROTO,
    "naive: rho = eta": NAIVEMAG,
    "AdamW": ADAMW,
}

# Every arm dict the regression test walks. Adding a figure means adding its
# dict here, so the discrimination guard covers it too.
ALL_ARM_DICTS = {
    "PANEL_ARMS": PANEL_ARMS,
    "ABLATION_ARMS": ABLATION_ARMS,
    "DERIVATION_ARMS": DERIVATION_ARMS,
    "RR_SLOT_ARMS": RR_SLOT_ARMS,
    "MAGNITUDE_RULE_ARMS": MAGNITUDE_RULE_ARMS,
    "PROTO_BETA2_ARMS": PROTO_BETA2_ARMS,
    "ADAMW_BETA2_ARMS": ADAMW_BETA2_ARMS,
}
