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
sweep joining the protagonist series), on the since-retired
``cw_no_rr_precond`` (an ``e2_noprod_norr`` sweep joining the factorwise arm),
and on ``beta2`` (the AdamW beta2 grid joining the AdamW baseline).

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


def pred_matches(cfg: dict, pred: dict) -> bool:
    """In-memory predicate check mirroring ``loader._matches``: literal equality,
    list-like membership, or callable truthiness per field. A run missing a
    referenced field does not match.

    Lives here, not in paper_plots_lib, so `paper_figs.py` can select arms with the
    same matcher without importing the notebook-support module. Pure dict logic —
    it adds no import to this leaf module.
    """
    return all(field_matches(cfg, k, v) for k, v in pred.items())


def field_matches(cfg: dict, field: str, want) -> bool:
    """Does ONE pinned field match? The single definition of a pin's semantics.

    Factored out because `paper_plots_lib._closest_arm` needs per-field verdicts
    to say WHICH pin excluded a run, and hand-rolling that comparison there gave
    two matchers that drifted: adding the list-vs-list branch below to
    `pred_matches` and not to the copy made the coverage diagnostic report
    `target_module_names: arm wants [], run has []` as a mismatch on a field
    that matches. Anything needing per-field detail calls this, so the diagnostic
    can never disagree with the predicate it explains.
    """
    if field not in cfg:
        return False
    c = cfg[field]
    if callable(want):
        return bool(want(c))
    if isinstance(want, (list, set, tuple, frozenset)):
        # Membership is for a SCALAR cfg value against a set of allowed ones
        # (`ADAMW`'s `precond_method=(None, "higham")`). When the cfg value is
        # itself list-like the pin cannot mean membership -- a list can only
        # equal a list -- so compare directly. Without this branch a field that
        # genuinely holds a list is unpinnable: `target_module_names=[]` was read
        # as "match nothing", silently rejecting every run, which is how a
        # group-derived predicate matched 0 of its own 4 runs.
        if isinstance(c, (list, set, tuple, frozenset)):
            return list(c) == list(want)
        return c in want
    return c == want


def arm_from_runs(cfgs: list[dict]) -> dict:
    """Derive an arm predicate from the runs that ACTUALLY RAN.

    Why this exists
    ---------------
    `arm()` fails closed field by field, but nothing made the READ path track
    the WRITE path: `scripts/sweep/*.sh` says what ran and `arms.py` separately
    re-declares what to look for, so a sweep nobody remembered to declare an arm
    for renders as absent rather than erroring. Measured on this repo: 361 of
    453 manifested groups in `logs/` are claimed by no arm in `ALL_ARM_DICTS`.

    The derivation removes the declaration. An arm is the fields CONSTANT across
    a set of runs, minus `manifest.SERIES_AXIS_FIELDS` (`lr`/`seed`/`max_steps`
    — the axes a series is allowed to vary on, and pinning them would split a
    series that should be averaged). Verified across every multi-run group on
    disk: the derived predicate matches all of its own runs in 409 of 409.

    LIMIT, and it is why this is not yet wired into any panel. A field is
    pinned only when it happens to AGREE across the set, so the derivation
    cannot tell a deliberate axis from an accident: pooling the small-batch
    sweep with the protagonist drops `global_batch_size` from the predicate
    precisely because they disagree on it, which is the first of the three
    incidents this module's header cites. Counting pins (131 against `arm()`'s
    ~70) measures the wrong thing -- fail-closed is about which fields are
    pinned when they DISAGREE. The honest source for "which fields did this
    sweep mean to vary" is the sweep's own `params_file` and `sweep_script`,
    recorded in `logs/<group>/run_info/meta.json`.

    A derived predicate deliberately also matches runs in OTHER groups that
    share the configuration. That is the point: one arm should span the sweeps
    that ran the same thing, which is what `NOPRODUCT` was hand-written to do
    when it had to admit both `kl-diag-polar-lora --precond factorwise` and the
    older `kl-shampoo-polar-lora`.
    """
    if not cfgs:
        raise ValueError("cannot derive an arm from zero runs")
    from ..manifest import SERIES_AXIS_FIELDS
    shared = set(cfgs[0])
    for c in cfgs[1:]:
        shared &= set(c)
    pred = {}
    for k in sorted(shared):
        if k in SERIES_AXIS_FIELDS or k.startswith("_"):
            continue
        v = cfgs[0][k]
        if isinstance(v, dict):
            continue          # unhashable and never a discriminator
        if len({repr(c.get(k)) for c in cfgs}) == 1:
            pred[k] = v
    return pred


def variant_key_fn(common: dict, arms: dict):
    """``cfg -> label`` selecting the FIRST arm in `arms` whose predicate matches,
    for cfgs that also match `common`; None when nothing matches.

    This is the one place a run is mapped to a display label. Passing arms built by
    `arm()` is what makes the mapping fail closed: a run carrying a field no arm
    pins falls out of every arm and renders nowhere, instead of joining whichever
    arm happened to omit that field.
    """
    def variant_key(cfg):
        if not pred_matches(cfg, common):
            return None
        for label, extra in arms.items():
            if pred_matches(cfg, extra):
                return label
        return None
    return variant_key


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
    inert = _inert_fields(optimizer)
    pins = {k: v for k, v in _pinned().items() if k not in inert}
    return {"optimizer": optimizer, **pins, **overrides}


@lru_cache(maxsize=None)
def _inert_fields(optimizer: str) -> frozenset:
    """Pinned fields the named optimizer's constructor never receives.

    Pinning such a field is the bug that shipped three times. `arm()` used to
    pin ALL ~76 `OptimizerConfig` fields at their dataclass defaults, including
    ones the arm's own optimizer cannot read — so any run that happened to log a
    non-default value for an INERT field was silently dropped, and the arm drew
    an empty series indistinguishable from "no data yet". Measured instances:

      - `cw_nesterov` on `ADAMW`: `LoRAPlusAdamW.__init__` takes no such
        argument, yet every adamw run at 5 of the 13 panel cells logs False
        against the pinned True. Those cells rendered with NO baseline and
        `leaderboard_rows` returned a NaN speed target.
      - `cw_nesterov` on `MUON` and `IMUON`: same shape, 6 runs each.
      - `muon_ns_steps` on `ADAMW`: 5 runs.

    Each was previously patched one arm at a time by widening the pin to a
    membership tuple. Deriving the set instead removes the class: a field the
    constructor does not accept is provenance, not an axis, so it is not pinned
    at all and every logged value is admitted.

    Derived from `optim_specs.REGISTRY[optimizer].cls.__init__`, which resolves
    for all 79 registered names. `spec.skip` is included because those fields are
    deliberately withheld from the constructor for that variant, so they stay at
    the class default rather than tracking the run. An unregistered name, or one
    built by a custom `build` callable whose real consumer cannot be introspected,
    returns the empty set — i.e. falls back to the old pin-everything behaviour,
    which is conservative: it can drop runs, never admit wrong ones.
    """
    import inspect
    try:
        from ..optim_specs import ALIAS, REGISTRY, _forwardable_params
    except Exception:
        return frozenset()
    spec = REGISTRY.get(optimizer)
    if spec is None:
        return frozenset()
    if spec.build is not None:
        # A custom `build` callable (imuon, sgd, adafactor) constructs the
        # optimizer its own way, so signature introspection does not apply.
        # Read the SOURCE for `config.<field>` instead: those are the only run
        # values that can reach it. `_build_imuon`, for instance, reads
        # `config.lr` and hardcodes everything else -- it runs the authors' code
        # verbatim -- so no pinned field can distinguish two of its runs, and
        # pinning any of them only drops runs (measured: 6, on cw_nesterov).
        import re
        try:
            src = inspect.getsource(spec.build)
        except (OSError, TypeError):
            return frozenset()
        # Bail out if `config` is used other than as `config.<attr>` (e.g. handed
        # to a helper), since then the reads are not visible here.
        body = "\n".join(src.split("\n")[1:])          # drop the def line
        if re.search(r"\bconfig\b(?!\s*\.)", body):
            return frozenset()
        read = set(re.findall(r"\bconfig\.(\w+)", body))
        return frozenset(k for k in _pinned() if k not in read)
    if spec.cls is None:
        return frozenset()
    try:
        params = list(_forwardable_params(spec.cls))
    except (TypeError, ValueError):
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return frozenset()          # **kwargs absorbs anything; cannot tell

    # Mirror `build_from_spec`'s forwarding, which is the only authority on which
    # CONFIG FIELD reaches the constructor. Introspecting the signature alone is
    # NOT enough and gets it wrong in the direction that matters: `betas` is
    # PACKED from beta1 and beta2, so a signature-only rule called both inert and
    # would have stopped ADAMW pinning beta2 -- reopening the exact incident this
    # module was written for, where the AdamW beta2 grid merged into the baseline.
    reachable = set()
    for prm in params:
        n = prm.name
        if n in (spec.fixed or {}):
            continue                                  # a constant, not the run's value
        if n == "betas":
            reachable |= {"beta1", "beta2"}           # packed, not passed through
            continue
        if n == "picard_iters":
            reachable |= {"picard_iters_override"}
            continue
        if n in (spec.defaults or {}):
            continue                                  # per-variant constant
        fld = (spec.alias or {}).get(n) or ALIAS.get(n, n)
        if fld in (spec.skip or set()):
            continue                                  # withheld on purpose
        reachable.add(fld)
    return frozenset(k for k in _pinned() if k not in reachable)


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

# Baselines. `precond_method` and `cw_nesterov` are both inert for AdamW —
# LoRAPlusAdamW.__init__ takes (model, lr, lora_plus_multiplier, betas, eps,
# weight_decay, adapter_name) and neither a preconditioner nor a Nesterov flag —
# but different sweep wrappers logged different values for each, so the baseline
# must admit both or it drops runs. `precond_method`: six older adamw groups
# inherited `--precond_method higham` from a shared wrapper while the newer ones
# log None. `cw_nesterov`: measured, every adamw run at the Llama-3.2-1B /
# openmath / r16 cell logs False against the `arm()` default True, so pinning it
# left `precond_panel(16)` with NO AdamW row and `leaderboard_rows` with a NaN
# speed target — the panel still rendered, just with no baseline to speak of.
# Any field an optimizer never reads is a provenance record, not an axis: admit
# every logged value rather than pinning one.
# `precond_method` and `cw_nesterov` used to be widened here to membership
# tuples, because pinning either dropped runs: LoRAPlusAdamW reads NEITHER, and
# different wrappers logged different values. `arm()` now derives its pin set
# from what `build_from_spec` actually forwards to the constructor, so an inert
# field is not pinned at all and every logged value is admitted. The widening is
# therefore unnecessary -- and its absence is what makes the fix general rather
# than one patch per arm. beta2 stays pinned by derivation (LoRAPlusAdamW takes
# `betas`), which is what keeps the AdamW beta2 grid out of the baseline.
ADAMW = arm("adamw")
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
# `precond` is stated explicitly on every curvature-whiten arm rather than left
# to the pin. The dataclass/CLI default is None ("inherit the optimizer's own
# setting"), but `loader._backfill_precond` resolves every run of this family to
# the branch it actually ran — "product" here — so a None pin would match none of
# them. Stating it also makes the arm table read as the three-branch selection it
# is: (C_B, C_A) = (B^T P B, A Q A^T) for product, (I, I) for one-sided,
# (P_A, Q_B) for factorwise.
PROTO = arm("kl-diag-polar-lora", **CW_PRODUCTION, global_batch_size=16,
            precond="product")

# E2 leave-one-out arms: the protagonist with one control removed.
NOSHAMPOO = {**PROTO, "cw_no_diag_curv": True}      # w/o curvature control
NOMAG = {**PROTO, "cw_unpinned": True, "lora_init_b": "symmetric"}  # w/o magnitude control
DOUBLE = {**PROTO, "cw_no_diag_curv": True, "cw_unpinned": True,
          "lora_init_b": "symmetric", "max_steps": 9000}  # both removed = the LoRA-Muon step

# Derivation ablations: each is its own registered optimizer, so the optimizer
# string selects it; CW_PRODUCTION holds the rest of the config at the
# protagonist's so the only difference is the derivation premise removed.
AVGLOSS = arm("kl-diag-lora", **CW_PRODUCTION,               # -per-sample: no msign, metric^-1
              precond="product")
HALFPOW = arm("kl-diag-flatout-lora", **CW_PRODUCTION,       # -msign at half metric power
              precond="product")
FLATOUT = arm("kl-diag-polar-flatout-lora", **CW_PRODUCTION,  # -outer un-whiten: msign only
              precond="product")

# ─── the `precond` axis: what fills (C_B, C_A) ───────────────────────────────
# Three branches, not four corners. All three share ONE (P, Q), the same p, q
# updates and the same rho = eta/(smax A + smax B) rule; they differ only in the
# slots. `kl-shampoo-polar-lora` IS the factorwise branch (its spec pins
# diag_metric=False), so its existing runs need no re-run to serve as that arm.
ONESIDED = {**PROTO, "precond": "one-sided"}     # C_B = C_A = I everywhere
# C_B = P_A, C_A = Q_B. TWO optimizer names produce this branch and they are the
# same computation: `kl-shampoo-polar-lora` pins diag_metric=False, which IS
# factorwise, and `kl-diag-polar-lora --precond factorwise` resolves to the same
# thing -- test_explicit_precond_reproduces_the_legacy_path_bitwise asserts the
# two are bit-identical. The older runs carry the first name and the newer sweeps
# the second, so pinning one name would drop half the arm: measured, the r16
# factorwise cells (5 runs, all at 9000) matched NOTHING while the figure showed
# the arm as absent. `precond` is what identifies the branch; the optimizer name
# is provenance.
NOPRODUCT = {**arm("kl-diag-polar-lora", **CW_PRODUCTION, precond="factorwise"),
             "optimizer": ("kl-diag-polar-lora", "kl-shampoo-polar-lora")}
# The diagonal slot: the same EMA of the factor's own whitened gradients as
# NOPRODUCT, but only its diagonal is kept. Third point on the slot-structure
# axis -- full r x r (NOPRODUCT), diagonal (this), identity (ONESIDED) -- and
# cheaper than either factorwise end at both ends of the step: the accumulation
# is a row-wise sum of squares, O(r d), rather than an r x r outer product,
# O(r^2 d), and the inverse square root is an elementwise rsqrt rather than a
# Gram Newton-Schulz (optim.py, the `rr_diagonal` branches).
NOPRODUCT_DIAG = arm(
    "kl-diag-polar-lora", **CW_PRODUCTION, precond="factorwise-diag")

# The same arm with the slots frozen at a checkpoint: the ablation asking
# whether the r x r ESTIMATE earns its place, or only its shape.
#
# Only the FROZEN side pins the field. The live side cannot: `field_matches`
# returns False for a field absent from the cfg, and `freeze_factorwise_slots`
# exists only on the branch implementing the ablation, so every run predating it
# lacks the key -- a `lambda v: not v` pin excluded 46 of them from the
# factorwise arm. Panels classify runs by `canonical_arm_label`, which carries
# " frozen-slots" only when the field is recorded True, so the two arms separate
# there without the live side pinning anything.
NOPRODUCT_FROZEN = {**NOPRODUCT, "freeze_factorwise_slots": True}

# ─── the `msign` axis: how accurately the matrix sign is applied ─────────────
# Orthogonal to `precond`. "diag" approximates the Gram inside the matrix sign by
# its diagonal, i.e. rownorm(Z_A) / colnorm(Z_B) — no r x r inverse sqrt. Run at
# BOTH ends of the precond axis, so the question "can the matrix sign be cheapened"
# is answered with the slot present and with it gone, not only after it is gone.
PROTO_DIAG = {**PROTO, "msign": "diag"}
ONESIDED_DIAG = {**PROTO, "precond": "one-sided", "msign": "diag"}

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
# `cw_nesterov=(False, True)` for the same reason `ADAMW` carries it:
# `LoRAPlusAdamW` never reads that flag, and runs at this cell log it BOTH ways
# depending on when they were launched, so pinning it to either value drops
# half the arm. This base is built by its own `arm()` call rather than from
# `ADAMW` (which pins `beta2=0.999`, the axis being swept), so the fix applied
# to `ADAMW` did not reach here: measured, all five non-0.999 arms rendered
# empty and `adamw_beta2_panel(256)` showed 1 of 6, with the grid runs present
# on disk the whole time.
ADAMW_BETA2_ARMS = beta2_arms(
    arm("adamw", precond_method=(None, "higham"), cw_nesterov=(False, True)),
    "beta2", _BETA2_GRID + [0.999], 0.999, "AdamW (shipped, b2=0.999)")


# Per-figure arm dicts (reader-facing label -> predicate), in legend order.
POLORA_LABEL = "PoLoRA"
PRECOND_PRODUCT_LABEL = r"Product: $C_B=B^\top P B,\ C_A=A Q A^\top$"
PANEL_ARMS = {
    "AdamW": ADAMW,
    POLORA_LABEL: PROTO,
    "iMuon": IMUON,
    "Muon (naive)": MUON,
    "LoRA-RITE": LORARITE,
    "w/o curvature+magnitude (LoRA-Muon step)": DOUBLE,
}
ABLATION_ARMS = {
    POLORA_LABEL: PROTO,
    # cw_no_diag_curv sets Q_isqrt=P_isqrt=1 (optim.py:2777).
    r"No curvature: $P=Q=I$": NOSHAMPOO,
    # cw_unpinned: rho=eta AND the final sigma_max(W) rescale is skipped, so
    # the raw whitened step is applied (optim.py:2979, :3041-3044).
    r"No magnitude rule: $\Delta A=-\eta W_A$": NOMAG,
    r"Neither: $P=Q=I$, $\Delta A=-\eta W_A$": DOUBLE,
}
DERIVATION_ARMS = {
    "AdamW": ADAMW,
    POLORA_LABEL: PROTO,
    # No msign, and the metric therefore applies TWICE (inner half composes
    # with the outer half) -- optim.py:1777-1782 spells this out.
    r"No $\mathrm{msign}$: $C_B^{-1}\widehat{M}_AQ^{-1}$": AVGLOSS,
    # No msign, outer un-whiten skipped, so the metric applies once at half
    # power -- the standard Shampoo/Adafactor whitening (optim.py:1778-1780).
    r"No $\mathrm{msign}$: $C_B^{-1/2}\widehat{M}_AQ^{-1/2}$": HALFPOW,
    # msign applied, outer un-whiten skipped (optim.py:3025-3027).
    r"$\mathrm{msign}(C_B^{-1/2}\widehat{M}_AQ^{-1/2})$": FLATOUT,
}
# The `precond` axis: product, identity, and two resolutions of the fitted
# factorwise slot. All four
# share one (P, Q), the same p, q updates and the same magnitude rule, and differ
# only in what fills (C_B, C_A).
PRECOND_ARMS = {
    "AdamW": ADAMW,
    PRECOND_PRODUCT_LABEL: PROTO,
    r"Identity: $C_B=C_A=I$": ONESIDED,
    r"Factorwise: $C_B=P_A,\ C_A=Q_B$": NOPRODUCT,
    r"Diagonal factorwise: $C_B=\operatorname{Diag}(P_A),\ C_A=\operatorname{Diag}(Q_B)$": NOPRODUCT_DIAG,
}

# `curvature_beta` crossed with `precond`, for the estimation-noise question:
# is factorwise's deficit at small r the cost of whitening by a NOISY estimate?
# `P_A` is a finite EMA, so it is anisotropic even when the true curvature is
# not, while one-sided's C_B = I has zero estimation variance and cannot make
# that error. Measured floor, feeding the EMA gradients whose true second moment
# is exactly I: injected anisotropy 0.098 / 0.126 / 0.125 at r = 16 / 64 / 256 —
# roughly FLAT in rank — against real anisotropy 0.195 / 0.338 / 0.447, which
# GROWS. beta2 0.99 -> 0.999 takes the effective sample size from 100 to 1000
# and should drop the floor with the signal untouched.
#
# BOTH branches appear at BOTH decays, and the one-sided rows are not padding:
# `curvature_beta` drives four EMAs, not one — P_A/Q_B (factorwise only) at
# optim.py:2184-2186, 2200-2201 and Q/P (BOTH arms) at 2191-2195,
# 2202-2203 — so without a one-sided control at the same decay, a shrinking gap
# cannot be told from beta2 simply helping everything. Measured in flight at
# step 750: beta2=0.999 moved factorwise -0.0006 and one-sided -0.0003, i.e.
# most of the effect is the shared diagonal metric.
PRECOND_BETA2_ARMS = {
    "AdamW": ADAMW,
    r"Factorwise, $\beta_2=0.9$": {**NOPRODUCT, "curvature_beta": 0.9},
    r"Factorwise, $\beta_2=0.99$": {**NOPRODUCT, "curvature_beta": 0.99},
    r"Factorwise, $\beta_2=0.999$": {**NOPRODUCT, "curvature_beta": 0.999},
    r"Identity, $\beta_2=0.9$": {**ONESIDED, "curvature_beta": 0.9},
    r"Identity, $\beta_2=0.99$": {**ONESIDED, "curvature_beta": 0.99},
    r"Identity, $\beta_2=0.999$": {**ONESIDED, "curvature_beta": 0.999},
}

# The `msign` axis, run at both ends of `precond`: can the matrix sign be replaced
# by its diagonal (row/column normalization) with the slot present, and with it
# gone? (one-sided, diag) is the O(rd) configuration — no r x r matmul or inverse
# square root anywhere in the direction.

MSIGN_ARMS = {
    "AdamW": ADAMW,
    "product, msign": PROTO,
    "product, diagonal msign": PROTO_DIAG,
    "one-sided, msign": ONESIDED,
    "one-sided, diagonal msign": ONESIDED_DIAG,
}

MAGNITUDE_RULE_ARMS = {
    r"PoLoRA: $\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$": PROTO,
    r"Naive: $\rho=\eta$": NAIVEMAG,
    "AdamW": ADAMW,
}

# Every arm dict the regression test walks. Adding a figure means adding its
# dict here, so the discrimination guard covers it too.
ALL_ARM_DICTS = {
    "PANEL_ARMS": PANEL_ARMS,
    "ABLATION_ARMS": ABLATION_ARMS,
    "DERIVATION_ARMS": DERIVATION_ARMS,
    "PRECOND_ARMS": PRECOND_ARMS,
    "MSIGN_ARMS": MSIGN_ARMS,
    "MAGNITUDE_RULE_ARMS": MAGNITUDE_RULE_ARMS,
    "PROTO_BETA2_ARMS": PROTO_BETA2_ARMS,
    "ADAMW_BETA2_ARMS": ADAMW_BETA2_ARMS,
    "PRECOND_BETA2_ARMS": PRECOND_BETA2_ARMS,
}
