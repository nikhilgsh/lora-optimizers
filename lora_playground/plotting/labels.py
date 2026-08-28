"""Single source of truth for optimizer-variant display labels, colors, and
legend order — used by every leaderboard notebook and the doc generator so
method names are identical everywhere.

Two contracts this module exists to enforce:
  * **Completeness** — `canonical_label` always encodes every distinguishing
    axis (family, polar method + iters, picard k, damping regime + value), so a
    label can never collapse two distinct configs into one bucket. That is what
    makes the silent-merge failure in `compare_variants_figure` impossible once
    the figure path defaults to this labeler + the `assert_label_discriminates`
    guard.
  * **AdamW is reserved** — AdamW maps to `OPTIM_COLORS['adamw']` (black) and is
    ordered FIRST, by construction (not per-notebook attention).

`canonical_key` is the compact pipe-form mechanistic key (e.g. `ct|ns8|k1|abs`)
for cross-setting aggregation; `canonical_label` is the human-readable form.
Both derive from one field extractor (`_axes`) so they can never diverge.
"""
from __future__ import annotations

from functools import lru_cache

from ..publication_identity import lora_init_label_suffix
from .colors import OPTIM_COLORS, series_colors

OPT_ADAMW = "adamw"
OPT_CT = "adam-polar-product-lora-coupled-spectral-chord-tight"
OPT_CT_CLEAN = OPT_CT + "-clean"


def _eps(x) -> str:
    """Format a damping epsilon as 1e-2 / 1e-6 etc."""
    return f"{float(x):.0e}".replace("e-0", "e-").replace("e+0", "e")


def _polar_quality_tag(cfg: dict) -> str:
    """Polar-quality suffix for CurvatureWhitenLoRA-backed +polar optimizers.

    These optimizers hardcoded their polar step to Newton-Schulz at
    ``ns_steps`` (train.py default 5), so every existing run is a PARTIAL
    polar; without this tag a future full-polar (polar_express / ns>=8) run
    would share a label with the ns=5 partial-polar runs. Prefers the loader's
    derived ``effective_polar_iters`` (step count) / ``effective_inner_polar``
    (method), then falls back to the RECORDED ``muon_ns_steps`` /
    ``polar_method`` those resolve from (the dependency set
    ``publication_semantics._EFFECTIVE_DEPENDENCIES`` declares for
    ``effective_polar_iters``); ``PE=N`` for polar_express, ``ns=N`` otherwise.
    Returns "" only when neither is recorded.

    The fallback is load-bearing, not belt-and-braces: no code populates
    ``_derived`` any more, so reading it alone made this tag "" for all 886 runs
    on disk and the polar-quality axis dropped silently out of every label —
    ``diag_shampoo_polar_*_opc_blackwell`` (ns=5) and its ``_pe8_`` rerun
    (polar_express=8) shared one label at four learning rates in three
    leaderboard cells, which `dedup_by_canonical` then refused to collapse."""
    d = cfg.get("_derived", {})
    n = d.get("effective_polar_iters")
    if n is None:
        n = cfg.get("muon_ns_steps")
    if n is None:
        return ""
    method = d.get("effective_inner_polar") or cfg.get("polar_method")
    prefix = "PE" if method == "polar_express" else "ns"
    return f" {prefix}={n}"


def effective_picard_iters(cfg: dict) -> int:
    """The Picard cross-coupling count ``k`` the run actually used.

    Four spellings, in decreasing authority. ``_derived`` carries what the
    optimizer itself emitted (``optim.py:6447`` logs ``int(self.picard_iters)``),
    so it is the ground truth where present; ``picard_iters_override`` and
    ``picard_iters`` are constructor inputs and can lag it; 1 is the default.

    Named because two callers resolved it differently and disagreed: a caller
    that stopped at ``picard_iters_override`` and fell back to the string
    ``"?"`` labelled six `chord-direction` runs ``k=?`` -- their recorded
    ``picard_iters`` was 3 or absent -- while series identity kept them
    distinct, giving 20 label collisions on one panel.
    """
    derived = cfg.get("_derived") or {}
    for value in (derived.get("effective_picard_iters"),
                  cfg.get("picard_iters_override"),
                  cfg.get("picard_iters")):
        if value is not None:
            return int(value)
    return 1


def _axes(cfg: dict) -> dict | None:
    """Extract the distinguishing axes from a run cfg. Returns None for
    optimizers outside the chord-tight family (and AdamW handled by callers).

    Reads loader-backfilled fields (`muon_ns_steps`, `polar_method`) and the
    loader-derived `effective_picard_iters` (raw `picard_iters_override` can
    lag the value the optimizer actually used — see project loader notes)."""
    opt = cfg.get("optimizer")
    if opt not in (OPT_CT, OPT_CT_CLEAN):
        return None
    ns = cfg.get("muon_ns_steps")
    pm = cfg.get("polar_method")
    k = effective_picard_iters(cfg)
    if cfg.get("precond_delta_relative"):
        damp_kind, damp_val = "epsrel", cfg.get("precond_delta")
    elif cfg.get("ssc_kappa") is not None:
        damp_kind, damp_val = "ssckap", cfg.get("ssc_kappa")
    elif cfg.get("ssc_c") is not None:
        damp_kind, damp_val = "sscc", cfg.get("ssc_c")
    else:
        damp_kind, damp_val = "abs", cfg.get("precond_delta")
    return {
        "family": "clean" if opt == OPT_CT_CLEAN else "ct",
        "polar_method": pm,
        "ns": ns,
        "k": int(k),
        "damp_kind": damp_kind,
        "damp_val": damp_val,
        # Whitening metric: default geometric factor-Gram (BᵀB) vs the
        # --curvature_whitening EMA of factor-grad outer products (S_curv).
        "curv": bool(cfg.get("curvature_whitening")),
    }


# Cross-optimizer axes with a SPELLING of their own in the label, in the order
# they appear in it. Each entry is `(field, render)`; `render(cfg, value)` is
# called only for a value that reached the label — see `_label_value` for the
# three gates — and may still return "" for a value that needs no suffix.
#
# This table replaced ten hand-written `if cfg.get(<field>) ...` lines that each
# re-implemented the same gate. The gate is the fact this module kept getting
# wrong: `cw_metric_init` compared against a literal "zero" after the real
# default moved to "1e-12" (`optim_config.py:101`, `train.py:790`), so
# `canonical_label` appended ` minit=1e-12` to EVERY run, bare "AdamW" resolved
# in 0 of 19 workload cells, and `docs/notes/leaderboard.md` regenerated with
# 168 "—" cells that the doc's own header explains as "never reached the
# target". One gate, derived defaults, and a per-field renderer that decides
# only how to SPELL a value it is already told is worth spelling.
#
# Every field name here must carry a `field_roles` LABELLED_ROLES role, or the
# knob is invisible to `canonical_arm_label`'s role exclusion and to
# `_residual_knobs`' duplicate suppression. `tests/test_field_roles.py` asserts
# it; `_declared_default` raises on a roleless name at call time. (The check is
# a test rather than an import-time assertion in this module: `loader` imports
# `plotting`, so a module-level `field_roles` import here would close an import
# cycle through `arms.PINNED_FIELDS()` -> `manifest` -> `loader` -> `plotting`.)
_FEATURED_KNOBS: tuple[tuple[str, object], ...] = (
    ("cw_no_diag_curv", lambda cfg, v: " w/o-curv" if v else ""),
    # `precond` is the three-branch (C_B, C_A) selector; only the two non-default
    # branches get a suffix so product runs keep their bare label. A family whose
    # SPEC already forces a branch needs no suffix: the optimizer name identifies
    # it. That keeps a recorded pre-flag KL-Shampoo run (no `precond`,
    # `diag_metric=False` pinned) and an explicit `--precond factorwise` run
    # under one canonical label.
    ("precond", lambda cfg, v: (
        " one-sided" if v == "one-sided"
        else " factorwise-diag" if v == "factorwise-diag"
        else " factorwise" if (v == "factorwise"
                               and not _branch_is_implied_by_the_optimizer(cfg))
        else "")),
    ("msign", lambda cfg, v: " msign-diag" if v == "diag" else ""),
    ("cw_unpinned", lambda cfg, v: " unpinned" if v else ""),
    ("higham_iters", lambda cfg, v: f" H={v}"),
    ("beta1", lambda cfg, v: f" β1={v:g}"),
    ("cw_metric_init", lambda cfg, v: f" minit={v}"),
    ("rdinv_variant", lambda cfg, v: f" rdinv={v}"),
    ("rdinv_delta", lambda cfg, v: f" rdδ={_eps(v)}"),
    # The factorwise-slot freeze. Not an `OptimizerConfig` field on this branch
    # -- it exists only where the ablation is implemented -- so it reaches the
    # label through `field_roles._RECORDED_ONLY_ALGORITHM`, which is what makes it
    # visible here at all. Without it a frozen-slot continuation and the dynamic
    # run it forked from share one label while `series_id` splits them on the
    # fork's `resume_debug_replay`, so they collide in one bucket instead of
    # being two arms.
    # `is True`, not truthiness: the run may record False, and there is no
    # declared default for the gate to compare that against.
    ("freeze_factorwise_slots", lambda cfg, v: " frozen-slots" if v is True else ""),
    # The optimizer's declared IMPLEMENTATION_REVISION. It exists precisely for
    # "update semantics changed with no corresponding resolved-config change"
    # (run_schema.optimizer_implementation_revision), so two revisions of one
    # config are two series and must not average: the factorwise free-slot fix
    # is revision 2, and `paper_view_semantics` EXCLUDES pre-fix factorwise runs
    # from the paper's precond views rather than pooling them. Its default comes
    # from `run_schema` via the derived default map, not a literal, so a bump
    # there does not suffix every run. `canonical_arm_label` drops it by ROLE.
    ("optimizer_impl_revision", lambda cfg, v: f" impl-rev={v}"),
)

_FEATURED_FIELDS = frozenset(field for field, _ in _FEATURED_KNOBS)


def _shared_knobs(cfg: dict, *, skip: frozenset = frozenset()) -> str:
    """Suffix for cross-optimizer axes that canonical_label must discriminate
    but that aren't in any per-optimizer template.

    One loop over the fields whose `field_roles` role puts them in the label
    (``ALGORITHM`` and ``REVISION``), keeping only values that are off their
    derived default, so default runs keep their bare label. `_FEATURED_KNOBS`
    spells the ones with a name of their own, in its declared order;
    `_residual_knobs` appends the rest generically, so a field added to
    ``OptimizerConfig`` keeps discriminating without being named here.

    ``skip`` drops whole roles' fields — `canonical_arm_label` passes the
    ``REVISION`` role's fields, which is how "the label minus the code-revision
    token" is expressed as a role exclusion rather than a regex over the string.
    """
    s = ""
    for field, render in _FEATURED_KNOBS:
        if field in skip:
            continue
        v = _label_value(cfg, field)
        if v is None:
            continue
        s += render(cfg, v)
    return (
        s
        + _residual_knobs(cfg, skip=skip)
        + lora_init_label_suffix(cfg.get("lora_init_b", "zero"))
    )


def _branch_is_implied_by_the_optimizer(cfg: dict) -> bool:
    """Whether the optimizer's own spec already forces this `precond` branch.

    Derived from the spec registry rather than a list of optimizer names: that
    list was hand-typed in four places and a missed copy makes naming and
    cohort membership disagree about which runs are factorwise.
    """
    from ..optim_specs import resolved_precond

    # From the OPTIMIZER NAME alone. Reading the run's recorded `diag_metric`
    # is wrong: `--precond factorwise` on a kl-diag variant also records
    # diag_metric=False, so the run would look "implied" and lose the suffix it
    # needs to stay separate from that optimizer's product runs.
    return resolved_precond({"optimizer": cfg.get("optimizer")}) == cfg.get("precond")


def _field_is_active(cfg: dict, field: str) -> bool:
    """Whether ``field`` can affect the configured optimizer.

    Launchers historically logged many optimizer knobs for every optimizer.
    Values an optimizer never receives are provenance, not variant identity:
    including them split one algorithm into labels such as
    ``AdamW precond_method=higham``.  ``arms.arm`` already derives this same
    distinction from the optimizer factory; labels consume that result so
    selection and naming cannot disagree about which fields matter.

    Unknown optimizers conservatively report no inert fields, preserving the
    previous fail-closed behavior.
    """
    optimizer = cfg.get("optimizer")
    if not optimizer:
        return True
    from .arms import _inert_fields
    return field not in _inert_fields(optimizer)


def _declared_default(cfg: dict, field: str) -> tuple[bool, object]:
    """``(declared, default)`` for one field, from the code rather than a literal.

    ``OptimizerConfig`` / the train.py CLI / `run_schema`'s revision counters
    first (`dedup._shipped_defaults`), then the run's own optimizer constructor
    (`dedup._constructor_defaults`, which layers the variant's spec constants on
    the signature). A literal copy of a default in this file goes stale the
    moment the real default moves, and then every run looks off-default and gets
    a suffix — see `_FEATURED_KNOBS`' note on ``cw_metric_init``.

    ``declared=False`` for a field the current code gives no default at all
    (``freeze_factorwise_slots``, whose ablation exists only where it is
    implemented). Its renderer decides what a recorded value means; there is
    nothing to compare against.

    Raises on a field in no `field_roles` role, matching `arms.arm()`'s refusal
    of an unknown override — a typo here would otherwise mean "never off
    default", i.e. a knob that stops appearing in labels and starts collapsing
    distinct sweeps onto one.
    """
    from ..field_roles import role_of
    from .dedup import _constructor_defaults, _shipped_defaults
    shipped = _shipped_defaults()
    if field in shipped:
        return True, shipped[field]
    ctor = _constructor_defaults(cfg.get("optimizer"))
    if field in ctor:
        return True, ctor[field]
    if role_of(field) is None:
        raise KeyError(
            f"{field!r} has no `field_roles` role and no declared default, so "
            f"there is nothing to compare a recorded value against. Fix the "
            f"name — a typo here silently stops the knob from appearing in any "
            f"label.")
    return False, None


def _label_value(cfg: dict, field: str):
    """`cfg[field]` when it should reach the label, else None. Three gates:

      1. **absent or ``None``** — a run logged before the flag existed ran the
         default by definition, so it is not a distinguishing value. This is
         load-bearing for `field_roles._EXTRA_ALGORITHM`: those flags are absent
         from most cfgs on disk (``freeze_factorwise_slots`` from 2458 of 2498),
         and reading absence as a value would suffix every one of them.
      2. **not a scalar** — this function is fed arm PREDICATE dicts as well as
         recorded configs (`paper_plots_lib._canonical_variant_key` derives an
         arm's expected label from its predicate), and a predicate may pin a
         field with a CALLABLE or a membership tuple, both meaning "any value
         satisfying this". Neither names one value to spell, and a callable is
         TRUTHY: testing truthiness made the slots-live arm — pinned
         ``lambda v: not v`` — label itself as frozen and collide with the
         slots-frozen arm.
      3. **inert, or equal to the derived default** — `_field_is_active` drops
         what this optimizer cannot read, and `_declared_default` drops what the
         run left alone, so a default run keeps its bare label.
    """
    v = cfg.get(field)
    if v is None:
        return None
    if callable(v) or isinstance(v, (list, set, tuple, frozenset, dict)):
        return None
    if not _field_is_active(cfg, field):
        return None
    declared, default = _declared_default(cfg, field)
    if declared and v == default:
        return None
    return v


# Fields the per-optimizer templates put in the label under a name of their own.
# `_FEATURED_KNOBS`' fields are added below rather than repeated here, and the
# CONSTRUCTOR spellings of both (`optim_config.ALIAS`) with them: `ns_steps` is
# the same knob as `muon_ns_steps` and is recorded alongside it on 1677 runs, so
# without the alias the polar-quality axis would be spelled twice per label.
#
# Everything else that is off its default is appended generically by
# `_residual_knobs`, so the FAILURE MODE INVERTS: a field added to
# OptimizerConfig is absent from this set, so it is appended automatically and
# the label keeps discriminating. Before, a new field was absent from the
# hand-written suffix and so was silently dropped -- which is how six buckets on
# the hero workload came to share one label, `AdamW minit=1e-12` at lr=1e-4
# covering six distinct series (the beta2 grid), five of which
# dedup_by_canonical then discarded.
_TEMPLATE_LABELLED = frozenset({
    "precond_refresh_every", "curvature_beta", "precond_delta",
    "precond_delta_relative", "polar_method", "muon_ns_steps", "ns_form",
    "polar_norm_dir", "polar_sigma_power", "cw_picard_iters",
    "picard_iters_override", "picard_alpha", "curvature_whitening",
    "ssc_c", "ssc_nsteps", "ssc_kappa",
})


def _labelled_elsewhere() -> frozenset:
    from ..optim_config import ALIAS
    named = _TEMPLATE_LABELLED | _FEATURED_FIELDS
    return named | frozenset(ctor for ctor, field in ALIAS.items()
                             if field in named)


_LABELLED_ELSEWHERE = _labelled_elsewhere()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@lru_cache(maxsize=1)
def _tail_fields() -> frozenset:
    """The fields `_residual_knobs` may append generically.

    `field_roles`' label-bearing roles (``ALGORITHM`` ∪ ``REVISION``) restricted
    to fields whose default ``OptimizerConfig`` or `run_schema` DECLARES. The
    restriction is the same argument `dedup._series_items`' third tier makes: a
    value can only be called off-default against a default the current code
    states. Two kinds of ALGORITHM field have none —

      * constructor spellings that live only inside a run's recorded
        ``optimizer_config`` block (``magnitude_rule``, ``picard_iters``,
        ``eps``, ``diag_metric``, …), whose only "default" is a per-variant
        constant the spec pins, so comparing against it says nothing about the
        run's choices;
      * train.py harness flags that are not optimizer knobs
        (``training_mode``, ``target_modules``, ``max_grad_norm``), which the
        caller's cell selection and `arms.arm()`'s explicit overrides carry.

    — and appending them would put a `k=v` token on runs that chose nothing.
    This keeps the tail's actual contract, which is the one its predecessor
    documented: a field added to ``OptimizerConfig`` is appended automatically,
    so the label keeps discriminating without being named here.
    """
    from ..field_roles import LABELLED_ROLES, fields_with_roles
    from ..run_schema import REVISION_FIELDS
    from .arms import _config_defaults
    declared = frozenset(_config_defaults()) | frozenset(REVISION_FIELDS)
    return fields_with_roles(*LABELLED_ROLES) & declared


def _residual_knobs(cfg: dict, *, skip: frozenset = frozenset()) -> str:
    """`k=v` for every label-bearing field that is off its default and is not
    already shown in the label.

    The field set is `_tail_fields()` — derived from `field_roles`, not a
    remembered list, so it cannot go stale as fields are added.
    """
    from .arms import _inert_fields
    out = []
    active = _tail_fields() - _inert_fields(cfg.get("optimizer"))
    for f in sorted(active - _LABELLED_ELSEWHERE - skip):
        v = _label_value(cfg, f)
        if v is None:
            continue
        # `+flag` / `-flag` rather than `flag=True` -- booleans dominate this
        # suffix and "=True" carries no information the name does not.
        if isinstance(v, bool):
            out.append(("+" if v else "-") + f)
        else:
            out.append(f"{f}={_fmt(v)}")
    return (" " + " ".join(out)) if out else ""


def canonical_arm_label(cfg: dict) -> str | None:
    """`canonical_label` minus the ``REVISION`` role's fields, for ARM matching.

    An arm names an algorithm CHOICE -- which branch fills the r x r slots,
    which magnitude rule -- and `arms.arm()` pins exactly the config fields
    that express one. `optimizer_impl_revision` is not such a field: no CLI
    flag sets it, `run_schema` stamps it from the optimizer class. So an arm
    predicate can never carry it, and comparing FULL labels made every run
    recording revision 2 match no arm at all -- which emptied the reference
    arm of the Qwen2.5/openmath/r16 matched panel and raised
    "has no recorded reference arm".

    Implemented by EXCLUDING the role from the label, not by regexing the
    rendered string: a regex has to be extended for every new revision
    counter's spelling (`run_schema.REVISION_FIELDS` already carries two), and
    a counter whose token the pattern did not match would silently be compared.

    The revision still splits SERIES identity: `dedup.series_id` is mechanical
    and does not read this, and a panel that would mix two revisions inside one
    arm raises from `comparison`'s semantic-signature check instead of
    averaging them.
    """
    from ..field_roles import REVISION, fields_with_roles
    return canonical_label(cfg, skip_fields=fields_with_roles(REVISION))


def canonical_label(cfg: dict, *,
                    skip_fields: frozenset = frozenset()) -> str | None:
    """Human-readable, fully-discriminating variant label (or None to exclude).

    Every distinguishing axis is present, so distinct configs never share a
    label. Examples:
      AdamW
      chord-tight ns=5 k=1 (abs)
      chord-tight PE=10 k=1 (abs)
      chord-tight ns=8 k=1 (ε_rel=1e-2)
      chord-tight-clean ns=8 k=2 (abs)
      chord-tight-clean ns=8 k=2 (κ_sr=0.75)

    ``skip_fields`` withholds named cfg fields from the suffix. Callers pass a
    `field_roles` ROLE's fields rather than a field list — `canonical_arm_label`
    is the one caller, dropping ``REVISION``.
    """
    if cfg.get("optimizer") == OPT_ADAMW:
        return "AdamW" + _shared_knobs(cfg, skip=skip_fields)
    opt = cfg.get("optimizer")
    if opt == "imuon-lora":
        return "iMuon" + _shared_knobs(cfg, skip=skip_fields)
    if opt == "lora-rite":
        return "LoRA-RITE" + _shared_knobs(cfg, skip=skip_fields)
    if opt == "muon-lora":
        steps = cfg.get("muon_ns_steps")
        method = cfg.get("polar_method")
        prefix = "PE" if method == "polar_express" else "ns"
        quality = f" {prefix}={steps}" if steps is not None else ""
        return f"Muon{quality}" + _shared_knobs(cfg, skip=skip_fields)
    if opt in ("curvature-whiten-lora", "curvature-whiten-polar-lora"):
        is_polar = opt == "curvature-whiten-polar-lora"
        polar = (" +polar" + _polar_quality_tag(cfg)) if is_polar else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        return f"SOAP-curv{polar} (f={f}{bc}{dd})" + _shared_knobs(cfg, skip=skip_fields)
    if opt in ("kl-shampoo-lora", "kl-shampoo-polar-lora"):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "kl-shampoo-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-Shampoo{polar}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg, skip=skip_fields)
    if opt == "kl-diag-polar-flatout-lora":
        pq = _polar_quality_tag(cfg)
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-diag-flatout +polar{pq}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg, skip=skip_fields)
    if opt in (
        "kl-diag-lora", "kl-diag-flatout-lora", "kl-diag-polar-lora",
    ):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "kl-diag-polar-lora" else ""
        family = "KL-diag-halfpow" if opt == "kl-diag-flatout-lora" else "KL-diag"
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"{family}{polar}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg, skip=skip_fields)
    if opt in ("diag-shampoo-lora", "diag-shampoo-polar-lora"):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "diag-shampoo-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        nes = " +nesterov" if cfg.get("cw_nesterov") else ""
        return f"diag-Shampoo{polar}{nes}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg, skip=skip_fields)
    a = _axes(cfg)
    if a is None:
        return None
    fam = "chord-tight" if a["family"] == "ct" else "chord-tight-clean"
    polar = f"PE={a['ns']}" if a["polar_method"] == "polar_express" else f"ns={a['ns']}"
    if a["damp_kind"] == "epsrel":
        damp = f"ε_rel={_eps(a['damp_val'])}"
    elif a["damp_kind"] == "ssckap":
        damp = f"κ_sr={a['damp_val']:g}"
    elif a["damp_kind"] == "sscc":
        damp = f"c={a['damp_val']:g}"
    else:
        damp = f"abs={_eps(a['damp_val'])}" if a["damp_val"] is not None else "abs"
    curv = " +curv" if a["curv"] else ""
    return f"{fam} {polar} k={a['k']} ({damp}){curv}" + _shared_knobs(cfg, skip=skip_fields)


def canonical_key(cfg: dict) -> str | None:
    """Compact mechanistic key for cross-setting aggregation (e.g.
    `ct|ns8|k1|abs`). polar_express and ns>=8 both collapse to `full` here —
    a deliberate aggregation choice; `canonical_label` keeps them distinct."""
    if cfg.get("optimizer") == OPT_ADAMW:
        return "AdamW"
    a = _axes(cfg)
    if a is None:
        return None
    full = a["polar_method"] == "polar_express" or (a["ns"] is not None and a["ns"] >= 8)
    polar = "full" if full else f"ns{a['ns']}"
    if a["damp_kind"] == "epsrel":
        damp = "epsrel"
    elif a["damp_kind"] == "ssckap":
        damp = f"ssckap{a['damp_val']}"
    elif a["damp_kind"] == "sscc":
        damp = f"sscc{a['damp_val']}"
    else:
        damp = "abs"
    curv = "|curv" if a["curv"] else ""
    return f"{a['family']}|{polar}|k{a['k']}|{damp}{curv}"


def order_labels(labels) -> list:
    """AdamW first; the rest stable-sorted alphabetically for determinism."""
    labels = list(dict.fromkeys(labels))  # de-dup, preserve first-seen
    rest = sorted(l for l in labels if l != "AdamW")
    return (["AdamW"] if "AdamW" in labels else []) + rest


# Display-label pins: recurring figure labels that must keep the SAME color in
# every figure regardless of which other labels are present (palette assignment
# is positional, so without pins a label's color shifts with the arm set).
# Any label starting with "PoLoRA" maps to the protagonist's optimizer
# color, so panel-specific suffixes ("(kl-diag)", "(ours)") stay consistent.
PINNED_LABEL_COLORS = {
    "iMuon": OPTIM_COLORS["imuon-lora"],
    "w/o curvature control": "#2a9d8f",
    "w/o magnitude control": "#e76f51",
    "w/o curvature+magnitude": "#8c510a",
    "Muon (naive)": "#e377c2",
    # Paper-panel series (the protagonist, the baselines, the three `precond`
    # branches) are NOT pinned here -- `paper_style.PAPER_SERIES_STYLES` owns
    # them, and pins their marker alongside their color. Adding them here too
    # would make two registries answer the same question, free to disagree.
}
PROTAGONIST_LABEL_PREFIX = "PoLoRA"
PROTAGONIST_COLOR = OPTIM_COLORS["kl-diag-polar-lora"]


def pinned_label_color(label: str) -> str | None:
    """Fixed color for a well-known display label, else None (palette-assigned)."""
    if label.startswith(PROTAGONIST_LABEL_PREFIX):
        return PROTAGONIST_COLOR
    return PINNED_LABEL_COLORS.get(label)


def canonical_colors(labels) -> dict:
    """AdamW → reserved black; pinned labels (protagonist, iMuon, E2 arms) →
    their fixed colors; every other label → a distinct color kept clear of the
    reserved set via `colors.series_colors`. Deterministic given the label
    set."""
    ordered = order_labels(labels)
    colors = {}
    if "AdamW" in ordered:
        colors["AdamW"] = OPTIM_COLORS["adamw"]
    for l in ordered:
        if l not in colors:
            pin = pinned_label_color(l)
            if pin is not None:
                colors[l] = pin
    rest = [l for l in ordered if l not in colors]
    palette = []
    if rest:
        reserved = ["#000000"] + sorted(set(colors.values()))
        palette = series_colors(len(rest), reserved=reserved)
    colors.update({l: palette[i] for i, l in enumerate(rest)})
    return colors
