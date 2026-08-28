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

from ..publication_identity import lora_init_label_suffix
from .colors import (
    OPTIM_COLORS, SERIES_PALETTE, _color_distance, _rgb_to_hex,
    distinct_palette,
)

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
    k = cfg.get("_derived", {}).get(
        "effective_picard_iters", cfg.get("picard_iters_override")) or 1
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


def _shared_knobs(cfg: dict) -> str:
    """Suffix for cross-optimizer axes that canonical_label must discriminate
    but that aren't in any per-optimizer template. Only non-default values
    appear, so default runs keep their bare label."""
    s = ""
    if _field_is_active(cfg, "cw_no_diag_curv") and cfg.get("cw_no_diag_curv"):
        s += " w/o-curv"
    # `precond` is the three-branch (C_B, C_A) selector; only the two non-default
    # branches get a suffix so product runs keep their bare label.
    if _field_is_active(cfg, "precond") and cfg.get("precond") == "one-sided":
        s += " one-sided"
    elif (
        _field_is_active(cfg, "precond")
        and cfg.get("precond") == "factorwise"
        and cfg.get("optimizer") not in (
            "kl-shampoo-lora", "kl-shampoo-polar-lora",
        )
    ):
        # KL-Shampoo is the legacy factorwise implementation, so the family
        # name already identifies this branch. Keeping the suffix off makes a
        # recorded pre-flag KL-Shampoo run and an explicit
        # ``--precond factorwise`` run share one canonical label.
        s += " factorwise"
    if _field_is_active(cfg, "msign") and cfg.get("msign") == "diag":
        s += " msign-diag"
    if _field_is_active(cfg, "cw_unpinned") and cfg.get("cw_unpinned"):
        s += " unpinned"
    # Each of these appends a suffix only when the run is OFF the default, so
    # the default run keeps a bare label. The default is DERIVED, never typed
    # here: a literal copy goes stale the moment the real default moves, and
    # then every run looks off-default and gets a suffix. That happened —
    # `cw_metric_init`'s default became "1e-12" (`optim_config.py:101`,
    # `train.py:790`) while this file still said "zero", so `canonical_label`
    # appended ` minit=1e-12` to EVERY run, bare "AdamW" resolved in 0 of 19
    # workload cells, and `docs/notes/leaderboard.md` regenerated with 168 "—"
    # cells that the doc's own header explains as "never reached the target".
    # `_residual_knobs` below already derives from `_config_defaults()`; this
    # hand-written block now does too.
    if (_field_is_active(cfg, "higham_iters")
            and (hi := _off_default(cfg, "higham_iters")) is not None):
        s += f" H={hi}"
    if (_field_is_active(cfg, "beta1")
            and (b1 := _off_default(cfg, "beta1")) is not None):
        s += f" β1={b1:g}"
    if (_field_is_active(cfg, "cw_metric_init")
            and (cm := _off_default(cfg, "cw_metric_init")) is not None):
        s += f" minit={cm}"
    if (_field_is_active(cfg, "rdinv_variant")
            and (rv := _off_default(cfg, "rdinv_variant")) is not None):
        s += f" rdinv={rv}"
    rd = cfg.get("rdinv_delta")
    if _field_is_active(cfg, "rdinv_delta") and rd is not None:
        s += f" rdδ={_eps(rd)}"
    # The optimizer's declared IMPLEMENTATION_REVISION. It exists precisely for
    # "update semantics changed with no corresponding resolved-config change"
    # (run_schema.optimizer_implementation_revision), so two revisions of one
    # config are two series and must not average: the factorwise free-slot fix
    # is revision 2, and `paper_view_semantics` EXCLUDES pre-fix factorwise runs
    # from the paper's precond views rather than pooling them. Not reachable via
    # `_residual_knobs` (that derives from PINNED_FIELDS, i.e. OptimizerConfig),
    # so it is spelled out here; the default comes from run_schema, not a
    # literal, so a bump there does not suffix every run.
    if (rev := cfg.get("optimizer_impl_revision")) is not None:
        from ..run_schema import DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION
        if rev != DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION:
            s += f" impl-rev={rev}"
    return (
        s
        + _residual_knobs(cfg)
        + lora_init_label_suffix(cfg.get("lora_init_b", "zero"))
    )


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


def _off_default(cfg: dict, field: str):
    """`cfg[field]` when it differs from the SHIPPED default, else None.

    The default comes from `OptimizerConfig` / the train.py CLI rather than a
    literal in this file, so a default that moves cannot silently turn every run
    into an off-default one. An absent field reads as on-default: a run logged
    before the flag existed ran the default by definition.

    Raises on an unknown field name, matching `arms.arm()` — a typo here would
    otherwise mean "never off default", i.e. a knob that stops appearing in
    labels and starts collapsing distinct sweeps onto one.
    """
    from .arms import _cli_defaults, _config_defaults
    defaults = {**_config_defaults(), **_cli_defaults()}
    if field not in defaults:
        raise KeyError(
            f"{field!r} is not an OptimizerConfig field or a train.py CLI flag, "
            f"so it has no default to compare against. Fix the name — a typo "
            f"here silently stops the knob from appearing in any label.")
    v = cfg.get(field)
    return None if v is None or v == defaults[field] else v


# Fields the per-optimizer templates and the hand-written suffix above already put
# in the label. Everything else that is off its default is appended generically by
# `_residual_knobs`, so the FAILURE MODE INVERTS: a field added to OptimizerConfig
# is absent from this set, so it is appended automatically and the label keeps
# discriminating. Before, a new field was absent from the hand-written suffix and
# so was silently dropped -- which is how six buckets on the hero workload came to
# share one label, `AdamW minit=1e-12` at lr=1e-4 covering six distinct series
# (the beta2 grid), five of which dedup_by_canonical then discarded.
_LABELLED_ELSEWHERE = frozenset({
    # per-optimizer templates (the headline name)
    "precond_refresh_every", "curvature_beta", "precond_delta",
    "precond_delta_relative", "polar_method", "muon_ns_steps", "ns_form",
    "polar_norm_dir", "polar_sigma_power", "cw_picard_iters",
    "picard_iters_override", "picard_alpha", "curvature_whitening",
    "ssc_c", "ssc_nsteps", "ssc_kappa",
    # the hand-written suffix above
    "cw_no_diag_curv", "precond", "msign", "cw_unpinned", "higham_iters",
    "beta1", "cw_metric_init", "rdinv_variant", "rdinv_delta",
})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _residual_knobs(cfg: dict) -> str:
    """`k=v` for every pinnable config field that is off its default and is not
    already shown in the label.

    Derived from `arms.PINNED_FIELDS()` (OptimizerConfig minus the per-series
    axes), not from a remembered list, so it cannot go stale as fields are added.
    """
    from .arms import PINNED_FIELDS, _config_defaults, _inert_fields
    defaults = _config_defaults()
    out = []
    active = PINNED_FIELDS() - _inert_fields(cfg.get("optimizer"))
    for f in sorted(active - _LABELLED_ELSEWHERE):
        if f not in cfg:
            continue
        v = cfg[f]
        if v == defaults.get(f):
            continue
        # `+flag` / `-flag` rather than `flag=True` -- booleans dominate this
        # suffix and "=True" carries no information the name does not.
        if isinstance(v, bool):
            out.append(("+" if v else "-") + f)
        else:
            out.append(f"{f}={_fmt(v)}")
    return (" " + " ".join(out)) if out else ""


def canonical_label(cfg: dict) -> str | None:
    """Human-readable, fully-discriminating variant label (or None to exclude).

    Every distinguishing axis is present, so distinct configs never share a
    label. Examples:
      AdamW
      chord-tight ns=5 k=1 (abs)
      chord-tight PE=10 k=1 (abs)
      chord-tight ns=8 k=1 (ε_rel=1e-2)
      chord-tight-clean ns=8 k=2 (abs)
      chord-tight-clean ns=8 k=2 (κ_sr=0.75)
    """
    if cfg.get("optimizer") == OPT_ADAMW:
        return "AdamW" + _shared_knobs(cfg)
    opt = cfg.get("optimizer")
    if opt == "imuon-lora":
        return "iMuon" + _shared_knobs(cfg)
    if opt == "lora-rite":
        return "LoRA-RITE" + _shared_knobs(cfg)
    if opt == "muon-lora":
        steps = cfg.get("muon_ns_steps")
        method = cfg.get("polar_method")
        prefix = "PE" if method == "polar_express" else "ns"
        quality = f" {prefix}={steps}" if steps is not None else ""
        return f"Muon{quality}" + _shared_knobs(cfg)
    if opt in ("curvature-whiten-lora", "curvature-whiten-polar-lora"):
        is_polar = opt == "curvature-whiten-polar-lora"
        polar = (" +polar" + _polar_quality_tag(cfg)) if is_polar else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        return f"SOAP-curv{polar} (f={f}{bc}{dd})" + _shared_knobs(cfg)
    if opt in ("kl-shampoo-lora", "kl-shampoo-polar-lora"):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "kl-shampoo-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-Shampoo{polar}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg)
    if opt == "kl-diag-polar-flatout-lora":
        pq = _polar_quality_tag(cfg)
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-diag-flatout +polar{pq}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg)
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
        return f"{family}{polar}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg)
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
        return f"diag-Shampoo{polar}{nes}{ks} (f={f}{bc}{dd})" + _shared_knobs(cfg)
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
    return f"{fam} {polar} k={a['k']} ({damp}){curv}" + _shared_knobs(cfg)


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
    reserved set via `distinct_palette`. Deterministic given the label set."""
    from .colors import ColorCollisionError
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
        # Take SERIES_PALETTE IN ORDER, skipping only entries a pin already
        # spent. Deliberately not `distinct_palette`'s greedy farthest-first:
        # that maximises separation for the arm count it is handed, so the
        # answer changes with the count and one arm's color moved between
        # panels. In order, the nth series always gets the nth free color.
        palette = [c for c in SERIES_PALETTE
                   if all(_color_distance(c, r) > 0.15 for r in reserved)]
        if len(palette) < len(rest):
            # More series than the palette holds. Widen the pool by appending,
            # so the series that already had colors keep them.
            import matplotlib.pyplot as plt
            big = [_rgb_to_hex(c) for cm in ("tab20", "tab20b", "tab20c")
                   for c in plt.get_cmap(cm).colors]
            try:
                palette = palette + distinct_palette(
                    len(rest) - len(palette),
                    reserved=reserved + palette, source=big,
                )
            except ColorCollisionError:  # still too tight — relax the spacing
                palette = palette + distinct_palette(
                    len(rest) - len(palette),
                    reserved=reserved + palette, source=big,
                    min_distance=0.08,
                )
    colors.update({l: palette[i] for i, l in enumerate(rest)})
    return colors
