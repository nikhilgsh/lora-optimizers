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

from .colors import OPTIM_COLORS, distinct_palette

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
    would share a label with the ns=5 partial-polar runs. Reads the loader's
    derived ``effective_polar_iters`` (step count) and ``effective_inner_polar``
    (method); ``PE=N`` for polar_express, ``ns=N`` otherwise. Returns "" when
    the step count couldn't be resolved (never silently implies a quality)."""
    d = cfg.get("_derived", {})
    n = d.get("effective_polar_iters")
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
        return "AdamW"
    opt = cfg.get("optimizer")
    if opt in ("curvature-whiten-lora", "curvature-whiten-polar-lora"):
        is_polar = opt == "curvature-whiten-polar-lora"
        polar = (" +polar" + _polar_quality_tag(cfg)) if is_polar else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        return f"SOAP-curv{polar} (f={f}{bc}{dd})"
    if opt in ("kl-shampoo-lora", "kl-shampoo-polar-lora"):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "kl-shampoo-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-Shampoo{polar}{ks} (f={f}{bc}{dd})"
    if opt == "kl-diag-polar-flatout-lora":
        pq = _polar_quality_tag(cfg)
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-diag-flatout +polar{pq}{ks} (f={f}{bc}{dd})"
    if opt in ("kl-diag-lora", "kl-diag-polar-lora"):
        polar = (" +polar" + _polar_quality_tag(cfg)) if opt == "kl-diag-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        dl = cfg.get("precond_delta")
        dd = f", δ={_eps(dl)}" if dl is not None else ""
        pic = cfg.get("cw_picard_iters", 1) or 1
        ks = f" k{pic}" if pic > 1 else ""
        return f"KL-diag{polar}{ks} (f={f}{bc}{dd})"
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
        return f"diag-Shampoo{polar}{nes}{ks} (f={f}{bc}{dd})"
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
    return f"{fam} {polar} k={a['k']} ({damp}){curv}"


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
        for src in ("tab10", "tab20", "tab20b", "Set3"):
            try:
                palette = distinct_palette(len(rest), reserved=reserved, source=src)
                break
            except ColorCollisionError:
                continue
        else:  # exhausted sources at min_distance — relax it as a last resort
            palette = distinct_palette(len(rest), reserved=reserved,
                                       source="tab20", min_distance=0.12)
    colors.update({l: palette[i] for i, l in enumerate(rest)})
    return colors
