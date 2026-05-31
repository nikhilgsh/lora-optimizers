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
    # One-sided SOAP (CurvatureWhitenLoRA) — a distinct optimizer, not the
    # chord-tight pipeline, so it has its own axes (refresh freq f, curvature
    # EMA β, optional polar add-on) rather than ns/polar_method/damping.
    opt = cfg.get("optimizer")
    if opt in ("curvature-whiten-lora", "curvature-whiten-polar-lora"):
        polar = " +polar" if opt == "curvature-whiten-polar-lora" else ""
        f = cfg.get("precond_refresh_every")
        cb = cfg.get("curvature_beta")
        bc = f", β_c={cb:g}" if cb is not None else ""
        return f"SOAP-1side{polar} (f={f}{bc})"
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


def canonical_colors(labels) -> dict:
    """AdamW → reserved black; every other label → a distinct color kept clear
    of black via `distinct_palette`. Deterministic given the label set."""
    from .colors import ColorCollisionError
    ordered = order_labels(labels)
    non_adamw = [l for l in ordered if l != "AdamW"]
    palette = []
    if non_adamw:
        for src in ("tab10", "tab20", "tab20b", "Set3"):
            try:
                palette = distinct_palette(len(non_adamw), reserved=["#000000"], source=src)
                break
            except ColorCollisionError:
                continue
        else:  # exhausted sources at min_distance — relax it as a last resort
            palette = distinct_palette(len(non_adamw), reserved=["#000000"],
                                       source="tab20", min_distance=0.12)
    colors = {}
    if "AdamW" in ordered:
        colors["AdamW"] = OPTIM_COLORS["adamw"]
    colors.update({l: palette[i] for i, l in enumerate(non_adamw)})
    return colors
