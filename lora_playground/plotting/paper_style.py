"""Semantic series styles shared by notebook and manuscript figures."""
from __future__ import annotations

from .colors import series_colors


_CURVATURE_ABLATION = {"color": "#009E73", "marker": "D"}
_MAGNITUDE_ABLATION = {"color": "#882255", "marker": "X"}
_DOUBLE_ABLATION = {"color": "#CC79A7", "marker": "P"}

PAPER_SERIES_STYLES = {
    "AdamW": {"color": "#000000", "marker": "o"},
    "PoLoRA": {"color": "#0072B2", "marker": "s"},
    "iMuon": {"color": "#E0A33D", "marker": "^"},
    "Muon (naive)": {"color": "#CB5A4C", "marker": "v"},
    "LoRA-RITE": {"color": "#8E6BAE", "marker": ">"},
    "No curvature or magnitude": _DOUBLE_ABLATION,
    # Two names per style: the current label (which carries the equation) and
    # the pre-rename one, still used by the sealed view in
    # publication/paper_views.json. Same dict object, so they cannot drift.
    "No curvature: $P=Q=I$": _CURVATURE_ABLATION,
    "Without curvature control": _CURVATURE_ABLATION,
    r"No magnitude rule: $\Delta A=-\eta W_A$": _MAGNITUDE_ABLATION,
    "Without magnitude rescale": _MAGNITUDE_ABLATION,
    r"Neither: $P=Q=I$, $\Delta A=-\eta W_A$": _DOUBLE_ABLATION,
    r"Product: $C_B=B^\top P B,\ C_A=A Q A^\top$": {
        "color": "#0072B2", "marker": "s",
    },
    r"Identity: $C_B=C_A=I$": {
        "color": "#E69F00", "marker": "^",
    },
    r"Factorwise: $C_B=P_A,\ C_A=Q_B$": {
        "color": "#8E6BAE", "marker": "D",
    },
}

_MARKERS = ("o", "s", "^", "D", "v", "P", "X", ">", "<", "h")


def resolve_paper_styles(tokens) -> dict[str, dict[str, str]]:
    """Resolve complete styles without using panel order for recurring series."""
    ordered = list(dict.fromkeys(tokens))
    resolved = {
        token: dict(PAPER_SERIES_STYLES[token])
        for token in ordered if token in PAPER_SERIES_STYLES
    }
    remaining = sorted(token for token in ordered if token not in resolved)
    if not remaining:
        return resolved

    # Every registry color is reserved, not just the ones this panel drew. A
    # color that means "PoLoRA" in one figure must not be handed to an ad-hoc
    # series in the next, or the reader learns a key that then lies to them.
    reserved = ["#000000", *(
        style["color"] for style in PAPER_SERIES_STYLES.values()
    )]
    palette = series_colors(len(remaining), reserved=reserved)

    used_markers = {style["marker"] for style in resolved.values()}
    marker_order = [marker for marker in _MARKERS if marker not in used_markers]
    marker_order += [marker for marker in _MARKERS if marker in used_markers]
    for index, (token, color) in enumerate(zip(remaining, palette)):
        resolved[token] = {
            "color": color,
            "marker": marker_order[index % len(marker_order)],
        }
    return resolved


__all__ = ["PAPER_SERIES_STYLES", "resolve_paper_styles"]
