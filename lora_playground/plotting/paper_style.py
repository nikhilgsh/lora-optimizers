"""Semantic series styles shared by notebook and manuscript figures."""
from __future__ import annotations

from matplotlib.pyplot import get_cmap as _get_cmap

from .colors import (
    SERIES_PALETTE, _color_distance, _rgb_to_hex, distinct_palette,
)


PAPER_SERIES_STYLES = {
    "AdamW": {"color": "#000000", "marker": "o"},
    "PoLoRA": {"color": "#0072B2", "marker": "s"},
    "iMuon": {"color": "#E0A33D", "marker": "^"},
    "Muon (naive)": {"color": "#CB5A4C", "marker": "v"},
    "LoRA-RITE": {"color": "#8E6BAE", "marker": ">"},
    "No curvature or magnitude": {
        "color": "#CC79A7", "marker": "P",
    },
    "Without curvature control": {
        "color": "#009E73", "marker": "D",
    },
    "Without magnitude rescale": {
        "color": "#882255", "marker": "X",
    },
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

    reserved = ["#000000", *(style["color"] for style in resolved.values())]
    # Take SERIES_PALETTE in order, keeping only entries clear of the colors
    # PAPER_SERIES_STYLES already spent. Order matters: the previous code tried
    # "tab10", then "tab20", "tab20b", "Set3", and inside each one
    # `distinct_palette` picks greedy farthest-first -- both the map that
    # answered and the colors it chose depended on HOW MANY unregistered series
    # the panel had, so one series came out #2ca02c green in a five-arm panel
    # and #ff7f0e orange in a three-arm one. In order, the nth series always
    # gets the nth free color, whatever else is on the panel.
    palette = [
        color for color in SERIES_PALETTE
        if all(_color_distance(color, r) > 0.15 for r in reserved)
    ]
    if len(palette) < len(remaining):
        # More unregistered series than the palette holds. Append rather than
        # reorder, so the series that already had colors keep them.
        palette = palette + distinct_palette(
            len(remaining) - len(palette),
            reserved=reserved + palette,
            source=[
                _rgb_to_hex(c) for cmap in ("tab20", "tab20b", "tab20c")
                for c in _get_cmap(cmap).colors
            ],
        )

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
