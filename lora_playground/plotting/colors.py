"""Optimizer color/family registries, marker overrides, ablation-axis style
constants, and the overlay-palette collision guard.

When adding a new optimizer:
  1. add an entry to ``OPTIM_COLORS`` below.
  2. add it to whichever family/families it belongs in in ``OPTIM_FAMILIES``.
The ``_validate_family_membership()`` check at module load surfaces any
``OPTIM_COLORS`` entry that's missing from every family as a warning.
"""
from __future__ import annotations


# NOTE: pure "black" is reserved for the AdamW baseline overlay; no candidate
# group should use it.
OPTIM_COLORS = {
    # 22 distinct colors — verified no-collision (see test_optim_colors_are_unique).
    # Grouped by family for readability; assignment is unique across the map.
    "adamw":                       "#000000",  # baseline — pure black overlay
    "adafactor":                   "#555555",  # grey baseline — Adafactor (rank-1 v factorization)
    # no-Adam family
    "scaled-lora":                 "#ff7f0e",
    "lin-lora":                    "#2ca02c",
    "muon-lora":                   "#e377c2",
    "imuon-lora":                  "#17becf",   # iMuon reference baseline (arXiv:2605.09238, v5)
    "polar-product-lora":          "#c49c94",
    # pre-Adam preconditioning (geometry → Adam)
    "adam-scaled-lora":            "#d62728",
    "adam-lin-lora":               "#9467bd",
    "adam-lin-core-lora":          "#b48ad6",  # cross-check: core-space Adam in lin-LoRA solver
    # post-Adam preconditioning (Adam → geometry, falsified family)
    "adam-lin-lora-post":          "#aec7e8",
    "adam-scaled-lora-post":       "#ffbb78",
    "adam-lin-lora-matrix":        "#98df8a",
    "adam-scaled-lora-matrix":     "#c5b0d5",
    # spectral / polar (the headline family)
    "adam-muon-lora":              "#3cb44b",   # vivid green, distinct from lin-lora's tab green
    "muon-adam-lora":              "#dbdb8d",
    "adam-polar-product-lora":     "#8c564b",
    "adam-polar-product-lora-coupled": "#5d342c",
    "adam-polar-product-lora-coupled-endrms": "#a04a3c",
    "adam-polar-product-lora-coupled-exact-chord": "#7d4a2e",
    "adam-polar-product-lora-coupled-spectral-chord": "#c75c2e",
    "adam-polar-product-lora-coupled-spectral-chord-tight": "#7a3520",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean": "#5a2d20",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw": "#3d1e16",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-rho": "#d97a40",
    "adam-polar-product-lora-coupled-spectral-chord-tight-exact": "#a8453c",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening": "#8e6b3c",
    "adam-polar-product-lora-coupled-spectral-chord-tight-outer": "#c95a35",
    "adam-polar-product-lora-coupled-spectral-chord-direction": "#a02a0d",
    "adam-soap-polar-product-lora": "#c97a3a",
    "adafactor-polar-product-lora": "#d4a04c",
    "sign-momentum-polar-product-lora": "#7c3aed",
    "adam-clip-product-lora":                 "#1f6f8c",
    "adam-clip-product-lora-coupled":         "#0f4f6f",
    "adam-clip-product-lora-coupled-endrms":  "#3a8fb5",
    "adam-polar-product-lora-gauge":          "#7e3a85",
    "adam-polar-product-lora-gauge-coupled":  "#5a2d63",
    "adam-polar-product-lora-clip-gauge":     "#3a857e",
    "adam-polar-product-lora-clip-gauge-coupled": "#1e6e63",
    "polar-coupled-core-lora":     "#7a3a2c",
    "polar-coupled-core-imbalance-scalar-lora":  "#a05030",
    "polar-coupled-core-balanced-scalar-lora":   "#c87858",
    "polar-coupled-core-imbalance-lora":         "#b86a48",
    "polar-coupled-core-imbalance-restore-lora": "#d8a070",
    "polar-coupled-core-state-rebalanced-lora":  "#5a2018",
    "polar-coupled-core-sign-lora":              "#9a4030",
    "polar-coupled-core-sign-rebalanced-lora":   "#bb6048",
    "polar-coupled-core-factor-adam-lora":       "#c08040",
    "polar-coupled-core-factor-adam-rebalanced-lora": "#e0a060",
    "muon-coupled-core-lora":      "#3a5a8a",
    "muon-coupled-core-imbalance-scalar-lora":   "#4a6aa8",
    "muon-coupled-core-balanced-scalar-lora":    "#7090c8",
    "muon-coupled-core-imbalance-lora":          "#5a8ac8",
    "muon-coupled-core-state-rebalanced-lora":   "#1a3a6a",
    "muon-coupled-core-sign-lora":               "#5a7ab8",
    "muon-coupled-core-sign-rebalanced-lora":    "#7a9ae0",
    "adamuon-polar-product-lora":  "#1f77b4",
    "adamuon-lora":                "#ff9896",
    # gauge-invariant variants
    "product-muon-lora":           "#0d3d66",
    "adam-product-muon-lora":      "#9edae5",
    # K-FAC / per-coord / dropped families (plotted only in legacy cells)
    "diag-scaled-lora":            "#3a6ea8",
    "kron-grad-lora":              "#bcbd22",
    "psi-lora":                    "#7f7f7f",
    "galore-adamw":                "#a55194",
    # orthogonal-core LoRA (UCV^T parameterization)
    "adam-ucv-core-lora":          "#2b8c4d",
    # SOAP on momentum in an S⊗D curvature basis; polar A/B
    "curvature-whiten-lora":       "#2e7d5b",
    "curvature-whiten-polar-lora": "#1f5f8c",
    # KL-Shampoo-LoRA: coupled KL fixed-point curvature, no v̂; polar A/B
    "kl-shampoo-lora":             "#d4801f",
    "kl-shampoo-polar-lora":       "#9c27b0",
    # Option (b): consistent diagonal global metric (geometric small side); polar A/B
    "kl-diag-lora":                "#c2185b",
    "kl-diag-polar-lora":          "#6a1b9a",
    # kl-diag-polar with the un-whiten removed (flat-spectrum robustness probe)
    "kl-diag-polar-flatout-lora":  "#ad1457",
    # Non-KL ablation of option (b): same diagonal metric, plain grad-energy EMA
    # diagonals (no KL fixed point); polar A/B
    "diag-shampoo-lora":           "#00897b",
    "diag-shampoo-polar-lora":     "#00695c",
}


OPTIM_FAMILIES = {
    # Headline polar/muon spectral family (post-Adam direction-shaping).
    "headline_polar": {
        "adamw",
        "adafactor",
        "adam-polar-product-lora",
        "adam-polar-product-lora-coupled",
        "adam-polar-product-lora-coupled-endrms",
        "adam-polar-product-lora-coupled-exact-chord",
        "adam-polar-product-lora-coupled-spectral-chord",
        "adam-polar-product-lora-coupled-spectral-chord-tight",
        "adam-polar-product-lora-coupled-spectral-chord-tight-clean",
        "adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw",
        "adam-polar-product-lora-coupled-spectral-chord-tight-no-rho",
        "adam-polar-product-lora-coupled-spectral-chord-tight-exact",
        "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
        "adam-polar-product-lora-coupled-spectral-chord-tight-outer",
        "adam-polar-product-lora-coupled-spectral-chord-direction",
        "adam-soap-polar-product-lora",
        "adafactor-polar-product-lora",
        "sign-momentum-polar-product-lora",
        "curvature-whiten-lora",
        "curvature-whiten-polar-lora",
        "kl-shampoo-lora",
        "kl-shampoo-polar-lora",
        "kl-diag-lora",
        "kl-diag-polar-lora",
        "kl-diag-polar-flatout-lora",
        "diag-shampoo-lora",
        "diag-shampoo-polar-lora",
        "adam-clip-product-lora",
        "adam-clip-product-lora-coupled",
        "adam-clip-product-lora-coupled-endrms",
        "adam-polar-product-lora-gauge",
        "adam-polar-product-lora-gauge-coupled",
        "adam-polar-product-lora-clip-gauge",
        "adam-polar-product-lora-clip-gauge-coupled",
        "polar-coupled-core-lora",
        "polar-coupled-core-imbalance-scalar-lora",
        "polar-coupled-core-balanced-scalar-lora",
        "polar-coupled-core-imbalance-lora",
        "polar-coupled-core-imbalance-restore-lora",
        "polar-coupled-core-state-rebalanced-lora",
        "polar-coupled-core-sign-lora",
        "polar-coupled-core-sign-rebalanced-lora",
        "polar-coupled-core-factor-adam-lora",
        "polar-coupled-core-factor-adam-rebalanced-lora",
        "muon-coupled-core-lora",
        "muon-coupled-core-imbalance-scalar-lora",
        "muon-coupled-core-balanced-scalar-lora",
        "muon-coupled-core-imbalance-lora",
        "muon-coupled-core-state-rebalanced-lora",
        "muon-coupled-core-sign-lora",
        "muon-coupled-core-sign-rebalanced-lora",
        "adamuon-lora",
        "adamuon-polar-product-lora",
        "adam-muon-lora",
    },
    "pre_adam_lin_scaled": {
        "adamw",
        "adam-lin-lora",
        "adam-lin-core-lora",
        "adam-scaled-lora",
    },
    "no_adam": {
        "adamw",
        "muon-lora",
        "imuon-lora",
        "polar-product-lora",
        "lin-lora",
        "scaled-lora",
    },
    "post_adam_h4": {
        "adamw",
        "adam-lin-lora-post",
        "adam-scaled-lora-post",
        "adam-lin-lora-matrix",
        "adam-scaled-lora-matrix",
        "muon-adam-lora",
    },
    "ucv_family": {
        "adamw",
        "adam-polar-product-lora",
        "adam-ucv-core-lora",
    },
    "bucket3_weak": {
        "adamw",
        "product-muon-lora",
        "adam-product-muon-lora",
        "diag-scaled-lora",
        "kron-grad-lora",
        "psi-lora",
        "galore-adamw",
    },
}


def _validate_family_membership() -> None:
    """Warn at module load when an OPTIM_COLORS entry is in no OPTIM_FAMILIES
    set. Soft check — some optimizers may be intentionally excluded from every
    cell-level comparison; the warning surfaces the more likely "you added a
    color but forgot a family" failure that silently empties plots."""
    in_some_family: set[str] = set().union(*OPTIM_FAMILIES.values())
    orphans = sorted(set(OPTIM_COLORS) - in_some_family)
    if orphans:
        import warnings
        warnings.warn(
            f"OPTIM_COLORS entries with no OPTIM_FAMILIES membership "
            f"(plots filtering by family will silently drop them): {orphans}. "
            f"Add to a family in plotting.colors or accept the exclusion.",
            stacklevel=2,
        )


_validate_family_membership()


# Linestyles for LoRA+ multiplier disambiguation. Convention: same color per
# base optimizer across all m, with linestyle carrying the m signal.
M_LINESTYLES = {
    1: "-",                 # solid
    4: (0, (5, 2)),         # long-dash
    16: (0, (1, 2)),        # dotted (gappy)
    32: (0, (3, 2, 1, 2)),  # dash-dot
}

# Marker styles to disambiguate near-color pairs. Default "o" elsewhere.
OPTIM_MARKERS = {
    # Greens cluster
    "lin-lora":                    "o",
    "adam-muon-lora":              "^",
    "muon-adam-lora":              "v",
    "adam-lin-lora-matrix":        "P",
    # Reds/pinks cluster
    "adam-scaled-lora":            "o",
    "adamuon-lora":                "X",
    "muon-lora":                   "*",
    "imuon-lora":                  "X",
    # Blues/cyans cluster
    "adamuon-polar-product-lora":  "o",
    "diag-scaled-lora":            "s",
    "adam-product-muon-lora":      "D",
    "adam-lin-lora-post":          "h",
    # Purples cluster
    "adam-lin-lora":               "o",
    "adam-scaled-lora-matrix":     "p",
    "galore-adamw":                "<",
    # Browns/oranges cluster
    "adam-polar-product-lora":     "o",
    "adam-polar-product-lora-coupled": "P",
    "adam-polar-product-lora-coupled-endrms": "X",
    "adam-polar-product-lora-coupled-exact-chord": "h",
    "adam-polar-product-lora-coupled-spectral-chord": "8",
    "adam-polar-product-lora-coupled-spectral-chord-tight": "H",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening": "x",
    "adam-polar-product-lora-coupled-spectral-chord-tight-outer": "P",
    "adam-polar-product-lora-coupled-spectral-chord-direction": "*",
    "adam-soap-polar-product-lora": "*",
    "adam-polar-product-lora-gauge":          "D",
    "adam-polar-product-lora-gauge-coupled":  "d",
    "adam-polar-product-lora-clip-gauge":     "v",
    "adam-polar-product-lora-clip-gauge-coupled": "^",
    "polar-product-lora":          "d",
    "scaled-lora":                 ">",
    "adam-scaled-lora-post":       "8",
    # Olive/yellow + dark
    "kron-grad-lora":              "o",
    "product-muon-lora":           "o",
    "psi-lora":                    "o",
    "adamw":                       "s",
}


# ─── overlay palette safety: no-collision guard ──────────────────────────────

class ColorCollisionError(ValueError):
    """Raised when an overlay palette contains a color visually too close to a
    reserved axis color (e.g. an OPTIM_COLORS entry or a per-notebook axis
    convention)."""


def _hex_to_rgb(c) -> tuple[float, float, float]:
    """Accept a '#rrggbb' string or an RGB(A) tuple; return RGB in [0,1]^3."""
    if isinstance(c, str):
        h = c.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"expected #rrggbb, got {c!r}")
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(float(x) for x in c[:3])


def _color_distance(c1, c2) -> float:
    """Euclidean distance in RGB ∈ [0,1]^3. Range is [0, sqrt(3) ≈ 1.732].

    RGB Euclidean is a coarse perceptual proxy — Lab ΔE would be more accurate,
    but RGB suffices for catching the practical failure mode (a Reds-cmap shade
    landing within visual confusion distance of tab-orange NS=5).
    """
    r1 = _hex_to_rgb(c1)
    r2 = _hex_to_rgb(c2)
    return sum((a - b) ** 2 for a, b in zip(r1, r2)) ** 0.5


def _rgb_to_hex(c) -> str:
    """Inverse of _hex_to_rgb for the tuple branch."""
    rgb = tuple(int(round(x * 255)) for x in c[:3])
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def assert_palette_distinct_from_reserved(
    candidates,
    reserved,
    *,
    min_distance: float = 0.30,
    name: str = "overlay",
) -> None:
    """Raise ColorCollisionError if any candidate color is within min_distance
    (RGB Euclidean) of a reserved color.

    min_distance=0.30 was tuned against the failure mode of `Reds`-cmap(0.45)
    (a pinkish-orange around `#e3927e`) vs tab orange `#ff7f0e` (distance
    ≈ 0.28) — the Reds light end was the original collision and 0.30 is the
    threshold that flags it while leaving Purples and pure cyans/magentas safe.
    """
    bad: list[tuple[str, str, float]] = []
    for c in candidates:
        c_hex = c if isinstance(c, str) else _rgb_to_hex(c)
        for r in reserved:
            r_hex = r if isinstance(r, str) else _rgb_to_hex(r)
            d = _color_distance(c_hex, r_hex)
            if d < min_distance:
                bad.append((c_hex, r_hex, d))
    if bad:
        details = "; ".join(f"{c} vs reserved {r} (d={d:.3f})" for c, r, d in bad)
        raise ColorCollisionError(
            f"{name} palette collides with reserved colors "
            f"(min_distance={min_distance}): {details}"
        )


def distinct_palette(
    n: int,
    *,
    reserved,
    source: str | list = "tab10",
    min_distance: float = 0.30,
    name: str = "distinct",
) -> list[str]:
    """Return `n` qualitative colors, each visually distinct from `reserved`
    and from each other (pairwise RGB distance > min_distance).

    Use when overlay series should be read as independent values (e.g. SSC
    c-knob), not as a gradient. For sequential progression (light-to-dark
    of one hue), use `overlay_palette` instead.

    Source = a matplotlib qualitative colormap name ('tab10', 'tab20',
    'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', …) or a list of
    candidate hex strings. The default 'tab10' provides 10 hues designed
    for categorical distinguishability.

    Selection is greedy farthest-first over candidates that pass the
    reserved filter — each pick maximizes the minimum distance to the
    already-selected set ∪ reserved. Raises ColorCollisionError if the
    candidate pool can't yield `n` colors satisfying min_distance.
    """
    import matplotlib.pyplot as plt

    if n <= 0:
        return []

    # Enumerate candidates from the source palette.
    if isinstance(source, str):
        cmap = plt.get_cmap(source)
        # ListedColormap exposes .colors directly; LinearSegmentedColormap
        # we sample at integer indices up to N (caller responsibility).
        if hasattr(cmap, "colors"):
            candidates = [_rgb_to_hex(c) for c in cmap.colors]
        else:
            # Fallback: 10 evenly-spaced samples.
            candidates = [_rgb_to_hex(cmap(i / 9)) for i in range(10)]
    else:
        candidates = [c if isinstance(c, str) else _rgb_to_hex(c) for c in source]

    reserved_hex = [r if isinstance(r, str) else _rgb_to_hex(r) for r in reserved]

    # Filter candidates that are too close to any reserved color.
    eligible = [
        c for c in candidates
        if all(_color_distance(c, r) >= min_distance for r in reserved_hex)
    ]

    # Greedy farthest-first selection.
    picked: list[str] = []
    pool = list(eligible)
    while len(picked) < n and pool:
        if not picked:
            # Seed: first eligible candidate.
            picked.append(pool.pop(0))
            continue
        # Pick candidate that maximizes min distance to (picked ∪ reserved).
        anchors = picked + reserved_hex
        best_idx, best_min_d = -1, -1.0
        for i, c in enumerate(pool):
            d = min(_color_distance(c, a) for a in anchors)
            if d > best_min_d:
                best_min_d, best_idx = d, i
        if best_min_d < min_distance:
            break  # remaining candidates too close to picked set
        picked.append(pool.pop(best_idx))

    if len(picked) < n:
        raise ColorCollisionError(
            f"{name}: source palette {source!r} yielded only {len(picked)} "
            f"color(s) at min_distance={min_distance} given {len(reserved_hex)} "
            f"reserved color(s); requested {n}. Try a larger source ('tab20', "
            f"'Set3', a custom list), or lower min_distance."
        )
    return picked


def overlay_palette(
    n: int,
    *,
    reserved,
    cmap: str = "Purples",
    value_range: tuple[float, float] = (0.55, 0.95),
    name: str = "overlay",
) -> list[str]:
    """Return `n` sequential colors from `cmap` over `value_range`, validated
    not to collide with `reserved`.

    Designed for overlay series where order should read as a visual
    progression (smaller index → lighter shade). The default value_range
    `(0.55, 0.95)` skips the very-light low end of most sequential colormaps
    so even the lightest overlay shade is legible against a white background.

    Use Purples for overlays on figures whose primary axis uses tab10
    (blue/orange/green) — Purples sits in a hue region far from those. For
    overlays on a different reserved set, pass a different `cmap`.

    Example:
        NS_AXIS = {2: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}
        colors = overlay_palette(len(cs), reserved=NS_AXIS.values())
    """
    import matplotlib.pyplot as plt

    if n <= 0:
        return []
    cmap_obj = plt.get_cmap(cmap)
    lo, hi = value_range
    palette = [
        _rgb_to_hex(cmap_obj(lo + (hi - lo) * (i / max(n - 1, 1))))
        for i in range(n)
    ]
    assert_palette_distinct_from_reserved(palette, reserved, name=name)
    return palette
