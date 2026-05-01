"""Predicate-based sweep loader and inventory.

``load_runs(where=...)`` selects runs whose cfg matches a per-field predicate;
``inventory_runs(logs_root)`` returns a structural audit (orphans, deprecated,
unknown optimizers, lr-pinning) for a notebook audit cell.

Scope tags on manifests are metadata only — they don't drive loading. To
exclude an old sweep set ``"deprecated": true`` in its own ``meta.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .manifest import live_manifests_newest_first, load_manifests, warn_untagged
from .plot_utils import (
    DIVERGE_THRESHOLD, OPTIM_COLORS, has_runs, load_sweep, max_loss, merge_runs,
)

# Default dedup axes used by 95% of notebook cells. Override per-cell when
# specialty axes (e.g. training_mode for the SVD oracle) matter.
DEFAULT_KEY_AXES: tuple[str, ...] = (
    "optimizer", "lr", "lora_r", "lora_plus_multiplier", "seed",
)

# Pinning categories returned in CoverageRow.pinning.
PINNING_INTERIOR = "interior"
PINNING_LOW = "pinned_low"
PINNING_HIGH = "pinned_high"
PINNING_SINGLE = "single_lr"
PINNING_ALL_DIVERGED = "all_diverged"


def _matches(spec: Any, value: Any) -> bool:
    """Predicate matcher for a single field.

    - callable                  → ``spec(value)`` truthy
    - list / set / tuple        → ``value in spec``
    - anything else (literal)   → ``value == spec``
    """
    if callable(spec):
        return bool(spec(value))
    if isinstance(spec, (list, set, tuple, frozenset)):
        return value in spec
    return value == spec


def _build_filter(where: dict[str, Any] | None) -> Callable[[dict], bool] | None:
    if not where:
        return None

    def predicate(cfg: dict) -> bool:
        for field_name, spec in where.items():
            if field_name not in cfg:
                return False
            if not _matches(spec, cfg[field_name]):
                return False
        return True

    return predicate


def load_runs(
    where: dict[str, Any] | None = None,
    *,
    key_axes: tuple[str, ...] = DEFAULT_KEY_AXES,
    cfg_postprocess: Callable[[dict, str], None] | None = None,
    logs_root: str = "../logs",
) -> list[tuple[dict, list[dict]]]:
    """Load all runs whose cfg matches every predicate in ``where``.

    Predicate types per field (see ``_matches``):
      - literal:               ``cfg[field] == value``
      - list/set/tuple:        ``cfg[field] in values``
      - callable:              ``predicate(cfg[field])`` truthy

    Omitted fields impose no constraint. A run missing a field referenced in
    ``where`` is excluded (treat absence as non-match).

    Dedup: ``key_fn(cfg) = tuple(cfg[a] for a in key_axes)``. ``merge_runs``
    keeps newest-trajectory-wins; group priority is newest-first (by
    ``submitted_at``). The hidden-axis collision check still fires if two
    runs share the dedup key but differ on another cfg axis.

    Deprecated groups (manifest field ``deprecated: true``) are excluded.
    """
    manifests = load_manifests(logs_root, strict=False)
    groups = [m["group"] for m in live_manifests_newest_first(manifests)]
    filter_fn = _build_filter(where)

    def key_fn(cfg: dict) -> tuple:
        return tuple(cfg.get(a) for a in key_axes)

    return merge_runs(
        groups,
        key_fn=key_fn,
        filter_fn=filter_fn,
        cfg_postprocess=cfg_postprocess,
        logs_root=logs_root,
    )


# ─── inventory ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoverageRow:
    """One row per (optimizer, lora_r, lora_plus_multiplier) cell."""
    optimizer: str
    lora_r: int
    lora_plus_multiplier: float
    lrs_swept: tuple[float, ...]
    best_lr: float | None              # None when all runs diverged
    final_loss_at_best: float | None
    pinning: str
    source_groups: tuple[str, ...]


@dataclass(frozen=True)
class RunInventory:
    groups_on_disk: tuple[str, ...]            # populated logs/<group>/run_info dirs
    groups_loaded: tuple[str, ...]             # subset that contributes runs
    groups_orphaned: tuple[str, ...]           # populated, no manifest or empty scope
    groups_deprecated: tuple[str, ...]         # self-flagged via meta.json deprecated:true
    optimizers_unknown: tuple[str, ...]        # in logs but not in OPTIM_COLORS
    coverage: tuple[CoverageRow, ...]

    @property
    def pinned(self) -> tuple[CoverageRow, ...]:
        """Subset of coverage with pinning ∈ {pinned_low, pinned_high}."""
        return tuple(r for r in self.coverage
                     if r.pinning in (PINNING_LOW, PINNING_HIGH))


def _classify_pinning(lrs_swept: tuple[float, ...], best_lr: float | None) -> str:
    if best_lr is None:
        return PINNING_ALL_DIVERGED
    if len(lrs_swept) <= 1:
        return PINNING_SINGLE
    if best_lr == min(lrs_swept):
        return PINNING_LOW
    if best_lr == max(lrs_swept):
        return PINNING_HIGH
    return PINNING_INTERIOR


def inventory_runs(logs_root: str = "../logs") -> RunInventory:
    """Walk all manifests + runs, return a structural audit.

    Each problem reported is a fact, not a threshold judgment:
      - groups_orphaned: populated dir without a valid scope-tagged manifest.
      - groups_deprecated: self-declared via ``meta.json`` ``deprecated: true``.
      - optimizers_unknown: optimizer present in some run's cfg but absent
        from ``OPTIM_COLORS`` — silently dropped from any cell that filters
        on color-map membership.
      - coverage: per (optimizer, lora_r, lora_plus_multiplier), the swept
        lrs, the best lr (lowest non-diverged final loss), and a pinning
        classification.
    """
    manifests = load_manifests(logs_root, strict=False)
    on_disk = sorted(m["group"] for m in manifests)
    deprecated = sorted(m["group"] for m in manifests if m.get("deprecated"))
    orphaned = sorted(warn_untagged(manifests))
    live = live_manifests_newest_first(manifests)
    live_groups = [m["group"] for m in live]

    # Single pass over all runs in live groups. We do NOT dedup here — the
    # inventory wants raw coverage across groups; downstream load_runs() does
    # the dedup for plotting.
    rows: dict[tuple[str, int, float], dict] = {}
    seen_optimizers: set[str] = set()
    contributing_groups: set[str] = set()
    for group in live_groups:
        if not has_runs(group, logs_root):
            continue
        for cfg, evs in load_sweep(group, logs_root):
            if not evs:
                continue
            optimizer = cfg.get("optimizer", "?")
            lora_r = int(cfg.get("lora_r", 16))
            mult = float(cfg.get("lora_plus_multiplier", 1.0))
            try:
                lr = float(cfg["lr"])
            except (KeyError, TypeError, ValueError):
                continue
            seen_optimizers.add(optimizer)
            contributing_groups.add(group)
            key = (optimizer, lora_r, mult)
            row = rows.setdefault(key, {"lrs": {}, "groups": set()})
            row["groups"].add(group)
            final = evs[-1]["eval_loss"]
            diverged = max_loss(evs) >= DIVERGE_THRESHOLD
            existing = row["lrs"].get(lr)
            if existing is None or (not diverged and (existing[1] or final < existing[0])):
                row["lrs"][lr] = (final, diverged)

    coverage: list[CoverageRow] = []
    for (optimizer, lora_r, mult), info in sorted(rows.items()):
        lrs = tuple(sorted(info["lrs"].keys()))
        non_diverged = [(lr, fl) for lr, (fl, div) in info["lrs"].items() if not div]
        if non_diverged:
            best_lr, best_loss = min(non_diverged, key=lambda x: x[1])
        else:
            best_lr, best_loss = None, None
        coverage.append(CoverageRow(
            optimizer=optimizer,
            lora_r=lora_r,
            lora_plus_multiplier=mult,
            lrs_swept=lrs,
            best_lr=best_lr,
            final_loss_at_best=best_loss,
            pinning=_classify_pinning(lrs, best_lr),
            source_groups=tuple(sorted(info["groups"])),
        ))

    optimizers_unknown = tuple(sorted(o for o in seen_optimizers if o not in OPTIM_COLORS))

    return RunInventory(
        groups_on_disk=tuple(on_disk),
        groups_loaded=tuple(sorted(contributing_groups)),
        groups_orphaned=tuple(orphaned),
        groups_deprecated=tuple(deprecated),
        optimizers_unknown=optimizers_unknown,
        coverage=tuple(coverage),
    )


def render_inventory(inv: RunInventory) -> str:
    """Plain-text report for the notebook audit cell."""
    lines: list[str] = []
    lines.append(f"Loaded {len(inv.groups_loaded)} of {len(inv.groups_on_disk)} groups on disk.")

    if inv.groups_orphaned:
        lines.append("")
        lines.append(f"ORPHANED ({len(inv.groups_orphaned)}) — populated but no valid manifest, will not load:")
        for g in inv.groups_orphaned:
            lines.append(f"  {g}")

    if inv.groups_deprecated:
        lines.append("")
        lines.append(f"DEPRECATED ({len(inv.groups_deprecated)}) — self-flagged, excluded:")
        for g in inv.groups_deprecated:
            lines.append(f"  {g}")

    if inv.optimizers_unknown:
        lines.append("")
        lines.append(f"UNKNOWN OPTIMIZERS ({len(inv.optimizers_unknown)}) — in logs but missing from OPTIM_COLORS, "
                     f"will be dropped by any cell that filters on it:")
        for o in inv.optimizers_unknown:
            lines.append(f"  {o}")

    lines.append("")
    lines.append(f"Coverage: {len(inv.coverage)} (optimizer, rank, mult) cells")
    if inv.pinned:
        lines.append(f"PINNED at lr-range boundary ({len(inv.pinned)}) — extension sweep recommended:")
        for r in inv.pinned:
            mult = f" m={r.lora_plus_multiplier:g}" if r.lora_plus_multiplier != 1.0 else ""
            lines.append(
                f"  {r.optimizer:<32}  r={r.lora_r:<4}{mult:<6}  "
                f"best_lr={r.best_lr:.0e} (final={r.final_loss_at_best:.4f}) "
                f"  swept={[f'{x:.0e}' for x in r.lrs_swept]} → {r.pinning}"
            )
    else:
        lines.append("No (optimizer, rank, mult) cells pinned at lr-range boundary.")

    return "\n".join(lines)
