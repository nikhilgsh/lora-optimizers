"""Series-identity helpers, label-collision detection, and per-group
baseline/variant filtering used by the high-level figure entry points.

Key contracts enforced here:
  - ``series_id(cfg)`` defines the equivalence class within which runs may be
    averaged (seeds, lr-grid points, horizon extensions of the same algorithm).
  - ``assert_label_discriminates`` raises if a display-label function collapses
    distinct ``series_id`` values into one bucket — the silent-averaging
    failure mode this contract exists to forbid.
  - ``filter_baseline`` / ``filter_variants`` hold non-target axes to the
    train.py argparse default within each per-axis group.
"""
from __future__ import annotations


class LabelCollisionError(ValueError):
    """Raised when distinct series_ids share one display label.

    The plotting layer averages within a display-label bucket as if all
    runs were seeds. When two distinct series_ids land in one bucket, the
    average silently mixes algorithms — exactly the failure mode this
    contract exists to forbid. The error message names the differing
    cfg field(s); the fix is to either extend the group_key/display_label
    function to discriminate, or (rarely) add the field to
    `manifest.SERIES_AXIS_FIELDS` if it's truly a per-series axis.
    """


def _hashable(v):
    """Recursive conversion to a hashable form (mirrors loader._hashable).
    Dicts → frozenset of items; lists → tuples; everything else → as-is.
    """
    if isinstance(v, dict):
        return frozenset((k, _hashable(vv)) for k, vv in v.items())
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def series_id(cfg: dict, *, axis_fields: frozenset[str] | None = None) -> frozenset:
    """Mechanical series identity = cfg minus SERIES_AXIS_FIELDS, with
    ``None``-valued fields treated as absent.

    Two cfgs with the same series_id are seeds / lr-grid points / horizon
    extensions of the same algorithm at the same model config and may be
    averaged together. Distinct series_ids cannot be averaged regardless
    of what a display-label function returns — see
    `assert_label_discriminates`.

    ``None`` is treated identically to "field not present": an older run
    whose schema didn't include flag X is informationally equivalent to a
    newer run with ``X=None`` (both fall through to argparse default at
    runtime). Treating them as distinct would split every old-vs-new
    cfg pair purely on schema growth. Explicit non-None values (``False``,
    ``0``, ``0.0``, ``""``) ARE series-defining.
    """
    if axis_fields is None:
        from lora_playground.manifest import SERIES_AXIS_FIELDS
        axis_fields = SERIES_AXIS_FIELDS
    # Exclude:
    #   - underscore-prefixed keys (loader enrichment namespaces:
    #     `_derived`, `_cli_args`, `_optim_steps`, etc.) — these mirror
    #     source-of-truth scalar fields and would double-count.
    #   - dict-valued composites like `optimizer_config` (loader backfill
    #     that mirrors a subset of scalar fields; older runs without it
    #     get backfilled and may not round-trip exactly).
    # Scalar top-level cfg fields ARE the source of truth.
    return frozenset(
        (k, _hashable(v)) for k, v in cfg.items()
        if k not in axis_fields
        and not k.startswith("_")
        and not isinstance(v, dict)
        and v is not None
    )


def _label_collision_report(runs, group_key_fn, *,
                            bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                            axis_fields: frozenset[str] | None = None,
                            ) -> list[dict]:
    """For every (label, *bucket_keys) bucket, return entries where the
    runs split into >1 distinct series_id. Each entry names the bucket,
    the differing cfg field(s), and the per-run values of those fields.
    """
    if axis_fields is None:
        from lora_playground.manifest import SERIES_AXIS_FIELDS
        axis_fields = SERIES_AXIS_FIELDS
    from collections import defaultdict
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for cfg, _ in runs:
        gk = group_key_fn(cfg)
        bk = tuple(cfg.get(k) for k in bucket_keys)
        buckets[(gk, bk)].append(cfg)

    reports: list[dict] = []
    for (label, bucket_vals), cfgs in buckets.items():
        if len(cfgs) < 2:
            continue
        ids = {series_id(c, axis_fields=axis_fields) for c in cfgs}
        if len(ids) == 1:
            continue
        # Identify the specific cfg fields that disagree across these runs.
        # Mirror series_id's exclusion rule so the reported diff is the
        # actual source-of-truth difference (not a derived/enriched field).
        all_keys: set[str] = set()
        for c in cfgs:
            all_keys.update(k for k, v in c.items()
                            if k not in axis_fields
                            and not k.startswith("_")
                            and not isinstance(v, dict))
        differing: dict[str, set] = {}
        for k in all_keys:
            vals = set()
            for c in cfgs:
                v = c.get(k)
                if v is None:
                    vals.add(None)
                    continue
                try:
                    vals.add(_hashable(v))
                except TypeError:
                    vals.add(repr(v))
            if len(vals) > 1:
                differing[k] = vals
        reports.append({
            "label": label,
            "bucket": dict(zip(bucket_keys, bucket_vals)),
            "n_runs": len(cfgs),
            "n_distinct_series": len(ids),
            "differing_fields": differing,
        })
    return reports


def assert_label_discriminates(runs, group_key_fn, *,
                               bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                               axis_fields: frozenset[str] | None = None,
                               ) -> None:
    """Hard-fail if any (label, *bucket_keys) bucket contains >1 series_id.

    Wire into every plot entry point that averages within a label bucket
    (`standard_sweep_figure`, `plot_eta_vs_final`, `_multi_seed_curve`).
    """
    reports = _label_collision_report(runs, group_key_fn,
                                      bucket_keys=bucket_keys,
                                      axis_fields=axis_fields)
    if not reports:
        return
    lines = []
    for r in reports:
        diffs = ", ".join(
            f"{k}={sorted(v, key=repr)}" for k, v in r["differing_fields"].items()
        )
        lines.append(
            f"  label={r['label']!r} bucket={r['bucket']} "
            f"n_runs={r['n_runs']} → {r['n_distinct_series']} distinct series_ids; "
            f"differing fields: {diffs}"
        )
    raise LabelCollisionError(
        "display label does not discriminate distinct series_ids; "
        "{} collision(s):\n{}\n"
        "Fix: extend the group_key / display-label function to discriminate "
        "on one of the listed fields, OR add the field to "
        "`lora_playground.manifest.SERIES_AXIS_FIELDS` if it is a true "
        "per-series axis (seed-like)."
        .format(len(reports), "\n".join(lines))
    )


def detect_group_collisions(runs, group_key_fn, *,
                            bucket_keys: tuple[str, ...] = ("lora_r", "lr"),
                            axis_fields: frozenset[str] | None = None,
                            ) -> list[dict]:
    """Non-raising sibling of `assert_label_discriminates`. Returns the
    same per-bucket collision reports for audit-cell printouts. Most
    callers should prefer the raising version — silent reports are how
    label/series drift accumulates.
    """
    return _label_collision_report(runs, group_key_fn,
                                   bucket_keys=bucket_keys,
                                   axis_fields=axis_fields)


def _last_step(evals) -> float:
    """Longest-trajectory comparator: max ``step`` among eval events, or -1
    for an empty trajectory (so any non-empty run wins over an empty one)."""
    if not evals:
        return -1.0
    return max(float(e.get("step", -1)) for e in evals)


def dedup_by_canonical(runs, *, keep_longest: bool = True):
    """Deduplicate ``(cfg, evals)`` runs by the canonical labeler so the dedup
    key can never drift from the displayed label.

    Dedup key is ``(canonical_label(cfg), float(cfg['lr']))``. When
    ``canonical_label(cfg)`` is ``None`` (a run outside the labeled family),
    the key falls back to ``cfg.get('optimizer')`` so unlabeled runs are NOT
    all collapsed into one bucket.

    With ``keep_longest`` (default), the run with the longest trajectory wins
    per key (max last ``step``; empty evals count as step -1). Input order of
    the kept runs is preserved (first-seen position of each key).
    """
    from .labels import canonical_label
    order: list = []
    best: dict = {}
    for cfg, evals in runs:
        label = canonical_label(cfg)
        key = (label, float(cfg["lr"])) if label is not None else cfg.get("optimizer")
        if key not in best:
            order.append(key)
            best[key] = (cfg, evals)
        elif keep_longest and _last_step(evals) > _last_step(best[key][1]):
            best[key] = (cfg, evals)
    return [best[k] for k in order]


def _baseline_values(runs, *, allowed_to_vary: set) -> dict:
    """Build {field: baseline_value} where baseline_value is the train.py
    argparse default. Only include fields that (a) vary across ``runs``
    AND (b) have their argparse default among the observed values.

    Why argparse default over modal: modal is fragile when an investigation's
    variant data takes >50% of the cell (the variant becomes "modal" → baseline
    runs get dropped instead of variants). Argparse default is stable.
    """
    from collections import defaultdict
    from lora_playground.loader import _argparse_defaults
    defaults = _argparse_defaults()
    observed: dict = defaultdict(set)
    for cfg, _ in runs:
        for k, v in cfg.items():
            if k in allowed_to_vary or k.startswith("_") or isinstance(v, dict):
                continue
            observed[k].add(_hashable(v))
    baseline: dict = {}
    for k, vals in observed.items():
        if len(vals) <= 1:
            continue
        if k not in defaults:
            continue
        default_val = _hashable(defaults[k])
        if default_val in vals:
            baseline[k] = default_val
    return baseline


def filter_baseline(runs, *, varying: tuple[str, ...] = ()):
    """Keep runs that match the argparse-default value on every scalar
    cfg field that VARIES WITHIN their ``varying``-tuple group (and has
    its default observed in the data), excluding fields in ``varying``
    and in ``manifest.SERIES_AXIS_FIELDS``.

    ``varying`` is the analysis cell's INTENDED axis of variation — fields
    whose differences the cell's display_label discriminates on (e.g.
    ``('optimizer', 'effective_picard_iters')`` for an optimizer-x-k
    leaderboard).

    Per-group (vs global) is essential for co-varying fields: e.g.
    ``polar_norm_dir`` is constant within each optimizer's runs but
    differs across optimizers. A global pass would hold it to the
    argparse default (one optimizer wins, others are excluded).
    """
    from collections import defaultdict
    from lora_playground.manifest import SERIES_AXIS_FIELDS
    allowed = set(varying) | SERIES_AXIS_FIELDS
    groups: dict = defaultdict(list)
    for cfg, evs in runs:
        gkey = tuple(_hashable(cfg.get(v)) for v in varying)
        groups[gkey].append((cfg, evs))
    kept = []
    for grp in groups.values():
        baseline = _baseline_values(grp, allowed_to_vary=allowed)
        for cfg, evs in grp:
            if all(_hashable(cfg.get(k)) == expected
                   for k, expected in baseline.items()):
                kept.append((cfg, evs))
    return kept


def filter_variants(runs, *, on: tuple[str, ...]):
    """Mirror of :func:`filter_baseline`. Keep runs where AT LEAST ONE of
    the fields in ``on`` is off its argparse default. Use in dedicated
    variant-investigation cells.
    """
    from lora_playground.loader import _argparse_defaults
    defaults = _argparse_defaults()
    kept = []
    for cfg, evs in runs:
        if any(k in defaults and _hashable(cfg.get(k)) != _hashable(defaults[k])
               for k in on):
            kept.append((cfg, evs))
    return kept
