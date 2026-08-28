"""Series-identity helpers, label-collision detection, and per-group
baseline/variant filtering used by the high-level figure entry points.

Key contracts enforced here:
  - ``series_id(cfg)`` defines the equivalence class within which runs may be
    averaged (seeds, lr-grid points, horizon extensions of the same algorithm).
  - ``assert_label_discriminates`` raises if a display-label function collapses
    distinct ``series_id`` values into one bucket — the silent-averaging
    failure mode this contract exists to forbid.
  - ``assert_curve_source_coherent`` raises if one displayed learning-rate
    curve silently joins runs produced by different execution-source snapshots.
  - ``filter_baseline`` / ``filter_variants`` hold non-target axes to the
    train.py argparse default within each per-axis group.
"""
from __future__ import annotations

from functools import lru_cache


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


class SourceCoherenceError(ValueError):
    """Raised when one displayed curve mixes unaudited execution sources.

    ``execution_source_sha`` is content-based and scoped to the training
    execution closure. It is deliberately a runtime field, so same-cell reruns
    still deduplicate newest/longest-first rather than becoming two algorithm
    series. Once those cells are deduplicated, however, joining different
    source hashes across learning rates can splice two algorithm versions into
    one curve. This error guards that second boundary.
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


@lru_cache(maxsize=1)
def _shipped_defaults() -> dict:
    """``{field: default}`` for every cfg field whose default the CURRENT code
    declares: ``OptimizerConfig``, train.py's parser, and the two run-schema
    revision counters.

    DERIVED, never typed. A literal copy of a default goes stale the moment the
    real default moves, and then every run looks off-default — that already cost
    a leaderboard regeneration once (see the `cw_metric_init` note in
    `labels._shared_knobs`).
    """
    from ..run_schema import DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION
    from .arms import _cli_defaults, _config_defaults
    return {
        **_config_defaults(),
        **_cli_defaults(),
        # `train.py` stamps these two only from run_schema.RUN_SCHEMA_VERSION 2
        # onward, so an older run records neither. Revision 1 is the first value
        # of both counters and the value `optimizer_implementation_revision`
        # returns for a class that declares none, i.e. what an unversioned run
        # ran under -- so recording revision 1 and not recording it at all are
        # the same series, while a LATER revision still splits.
        # `comparison._recorded_revision` resolves the same two fields the same
        # way for the measurement/pipeline conflict check.
        "optimizer_impl_revision": DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION,
        "measurement_semantics_revision": DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION,
    }


@lru_cache(maxsize=None)
def _constructor_defaults(optimizer: str | None) -> dict:
    """``{constructor kwarg: default}`` for the class the named optimizer builds.

    The flattened cfg carries the run's recorded ``optimizer_config`` block, i.e.
    the CONSTRUCTOR's kwargs — spellings like ``ns_steps`` or ``flat_outer`` that
    are not ``OptimizerConfig`` fields and so have no default in
    `_shipped_defaults`. The class signature is where their default lives, with
    the spec's per-variant constants layered on top (those are what the variant
    actually passes). ``{}`` for an unregistered name or a ``build``-callable
    spec, which is conservative: nothing is normalized away.
    """
    import inspect
    if not optimizer:
        return {}
    try:
        from ..optim_specs import REGISTRY
    except Exception:
        return {}
    spec = REGISTRY.get(optimizer)
    if spec is None or spec.cls is None:
        return {}
    try:
        sig = inspect.signature(spec.cls.__init__)
    except (TypeError, ValueError):
        return {}
    defaults = {p.name: p.default for p in sig.parameters.values()
                if p.default is not inspect.Parameter.empty}
    defaults.update(spec.defaults or {})
    defaults.update(spec.fixed or {})
    return defaults


def _recorded_block_keys(cfg: dict) -> frozenset:
    """Keys the run recorded inside its own optimizer blocks.

    ``run_records.logged_effective_config`` flattens ``optimizer_config`` and
    ``optimizer_effective`` onto the cfg, so those blocks' keys arrive as
    top-level scalars alongside train.py's CLI arguments.
    """
    keys: set[str] = set()
    for block in ("optimizer_config", "optimizer_effective"):
        b = cfg.get(block)
        if isinstance(b, dict):
            keys |= {k for k in b if isinstance(k, str)}
    return frozenset(keys)


def _series_items(cfg: dict, axis_fields: frozenset[str]):
    """Yield the ``(field, hashable value)`` pairs that define a run's series.

    A field contributes only when the run recorded a value that DIFFERS from
    what the current code does by default. Three exclusions, in order:

      1. ``None`` — "field not present": an older run whose schema didn't
         include flag X is informationally equivalent to a newer run with
         ``X=None``.
      2. a value equal to the field's derived default — the same argument one
         step further. The docstring's own justification for (1) is that both
         "fall through to argparse default at runtime"; a run that RECORDS the
         default did exactly that, so `X=<default>` and absent are one series.
         Without this, every old-vs-new cfg pair splits purely on schema growth
         (measured: 43 colliding buckets across 6 leaderboard cells, on fields
         like `cw_solved_rho=False` and `rdinv_variant='A'`).
      3. a field the run recorded only inside its optimizer blocks that the
         run's optimizer constructor NO LONGER ACCEPTS — a retired or
         side-branch knob (`cw_picard_mode`, `cw_no_rr_precond`,
         `freeze_factorwise_slots`). It has no default to compare against and
         names no behaviour the current code can express, so it is the record of
         a code revision, not a choice: `arms._inert_fields`' rule that "a field
         the constructor does not accept is provenance, not an axis".

    Everything else is series-defining BY DEFAULT — an off-default value, and an
    unrecognized field the current schema has never seen, both split the series.
    Explicit off-default ``False`` / ``0`` / ``""`` are series-defining.
    """
    shipped = _shipped_defaults()
    ctor = _constructor_defaults(cfg.get("optimizer"))
    block_keys = None
    for k, v in cfg.items():
        # Underscore-prefixed keys are loader/parser namespaces (`_derived`,
        # `_cli_args`, `_optim_steps`); dict values are the composite blocks.
        # Both mirror source-of-truth scalars and would double-count.
        if (k in axis_fields or k.startswith("_")
                or isinstance(v, dict) or v is None):
            continue
        h = _hashable(v)
        if k in shipped:
            if h == _hashable(shipped[k]):
                continue
        elif k in ctor:
            if h == _hashable(ctor[k]):
                continue
        else:
            if block_keys is None:
                block_keys = _recorded_block_keys(cfg)
            if k in block_keys:
                continue
        yield k, h


def series_id(cfg: dict, *, axis_fields: frozenset[str] | None = None) -> frozenset:
    """Mechanical series identity = cfg minus SERIES_AXIS_FIELDS, minus every
    field that carries no information relative to the shipped default.

    Two cfgs with the same series_id are seeds / lr-grid points / horizon
    extensions of the same algorithm at the same model config and may be
    averaged together. Distinct series_ids cannot be averaged regardless
    of what a display-label function returns — see
    `assert_label_discriminates`. `_series_items` documents exactly which
    fields are dropped and why.
    """
    if axis_fields is None:
        from lora_playground.manifest import SERIES_AXIS_FIELDS
        axis_fields = SERIES_AXIS_FIELDS
    return frozenset(_series_items(cfg, axis_fields))


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
        # Read through `_series_items` — the SAME normalization series_id used —
        # so the reported diff is a field that actually split the bucket, not a
        # schema-growth artifact the identity already discounted.
        norm = [dict(_series_items(c, axis_fields)) for c in cfgs]
        all_keys: set[str] = set().union(*(set(n) for n in norm))
        differing: dict[str, set] = {}
        for k in all_keys:
            vals = {n.get(k) for n in norm}
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


def _execution_source(cfg: dict) -> tuple[str, str] | None:
    """Strongest available execution-source identifier for one run."""
    source_sha = cfg.get("execution_source_sha")
    if source_sha:
        return "execution_source_sha", str(source_sha)
    git_commit = cfg.get("git_commit")
    if git_commit:
        return "git_commit", str(git_commit)
    return None


def filter_curve_sources(runs, group_key_fn, allowed_sources_by_label: dict):
    """Filter selected labels to exact execution sources, with a safe labeler.

    Returns ``(kept_runs, excluded_runs, checked_label_fn)``. Labels absent from
    ``allowed_sources_by_label`` are unchanged. A constrained label keeps only
    an exact ``execution_source_sha`` match; missing provenance is an error.
    ``checked_label_fn`` repeats the check and raises on an excluded source, so
    accidentally passing the original unfiltered run list to a downstream
    figure fails instead of silently restoring the rejected cohort.
    """
    allowed = {
        label: {str(source) for source in sources}
        for label, sources in allowed_sources_by_label.items()
    }

    def checked_label(cfg):
        label = group_key_fn(cfg)
        if label is None or label not in allowed:
            return label
        source = cfg.get("execution_source_sha")
        if not source:
            raise SourceCoherenceError(
                f"label={label!r} is source-restricted but the run has no "
                "execution_source_sha"
            )
        if str(source) not in allowed[label]:
            raise SourceCoherenceError(
                f"label={label!r} execution_source_sha={source} is not in the "
                f"allowed set {sorted(allowed[label])}"
            )
        return label

    kept, excluded = [], []
    for run in runs:
        cfg, _hist = run
        label = group_key_fn(cfg)
        if label not in allowed:
            kept.append(run)
            continue
        try:
            checked_label(cfg)
        except SourceCoherenceError as exc:
            if not cfg.get("execution_source_sha"):
                raise
            excluded.append((run, str(exc)))
        else:
            kept.append(run)
    return kept, excluded, checked_label


def assert_curve_source_coherent(
    runs,
    group_key_fn,
    *,
    bucket_keys: tuple[str, ...] = (
        "model_name", "data_dir", "data_pipeline_version", "lora_r",
    ),
    equivalent_source_groups: dict | None = None,
) -> None:
    """Hard-fail when one displayed LR curve spans unaudited source snapshots.

    Curves are grouped by ``(group_key_fn(cfg), *bucket_keys)``. Within a
    curve, every non-empty run must resolve to one ``execution_source_sha``
    (or, for older logs, one ``git_commit``). This is intentionally separate
    from :func:`series_id`: source provenance must *not* split same-cell reruns
    before the loader's newest/longest-wins dedup, but it must be coherent when
    the surviving LR cells are joined into a line.

    ``equivalent_source_groups`` is the explicit audit escape hatch. It maps a
    displayed label to an iterable of exact source-value groups, for example
    ``{"one-sided": [{"old-source-sha", "new-source-sha"}]}``. Only that
    label receives the equivalence; the same two hashes remain a collision for
    every other algorithm. Keep groups exact and narrow: this records a code-
    review conclusion that the source change is inert for that displayed arm.

    Empty histories and ``None`` labels are skipped because the figure path
    does not display them. Missing provenance raises rather than silently
    treating all unknown sources as equivalent.
    """
    from collections import defaultdict

    equivalent_source_groups = equivalent_source_groups or {}
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for cfg, evals in runs:
        if not evals:
            continue
        label = group_key_fn(cfg)
        if label is None:
            continue
        bucket = tuple(_hashable(cfg.get(k)) for k in bucket_keys)
        buckets[(_hashable(label), bucket)].append(cfg)

    failures: list[str] = []
    for (_label_key, bucket), cfgs in buckets.items():
        label = group_key_fn(cfgs[0])
        equivalence_groups = [
            {str(source) for source in group}
            for group in equivalent_source_groups.get(label, ())
        ]

        resolved: dict[tuple, dict[str, set]] = defaultdict(
            lambda: {"sources": set(), "lrs": set(), "commits": set()}
        )
        missing_lrs: set = set()
        for cfg in cfgs:
            source = _execution_source(cfg)
            if source is None:
                missing_lrs.add(cfg.get("lr"))
                continue
            source_field, source_value = source
            equivalence_idx = next(
                (i for i, group in enumerate(equivalence_groups)
                 if source_value in group),
                None,
            )
            source_key = (("equivalence", equivalence_idx)
                          if equivalence_idx is not None else source)
            entry = resolved[source_key]
            entry["sources"].add((source_field, source_value))
            entry["lrs"].add(cfg.get("lr"))
            if cfg.get("git_commit"):
                entry["commits"].add(str(cfg["git_commit"]))

        bucket_desc = dict(zip(bucket_keys, bucket))
        if missing_lrs:
            failures.append(
                f"  label={label!r} bucket={bucket_desc}: missing both "
                f"execution_source_sha and git_commit at "
                f"lr={sorted(missing_lrs, key=repr)}"
            )
        if len(resolved) > 1:
            source_lines = []
            for entry in resolved.values():
                source_desc = ", ".join(
                    f"{field}={value}" for field, value in sorted(entry["sources"])
                )
                source_lines.append(
                    f"[{source_desc}] lr={sorted(entry['lrs'], key=repr)} "
                    f"git_commit={sorted(entry['commits'])}"
                )
            failures.append(
                f"  label={label!r} bucket={bucket_desc}: "
                + "; ".join(source_lines)
            )

    if failures:
        raise SourceCoherenceError(
            "displayed learning-rate curve mixes execution sources whose "
            "semantic equivalence has not been audited:\n"
            + "\n".join(failures)
            + "\nFix: keep only runs from one semantic source, or pass "
              "equivalent_source_groups={label: [{sha_a, sha_b}]} after "
              "code-reviewing that exact source change for that arm."
        )


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
            continue
        # Two runs under one key are a CONTINUATION (same series, resumed or
        # re-run) only if they agree outside SERIES_AXIS_FIELDS. When they do not,
        # collapsing them discards a real result: keep_longest would pick whichever
        # trained further, with no error and no notice. Measured on
        # Llama-3.2-1B/openmath/r256 before canonical_label was made to
        # discriminate: six buckets collapsed distinct series, one of them
        # `AdamW ... lr=1e-4` covering six -- the beta2 grid -- of which five were
        # silently dropped and a leaderboard rendered cleanly on the survivor.
        if series_id(cfg) != series_id(best[key][0]):
            raise LabelCollisionError(
                f"dedup_by_canonical: two runs share the dedup key {key!r} but are "
                f"DIFFERENT series, so collapsing them would discard a result.\n"
                f"  differing fields: {_series_diff(cfg, best[key][0])}\n"
                f"Fix: make canonical_label discriminate on one of those fields "
                f"(labels._residual_knobs derives the suffix from "
                f"arms.PINNED_FIELDS(), so a field it misses is one listed in "
                f"labels._LABELLED_ELSEWHERE), OR add the field to "
                f"manifest.SERIES_AXIS_FIELDS if it is a true per-series axis."
            )
        if keep_longest and _last_step(evals) > _last_step(best[key][1]):
            best[key] = (cfg, evals)
    return [best[k] for k in order]


def _series_diff(a: dict, b: dict) -> str:
    """Fields that make two cfgs different series, for the error message.

    Compares through `_series_items` so only fields that genuinely split the
    identity are named, and prints the RECORDED values so the reader can see
    what each run logged.
    """
    from ..manifest import SERIES_AXIS_FIELDS
    na = dict(_series_items(a, SERIES_AXIS_FIELDS))
    nb = dict(_series_items(b, SERIES_AXIS_FIELDS))
    diffs = [f"{k}={a.get(k)!r} vs {b.get(k)!r}"
             for k in sorted(set(na) | set(nb)) if na.get(k) != nb.get(k)]
    return ", ".join(diffs[:6]) + ("..." if len(diffs) > 6 else "")


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
