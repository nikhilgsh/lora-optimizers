"""Cheap correctness checks for `notebooks/packed_v1_leaderboard.ipynb`.

The notebook itself takes ~30-60s to fully execute and produces image
outputs that are hard to inspect programmatically. This test instead
re-loads `cand_runs` the same way the notebook does, runs the same
filters, and asserts on the resulting run/series counts. Fast (~5s)
and surfaces breakage from upstream changes to the loader, filter
helpers, or sweep contents.

When adding sweep data that changes these counts, update the literals
below — the assertion should fire as a deliberate review prompt.
"""
from collections import Counter

import pytest

from lora_playground.loader import load_runs
from lora_playground.plot_utils import (
    assert_label_discriminates,
    filter_baseline,
    filter_variants,
    series_id,
)


CANDIDATE_FAMILY = [
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
    "adam-polar-product-lora-coupled-spectral-chord-direction",
    "adam-polar-product-lora-coupled",
]
WHITEN = CANDIDATE_FAMILY[0]
NOWHITEN = CANDIDATE_FAMILY[1]
DIRECT = CANDIDATE_FAMILY[2]
FROB = CANDIDATE_FAMILY[3]

# Match cell f4de1aa2's group_key — keep in sync.
SHORT = {
    WHITEN:   "chord-tight",
    NOWHITEN: "chord-tight (no-whiten)",
    DIRECT:   "chord-direction",
}


def _damping_tag(cfg):
    pdr = bool(cfg.get("precond_delta_relative"))
    pd = cfg.get("precond_delta")
    if pd is None:
        return ""
    if pdr:
        return f" [eps_rel={pd:g}]"
    if abs(float(pd) - 1e-6) < 1e-12:
        return ""
    return f" [d={pd:g} abs]"


def group_key(cfg):
    opt = cfg["optimizer"]
    k = cfg.get("_derived", {}).get("effective_picard_iters", "?")
    init_b = cfg.get("lora_init_b") or "zero"
    init_tag = ("" if init_b == "zero"
                else " [Init[AB]]" if init_b == "symmetric"
                else f" [{init_b}]")
    damp_tag = _damping_tag(cfg)
    if opt == FROB:
        return f"frob k={k}{init_tag}{damp_tag}"
    base = SHORT.get(opt, opt)
    return f"{base} k={k}{init_tag}{damp_tag}"


@pytest.fixture(scope="module")
def cand_runs():
    runs = load_runs(
        where={
            "optimizer": CANDIDATE_FAMILY,
            "data_pipeline_version": "packed_v1",
            "max_steps": 4000,
        },
        warn_cross_commit=False,
    )
    return runs


@pytest.fixture(scope="module")
def ref_runs():
    return load_runs(
        where={"optimizer": "adamw", "data_pipeline_version": "packed_v1"},
        warn_cross_commit=False,
    )


def test_cand_runs_lower_bound(cand_runs):
    # Sanity check: there's data. Lower bound, not exact — sweeps in flight grow this.
    assert len(cand_runs) >= 100, f"cand_runs={len(cand_runs)} unexpectedly low"


def test_ref_runs_present(ref_runs):
    # Threshold post-HISTORICAL_DEFAULTS_WHEN_MISSING fix: only true
    # packed_v1 runs (explicitly logged) are counted. Pre-fix this would
    # bloat to ~140 with unpacked_v0 runs mis-tagged as packed_v1.
    assert len(ref_runs) >= 20, f"ref_runs={len(ref_runs)} unexpectedly low"


def test_cell11_competitive_supplementary_split(cand_runs):
    """Cell 7876dcfe: filter_baseline + competitive/supplementary partition.
    Both partitions must be non-empty; competitive must have all 6
    (variant, k) cells represented; supplementary must contain frob runs."""
    COMPETITIVE_VARIANT_K = {
        (WHITEN, 1), (WHITEN, 3),
        (NOWHITEN, 1), (NOWHITEN, 3),
        (DIRECT, 1), (DIRECT, 3),
    }

    def is_competitive(cfg):
        k = cfg.get("_derived", {}).get("effective_picard_iters")
        return (cfg["optimizer"], k) in COMPETITIVE_VARIANT_K

    clean = filter_baseline(
        cand_runs, varying=("optimizer", "effective_picard_iters"),
    )
    competitive = [r for r in clean if is_competitive(r[0])]
    supplementary = [r for r in clean if not is_competitive(r[0])]

    assert len(clean) >= 80, f"clean_runs={len(clean)} unexpectedly low"
    assert len(competitive) >= 60, f"competitive={len(competitive)}"
    # Supplementary (frob) is sparse at packed_v1 4k — only the few runs
    # at the project's canonical lr=3e-4 anchor exist post-exclusion.
    assert len(supplementary) >= 4, f"supplementary={len(supplementary)} — frob runs missing"

    # All 6 competitive (variant, k) cells present
    present_cells = {(c["optimizer"], c.get("_derived", {}).get("effective_picard_iters"))
                     for c, _ in competitive}
    assert COMPETITIVE_VARIANT_K <= present_cells, (
        f"Missing competitive cells: {COMPETITIVE_VARIANT_K - present_cells}")

    # Supplementary contains only frob
    sup_opts = {c["optimizer"] for c, _ in supplementary}
    assert sup_opts == {FROB}, f"supplementary should be frob-only, got {sup_opts}"

    # Variant exclusion check: no Init[AB] or epsrel runs in clean
    for c, _ in clean:
        assert (c.get("lora_init_b") or "zero") == "zero", \
            f"Init[AB] leaked into clean: lr={c['lr']}"
        assert not c.get("precond_delta_relative"), \
            f"eps_rel leaked into clean: lr={c['lr']}"


def test_cell13_init_ab_envelope_has_all_four_series(cand_runs):
    """Cell de6ef354: Init[A] vs Init[AB] at r=256 k=1 across both
    whiten and no-whiten. All four series must be present, otherwise
    the comparison plot is silently missing a curve."""
    def in_env(c):
        return (c.get("lora_r") == 256
                and c.get("_derived", {}).get("effective_picard_iters") == 1
                and c["optimizer"] in (WHITEN, NOWHITEN))

    env = [(c, e) for c, e in cand_runs if in_env(c)]
    assert len(env) >= 20, f"envelope={len(env)} unexpectedly low"

    init_var = filter_variants(env, on=("lora_init_b",))
    init_base = filter_baseline(env, varying=("optimizer",))
    combined = init_var + init_base

    series = Counter(group_key(c) for c, _ in combined)
    expected_series = {
        "chord-tight k=1",                         # WHITEN Init[A]
        "chord-tight k=1 [Init[AB]]",              # WHITEN Init[AB]
        "chord-tight (no-whiten) k=1",             # NOWHITEN Init[A]
        "chord-tight (no-whiten) k=1 [Init[AB]]",  # NOWHITEN Init[AB]
    }
    missing = expected_series - set(series)
    assert not missing, (
        f"Init[A]/Init[AB] cell missing series: {missing}. "
        f"Present: {dict(series)}"
    )
    # Each series should have at least 2 lr points to be informative
    for s in expected_series:
        assert series[s] >= 2, f"series {s!r} has only {series[s]} runs (<2 lrs)"


def test_cell15_eps_rel_runs_present(cand_runs):
    """Cell fd89189b: eps_rel sweep at r=256 chord-tight whiten k=1
    Init[A] requires at least the default-damping baseline + multiple
    eps_rel values."""
    eps_runs = [
        (c, e) for c, e in cand_runs
        if c.get("optimizer") == WHITEN
        and c.get("lora_r") == 256
        and c.get("_derived", {}).get("effective_picard_iters") == 1
        and (c.get("lora_init_b") or "zero") == "zero"
        and (e and e[-1].get("step") == 4000)
    ]
    assert len(eps_runs) >= 8, f"eps_rel envelope: {len(eps_runs)} runs"

    pdr_values = Counter(bool(c.get("precond_delta_relative")) for c, _ in eps_runs)
    assert pdr_values[False] >= 3, "Need default-damping baseline runs"
    assert pdr_values[True] >= 3, "Need eps_rel variant runs"


def test_label_discrimination_holds_on_filtered_run_set(cand_runs):
    """The series-id contract must pass on the run set the plot cells
    actually consume (post-filter_baseline)."""
    clean = filter_baseline(
        cand_runs, varying=("optimizer", "effective_picard_iters"),
    )
    # No collision — raises LabelCollisionError if violated.
    assert_label_discriminates(clean, group_key)


def test_no_stale_commit_survivors(cand_runs):
    """EXCLUDED_COMMITS in manifest must keep all known-stale runs out of
    cand_runs. If a stale commit's runs reappear, the loader exclusion
    is broken or a new stale commit slipped in undetected."""
    from lora_playground.manifest import EXCLUDED_COMMITS
    for cfg, _ in cand_runs:
        commit = cfg.get("git_commit") or ""
        for prefix in EXCLUDED_COMMITS:
            assert not commit.startswith(prefix), (
                f"Stale-commit run leaked: commit={commit[:7]} "
                f"group={cfg.get('log_group')!r} — EXCLUDED_COMMITS not honored"
            )


def test_competitive_no_loss_outliers_at_canonical_horizon(cand_runs):
    """For every (variant, k, rank) cell in the competitive partition,
    EVERY run completing the canonical horizon (4000 steps) must reach
    a healthy final loss (<0.65). Anything ≥0.65 for these algorithms at
    4k steps indicates a stale-commit regression — the run would render
    as a clipped point off the y-axis (silent empty curve in the legend).

    This is the test that catches the empty r=32/r=128 panels and missing
    chord-tight k=3 curves at r=16/256 that surfaced when EXCLUDED_COMMITS
    didn't yet cover the stale commits."""
    COMPETITIVE = {
        (WHITEN, 1), (WHITEN, 3),
        (NOWHITEN, 1), (NOWHITEN, 3),
        (DIRECT, 1), (DIRECT, 3),
    }
    HEALTHY_MAX = 0.65  # chord variants at packed_v1 4k reach 0.51-0.55
    violators = []
    for cfg, evs in cand_runs:
        k = cfg.get("_derived", {}).get("effective_picard_iters")
        if (cfg["optimizer"], k) not in COMPETITIVE:
            continue
        if not evs or evs[-1].get("step") != 4000:
            continue
        loss = evs[-1].get("eval_loss")
        if loss is None or loss < HEALTHY_MAX:
            continue
        violators.append((
            cfg["optimizer"], k, cfg.get("lora_r"), cfg["lr"],
            (cfg.get("git_commit") or "?")[:7],
            cfg.get("log_group"), loss,
        ))
    assert not violators, (
        f"{len(violators)} competitive runs at 4k have loss ≥ {HEALTHY_MAX} "
        f"— likely stale-commit regression. First few:\n"
        + "\n".join(f"  {v}" for v in violators[:5])
    )


def test_every_competitive_cell_has_at_least_one_visible_run(cand_runs):
    """Coverage: for every (variant, k, rank) cell with ANY presence in
    cand_runs, at least one run at the canonical horizon must reach
    loss < 0.65 — otherwise the panel's curve is invisible (clipped)
    and the legend entry has no data behind it."""
    COMPETITIVE = {
        (WHITEN, 1), (WHITEN, 3),
        (NOWHITEN, 1), (NOWHITEN, 3),
        (DIRECT, 1), (DIRECT, 3),
    }
    healthy_at = {}  # (opt, k, rank) -> bool any healthy
    present_at = {}  # (opt, k, rank) -> bool any run
    for cfg, evs in cand_runs:
        k = cfg.get("_derived", {}).get("effective_picard_iters")
        cell = (cfg["optimizer"], k, cfg.get("lora_r"))
        if (cfg["optimizer"], k) not in COMPETITIVE:
            continue
        present_at[cell] = True
        if evs and evs[-1].get("step") == 4000:
            loss = evs[-1].get("eval_loss")
            if loss is not None and loss < 0.65:
                healthy_at[cell] = True
    invisible = sorted(set(present_at) - set(healthy_at))
    assert not invisible, (
        f"{len(invisible)} competitive (variant, k, rank) cells have runs "
        f"loaded but none with healthy loss < 0.65 — these render as empty "
        f"curves with legend entries. Cells: {invisible}"
    )


def test_per_rank_competitive_coverage(cand_runs):
    """For each rank present in the competitive partition, at least 2
    distinct (variant, k) cells must have visible (healthy) data. A rank
    panel with only one variant rendering is not a comparison."""
    from collections import defaultdict
    from lora_playground.plot_utils import filter_baseline
    COMPETITIVE = {
        (WHITEN, 1), (WHITEN, 3),
        (NOWHITEN, 1), (NOWHITEN, 3),
        (DIRECT, 1), (DIRECT, 3),
    }
    clean = filter_baseline(
        cand_runs, varying=("optimizer", "effective_picard_iters"),
    )
    by_rank: dict = defaultdict(set)
    for cfg, evs in clean:
        k = cfg.get("_derived", {}).get("effective_picard_iters")
        if (cfg["optimizer"], k) not in COMPETITIVE:
            continue
        if not evs or evs[-1].get("step") != 4000:
            continue
        if evs[-1].get("eval_loss", 1) >= 0.65:
            continue
        by_rank[cfg.get("lora_r")].add((cfg["optimizer"], k))
    weak_ranks = [r for r, cells in by_rank.items() if len(cells) < 2]
    assert not weak_ranks, (
        f"Ranks with <2 healthy competitive (variant, k) cells "
        f"render as single-line panels: {weak_ranks}. "
        f"Per-rank cell coverage: { {r: sorted(c) for r, c in by_rank.items()} }"
    )


def test_adamw_reference_present_at_competitive_ranks(cand_runs, ref_runs):
    """For every rank that has competitive runs at 4k, AdamW reference
    must have AT LEAST 3 complete (step=4000) runs so the left-panel
    η-sweep curve is informative (not a single dot or a 2-point line)."""
    from collections import defaultdict
    comp_ranks = set()
    for cfg, evs in cand_runs:
        if evs and evs[-1].get("step") == 4000:
            comp_ranks.add(cfg.get("lora_r"))
    ref_by_rank: dict = defaultdict(int)
    for cfg, evs in ref_runs:
        if evs and evs[-1].get("step") == 4000:
            ref_by_rank[cfg.get("lora_r")] += 1
    weak = {r: ref_by_rank.get(r, 0) for r in comp_ranks if ref_by_rank.get(r, 0) < 3}
    assert not weak, (
        f"Ranks with <3 complete AdamW reference runs at 4k will render "
        f"erratic left-panel curves: {weak}"
    )


def test_adamw_reference_only_packed_v1_runs(ref_runs):
    """Regression: argparse-default backfill was tagging old unpacked_v0
    runs (missing the field) as packed_v1, polluting packed_v1 queries
    with stale-pipeline data. The fix uses HISTORICAL_DEFAULTS_WHEN_MISSING
    to backfill the historical default.

    Test shape: at every rank with AdamW data, the BEST run must reach
    healthy loss (<0.60). Diverged-lr runs (legitimately bad lrs in an
    lr sweep) are allowed; they show where the optimum bounds are.
    Stale-pipeline pollution shows up as ranks where every run is bad
    or where the step count is anomalous.
    """
    from collections import defaultdict
    # Bucket by rank, only 4k-step runs
    by_rank: dict = defaultdict(list)
    for cfg, evs in ref_runs:
        if not evs or evs[-1].get("step") != 4000:
            continue
        loss = evs[-1].get("eval_loss")
        if loss is None:
            continue
        by_rank[cfg.get("lora_r")].append((cfg, loss))
    for r, runs in by_rank.items():
        best = min(loss for _, loss in runs)
        assert best < 0.60, (
            f"AdamW packed_v1 r={r}: best of {len(runs)} runs at 4k is {best:.4f} "
            f"(>= 0.60). All runs:\n"
            + "\n".join(f"  lr={c['lr']} loss={l:.4f} group={c.get('log_group')!r}"
                        for c, l in sorted(runs, key=lambda x: x[1]))
        )

    # Also: no AdamW packed_v1 run at the canonical horizon should have
    # come from a sweep group whose name suggests a different pipeline
    # (e.g. _8k_, _2k_, _supp_) — those are old-pipeline indicators.
    LEGACY_MARKERS = ("_8k_", "_2k_", "_supp_", "_salvage_")
    legacy = []
    for cfg, evs in ref_runs:
        if not evs or evs[-1].get("step") != 4000:
            continue
        group = cfg.get("log_group") or ""
        if any(m in group for m in LEGACY_MARKERS):
            legacy.append((group, cfg.get("lr")))
    assert not legacy, (
        f"AdamW packed_v1 at 4k includes runs from legacy-pipeline groups: {legacy[:5]}"
    )


def test_historical_defaults_when_missing_applied(cand_runs):
    """For runs with `data_pipeline_version` missing in cfg, the loader
    must backfill with the historical default ('unpacked_v0'), NOT the
    current argparse default ('packed_v1'). This is the regression guard
    for the data-hygiene bug found in cell 11 Adam-baseline erratic
    behavior."""
    from lora_playground.loader import (
        HISTORICAL_DEFAULTS_WHEN_MISSING, _argparse_defaults,
    )
    # The map must include data_pipeline_version because its default flipped.
    assert "data_pipeline_version" in HISTORICAL_DEFAULTS_WHEN_MISSING
    assert HISTORICAL_DEFAULTS_WHEN_MISSING["data_pipeline_version"] == "unpacked_v0"
    # The current argparse default differs from the historical, otherwise the
    # override would be a no-op (and the entry can be removed).
    defaults = _argparse_defaults()
    assert defaults.get("data_pipeline_version") != "unpacked_v0", (
        "data_pipeline_version argparse default is now 'unpacked_v0' — "
        "remove the HISTORICAL_DEFAULTS_WHEN_MISSING entry."
    )


def test_no_inflated_seed_sigma_at_k3_r64_lr3e3(cand_runs):
    """Regression: k=3, r=64, lr=3e-3, seed=0 had a phantom outlier
    from a stale-commit log group that wasn't getting deduped because
    `picard_iters_override=3` vs `None` produced distinct loader keys.
    After the loader fix, all surviving same-seed runs at this cell
    should land within the workload sigma floor (~0.001)."""
    hits = [
        (c, e) for c, e in cand_runs
        if c.get("optimizer") == WHITEN
        and c.get("lora_r") == 64
        and c.get("_derived", {}).get("effective_picard_iters") == 3
        and abs(float(c["lr"]) - 3e-3) < 1e-8
        and c.get("seed") == 0
        and (e and e[-1].get("step") == 4000)
    ]
    # The legitimate seed=0 run survives; the buggy commit's seed=0 should
    # have been deduped or remains a separate series_id (different config).
    # Either way, when grouped by series_id, the seed=0 winners shouldn't
    # span >2sigma (~0.002).
    by_id = {}
    for c, e in hits:
        sid = series_id(c)
        by_id.setdefault(sid, []).append(e[-1]["eval_loss"])
    for sid, losses in by_id.items():
        if len(losses) > 1:
            spread = max(losses) - min(losses)
            assert spread <= 0.005, (
                f"seed=0 spread {spread:.4f} > 0.005 at WHITEN k=3 r=64 lr=3e-3; "
                f"losses={losses}. Loader dedup regression suspected."
            )
