"""Logic tests for the loader / filter / backfill machinery used by
`notebooks/leaderboard_old/packed_v1_leaderboard.ipynb`.

Scope: rules that should always hold regardless of which sweeps happen
to be on disk — label-collision contracts, EXCLUDED_COMMITS application,
HISTORICAL_DEFAULTS_WHEN_MISSING backfill, and filter_baseline's variant
exclusion. Data-state alarms (run counts, leaderboard quality, specific
cell coverage) were removed in favor of testing logic over state.
"""
import pytest

from lora_playground.loader import load_runs
from lora_playground.plotting import (
    assert_label_discriminates,
    filter_baseline,
    series_id,
)


CANDIDATE_FAMILY = [
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
    "adam-polar-product-lora-coupled-spectral-chord-direction",
    "adam-polar-product-lora-coupled",
]
WHITEN, NOWHITEN, DIRECT, FROB = CANDIDATE_FAMILY

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
    return load_runs(
        where={
            "optimizer": CANDIDATE_FAMILY,
            "data_pipeline_version": "packed_v1",
            "max_steps": 4000,
        },
        warn_cross_commit=False,
    )


@pytest.fixture(scope="module")
def ref_runs():
    return load_runs(
        where={"optimizer": "adamw", "data_pipeline_version": "packed_v1"},
        warn_cross_commit=False,
    )


def test_label_discrimination_holds_on_filtered_run_set(cand_runs):
    """The series-id contract must pass on the run set the plot cells
    actually consume (post-filter_baseline). If two distinct cfgs collapse
    to the same group_key, the leaderboard would silently overlay them."""
    clean = filter_baseline(
        cand_runs, varying=("optimizer", "effective_picard_iters"),
    )
    assert_label_discriminates(clean, group_key)


def test_filter_baseline_excludes_variant_runs(cand_runs):
    """filter_baseline(varying=(optimizer, effective_picard_iters)) must
    drop runs that deviate on any other axis — Init[AB], eps_rel damping,
    etc. Those belong to dedicated variant cells, not the baseline panel."""
    clean = filter_baseline(
        cand_runs, varying=("optimizer", "effective_picard_iters"),
    )
    for c, _ in clean:
        assert (c.get("lora_init_b") or "zero") == "zero", (
            f"Init[AB] leaked into baseline-clean: optimizer={c['optimizer']} "
            f"lr={c['lr']} init_b={c.get('lora_init_b')}"
        )
        assert not c.get("precond_delta_relative"), (
            f"eps_rel-damped run leaked into baseline-clean: "
            f"optimizer={c['optimizer']} lr={c['lr']}"
        )


def test_no_stale_commit_survivors(cand_runs):
    """COMMIT_EXCLUSIONS must keep all known-stale runs out of cand_runs.
    If a stale commit's runs reappear, the loader exclusion is broken or
    a new stale commit slipped in undetected."""
    from lora_playground.commit_exclusions import COMMIT_EXCLUSIONS
    for cfg, _ in cand_runs:
        commit = cfg.get("git_commit") or ""
        for prefix, _reason in COMMIT_EXCLUSIONS:
            assert not commit.startswith(prefix), (
                f"Stale-commit run leaked: commit={commit[:7]} "
                f"group={cfg.get('log_group')!r} — COMMIT_EXCLUSIONS not honored"
            )


def test_historical_defaults_when_missing_applied():
    """For runs with `data_pipeline_version` missing in cfg, the loader
    must backfill with the historical default ('unpacked_v0'), NOT the
    current argparse default ('packed_v1'). This is the regression guard
    for the data-hygiene bug found in cell 11 Adam-baseline erratic
    behavior."""
    from lora_playground.loader import (
        HISTORICAL_DEFAULTS_WHEN_MISSING, _argparse_defaults,
    )
    assert "data_pipeline_version" in HISTORICAL_DEFAULTS_WHEN_MISSING
    assert HISTORICAL_DEFAULTS_WHEN_MISSING["data_pipeline_version"] == "unpacked_v0"
    defaults = _argparse_defaults()
    assert defaults.get("data_pipeline_version") != "unpacked_v0", (
        "data_pipeline_version argparse default is now 'unpacked_v0' — "
        "remove the HISTORICAL_DEFAULTS_WHEN_MISSING entry."
    )


def test_no_seed_collision_within_series(cand_runs):
    """Silent-merge guard: within each series_id, every seed value must
    appear at most once. If two distinct cfgs collapse to the same
    series_id but report the same seed, the leaderboard will treat them
    as duplicate seed runs and either silently dedup one or — worse —
    average their losses as seed replicates, producing a phantom spike.

    Failure modes this catches:
      - new optimizer kwarg added; not in SERIES_AXIS_FIELDS; two runs
        differing only on the kwarg collapse to one series_id.
      - argparse default flips; HISTORICAL_DEFAULTS_WHEN_MISSING entry
        forgotten; old + new runs land in the same series_id.
      - loader dedup regression leaves two physical runs of the same
        config under different log paths.
    """
    # Key on every axis the leaderboard plot pins per cell:
    #   - series_id (which collapses across SERIES_AXIS_FIELDS by design)
    #   - lr, lora_r (the panel coordinates — part of SERIES_AXIS_FIELDS but
    #     re-fanned-out in the plot, so two runs sharing them are treated as
    #     replicates within one panel cell)
    #   - seed (the replicate axis the plot averages over)
    # Multiple runs in the same bucket means the plot averaged them as if
    # they were duplicate seed runs.
    by_cell: dict[tuple, list] = {}
    for cfg, _ in cand_runs:
        sid = series_id(cfg)
        cell = (sid, cfg.get("lr"), cfg.get("lora_r"), cfg.get("seed"))
        by_cell.setdefault(cell, []).append(cfg)
    collisions = []
    for cell, cfgs in by_cell.items():
        if len(cfgs) <= 1:
            continue
        sample = cfgs[0]
        opt = sample.get("optimizer")
        _, lr, r, seed = cell
        groups = sorted({c.get("log_group") for c in cfgs})
        commits = sorted({(c.get("git_commit") or "?")[:7] for c in cfgs})
        collisions.append(
            f"  optimizer={opt} r={r} lr={lr} seed={seed}: "
            f"{len(cfgs)} cfgs share series_id at this panel cell; "
            f"groups={groups} commits={commits}"
        )
    assert not collisions, (
        f"{len(collisions)} (series_id, lr, lora_r, seed) collisions — "
        f"distinct cfgs collapsed to the same panel cell with the same seed "
        f"will be silently deduped or averaged as replicates. First few:\n"
        + "\n".join(collisions[:5])
    )


def test_adamw_reference_excludes_legacy_pipeline_groups(ref_runs):
    """Regression: argparse-default backfill was tagging old unpacked_v0
    runs (missing the field) as packed_v1, polluting packed_v1 queries
    with stale-pipeline data. After the HISTORICAL_DEFAULTS_WHEN_MISSING
    fix, no AdamW packed_v1 run at the canonical horizon should come
    from a sweep group whose name marks it as a different pipeline."""
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
