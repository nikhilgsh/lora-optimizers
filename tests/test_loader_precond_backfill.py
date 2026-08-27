"""The `precond` backfill in `lora_playground.loader`: derivation, visibility, expiry.

Two backward-compatibility shims live in `loader._backfill_precond`, and each
had a way of going wrong silently.

`_precond_by_optimizer` used to be a hand-written ten-entry table mapping an
optimizer NAME to the `precond` branch its pre-flag runs ran. That duplicated
the `diag_metric` pins in `optim_specs.REGISTRY` with nothing asserting the two
agreed: an eleventh `CurvatureWhitenLoRA` spec would have been absent from the
table, its runs would have carried `precond=None`, and every arm predicate that
pins `precond` (`arms.PROTO`, `arms.NOPRODUCT`, `arms.ONESIDED`, …) would have
skipped them — the arm renders as absent, no error anywhere. It is now DERIVED
from the registry; the tests below pin that derivation against the ten entries
the table held and assert it cannot miss a new spec.

The backfill also MUTATES each cfg away from its own `config.json`: it invents a
`precond` that was never logged and deletes the retired `cw_no_rr_precond`. That
behavior is correct and deliberately unchanged here — the alternative, every arm
predicate accepting `{None, "product"}`, is worse. What it gains is a record of
what it touched (`loader.PRECOND_BACKFILL_MARKER`) and an expiry test, so the
shim is deleted when the last run needing it is gone instead of living forever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground import loader  # noqa: E402


# The mapping `loader._PRECOND_BY_OPTIMIZER` held before it was derived, frozen
# here as the equivalence target. Every run analysed to date was backfilled with
# these values, so the derivation is only a safe replacement if it reproduces
# them exactly. `git log -p -- lora_playground/optim_specs.py` shows no spec's
# `diag_metric` pin has ever changed value, so there is no historical branch the
# current REGISTRY has forgotten.
FROZEN_PRECOND_BY_OPTIMIZER: dict[str, str] = {
    "kl-diag-lora": "product",
    "kl-diag-polar-lora": "product",
    "kl-diag-polar-flatout-lora": "product",
    "kl-diag-flatout-lora": "product",
    "diag-shampoo-lora": "product",
    "diag-shampoo-polar-lora": "product",
    "kl-shampoo-lora": "factorwise",
    "kl-shampoo-polar-lora": "factorwise",
    "curvature-whiten-lora": "factorwise",
    "curvature-whiten-polar-lora": "factorwise",
}


# ─── Shim A: the name → branch map is derived, and reproduces the old table ────

def test_derived_map_reproduces_the_retired_hardcoded_table():
    """The equivalence proof for retiring the table.

    Scoped to the ten names the table covered: a NEW curvature-whiten spec
    appearing in the derived map is the point of deriving it and is not a
    failure here (`test_every_curvature_whiten_spec_is_covered` is what watches
    the full set). What must not change is the branch any of these ten resolves
    to.

    A difference is NOT a licence to edit `FROZEN_PRECOND_BY_OPTIMIZER`: it means
    a spec's `diag_metric` pin moved, which re-labels the branch of every
    already-analysed pre-flag run of that optimizer. Work out which is right
    first.
    """
    derived = loader._derive_precond_by_optimizer()
    covered = {k: derived[k] for k in FROZEN_PRECOND_BY_OPTIMIZER if k in derived}
    assert covered == FROZEN_PRECOND_BY_OPTIMIZER


def test_cached_accessor_agrees_with_the_derivation():
    """`_precond_by_optimizer` reads a JSON snapshot when it is fresh. A stale
    or corrupt snapshot would feed the backfill a wrong branch, so the two must
    agree on the value the loader actually uses."""
    assert loader._precond_by_optimizer() == loader._derive_precond_by_optimizer()


def test_every_curvature_whiten_spec_is_covered():
    """The mechanism, not today's output: membership is `spec.cls is
    CurvatureWhitenLoRA`, so a new variant joins the map on the next import
    whether or not it pins `diag_metric`. This is what the hand-written table
    could not do."""
    from lora_playground import optim as _optim
    from lora_playground.optim_specs import REGISTRY

    cw_specs = {n for n, s in REGISTRY.items() if s.cls is _optim.CurvatureWhitenLoRA}
    assert cw_specs, "no CurvatureWhitenLoRA specs in REGISTRY — has the class moved?"
    assert set(loader._derive_precond_by_optimizer()) == cw_specs


def test_derivation_mirrors_the_constructors_resolution():
    """`CurvatureWhitenLoRA.__init__` resolves an unset `--precond` as
    ``"product" if diag_metric else "factorwise"``. That line is the definition
    of the branch; the map must be exactly it applied to each spec's pin, with
    the constructor's own default where a spec states none."""
    import inspect

    from lora_playground import optim as _optim
    from lora_playground.optim_specs import REGISTRY

    cw = _optim.CurvatureWhitenLoRA
    class_default = inspect.signature(cw.__init__).parameters["diag_metric"].default
    derived = loader._derive_precond_by_optimizer()
    for name, branch in derived.items():
        dm = REGISTRY[name].fixed.get("diag_metric", class_default)
        assert branch == ("product" if dm else "factorwise"), (
            f"{name}: spec pins diag_metric={dm!r} but the map says {branch!r}"
        )


def test_a_new_spec_cannot_be_forgotten(monkeypatch):
    """The failure the table had: an eleventh CW spec absent from it left its
    runs at `precond=None` and every arm pinning `precond` silently skipped
    them. Register one and check the derivation picks it up unprompted."""
    from lora_playground import optim as _optim
    from lora_playground import optim_specs

    extra = optim_specs.OptimizerSpec(
        cls=_optim.CurvatureWhitenLoRA,
        fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True,
               "use_polar": True},
    )
    patched = dict(optim_specs.REGISTRY)
    patched["kl-diag-polar-hypothetical-lora"] = extra
    monkeypatch.setattr(optim_specs, "REGISTRY", patched)

    derived = loader._derive_precond_by_optimizer()
    assert derived.get("kl-diag-polar-hypothetical-lora") == "product"


def test_snapshot_is_regenerated_when_its_sources_move(tmp_path):
    """The JSON snapshot exists so the loader stays torch-free at import; it
    must not outlive an edit to what it snapshots."""
    cache = tmp_path / "snap.json"
    src = tmp_path / "src.py"
    src.write_text("x = 1\n")
    assert not loader._cache_is_fresh(cache, src), "missing cache reads as fresh"
    cache.write_text("{}")
    import os
    os.utime(cache, ns=(src.stat().st_mtime_ns + 10**9,) * 2)
    assert loader._cache_is_fresh(cache, src)
    os.utime(src, ns=(cache.stat().st_mtime_ns + 10**9,) * 2)
    assert not loader._cache_is_fresh(cache, src), "edited source did not invalidate"
    assert loader._cache_is_fresh(cache, tmp_path / "absent.py"), (
        "a source that does not exist should be skipped, not read as stale")


def test_snapshot_on_disk_matches_the_derivation():
    """If a snapshot has been written, it is what the loader reads."""
    if not loader._PRECOND_CACHE.exists():
        pytest.skip("no snapshot written yet in this checkout")
    with open(loader._PRECOND_CACHE) as f:
        snap = json.load(f)
    assert snap == loader._derive_precond_by_optimizer()


# ─── Shim B: what the backfill touched is visible on the cfg ──────────────────

def _cfg(**kw) -> dict:
    base = {"optimizer": "kl-diag-polar-lora"}
    base.update(kw)
    return base


def test_marker_records_a_synthesized_precond():
    cfg = _cfg()
    loader._backfill_precond(cfg, {})
    assert cfg["precond"] == "product"
    assert cfg[loader.PRECOND_BACKFILL_MARKER] == ("precond",)


def test_marker_absent_when_the_run_logged_its_own_precond():
    cfg = _cfg(precond="factorwise")
    loader._backfill_precond(cfg, {})
    assert cfg["precond"] == "factorwise", "a logged value must never be overwritten"
    assert loader.PRECOND_BACKFILL_MARKER not in cfg


def test_recorded_optimizer_config_wins_over_the_name_map():
    """`diag_metric` in the run's own optimizer_config is what it constructed
    with, so it outranks the spec's pin (which the run may have overridden)."""
    cfg = _cfg()
    loader._backfill_precond(cfg, {"diag_metric": False})
    assert cfg["precond"] == "factorwise"
    assert cfg[loader.PRECOND_BACKFILL_MARKER] == ("precond",)


def test_marker_records_the_retired_field_drop():
    cfg = _cfg(precond="product", cw_no_rr_precond=False)
    loader._backfill_precond(cfg, {})
    assert "cw_no_rr_precond" not in cfg
    assert cfg[loader.PRECOND_BACKFILL_MARKER] == ("cw_no_rr_precond",)


def test_marker_accumulates_and_is_idempotent():
    """`_enrich_cfg` runs twice per run under `load_runs`; the second pass finds
    the work already done and must not reset the record."""
    cfg = _cfg(cw_no_rr_precond=False)
    loader._backfill_precond(cfg, {})
    first = cfg[loader.PRECOND_BACKFILL_MARKER]
    assert first == ("cw_no_rr_precond", "precond")
    loader._backfill_precond(cfg, {})
    assert cfg[loader.PRECOND_BACKFILL_MARKER] == first
    assert cfg["precond"] == "product"


def test_non_curvature_whiten_runs_are_untouched():
    cfg = _cfg(optimizer="adamw")
    loader._backfill_precond(cfg, {})
    assert cfg.get("precond") is None
    assert loader.PRECOND_BACKFILL_MARKER not in cfg


def test_marker_cannot_split_a_series_or_a_dedup_key():
    """The marker is an extra cfg key, so it has to be invisible to everything
    that decides run identity. Underscore-prefixed is the mechanism; this is the
    behavioural check on the four functions that walk cfg keys."""
    from lora_playground.plotting import RUNTIME_FIELDS
    from lora_playground.plotting.dedup import series_id

    a = {"optimizer": "kl-diag-polar-lora", "lr": 1e-3, "precond": "product"}
    b = dict(a)
    b[loader.PRECOND_BACKFILL_MARKER] = ("precond",)
    assert series_id(a) == series_id(b)
    assert loader._denylist_key(a, RUNTIME_FIELDS) == loader._denylist_key(b, RUNTIME_FIELDS)
    # `_check_unique_on` must not report the marker as an uncontrolled axis.
    loader._check_unique_on([(a, []), (b, [])], ("optimizer",), RUNTIME_FIELDS, ())
    assert loader.PRECOND_BACKFILL_MARKER.startswith("_")


def test_marker_is_not_a_pinnable_arm_field():
    """`arms.arm()` pins every OptimizerConfig field; if the marker were one, an
    arm would pin it and drop every run that was not backfilled."""
    from lora_playground.plotting.arms import PINNED_FIELDS
    assert loader.PRECOND_BACKFILL_MARKER not in PINNED_FIELDS()


# ─── expiry: delete the shim when no run needs it ─────────────────────────────

pytestmark_logs = pytest.mark.skipif(
    not (ROOT / "logs").is_dir(), reason="no logs/ tree")


@pytest.fixture(scope="module")
def backfill_counts() -> dict[str, int]:
    """How many runs in `logs/` each half of the shim is still serving."""
    runs = loader.load_runs(warn_cross_commit=False, quiet=True,
                            logs_root=str(ROOT / "logs"))
    if not runs:
        pytest.skip("no runs on disk")
    counts = {"precond": 0, "cw_no_rr_precond": 0}
    for cfg, _ in runs:
        for key in cfg.get(loader.PRECOND_BACKFILL_MARKER) or ():
            counts[key] = counts.get(key, 0) + 1
    counts["_n_runs"] = len(runs)
    return counts


@pytestmark_logs
@pytest.mark.parametrize("key,what", [
    ("precond", "the `precond` synthesis (the derived name map and the "
                "`optimizer_config['diag_metric']` read)"),
    ("cw_no_rr_precond", "the `cw_no_rr_precond` drop"),
])
def test_shim_still_has_runs_to_serve(backfill_counts, key, what):
    """Fails when the count reaches ZERO — that is the signal to delete, not a
    regression.

    A backward-compatibility shim with no runs left is pure risk: it keeps
    mutating cfgs away from what is on disk for no reason, and the next reader
    has to re-derive why it is there. This test is the expiry date.
    """
    n = backfill_counts.get(key, 0)
    assert n > 0, (
        f"THIS SHIM IS NOW DEAD — DELETE IT. No run in logs/ "
        f"({backfill_counts['_n_runs']} loaded) still needs {what}.\n"
        f"Remove the corresponding block from `loader._backfill_precond`"
        + (", along with `_derive_precond_by_optimizer`, "
           "`_precond_by_optimizer`, `_PRECOND_CACHE` and "
           "`logs/_precond_by_optimizer.json`" if key == "precond" else
           " and the `cw_no_rr_precond` entry in `_ENRICHMENT_WRITTEN_FIELDS`")
        + f", then delete this test's `{key}` case.\n"
        f"Counts this run: {dict(backfill_counts)}"
    )


@pytestmark_logs
def test_no_loaded_cfg_still_carries_the_retired_field(backfill_counts):
    """The drop half of the shim, checked at its output rather than its input:
    `merge_runs`' hidden-axis check reads a False-vs-absent split on this key as
    two distinct series under one label, so no cfg may survive with it."""
    runs = loader.load_runs(warn_cross_commit=False, quiet=True,
                            logs_root=str(ROOT / "logs"))
    survivors = [(c.get("log_group"), c.get("_log_filename"))
                 for c, _ in runs if "cw_no_rr_precond" in c]
    assert not survivors, f"cfgs still carrying the retired key: {survivors[:5]}"
