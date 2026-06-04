"""Tests for the shared leaderboard workload registry (lora_playground.workloads).

Two layers:
  * Pure-logic tests (always run): dataset resolution, registry shape, deny-pattern.
  * Data-backed tests (skip when logs/ is empty, e.g. clean checkout): each cell
    resolves an AdamW baseline + ≥1 variant, no dataset/deny leakage, the
    previously-dropped epsrel/curvature campaigns are members of the OLMo-opc
    cells (regression for the hand-list drift bug), and canonical_label
    discriminates within every cell.
"""
import re
import pytest

from lora_playground.workloads import (
    Workload, WORKLOADS, iter_workloads, find_workload, workload_runs,
    resolve_dataset, _denied, _EXCLUDE_RE, ALLOW_GROUPS,
)
from lora_playground.plotting import canonical_label, assert_label_discriminates


# ── pure logic ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("data_dir, expected", [
    ("data/opc_sft_stage2_all_packed_seq2048", "opc"),
    ("data/opc_sft_stage2_all_packed_seq2048_llama32", "opc"),
    ("data/opc_sft_stage2_all_packed_seq2048_qwen25", "opc"),
    ("data/openmath_instruct_2_2m_packed_seq2048", "openmath"),
    ("data/openmath_instruct_2_2m_packed_seq2048_llama32", "openmath"),
    ("data/tulu3_sft_mixture_400k_packed_seq2048", "tulu3"),
    ("data/magicoder_seq512_32k_packed", "magicoder"),
])
def test_resolve_dataset_from_data_dir(data_dir, expected):
    assert resolve_dataset({"command": f"python train_lora.py --data_dir {data_dir} --lora_r 64"}) == expected


def test_resolve_dataset_none_when_no_data_dir():
    assert resolve_dataset({"command": "python train_lora.py --lora_r 64"}) is None
    assert resolve_dataset({}) is None


def test_resolve_dataset_ignores_stale_dataset_name():
    # dataset_name stays at the Magicoder argparse default for packed runs; must be ignored.
    cfg = {"dataset_name": "ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response", "command": ""}
    assert resolve_dataset(cfg) is None


def test_deny_pattern():
    assert _denied("chord_tight_r256_k1_snapshot_blackwell")
    # The `_slack_` pattern still denies slack-probe groups in general...
    assert _denied("chord_tight_slack_phase_L_1b_r256_lr3_blackwell")
    assert _denied("some_smoke_run")
    # ...except the one explicitly allow-listed (its only ≥8000-step opc-r256 runs
    # are legit abs=1e-6 ns=5 k=1 low-lr points; see ALLOW_GROUPS).
    assert "chord_tight_slack_phase_L_1b_r256_lr2_blackwell" in ALLOW_GROUPS
    assert not _denied("chord_tight_slack_phase_L_1b_r256_lr2_blackwell")
    assert not _denied("chord_tight_phase_L_lrsweep_r256_blackwell")
    assert not _denied("epsrel_fullpolar_r256_olmo_ctk1_eps1e-3_lrext_blackwell")
    # legit ε_rel=1e-2 arm — must NOT be denied (the "_probe_" trap)
    assert not _denied("chord_tight_phase_L_r256_epsrel_ns_probe_blackwell_v2")
    assert not _denied(None)


def test_registry_shape():
    assert len(WORKLOADS) == 13
    # the 10 OLMo/Llama-3.2 cells + two Qwen cells (opc, bengali) + one
    # Meta-Llama-3-8B cell, all unique
    keys = {(w.model_name, w.dataset, w.rank) for w in WORKLOADS}
    assert len(keys) == len(WORKLOADS)
    assert ("Qwen/Qwen2.5-1.5B", "opc", 256) in keys
    assert ("Qwen/Qwen2.5-1.5B", "bengali", 256) in keys
    assert ("meta-llama/Meta-Llama-3-8B", "opc", 256) in keys
    for w in WORKLOADS:
        assert isinstance(w, Workload)
        assert w.horizon >= w.min_completed_steps
        assert w.dataset in {"opc", "openmath", "tulu3", "bengali"}


def test_find_workload_roundtrip_and_miss():
    for w in WORKLOADS:
        assert find_workload(w.model_name, w.dataset, w.rank) is w
        assert find_workload(w.model_display, w.dataset, w.rank) is w  # display name too
    with pytest.raises(KeyError):
        find_workload("allenai/OLMo-2-0425-1B", "opc", 1234)


def test_label_and_title():
    wl = find_workload("allenai/OLMo-2-0425-1B", "opc", 64)
    assert wl.label == "OLMo/opc/r64"
    assert wl.title == "OLMo-2-1B × opc-sft-stage2 (Magicoder) × r=64"


# ── data-backed (skip on empty logs/) ────────────────────────────────────────

def _logs_present():
    from pathlib import Path
    from lora_playground.workloads import DEFAULT_LOGS_ROOT
    p = Path(DEFAULT_LOGS_ROOT)
    return p.is_dir() and any(p.iterdir())


requires_logs = pytest.mark.skipif(not _logs_present(), reason="no logs/ on disk")


@requires_logs
@pytest.mark.parametrize("wl", iter_workloads(), ids=lambda w: w.label)
def test_cell_resolves_adamw_and_variant(wl):
    runs = workload_runs(wl)
    # A freshly-registered workload whose sweeps have not yet produced logs has
    # zero resolved runs — skip rather than fail (the cell becomes live as soon
    # as its first AdamW/variant runs land).
    if not runs:
        pytest.skip(f"{wl.label}: no runs on disk yet (pre-launch workload)")
    labels = {canonical_label(c) for c, _ in runs if canonical_label(c) is not None}
    assert "AdamW" in labels, f"{wl.label}: no AdamW baseline resolved"
    assert len(labels - {"AdamW"}) >= 1, f"{wl.label}: no non-AdamW variant resolved"


@requires_logs
@pytest.mark.parametrize("wl", iter_workloads(), ids=lambda w: w.label)
def test_no_dataset_or_deny_leakage(wl):
    for c, _ in workload_runs(wl):
        assert resolve_dataset(c) == wl.dataset, f"{wl.label}: foreign dataset {c.get('log_group')}"
        assert not _denied(c.get("log_group")), f"{wl.label}: denied group leaked"


@requires_logs
@pytest.mark.parametrize("wl", iter_workloads(), ids=lambda w: w.label)
def test_canonical_label_discriminates_per_cell(wl):
    runs = [(c, h) for c, h in workload_runs(wl) if canonical_label(c) is not None]
    assert_label_discriminates(runs, canonical_label, bucket_keys=("lora_r", "lr"))


@requires_logs
def test_leaderboard_panel_renders():
    """The shared notebook render helper produces a figure from the registry."""
    import matplotlib
    matplotlib.use("Agg")
    from lora_playground.plotting import leaderboard_panel
    fig, table_df, summary_df = leaderboard_panel(
        "allenai/OLMo-2-0425-1B", "opc", 64, "smoke")
    assert fig is not None
    variants = set(summary_df["variant"]) if "variant" in summary_df else set(summary_df.index)
    assert "AdamW" in variants
    # a label_filter sub-view narrows the variant set
    fig2, _, sdf2 = leaderboard_panel(
        "allenai/OLMo-2-0425-1B", "opc", 64, "smoke-k1",
        label_filter=lambda lbl, c: lbl == "AdamW")
    v2 = set(sdf2["variant"]) if "variant" in sdf2 else set(sdf2.index)
    assert v2 == {"AdamW"}


@requires_logs
def test_regression_olmo_opc_includes_epsrel_and_curvature():
    """The hand-list drift bug: these campaigns were silently missing from the doc."""
    for rank in (64, 256):
        wl = find_workload("allenai/OLMo-2-0425-1B", "opc", rank)
        groups = {c.get("log_group") for c, _ in workload_runs(wl)}
        assert any("epsrel_fullpolar" in g for g in groups), f"r{rank}: epsrel groups missing"
    # curvature/SOAP campaign is r256-only
    wl256 = find_workload("allenai/OLMo-2-0425-1B", "opc", 256)
    g256 = {c.get("log_group") for c, _ in workload_runs(wl256)}
    assert any("curvature_whiten" in g for g in g256), "r256: curvature/SOAP groups missing"
