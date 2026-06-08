"""Tests for the canonical variant labeler (single source of truth)."""
from __future__ import annotations

from lora_playground.plotting import (
    canonical_colors, canonical_key, canonical_label, order_labels,
)

OPT_CT = "adam-polar-product-lora-coupled-spectral-chord-tight"
OPT_CLEAN = OPT_CT + "-clean"


def _cfg(opt, **kw):
    return {"optimizer": opt, **kw}


def test_label_table():
    assert canonical_label(_cfg("adamw")) == "AdamW"
    assert canonical_label(_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns")) \
        == "chord-tight ns=5 k=1 (abs)"
    assert canonical_label(_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns",
                                precond_delta_relative=True, precond_delta=0.01)) \
        == "chord-tight ns=8 k=1 (ε_rel=1e-2)"
    assert canonical_label(_cfg(OPT_CT, polar_method="polar_express", muon_ns_steps=10)) \
        == "chord-tight PE=10 k=1 (abs)"
    assert canonical_label(_cfg(OPT_CLEAN, muon_ns_steps=8, polar_method="ns",
                                picard_iters_override=2)) \
        == "chord-tight-clean ns=8 k=2 (abs)"
    assert canonical_label(_cfg(OPT_CLEAN, muon_ns_steps=8, picard_iters_override=2,
                                ssc_kappa=0.75)) \
        == "chord-tight-clean ns=8 k=2 (κ_sr=0.75)"
    assert canonical_label(_cfg(
        "kl-diag-ssc-history-picard-lora",
        precond_refresh_every=10,
        curvature_beta=0.99,
        precond_delta=0.001,
        cw_picard_iters=2,
        optimizer_config={"ssc_kappa": 0.75},
    )) == "KL-diag +SSC-history k2 (f=10, β_c=0.99, δ=1e-3, κ=0.75)"


def test_non_family_optimizer_is_none():
    assert canonical_label(_cfg("adam-lin-lora")) is None


def test_distinct_configs_get_distinct_labels():
    # the exact bug class: ns must appear so ns5/ns8 never collapse
    a = canonical_label(_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns"))
    b = canonical_label(_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns"))
    assert a != b
    # eps_rel value must appear
    e1 = canonical_label(_cfg(OPT_CT, muon_ns_steps=8, precond_delta_relative=True, precond_delta=0.001))
    e2 = canonical_label(_cfg(OPT_CT, muon_ns_steps=8, precond_delta_relative=True, precond_delta=0.1))
    assert e1 != e2
    # curvature-family (kl-shampoo / SOAP-curv) damping δ must appear, else a
    # precond_delta sweep collapses to one curve in the leaderboard panel.
    k1 = canonical_label(_cfg("kl-shampoo-polar-lora", precond_refresh_every=10,
                              curvature_beta=0.99, precond_delta=1e-4))
    k2 = canonical_label(_cfg("kl-shampoo-polar-lora", precond_refresh_every=10,
                              curvature_beta=0.99, precond_delta=1e-2))
    assert k1 != k2 and "δ=1e-4" in k1 and "δ=1e-2" in k2


def test_derived_picard_overrides_raw():
    # loader-derived effective_picard_iters takes precedence over raw override
    cfg = _cfg(OPT_CLEAN, muon_ns_steps=8, picard_iters_override=1,
               _derived={"effective_picard_iters": 2})
    assert "k=2" in canonical_label(cfg)


def test_adamw_black_and_first():
    labels = ["chord-tight ns=8 k=1 (abs)", "AdamW", "chord-tight ns=5 k=1 (abs)"]
    assert order_labels(labels)[0] == "AdamW"
    colors = canonical_colors(labels)
    assert colors["AdamW"] == "#000000"
    # no non-AdamW label is assigned black
    assert all(v != "#000000" for k, v in colors.items() if k != "AdamW")


def test_canonical_key_compact_form():
    assert canonical_key(_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns")) == "ct|ns5|k1|abs"
    # ns>=8 and polar_express both collapse to "full" in the aggregation key
    assert canonical_key(_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns")) == "ct|full|k1|abs"
    assert canonical_key(_cfg(OPT_CT, polar_method="polar_express", muon_ns_steps=10)) == "ct|full|k1|abs"
