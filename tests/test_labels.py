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


def test_adamw_label_ignores_optimizer_inert_wrapper_fields():
    cfg = _cfg(
        "adamw",
        precond_method="higham",
        higham_iters=8,
        cw_nesterov=False,
        muon_ns_steps=8,
    )
    assert canonical_label(cfg) == "AdamW"


def test_adamw_label_keeps_fields_the_optimizer_consumes():
    assert "beta2=0.95" in canonical_label(_cfg("adamw", beta2=0.95))
    assert "lora_plus_multiplier=4" in canonical_label(
        _cfg("adamw", lora_plus_multiplier=4.0)
    )


def test_canonical_key_compact_form():
    assert canonical_key(_cfg(OPT_CT, muon_ns_steps=5, polar_method="ns")) == "ct|ns5|k1|abs"
    # ns>=8 and polar_express both collapse to "full" in the aggregation key
    assert canonical_key(_cfg(OPT_CT, muon_ns_steps=8, polar_method="ns")) == "ct|full|k1|abs"
    assert canonical_key(_cfg(OPT_CT, polar_method="polar_express", muon_ns_steps=10)) == "ct|full|k1|abs"


def test_pinned_labels_stable_across_label_sets():
    # the protagonist (any "PoLoRA*" label) keeps one color no matter which
    # other arms are present — this is what makes figure colors consistent
    from lora_playground.plotting.labels import canonical_colors, PROTAGONIST_COLOR
    sets = [
        ["AdamW", "PoLoRA (kl-diag)", "iMuon"],
        ["PoLoRA (kl-diag)", "w/o curvature control", "w/o magnitude control"],
        ["AdamW", "PoLoRA (ours)"],
        ["PoLoRA (kl-diag)"],
    ]
    for labels in sets:
        colors = canonical_colors(labels)
        proto = [l for l in labels if l.startswith("PoLoRA")][0]
        assert colors[proto] == PROTAGONIST_COLOR
        # pins never collide with palette-assigned labels in the same figure
        assert len(set(colors.values())) == len(colors)


def test_paper_series_styles_are_stable_and_collision_free():
    from lora_playground.plotting.paper_style import resolve_paper_styles

    product = r"Product: $C_B=B^\top P B,\ C_A=A Q A^\top$"
    identity = r"Identity: $C_B=C_A=I$"
    factorwise = r"Factorwise: $C_B=P_A,\ C_A=Q_B$"
    factorwise_diag = (
        r"Diagonal factorwise: $C_B=\operatorname{Diag}(P_A),\ "
        r"C_A=\operatorname{Diag}(Q_B)$"
    )
    sets = (
        ("AdamW", "PoLoRA", "iMuon", "Muon (naive)", "LoRA-RITE"),
        ("AdamW", product, identity, factorwise, factorwise_diag),
        (product, identity, factorwise, factorwise_diag),
    )
    observed = {}
    for tokens in sets:
        styles = resolve_paper_styles(tokens)
        assert len({style["color"] for style in styles.values()}) == len(tokens)
        for token, style in styles.items():
            current = (style["color"], style["marker"])
            assert token not in observed or observed[token] == current
            observed[token] = current
    assert observed["AdamW"] == ("#000000", "o")
