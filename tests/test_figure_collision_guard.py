"""Regression test: compare_variants_figure must RAISE (not silently min-merge)
when a label buckets two distinct configs at one (lora_r, lr).

Reproduces the eps_rel ns=5 / ns=8 bug: a coarse label that omits `ns` put both
runs in one bucket and the figure kept the lower-loss one, displaying a wrong
curve. The guard turns that into a hard error.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from lora_playground.plotting import (
    LabelCollisionError, canonical_label, compare_variants_figure,
)

OPT_CT = "adam-polar-product-lora-coupled-spectral-chord-tight"


def _run(ns, lr, loss, r=256):
    cfg = {"optimizer": OPT_CT, "lr": lr, "lora_r": r,
           "muon_ns_steps": ns, "polar_method": "ns"}
    return cfg, [{"step": 9000, "eval_loss": loss}]


def test_coarse_label_raises_on_ns_collision():
    runs = [_run(5, 1e-2, 0.74), _run(8, 1e-2, 0.81)]   # same lr, different ns
    coarse = lambda c: "chord" if c["optimizer"] == OPT_CT else None
    with pytest.raises(LabelCollisionError):
        compare_variants_figure(
            variants={"chord": {}}, common_where={}, ref_label="chord",
            max_steps=9000, prefetched_runs=runs, variant_key=coarse)
    plt.close("all")


def test_canonical_label_avoids_collision():
    runs = [_run(5, 1e-2, 0.74), _run(8, 1e-2, 0.81)]
    labels = {canonical_label(c) for c, _ in runs}
    assert len(labels) == 2  # ns-explicit → distinct
    fig, *_ = compare_variants_figure(
        variants={l: {} for l in labels}, common_where={},
        ref_label=sorted(labels)[0], max_steps=9000,
        prefetched_runs=runs, variant_key=canonical_label)
    plt.close("all")


def test_allow_label_collision_escape_hatch():
    runs = [_run(5, 1e-2, 0.74), _run(8, 1e-2, 0.81)]
    fig, *_ = compare_variants_figure(
        variants={"chord": {}}, common_where={}, ref_label="chord",
        max_steps=9000, prefetched_runs=runs, variant_key=lambda c: "chord",
        allow_label_collision=True)
    plt.close("all")
