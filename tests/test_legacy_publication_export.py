"""One-time legacy leaderboard export into the sealed archive schema."""
from __future__ import annotations

import pytest

from lora_playground.publication_archive import PublicationArchiveError
from lora_playground.publication_semantics import PublicationVariantSemantics
from lora_playground.workloads import Workload
from scripts.analysis.export_legacy_leaderboard_archive import (
    LegacyVariantProjection,
    build_archive_payload,
    legacy_publication_semantics,
    project_legacy_variant,
)


def _workload():
    return Workload(
        "model", "openmath", 64, "Model", "OpenMath", 1000, 0.001, True
    )


def _run(
    name,
    *,
    label="AdamW",
    variant_id="adamw.default",
    lr=1e-3,
    final_step=1000,
    sources=None,
):
    cfg = {
        "optimizer": "adamw",
        "model_name": "model",
        "data_dir": "/data/openmath_instruct_2_2m_packed",
        "lora_r": 64,
        "lr": lr,
        "max_steps": 1000,
        "data_pipeline_version": "packed_v1.1",
        "seed": 0,
        "label": label,
        "variant_id": variant_id,
        "log_group": "group",
        "_log_filename": f"{name}.out",
    }
    if sources is not None:
        cfg["_legacy_source_physical_ids"] = tuple(sources)
        cfg["_legacy_source_segments"] = tuple(
            {
                "physical_id": source,
                "contributed_start_step": 500 if index == 0 else 1000,
                "contributed_end_step": 500 if index == 0 else 1000,
            }
            for index, source in enumerate(sources)
        )
    return cfg, [
        {"step": 500, "eval_loss": 0.9},
        {"step": final_step, "eval_loss": 0.8},
    ]


def _build(runs):
    def adapter(cfg):
        semantics = PublicationVariantSemantics(
            optimizer=cfg["optimizer"],
            config={"variant": cfg["variant_id"]},
            effective={},
            semantic_revision=1,
            implementation_class="tests.Method",
            implementation_revision=1,
        )
        return LegacyVariantProjection(
            semantics=semantics,
            label=cfg["label"],
            style_key=cfg["label"],
        )

    return build_archive_payload(
        [(_workload(), runs)],
        variant_adapter=adapter,
    )


def test_export_keeps_completed_logical_trajectories_and_source_segments():
    payload = _build([
        _run(
            "resume",
            sources=("old/log_0.out", "new/log_0.out.resume_1"),
        ),
        _run(
            "candidate",
            label="Method",
            variant_id="method.v1",
            lr=3e-3,
        ),
        _run("partial", lr=1e-2, final_step=500),
    ])

    assert payload["projection_id"] == "legacy_leaderboard_v1"
    assert [variant["label"] for variant in payload["variants"]] == [
        "AdamW", "Method",
    ]
    assert len(payload["runs"]) == 2
    resumed = next(
        run for run in payload["runs"]
        if run["logical_id"] == "new/log_0.out.resume_1"
    )
    assert resumed["logical_id"] == "new/log_0.out.resume_1"
    assert resumed["source_segments"] == [
        {
            "physical_id": "old/log_0.out",
            "contributed_start_step": 500,
            "contributed_end_step": 500,
        },
        {
            "physical_id": "new/log_0.out.resume_1",
            "contributed_start_step": 1000,
            "contributed_end_step": 1000,
        },
    ]
    assert resumed["history"][-1] == {"step": 1000, "eval_loss": 0.8}


def test_export_excludes_optimizer_outside_reviewed_publication_families():
    excluded_cfg, history = _run("excluded")
    excluded_cfg.update({
        "optimizer": "muon-lora",
        "optimizer_config": {
            "_optim_class": "MuonLoRA",
            "lr": excluded_cfg["lr"],
        },
        "optimizer_effective": {},
        "git_commit": "reviewed-commit",
    })
    included_cfg, included_history = _run("included")
    included_cfg.update({
        "optimizer_config": {
            "_optim_class": "LoRAPlusAdamW",
            "lr": included_cfg["lr"],
        },
        "optimizer_effective": {},
        "git_commit": "reviewed-commit",
    })

    payload = build_archive_payload(
        [(_workload(), [
            (excluded_cfg, history),
            (included_cfg, included_history),
        ])],
        variant_adapter=project_legacy_variant,
    )

    assert [variant["label"] for variant in payload["variants"]] == ["AdamW"]
    assert [run["logical_id"] for run in payload["runs"]] == [
        "group/included.out"
    ]


@pytest.mark.parametrize(
    "runs",
    [
        [
            _run("a", label="First", variant_id="same", lr=1e-3),
            _run("b", label="Second", variant_id="same", lr=3e-3),
        ],
        [
            _run("a", label="Same", variant_id="first", lr=1e-3),
            _run("b", label="Same", variant_id="second", lr=3e-3),
        ],
    ],
)
def test_export_rejects_non_bijective_variant_identity(runs):
    with pytest.raises(
        PublicationArchiveError,
        match="maps to both|conflicting metadata",
    ):
        _build(runs)


def test_legacy_label_does_not_backfill_nesterov_into_pre_flag_run():
    common = {
        "optimizer": "diag-shampoo-polar-lora",
        "cw_nesterov": True,  # compatibility default, not producer evidence
        "precond_refresh_every": 10,
        "curvature_beta": 0.99,
        "precond_delta": 1e-4,
        "optimizer_effective": {
            "effective_inner_polar": "polar_express",
            "effective_polar_iters": 8,
        },
        "_derived": {
            "effective_inner_polar": "polar_express",
            "effective_polar_iters": 8,
        },
        "git_commit": "69c0ce9",
        "execution_source_dirty": False,
    }
    old = {
        **common,
        "optimizer_config": {
            "precond_refresh_every": 10,
            "curvature_beta": 0.99,
            "delta": 1e-4,
            "soap_v": False,
            "use_polar": True,
            "polar_method": "polar_express",
            "ns_steps": 8,
        },
    }
    nesterov = {
        **common,
        "optimizer_config": {
            "precond_refresh_every": 10,
            "curvature_beta": 0.99,
            "delta": 1e-4,
            "soap_v": False,
            "use_polar": True,
            "polar_method": "polar_express",
            "ns_steps": 8,
            "cw_nesterov": True,
        },
        "git_commit": "889450a",
    }

    old_projection = project_legacy_variant(old)
    nesterov_projection = project_legacy_variant(nesterov)
    assert "+nesterov" not in old_projection.label
    assert "+nesterov" in nesterov_projection.label
    assert old_projection.semantics.view_key != nesterov_projection.semantics.view_key


def test_missing_pre_flag_and_explicit_false_share_plain_ema_view_identity():
    old = {
        "optimizer": "diag-shampoo-polar-lora",
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "soap_v": False,
            "cw_nesterov": False,
        },
        "optimizer_effective": {},
        "git_commit": "889450a",
        "execution_source_dirty": False,
    }
    pre_flag = {
        **old,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "soap_v": False,
        },
        "git_commit": "69c0ce9",
        "execution_source_dirty": False,
    }

    assert (
        legacy_publication_semantics(old).view_key
        == legacy_publication_semantics(pre_flag).view_key
    )


def test_missing_pre_feature_curvature_controls_match_explicit_defaults():
    common = {
        "optimizer": "kl-shampoo-polar-lora",
        "optimizer_effective": {
            "effective_inner_polar": "ns",
            "effective_polar_iters": 5,
        },
        "git_commit": "reviewed-clean-source",
        "execution_source_dirty": False,
    }
    old = {
        **common,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "use_polar": True,
            "ns_steps": 5,
        },
    }
    explicit = {
        **common,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "use_polar": True,
            "ns_steps": 5,
            "polar_method": "ns",
            "flat_outer": False,
        },
    }

    assert (
        legacy_publication_semantics(old).view_key
        == legacy_publication_semantics(explicit).view_key
    )


def test_retired_neutral_legacy_control_does_not_split_publication_view():
    common = {
        "optimizer": "kl-shampoo-polar-lora",
        "optimizer_effective": {},
        "git_commit": "reviewed-clean-source",
        "execution_source_dirty": False,
    }
    absent = {
        **common,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "use_polar": True,
        },
    }
    explicit_neutral = {
        **common,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "use_polar": True,
            "cw_no_rr_precond": False,
        },
    }
    explicit_non_neutral = {
        **common,
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "use_polar": True,
            "cw_no_rr_precond": True,
        },
    }

    absent_semantics = legacy_publication_semantics(absent)
    assert (
        absent_semantics.view_key
        == legacy_publication_semantics(explicit_neutral).view_key
    )
    assert (
        absent_semantics.view_key
        != legacy_publication_semantics(explicit_non_neutral).view_key
    )


def test_legacy_semantics_preserve_reviewed_polar_short_circuit_resolution():
    cfg = {
        "optimizer": "diag-shampoo-polar-lora",
        "precond_refresh_every": 10,
        "curvature_beta": 0.99,
        "precond_delta": 1e-4,
        "optimizer_config": {
            "precond_refresh_every": 10,
            "curvature_beta": 0.99,
            "delta": 1e-4,
            "polar_method": "polar_express",
            "ns_steps": 8,
        },
        "optimizer_effective": {},
        "_derived": {
            "effective_inner_polar": "svd_exact",
            "effective_polar_iters": 8,
        },
        "git_commit": "commit",
        "beta1": 0.9,
        "beta2": 0.999,
    }

    semantics = legacy_publication_semantics(cfg)
    assert semantics.effective["effective_inner_polar"] == "svd_exact"
    assert semantics.effective["effective_polar_iters"] == 8


def test_legacy_adamw_removes_only_inert_picard_effective_field():
    common = {
        "optimizer": "adamw",
        "optimizer_config": {
            "_optim_class": "LoRAPlusAdamW",
            "betas": [None, None],
            "eps": 1e-8,
            "lora_plus_multiplier": 1.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "git_commit": "commit",
        "beta1": 0.9,
        "beta2": 0.999,
    }

    baseline = legacy_publication_semantics({
        **common,
        "optimizer_effective": {},
    })
    repaired = legacy_publication_semantics({
        **common,
        "optimizer_effective": {
            "effective_picard_iters": 1,
            "future_recorded_semantic": "kept",
        },
    })
    assert "effective_picard_iters" not in repaired.effective
    assert repaired.effective["future_recorded_semantic"] == "kept"
    assert baseline.view_key != repaired.view_key


def test_legacy_adamw_repairs_param_group_betas_without_trusting_old_snapshot():
    common = {
        "optimizer": "adamw",
        "optimizer_config": {
            "_optim_class": "LoRAPlusAdamW",
            "betas": [None, None],
            "lr": 1e-3,
        },
        "optimizer_effective": {},
        "execution_source_dirty": False,
    }
    default = legacy_publication_semantics({
        **common,
        "beta1": 0.9,
        "beta2": 0.999,
        "git_commit": "old-hardcoded-source",
    })
    forwarded = legacy_publication_semantics({
        **common,
        "beta1": 0.9,
        "beta2": 0.81,
        "git_commit": "50299f6",
    })

    assert default.config["beta1"] == 0.9
    assert default.config["beta2"] == 0.999
    assert forwarded.config["beta2"] == 0.81
    assert default.view_key != forwarded.view_key


def test_legacy_adamw_rejects_unattested_nondefault_beta_repair():
    with pytest.raises(PublicationArchiveError, match="lost its executed betas"):
        legacy_publication_semantics({
            "optimizer": "adamw",
            "optimizer_config": {
                "_optim_class": "LoRAPlusAdamW",
                "betas": [None, None],
            },
            "optimizer_effective": {},
            "beta1": 0.9,
            "beta2": 0.81,
            "git_commit": "unknown-source",
            "execution_source_dirty": False,
        })


def test_legacy_identity_uses_executed_optimizer_snapshot_not_requested_beta1():
    common = {
        "optimizer": "diag-shampoo-polar-lora",
        "optimizer_config": {
            "_optim_class": "CurvatureWhitenLoRA",
            "betas": [0.9, 0.999],
            "cw_nesterov": True,
            "soap_v": False,
        },
        "optimizer_effective": {},
        "git_commit": "same-executed-source",
    }

    requested_default = legacy_publication_semantics({**common, "beta1": 0.9})
    requested_ignored = legacy_publication_semantics({**common, "beta1": 0.95})

    assert requested_default.view_key == requested_ignored.view_key
    assert requested_ignored.config["beta1"] == 0.9
