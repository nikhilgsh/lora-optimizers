"""Export the reviewed legacy leaderboard input as one sealed JSON archive.

This is a migration tool, not a live loader. It runs the deprecated publication
path once, after review, and stores the logical trajectories and variant labels
that the records-native generator will continue to consume. Future versioned
runs are discovered through ``RunCatalog`` and are never added here.

Run:
  python scripts/analysis/export_legacy_leaderboard_archive.py \
    --output publication/legacy_leaderboard_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lora_playground.leaderboard import is_final
from lora_playground.plotting.labels import canonical_label
from lora_playground.publication_archive import (
    ARCHIVE_SCHEMA_VERSION,
    PublicationArchiveError,
    publication_archive_from_payload,
)
from lora_playground.publication_semantics import (
    PublicationVariantSemantics,
    normalize_legacy_optimizer_variant_fields,
)
from lora_playground.workloads import Workload, iter_workloads, workload_runs


# Preserve the caller-visible workspace path. ``resolve()`` follows managed
# workspace symlinks into backing mounts that may be read-only even though the
# mounted project root is writable.
ROOT = Path(__file__).absolute().parents[2]
DEFAULT_LOGS = ROOT / "logs"
DEFAULT_OUTPUT = ROOT / "publication" / "legacy_leaderboard_v1.json"
PROJECTION_ID = "legacy_leaderboard_v1"
MEASUREMENT_SEMANTICS_REVISION = "legacy_leaderboard.measurement.v1"
BEHAVIOR_REVISION = "legacy_leaderboard.behavior.v1"

# Reviewed pre-feature executions. Their clean recorded source implements only
# plain EMA; ``cw_nesterov`` did not exist. This allow-list belongs to the
# one-time projection and is never consulted by live loading.
_PRE_NESTEROV_SOURCE_COMMITS = frozenset({
    "04419a0",
    "1437268",
    "69c0ce9",
    "95cea86",
})

# These reviewed clean sources use the declarative factory, whose AdamW spec
# forwards beta1/beta2. Older branches hardcoded (0.9, 0.999). Historical
# ``optimizer_config_dict`` nevertheless recorded (None, None) for both because
# LoRAPlusAdamW keeps betas in param_groups rather than instance attributes.
_ADAMW_BETA_FORWARDING_SOURCE_COMMITS = frozenset({
    "50299f6",
    "601667c",
    "811baa2",
    "9b69f74",
})

# Constructor fields added after some archived executions, with values equal to
# the behavior that the older implementation hardcoded.  This is a sealed
# migration schema, not a live default lookup: neutral values are omitted so
# "field did not exist" and "later field explicitly recorded the old behavior"
# have one canonical representation. Non-neutral and unknown values remain in
# the exact/view payload and therefore fail closed.
_LEGACY_NEUTRAL_FIELDS = {
    "CurvatureWhitenLoRA": {
        "polar_method": "ns",
        "flat_outer": False,
        "precond_method": "eigh",
        "higham_iters": 10,
        "cw_picard_iters": 1,
        "cw_no_radius": False,
        "cw_no_diag_curv": False,
        "cw_no_rr_precond": False,  # retired control
        "cw_unpinned": False,
        "cw_solved_rho": False,
        "msign": "full",
        "cw_factor_a": 0.0,
        "cw_factor_b": 0.0,
        "rdinv_variant": "A",
        "rdinv_delta": None,
        "cw_metric_init": "1e-12",
    },
    "AdamPolarProductLoRA": {
        "htmuon_p": None,
        "ns_form": "gram",
        "higham_compute_dtype": "fp32",
        "fw_linearization": "anchored",
        "ssc_c": None,
        "ssc_nsteps": 10,
        "ssc_kappa": None,
        "ssc_kappa_refresh_every": 1,
        "ssc_kappa_warmup_steps": 5,
        "ssc_kappa_solver": "eigvalsh",
        "ssc_kappa_bisect_iters": 3,
        "ssc_kappa_bisect_mode": "sequential",
        "ssc_kappa_bisect_nsteps_eval": None,
        "ssc_kappa_cache_share_picard": False,
        "ssc_kappa_cache_ema_beta": None,
        "ssc_kappa_cross_group_eigvalsh": True,
        "curvature_whitening": False,
        "curvature_beta": 0.99,
    },
}


@dataclass(frozen=True, slots=True)
class LegacyVariantProjection:
    semantics: PublicationVariantSemantics
    label: str | None
    style_key: str | None


def legacy_publication_semantics(
    cfg: Mapping[str, Any],
) -> PublicationVariantSemantics:
    """Return the sole normalized semantic surface for one legacy run."""
    optimizer_config = cfg.get("optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        raise PublicationArchiveError("legacy run has no optimizer_config object")
    optimizer_effective = cfg.get("optimizer_effective")
    if not isinstance(optimizer_effective, Mapping):
        optimizer_effective = {}
    derived = cfg.get("_derived")
    if not isinstance(derived, Mapping):
        derived = {}

    normalized_source = dict(optimizer_config)
    optimizer = cfg.get("optimizer")
    optim_class = optimizer_config.get("_optim_class")
    clean = cfg.get("execution_source_dirty") is False
    if (
        optimizer == "adamw"
        and optim_class == "LoRAPlusAdamW"
        and normalized_source.get("betas") in ([None, None], (None, None))
    ):
        beta1 = cfg.get("beta1")
        beta2 = cfg.get("beta2")
        default_betas = (0.9, 0.999)
        requested_betas = (beta1, beta2)
        if requested_betas == default_betas:
            # Both the old hardcoded factory and the later forwarding spec
            # execute the same default pair, so no source-mode inference is
            # required (including old runs without scoped cleanliness fields).
            normalized_source["betas"] = default_betas
        elif (
            cfg.get("git_commit") in _ADAMW_BETA_FORWARDING_SOURCE_COMMITS
            and clean
            and all(isinstance(value, (int, float)) for value in requested_betas)
        ):
            normalized_source["betas"] = requested_betas
        else:
            raise PublicationArchiveError(
                "legacy AdamW snapshot lost its executed betas and the "
                "non-default request lacks a reviewed clean forwarding source: "
                f"requested={requested_betas!r}, "
                f"commit={cfg.get('git_commit')!r}, "
                f"execution_source_dirty={cfg.get('execution_source_dirty')!r}"
            )
    if (
        optimizer in {"diag-shampoo-lora", "diag-shampoo-polar-lora"}
        and normalized_source.get("soap_v") is False
        and "cw_nesterov" not in normalized_source
    ):
        commit = cfg.get("git_commit")
        if commit not in _PRE_NESTEROV_SOURCE_COMMITS or not clean:
            raise PublicationArchiveError(
                "legacy diag-Shampoo run omits cw_nesterov without a reviewed "
                f"clean pre-feature source attestation: commit={commit!r}, "
                f"execution_source_dirty={cfg.get('execution_source_dirty')!r}"
            )
        normalized_source["cw_nesterov"] = False

    config, effective = normalize_legacy_optimizer_variant_fields(
        normalized_source,
        optimizer_effective,
        derived_fallback=derived,
    )
    config = dict(config)
    for field, neutral in _LEGACY_NEUTRAL_FIELDS.get(optim_class, {}).items():
        if config.get(field, object()) == neutral:
            config.pop(field)
    source_revision = cfg.get("git_commit")
    if not isinstance(source_revision, str) or not source_revision:
        raise PublicationArchiveError("legacy run has no recorded git_commit")
    return PublicationVariantSemantics(
        optimizer=optimizer,
        config=config,
        effective=effective,
        semantic_revision=BEHAVIOR_REVISION,
        implementation_class=str(
            optimizer_config.get("_optim_class", "legacy.unknown")
        ),
        implementation_revision=source_revision,
    )


def project_legacy_variant(cfg: Mapping[str, Any]) -> LegacyVariantProjection:
    """Derive exact ID, view key, label, and style from one snapshot."""
    semantics = legacy_publication_semantics(cfg)
    label = canonical_label(dict(semantics.label_config))
    if label is not None and (not isinstance(label, str) or not label):
        raise PublicationArchiveError("publication labels must be non-empty strings")
    return LegacyVariantProjection(semantics, label, label)


def _source_segments(
    cfg: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    recorded = cfg.get("_legacy_source_segments")
    if isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes)):
        segments = tuple(dict(segment) for segment in recorded)
        if segments:
            return segments
    group = cfg.get("log_group")
    filename = cfg.get("_log_filename")
    if isinstance(group, str) and group and isinstance(filename, str) and filename:
        steps = [event.get("step") for event in history if event.get("step") is not None]
        if not steps:
            raise PublicationArchiveError(
                "legacy publication run has no step-bearing history"
            )
        return ({
            "physical_id": f"{group}/{filename}",
            "contributed_start_step": min(steps),
            "contributed_end_step": max(steps),
        },)
    raise PublicationArchiveError(
        "legacy publication run has no reviewed source segments"
    )


def _archive_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    required = ("optimizer", "model_name", "data_dir", "lora_r", "lr", "max_steps")
    missing = [field for field in required if cfg.get(field) is None]
    if missing:
        raise PublicationArchiveError(
            f"legacy run is missing archive config fields {missing!r}"
        )
    fields = {
        field: cfg[field]
        for field in (
            "optimizer", "model_name", "data_dir", "lora_r", "lr",
            "max_steps", "data_pipeline_version", "seed",
        )
        if cfg.get(field) is not None
    }
    fields["measurement_semantics_revision"] = MEASUREMENT_SEMANTICS_REVISION
    return fields


def build_archive_payload(
    cells: Iterable[tuple[Workload, Sequence[tuple[dict, list[dict]]]]],
    *,
    variant_adapter: Callable[[Mapping[str, Any]], LegacyVariantProjection],
) -> dict[str, Any]:
    """Build and self-validate a deterministic archive payload."""
    views: dict[str, dict[str, Any]] = {}
    exact_to_view: dict[str, str] = {}
    labels: dict[str, str] = {}
    archived_runs: list[dict[str, Any]] = []
    logical_ids: set[str] = set()

    for workload, runs in cells:
        for cfg, history in runs:
            if not history:
                continue
            last = max(history, key=lambda event: event.get("step", 0) or 0)
            if not is_final(last.get("step"), workload.horizon):
                continue
            projection = variant_adapter(cfg)
            # ``canonical_label`` returning None is the established declaration
            # that this optimizer is outside the reviewed leaderboard families.
            # Preserve that policy boundary instead of inventing archive labels
            # for every optimizer that happens to share a workload directory.
            if projection.label is None:
                continue
            if projection.style_key is None:
                raise PublicationArchiveError(
                    "a publication variant with a label must have a style key"
                )
            semantics = projection.semantics
            exact_id = semantics.exact_id
            view_key = semantics.view_key
            previous_view = exact_to_view.get(exact_id)
            if previous_view is not None and previous_view != view_key:
                raise PublicationArchiveError(
                    f"exact id {exact_id!r} maps to both "
                    f"{previous_view!r} and {view_key!r}"
                )
            metadata = views.get(view_key)
            expected = {
                "label": projection.label,
                "style_key": projection.style_key,
                "optimizer_semantic_key": view_key,
            }
            if metadata is not None and any(
                metadata[key] != value for key, value in expected.items()
            ):
                raise PublicationArchiveError(
                    f"view key {view_key!r} has conflicting metadata"
                )
            previous_view = labels.get(projection.label)
            if previous_view is not None and previous_view != view_key:
                raise PublicationArchiveError(
                    f"variant label {projection.label!r} maps to both "
                    f"{previous_view!r} and {view_key!r}"
                )
            if metadata is None:
                metadata = {**expected, "exact_ids": set()}
                views[view_key] = metadata
            metadata["exact_ids"].add(exact_id)
            exact_to_view[exact_id] = view_key
            labels[projection.label] = view_key

            source_segments = _source_segments(cfg, history)
            sources = tuple(
                str(segment["physical_id"]) for segment in source_segments
            )
            logical_id = sources[-1]
            if logical_id in logical_ids:
                raise PublicationArchiveError(
                    f"legacy logical run {logical_id!r} appears more than once"
                )
            logical_ids.add(logical_id)
            eval_history = [
                {"step": event["step"], "eval_loss": event["eval_loss"]}
                for event in sorted(
                    history, key=lambda event: event.get("step", 0) or 0
                )
                if event.get("step") is not None
                and event.get("eval_loss") is not None
            ]
            archived_runs.append({
                "logical_id": logical_id,
                "exact_id": exact_id,
                "source_segments": list(source_segments),
                "config": _archive_config(cfg),
                "history": eval_history,
                **({"aborted": cfg["_aborted"]} if "_aborted" in cfg else {}),
            })

    ordered_views = sorted(
        views.items(),
        key=lambda item: (item[1]["label"] != "AdamW", item[1]["label"], item[0]),
    )
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "projection_id": PROJECTION_ID,
        "variants": [
            {
                "view_key": view_key,
                "label": metadata["label"],
                "style_key": metadata["style_key"],
                "optimizer_semantic_key": metadata["optimizer_semantic_key"],
                "exact_ids": sorted(metadata["exact_ids"]),
            }
            for view_key, metadata in ordered_views
        ],
        "runs": sorted(archived_runs, key=lambda run: run["logical_id"]),
    }
    publication_archive_from_payload(payload)
    return payload


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check", action="store_true",
        help="compare with an existing archive without writing",
    )
    args = parser.parse_args(argv)
    logs_root = str(Path(args.logs_root).absolute())
    output = Path(args.output).absolute()

    cells = (
        (workload, workload_runs(workload, logs_root=logs_root))
        for workload in iter_workloads()
    )
    payload = build_archive_payload(
        cells,
        variant_adapter=project_legacy_variant,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        current = output.read_text() if output.exists() else ""
        if current != rendered:
            print(f"STALE: {output} differs from the reviewed legacy projection")
            return 1
        print(f"up to date: {output}")
        return 0

    _write_atomic(output, rendered)
    print(
        f"wrote {output}: {len(payload['runs'])} logical runs, "
        f"{len(payload['variants'])} variants"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
