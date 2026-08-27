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
from lora_playground.publication_identity import (
    LORA_INIT_B_MODES,
    composite_publication_identity,
    lora_init_label_suffix,
    require_lora_init_b,
)
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
    lora_init_b: str
    label: str | None
    style_key: str | None


def _require_lora_init_b(cfg: Mapping[str, Any]) -> str:
    mode = cfg.get("lora_init_b")
    try:
        return require_lora_init_b(mode)
    except ValueError as exc:
        raise PublicationArchiveError(
            f"legacy run must explicitly record a valid mode: {exc}"
        ) from exc


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
    lora_init_b = _require_lora_init_b(cfg)
    label_config = dict(semantics.label_config)
    label_config["lora_init_b"] = lora_init_b
    label = canonical_label(label_config)
    if label is not None and (not isinstance(label, str) or not label):
        raise PublicationArchiveError("publication labels must be non-empty strings")
    return LegacyVariantProjection(semantics, lora_init_b, label, label)


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
    required = (
        "optimizer", "model_name", "data_dir", "lora_r", "lr", "max_steps",
        "data_pipeline_version", "lora_init_b",
    )
    missing = [field for field in required if cfg.get(field) is None]
    if missing:
        raise PublicationArchiveError(
            f"legacy run is missing archive config fields {missing!r}"
        )
    fields = {
        field: cfg[field]
        for field in (
            "optimizer", "model_name", "data_dir", "lora_r", "lr",
            "max_steps", "data_pipeline_version", "lora_init_b", "seed",
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
            lora_init_b = _require_lora_init_b(cfg)
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
            if projection.lora_init_b != lora_init_b:
                raise PublicationArchiveError(
                    "legacy variant projection lora_init_b disagrees with the "
                    f"recorded run: {projection.lora_init_b!r} != {lora_init_b!r}"
                )
            exact_id = composite_publication_identity(
                semantics.exact_id, lora_init_b
            )
            view_key = composite_publication_identity(
                semantics.view_key, lora_init_b
            )
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
                "optimizer_semantic_key": semantics.view_key,
                "lora_init_b": lora_init_b,
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
                "lora_init_b": metadata["lora_init_b"],
                "exact_ids": sorted(metadata["exact_ids"]),
            }
            for view_key, metadata in ordered_views
        ],
        "runs": sorted(archived_runs, key=lambda run: run["logical_id"]),
    }
    publication_archive_from_payload(payload)
    return payload


def migrate_sealed_archive_initialization_identity(
    payload: Mapping[str, Any],
    *,
    source_config: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Upgrade reviewed schema-v1 membership with recorded init semantics.

    This migration deliberately does not rediscover workloads or runs.  The
    sealed archive already owns that reviewed evidence set; every logical run,
    source segment, and history is copied unchanged.  The only new input is
    ``lora_init_b`` read from each named physical source config.  Missing,
    unknown, or cross-segment disagreement fails instead of defaulting.
    """
    if payload.get("schema_version") != 1:
        raise PublicationArchiveError(
            "initialization migration requires publication archive schema 1; "
            f"got {payload.get('schema_version')!r}"
        )
    if not callable(source_config):
        raise TypeError("source_config must be callable")

    raw_variants = payload.get("variants")
    raw_runs = payload.get("runs")
    if not isinstance(raw_variants, Sequence) or isinstance(
        raw_variants, (str, bytes)
    ):
        raise PublicationArchiveError("schema-v1 variants must be an array")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise PublicationArchiveError("schema-v1 runs must be an array")

    variants_by_exact: dict[str, Mapping[str, Any]] = {}
    for index, raw_variant in enumerate(raw_variants):
        if not isinstance(raw_variant, Mapping):
            raise PublicationArchiveError(f"schema-v1 variants[{index}] must be an object")
        exact_ids = raw_variant.get("exact_ids")
        if not isinstance(exact_ids, Sequence) or isinstance(exact_ids, (str, bytes)):
            raise PublicationArchiveError(
                f"schema-v1 variants[{index}].exact_ids must be an array"
            )
        for exact_id in exact_ids:
            if not isinstance(exact_id, str) or not exact_id:
                raise PublicationArchiveError(
                    f"schema-v1 variants[{index}] has an invalid exact id"
                )
            if exact_id in variants_by_exact:
                raise PublicationArchiveError(
                    f"schema-v1 exact id {exact_id!r} belongs to multiple variants"
                )
            variants_by_exact[exact_id] = raw_variant

    views: dict[str, dict[str, Any]] = {}
    migrated_runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, Mapping):
            raise PublicationArchiveError(f"schema-v1 runs[{index}] must be an object")
        exact_id = raw_run.get("exact_id")
        old_variant = variants_by_exact.get(exact_id)
        if old_variant is None:
            raise PublicationArchiveError(
                f"schema-v1 runs[{index}] references unknown exact id {exact_id!r}"
            )
        raw_segments = raw_run.get("source_segments")
        if not isinstance(raw_segments, Sequence) or isinstance(
            raw_segments, (str, bytes)
        ) or not raw_segments:
            raise PublicationArchiveError(
                f"schema-v1 runs[{index}].source_segments must be non-empty"
            )
        modes = []
        for segment_index, segment in enumerate(raw_segments):
            if not isinstance(segment, Mapping):
                raise PublicationArchiveError(
                    f"schema-v1 runs[{index}].source_segments[{segment_index}] "
                    "must be an object"
                )
            physical_id = segment.get("physical_id")
            if not isinstance(physical_id, str) or not physical_id:
                raise PublicationArchiveError(
                    f"schema-v1 runs[{index}] has an invalid source physical_id"
                )
            config = source_config(physical_id)
            if not isinstance(config, Mapping):
                raise PublicationArchiveError(
                    f"source {physical_id!r} has no recorded config object"
                )
            modes.append(_require_lora_init_b(config))
        if len(set(modes)) != 1:
            raise PublicationArchiveError(
                f"schema-v1 run {raw_run.get('logical_id')!r} source segments "
                f"disagree on lora_init_b: {modes!r}"
            )
        lora_init_b = modes[0]

        optimizer_semantic_key = old_variant.get("optimizer_semantic_key")
        if not isinstance(optimizer_semantic_key, str) or not optimizer_semantic_key:
            raise PublicationArchiveError(
                f"schema-v1 variant for exact id {exact_id!r} has no optimizer key"
            )
        view_key = composite_publication_identity(
            optimizer_semantic_key, lora_init_b
        )
        migrated_exact_id = composite_publication_identity(exact_id, lora_init_b)
        old_label = old_variant.get("label")
        old_style = old_variant.get("style_key")
        if not isinstance(old_label, str) or not old_label:
            raise PublicationArchiveError("schema-v1 variant has no label")
        if not isinstance(old_style, str) or not old_style:
            raise PublicationArchiveError("schema-v1 variant has no style key")
        label = old_label + lora_init_label_suffix(lora_init_b)
        style_key = label if old_style == old_label else old_style
        expected = {
            "view_key": view_key,
            "label": label,
            "style_key": style_key,
            "optimizer_semantic_key": optimizer_semantic_key,
            "lora_init_b": lora_init_b,
        }
        metadata = views.get(view_key)
        if metadata is not None and any(
            metadata[field] != value for field, value in expected.items()
        ):
            raise PublicationArchiveError(
                f"migrated view {view_key!r} has conflicting metadata"
            )
        if metadata is None:
            metadata = {**expected, "exact_ids": set()}
            views[view_key] = metadata
        metadata["exact_ids"].add(migrated_exact_id)

        config = raw_run.get("config")
        if not isinstance(config, Mapping):
            raise PublicationArchiveError(
                f"schema-v1 runs[{index}].config must be an object"
            )
        migrated_runs.append({
            **dict(raw_run),
            "exact_id": migrated_exact_id,
            "config": {**dict(config), "lora_init_b": lora_init_b},
        })

    migrated = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "projection_id": payload.get("projection_id"),
        "variants": [
            {
                **{field: metadata[field] for field in (
                    "view_key", "label", "style_key",
                    "optimizer_semantic_key", "lora_init_b",
                )},
                "exact_ids": sorted(metadata["exact_ids"]),
            }
            for _, metadata in sorted(
                views.items(),
                key=lambda item: (
                    item[1]["label"] != "AdamW", item[1]["label"], item[0]
                ),
            )
        ],
        "runs": sorted(migrated_runs, key=lambda run: run["logical_id"]),
    }
    publication_archive_from_payload(migrated)
    return migrated


def verify_sealed_archive_initialization_sources(
    payload: Mapping[str, Any],
    *,
    source_config: Callable[[str], Mapping[str, Any]],
) -> None:
    """Verify current sealed membership against recorded source init modes."""
    publication_archive_from_payload(payload)
    for index, raw_run in enumerate(payload["runs"]):
        archived_mode = _require_lora_init_b(raw_run["config"])
        source_modes = []
        for segment in raw_run["source_segments"]:
            physical_id = segment["physical_id"]
            config = source_config(physical_id)
            if not isinstance(config, Mapping):
                raise PublicationArchiveError(
                    f"source {physical_id!r} has no recorded config object"
                )
            source_modes.append(_require_lora_init_b(config))
        if set(source_modes) != {archived_mode}:
            raise PublicationArchiveError(
                f"sealed run {raw_run.get('logical_id')!r} records "
                f"lora_init_b={archived_mode!r}, but source segments record "
                f"{source_modes!r}"
            )


def _source_config_from_logs(logs_root: Path, physical_id: str) -> Mapping[str, Any]:
    from lora_playground.run_parsing import parse_run_file

    group, separator, filename = physical_id.partition("/")
    if not separator or not group or not filename:
        raise PublicationArchiveError(
            f"source physical_id must be '<group>/<filename>', got {physical_id!r}"
        )
    path = logs_root / group / "run_info" / "logs" / filename
    config = parse_run_file(path).raw_config()
    if config is None:
        raise PublicationArchiveError(f"source {physical_id!r} has no config event")
    return config


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
        "--source-archive",
        help=(
            "migrate reviewed schema-v1 membership instead of rediscovering "
            "runs from the current workload registry"
        ),
    )
    parser.add_argument(
        "--check", action="store_true",
        help="compare with an existing archive without writing",
    )
    args = parser.parse_args(argv)
    logs_root = str(Path(args.logs_root).absolute())
    output = Path(args.output).absolute()

    source_path = Path(args.source_archive).absolute() if args.source_archive else None
    if source_path is not None:
        try:
            source_payload = json.loads(source_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationArchiveError(
                f"could not read source archive {source_path}: {exc}"
            ) from exc
        payload = migrate_sealed_archive_initialization_identity(
            source_payload,
            source_config=lambda physical_id: _source_config_from_logs(
                Path(logs_root), physical_id
            ),
        )
    elif output.exists():
        try:
            payload = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationArchiveError(
                f"could not read sealed archive {output}: {exc}"
            ) from exc
        if payload.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise PublicationArchiveError(
                f"existing archive {output} uses schema "
                f"{payload.get('schema_version')!r}; migrate it explicitly with "
                "--source-archive before replacing reviewed membership"
            )
        verify_sealed_archive_initialization_sources(
            payload,
            source_config=lambda physical_id: _source_config_from_logs(
                Path(logs_root), physical_id
            ),
        )
    else:
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
