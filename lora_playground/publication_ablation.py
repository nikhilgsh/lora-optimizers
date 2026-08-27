"""Sealed publication evidence for the derivation-ablation reports.

The reports in :mod:`scripts.ablation_table`, :mod:`scripts.ablation_speedup`,
and :mod:`scripts.cell_sigma` all describe one declared publication workload.
This module keeps their data boundary and stable variant identities in one
place.  It never reconstructs historical defaults or searches live logs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .comparison import (
    AggregatedCurve,
    ComparisonResult,
    VariantSpec,
    build_comparison,
)
from .leaderboard_variants import publication_variant_specs
from .publication_archive import (
    ArchivedPublicationRun,
    load_publication_archive,
)
from .publication_queries import publication_workload_runs
from .workloads import Workload, find_workload


DEFAULT_PUBLICATION_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "publication"
    / "legacy_leaderboard_v1.json"
)
ABLATION_WORKLOAD_KEY = (
    "meta-llama/Llama-3.2-1B",
    "openmath",
    256,
    "packed_v1.1",
)
ABLATION_HORIZON = 9000


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One report row keyed by a stable publication variant identity.

    ``variant_id=None`` records that the planned arm has no completed run in
    the sealed archive.  Keeping that state explicit preserves the report's
    no-data row without inventing an identity or falling back to live logs.
    """

    label: str
    variant_id: str | None


PROTAGONIST_ID = (
    "publication.view.v1:kl-diag-polar-lora:"
    "d837ff8c700deedfe262d16f|lora_init_b=zero"
)
KL_DIAG_ID = (
    "publication.view.v1:kl-diag-lora:"
    "0b0200fb9b37efb1b47ffa06|lora_init_b=zero"
)
POLAR_FLATOUT_ID = (
    "publication.view.v1:kl-diag-polar-flatout-lora:"
    "e5b662ea8fd56d983371404f|lora_init_b=zero"
)
KL_SHAMPOO_ID = (
    "publication.view.v1:kl-shampoo-polar-lora:"
    "0e6404be2b21ba3b206d0de0|lora_init_b=zero"
)
ONE_SIDED_ID = (
    "publication.view.v1:kl-diag-polar-lora:"
    "44210b57faa98f4abd91bd6f|lora_init_b=zero"
)
NO_DIAGONAL_METRIC_ID = (
    "publication.view.v1:kl-diag-polar-lora:"
    "8727c82e5f247736ee64d926|lora_init_b=zero"
)
ADAMW_ID = (
    "publication.view.v1:adamw:"
    "97704dc018bbe0de5fc79a67|lora_init_b=zero"
)


ABLATION_ARMS: tuple[AblationArm, ...] = (
    AblationArm("PoLoRA (protagonist)", PROTAGONIST_ID),
    AblationArm("w/o msign (metric^-1)", KL_DIAG_ID),
    # The submitted half-power sweep has no completed log and therefore no
    # reviewed archive identity.  The reports retain an explicit no-data row.
    AblationArm("w/o msign (metric^-1/2)", None),
    AblationArm("w/o outer un-whiten", POLAR_FLATOUT_ID),
    AblationArm("w/o rxr metric CONTENTS", KL_SHAMPOO_ID),
    AblationArm("w/o rxr preconditioner", ONE_SIDED_ID),
    AblationArm("w/o diagonal P,Q", NO_DIAGONAL_METRIC_ID),
    AblationArm("AdamW", ADAMW_ID),
)


@dataclass(frozen=True, slots=True)
class AblationEvidence:
    """One archive-selected workload and its validated variant specs."""

    workload: Workload
    runs: tuple[ArchivedPublicationRun, ...]
    specs_by_id: Mapping[str, VariantSpec]

    @property
    def selected_specs(self) -> tuple[VariantSpec, ...]:
        return tuple(
            self.specs_by_id[arm.variant_id]
            for arm in ABLATION_ARMS
            if arm.variant_id is not None
        )


class PublicationAblationError(ValueError):
    """The sealed archive does not contain a declared ablation identity."""


def load_ablation_evidence(
    archive_path: str | Path = DEFAULT_PUBLICATION_ARCHIVE,
) -> AblationEvidence:
    """Load the exact publication workload and validate every declared ID."""
    archive = load_publication_archive(archive_path)
    workload = find_workload(*ABLATION_WORKLOAD_KEY)
    runs = publication_workload_runs(archive, workload)
    specs = publication_variant_specs(runs)
    specs_by_id = {spec.id: spec for spec in specs}
    missing = [
        arm.variant_id
        for arm in ABLATION_ARMS
        if arm.variant_id is not None and arm.variant_id not in specs_by_id
    ]
    if missing:
        raise PublicationAblationError(
            "declared ablation variant IDs are absent from the selected "
            f"publication workload: {missing!r}"
        )
    return AblationEvidence(
        workload=workload,
        runs=runs,
        specs_by_id=MappingProxyType(specs_by_id),
    )


def build_ablation_comparison(
    evidence: AblationEvidence,
    *,
    horizon: int = ABLATION_HORIZON,
) -> ComparisonResult:
    """Build the shared records-native reduction for table and speed reports."""
    return build_comparison(
        evidence.runs,
        evidence.selected_specs,
        horizon=horizon,
    )


def comparison_curves(
    comparison: ComparisonResult,
    variant_id: str,
) -> Mapping[float, AggregatedCurve]:
    """Return one curve per LR, with completed evidence superseding partials.

    A completed and an in-flight attempt can coexist at one LR.  The completed
    logical trajectory is the reportable measurement; the partial is used only
    where no completed curve exists.  This precedence is explicit here rather
    than depending on input/update order.
    """
    curves = dict(comparison.partials.get(variant_id, {}))
    curves.update(comparison.completed.get(variant_id, {}))
    return MappingProxyType(curves)


def eval_trajectory(
    events: Sequence[Mapping[str, Any]],
) -> dict[int, float]:
    """Return the explicit step→loss trajectory from archived eval events."""
    return {
        int(event["step"]): float(event["eval_loss"])
        for event in events
        if event.get("eval_loss") is not None
    }


def seed_trajectories(
    runs: Sequence[ArchivedPublicationRun],
    *,
    variant_id: str,
    lr: float,
) -> Mapping[int, Mapping[int, float]]:
    """Select one deepest archived trajectory per seed for one exact arm/LR.

    Deeper logical trajectories supersede shorter attempts.  Equal-depth
    duplicates fail closed because choosing one by filesystem/archive order
    would make the reported spread depend on incidental ordering.
    """
    selected: dict[int, tuple[str, dict[int, float]]] = {}
    for run in runs:
        cfg = run.effective_config
        if (
            cfg.get("_publication_variant_id") != variant_id
            or cfg.get("lr") != lr
        ):
            continue
        seed = cfg.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PublicationAblationError(
                f"run {run.physical_id!r} has no integer recorded seed"
            )
        trajectory = eval_trajectory(run.history)
        if not trajectory:
            continue
        prior = selected.get(seed)
        if prior is None or max(trajectory) > max(prior[1]):
            selected[seed] = (run.physical_id, trajectory)
            continue
        if max(trajectory) == max(prior[1]):
            raise PublicationAblationError(
                f"variant {variant_id!r} lr={lr:g} has equal-depth archived "
                f"trajectories for seed {seed}: {prior[0]!r} and "
                f"{run.physical_id!r}"
            )
    return MappingProxyType({
        seed: MappingProxyType(trajectory)
        for seed, (_run_id, trajectory) in selected.items()
    })


__all__ = [
    "ABLATION_ARMS",
    "ABLATION_HORIZON",
    "ABLATION_WORKLOAD_KEY",
    "ADAMW_ID",
    "AblationArm",
    "AblationEvidence",
    "DEFAULT_PUBLICATION_ARCHIVE",
    "KL_SHAMPOO_ID",
    "ONE_SIDED_ID",
    "PROTAGONIST_ID",
    "PublicationAblationError",
    "build_ablation_comparison",
    "comparison_curves",
    "eval_trajectory",
    "load_ablation_evidence",
    "seed_trajectories",
]
