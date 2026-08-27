"""Normalized optimizer semantics for publication views.

Exact producer provenance, displayed publication cohorts, and presentation
labels are separate concerns.  This module provides the one normalized
optimizer surface shared by all three:

* ``exact_id`` includes the recorded source/implementation revision;
* ``view_key`` includes the reviewed behavior revision but not source-only
  provenance; and
* ``label_config`` is a deterministic rendering adapter over the same fields.

The normalizer consumes snapshots supplied by the producer or by a named
historical projection.  It never imports optimizer defaults or plotting code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .optim_config import ALIAS as OPTIMIZER_CONFIG_ALIASES
from .run_records import freeze_value, thaw_value


PUBLICATION_SEMANTICS_SCHEMA_VERSION = 1

# Explicit code-owned contract for constructor inputs that only observe an
# update.  This is intentionally an exact allow-list, not a name-prefix
# heuristic: a new constructor field remains semantic until its owner reviews
# and adds it here.
OBSERVATION_ONLY_OPTIMIZER_FIELDS = frozenset({
    "diagnostics_every",
    "log_basic_diagnostics",
    "log_heavy_diagnostics",
    "log_non_finite",
    "log_non_finite_start_step",
    "debug_optimizer_state",
    "debug_optimizer_state_every",
    "debug_optimizer_state_start_step",
    "debug_snapshot_dir",
    "debug_snapshot_limit",
    "debug_abort_on_non_finite",
    "ssc_kappa_diagnose_eigvalsh",
    "ssc_kappa_diagnose_start_step",
    "ssc_kappa_diag_ema_beta",
    "dump_pre_polar_dir",
    "dump_pre_polar_every",
    "dump_pre_polar_pairs",
    "dump_pre_polar_max_pairs",
})

LAYOUT_AUDIT_OPTIMIZER_FIELDS = frozenset({"muon_params", "adamw_params"})

# Constructor spelling -> run-config spelling.  The normalized surface uses
# one representation, so class refactors such as ``betas`` tuple vs split
# attributes do not create a new publication identity.
_CONFIG_ALIASES = {
    **OPTIMIZER_CONFIG_ALIASES,
    # Per-spec spelling used by AdamPolarProductLoRA. It belongs on the
    # producer schema too: the normalized payload uses OptimizerConfig names,
    # never implementation-local constructor names.
    "core_remix_alpha": "polar_core_remix_alpha",
}

# Known resolved fields are retained only when the normalized constructor
# surface contains the mechanism they resolve.  Unknown producer-emitted
# effective fields remain semantic (fail closed).
_EFFECTIVE_DEPENDENCIES = {
    "effective_picard_iters": frozenset({
        "cw_picard_iters", "picard_iters", "picard_iters_override",
    }),
    "effective_inner_polar": frozenset({
        "polar_method", "polar_sigma_power", "use_polar",
    }),
    "effective_polar_iters": frozenset({
        "muon_ns_steps", "polar_method", "use_polar",
    }),
    "effective_polar_pre_norm": frozenset({"polar_norm_dir"}),
}


class PublicationSemanticsError(ValueError):
    """A producer or historical projection supplied ambiguous semantics."""


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationSemanticsError(f"{context} must be a non-empty string")
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationSemanticsError(f"{context} must be an object")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            thaw_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PublicationSemanticsError(
            f"publication semantics are not canonical JSON: {exc}"
        ) from exc


def normalize_optimizer_variant_fields(
    optimizer_config: Mapping[str, Any],
    optimizer_effective: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Normalize a producer-owned optimizer snapshot into variant fields.

    Learning rate is a comparison axis, not a method identity.  Class names are
    audit metadata.  Observation-only constructor controls are removed through
    the explicit contract above.  All other fields remain semantic by default.

    The effective block is copied verbatim.  It was resolved by the optimizer
    that executed the update; read-side code never guesses which fields apply.
    """
    raw_config = _require_mapping(optimizer_config, "optimizer_config")
    raw_effective = _require_mapping(optimizer_effective, "optimizer_effective")
    config: dict[str, Any] = {}
    for key, value in raw_config.items():
        if not isinstance(key, str):
            raise PublicationSemanticsError(
                "optimizer_config keys must be strings"
            )
        if key in {"_optim_class", "lr"}:
            continue
        if (
            key in OBSERVATION_ONLY_OPTIMIZER_FIELDS
            or key in LAYOUT_AUDIT_OPTIMIZER_FIELDS
        ):
            continue
        if key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise PublicationSemanticsError(
                    "optimizer_config.betas must contain beta1 and beta2"
                )
            config["beta1"], config["beta2"] = value
            continue
        config[_CONFIG_ALIASES.get(key, key)] = value

    return freeze_value(config), freeze_value(dict(raw_effective))


def normalize_legacy_optimizer_variant_fields(
    optimizer_config: Mapping[str, Any],
    optimizer_effective: Mapping[str, Any],
    *,
    derived_fallback: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Archive-only normalization for pre-producer-schema records.

    The compatibility loader's derived block may supply only the four known
    resolved fields, and only when the recorded constructor surface contains
    the mechanism they resolve. Unknown producer-emitted effective fields stay
    semantic. This function is never used for new versioned runs.
    """
    config, effective = normalize_optimizer_variant_fields(
        optimizer_config, optimizer_effective
    )
    fallback = _require_mapping(derived_fallback, "derived_fallback")
    resolved = {
        key: fallback[key]
        for key in _EFFECTIVE_DEPENDENCIES
        if key in fallback
    }
    resolved.update(effective)
    for key, dependencies in _EFFECTIVE_DEPENDENCIES.items():
        if key in resolved and (
            resolved[key] is None or not dependencies.intersection(config)
        ):
            resolved.pop(key)
    return config, freeze_value(resolved)


@dataclass(frozen=True, slots=True)
class PublicationVariantSemantics:
    """One normalized semantic snapshot with exact and view identities."""

    optimizer: str
    config: Mapping[str, Any]
    effective: Mapping[str, Any]
    semantic_revision: str | int
    implementation_class: str
    implementation_revision: str | int

    def __post_init__(self) -> None:
        _require_text(self.optimizer, "optimizer")
        _require_mapping(self.config, "config")
        _require_mapping(self.effective, "effective")
        _require_text(self.implementation_class, "implementation_class")
        for name, value in (
            ("semantic_revision", self.semantic_revision),
            ("implementation_revision", self.implementation_revision),
        ):
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise PublicationSemanticsError(
                    f"{name} must be a non-empty string or integer"
                )
            if isinstance(value, str) and not value.strip():
                raise PublicationSemanticsError(f"{name} must not be empty")
        object.__setattr__(self, "config", freeze_value(dict(self.config)))
        object.__setattr__(self, "effective", freeze_value(dict(self.effective)))

    @property
    def view_payload(self) -> Mapping[str, Any]:
        return MappingProxyType({
            "schema_version": PUBLICATION_SEMANTICS_SCHEMA_VERSION,
            "optimizer": self.optimizer,
            "config": self.config,
            "effective": self.effective,
            "semantic_revision": self.semantic_revision,
        })

    @property
    def exact_payload(self) -> Mapping[str, Any]:
        return MappingProxyType({
            **dict(self.view_payload),
            "implementation": {
                "class": self.implementation_class,
                "revision": self.implementation_revision,
            },
        })

    @property
    def view_key(self) -> str:
        return _digest_key("publication.view", self.optimizer, self.view_payload)

    @property
    def exact_id(self) -> str:
        return _digest_key("publication.exact", self.optimizer, self.exact_payload)

    @property
    def label_config(self) -> Mapping[str, Any]:
        return freeze_value({
            "optimizer": self.optimizer,
            **thaw_value(self.config),
            "_derived": thaw_value(self.effective),
        })


def _digest_key(prefix: str, optimizer: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.blake2s(
        _canonical_json(payload).encode("utf-8"), digest_size=12
    ).hexdigest()
    return (
        f"{prefix}.v{PUBLICATION_SEMANTICS_SCHEMA_VERSION}:"
        f"{optimizer}:{digest}"
    )


def publication_semantics_from_payload(
    payload: Mapping[str, Any],
) -> PublicationVariantSemantics:
    """Parse the producer/archive JSON representation of one snapshot."""
    item = _require_mapping(payload, "publication optimizer semantics")
    if item.get("schema_version") != PUBLICATION_SEMANTICS_SCHEMA_VERSION:
        raise PublicationSemanticsError(
            "unsupported publication optimizer semantics schema_version "
            f"{item.get('schema_version')!r}; expected "
            f"{PUBLICATION_SEMANTICS_SCHEMA_VERSION}"
        )
    return PublicationVariantSemantics(
        optimizer=item.get("optimizer"),
        config=_require_mapping(item.get("config"), "config"),
        effective=_require_mapping(item.get("effective"), "effective"),
        semantic_revision=item.get("semantic_revision"),
        implementation_class=_require_text(
            _require_mapping(item.get("implementation"), "implementation").get(
                "class"
            ),
            "implementation.class",
        ),
        implementation_revision=_require_mapping(
            item.get("implementation"), "implementation"
        ).get("revision"),
    )


def build_optimizer_variant_semantics_payload(
    *,
    optimizer: str,
    optimizer_instance: object,
    optimizer_config: Mapping[str, Any],
    optimizer_effective: Mapping[str, Any],
    semantic_revision: str | int,
    implementation_revision: str | int,
) -> dict[str, Any]:
    """Build the producer-owned JSON block for one constructed optimizer.

    Both identity and presentation adapters consume this exact normalized
    snapshot. The producer supplies the already-constructed instance plus its
    recorded config/effective blocks; no run reader imports defaults or
    re-introspects a newer class later.
    """
    optimizer_class = type(optimizer_instance)
    implementation_class = (
        f"{optimizer_class.__module__}.{optimizer_class.__qualname__}"
    )
    config, effective = normalize_optimizer_variant_fields(
        optimizer_config, optimizer_effective
    )
    semantics = PublicationVariantSemantics(
        optimizer=optimizer,
        config=config,
        effective=effective,
        semantic_revision=semantic_revision,
        implementation_class=implementation_class,
        implementation_revision=implementation_revision,
    )
    # Validate finite, canonical JSON at the producer boundary. A run must not
    # advertise schema-v2 semantics that a later reader cannot identify.
    _canonical_json(semantics.exact_payload)
    return thaw_value(semantics.exact_payload)


__all__ = [
    "OBSERVATION_ONLY_OPTIMIZER_FIELDS",
    "LAYOUT_AUDIT_OPTIMIZER_FIELDS",
    "PUBLICATION_SEMANTICS_SCHEMA_VERSION",
    "PublicationSemanticsError",
    "PublicationVariantSemantics",
    "build_optimizer_variant_semantics_payload",
    "normalize_optimizer_variant_fields",
    "normalize_legacy_optimizer_variant_fields",
    "publication_semantics_from_payload",
]
