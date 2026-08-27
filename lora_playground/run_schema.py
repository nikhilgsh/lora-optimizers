"""Small, versioned fields for newly emitted run config events.

This module records semantic revision labels and explicit execution-attempt
identity.  It intentionally does not snapshot a second full configuration and
does not compute hashes or digests.  Callers merge the returned dictionaries
into the config event alongside the already-resolved configuration.

Resume lineage comes only from a checkpoint that was successfully loaded.
Checkpoint paths, launcher environment, SLURM restart counters, and step
ranges are never used to infer ancestry.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


RUN_SCHEMA_VERSION = 2
"""Version of the top-level run config-event schema fields in this module."""

DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION = 1
OPTIMIZER_IMPLEMENTATION_REVISION_ATTR = "IMPLEMENTATION_REVISION"
MEASUREMENT_SEMANTICS_REVISION = 1

ATTEMPT_ID_ENV = "LORA_ATTEMPT_ID"
CHECKPOINT_IDENTITY_ENV = "LORA_CHECKPOINT_IDENTITY"


def optimizer_implementation_revision(optimizer: object | type) -> int:
    """Return the optimizer class's declared revision, or the stable default.

    Project-owned optimizer classes may increment ``IMPLEMENTATION_REVISION``
    when their update semantics change without a corresponding resolved-config
    change.  Classes without that attribute remain at revision 1.  Both an
    optimizer instance and its class are accepted so this helper is cheap to
    test and usable before or after construction.
    """
    optimizer_class = optimizer if isinstance(optimizer, type) else type(optimizer)
    revision = getattr(
        optimizer_class,
        OPTIMIZER_IMPLEMENTATION_REVISION_ATTR,
        DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION,
    )
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError(
            f"{optimizer_class.__name__}."
            f"{OPTIMIZER_IMPLEMENTATION_REVISION_ATTR} must be a positive int, "
            f"got {revision!r}"
        )
    return revision


def semantic_revisions(
    optimizer: object | type,
    resolved_config: Mapping[str, Any],
) -> dict[str, int | str]:
    """Build the bounded component revisions for a run config event.

    ``data_pipeline`` is copied from the executed configuration; it is never
    reconstructed from current defaults.  The output contains only revision
    labels, not the resolved configuration itself.
    """
    if not isinstance(resolved_config, Mapping):
        raise TypeError("resolved_config must be a mapping")
    pipeline = resolved_config.get("data_pipeline_version")
    if not isinstance(pipeline, str) or not pipeline.strip():
        raise ValueError(
            "resolved_config must contain a non-empty data_pipeline_version"
        )
    return {
        "optimizer_impl": optimizer_implementation_revision(optimizer),
        "data_pipeline": pipeline,
        "measurement": MEASUREMENT_SEMANTICS_REVISION,
    }


def _required_identity(
    field: str,
    explicit: str | None,
    environ: Mapping[str, str],
    env_name: str,
) -> str:
    value = explicit if explicit is not None else environ.get(env_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field} must be supplied explicitly or via {env_name}"
        )
    return value


def attempt_metadata(
    *,
    attempt_id: str | None = None,
    checkpoint_identity: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, int | str | None]:
    """Return root-attempt fields ready to merge into a config event.

    Explicit values take precedence over the dedicated ``LORA_*`` environment
    variables.  Parent identity is deliberately not accepted from either
    source: train emits it later in the ``resume`` event only after loading a
    checkpoint whose metadata names the parent attempt.

    ``attempt_id`` and ``checkpoint_identity`` are required after environment
    lookup.  The helper does not invent either value, keeping launcher/runtime
    ownership visible.
    """
    env = os.environ if environ is None else environ
    if not isinstance(env, Mapping):
        raise TypeError("environ must be a mapping")

    resolved_attempt = _required_identity(
        "attempt_id", attempt_id, env, ATTEMPT_ID_ENV
    )
    resolved_checkpoint = _required_identity(
        "checkpoint_identity",
        checkpoint_identity,
        env,
        CHECKPOINT_IDENTITY_ENV,
    )
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "attempt_id": resolved_attempt,
        "resume_parent_attempt_id": None,
        "checkpoint_identity": resolved_checkpoint,
    }


__all__ = [
    "ATTEMPT_ID_ENV",
    "CHECKPOINT_IDENTITY_ENV",
    "DEFAULT_OPTIMIZER_IMPLEMENTATION_REVISION",
    "MEASUREMENT_SEMANTICS_REVISION",
    "OPTIMIZER_IMPLEMENTATION_REVISION_ATTR",
    "RUN_SCHEMA_VERSION",
    "attempt_metadata",
    "optimizer_implementation_revision",
    "semantic_revisions",
]
