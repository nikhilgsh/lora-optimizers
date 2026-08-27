"""Tidy-DataFrame wrapper over immutable run records.

``load_runs_df`` exposes the producer-recorded semantic configuration as one
row per physical or explicitly resolved logical run. Physical provenance is
kept in named columns instead of being reconstructed from command strings::

    df = load_runs_df(where={"lora_r": 256})
    df[(df.polar_method == "ns") & df.ssc_kappa.isna()]

``df.columns`` is the recorded schema. Missing historical values stay missing;
this adapter never imports current defaults or the tuple loader.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .run_catalog import load_records
from .run_records import run_view


_COLLECTION_TYPES = (list, set, tuple, frozenset)


def _matches(want: Any, value: Any) -> bool:
    """Match one recorded field using the public predicate convention."""
    if callable(want):
        return bool(want(value))
    if isinstance(want, _COLLECTION_TYPES):
        if isinstance(value, _COLLECTION_TYPES):
            return list(value) == list(want)
        return value in want
    return value == want


def _where_value(view, field: str) -> tuple[bool, Any]:
    if field in {"group", "log_group", "_group"}:
        return view.group is not None, view.group
    if field == "log_filename":
        return view.log_filename is not None, view.log_filename
    if field == "physical_id":
        return True, view.physical_id
    if field not in view.semantic_config:
        return False, None
    return True, view.semantic_config[field]


def _where_matches(view, where: Mapping[str, Any]) -> bool:
    for field, want in where.items():
        present, value = _where_value(view, field)
        if not present or not _matches(want, value):
            return False
    return True


def load_runs_df(
    where: Mapping[str, Any] | None = None,
    *,
    logs_root: str | None = None,
    catalog=None,
    resolve_lineages: bool = True,
) -> pd.DataFrame:
    """Return one row per selected immutable run.

    ``where`` supports literals, explicit collections, and callables over
    recorded semantic fields. ``group``/``log_group``, ``log_filename``, and
    ``physical_id`` select explicit physical metadata. The loader accepts
    either ``logs_root`` or an already-discovered ``RunCatalog`` snapshot.
    """
    if where is None:
        predicates: Mapping[str, Any] = {}
    elif isinstance(where, Mapping):
        predicates = where
    else:
        raise TypeError("where must be a field-to-predicate mapping")

    records = load_records(
        logs_root=logs_root,
        catalog=catalog,
        resolve_lineages=resolve_lineages,
    )
    rows = []
    for index, record in enumerate(records):
        view = run_view(record, index)
        if not _where_matches(view, predicates):
            continue
        row = dict(view.semantic_config)
        losses = [
            event["eval_loss"]
            for event in view.history
            if event.get("eval_loss") is not None
        ]
        steps = [event.get("step") for event in view.history]
        row.update({
            "physical_id": view.physical_id,
            "group": view.group,
            "log_filename": view.log_filename,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "n_evals": len(losses),
            "max_step": max(
                (step for step in steps if step is not None),
                default=None,
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)
