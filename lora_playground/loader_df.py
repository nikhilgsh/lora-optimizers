"""Tidy-DataFrame wrapper over `load_runs`.

`load_runs` returns a list of `(cfg_dict, evals_list)` tuples — easy to mis-access
(the cfg is a positional element, not a `.cfg` attribute). This flattens each run
to one row: every cfg field becomes a column, plus `final_loss` / `min_loss` /
`n_evals` / `max_step` / `group`. Query with pandas instead of unpacking tuples:

    df = load_runs_df(where={"lora_r": 256})
    df[(df.polar_method == "ns") & df.ssc_kappa.isna() & ~df.precond_delta_relative]

`df.columns` is the schema — no field guessing.
"""
from __future__ import annotations

import re

import pandas as pd

from .loader import load_runs


def _group_from_cfg(cfg: dict) -> str | None:
    """Best-effort run-group name from the logged command / checkpoint paths."""
    cmd = cfg.get("command", "") or ""
    for flag in ("--checkpoint_dir", "--snapshot_dir", "--resume_from"):
        m = re.search(rf"{flag}\s+(\S+)", cmd)
        if m:
            # .../logs/<group>/run_info/... or .../<group>/task_N
            parts = [p for p in m.group(1).split("/") if p]
            for i, p in enumerate(parts):
                if p in ("logs", "lora_snapshots") and i + 1 < len(parts):
                    return parts[i + 1]
    return cfg.get("wandb_run_name")


def load_runs_df(where: dict | None = None, **kwargs) -> pd.DataFrame:
    """One row per run: cfg fields as columns + final_loss/min_loss/n_evals/max_step/group."""
    runs = load_runs(where=where or {}, **kwargs)
    rows = []
    for cfg, evals in runs:
        row = dict(cfg)
        losses = [e["eval_loss"] for e in evals
                  if isinstance(e, dict) and e.get("eval_loss") is not None]
        steps = [e.get("step") for e in evals if isinstance(e, dict)]
        row["final_loss"] = losses[-1] if losses else None
        row["min_loss"] = min(losses) if losses else None
        row["n_evals"] = len(losses)
        row["max_step"] = max((s for s in steps if s is not None), default=None)
        row["group"] = _group_from_cfg(cfg)
        rows.append(row)
    return pd.DataFrame(rows)
