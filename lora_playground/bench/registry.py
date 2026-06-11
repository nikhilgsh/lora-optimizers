"""Close the bench → ``timing_registry.json`` loop.

``lora_playground.timing`` records per-step cost from a *training log* (which is
wall-inclusive: fwd+bwd+opt+eval+ckpt). A bench step-wall result is a *different*
quantity — fwd+bwd+opt+zero only, no eval/checkpoint — so it slightly
UNDER-estimates the real wall. We therefore:

  * tag bench-derived entries ``source="bench_step_wall"``, ``wall_inclusive=false``;
  * NEVER clobber an authoritative train-log entry (no ``source`` field, or
    ``source="train_log"``) with a bench entry unless ``force=True``.

This lets a bench *bootstrap* a registry key that no run has produced yet (so the
sbatch wall guard stops blocking on a miss), while keeping log-derived entries —
the ones the wall buffer is calibrated against — authoritative.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .. import timing


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(timing._repo_root()),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _cfg_from_result(result: dict) -> dict:
    """Shape a bench step-wall result into the cfg-dict ``timing.key_fields``
    consumes. ``cw_picard_iters`` maps to the registry's ``picard`` axis."""
    return {
        "model_name": result.get("model_name"),
        "lora_r": result.get("lora_r"),
        "optimizer": result.get("optimizer"),
        "muon_ns_steps": result.get("muon_ns_steps"),
        "picard_iters": result.get("cw_picard_iters", result.get("picard_iters", 1)),
        "batch_size": result.get("batch_size"),
        "grad_accum_steps": result.get("grad_accum_steps"),
        "max_seq_length": result.get("max_seq_length"),
        "bf16": result.get("bf16", True),
        "compile": bool(result.get("compile_mode")),
        "git_commit": result.get("git_commit") or _git_commit(),
    }


def record_bench(result: dict, hardware: str, *, force: bool = False,
                 path: Optional[Path] = None) -> dict:
    """Upsert a registry entry from a harness step-wall ``result``.

    ``result`` must carry ``mean_sec_per_step`` plus the key/provenance fields
    (model_name, lora_r, optimizer, muon_ns_steps, cw_picard_iters, batch_size,
    grad_accum_steps, max_seq_length, compile_mode, bf16). Returns the entry that
    is now in the registry (existing authoritative one if not overwritten).
    """
    if "mean_sec_per_step" not in result:
        raise KeyError("result must contain 'mean_sec_per_step'")
    cfg = _cfg_from_result(result)
    fields = timing.key_fields(cfg, hardware=hardware)
    key = timing.make_key(fields)

    entries = timing.load_registry(path)
    existing = next((e for e in entries if e.get("key") == key), None)
    if existing is not None and not force:
        # Do not let a (non-wall-inclusive) bench entry overwrite an authoritative
        # train-log entry. A prior bench entry (same source) is fine to refresh.
        if existing.get("source", "train_log") != "bench_step_wall":
            return existing

    entry = {
        "key": key,
        "s_per_step": round(float(result["mean_sec_per_step"]), 4),
        "measured_at_step": None,
        "source_log": None,
        "source": "bench_step_wall",
        "wall_inclusive": False,
        "n_reps": result.get("n_reps"),
        "precond_method": result.get("precond_method"),
        **fields,
    }
    entries = [e for e in entries if e.get("key") != key]
    entries.append(entry)
    entries.sort(key=lambda e: e["key"])
    timing.save_registry(entries, path)
    return entry
