"""Per-step wall-time registry — so the SLURM wall guard anchors on a measured
cost instead of a hand-typed ETA, and we never re-measure the same config.

Motivation: a 9000-step full-polar r64 sweep was given a 12h wall on a ~3×-too-low
ETA and timed out at 98.7%. The fix anchors ``--time`` on ``max_steps × measured
s/step × tasks_per_gpu × buffer``. This module is the measurement store.

Registry file: ``timing_registry.json`` at repo root. Each entry is keyed by the
axes that actually move per-step cost (matrix ops are O(d^3)):

    hardware, model (short), lora_r, optimizer_family, ns_steps, picard

The remaining run knobs (batch_size, grad_accum_steps, max_seq_length, dtype,
compile) are fixed across this project's optimizer comparisons (see CLAUDE.md
"Fix across all optimizer comparisons"), so they're recorded on each entry for
provenance but are NOT part of the lookup key. ``git_commit`` is stored so the
wall hook can warn when the entry predates an optimizer-code change that may have
shifted cost.

``s_per_step`` is stored *wall-inclusive* (it already covers eval + diagnostics +
checkpoint overhead, since it's derived from cumulative ``train_elapsed_sec``).
Do NOT pre-inflate it with a safety factor — the wall guard applies the buffer.

CLI:
    python -m lora_playground.timing record <log.out> --hardware blackwell
    python -m lora_playground.timing lookup  --key blackwell/OLMo-2-0425-1B/r64/polar-product/ns8/k1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def registry_path() -> Path:
    return _repo_root() / "timing_registry.json"


# ─── config → key field extraction ────────────────────────────────────────────

def _model_short(model_name: Optional[str]) -> str:
    if not model_name:
        return "unknown"
    return str(model_name).rstrip("/").split("/")[-1]


def optimizer_family(optimizer: Optional[str]) -> str:
    """Coarse family that shares a per-step cost profile. Substring priority;
    falls back to the full optimizer name so an unknown optimizer still keys
    deterministically (just without family-level sharing)."""
    o = (optimizer or "").lower()
    if "polar-product" in o:
        return "polar-product"
    if "adamuon" in o or "ada-muon" in o:
        return "adamuon"
    if "muon" in o:
        return "muon"
    if "lin-lora" in o or "linlora" in o:
        return "lin-lora"
    if "curvature-whiten" in o or "curvature_whiten" in o:
        return "curvature-whiten"
    if "galore" in o:
        return "galore"
    if o in ("adamw", "adam"):
        return "adamw"
    return o or "unknown"


def _first(cfg: dict, *keys, default=None):
    """Return the first present, non-None value among cfg[key] and
    cfg['_cli_args'][key] / cfg['optimizer_config'][key] for each key."""
    cli = cfg.get("_cli_args") or {}
    oc = cfg.get("optimizer_config") or {}
    der = cfg.get("_derived") or {}
    for k in keys:
        for src in (cfg, cli, oc, der):
            v = src.get(k)
            if v is not None:
                return v
    return default


def key_fields(cfg: dict, hardware: Optional[str] = None) -> dict:
    """Extract the registry key fields + provenance fields from a config event
    (or any cfg-shaped dict)."""
    ns = _first(cfg, "muon_ns_steps", "ns_steps")
    picard = _first(cfg, "picard_iters_override", "picard_iters", "effective_picard_iters")
    return {
        "hardware": (hardware or cfg.get("hardware") or "unknown"),
        "model": _model_short(cfg.get("model_name") or _first(cfg, "model_name")),
        "lora_r": _first(cfg, "lora_r"),
        "optimizer_family": optimizer_family(cfg.get("optimizer") or _first(cfg, "optimizer")),
        "ns_steps": (int(ns) if ns is not None else None),
        "picard": (int(picard) if picard is not None else 1),
        # provenance (not part of key)
        "batch_size": _first(cfg, "batch_size"),
        "grad_accum_steps": _first(cfg, "grad_accum_steps"),
        "max_seq_length": _first(cfg, "max_seq_length"),
        "compile": bool(_first(cfg, "compile", default=False)),
        "dtype": ("bf16" if _first(cfg, "bf16", default=False) else "fp32"),
        "git_commit": (cfg.get("git_commit") or _first(cfg, "git_commit")),
    }


def make_key(fields: dict) -> str:
    """The string form used by the wall hook's ``# TIMING_KEY:`` header."""
    return (f"{fields['hardware']}/{fields['model']}/r{fields['lora_r']}/"
            f"{fields['optimizer_family']}/ns{fields['ns_steps']}/k{fields['picard']}")


# ─── registry I/O ─────────────────────────────────────────────────────────────

def load_registry(path: Optional[Path] = None) -> list[dict]:
    p = Path(path) if path else registry_path()
    if not p.exists():
        return []
    return json.loads(p.read_text())


def save_registry(entries: list[dict], path: Optional[Path] = None) -> None:
    p = Path(path) if path else registry_path()
    p.write_text(json.dumps(entries, indent=2) + "\n")


def lookup(cfg_or_key, hardware: Optional[str] = None,
           path: Optional[Path] = None) -> Optional[tuple[float, dict]]:
    """Look up s/step. ``cfg_or_key`` is either a cfg-shaped dict or a key string
    (``hardware/model/rN/family/nsN/kN``). Returns (s_per_step, entry) or None."""
    key = cfg_or_key if isinstance(cfg_or_key, str) else make_key(key_fields(cfg_or_key, hardware))
    for e in load_registry(path):
        if e.get("key") == key:
            return (float(e["s_per_step"]), e)
    return None


# ─── measurement from a run log ───────────────────────────────────────────────

def s_per_step_from_log(log_path: str | Path) -> tuple[float, int, float]:
    """Parse a training stdout JSONL log and return
    (s_per_step, last_step, last_elapsed_sec).

    Uses cumulative ``train_elapsed_sec`` at ``eval`` events. The per-step rate is
    the median of consecutive-eval deltas, skipping the first two evals (compile
    warmup biases the very first interval). Wall-inclusive by construction.
    """
    evals: list[tuple[int, float]] = []
    for line in Path(log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "eval" and "step" in e and "train_elapsed_sec" in e:
            evals.append((int(e["step"]), float(e["train_elapsed_sec"])))
    if len(evals) < 3:
        raise ValueError(f"{log_path}: need >=3 eval events to measure per-step "
                         f"(found {len(evals)})")
    evals.sort()
    deltas = []
    for i in range(2, len(evals)):  # skip evals[0] (warmup) and the first interval
        ds = evals[i][0] - evals[i - 1][0]
        dt = evals[i][1] - evals[i - 1][1]
        if ds > 0:
            deltas.append(dt / ds)
    last_step, last_el = evals[-1]
    # Conservative: the larger of steady-state and warmup-inclusive overall rate.
    return (max(median(deltas), last_el / last_step), last_step, last_el)


def _config_event(log_path: str | Path) -> dict:
    for line in Path(log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "config":
            return e
    raise ValueError(f"{log_path}: no config event found")


def record(log_path: str | Path, hardware: str,
           path: Optional[Path] = None) -> dict:
    """Measure s/step from ``log_path``, build a registry entry keyed off its
    config event, and upsert it into the registry (replacing any entry with the
    same key). ``hardware`` is supplied by the caller because run logs don't
    record the GPU model reliably. Returns the entry."""
    cfg = _config_event(log_path)
    s_per_step, last_step, _ = s_per_step_from_log(log_path)
    fields = key_fields(cfg, hardware=hardware)
    entry = {
        "key": make_key(fields),
        "s_per_step": round(s_per_step, 3),
        "measured_at_step": last_step,
        "source_log": str(log_path),
        **fields,
    }
    entries = [e for e in load_registry(path) if e.get("key") != entry["key"]]
    entries.append(entry)
    entries.sort(key=lambda e: e["key"])
    save_registry(entries, path)
    return entry


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="per-step timing registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="measure a run log and upsert an entry")
    r.add_argument("log_path")
    r.add_argument("--hardware", required=True)
    lk = sub.add_parser("lookup", help="look up s/step by key")
    lk.add_argument("--key", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "record":
        entry = record(args.log_path, args.hardware)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "lookup":
        hit = lookup(args.key)
        if hit is None:
            print(f"MISS: {args.key}")
            return 1
        s, entry = hit
        print(f"{s} s/step  ({entry['source_log']}, commit {str(entry.get('git_commit'))[:8]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
