"""Audit a sweep params JSON against existing logs/ for already-completed cells.

Usage:
    python scripts/audit_sweep_overlap.py params/<sweep>.json

For each (config-tuple) in the cartesian product of the params file, search
logs/ for an existing run matching all keys (allowing string/numeric coercion).
Prints one line per cell: ✓ <group> if already exists, ☐ NEW otherwise.
At the end prints a one-line "drop these cells before submit" suggestion.

Designed as a pre-submit forcing function: ALWAYS run before
`./slurm_scripts/submit.sh` to enforce CLAUDE.md's "reuse existing data
before any new run" rule. Cells marked ✓ are wasted GPU.

NOTE: matches on cfg fields recorded by train.py at run time (lr, optimizer,
lora_r, seed, picard_alpha, picard_iters, lora_plus_multiplier, etc.).
For sweeps that vary a NEW config field train.py just learned about,
existing logs won't have that field — the audit treats missing-field as
"unknown, may or may not match"; check the printed match_keys to confirm
intent.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

# Map JSON sweep keys → cfg key as recorded by train.py log_event(config).
KEY_ALIASES = {
    "picard_iters_override": "picard_iters",   # config records the actual value
}

# Semantic defaults for fields that postdate older runs. If a sweep cell
# specifies one of these at the listed default value, an older run that
# lacks the field entirely is still a semantic match (the disabled/no-op
# value is what the older code path implicitly produced). Keep this list
# narrow — only fields whose default is a true no-op.
FIELD_DEFAULTS = {
    "anderson_m": 0,        # m=0 disables Anderson; older runs predate the path
    "anderson_reg": 1e-10,  # regularizer is a no-op when m=0
}


def load_params(p: Path) -> dict:
    return json.loads(p.read_text())


def cartesian(params: dict) -> list[dict]:
    keys = list(params.keys())
    rows = []
    for combo in itertools.product(*(params[k] for k in keys)):
        rows.append(dict(zip(keys, combo)))
    return rows


def coerce(v):
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v
    return v


def normalize_cell(cell: dict) -> list[dict]:
    """Expand a cell into one or more SEMANTICALLY-equivalent canonical forms.

    The cartesian-product cell's literal config may be equivalent to a
    different config recorded in older logs. Encoded equivalences:

    - `picard_alpha=0.0` zeros the cross-coupling term, making any
      picard_iters≥2 run equivalent to picard_iters=1 (uncoupled).
      Canonical form: drop picard_alpha, set picard_iters_override=1
      OR optimizer=adam-polar-product-lora.
    - `picard_iters_override=1` is equivalent to optimizer=
      adam-polar-product-lora (the uncoupled variant) regardless of
      picard_alpha.
    - `picard_alpha=1.0` with picard_iters_override=2 is the original
      coupled default, equivalent to optimizer=
      adam-polar-product-lora-coupled with no overrides recorded.

    Returns a list of cell dicts; existence of ANY matching log of any
    canonical form counts as overlap.
    """
    forms = [dict(cell)]

    pa = coerce(cell.get("picard_alpha"))
    pi = coerce(cell.get("picard_iters_override"))
    opt = cell.get("optimizer")

    # α=0 → uncoupled
    if pa == 0:
        c = {k: v for k, v in cell.items() if k != "picard_alpha"}
        c["optimizer"] = "adam-polar-product-lora"
        c.pop("picard_iters_override", None)
        forms.append(c)

    # picard_iters=1 → uncoupled regardless of α
    if pi == 1:
        c = {k: v for k, v in cell.items() if k not in ("picard_alpha", "picard_iters_override")}
        c["optimizer"] = "adam-polar-product-lora"
        forms.append(c)

    # α=1 with picard_iters=2 (or picard_iters_override absent) → original coupled default
    if (pa == 1 or pa is None) and (pi is None or pi == 2):
        c = {k: v for k, v in cell.items() if k not in ("picard_alpha", "picard_iters_override")}
        if opt == "adam-polar-product-lora-coupled":
            forms.append(c)

    return forms


def cfg_field_or_command(cfg: dict, key: str, command: str):
    """Return cfg[key] if present (and not None), else parse `--key VALUE`
    from the recorded command line. Returns None if neither has the field.

    Older runs / runs predating a CLI flag won't record the field in the
    cfg dict — falling back to the command string lets us still match.
    """
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    flag = "--" + key
    if flag in command:
        toks = command.split()
        try:
            idx = toks.index(flag)
            return toks[idx + 1]
        except (ValueError, IndexError):
            return None
    return None


def cell_matches_cfg(cell: dict, cfg: dict) -> bool:
    """A normalized cell-form matches cfg iff all its keys are present and equal.

    Tries every canonical form returned by normalize_cell — overlap is
    declared if ANY form matches.
    """
    command = cfg.get("command", "")
    for form in normalize_cell(cell):
        ok = True
        for k, v in form.items():
            cfg_key = KEY_ALIASES.get(k, k)
            cfg_val = cfg_field_or_command(cfg, cfg_key, command)
            if cfg_val is None:
                # Field doesn't exist in older run. If the cell value equals
                # the field's semantic default, treat as match (older code
                # path implicitly produced the same behavior).
                default = FIELD_DEFAULTS.get(cfg_key, "__SENTINEL_NO_DEFAULT__")
                if coerce(v) == default:
                    continue
                ok = False
                break
            if coerce(cfg_val) != coerce(v):
                ok = False
                break
        if ok:
            return True
    return False


def find_match(cell: dict, logs_root: Path) -> str | None:
    """Return group name of first matching run, or None."""
    for group_dir in logs_root.iterdir():
        if not group_dir.is_dir():
            continue
        ri = group_dir / "run_info" / "logs"
        if not ri.is_dir():
            continue
        for log_file in ri.glob("log_*.out"):
            cfg = None
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("event") == "config":
                            cfg = rec
                            break
            except Exception:
                continue
            if cfg is None:
                continue
            if cell_matches_cfg(cell, cfg):
                return group_dir.name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params_file")
    ap.add_argument("--logs-root", default="logs")
    args = ap.parse_args()

    params = load_params(Path(args.params_file))
    cells = cartesian(params)
    logs_root = Path(args.logs_root)

    print(f"Sweep params: {args.params_file}")
    print(f"Cartesian product size: {len(cells)} cells")
    print()
    overlap = []
    for cell in cells:
        match = find_match(cell, logs_root)
        if match:
            print(f"  ✓ EXISTS in {match:<40s}  {cell}")
            overlap.append(cell)
        else:
            print(f"  ☐ NEW                                       {cell}")
    print()
    if overlap:
        print(f"OVERLAP: {len(overlap)}/{len(cells)} cells already exist.")
        print("DROP these cells from the params file before submitting:")
        for c in overlap:
            print(f"  {c}")
        sys.exit(1)
    else:
        print(f"No overlap. All {len(cells)} cells are new.")
        sys.exit(0)


if __name__ == "__main__":
    main()
