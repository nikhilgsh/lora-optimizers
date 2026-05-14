#!/usr/bin/env python
"""scripts/backfill_cfg_events.py — one-shot backfill of legacy cfg events.

Phase 2 of the data-loading redesign (see
~/.claude/plans/the-data-loading-in-wobbly-pelican.md). For each
logs/<group>/run_info/logs/log_N.out:

  - parse line 1 as the cfg event (JSON)
  - skip if the file looks in-flight (mtime newer than IN_FLIGHT_GRACE_S)
  - skip if cfg already carries `optimizer_effective` (Phase 1 emit, idempotent)
  - else run forensic reconstruction (lora_playground.loader internals) and
    merge resolved fields onto the cfg
  - atomically rewrite the file: write to .tmp, verify (line 1 is valid JSON,
    total event count unchanged), rename original to .bak, rename .tmp to
    the original. The .bak sibling stays around as one-shot recovery.

A run-by-run audit manifest is written to backfill_records/backfill_<ts>.json:
path, group, log_filename, sha256 before/after, list of uncertain fields. This
file IS the durable audit trail (logs/ is gitignored).

Usage:
  python scripts/backfill_cfg_events.py --dry-run
  python scripts/backfill_cfg_events.py --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.loader import (  # noqa: E402
    HISTORICAL_DEFAULTS_WHEN_MISSING,
    _argparse_defaults,
    _backfill_optimizer_config,
    _derive_effective_inner_polar,
    _derive_effective_picard_iters,
)
from lora_playground.manifest import load_manifests  # noqa: E402


# Files modified within this many seconds are treated as potentially in-flight
# and skipped. Real training runs write a final `eval` event then exit; their
# mtime stops advancing immediately after. 10 minutes is a safe cushion for
# in-flight detection without being long enough to block backfill of recently
# finished work indefinitely.
IN_FLIGHT_GRACE_S = 600


def _resolve_effective(cfg: dict) -> tuple[dict, list[str]]:
    """Return (optimizer_effective_dict, uncertain_fields).

    Mirrors what loader._enrich_cfg currently computes via `_derive_*`. We
    call the same helpers directly so a Phase-3 forensic-layer deletion
    doesn't accidentally change semantics between backfill and live load.
    """
    opt_cfg = cfg.get("optimizer_config")
    if opt_cfg is None:
        opt_cfg = _backfill_optimizer_config(cfg)
    out: dict = {}
    uncertain: list[str] = []
    inner = _derive_effective_inner_polar(cfg, opt_cfg)
    if inner is not None:
        out["effective_inner_polar"] = inner
    # picard_iters is a polar-product-family concept. For non-polar optimizers
    # (AdamW, Muon, …) `_derive_effective_picard_iters` returns (None, False)
    # but the False is N/A noise, not genuine uncertainty. We only flag
    # uncertain when the optimizer plausibly HAS the concept.
    optimizer_name = cfg.get("optimizer", "") or ""
    is_polar_family = "polar-product" in optimizer_name
    k, k_certain = _derive_effective_picard_iters(cfg, opt_cfg)
    if k is not None:
        out["effective_picard_iters"] = int(k)
        if is_polar_family and not k_certain:
            uncertain.append("effective_picard_iters")
    elif is_polar_family:
        # Polar-product optimizer that couldn't be resolved at all — genuine
        # uncertainty. Phase 3 invariants/attestations resolve.
        out["effective_picard_iters"] = None
        uncertain.append("effective_picard_iters")
    # For non-polar optimizers, omit the field entirely (N/A, not unresolved).
    return out, uncertain


def _resolve_diagnostics(cfg: dict) -> dict:
    """Canonical-name diagnostics block, resolving the legacy three-way alias
    chain that loader._enrich_cfg currently maintains in-memory at load time.
    """
    opt_cfg = cfg.get("optimizer_config") or {}
    basic = (
        cfg.get("log_basic_diagnostics")
        or cfg.get("log_optim_diagnostics")
        or opt_cfg.get("log_basic_diagnostics")
        or opt_cfg.get("log_diagnostics")
    )
    heavy = cfg.get("log_heavy_diagnostics") or opt_cfg.get("log_heavy_diagnostics")
    every = cfg.get("optim_diagnostics_every") or opt_cfg.get("diagnostics_every")
    return {
        "basic": bool(basic),
        "heavy": bool(heavy),
        "every": int(every) if every not in (None, "None") else 0,
    }


def _backfill_provenance(cfg: dict, manifest: dict | None) -> dict:
    """Provenance fields. Legacy cfgs don't carry git_dirty / git_diff_sha /
    git_untracked_files — fall back to manifest-level git_dirty (sweep-level,
    less precise but our only retrospective signal). git_diff_sha and
    git_untracked_files default to None / [] for legacy.
    """
    if "git_dirty" in cfg:
        dirty = bool(cfg["git_dirty"])
    elif manifest is not None and "git_dirty" in manifest:
        dirty = bool(manifest["git_dirty"])
    else:
        dirty = False
    return {
        "git_dirty": dirty,
        "git_diff_sha": cfg.get("git_diff_sha"),
        "git_untracked_files": cfg.get("git_untracked_files") or [],
    }


def _find_cfg_event(path: Path) -> tuple[dict | None, list[str], int]:
    """Locate the config event in a .out file. Lines BEFORE it (banner text,
    pre-event prints) are preserved verbatim; the config event itself is
    parsed; lines AFTER it are preserved verbatim too.

    Returns (parsed cfg or None, full list of original lines, index of the
    cfg line within that list). When no config event exists, returns
    (None, lines, -1) so the caller can report unparseable.
    """
    text = path.read_text()
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "config":
            return obj, lines, i
    return None, lines, -1


def _atomic_rewrite(path: Path, lines: list[str], cfg_idx: int,
                    new_cfg_line: str) -> None:
    """Replace `lines[cfg_idx]` with `new_cfg_line`, write atomically.

    Verification: tmp's line at cfg_idx parses as JSON AND has event==config;
    total non-empty-line count unchanged. Then original → .bak, .tmp → original.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    new_lines = list(lines)
    new_lines[cfg_idx] = new_cfg_line
    # Preserve the original's trailing-newline convention: read_text + splitlines
    # drops trailing newline info, so we add one back so log files end in '\n'.
    new_text = "\n".join(new_lines) + "\n"
    tmp.write_text(new_text)
    # Verify: at cfg_idx, line is valid JSON with event==config.
    tmp_lines = tmp.read_text().splitlines()
    try:
        verify_cfg = json.loads(tmp_lines[cfg_idx])
    except (json.JSONDecodeError, IndexError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Phase-2 verify: tmp line {cfg_idx} not valid JSON: {e}")
    if verify_cfg.get("event") != "config":
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Phase-2 verify: line {cfg_idx} event={verify_cfg.get('event')!r} "
            "(expected 'config')"
        )
    orig_evs = sum(1 for ln in lines if ln.strip())
    new_evs = sum(1 for ln in tmp_lines if ln.strip())
    if orig_evs != new_evs:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Phase-2 verify: non-empty line count {orig_evs} → {new_evs} (must match)"
        )
    if bak.exists():
        bak.unlink()
    path.rename(bak)
    tmp.rename(path)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="Report planned changes; do not write.")
    g.add_argument("--apply", action="store_true",
                   help="Atomically rewrite files; create .bak siblings; "
                        "write audit manifest to --records-dir.")
    ap.add_argument("--logs-root", default=str(ROOT / "logs"))
    ap.add_argument("--records-dir", default=str(ROOT / "backfill_records"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N files (testing).")
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    records_dir = Path(args.records_dir)
    if args.apply:
        records_dir.mkdir(exist_ok=True)

    manifests = load_manifests(str(logs_root), strict=False)
    manifest_by_group = {m.get("group"): m for m in manifests}

    out_files = sorted(logs_root.glob("*/run_info/logs/*.out"))
    if args.limit is not None:
        out_files = out_files[: args.limit]
    print(f"Found {len(out_files)} .out files under {logs_root}.")

    summary = {
        "backfilled": 0,
        "skipped_inflight": 0,
        "skipped_already_new_schema": 0,
        "skipped_unparseable_cfg": 0,
        "errors": 0,
        "uncertain_runs": 0,
    }
    records = []
    defaults = _argparse_defaults()

    for path in out_files:
        try:
            mtime = path.stat().st_mtime
            if (time.time() - mtime) < IN_FLIGHT_GRACE_S:
                summary["skipped_inflight"] += 1
                continue
            cfg, lines, cfg_idx = _find_cfg_event(path)
            if cfg is None:
                summary["skipped_unparseable_cfg"] += 1
                continue
            if "optimizer_effective" in cfg:
                summary["skipped_already_new_schema"] += 1
                continue
            group = path.parent.parent.parent.name
            manifest = manifest_by_group.get(group)

            opt_eff, uncertain = _resolve_effective(cfg)
            diag = _resolve_diagnostics(cfg)
            prov = _backfill_provenance(cfg, manifest)

            # IMPORTANT: do NOT write argparse defaults into the cfg event.
            # Many flags' argparse defaults are `None` (e.g. data_dir), and
            # the real values come from the `_cli_args` blanket dump that
            # load_run.setdefault-chains into cfg at read time. Writing
            # `None` explicitly here would pre-empt that fill and the loader
            # would see `None` instead of the actual CLI value. The loader's
            # _argparse_defaults() path stays alive through Phase 3; we
            # only retire it when Phase 3 also retires `_cli_args` reliance.

            cfg["optimizer_effective"] = opt_eff
            cfg["diagnostics"] = diag
            for k, v in prov.items():
                cfg.setdefault(k, v)
            if uncertain:
                cfg["_backfill_uncertain"] = uncertain
                summary["uncertain_runs"] += 1

            new_cfg_line = json.dumps(cfg)
            record = {
                "path": str(path.relative_to(ROOT)),
                "group": group,
                "log_filename": path.name,
                "cfg_line_index": cfg_idx,
                "uncertain_fields": uncertain,
            }
            if args.apply:
                record["sha256_original"] = _sha256_of(path)
                _atomic_rewrite(path, lines, cfg_idx, new_cfg_line)
                record["sha256_new"] = _sha256_of(path)
            records.append(record)
            summary["backfilled"] += 1
        except Exception as e:
            summary["errors"] += 1
            print(f"[error] {path}: {type(e).__name__}: {e}")

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:>30}: {v}")

    if args.apply and records:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = records_dir / f"backfill_{ts}.json"
        manifest_path.write_text(json.dumps({
            "timestamp_iso": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "logs_root": str(logs_root),
            "summary": summary,
            "records": records,
        }, indent=2) + "\n")
        print(f"\nAudit manifest written: {manifest_path.relative_to(ROOT)}")
        print(f"({len(records)} run(s) recorded.)")


if __name__ == "__main__":
    main()
