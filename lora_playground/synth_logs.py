"""Synthesize subhorizon log groups from longer-horizon runs.

When the LR schedule is constant, an 8k-step run's step-2000 eval is
behaviorally identical to a 2k-step run's step-2000 eval (same iterates,
same LR at every step, just stopped earlier). This module materializes that
equivalence as a new log group on disk so the canonical loader picks it up
and the 2k-horizon notebook automatically gains the missing cells.

Hard preconditions per source log file:
  1. The cfg's `optimizer_config` does not declare a non-constant schedule
     (no warmup, no decay, no cosine — anything that varies LR with step).
  2. Every `eval` event's `lr` equals the cfg's `lr` (sanity check that the
     run actually ran with constant LR — catches schedule bugs in cfg).
  3. `target_horizon` matches an eval event's step exactly.
  4. `target_horizon` does not exceed steps-per-epoch — i.e. the cut lands
     within epoch 0, where the data loader's batch ordering is identical
     to a fresh `max_steps=target_horizon` run with the same seed.
     Beyond epoch 0, a multi-epoch source has reshuffled and consumed extra
     RNG state; the trajectory diverges from a single-pass twin.

If any precondition fails, the source log is skipped with a loud warning
and the group is NOT silently truncated. Synthesized logs are tagged in
the manifest with `synthesized_from` and `synthesized_horizon` for
provenance.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path


_NON_CONSTANT_SCHEDULE_HINTS = (
    "warmup", "warmup_steps", "lr_warmup", "warmup_ratio",
    "schedule", "lr_schedule", "decay", "lr_decay",
    "cosine", "lr_cosine", "min_lr",
)


def _has_non_constant_schedule(cfg: dict) -> tuple[bool, str]:
    """Return (True, reason) if the cfg declares a non-constant LR schedule.

    Looks at top-level cfg keys plus optimizer_config nested keys. A field
    counts as "declaring a schedule" only when its value is truthy
    (non-zero, non-empty, non-None) — leaving the field at default 0/None
    is the constant case.
    """
    def _check(d: dict, prefix: str = "") -> tuple[bool, str]:
        for k, v in d.items():
            kl = k.lower()
            for hint in _NON_CONSTANT_SCHEDULE_HINTS:
                if hint in kl and v not in (None, 0, 0.0, False, "", "none", "constant"):
                    return True, f"{prefix}{k}={v!r}"
        return False, ""

    bad, why = _check(cfg)
    if bad:
        return True, why
    optcfg = cfg.get("optimizer_config", {})
    if isinstance(optcfg, dict):
        bad, why = _check(optcfg, prefix="optimizer_config.")
        if bad:
            return True, why
    return False, ""


def _verify_constant_lr(cfg: dict, eval_events: list[dict]) -> tuple[bool, str]:
    """All eval events must have lr == cfg.lr."""
    nominal = float(cfg["lr"])
    for ev in eval_events:
        actual = ev.get("lr")
        if actual is None:
            continue  # older logs may omit it; not authoritative
        if abs(float(actual) - nominal) > 1e-12:
            return False, f"step={ev['step']} lr={actual} ≠ cfg.lr={nominal}"
    return True, ""


def materialize_subhorizon_log(
    src_log_path: Path,
    target_horizon: int,
    dest_log_path: Path,
) -> dict:
    """Read one source log file, synthesize a target_horizon log file at
    `dest_log_path`. Returns a status dict; raises ValueError on any
    precondition failure (caller decides whether to abort the whole group
    or skip this file)."""
    src_log_path = Path(src_log_path)
    dest_log_path = Path(dest_log_path)

    config_event = None
    eval_events = []
    other_events = []
    for line in src_log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # non-JSON lines (tqdm, wandb noise) — skip
        if not isinstance(ev, dict) or "event" not in ev:
            continue
        if ev["event"] == "config":
            config_event = ev
        elif ev["event"] == "eval":
            eval_events.append(ev)
        else:
            other_events.append(ev)

    if config_event is None:
        raise ValueError(f"{src_log_path}: no config event found")
    if not eval_events:
        raise ValueError(f"{src_log_path}: no eval events found")

    bad, why = _has_non_constant_schedule(config_event)
    if bad:
        raise ValueError(f"{src_log_path}: non-constant LR schedule declared ({why})")

    ok, why = _verify_constant_lr(config_event, eval_events)
    if not ok:
        raise ValueError(f"{src_log_path}: per-step LR varies ({why})")

    src_max_steps = int(config_event.get("max_steps", 0))
    if target_horizon >= src_max_steps:
        raise ValueError(
            f"{src_log_path}: target_horizon={target_horizon} not strictly less "
            f"than source max_steps={src_max_steps}; would not be a sub-run"
        )

    # Within-epoch-0 check: salvage is only valid where the data loader has
    # not reshuffled. steps_per_epoch = train_samples / (batch_size * grad_accum).
    train_samples = config_event.get("train_samples")
    batch_size = config_event.get("batch_size")
    grad_accum = config_event.get("grad_accum_steps")
    if train_samples and batch_size and grad_accum:
        steps_per_epoch = int(train_samples) // (int(batch_size) * int(grad_accum))
        if target_horizon > steps_per_epoch:
            raise ValueError(
                f"{src_log_path}: target_horizon={target_horizon} exceeds "
                f"steps_per_epoch={steps_per_epoch} "
                f"(train_samples={train_samples}, batch_size={batch_size}, "
                f"grad_accum={grad_accum}); salvage is only valid within "
                f"epoch 0 where the data loader ordering is deterministic."
            )

    target_evals = [e for e in eval_events if int(e["step"]) <= target_horizon]
    if not target_evals or int(target_evals[-1]["step"]) != target_horizon:
        seen = sorted({int(e["step"]) for e in eval_events})
        raise ValueError(
            f"{src_log_path}: no eval event at step={target_horizon}; "
            f"available steps near target: {[s for s in seen if s <= target_horizon * 1.5][-6:]}"
        )

    # Build the synthesized config.
    new_cfg = dict(config_event)
    new_cfg["max_steps"] = target_horizon
    new_cfg["_synth_source"] = str(src_log_path)
    new_cfg["_synth_source_max_steps"] = src_max_steps
    # Patch the recorded command-line max_steps so reproducibility checks
    # don't get confused.
    cmd = new_cfg.get("command", "")
    new_cfg["command"] = _patch_command_max_steps(cmd, target_horizon)

    # Write the synthesized log.
    dest_log_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_log_path.open("w") as f:
        f.write(json.dumps(new_cfg) + "\n")
        for ev in target_evals:
            f.write(json.dumps(ev) + "\n")

    return {
        "src": str(src_log_path),
        "dest": str(dest_log_path),
        "n_eval_events": len(target_evals),
        "src_max_steps": src_max_steps,
        "target_horizon": target_horizon,
        "lr": float(new_cfg["lr"]),
        "optimizer": new_cfg.get("optimizer"),
        "lora_r": new_cfg.get("lora_r"),
    }


def _patch_command_max_steps(cmd: str, target: int) -> str:
    """Replace `--max_steps N` token in command string with the target."""
    if not cmd:
        return cmd
    parts = cmd.split()
    out = []
    skip_next = False
    for i, p in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if p == "--max_steps" and i + 1 < len(parts):
            out.append("--max_steps")
            out.append(str(target))
            skip_next = True
        else:
            out.append(p)
    return " ".join(out)


def materialize_subhorizon_group(
    src_group_dir: Path,
    target_horizon: int,
    dest_group_dir: Path,
    *,
    dest_group_name: str | None = None,
    keep_partial: bool = False,
) -> dict:
    """Synthesize a whole sub-horizon group. Reads every `log_*.out` in
    `src_group_dir/run_info/logs/`, verifies preconditions, writes the
    truncated logs to `dest_group_dir/run_info/logs/`, and emits a fresh
    `run_info/meta.json` with provenance fields.

    Returns {"materialized": [...], "skipped": [...], "dest": dest_group_dir}.
    """
    src_group_dir = Path(src_group_dir)
    dest_group_dir = Path(dest_group_dir)
    dest_group_name = dest_group_name or dest_group_dir.name

    src_logs_dir = src_group_dir / "run_info" / "logs"
    if not src_logs_dir.is_dir():
        raise FileNotFoundError(f"{src_logs_dir} not found")

    src_meta_path = src_group_dir / "run_info" / "meta.json"
    if not src_meta_path.exists():
        raise FileNotFoundError(f"{src_meta_path} not found")
    src_meta = json.loads(src_meta_path.read_text())

    dest_logs_dir = dest_group_dir / "run_info" / "logs"
    dest_logs_dir.mkdir(parents=True, exist_ok=True)

    materialized = []
    skipped = []
    for src in sorted(src_logs_dir.glob("log_*.out")):
        dest = dest_logs_dir / src.name
        try:
            info = materialize_subhorizon_log(src, target_horizon, dest)
            materialized.append(info)
        except ValueError as e:
            skipped.append({"src": str(src), "reason": str(e)})
            warnings.warn(f"skip {src.name}: {e}")

    if not materialized and not keep_partial:
        raise RuntimeError(
            f"materialize_subhorizon_group: no logs were materialized "
            f"({len(skipped)} skipped). Pass keep_partial=True to write "
            f"an empty group anyway."
        )

    # Emit a fresh manifest tagging provenance.
    new_meta = dict(src_meta)
    new_meta["group"] = dest_group_name
    new_meta["synthesized_from"] = src_meta.get("group")
    new_meta["synthesized_horizon"] = target_horizon
    new_meta["synthesized_n_logs"] = len(materialized)
    new_meta["synthesized_n_skipped"] = len(skipped)
    new_meta["purpose"] = (
        f"Synthesized step-{target_horizon} subhorizon copy of "
        f"{src_meta.get('group')!r} (constant-LR schedule verified). "
        f"Original purpose: {src_meta.get('purpose', '')!r}"
    )
    # Add 'synthesized' scope tag so analysts can filter.
    scopes = list(new_meta.get("scope", []))
    if "synthesized" not in scopes:
        scopes.append("synthesized")
    new_meta["scope"] = scopes

    (dest_group_dir / "run_info").mkdir(parents=True, exist_ok=True)
    (dest_group_dir / "run_info" / "meta.json").write_text(
        json.dumps(new_meta, indent=2) + "\n"
    )

    return {
        "materialized": materialized,
        "skipped": skipped,
        "dest_group_dir": str(dest_group_dir),
        "dest_group_name": dest_group_name,
    }


def _cli():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--src-group", required=True,
                   help="Path to source log group dir (e.g. logs/polar_k3_8k_partA)")
    p.add_argument("--target-horizon", required=True, type=int,
                   help="Step count to truncate at (must match an eval event)")
    p.add_argument("--dest-group", required=True,
                   help="Path to destination log group dir (will be created)")
    p.add_argument("--dest-group-name", default=None,
                   help="Manifest 'group' field (default: basename of --dest-group)")
    p.add_argument("--keep-partial", action="store_true",
                   help="Write the destination even if 0 logs materialized")
    args = p.parse_args()

    res = materialize_subhorizon_group(
        Path(args.src_group),
        args.target_horizon,
        Path(args.dest_group),
        dest_group_name=args.dest_group_name,
        keep_partial=args.keep_partial,
    )
    print(f"materialized {len(res['materialized'])} logs to {res['dest_group_dir']}")
    if res["skipped"]:
        print(f"skipped {len(res['skipped'])}:")
        for s in res["skipped"]:
            print(f"  - {Path(s['src']).name}: {s['reason']}")


if __name__ == "__main__":
    _cli()
