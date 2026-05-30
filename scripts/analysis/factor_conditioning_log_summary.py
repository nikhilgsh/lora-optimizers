#!/usr/bin/env python
"""Extract factor-conditioning diagnostics from training JSONL logs.

The goal is to compare already-logged LoRA factor spectra and whitened
direction spectra across model/rank/run settings without re-running training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


METRIC_FIELDS = [
    "stable_rank_A_median",
    "stable_rank_B_median",
    "cond_SA_median",
    "cond_SB_median",
    "xunc_A_stable_rank_median",
    "xunc_B_stable_rank_median",
    "xunc_A_smax_over_tau_equal_median",
    "xunc_B_smax_over_tau_equal_median",
    "xunc_A_smedian_median",
    "xunc_B_smedian_median",
    "cos_A_median",
    "cos_B_median",
    "cos_polar_clip_A_median",
    "cos_polar_clip_B_median",
    "cos_polar_clip_tight_A_median",
    "cos_polar_clip_tight_B_median",
    "sat_frac_tight_A_median",
    "sat_frac_tight_B_median",
    "frac_dA_through_B_median",
    "frac_dB_through_A_median",
    "awc_q_agree_median",
    "awc_lambda_core_median",
    "chord_slack_median",
]

CONFIG_FIELDS = [
    "model_name",
    "data_pipeline_version",
    "optimizer",
    "lr",
    "lora_r",
    "lora_alpha",
    "muon_ns_steps",
    "picard_iters_override",
    "polar_method",
    "ns_form",
    "precond_method",
    "optim_diagnostics_every",
    "seed",
    "max_steps",
]


def _json_records(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _run_label(path: Path) -> str:
    parts = path.parts
    if "logs" in parts:
        i = len(parts) - 1 - parts[::-1].index("logs")
        if i > 0:
            return parts[i - 1]
    return path.parent.name


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted((p / "run_info" / "logs").glob("log_*.out"))
            if not candidates:
                candidates = sorted(p.glob("log_*.out"))
            paths.extend(candidates)
        else:
            paths.append(p)
    return [p for p in paths if p.exists()]


def parse_log(path: Path) -> list[dict]:
    cfg: dict = {}
    eval_losses: dict[int, float] = {}
    rows: list[dict] = []
    for rec in _json_records(path):
        event = rec.get("event")
        if event == "config":
            cfg = {k: rec.get(k) for k in CONFIG_FIELDS}
            opt_cfg = rec.get("optimizer_config") or {}
            if cfg.get("muon_ns_steps") is None:
                cfg["muon_ns_steps"] = opt_cfg.get("ns_steps")
        elif event == "eval" and rec.get("step") is not None:
            eval_losses[int(rec["step"])] = rec.get("eval_loss")
        elif event == "optim_step":
            r = cfg.get("lora_r") or rec.get("n_pairs")
            row = {
                "group": _run_label(path),
                "log": str(path),
                "task": path.stem,
                "step": rec.get("step"),
                **cfg,
            }
            for k in METRIC_FIELDS:
                row[k] = rec.get(k)
            if r:
                for base in (
                    "stable_rank_A",
                    "stable_rank_B",
                    "xunc_A_stable_rank",
                    "xunc_B_stable_rank",
                ):
                    v = row.get(f"{base}_median")
                    row[f"{base}_frac_median"] = (
                        float(v) / float(r) if v is not None else None
                    )
            rows.append(row)

    # Attach nearest eval loss at or after the diagnostic step for orientation.
    if eval_losses:
        eval_steps = sorted(eval_losses)
        for row in rows:
            step = row.get("step")
            if step is None:
                continue
            future = [s for s in eval_steps if s >= int(step)]
            if future:
                s = future[0]
                row["next_eval_step"] = s
                row["next_eval_loss"] = eval_losses[s]
    return rows


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict], steps: set[int]) -> list[dict]:
    selected = [r for r in rows if int(r.get("step") or -1) in steps]
    keys = ["model_name", "lora_r", "lr", "muon_ns_steps", "step"]
    buckets: dict[tuple, list[dict]] = {}
    for row in selected:
        key = tuple(row.get(k) for k in keys)
        buckets.setdefault(key, []).append(row)

    summary: list[dict] = []
    summary_fields = [
        "stable_rank_A_frac_median",
        "stable_rank_B_frac_median",
        "cond_SA_median",
        "cond_SB_median",
        "xunc_A_stable_rank_frac_median",
        "xunc_B_stable_rank_frac_median",
        "xunc_A_smax_over_tau_equal_median",
        "xunc_B_smax_over_tau_equal_median",
        "cos_A_median",
        "cos_B_median",
        "cos_polar_clip_A_median",
        "cos_polar_clip_B_median",
        "sat_frac_tight_A_median",
        "sat_frac_tight_B_median",
        "frac_dA_through_B_median",
        "frac_dB_through_A_median",
        "chord_slack_median",
        "next_eval_loss",
    ]
    for key, bucket in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out = dict(zip(keys, key))
        out["n_logs"] = len(bucket)
        out["groups"] = ";".join(sorted({str(r["group"]) for r in bucket}))
        for field in summary_fields:
            vals = [_safe_float(r.get(field)) for r in bucket]
            vals = [v for v in vals if v is not None]
            out[field] = median(vals) if vals else None
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Run group dirs or log_*.out files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--steps", default="100,200,500,1000,4000,9000")
    args = ap.parse_args()

    paths = _expand_inputs(args.inputs)
    rows: list[dict] = []
    for path in paths:
        rows.extend(parse_log(path))
    steps = {int(x) for x in args.steps.split(",") if x}
    summary = summarize(rows, steps)

    write_csv(Path(args.out), rows)
    write_csv(Path(args.summary_out), summary)

    print(f"logs={len(paths)} optim_rows={len(rows)} summary_rows={len(summary)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    for row in summary:
        model = str(row.get("model_name", "")).split("/")[-1]
        print(
            " ".join(
                [
                    f"model={model}",
                    f"r={row.get('lora_r')}",
                    f"lr={row.get('lr')}",
                    f"ns={row.get('muon_ns_steps')}",
                    f"step={row.get('step')}",
                    f"srB/r={row.get('stable_rank_B_frac_median'):.3f}",
                    f"condSB={row.get('cond_SB_median'):.1f}",
                    f"xA/r={row.get('xunc_A_stable_rank_frac_median'):.3f}",
                    f"xB/r={row.get('xunc_B_stable_rank_frac_median'):.3f}",
                    f"cosA={row.get('cos_A_median'):.3f}",
                    f"throughA={row.get('frac_dA_through_B_median'):.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
