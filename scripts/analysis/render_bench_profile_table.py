"""Render docs/notes/polar_product/walltime_profile.md packed_v1 cells.

Reads logs/bench_profile_packed_v1/blackwell_runs.jsonl (one JSONL record
per (model, lora_r, optimizer) cell, written by
scripts/bench/bench_optimizer_step.py) and emits a markdown table that
fits under §"Wall-time + MFU under packed_v1 (Blackwell)".

Usage:
    python scripts/analysis/render_bench_profile_table.py \
        [--in logs/bench_profile_packed_v1/blackwell_runs.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Tier inferred from model_name. Add new entries when introducing a new
# base model in the bench.
TIER_FROM_MODEL: dict[str, str] = {
    "allenai/OLMo-2-0425-1B":      "1B",
    "meta-llama/Llama-3.2-3B":     "3B",
    "meta-llama/Meta-Llama-3-8B":  "8B",
}


def short_optim(s: str) -> str:
    """Compact optimizer name for the markdown table."""
    if s == "adamw":
        return "adamw"
    if s == "adam-polar-product-lora-coupled-spectral-chord-tight":
        return "tight-chord-higham"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="src",
        default=str(Path(__file__).resolve().parent.parent.parent
                    / "logs" / "bench_profile_packed_v1" / "blackwell_runs.jsonl"),
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"# (no data yet at {src})", file=sys.stderr)
        return 0

    rows: list[dict] = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"# skip bad line: {e}", file=sys.stderr)
                continue
            if r.get("event") != "bench_step":
                continue
            rows.append(r)

    if not rows:
        print("# (no bench_step records found)", file=sys.stderr)
        return 0

    # Header.
    print(
        "| tier | r   | seq  | batch×accum | optim              | "
        "fwd ms | bwd ms | opt ms | total ms |  MFU  | peak MB |"
    )
    print(
        "|------|----:|-----:|-------------|--------------------|"
        "-------:|-------:|-------:|---------:|------:|--------:|"
    )
    # Sort: tier (1B, 3B, 8B), r ascending, optim (adamw before tight-chord).
    def sort_key(r):
        tier = TIER_FROM_MODEL.get(r["model_name"], r["model_name"])
        return (
            ["1B", "3B", "8B"].index(tier) if tier in ["1B", "3B", "8B"] else 99,
            r.get("lora_r", 0),
            0 if r.get("optimizer") == "adamw" else 1,
        )
    rows.sort(key=sort_key)
    for r in rows:
        tier = TIER_FROM_MODEL.get(r["model_name"], r["model_name"])
        mfu = r.get("mfu")
        mfu_str = f"{mfu*100:.1f}%" if mfu is not None else "—"
        print(
            f"| {tier:<4} | "
            f"{r['lora_r']:>3} | "
            f"{r['seq_len']:>4} | "
            f"{r['batch_size']}×{r['grad_accum_steps']:<10} | "
            f"{short_optim(r['optimizer']):<18} | "
            f"{r['fwd_sec_per_step']*1000:>6.0f} | "
            f"{r['bwd_sec_per_step']*1000:>6.0f} | "
            f"{r['opt_sec_per_step']*1000:>6.1f} | "
            f"{r['mean_sec_per_step']*1000:>8.0f} | "
            f"{mfu_str:>5} | "
            f"{r['peak_memory_mb']:>7.0f} |"
        )

    # Footer summary: per (tier, r), wall for 6k and 8.2k steps under AdamW.
    print()
    print("### Phase B/C wall-budget verdict (packed_v1, Blackwell)")
    print()
    print("| tier | r   | per-step (AdamW) | 6k wall | 8.2k wall (270M) |")
    print("|------|----:|-----------------:|--------:|-----------------:|")
    by_cell: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("optimizer") != "adamw":
            continue
        tier = TIER_FROM_MODEL.get(r["model_name"], r["model_name"])
        by_cell[(tier, r["lora_r"])] = r
    for (tier, lora_r), r in sorted(
        by_cell.items(),
        key=lambda kv: (
            ["1B", "3B", "8B"].index(kv[0][0]) if kv[0][0] in ["1B", "3B", "8B"] else 99,
            kv[0][1],
        ),
    ):
        per_step = r["mean_sec_per_step"]
        wall_6k = per_step * 6000 / 3600
        wall_8k = per_step * 8200 / 3600
        verdict_6k = "✓" if wall_6k <= 24 else "⚠"
        verdict_8k = "✓" if wall_8k <= 24 else ("⚠" if wall_8k <= 48 else "⚠⚠")
        print(
            f"| {tier:<4} | {lora_r:>3} | "
            f"{per_step*1000:>14.0f} ms | "
            f"{wall_6k:>5.1f}h {verdict_6k} | "
            f"{wall_8k:>13.1f}h {verdict_8k}     |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
