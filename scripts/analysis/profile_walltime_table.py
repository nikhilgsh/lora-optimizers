"""Render walltime_profile.md table from baseline + flash_compile JSONL outputs.

Reads:
  logs/bench_profile_walltime/sweep_baseline_{1B,3B,8B}.jsonl
  logs/bench_profile_walltime/sweep_flash_compile_{1B,3B,8B}.jsonl  (optional)

Writes a Markdown report to docs/notes/polar_product/walltime_profile.md.
For each (base, r) cell reports per-step ms breakdown (fwd/bwd/opt/zero), peak
memory, and — when both conditions exist — the after/before speedup.

Usage:
  python scripts/analysis/profile_walltime_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "bench_profile_walltime"
OUT = ROOT / "docs" / "notes" / "polar_product" / "walltime_profile.md"

TIERS = ["1B", "3B", "8B"]
CONDITIONS = ["baseline", "flash_compile"]


def load(cond: str, tier: str) -> list[dict]:
    p = LOGS / f"sweep_{cond}_{tier}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def cell_key(rec: dict) -> tuple:
    return (rec["model_name"], rec["lora_r"], rec["optimizer"],
            rec.get("precond_method", "n/a"))


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}"


def render() -> str:
    lines: list[str] = []
    lines.append("# Wall-time profile across the 1B/3B/8B base ladder")
    lines.append("")
    lines.append("Baseline (HF default attn, no compile) vs after (flash_attention_2 + "
                 "torch.compile mode='default'). Single A100 (shared, not "
                 "exclusive — relative ordering preserved per profiling_a100_canonical_2026_05_04.md).")
    lines.append("")
    lines.append("Per-step times: forward + backward summed across grad_accum_steps "
                 "microbatches; optimizer.step() and zero_grad timed separately. All numbers in ms.")
    lines.append("")

    for tier in TIERS:
        base = load("baseline", tier)
        after = load("flash_compile", tier)
        if not base and not after:
            lines.append(f"## {tier}")
            lines.append("_(no data — sweep not yet run)_")
            lines.append("")
            continue

        # Group by cell key
        rows: dict[tuple, dict[str, dict]] = {}
        for rec in base:
            rows.setdefault(cell_key(rec), {})["baseline"] = rec
        for rec in after:
            rows.setdefault(cell_key(rec), {})["flash_compile"] = rec

        # First record gives us per-tier batch/seq context
        any_rec = (base or after)[0]
        lines.append(f"## {tier} — {any_rec['model_name']}, "
                     f"seq={any_rec.get('seq_len','?')}, "
                     f"batch={any_rec.get('batch_size','?')}×accum={any_rec.get('grad_accum_steps','?')}")
        lines.append("")
        lines.append("| r | optim | method | cond | fwd ms | bwd ms | opt ms | zero ms | total ms | peak MB | attn | compile |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        for key in sorted(rows.keys(), key=lambda k: (k[1], k[2], k[3])):
            _, r, opt, method = key
            for cond in CONDITIONS:
                rec = rows[key].get(cond)
                if not rec:
                    continue
                total = (rec["fwd_sec_per_step"] + rec["bwd_sec_per_step"]
                         + rec["opt_sec_per_step"] + rec["zero_sec_per_step"])
                lines.append(
                    f"| {r} | {opt} | {method} | {cond} | "
                    f"{fmt_ms(rec['fwd_sec_per_step'])} | "
                    f"{fmt_ms(rec['bwd_sec_per_step'])} | "
                    f"{fmt_ms(rec['opt_sec_per_step'])} | "
                    f"{fmt_ms(rec['zero_sec_per_step'])} | "
                    f"{fmt_ms(total)} | "
                    f"{rec['peak_memory_mb']:.0f} | "
                    f"{rec.get('attn_implementation', '?')} | "
                    f"{rec.get('compile_mode') or 'eager'} |"
                )
        lines.append("")

        # Speedup summary if both conds present
        speedups = []
        for key, conds in rows.items():
            if "baseline" in conds and "flash_compile" in conds:
                b = conds["baseline"]
                a = conds["flash_compile"]
                bt = (b["fwd_sec_per_step"] + b["bwd_sec_per_step"]
                      + b["opt_sec_per_step"] + b["zero_sec_per_step"])
                at = (a["fwd_sec_per_step"] + a["bwd_sec_per_step"]
                      + a["opt_sec_per_step"] + a["zero_sec_per_step"])
                speedups.append((key, bt / at, b["peak_memory_mb"], a["peak_memory_mb"]))
        if speedups:
            lines.append("**Speedup (after / before):**")
            lines.append("")
            lines.append("| r | optim | method | speedup | peak MB before | peak MB after |")
            lines.append("|---|---|---|---:|---:|---:|")
            for (_, r, opt, method), spd, pb, pa in sorted(speedups,
                                                            key=lambda x: (x[0][1], x[0][2])):
                lines.append(f"| {r} | {opt} | {method} | {spd:.2f}× | {pb:.0f} | {pa:.0f} |")
            lines.append("")

    return "\n".join(lines)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
