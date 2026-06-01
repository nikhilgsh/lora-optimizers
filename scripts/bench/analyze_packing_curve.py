"""Summarize the timing/packing benchmark outputs into the tables that go in
docs/notes/timing_benchmarks_r256_blackwell.md.

Reads (whatever exists under --bench-dir):
  - intrinsic_r256_blackwell.jsonl  -> intrinsic optimizer cost table (Part 1)
  - packing/pack_N*_gpu*.jsonl      -> packing curve penalty(N) (Part 4)
  - packing/telemetry_N*.csv        -> clocks/power mean per N (mechanism)

No GPU, no torch — pure stdlib parsing of the bench JSONL + CSV. Prints to stdout;
copy the tables into the notes doc with provenance.

Usage:
    python scripts/bench/analyze_packing_curve.py [--bench-dir logs/bench]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict


def _load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def intrinsic_table(bench_dir):
    path = os.path.join(bench_dir, "intrinsic_r256_blackwell.jsonl")
    if not os.path.exists(path):
        print(f"[intrinsic] {path} not found — skip\n")
        return
    rows = [r for r in _load_jsonl(path) if r.get("event") == "bench_step"]
    if not rows:
        print(f"[intrinsic] no bench_step rows in {path}\n")
        return
    print(f"## Intrinsic optimizer cost (isolated)  [{path}]")
    print(f"{'optimizer':<52}{'K':>4}{'opt_ms':>9}{'total_ms':>10}"
          f"{'xAdamW':>9}{'refresh_ms':>12}{'stale_ms':>10}")
    for r in sorted(rows, key=lambda r: (r.get("optimizer", ""), r.get("precond_refresh_every", 0))):
        print(f"{r.get('optimizer',''):<52}{r.get('precond_refresh_every',''):>4}"
              f"{r.get('opt_sec_per_step',0)*1000:>9.1f}"
              f"{r.get('mean_sec_per_step',0)*1000:>10.1f}"
              f"{r.get('ratio_vs_adamw_step_only',float('nan')):>8.2f}x"
              f"{r.get('refresh_sec_per_step',float('nan'))*1000:>12.1f}"
              f"{r.get('stale_sec_per_step',float('nan'))*1000:>10.1f}")
    print()


def packing_table(bench_dir):
    pat = os.path.join(bench_dir, "packing", "pack_N*_gpu*.jsonl")
    files = glob.glob(pat)
    if not files:
        print(f"[packing] no files matching {pat} — skip\n")
        return
    # group mean_sec_per_step by (opt, N) across GPUs
    by = defaultdict(list)  # (opt, N) -> [s/step per gpu]
    rx = re.compile(r"pack_N(\d+)(cross)?_(?:gpu\d+|.*?gpu\d+)")
    for f in files:
        base = os.path.basename(f)
        m = re.search(r"pack_N(\d+)(cross)?_", base)
        if not m:
            continue
        N = int(m.group(1)); cross = bool(m.group(2))
        for r in _load_jsonl(f):
            if r.get("event") != "bench_step":
                continue
            opt = r.get("optimizer", "?")
            key = (opt, f"{N}{'x' if cross else ''}")
            by[key].append(r.get("mean_sec_per_step", float("nan")))
    print("## Packing curve  [logs/bench/packing/pack_N*_gpu*.jsonl]")
    print(f"{'optimizer':<34}{'N':>5}{'n_gpu':>6}{'mean_s/step':>13}{'penalty(N)':>12}")
    # baseline N=1 per optimizer
    base1 = {}
    for (opt, N), vals in by.items():
        if N == "1":
            base1[opt] = st.mean(vals)
    for (opt, N) in sorted(by, key=lambda k: (k[0], int(re.sub(r'\D', '', k[1])), 'x' in k[1])):
        vals = by[(opt, N)]
        mean = st.mean(vals)
        pen = (mean / base1[opt]) if base1.get(opt) else float("nan")
        print(f"{opt:<34}{N:>5}{len(vals):>6}{mean:>13.3f}{pen:>11.2f}x")
    print()


def telemetry_summary(bench_dir):
    pat = os.path.join(bench_dir, "packing", "telemetry_N*.csv")
    files = glob.glob(pat)
    if not files:
        print(f"[telemetry] no files matching {pat} — skip\n")
        return
    print("## Telemetry (mechanism: throttling vs bandwidth)  [logs/bench/packing/telemetry_N*.csv]")
    print(f"{'file':<40}{'mean_sm_mhz':>13}{'mean_power_w':>14}{'max_temp_c':>12}")
    for f in sorted(files):
        sm, pw, tp = [], [], []
        try:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    try:
                        sm.append(float(row["clocks_sm_mhz"]))
                        pw.append(float(row["power_w"]))
                        tp.append(float(row["temp_c"]))
                    except (KeyError, ValueError):
                        continue
        except OSError:
            continue
        if not sm:
            continue
        print(f"{os.path.basename(f):<40}{st.mean(sm):>13.0f}{st.mean(pw):>14.0f}{max(tp):>12.0f}")
    print("\n# Read: if mean_sm_mhz drops and power/temp pin at limits as N rises -> throttling;")
    print("# if clocks hold but packing penalty still >1 -> RAM/PCIe bandwidth.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="logs/bench")
    args = ap.parse_args()
    intrinsic_table(args.bench_dir)
    packing_table(args.bench_dir)
    telemetry_summary(args.bench_dir)


if __name__ == "__main__":
    main()
