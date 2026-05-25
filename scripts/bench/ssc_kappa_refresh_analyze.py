"""Analyze the refresh-every-N bench logs: extract c trajectories and
quantify drift implications.

For each refresh log:
- pull eval_loss / tok_per_s / per-step wall
- pull per-step ssc_c_{A,B}_{median,min,max} from optim_step events
- compute, from the N=1 trace (treated as ground truth):
    rolling drift Δc/c over windows of length 10 and 50
- print summary table and per-step drift histogram.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / 'logs/bench_ssc_drift'


def parse_log(path: Path):
    cfg = None
    evals = []
    ssc = []  # per-step (step, c_A_median, c_A_min, c_A_max, c_B_median, c_B_min, c_B_max)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ev = d.get('event')
            if ev == 'config':
                cfg = d
            elif ev == 'eval':
                evals.append(d)
            elif ev == 'optim_step':
                if 'ssc_c_A_median' in d:
                    ssc.append((
                        int(d['step']),
                        d['ssc_c_A_median'], d.get('ssc_c_A_min'), d.get('ssc_c_A_max'),
                        d['ssc_c_B_median'], d.get('ssc_c_B_min'), d.get('ssc_c_B_max'),
                    ))
    return cfg, evals, ssc


def rolling_drift(c: np.ndarray, window: int) -> np.ndarray:
    """For each step t ≥ window, |c[t] - c[t-window]| / c[t-window]."""
    if len(c) <= window:
        return np.array([])
    return np.abs(c[window:] - c[:-window]) / np.maximum(c[:-window], 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', type=str, default=str(LOG_DIR))
    ap.add_argument('--rank', type=int, default=256)
    args = ap.parse_args()

    runs = {}  # N -> (cfg, evals, ssc)
    for N in (1, 10, 50):
        p = Path(args.logdir) / f'refresh_r{args.rank}_N{N}.log'
        if not p.exists():
            print(f"missing: {p}")
            continue
        runs[N] = parse_log(p)

    print(f"\n# r={args.rank} bench summary\n")
    print(f"{'N':>4} {'tok/s':>8} {'wall':>8} {'eval_loss':>10} {'Δ vs N=1':>10} {'n_optim_evts':>12}")
    base_loss = None
    for N, (cfg, evals, ssc) in runs.items():
        if not evals:
            print(f"{N:>4}  no eval rows")
            continue
        ev = evals[-1]
        loss = ev['eval_loss']
        if N == 1:
            base_loss = loss
        delta = (loss - base_loss) if base_loss is not None else float('nan')
        print(f"{N:>4} {ev['tokens_per_sec']:>8.0f} "
              f"{ev['train_elapsed_sec']:>7.1f}s "
              f"{loss:>10.5f} {delta:>+10.5f} {len(ssc):>12}")

    # Drift analysis on N=1 trace (ground truth, every-step c values).
    if 1 in runs and runs[1][2]:
        ssc = runs[1][2]
        steps = np.array([r[0] for r in ssc])
        cA_med = np.array([r[1] for r in ssc])
        cA_min = np.array([r[2] for r in ssc])
        cA_max = np.array([r[3] for r in ssc])
        cB_med = np.array([r[4] for r in ssc])
        cB_min = np.array([r[5] for r in ssc])
        cB_max = np.array([r[6] for r in ssc])

        print(f"\n# c trajectory from N=1 (ground-truth per-step c, {len(steps)} steps)\n")
        print(f"{'side':>6} {'agg':>6}  first  last  min    max    range")
        for label, c in [('A', cA_med), ('A_min', cA_min), ('A_max', cA_max),
                         ('B', cB_med), ('B_min', cB_min), ('B_max', cB_max)]:
            print(f"  {label:>4}  {c[0]:7.4f} {c[-1]:7.4f} {c.min():7.4f} {c.max():7.4f} {c.max()-c.min():7.4f}")

        print(f"\n# rolling drift Δc/c over windows (the 'what cached c misses' if you refresh every W')\n")
        print(f"{'window':>6}  {'agg':>6} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}")
        for w in (1, 5, 10, 25, 50):
            for label, c in [('A_med', cA_med), ('A_min', cA_min), ('A_max', cA_max),
                             ('B_med', cB_med), ('B_min', cB_min), ('B_max', cB_max)]:
                d = rolling_drift(c, w)
                if len(d) == 0:
                    continue
                print(f"  {w:>4}    {label:>6} "
                      f"{np.median(d):>7.4f} {np.percentile(d, 90):>7.4f} "
                      f"{np.percentile(d, 99):>7.4f} {d.max():>7.4f}")

    # Direct comparison: cached-c trajectory in N=10 vs N=50 runs against ground-truth N=1.
    # The cached-c trace is what the optimizer actually used; the N=1 trace is what it
    # WOULD have used if refreshing every step. Plotting both side-by-side shows how
    # much the cache deviates from optimal.
    if 1 in runs:
        ssc1 = runs[1][2]
        steps1 = np.array([r[0] for r in ssc1])
        cA1 = np.array([r[1] for r in ssc1])
        for N in (10, 50):
            if N not in runs:
                continue
            ssc_n = runs[N][2]
            if not ssc_n:
                continue
            steps_n = np.array([r[0] for r in ssc_n])
            cA_n = np.array([r[1] for r in ssc_n])
            # Align by step.
            common = np.intersect1d(steps1, steps_n)
            i1 = np.isin(steps1, common)
            iN = np.isin(steps_n, common)
            diff = np.abs(cA_n[iN] - cA1[i1]) / np.maximum(cA1[i1], 1e-30)
            print(f"\n# cached-vs-true c diff at N={N}  (side A median, across {len(common)} aligned steps)")
            print(f"  p50={np.median(diff):.4f}  p90={np.percentile(diff, 90):.4f}  "
                  f"p99={np.percentile(diff, 99):.4f}  max={diff.max():.4f}")


if __name__ == '__main__':
    main()
