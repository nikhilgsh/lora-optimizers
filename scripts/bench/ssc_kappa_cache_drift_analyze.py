"""Post-hoc c-drift analysis: from the per-step c trajectory of an N=1
run, compute the drift between the cached c (refreshed every N) and the
ground-truth c (refreshed every step) for various N and warmup-M.

Methodology beats running multiple cells: a single per-step c trace
contains all the info needed to simulate any refresh schedule.

Caveat: this assumes the cached-c trajectory doesn't diverge the model
state appreciably (within the regime where the cache is effective). End-
to-end eval_loss validation still needed at one chosen N.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_trace(path: Path):
    """Return list of (step, c_A_med, c_A_min, c_A_max, c_B_med, c_B_min, c_B_max)."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('event') != 'optim_step':
                continue
            if 'ssc_c_A_median' not in d:
                continue
            rows.append((
                int(d['step']),
                d['ssc_c_A_median'], d.get('ssc_c_A_min', np.nan), d.get('ssc_c_A_max', np.nan),
                d['ssc_c_B_median'], d.get('ssc_c_B_min', np.nan), d.get('ssc_c_B_max', np.nan),
            ))
    rows.sort()
    return rows


def simulate_cache(c_true: np.ndarray, N: int, M: int) -> np.ndarray:
    """For each step t (1-indexed), return the c that would be used under
    refresh-every-N with warmup-M.

    Steps 1..M: cached c = c_true (refresh every step in warmup).
    Steps M+1..: cached c = c_true[refresh_step] where refresh_step is
    the most recent step satisfying ((s-1) % N == 0).
    """
    out = np.empty_like(c_true)
    for t in range(len(c_true)):  # t is 0-indexed; step = t+1
        step = t + 1
        if step <= M:
            # Warmup: refresh every step.
            out[t] = c_true[t]
            continue
        # Post-warmup: last refresh is max(M, last cadence boundary ≤ step).
        cadence_refresh = ((step - 1) // N) * N + 1
        last_refresh_step = max(M, cadence_refresh)
        out[t] = c_true[last_refresh_step - 1]
    return out


def report_for_N(c_true, N, M, side_label):
    cached = simulate_cache(c_true, N, M)
    # Exclude saturated entries from drift (c==c_lo) since they don't reflect
    # real spectrum but the saturation floor.
    valid = c_true > 0.002  # c_lo=1e-3 with small buffer
    if not valid.any():
        return None
    rel = np.abs(cached[valid] - c_true[valid]) / np.maximum(c_true[valid], 1e-30)
    return {
        'N': N, 'side': side_label, 'n_valid': int(valid.sum()),
        'p50': float(np.median(rel)),
        'p90': float(np.percentile(rel, 90)),
        'p99': float(np.percentile(rel, 99)),
        'max': float(rel.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n1_log', type=str,
                    default=str(REPO / 'logs/bench_ssc_drift/refresh_r256_N1.log'),
                    help="Path to the N=1 per-step c trace log.")
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--Ns', type=int, nargs='+', default=[5, 10, 20, 50, 100])
    args = ap.parse_args()

    rows = load_trace(Path(args.n1_log))
    if not rows:
        print(f"no optim_step rows with ssc_c in {args.n1_log}")
        return
    print(f"\n# Loaded {len(rows)} steps from {args.n1_log}\n")

    steps = np.array([r[0] for r in rows])
    cA_med = np.array([r[1] for r in rows])
    cA_min = np.array([r[2] for r in rows])
    cA_max = np.array([r[3] for r in rows])
    cB_med = np.array([r[4] for r in rows])
    cB_min = np.array([r[5] for r in rows])
    cB_max = np.array([r[6] for r in rows])

    print(f"# c-trajectory (true, per-step) summary:")
    for label, c in [('cA_med', cA_med), ('cA_min', cA_min), ('cA_max', cA_max),
                     ('cB_med', cB_med), ('cB_min', cB_min), ('cB_max', cB_max)]:
        n_sat = int((c <= 0.002).sum())
        print(f"  {label:>7}: range [{c.min():.5f}, {c.max():.5f}]  saturated_steps={n_sat}")

    print(f"\n# Cached-vs-true drift, warmup M={args.warmup}, non-saturated steps only:\n")
    print(f"{'N':>4}  {'side':>7}  {'n':>5}  {'p50':>8}  {'p90':>8}  {'p99':>8}  {'max':>8}")
    for N in args.Ns:
        for label, c in [('A_med', cA_med), ('A_min', cA_min), ('A_max', cA_max),
                         ('B_med', cB_med), ('B_min', cB_min), ('B_max', cB_max)]:
            r = report_for_N(c, N, args.warmup, label)
            if r is None:
                continue
            print(f"{N:>4}  {r['side']:>7}  {r['n_valid']:>5}  "
                  f"{r['p50']:>8.4f}  {r['p90']:>8.4f}  {r['p99']:>8.4f}  {r['max']:>8.4f}")
        print()


if __name__ == '__main__':
    main()
