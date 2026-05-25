"""Measure kpar K=3 c-accuracy with a STALE warm start (= R=5 scenario).

At refresh=5, cached c is up to 5 steps old. We simulate that by using c
solved at snapshot step S-5 as c_init for kpar at snapshot step S, then
comparing kpar's solved c against the true c (eigvalsh) at step S.

If the cached c has drifted more than log_window/(K-1), the kpar argmin
picks a bracket-boundary c → wrong. Reports per-pair log-c-error.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

from lora_playground.optim import (
    _ssc_adaptive_kappa_batched, _ssc_misr_bisect_batched_kpar,
)
from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.utils import spd_frac_power_inv


def _spd_half_inv_loop(S, eps=1e-6):
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _whitened_X_groups(snap):
    groups = defaultdict(list)
    for pi, p in snap['pair_state'].items():
        if not all(k in p for k in ('A', 'B', 'u_A', 'u_B')):
            continue
        A, B = p['A'].float(), p['B'].float()
        u_A, u_B = p['u_A'].float(), p['u_B'].float()
        key = (A.shape[0], A.shape[1], B.shape[0])
        groups[key].append((pi, A, B, u_A, u_B))
    out = {'A': defaultdict(list), 'B': defaultdict(list)}
    for shape_key, group in groups.items():
        A = torch.stack([g[1] for g in group])
        B = torch.stack([g[2] for g in group])
        u_A = torch.stack([g[3] for g in group])
        u_B = torch.stack([g[4] for g in group])
        pis = [g[0] for g in group]
        SA = A @ A.transpose(-2, -1)
        SB = B.transpose(-2, -1) @ B
        W_for_A = _spd_half_inv_loop(SB)
        W_for_B = _spd_half_inv_loop(SA)
        X_A = W_for_A @ u_A
        X_B = u_B @ W_for_B
        for side, X in (('A', X_A), ('B', X_B)):
            sigma = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X])
            X = X / sigma.view(-1, *([1] * (X.dim() - 1)))
            out[side][shape_key] = (pis, X)
    return out


def _solve_true_c_by_pair(groups_dict, kappa, nsteps=20):
    """Returns {pi: c_true} for all pairs in the snapshot."""
    out = {}
    for shape_key, (pis, X) in groups_dict.items():
        _, c_true = _ssc_adaptive_kappa_batched(X, kappa=kappa, nsteps=nsteps)
        for pi, c in zip(pis, c_true.tolist()):
            out[pi] = c
    return out


def diagnose(run_key, label, K=3, log_window=0.5, kappa=0.6, nsteps=20):
    if SNAP_ROOTS.get(run_key) is None or not SNAP_ROOTS[run_key].exists():
        print(f"\n[{label}] SKIP — snapshot files not present")
        return
    steps_avail = sorted(s for s in STEPS_BY_ROOT[run_key] if s > 0)
    if len(steps_avail) < 2:
        print(f"\n[{label}] SKIP — need >=2 snapshot steps")
        return
    print(f"\n[{label}] kpar K={K}  log_window={log_window}  nsteps={nsteps}  "
          f"snapshot steps={steps_avail}")

    # Snapshots have non-uniform spacing; for each (prev, cur) pair we use
    # `prev`'s solved c as c_init for kpar at `cur` step. The gap (cur-prev)
    # is the simulated refresh interval.
    for i in range(1, len(steps_avail)):
        prev, cur = steps_avail[i - 1], steps_avail[i]
        gap = cur - prev
        snap_prev = load_snapshot(prev, root=SNAP_ROOTS[run_key])
        snap_cur = load_snapshot(cur, root=SNAP_ROOTS[run_key])
        grp_prev = _whitened_X_groups(snap_prev)
        grp_cur = _whitened_X_groups(snap_cur)

        c_true_prev = {
            'A': _solve_true_c_by_pair(grp_prev['A'], kappa, nsteps),
            'B': _solve_true_c_by_pair(grp_prev['B'], kappa, nsteps),
        }

        boundary_count = 0
        total_pairs = 0
        log_errs = []
        drift_logs = []
        for side in ('A', 'B'):
            for shape_key, (pis_cur, X_cur) in grp_cur[side].items():
                # c_init = prev-step's solved c (the "stale cache" we'd reuse).
                # Skip pairs that appear only at cur (none in this dataset, but defensive).
                c_init_list = []
                keep_idx = []
                for j, pi in enumerate(pis_cur):
                    if pi in c_true_prev[side]:
                        c_init_list.append(c_true_prev[side][pi])
                        keep_idx.append(j)
                if not c_init_list:
                    continue
                X_sub = X_cur[keep_idx]
                c_init = torch.tensor(c_init_list)
                _, c_par = _ssc_misr_bisect_batched_kpar(
                    X_sub, kappa=kappa, K=K, nsteps=nsteps,
                    c_init=c_init, log_window=log_window,
                )
                _, c_true_cur = _ssc_adaptive_kappa_batched(
                    X_sub, kappa=kappa, nsteps=nsteps,
                )
                # Drift of the cache itself (log-c distance prev -> cur).
                drift = (torch.tensor(c_init_list).log() - c_true_cur.log()).abs().tolist()
                drift_logs.extend(drift)
                # kpar error vs true.
                err = (c_par.log() - c_true_cur.log()).abs().tolist()
                log_errs.extend(err)
                # How many landed on a bracket boundary?
                #   The bracket is c_init * exp([-log_window, +log_window]).
                #   "Boundary" = |log(c_par/c_init)| > log_window - eps.
                edge_eps = 1e-3
                boundary = ((c_par.log() - torch.tensor(c_init_list).log()).abs()
                            > log_window - edge_eps)
                boundary_count += int(boundary.sum().item())
                total_pairs += c_par.numel()

        if not log_errs:
            continue
        log_errs = np.array(log_errs)
        drift_arr = np.array(drift_logs)
        print(f"  gap={gap:>3} steps  N={total_pairs:>4}  "
              f"cache_drift_log  p50={float(np.median(drift_arr)):.3f}  "
              f"p99={float(np.percentile(drift_arr, 99)):.3f}  "
              f"max={float(drift_arr.max()):.3f}")
        print(f"                              kpar_err_log     "
              f"p50={float(np.median(log_errs)):.3f}  "
              f"p99={float(np.percentile(log_errs, 99)):.3f}  "
              f"max={float(log_errs.max()):.3f}  "
              f"boundary_winners={boundary_count}/{total_pairs}")


def main():
    torch.manual_seed(0)
    for K, log_window in [(3, 0.5), (9, 0.5), (3, 0.3)]:
        print(f"\n===== K={K}  log_window={log_window} =====")
        for run_key, label in [
            (RUN_A, "RUN_A r=64 lr=3e-2"),
            (RUN_C, "RUN_C r=64 lr=1e-1"),
            (RUN_B, "RUN_B r=256 lr=1e-1"),
            (RUN_D, "RUN_D r=256 lr=1e-3"),
        ]:
            diagnose(run_key, label, K=K, log_window=log_window, kappa=0.6, nsteps=20)


if __name__ == "__main__":
    main()
