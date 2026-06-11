"""Accuracy of the inverse-sqrt methods on REAL snapshot Grams — the accuracy
half of the QR-eigenbasis-vs-Higham retiming (HANDOFF §4: "Accuracy on REAL S_a
from snapshots, not synthetic SPD").

The protagonist's small-side Shampoo inverse-sqrt S^{-1/2} is the one quantity
that differs between the QR-eigenbasis path (a full eigh seed + orthogonal-
iteration refresh → tracks eigh) and the Higham/Newton-Schulz path (eigh-free,
fresh every step). The decision needs: at the *real conditioning* the factors
exhibit, how far is each from the eigh ground truth?

We load real (A, B) factors from the chord-tight snapshots, form the small-side
Grams S_A = A Aᵀ and S_B = Bᵀ B (r×r), and compare Higham(n_iters) to the eigh
truth via relative Frobenius error, the independent residual ‖Z S Z − I‖_F, and
the top-singular-value error — across the real condition-number distribution.

Consolidates ``scripts/bench/bench_higham_accuracy.py`` (synthetic) and
``scripts/bench/higham_convergence_snapshot.py`` (real snapshots).
"""
from __future__ import annotations

from statistics import median
from typing import Optional

import torch

from ..optim import _spd_inv_half
from ..snapshot_analysis.snapshots import SNAP_ROOTS, STEPS_BY_ROOT, load_snapshot
from ..utils import spd_inv_sqrt_higham


def load_real_grams(snapshot_key, *, steps=None, max_pairs: Optional[int] = None):
    """Yield (label, S) real SPD Grams from a snapshot run. ``snapshot_key`` is a
    key into ``SNAP_ROOTS`` (e.g. ('3e-2', 256, 5, 'chord-tight-k1') for r256 k1).
    Both sides: S_A = A Aᵀ (whitens B-side) and S_B = Bᵀ B (whitens A-side)."""
    root = SNAP_ROOTS.get(snapshot_key)
    if root is None or not root.exists():
        raise FileNotFoundError(f"snapshot root for {snapshot_key} not present: {root}")
    use_steps = steps if steps is not None else [s for s in STEPS_BY_ROOT[snapshot_key] if s > 0]
    for step in use_steps:
        snap = load_snapshot(step, root=root)
        for pi, p in snap["pair_state"].items():
            if not all(k in p for k in ("A", "B")):
                continue
            if max_pairs is not None and pi >= max_pairs:
                continue
            A, B = p["A"].float(), p["B"].float()
            yield (f"step{step}/pair{pi}/A", A @ A.transpose(-2, -1))
            yield (f"step{step}/pair{pi}/B", B.transpose(-2, -1) @ B)


def accuracy_vs_eigh(S, n_iters_list=(5, 8, 10, 16), eps=1e-4):
    """For one SPD S, compare Higham(n_iters) to the eigh ground truth (S+δI)^{-1/2}.
    Returns dict: cond, and per-n_iters (rel_err_F, residual_F, rel_err_top)."""
    r = S.shape[-1]
    eye = torch.eye(r, dtype=S.dtype, device=S.device)
    S_damped = S + eps * eye
    eigs = torch.linalg.eigvalsh(S).clamp_min(0)
    cond = float(eigs.max() / eigs.min().clamp_min(1e-30))
    X_eigh = _spd_inv_half(S, eps=eps, method="eigh")
    sig_eigh = float(torch.linalg.eigvalsh(X_eigh).max())
    out = {"cond": cond, "by_iters": {}}
    for n in n_iters_list:
        X = spd_inv_sqrt_higham(S, n_iters=n, eps=eps)
        rel_err_F = float((X - X_eigh).norm() / X_eigh.norm().clamp_min(1e-30))
        residual_F = float((X @ S_damped @ X - eye).norm())
        sig = float(torch.linalg.eigvalsh(X).max())
        rel_err_top = abs(sig - sig_eigh) / max(sig_eigh, 1e-30)
        out["by_iters"][n] = {"rel_err_F": rel_err_F, "residual_F": residual_F,
                              "rel_err_top": rel_err_top}
    return out


def run_inverse_sqrt_accuracy(snapshot_key, *, n_iters_list=(5, 8, 10, 16),
                              eps=1e-4, steps=None, max_pairs=None) -> dict:
    """Aggregate Higham-vs-eigh accuracy over a snapshot run's real Grams.
    Prints a p50/p99/max table per n_iters and returns the raw aggregates."""
    conds, by_iters = [], {n: {"rel_err_F": [], "residual_F": [], "rel_err_top": []}
                           for n in n_iters_list}
    n_grams = 0
    for _label, S in load_real_grams(snapshot_key, steps=steps, max_pairs=max_pairs):
        a = accuracy_vs_eigh(S, n_iters_list=n_iters_list, eps=eps)
        conds.append(a["cond"]); n_grams += 1
        for n in n_iters_list:
            for m in ("rel_err_F", "residual_F", "rel_err_top"):
                by_iters[n][m].append(a["by_iters"][n][m])

    def pct(xs, q):
        xs = sorted(xs); i = min(len(xs) - 1, int(q * (len(xs) - 1)))
        return xs[i] if xs else float("nan")

    print(f"# inverse-sqrt accuracy vs eigh  snapshot={snapshot_key}  "
          f"eps={eps}  n_grams={n_grams}", flush=True)
    print(f"#   cond(S): p50={median(conds):.2e}  p99={pct(conds,0.99):.2e}  "
          f"max={max(conds):.2e}", flush=True)
    print(f"#   {'n_iters':>7} {'rel_err_F(p50/p99)':>26} {'residual_F(p50/p99)':>26} "
          f"{'rel_err_top(p99)':>16}", flush=True)
    for n in n_iters_list:
        d = by_iters[n]
        print(f"#   {n:>7} "
              f"{median(d['rel_err_F']):>11.2e}/{pct(d['rel_err_F'],0.99):>11.2e} "
              f"{median(d['residual_F']):>11.2e}/{pct(d['residual_F'],0.99):>11.2e} "
              f"{pct(d['rel_err_top'],0.99):>16.2e}", flush=True)
    return {"snapshot_key": snapshot_key, "eps": eps, "n_grams": n_grams,
            "conds": conds, "by_iters": by_iters}
