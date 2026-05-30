#!/usr/bin/env python
"""Compare finite NS to stronger relative damping on stored snapshots.

This is a bounded offline probe. It asks whether the A-side direction

    W_abs NS_K(W_abs u_A)

is close to

    W_rel polar(W_rel u_A)

for relative damping choices W_rel = (B^T B + eps_eff I)^(-1/2).
If yes, "fewer NS iters" can be partly interpreted as acting like a more
conservative damping policy on the same weak B directions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lora_playground.optim import _newton_schulz
from lora_playground.snapshot_analysis.snapshots import RUNS, SNAP_ROOTS, STEPS_BY_ROOT, load_snapshot
from lora_playground.snapshot_analysis.whitening import DELTA_ABS


REL_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _eigh_desc(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigs, vecs = torch.linalg.eigh(0.5 * (S.float() + S.float().T))
    order = torch.argsort(eigs, descending=True)
    return eigs[order].clamp_min(0.0), vecs[:, order]


def _invsqrt(eigs: torch.Tensor, vecs: torch.Tensor, delta: float) -> torch.Tensor:
    return vecs @ torch.diag((eigs + float(delta)).clamp_min(1e-30).rsqrt()) @ vecs.T


def _polar(X: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    return U @ Vh


def _cos(A: torch.Tensor, B: torch.Tensor) -> float:
    Af = A.float().flatten()
    Bf = B.float().flatten()
    return float((Af @ Bf) / (Af.norm() * Bf.norm()).clamp_min(1e-30))


def _tail25_mass_in_B_basis(G: torch.Tensor, vecB: torch.Tensor) -> float:
    r = vecB.shape[0]
    n_tail = max(1, r // 4)
    Gb = vecB.T @ G.float()
    energy = Gb.square().sum(dim=1)
    return float(energy[-n_tail:].sum() / energy.sum().clamp_min(1e-30))


def _through_B(B: torch.Tensor, G: torch.Tensor) -> float:
    Gf = G.float()
    return float((B.float() @ Gf).norm() / Gf.norm().clamp_min(1e-30))


def pair_rows(pair: dict, run_key: tuple, step: int, pair_index: int, ns_steps: tuple[int, ...]) -> list[dict]:
    B = pair["B"].float()
    u_A = pair["u_A"].float()
    S_B = B.T @ B
    eigB, vecB = _eigh_desc(S_B)
    r = int(eigB.numel())
    lam_max = float(eigB[0])
    lam_mean = float(eigB.mean())
    W_abs = _invsqrt(eigB, vecB, DELTA_ABS)

    rows = []
    finite = {}
    X_abs = W_abs @ u_A
    for k in ns_steps:
        P_k = _newton_schulz(X_abs, nsteps=k)
        G_k = W_abs @ P_k
        finite[k] = G_k
        rows.append({
            "lr": run_key[0],
            "lora_r": run_key[1],
            "snapshot_ns": run_key[2],
            "variant": run_key[3],
            "step": step,
            "pair": pair_index,
            "ns_steps": k,
            "damping_type": "finite_ns_abs",
            "rel": None,
            "delta_eff": DELTA_ABS,
            "cos_to_finite_ns": 1.0,
            "tail25_mass": _tail25_mass_in_B_basis(G_k, vecB),
            "through_B": _through_B(B, G_k),
            "stable_rank_B_frac": float(eigB.sum() / eigB[0].clamp_min(1e-30)) / r,
            "cond_SB": float(eigB[0] / eigB[-1].clamp_min(1e-30)),
        })

    for damping_type, scale in (("op", lam_max), ("trace", lam_mean)):
        for rel in REL_GRID:
            delta = DELTA_ABS if rel == 0.0 else float(rel) * scale
            W = _invsqrt(eigB, vecB, delta)
            G = W @ _polar(W @ u_A)
            for k, G_k in finite.items():
                rows.append({
                    "lr": run_key[0],
                    "lora_r": run_key[1],
                    "snapshot_ns": run_key[2],
                    "variant": run_key[3],
                    "step": step,
                    "pair": pair_index,
                    "ns_steps": k,
                    "damping_type": damping_type,
                    "rel": rel,
                    "delta_eff": delta,
                    "cos_to_finite_ns": _cos(G, G_k),
                    "tail25_mass": _tail25_mass_in_B_basis(G, vecB),
                    "through_B": _through_B(B, G),
                    "stable_rank_B_frac": float(eigB.sum() / eigB[0].clamp_min(1e-30)) / r,
                    "cond_SB": float(eigB[0] / eigB[-1].clamp_min(1e-30)),
                })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["lr", "lora_r", "snapshot_ns", "variant", "step", "ns_steps", "damping_type", "rel"]
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    fields = [
        "cos_to_finite_ns",
        "tail25_mass",
        "through_B",
        "stable_rank_B_frac",
        "cond_SB",
    ]
    out = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(keys, key))
        row["n_pairs"] = len(bucket)
        for field in fields:
            vals = [float(b[field]) for b in bucket if b.get(field) is not None]
            row[field + "_median"] = float(np.median(vals)) if vals else None
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--steps", default="4000")
    ap.add_argument("--max-pairs", type=int, default=4)
    ap.add_argument("--ns-steps", default="2,3,5")
    args = ap.parse_args()

    wanted_steps = {int(s) for s in args.steps.split(",") if s}
    ns_steps = tuple(int(s) for s in args.ns_steps.split(",") if s)
    rows = []
    for run_key in RUNS:
        root = SNAP_ROOTS[run_key]
        steps = [s for s in STEPS_BY_ROOT.get(run_key, []) if s in wanted_steps]
        for step in steps:
            snap = load_snapshot(step, root)
            n_total = len(snap["pair_state"])
            pair_indices = np.linspace(0, n_total - 1, min(args.max_pairs, n_total), dtype=int)
            for pi in pair_indices:
                rows.extend(pair_rows(snap["pair_state"][int(pi)], run_key, step, int(pi), ns_steps))

    summary = summarize(rows)
    write_csv(Path(args.out), rows)
    write_csv(Path(args.summary_out), summary)

    print(f"rows={len(rows)} summary_rows={len(summary)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")

    # Print best nonzero rel per run/K/type by cosine to finite-NS.
    grouped: dict[tuple, list[dict]] = {}
    for row in summary:
        if row["damping_type"] not in ("op", "trace") or not row["rel"]:
            continue
        key = (row["lr"], row["lora_r"], row["variant"], row["step"], row["ns_steps"], row["damping_type"])
        grouped.setdefault(key, []).append(row)
    for key, vals in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        best = max(vals, key=lambda r: r["cos_to_finite_ns_median"])
        print(
            " ".join([
                f"run=({key[0]},r{key[1]},{key[2]})",
                f"step={key[3]}",
                f"K={key[4]}",
                f"type={key[5]}",
                f"best_rel={best['rel']}",
                f"cos={best['cos_to_finite_ns_median']:.3f}",
                f"tail25={best['tail25_mass_median']:.3f}",
                f"through={best['through_B_median']:.3f}",
                f"srB/r={best['stable_rank_B_frac_median']:.3f}",
                f"condSB={best['cond_SB_median']:.1f}",
            ])
        )


if __name__ == "__main__":
    main()
