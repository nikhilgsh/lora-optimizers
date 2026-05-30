#!/usr/bin/env python
"""Probe LoRA factor tail amplification in stored optimizer snapshots.

For side A, the relevant small-side metric is S_B = B^T B because the
whitened NS input is S_B^{-1/2} u_A. This script decomposes u_A in the
eigenbasis of S_B and measures how much energy is in B's low-singular
directions before and after whitening. Side B is analogous with S_A = A A^T.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lora_playground.snapshot_analysis.snapshots import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    load_snapshot,
)
from lora_playground.snapshot_analysis.whitening import DELTA_ABS


def _stable_rank_from_eigs(eigs: torch.Tensor) -> float:
    eigs = eigs.clamp_min(0.0)
    return float(eigs.sum() / eigs.max().clamp_min(1e-30))


def _eig_desc(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigs, vecs = torch.linalg.eigh(S.float())
    order = torch.argsort(eigs, descending=True)
    eigs = eigs[order].clamp_min(0.0)
    vecs = vecs[:, order]
    return eigs, vecs


def _tail_fractions(energy: torch.Tensor, eigs: torch.Tensor, frac: float) -> dict:
    r = int(eigs.numel())
    n_tail = max(1, int(round(frac * r)))
    tail = torch.arange(r - n_tail, r)
    raw = energy.clamp_min(0.0)
    white = raw / (eigs + DELTA_ABS)
    raw_total = raw.sum().clamp_min(1e-30)
    white_total = white.sum().clamp_min(1e-30)
    raw_tail = float(raw[tail].sum() / raw_total)
    white_tail = float(white[tail].sum() / white_total)
    return {
        f"raw_tail{int(frac * 100)}": raw_tail,
        f"white_tail{int(frac * 100)}": white_tail,
        f"tail{int(frac * 100)}_gain": white_tail / max(raw_tail, 1e-30),
        f"s_tail{int(frac * 100)}_max_rel": float(
            eigs[tail].sqrt().max() / eigs[0].sqrt().clamp_min(1e-30)
        ),
    }


def pair_metrics(pair: dict, run_key: tuple, step: int, pair_index: int) -> dict:
    A = pair["A"].float()
    B = pair["B"].float()
    u_A = pair["u_A"].float()
    u_B = pair["u_B"].float()

    eigA, vecA = _eig_desc(A @ A.T)
    eigB, vecB = _eig_desc(B.T @ B)
    r = int(A.shape[0])

    # Side A: row/rank coordinates of u_A in eigenbasis of B^T B.
    uA_basis = vecB.T @ u_A
    uA_energy = uA_basis.square().sum(dim=1)
    xA_basis = uA_basis / (eigB + DELTA_ABS).sqrt().unsqueeze(1)

    # Side B: column/rank coordinates of u_B in eigenbasis of A A^T.
    uB_basis = u_B @ vecA
    uB_energy = uB_basis.square().sum(dim=0)
    xB_basis = uB_basis / (eigA + DELTA_ABS).sqrt().unsqueeze(0)

    sv_xA = torch.linalg.svdvals(xA_basis)
    sv_xB = torch.linalg.svdvals(xB_basis)

    row = {
        "lr": run_key[0],
        "lora_r": run_key[1],
        "ns": run_key[2],
        "variant": run_key[3],
        "step": step,
        "pair": pair_index,
        "shape": f"{tuple(A.shape)}|{tuple(B.shape)}",
        "stable_rank_A_frac": _stable_rank_from_eigs(eigA) / r,
        "stable_rank_B_frac": _stable_rank_from_eigs(eigB) / r,
        "cond_SA": float(eigA[0] / eigA[-1].clamp_min(1e-30)),
        "cond_SB": float(eigB[0] / eigB[-1].clamp_min(1e-30)),
        "smin_A_rel": float(eigA[-1].sqrt() / eigA[0].sqrt().clamp_min(1e-30)),
        "smin_B_rel": float(eigB[-1].sqrt() / eigB[0].sqrt().clamp_min(1e-30)),
        "amp_A_frob": float(xA_basis.norm() / u_A.norm().clamp_min(1e-30)),
        "amp_B_frob": float(xB_basis.norm() / u_B.norm().clamp_min(1e-30)),
        "xunc_A_stable_rank_frac": float(
            sv_xA.square().sum() / sv_xA.square().max().clamp_min(1e-30)
        ) / r,
        "xunc_B_stable_rank_frac": float(
            sv_xB.square().sum() / sv_xB.square().max().clamp_min(1e-30)
        ) / r,
    }
    for frac in (0.25, 0.50):
        for k, v in _tail_fractions(uA_energy, eigB, frac).items():
            row[f"A_in_B_{k}"] = v
        for k, v in _tail_fractions(uB_energy, eigA, frac).items():
            row[f"B_in_A_{k}"] = v
    return row


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["lr", "lora_r", "ns", "variant", "step"]
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(row)
    fields = [
        "stable_rank_A_frac",
        "stable_rank_B_frac",
        "cond_SA",
        "cond_SB",
        "amp_A_frob",
        "amp_B_frob",
        "xunc_A_stable_rank_frac",
        "xunc_B_stable_rank_frac",
        "A_in_B_raw_tail25",
        "A_in_B_white_tail25",
        "A_in_B_tail25_gain",
        "B_in_A_raw_tail25",
        "B_in_A_white_tail25",
        "B_in_A_tail25_gain",
        "A_in_B_raw_tail50",
        "A_in_B_white_tail50",
        "B_in_A_raw_tail50",
        "B_in_A_white_tail50",
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
    ap.add_argument("--steps", default="200,500,1000,4000,9000")
    ap.add_argument("--max-pairs", type=int, default=24)
    args = ap.parse_args()

    wanted_steps = {int(s) for s in args.steps.split(",") if s}
    rows = []
    for run_key in RUNS:
        root = SNAP_ROOTS[run_key]
        steps = [s for s in STEPS_BY_ROOT.get(run_key, []) if s in wanted_steps]
        for step in steps:
            snap = load_snapshot(step, root)
            n_total = len(snap["pair_state"])
            pair_indices = np.linspace(0, n_total - 1, min(args.max_pairs, n_total), dtype=int)
            for pi in pair_indices:
                rows.append(pair_metrics(snap["pair_state"][int(pi)], run_key, step, int(pi)))

    summary = summarize(rows)
    write_csv(Path(args.out), rows)
    write_csv(Path(args.summary_out), summary)

    print(f"snapshot_rows={len(rows)} summary_rows={len(summary)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    for row in summary:
        print(
            " ".join(
                [
                    f"run=({row['lr']},r{row['lora_r']},ns{row['ns']},{row['variant']})",
                    f"step={row['step']}",
                    f"srB/r={row['stable_rank_B_frac_median']:.3f}",
                    f"condSB={row['cond_SB_median']:.1f}",
                    f"Araw25={row['A_in_B_raw_tail25_median']:.3f}",
                    f"Awht25={row['A_in_B_white_tail25_median']:.3f}",
                    f"gain25={row['A_in_B_tail25_gain_median']:.1f}",
                    f"xA/r={row['xunc_A_stable_rank_frac_median']:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
