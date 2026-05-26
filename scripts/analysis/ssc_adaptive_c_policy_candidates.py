#!/usr/bin/env python
"""Offline diagnostics for cheap/stabilized SSC adaptive-c policies.

This script is intentionally snapshot-only: it compares candidate c-selection
policies against exact eigvalsh κ on saved optimizer states before any new
training run is launched.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_playground.optim import _solve_c_from_kappa_batched
from lora_playground.snapshot_analysis.snapshots import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    load_snapshot,
)
from lora_playground.snapshot_analysis.ssc import _prerescale_unit_op
from lora_playground.snapshot_analysis.whitening import whitened_NS_input


DEFAULT_OUT = (
    REPO
    / "notebooks/snapshot_analysis/_data/ssc_adaptive_c_policy_candidates.csv"
)
DEFAULT_CACHE_OUT = (
    REPO
    / "notebooks/snapshot_analysis/_data/ssc_kpar_cache_policy_candidates.csv"
)
DEFAULT_FAIL_OUT = (
    REPO
    / "notebooks/snapshot_analysis/_data/ssc_failing_snapshot_policy_candidates.csv"
)
DEFAULT_CACHE_CKPT = (
    REPO
    / "logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_sigma_guard_validate_2k"
    / "checkpoints/ckpt_step1750/optimizer.pt"
)
DEFAULT_FAIL_SNAPSHOT = (
    REPO
    / "logs/bench_ssc_drift/debug_bs32_kpar_K3R5_p2_picard_trace_replay_1750_1772_6446685"
    / "snapshots/step001771_pair069_non_finite_intermediate_base_model.model.model.layers.9.mlp.down_proj_default_.pt"
)


def _small_side_batch(X: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    Xf = X.float()
    if Xf.shape[-2] > Xf.shape[-1]:
        Xf = Xf.transpose(-2, -1)
    leading = Xf.shape[:-2]
    return Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1]), leading


def exact_c_from_kappa(X: torch.Tensor, kappa: float, eps: float = 1e-12) -> torch.Tensor:
    """Exact eigvalsh κ solve, without applying SSC."""
    Xb, leading = _small_side_batch(X)
    G = torch.bmm(Xb, Xb.transpose(-2, -1))
    lam = torch.linalg.eigvalsh(G).clamp_min(0.0)
    s_sq = lam / lam.max(dim=-1, keepdim=True).values.clamp_min(eps)
    c = _solve_c_from_kappa_batched(s_sq, kappa, c_lo=1e-3, c_hi=1e3, iters=40)
    return c.reshape(*leading)


def stable_rank_c_from_kappa(
    X: torch.Tensor,
    kappa: float,
    eps: float = 1e-6,
    c_min: float = 1e-3,
    c_max: float = 1e3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-spike-plus-flat-tail adaptive c from normalized stable rank.

    Returns (c, mu, m_tail), where mu = ||X||_F^2 / r under ||X||_op = 1.
    """
    Xb, leading = _small_side_batch(X)
    _, r, _ = Xb.shape
    if r <= 1:
        c = torch.full(Xb.shape[:1], float(c_max), device=Xb.device, dtype=Xb.dtype)
        mu = torch.ones_like(c)
        m = torch.ones_like(c)
        return c.reshape(*leading), mu.reshape(*leading), m.reshape(*leading)

    F2 = Xb.square().sum(dim=(-2, -1))
    mu = F2 / float(r)
    m = ((float(r) * mu - 1.0) / float(r - 1)).clamp(eps, 1.0 - eps)
    k_tail_value = (float(r) * float(kappa) - 1.0) / float(r - 1)
    k_tail = torch.full_like(m, k_tail_value).clamp(eps, 1.0 - eps)

    denom = (k_tail - m).clamp_min(eps)
    c2 = m * (1.0 - k_tail) / denom
    c = torch.sqrt(c2).clamp(float(c_min), float(c_max))
    c = torch.where(k_tail <= m + eps, torch.full_like(c, float(c_max)), c)
    c = torch.where(k_tail >= 1.0 - eps, torch.full_like(c, float(c_min)), c)
    c = torch.nan_to_num(c, nan=float(c_max), posinf=float(c_max), neginf=float(c_min))
    return c.reshape(*leading), mu.reshape(*leading), m.reshape(*leading)


def _sample_indices(n: int, k: int | None) -> list[int]:
    if k is None or k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _group_pair_items(pair_state: dict, pairs_per_group: int | None):
    groups = defaultdict(list)
    for pair_idx, pair in pair_state.items():
        A = pair["A"]
        B = pair["B"]
        groups[(A.shape[0], A.shape[1], B.shape[0])].append((int(pair_idx), pair))
    for shape, items in groups.items():
        indices = _sample_indices(len(items), pairs_per_group)
        yield shape, [items[i] for i in indices]


def write_snapshot_policy_csv(
    out_path: Path,
    *,
    kappa: float,
    steps: tuple[int, ...],
    pairs_per_group: int | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "lr", "lora_r", "ns", "variant", "step", "side",
        "shape", "pair", "r", "c_exact", "c_sr", "c_exact_group_median",
        "c_sr_group_median", "mu", "m_tail", "log_err_sr",
        "log_err_exact_group_median", "log_err_sr_group_median",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run_key in RUNS:
            available = set(STEPS_BY_ROOT.get(run_key, []))
            steps_here = [s for s in steps if s in available]
            root = SNAP_ROOTS[run_key]
            for step in steps_here:
                snap = load_snapshot(step, root=root)
                for side in ("A", "B"):
                    for shape, items in _group_pair_items(
                        snap["pair_state"], pairs_per_group
                    ):
                        Xs = []
                        pair_indices = []
                        for pair_idx, pair in items:
                            X = _prerescale_unit_op(whitened_NS_input(pair, side=side))
                            Xs.append(X)
                            pair_indices.append(pair_idx)
                        X_batch = torch.stack(Xs)
                        c_exact = exact_c_from_kappa(X_batch, kappa=kappa)
                        c_sr, mu, m_tail = stable_rank_c_from_kappa(
                            X_batch, kappa=kappa
                        )
                        exact_med = c_exact.median()
                        sr_med = c_sr.median()
                        r = min(X_batch.shape[-2], X_batch.shape[-1])
                        for j, pair_idx in enumerate(pair_indices):
                            ce = c_exact[j].clamp_min(1e-30)
                            cs = c_sr[j].clamp_min(1e-30)
                            row = {
                                "run": repr(run_key),
                                "lr": run_key[0],
                                "lora_r": run_key[1],
                                "ns": run_key[2],
                                "variant": run_key[3],
                                "step": step,
                                "side": side,
                                "shape": repr(shape),
                                "pair": pair_idx,
                                "r": r,
                                "c_exact": float(c_exact[j]),
                                "c_sr": float(c_sr[j]),
                                "c_exact_group_median": float(exact_med),
                                "c_sr_group_median": float(sr_med),
                                "mu": float(mu[j]),
                                "m_tail": float(m_tail[j]),
                                "log_err_sr": float((cs.log() - ce.log()).abs()),
                                "log_err_exact_group_median": float(
                                    (exact_med.clamp_min(1e-30).log() - ce.log()).abs()
                                ),
                                "log_err_sr_group_median": float(
                                    (sr_med.clamp_min(1e-30).log() - ce.log()).abs()
                                ),
                            }
                            writer.writerow(row)
                print(f"snapshot policy rows: {run_key} step={step}", flush=True)


def write_kpar_cache_csv(out_path: Path, ckpt_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint", "group", "slot", "side", "n", "local_index",
        "c_raw", "c_floor_1e_3", "c_floor_1e_4", "c_group_median",
        "raw_over_median", "floor_1e_3_over_median",
    ]
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for group_idx, gs in enumerate(sd.get("group_state", [])):
            for slot, value in sorted(gs.items()):
                if not (slot.startswith("ssc_c_cached_") or slot.startswith("ssc_c_last_")):
                    continue
                if not torch.is_tensor(value):
                    continue
                vals = value.float().reshape(-1)
                median = vals[torch.isfinite(vals)].median()
                parts = slot.split("_")
                side = parts[3] if len(parts) >= 4 else "?"
                n = parts[4] if len(parts) >= 5 else "last"
                for local_idx, raw in enumerate(vals):
                    c_floor_1e_3 = raw.clamp_min(1e-3)
                    c_floor_1e_4 = raw.clamp_min(1e-4)
                    writer.writerow({
                        "checkpoint": str(ckpt_path.relative_to(REPO)),
                        "group": group_idx,
                        "slot": slot,
                        "side": side,
                        "n": n,
                        "local_index": local_idx,
                        "c_raw": float(raw),
                        "c_floor_1e_3": float(c_floor_1e_3),
                        "c_floor_1e_4": float(c_floor_1e_4),
                        "c_group_median": float(median),
                        "raw_over_median": float(raw / median.clamp_min(1e-30)),
                        "floor_1e_3_over_median": float(
                            c_floor_1e_3 / median.clamp_min(1e-30)
                        ),
                    })
    print(f"kpar cache rows: {out_path}", flush=True)


def write_failing_snapshot_csv(out_path: Path, snapshot_path: Path, kappa: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "snapshot", "step", "pair_index", "pair_name", "x_key", "c_key",
        "side", "n", "finite", "op_norm_before_rescale", "frob_norm_before_rescale",
        "mu", "m_tail", "c_used", "c_exact", "c_sr", "log_err_used",
        "log_err_sr",
    ]
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    tensors = snap.get("tensors", {})
    candidates = [
        ("X_A_eff_n0", "ssc_c_A_n0", "A", "n0"),
        ("X_B_eff_n0", "ssc_c_B_n0", "B", "n0"),
        ("X_A_eff_n1", "ssc_c_A_n1", "A", "n1"),
        ("X_B_eff_n1", "ssc_c_B_n1", "B", "n1"),
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for x_key, c_key, side, n in candidates:
            X_raw = tensors.get(x_key)
            c_used = tensors.get(c_key)
            if not torch.is_tensor(X_raw):
                continue
            finite = bool(torch.isfinite(X_raw).all())
            op_before = float("nan")
            frob_before = float("nan")
            mu = torch.tensor(float("nan"))
            m_tail = torch.tensor(float("nan"))
            c_exact = torch.tensor(float("nan"))
            c_sr = torch.tensor(float("nan"))
            if finite:
                X_raw = X_raw.float()
                op_before = float(torch.linalg.matrix_norm(X_raw, ord=2))
                frob_before = float(X_raw.norm())
                X = _prerescale_unit_op(X_raw).unsqueeze(0)
                c_exact = exact_c_from_kappa(X, kappa=kappa).reshape(-1)[0]
                c_sr, mu, m_tail = stable_rank_c_from_kappa(X, kappa=kappa)
                c_sr = c_sr.reshape(-1)[0]
                mu = mu.reshape(-1)[0]
                m_tail = m_tail.reshape(-1)[0]
            used = float(c_used) if torch.is_tensor(c_used) else float("nan")
            ce = c_exact.clamp_min(1e-30)
            writer.writerow({
                "snapshot": str(snapshot_path.relative_to(REPO)),
                "step": snap.get("step"),
                "pair_index": snap.get("pair_index"),
                "pair_name": snap.get("pair_name"),
                "x_key": x_key,
                "c_key": c_key,
                "side": side,
                "n": n,
                "finite": finite,
                "op_norm_before_rescale": op_before,
                "frob_norm_before_rescale": frob_before,
                "mu": float(mu),
                "m_tail": float(m_tail),
                "c_used": used,
                "c_exact": float(c_exact),
                "c_sr": float(c_sr),
                "log_err_used": float(
                    abs(torch.tensor(used).clamp_min(1e-30).log() - ce.log())
                ),
                "log_err_sr": float(abs(c_sr.clamp_min(1e-30).log() - ce.log())),
            })
    print(f"failing snapshot rows: {out_path}", flush=True)


def _parse_steps(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", type=float, default=0.6)
    parser.add_argument("--steps", type=_parse_steps, default=(200, 1000, 4000))
    parser.add_argument("--pairs-per-group", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-out", type=Path, default=DEFAULT_CACHE_OUT)
    parser.add_argument("--fail-out", type=Path, default=DEFAULT_FAIL_OUT)
    parser.add_argument("--cache-ckpt", type=Path, default=DEFAULT_CACHE_CKPT)
    parser.add_argument("--fail-snapshot", type=Path, default=DEFAULT_FAIL_SNAPSHOT)
    args = parser.parse_args()

    write_snapshot_policy_csv(
        args.out,
        kappa=args.kappa,
        steps=args.steps,
        pairs_per_group=args.pairs_per_group,
    )
    if args.cache_ckpt.exists():
        write_kpar_cache_csv(args.cache_out, args.cache_ckpt)
    else:
        print(f"missing cache checkpoint: {args.cache_ckpt}", flush=True)
    if args.fail_snapshot.exists():
        write_failing_snapshot_csv(args.fail_out, args.fail_snapshot, args.kappa)
    else:
        print(f"missing failing snapshot: {args.fail_snapshot}", flush=True)


if __name__ == "__main__":
    main()
