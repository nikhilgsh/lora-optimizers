#!/usr/bin/env python
"""Probe Muon/Keller/MuP shape scaling on stored LoRA optimizer snapshots.

The Scientific Spaces note derives shape factors by asking for constant
RMS feature increments for a dense linear update. These snapshots do not
store layer activations, so this script tests the parts available from
optimizer state:

* which Keller/MuP multiplier each LoRA pair shape would receive;
* a no-activation isotropic-input proxy for current merged-step RMS,
  using logged ||dA||_F, ||dB||_F and throughput fractions;
* existing diagnostics that reveal non-isotropy of the polar input/update.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

from lora_playground.snapshot_analysis.snapshots import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    load_snapshot,
)


def _parse_steps(spec: str, available: list[int]) -> list[int]:
    if spec == "latest":
        return [max(available)] if available else []
    wanted = {int(s) for s in spec.split(",") if s.strip()}
    return [s for s in available if s in wanted]


def _shape_class(d_in: int, d_out: int) -> str:
    if d_out > d_in:
        return "expand"
    if d_out < d_in:
        return "compress"
    return "square"


def _float_diag(diag: dict, key: str) -> float | None:
    value = diag.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _row_from_pair(run_key: tuple, step: int, pair_index: int, pair: dict) -> dict | None:
    diag = pair.get("last_diag") or {}
    required = ("norm_dA", "norm_dB", "frac_dA_through_B", "frac_dB_through_A")
    if any(_float_diag(diag, key) is None for key in required):
        return None

    A = pair["A"]
    B = pair["B"]
    r, d_in = int(A.shape[0]), int(A.shape[1])
    d_out = int(B.shape[0])
    lr = float(run_key[0])
    ratio = d_out / d_in
    kj_alpha = math.sqrt(max(1.0, ratio))
    mup_alpha = math.sqrt(ratio)

    bdA_frob = _float_diag(diag, "frac_dA_through_B") * _float_diag(diag, "norm_dA")
    dBA_frob = _float_diag(diag, "frac_dB_through_A") * _float_diag(diag, "norm_dB")
    # If input coordinates were isotropic with RMS 1, E||x M^T||_2^2 / d_out
    # is ||M||_F^2 / d_out. We do not know the cross-term between B dA and dB A
    # from last_diag, so report a triangle upper proxy and per-block pieces.
    iso_A_over_lr = bdA_frob / math.sqrt(d_out) / lr
    iso_B_over_lr = dBA_frob / math.sqrt(d_out) / lr
    iso_sum_over_lr = iso_A_over_lr + iso_B_over_lr

    # The current tangent cap implies ||Delta W||_2 <= lr. For an adversarial
    # input direction with RMS 1, the corresponding output-RMS bound is this.
    worst_case_rms_over_lr = math.sqrt(d_in / d_out)

    return {
        "lr": run_key[0],
        "lora_r": run_key[1],
        "ns": run_key[2],
        "variant": run_key[3],
        "step": step,
        "pair": pair_index,
        "pair_step": pair.get("step"),
        "shape": f"{tuple(A.shape)}|{tuple(B.shape)}",
        "shape_class": _shape_class(d_in, d_out),
        "r": r,
        "d_in": d_in,
        "d_out": d_out,
        "d_out_over_d_in": ratio,
        "keller_alpha": kj_alpha,
        "mup_alpha": mup_alpha,
        "mup_over_keller": mup_alpha / kj_alpha,
        "current_worst_case_rms_over_lr_bound": worst_case_rms_over_lr,
        "current_iso_BdA_rms_over_lr": iso_A_over_lr,
        "current_iso_dBA_rms_over_lr": iso_B_over_lr,
        "current_iso_tangent_triangle_rms_over_lr": iso_sum_over_lr,
        "xunc_A_stable_rank_frac": (
            _float_diag(diag, "xunc_A_stable_rank") / r
            if _float_diag(diag, "xunc_A_stable_rank") is not None else None
        ),
        "xunc_B_stable_rank_frac": (
            _float_diag(diag, "xunc_B_stable_rank") / r
            if _float_diag(diag, "xunc_B_stable_rank") is not None else None
        ),
        "stable_rank_A_frac": (
            _float_diag(diag, "stable_rank_A") / r
            if _float_diag(diag, "stable_rank_A") is not None else None
        ),
        "stable_rank_B_frac": (
            _float_diag(diag, "stable_rank_B") / r
            if _float_diag(diag, "stable_rank_B") is not None else None
        ),
        "geoA_row_norm_cv": _float_diag(diag, "geoA_row_norm_cv"),
        "geoA_col_norm_cv": _float_diag(diag, "geoA_col_norm_cv"),
        "geoB_row_norm_cv": _float_diag(diag, "geoB_row_norm_cv"),
        "geoB_col_norm_cv": _float_diag(diag, "geoB_col_norm_cv"),
        "frac_dA_through_B": _float_diag(diag, "frac_dA_through_B"),
        "frac_dB_through_A": _float_diag(diag, "frac_dB_through_A"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> list[dict]:
    keys = ["lr", "lora_r", "ns", "variant", "step", "shape_class", "d_out_over_d_in"]
    fields = [
        "keller_alpha",
        "mup_alpha",
        "mup_over_keller",
        "current_worst_case_rms_over_lr_bound",
        "current_iso_tangent_triangle_rms_over_lr",
        "xunc_A_stable_rank_frac",
        "xunc_B_stable_rank_frac",
        "stable_rank_A_frac",
        "stable_rank_B_frac",
        "geoA_row_norm_cv",
        "geoA_col_norm_cv",
        "geoB_row_norm_cv",
        "geoB_col_norm_cv",
        "frac_dA_through_B",
        "frac_dB_through_A",
    ]
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(row)

    out = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(keys, key))
        row["n_pairs"] = len(bucket)
        for field in fields:
            vals = [b[field] for b in bucket if b.get(field) is not None]
            row[field + "_median"] = median(vals) if vals else None
        out.append(row)
    return out


def _iter_pairs(pair_state, max_pairs: int):
    if isinstance(pair_state, dict):
        keys = sorted(pair_state)
        for key in keys[:max_pairs]:
            yield int(key), pair_state[key]
        return
    for pair_index, pair in enumerate(pair_state[:max_pairs]):
        yield pair_index, pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/snapshot_analysis/_data/muon_scaling_snapshot_rows.csv")
    ap.add_argument("--summary-out", default="notebooks/snapshot_analysis/_data/muon_scaling_snapshot_summary.csv")
    ap.add_argument("--steps", default="4000,9000")
    ap.add_argument("--max-pairs", type=int, default=112)
    args = ap.parse_args()

    rows: list[dict] = []
    skipped = 0
    for run_key in RUNS:
        available = STEPS_BY_ROOT.get(run_key, [])
        for step in _parse_steps(args.steps, available):
            snap = load_snapshot(step, SNAP_ROOTS[run_key])
            pairs = snap["pair_state"]
            for pair_index, pair in _iter_pairs(pairs, args.max_pairs):
                row = _row_from_pair(run_key, step, pair_index, pair)
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)

    summary = _summarize(rows)
    _write_csv(Path(args.out), rows)
    _write_csv(Path(args.summary_out), summary)

    print(f"rows={len(rows)} skipped={skipped} summary_rows={len(summary)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    for row in summary:
        print(
            " ".join(
                [
                    f"run=({row['lr']},r{row['lora_r']},{row['variant']})",
                    f"step={row['step']}",
                    f"{row['shape_class']} q={row['d_out_over_d_in']:.3g}",
                    f"KJ={row['keller_alpha_median']:.3g}",
                    f"muP={row['mup_alpha_median']:.3g}",
                    f"iso/eta={row['current_iso_tangent_triangle_rms_over_lr_median']:.3g}",
                    f"xA/r={row['xunc_A_stable_rank_frac_median']:.3g}",
                    f"xB/r={row['xunc_B_stable_rank_frac_median']:.3g}",
                ]
            )
        )


if __name__ == "__main__":
    main()
