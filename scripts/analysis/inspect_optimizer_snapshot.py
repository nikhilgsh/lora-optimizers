#!/usr/bin/env python
"""Summarize optimizer debug snapshots emitted by `debug_snapshot_dir`.

The polar-product NaN tracer saves one `.pt` file per offending pair. This
script reports which tensors are non-finite and the scale of every tensor so
the first failure link is visible without interactive torch spelunking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch


def _snapshot_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pt")))
        elif p.exists():
            paths.append(p)
    return sorted(paths)


def _tensor_stats(t: torch.Tensor) -> dict[str, object]:
    td = t.detach().cpu()
    finite = torch.isfinite(td)
    finite_count = int(finite.sum().item())
    total = td.numel()
    out: dict[str, object] = {
        "shape": tuple(td.shape),
        "dtype": str(td.dtype),
        "finite": finite_count == total,
        "nan": int(torch.isnan(td).sum().item()),
        "inf": int(torch.isinf(td).sum().item()),
    }
    if finite_count:
        vals = td[finite].float()
        out["absmax_finite"] = float(vals.abs().max().item())
        out["norm_finite"] = float(vals.norm().item())
        out["min_finite"] = float(vals.min().item())
        out["max_finite"] = float(vals.max().item())
    return out


def _load_snapshot(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize(path: Path) -> None:
    snap = _load_snapshot(path)
    print(f"== {path} ==")
    for key in ("reason", "step", "pair_index", "local_index", "group_id", "pair_name"):
        print(f"{key}: {snap.get(key)}")
    where = snap.get("where")
    if where is not None:
        print(f"where: {where}")

    scalars = snap.get("scalars") or {}
    if scalars:
        print("scalars:")
        for name in sorted(scalars):
            print(f"  {name}: {scalars[name]}")

    tensors = snap.get("tensors") or {}
    if not tensors:
        print("no tensors")
        return

    bad = []
    print("tensors:")
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            print(f"  {name}: {type(value).__name__} {value!r}")
            continue
        stats = _tensor_stats(value)
        if not stats["finite"]:
            bad.append(name)
        scale = (
            f"absmax={stats.get('absmax_finite'):.6g} "
            f"norm={stats.get('norm_finite'):.6g}"
            if "absmax_finite" in stats
            else "no finite entries"
        )
        print(
            f"  {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"finite={stats['finite']} nan={stats['nan']} inf={stats['inf']} "
            f"{scale}"
        )
    print(f"non_finite_tensors: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Snapshot .pt files or directories")
    args = parser.parse_args()
    paths = _snapshot_paths(args.paths)
    if not paths:
        raise SystemExit("no snapshot .pt files found")
    for i, path in enumerate(paths):
        if i:
            print()
        summarize(path)


if __name__ == "__main__":
    main()
