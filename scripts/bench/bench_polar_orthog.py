"""Microbenchmark: NS_rect vs NS_gram vs Polar Express on production LoRA shapes.

Times the three polar-orthogonalization primitives in `lora_playground.optim`:

  - `_newton_schulz`             — cubic, per-matrix, fp32 (canonical Muon)
  - `_newton_schulz_gram_batched`— Dao 2026 Algorithm 3 (fp16+restart by default)
  - `_polar_express`             — Amsel 2025, degree-5 with optimal coefficients

at production shapes (r ∈ {16, 64, 256}, d ∈ {2048} for OLMo-2-1B q/k/v/o), at
a sweep of iteration counts K ∈ {3, 5, 8, 10}, with both:

  - wall-time (CUDA-event timing, n_warmup + n_reps)
  - accuracy: max|σ_i − 1| on a sample of REAL u_A snapshot tensors

so we can answer the cost-matched question raised in the snapshot notebook:
does PE-j=8 do better than NS-j=10 on the actual u_A distribution? The K=10
NS > K=5 NS finding in `notebooks/packed_v1_leaderboard.ipynb` means residual
to true polar at fixed K is downstream-visible.

Snapshot source: `/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell`
(the same one cell 3 of muon_squared_snapshot_analysis.ipynb loads).

Output: JSONL to `logs/bench/polar_orthog_<hardware>.jsonl`, one row per
(variant, K, shape_kind, source). Rendered to
`docs/notes/polar_product/ns_polar_express_bench.md` by
`scripts/analysis/render_polar_orthog_table.py`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lora_playground.optim import (  # noqa: E402
    _newton_schulz,
    _newton_schulz_gram_batched,
    _polar_express,
)


# ---------------------------------------------------------------------------
# Variant adapters: unify the call signatures to `fn(X_2d, nsteps) -> X_2d`.
# `_newton_schulz_gram_batched` takes (..., m, n); wrap with unsqueeze/squeeze.
# Default fp16+restart matches the production path picked by `_pick_orthog_fn`.

def _ns_rect(X, nsteps):
    return _newton_schulz(X, nsteps=nsteps)


def _ns_gram(X, nsteps):
    # Production defaults: fp16 iteration, restart at iter 2.
    return _newton_schulz_gram_batched(
        X.unsqueeze(0), nsteps=nsteps,
        dtype=torch.float16, restart_at=2, safety_factor=1.05,
    ).squeeze(0)


def _pe(X, nsteps):
    return _polar_express(X, nsteps=nsteps)


VARIANTS = {
    "ns_rect": _ns_rect,
    "ns_gram_fp16": _ns_gram,
    "polar_express": _pe,
}


# ---------------------------------------------------------------------------
# Timing

def time_call(fn, X, nsteps, n_warmup, n_reps):
    """CUDA-event timing. Returns (mean_ms, std_ms, all_ms)."""
    for _ in range(n_warmup):
        _ = fn(X, nsteps)
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = fn(X, nsteps)
        e.record()
        torch.cuda.synchronize()
        times_ms.append(s.elapsed_time(e))
    t = torch.tensor(times_ms)
    return t.mean().item(), t.std().item(), times_ms


# ---------------------------------------------------------------------------
# Accuracy: max|σ_i − 1| on the output

@torch.no_grad()
def sigma_residual(Y):
    """max_i |σ_i(Y) − 1|. Y on GPU, returns CPU float."""
    s = torch.linalg.svdvals(Y.float())
    return (s - 1.0).abs().max().item()


# ---------------------------------------------------------------------------
# Real snapshot inputs

def load_snapshot_u_A(snapshot_root: Path, step: int, n_pairs: int, device: str):
    """Return a list of (pair_idx, u_A) for `n_pairs` spread across the layers."""
    sd = torch.load(
        snapshot_root / f"step_{step}" / "optimizer.pt",
        map_location="cpu", weights_only=False,
    )
    pair_state = sd["pair_state"]
    n_total = len(pair_state)
    import numpy as np
    indices = np.linspace(0, n_total - 1, n_pairs, dtype=int).tolist()
    out = []
    for pi in indices:
        u_A = pair_state[pi]["u_A"].to(device).float()
        out.append((int(pi), u_A))
    return out


# ---------------------------------------------------------------------------
# Synthetic shapes (rectangular A-side: (r, d))

SYNTHETIC_SHAPES = [
    # (r, d, label)
    (16,  2048, "r16_d2048"),
    (64,  2048, "r64_d2048"),
    (256, 2048, "r256_d2048"),
]


def make_random(r, d, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(r, d, device=device, dtype=torch.float32, generator=g)


# ---------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_reps", type=int, default=30)
    p.add_argument("--Ks", type=int, nargs="+", default=[3, 5, 8, 10])
    p.add_argument("--n_pairs", type=int, default=8,
                   help="number of real snapshot u_A tensors to evaluate accuracy on")
    p.add_argument(
        "--snapshot_root", type=Path,
        default=Path("/mnt/ceph/users/nghosh/lora_snapshots/"
                     "chord_tight_r64_k3_snapshot_blackwell/task_0"),
    )
    p.add_argument("--snapshot_step", type=int, default=2000)
    p.add_argument("--hardware", required=True,
                   choices=["blackwell", "a6000", "h100", "a100"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    sha = git_commit()
    gpu = gpu_name()
    print(f"# device={device}  gpu={gpu}  commit={sha[:10]}", flush=True)

    # ------------------------------------------------------------------
    # Equivalence sanity floor: at K=10 on random fp32, all variants
    # should pull σ → 1 to within < 0.1.
    print("# equivalence sanity floor at K=10 (random fp32)", flush=True)
    for r, d, label in SYNTHETIC_SHAPES:
        X = make_random(r, d, device, seed=0)
        for name, fn in VARIANTS.items():
            res = sigma_residual(fn(X, 10))
            print(f"#   {label:14s} {name:15s} K=10  max|σ-1|={res:.2e}", flush=True)
            assert res < 0.2, f"{name} at {label} K=10 has residual {res} — broken"

    # ------------------------------------------------------------------
    # Load real snapshot inputs once.
    snapshot_pairs = []
    if args.snapshot_root.exists():
        print(f"# loading {args.n_pairs} u_A tensors from "
              f"{args.snapshot_root}/step_{args.snapshot_step}", flush=True)
        snapshot_pairs = load_snapshot_u_A(
            args.snapshot_root, args.snapshot_step, args.n_pairs, device,
        )
        if snapshot_pairs:
            shapes = sorted({tuple(u.shape) for _, u in snapshot_pairs})
            print(f"#   loaded {len(snapshot_pairs)} pairs; unique shapes: {shapes}",
                  flush=True)
    else:
        print(f"# WARNING: snapshot_root {args.snapshot_root} not found — "
              f"skipping real-input accuracy rows", flush=True)

    # ------------------------------------------------------------------
    # Sweep: (variant, K) on each synthetic shape (timing) and each snapshot
    # pair (accuracy).
    rows = []
    t0 = time.time()

    # Synthetic-shape timing rows.
    for r, d, label in SYNTHETIC_SHAPES:
        X = make_random(r, d, device, seed=0)
        # One residual sample on random input too, for completeness.
        for K in args.Ks:
            for name, fn in VARIANTS.items():
                mean_ms, std_ms, _ = time_call(
                    fn, X, K, args.n_warmup, args.n_reps,
                )
                resid = sigma_residual(fn(X, K))
                row = {
                    "source": "random_fp32",
                    "shape_kind": label,
                    "r": r,
                    "d": d,
                    "variant": name,
                    "K": K,
                    "ms_mean": mean_ms,
                    "ms_std": std_ms,
                    "sigma_residual_max": resid,
                    "hardware": args.hardware,
                    "gpu_name": gpu,
                    "git_commit": sha,
                    "n_reps": args.n_reps,
                }
                rows.append(row)
                print(
                    f"  random {label:14s} {name:15s} K={K:2d}  "
                    f"{mean_ms:7.3f}±{std_ms:5.3f} ms  resid={resid:.2e}",
                    flush=True,
                )

    # Snapshot-input accuracy rows (one residual per pair × variant × K).
    # Timing on snapshot inputs duplicates the synthetic-shape timing (same
    # shapes); we only collect accuracy here.
    for pair_idx, u_A in snapshot_pairs:
        r, d = u_A.shape
        label = f"snapshot_r{r}_d{d}_pair{pair_idx}"
        for K in args.Ks:
            for name, fn in VARIANTS.items():
                resid = sigma_residual(fn(u_A, K))
                row = {
                    "source": "snapshot_u_A",
                    "shape_kind": label,
                    "snapshot_step": args.snapshot_step,
                    "pair_idx": pair_idx,
                    "r": r,
                    "d": d,
                    "variant": name,
                    "K": K,
                    "ms_mean": None,
                    "ms_std": None,
                    "sigma_residual_max": resid,
                    "hardware": args.hardware,
                    "gpu_name": gpu,
                    "git_commit": sha,
                }
                rows.append(row)

    # ------------------------------------------------------------------
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    print(f"# wrote {len(rows)} rows to {args.out} in {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
