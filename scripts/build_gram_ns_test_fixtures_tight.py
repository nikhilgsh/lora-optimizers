"""Build real-input precision test fixtures for Gram Newton-Schulz.

Walks the chord-tight r=64 k=3 snapshots, reconstructs the polar input
X_eff = S_B^{-1/2} u_A (and X_eff = u_B S_A^{-1/2}) that would feed
_newton_schulz_gram_batched in production, and saves the worst-conditioned
ones to a fixture file.

Snapshot source:
  /mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0/

Snapshot command (from meta.json step_2000):
  --precond_method higham --precond_delta_relative --precond_delta 1e-2
  --lora_r 64 --picard_iters_override 3 --muon_ns_steps 5

Output: tests/fixtures/gram_ns_real_inputs.pt with keys:
  - "X_effs":  list of (m, n) fp32 tensors (m >= n; transposed if needed)
  - "metadata": list of dicts {pair, step, side, cond_G, sigma_min, sigma_max}
  - "audit": dict with cond histogram stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lora_playground.utils import spd_inv_sqrt_higham_batched

SNAP_ROOT = Path(
    "/mnt/ceph/users/nghosh/lora_snapshots/"
    "chord_tight_r64_k3_snapshot_blackwell/task_0"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "tests/fixtures/gram_ns_real_inputs_tight.pt"

# Use tight (production-min) damping to stress-test conditioning. The original
# snapshot used precond_delta=1e-2 (well-damped); chord-tight-clean sweeps also
# use 1e-6 / 1e-4 / 1e-3 depending on the cfg, and tighter damping pushes
# σ_min(S_B) lower → cond(X_eff) higher. This rebuild exercises the regime
# we're hedging against.
PRECOND_EPS = 1e-6
PRECOND_EPS_RELATIVE = True
HIGHAM_ITERS = 10


def load_snapshot(step: int) -> dict[int, dict]:
    sd = torch.load(SNAP_ROOT / f"step_{step}" / "optimizer.pt",
                    map_location="cpu", weights_only=False)
    return sd["pair_state"]


def reconstruct_X_eff(pair: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_eff_A, S_A, X_eff_B, S_B) for one pair, all fp32."""
    A = pair["A"].float()                      # (r, d_in)
    B = pair["B"].float()                      # (d_out, r)
    u_A = pair["u_A"].float()                  # (r, d_in)
    u_B = pair["u_B"].float()                  # (d_out, r)

    S_A = A @ A.transpose(-2, -1)              # (r, r)
    S_B = B.transpose(-2, -1) @ B              # (r, r)

    # Batch dim of 1 for the spd_inv_sqrt API.
    S_A_half_inv = spd_inv_sqrt_higham_batched(
        S_A.unsqueeze(0), n_iters=HIGHAM_ITERS,
        eps=PRECOND_EPS, eps_relative=PRECOND_EPS_RELATIVE,
    )[0]
    S_B_half_inv = spd_inv_sqrt_higham_batched(
        S_B.unsqueeze(0), n_iters=HIGHAM_ITERS,
        eps=PRECOND_EPS, eps_relative=PRECOND_EPS_RELATIVE,
    )[0]

    X_eff_A = S_B_half_inv @ u_A               # (r, d_in)
    X_eff_B = u_B @ S_A_half_inv               # (d_out, r)
    return X_eff_A, S_A, X_eff_B, S_B


def gram_conditioning(X: torch.Tensor) -> tuple[float, float, float]:
    """Return (cond(G), sigma_min(X), sigma_max(X)) for G = smaller-side Gram of X."""
    # Compute singular values directly; cond(G) = cond(X)².
    svals = torch.linalg.svdvals(X.float())
    s_max = float(svals[0])
    s_min = float(svals[-1])
    if s_min <= 0:
        cond_G = float("inf")
    else:
        cond_G = (s_max / s_min) ** 2
    return cond_G, s_min, s_max


def main():
    print(f"Loading snapshots from {SNAP_ROOT}")
    steps = sorted(int(p.name.split("_")[1]) for p in SNAP_ROOT.glob("step_*"))
    print(f"Steps: {steps}")

    records = []  # list of (cond_G, X_eff fp32, metadata dict)

    # Skip step=0 — u_A is zero at init (degenerate singular X_eff,
    # cond=inf for all pairs); not a useful precision test case.
    steps = [s for s in steps if s > 0]

    for step in steps:
        pairs = load_snapshot(step)
        print(f"  step {step:5d}: {len(pairs)} pairs", flush=True)
        for pair_idx, pair in sorted(pairs.items()):
            try:
                X_eff_A, _, X_eff_B, _ = reconstruct_X_eff(pair)
            except Exception as exc:
                print(f"    pair {pair_idx}: skipped ({exc})")
                continue

            for side, X in [("A", X_eff_A), ("B", X_eff_B)]:
                # Convention: ensure rows are the smaller side (matches the
                # tall-transpose in _newton_schulz_*_batched).
                if X.shape[0] > X.shape[1]:
                    X = X.transpose(-2, -1).contiguous()
                cond_G, s_min, s_max = gram_conditioning(X)
                records.append((cond_G, X.clone(), {
                    "pair": pair_idx, "step": step, "side": side,
                    "shape": tuple(X.shape),
                    "cond_G": cond_G, "sigma_min": s_min, "sigma_max": s_max,
                }))

    # Audit summary across all records.
    conds = torch.tensor([r[0] for r in records if torch.isfinite(torch.tensor(r[0]))])
    print(f"\nAudit ({len(records)} X_eff matrices):")
    print(f"  cond(G):  min={conds.min().item():.2e}  "
          f"median={conds.median().item():.2e}  "
          f"max={conds.max().item():.2e}")
    for thresh in (1e2, 1e3, 1e4, 1e5):
        frac = float((conds > thresh).float().mean())
        print(f"  fraction with cond(G) > {thresh:.0e}: {frac:.3f}")

    # Pick top-20 worst-conditioned for the fixture.
    records.sort(key=lambda r: -r[0])
    top20 = records[:20]
    print(f"\nTop-20 worst-conditioned (saved to fixture):")
    for i, (cond_G, _, meta) in enumerate(top20):
        print(f"  [{i:2d}] step={meta['step']:5d} pair={meta['pair']:3d} "
              f"side={meta['side']} shape={meta['shape']}  cond(G)={cond_G:.2e}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X_effs": [r[1] for r in top20],
        "metadata": [r[2] for r in top20],
        "audit": {
            "n_total": len(records),
            "cond_min": float(conds.min()),
            "cond_median": float(conds.median()),
            "cond_max": float(conds.max()),
            "frac_above_1e3": float((conds > 1e3).float().mean()),
            "frac_above_1e4": float((conds > 1e4).float().mean()),
        },
        "source": {
            "snap_root": str(SNAP_ROOT),
            "steps": steps,
            "precond_eps": PRECOND_EPS,
            "precond_eps_relative": PRECOND_EPS_RELATIVE,
            "higham_iters": HIGHAM_ITERS,
        },
    }
    torch.save(payload, OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
