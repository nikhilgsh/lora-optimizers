"""Build real-input precision test fixtures for the bf16 Higham port.

Walks the chord-tight r=64 k=3 snapshots and extracts the per-pair Gram
matrices S_A = A·A^T, S_B = B^T·B that feed
`spd_inv_sqrt_higham_batched` in production. Picks the top-20
worst-conditioned across all (step, pair, side) for the Tier-1 fixture.

Snapshot source (same as `build_gram_ns_test_fixtures.py` and the
muon² hypothesis-test notebook):
  /mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0/

Output: tests/fixtures/higham_real_grams.pt with keys:
  - "S_grams": list of fp32 (r, r) tensors
  - "metadata": list of dicts {pair, step, side, shape, cond_S, sigma_min, sigma_max}
  - "audit": condition-number distribution stats
  - "source": snapshot path and selection params
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SNAP_ROOT = Path(
    "/mnt/ceph/users/nghosh/lora_snapshots/"
    "chord_tight_r64_k3_snapshot_blackwell/task_0"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests/fixtures/higham_real_grams.pt"
)


def load_snapshot(step: int) -> dict[int, dict]:
    sd = torch.load(
        SNAP_ROOT / f"step_{step}" / "optimizer.pt",
        map_location="cpu", weights_only=False,
    )
    return sd["pair_state"]


def gram_conditioning(S: torch.Tensor) -> tuple[float, float, float]:
    """Return (cond(S), σ_min(S), σ_max(S)). S is SPD so singular values
    equal eigenvalues, but svdvals is the safer call since pair_state
    tensors carry no PSD guarantee — A·A^T may have tiny negative
    eigenvalues from floating-point rounding."""
    svals = torch.linalg.svdvals(S.float())
    s_max = float(svals[0])
    s_min = float(svals[-1])
    cond_S = float("inf") if s_min <= 0 else s_max / s_min
    return cond_S, s_min, s_max


def main():
    print(f"Loading snapshots from {SNAP_ROOT}")
    steps = sorted(int(p.name.split("_")[1]) for p in SNAP_ROOT.glob("step_*"))
    print(f"Steps: {steps}")

    # Skip step=0: B is zero at LoRA init, so S_B is exactly singular
    # (cond = ∞) for every pair. Not a useful precision test case —
    # the damping `eps_relative=True` path lifts the smallest eigenvalue
    # to eps·λ_max which would dominate. Keep the natural mid-training
    # cond range.
    steps = [s for s in steps if s > 0]

    records = []  # list of (cond_S, S fp32, metadata dict)
    for step in steps:
        pairs = load_snapshot(step)
        print(f"  step {step:5d}: {len(pairs)} pairs", flush=True)
        for pair_idx, pair in sorted(pairs.items()):
            try:
                A = pair["A"].float()              # (r, d_in)
                B = pair["B"].float()              # (d_out, r)
            except KeyError as exc:
                print(f"    pair {pair_idx}: skipped ({exc})")
                continue
            S_A = A @ A.transpose(-2, -1)          # (r, r)
            S_B = B.transpose(-2, -1) @ B          # (r, r)
            for side, S in [("A", S_A), ("B", S_B)]:
                cond_S, s_min, s_max = gram_conditioning(S)
                records.append((cond_S, S.clone(), {
                    "pair": pair_idx, "step": step, "side": side,
                    "shape": tuple(S.shape),
                    "cond_S": cond_S, "sigma_min": s_min, "sigma_max": s_max,
                }))

    # Audit distribution before selecting top-20.
    conds = torch.tensor(
        [r[0] for r in records if torch.isfinite(torch.tensor(r[0]))]
    )
    print(f"\nAudit ({len(records)} Gram matrices):")
    print(
        f"  cond(S):  min={conds.min().item():.2e}  "
        f"median={conds.median().item():.2e}  "
        f"max={conds.max().item():.2e}"
    )
    for thresh in (1e2, 1e3, 1e4, 1e5):
        frac = float((conds > thresh).float().mean())
        print(f"  fraction with cond(S) > {thresh:.0e}: {frac:.3f}")

    # Top-20 worst-conditioned (highest cond first).
    records.sort(key=lambda r: -r[0])
    top20 = records[:20]
    print(f"\nTop-20 worst-conditioned (saved to fixture):")
    for i, (cond_S, _, meta) in enumerate(top20):
        print(
            f"  [{i:2d}] step={meta['step']:5d} pair={meta['pair']:3d} "
            f"side={meta['side']} shape={meta['shape']}  cond(S)={cond_S:.2e}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "S_grams": [r[1] for r in top20],
        "metadata": [r[2] for r in top20],
        "audit": {
            "n_total": len(records),
            "cond_min": float(conds.min()),
            "cond_median": float(conds.median()),
            "cond_max": float(conds.max()),
            "frac_above_1e2": float((conds > 1e2).float().mean()),
            "frac_above_1e3": float((conds > 1e3).float().mean()),
            "frac_above_1e4": float((conds > 1e4).float().mean()),
            "frac_above_1e5": float((conds > 1e5).float().mean()),
        },
        "source": {
            "snap_root": str(SNAP_ROOT),
            "steps": steps,
        },
    }
    torch.save(payload, OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
