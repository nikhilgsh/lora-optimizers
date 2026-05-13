#!/usr/bin/env python
"""Summarize the short init×damping pilot once it finishes.

Reads logs/short_init_damping_pilot_r256_blackwell/run_info/logs/log_{0..8}.out,
extracts eval_loss and σ_max(S_A), σ_max(S_B) trajectories per cell, plots a
3×2 figure (3 lrs × {loss, σ_max(S_B)}) with one line per init, and writes a
text summary table. Designed to be run after the pilot completes.

Layout assumes the task file order: lr ∈ {3e-3, 1e-2, 3e-2} outer × init ∈
{zero, gaussian, symmetric} inner — log_0..8 = (lr, init) pairs.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("logs/short_init_damping_pilot_r256_blackwell/run_info/logs")
OUT_DIR = Path("docs/notes/polar_product")

CELLS = [
    (0, "3e-3", "zero"),      (1, "3e-3", "gaussian"), (2, "3e-3", "symmetric"),
    (3, "1e-2", "zero"),      (4, "1e-2", "gaussian"), (5, "1e-2", "symmetric"),
    (6, "3e-2", "zero"),      (7, "3e-2", "gaussian"), (8, "3e-2", "symmetric"),
]
INIT_COLOR = {"zero": "C0", "gaussian": "C1", "symmetric": "C2"}
LRS = ["3e-3", "1e-2", "3e-2"]


def load_cell(log_path):
    eval_steps, eval_loss = [], []
    opt_steps, sa_max, sb_max = [], [], []
    if not log_path.exists():
        return None
    for line in log_path.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        if ev == "eval":
            eval_steps.append(d["step"])
            eval_loss.append(d["eval_loss"])
        elif ev == "optim_step":
            opt_steps.append(d["step"])
            sa_max.append(d.get("SA_max_median"))
            sb_max.append(d.get("SB_max_median"))
    return {
        "eval_steps": eval_steps, "eval_loss": eval_loss,
        "opt_steps": opt_steps, "sa_max": sa_max, "sb_max": sb_max,
    }


def main():
    data = {}
    for i, lr, init in CELLS:
        data[(lr, init)] = load_cell(LOG_DIR / f"log_{i}.out")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    for col, lr in enumerate(LRS):
        for init in ("zero", "gaussian", "symmetric"):
            d = data.get((lr, init))
            if d is None or not d["eval_steps"]:
                continue
            c = INIT_COLOR[init]
            axes[0, col].plot(d["eval_steps"], d["eval_loss"], color=c, label=init, marker='o', ms=3)
            axes[1, col].plot(d["opt_steps"], d["sa_max"], color=c, label=init)
            axes[2, col].plot(d["opt_steps"], d["sb_max"], color=c, label=init)
        axes[0, col].set_title(f"lr={lr}")
        axes[0, col].grid(alpha=0.3); axes[1, col].grid(alpha=0.3); axes[2, col].grid(alpha=0.3)
        axes[1, col].set_yscale("log"); axes[2, col].set_yscale("log")
    axes[0, 0].set_ylabel("eval_loss")
    axes[1, 0].set_ylabel("σ_max(S_A) median")
    axes[2, 0].set_ylabel("σ_max(S_B) median")
    for ax in axes[2, :]:
        ax.set_xlabel("step")
    axes[0, 0].legend(loc="best", fontsize=9)
    plt.suptitle("Pilot: init × lr at r=256 chord-tight whiten k=1, 600 steps")
    plt.tight_layout()
    out_png = OUT_DIR / "pilot_init_damping_r256.png"
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")

    # Text summary: final eval_loss and final σ_max per cell
    print(f"\n{'lr':>6} {'init':>10} {'final_step':>10} {'eval_loss':>10} {'σmax(S_A)':>10} {'σmax(S_B)':>10}")
    for i, lr, init in CELLS:
        d = data.get((lr, init))
        if d is None or not d["eval_steps"]:
            print(f"{lr:>6} {init:>10} {'(missing)':>10}")
            continue
        print(f"{lr:>6} {init:>10} {d['eval_steps'][-1]:>10} "
              f"{d['eval_loss'][-1]:>10.4f} "
              f"{d['sa_max'][-1] if d['sa_max'] else 'NA':>10} "
              f"{d['sb_max'][-1] if d['sb_max'] else 'NA':>10}")


if __name__ == "__main__":
    main()
