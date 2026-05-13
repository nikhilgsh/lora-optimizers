#!/usr/bin/env python
"""Compare S_A/S_B conditioning trajectories across ranks for chord-tight at lr=3e-3."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = {
    16:  Path("logs/chord_tight_diag_4k_r16r64_blackwell/run_info/logs/log_0.out"),
    64:  Path("logs/chord_tight_diag_4k_r16r64_blackwell/run_info/logs/log_1.out"),
    256: Path("logs/r256_chord_whiten_k1_lrsweep_4k_blackwell/run_info/logs/log_0.out"),
}

def load(path):
    steps, sa_min, sa_max, sb_min, sb_max = [], [], [], [], []
    eval_steps, eval_loss = [], []
    for line in path.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") == "optim_step":
            steps.append(d["step"])
            sa_min.append(d["SA_min_median"]); sa_max.append(d["SA_max_median"])
            sb_min.append(d["SB_min_median"]); sb_max.append(d["SB_max_median"])
        elif d.get("event") == "eval":
            eval_steps.append(d["step"]); eval_loss.append(d["eval_loss"])
    return steps, sa_min, sa_max, sb_min, sb_max, eval_steps, eval_loss

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = {16: "C0", 64: "C1", 256: "C2"}
for r, p in LOGS.items():
    s, samn, samx, sbmn, sbmx, es, el = load(p)
    condA = [mx/mn for mx, mn in zip(samx, samn)]
    condB = [mx/mn for mx, mn in zip(sbmx, sbmn)]
    axes[0].plot(s, condA, color=colors[r], label=f"r={r}")
    axes[1].plot(s, condB, color=colors[r], label=f"r={r}")
    axes[2].plot(es, el, color=colors[r], label=f"r={r}")

axes[0].set_yscale("log"); axes[0].set_title("cond(S_A) median over pairs")
axes[1].set_yscale("log"); axes[1].set_title("cond(S_B) median over pairs")
axes[2].set_title("eval_loss")
for ax in axes:
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)
plt.suptitle("chord-tight (k=1) lr=3e-3, conditioning vs rank, full 4k horizon")
plt.tight_layout()
out = Path("docs/notes/polar_product/conditioning_r_compare.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"saved {out}")

# Print numerical summary at first few logged steps
print("\nFirst-eval conditioning summary (median over pairs):")
print(f"{'r':>5} {'step':>5} {'cond(SA)':>12} {'cond(SB)':>12} {'eval_loss':>10}")
for r, p in LOGS.items():
    s, samn, samx, sbmn, sbmx, es, el = load(p)
    n = len(s)
    for i in [0, 1, 4, 9, n//4, n//2, 3*n//4, n-1]:
        if 0 <= i < n:
            print(f"{r:>5} {s[i]:>5} {samx[i]/samn[i]:>12.3g} {sbmx[i]/sbmn[i]:>12.3g} {'-':>10}")
