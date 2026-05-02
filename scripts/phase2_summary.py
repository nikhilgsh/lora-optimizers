"""Phase 2 sweep summary — pulls eval_loss and key diagnostics from
all 3 phase-2 sweep dirs + baselines, prints a final comparison table.

Usage: conda run -n ffcv-pl python scripts/phase2_summary.py [--step 2000]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.loader import load_runs


def latest_eval(logfile, target_step=None):
    """Return (step, eval_loss) of the latest eval at-or-before target_step
    (or absolute latest if target_step is None)."""
    best = None
    for line in open(logfile):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "eval":
            continue
        s = d.get("step", 0)
        if target_step is not None and s > target_step:
            continue
        if best is None or s > best[0]:
            best = (s, d.get("eval_loss"))
    return best


def cell_config(logfile):
    """Read the config event from the head of the log."""
    for line in open(logfile):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") == "config":
            return d
    return None


def collect_sweep(group, target_step=2000):
    """Walk a sweep group's log directory and return per-cell (cfg, step, loss)."""
    log_dir = ROOT / "logs" / group / "run_info" / "logs"
    rows = []
    if not log_dir.exists():
        return rows
    for f in sorted(log_dir.glob("log_*.out")):
        cfg = cell_config(f)
        if cfg is None:
            continue
        ev = latest_eval(f, target_step)
        if ev is None:
            continue
        rows.append({
            "log": f.name,
            "optimizer": cfg["optimizer"],
            "lr": cfg["lr"],
            "lora_r": cfg["lora_r"],
            "step": ev[0],
            "eval_loss": ev[1],
        })
    return rows


def best_per_rank(rows, optimizer_filter=None):
    """Return {(opt, r): best (lr, step, eval_loss)} from a row list."""
    out = {}
    for r in rows:
        if optimizer_filter and r["optimizer"] not in optimizer_filter:
            continue
        key = (r["optimizer"], r["lora_r"])
        if key not in out or r["eval_loss"] < out[key][2]:
            out[key] = (r["lr"], r["step"], r["eval_loss"])
    return out


def baseline_summary(target_step=2000):
    """Pull AdamW + adam-polar-product-lora-coupled at target_step via
    canonical loader, m=1 only.
    """
    runs = load_runs(where={
        "optimizer": ["adamw", "adam-polar-product-lora-coupled"],
        "lora_r": [16, 64],
        "lora_plus_multiplier": 1.0,
    })
    out = {}
    for cfg, evs in runs:
        finals = [e for e in evs if e.get("event") == "eval" and e.get("step") == target_step]
        if not finals:
            last = [e for e in evs if e.get("event") == "eval"]
            if not last:
                continue
            finals = [last[-1]]
        e = finals[-1]
        key = (cfg["optimizer"], cfg["lora_r"])
        if key not in out or e["eval_loss"] < out[key][2]:
            out[key] = (cfg["lr"], e["step"], e["eval_loss"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, default=2000,
                   help="Target step for comparison (default 2000)")
    args = p.parse_args()

    print(f"Phase 2 summary — eval_loss at step ≤ {args.step}")
    print("=" * 78)
    print()

    # Baselines
    print("=== Baselines (canonical loader, m=1, step 2000) ===")
    bl = baseline_summary(args.step)
    print(f"{'optimizer':<40} {'r':>3} {'best_lr':>10} {'step':>5} {'eval':>9}")
    print("-" * 70)
    for (opt, r), (lr, step, loss) in sorted(bl.items()):
        print(f"{opt:<40} {r:>3} {lr:>10.0e} {step:>5} {loss:>9.4f}")
    print()

    # Phase 1 reference: vanilla variant 1
    print("=== Phase 1 reference: polar-coupled-core-2k group ===")
    rows_v1 = collect_sweep("polar_coupled_core_2k", args.step)
    best_v1 = best_per_rank(rows_v1)
    print(f"{'optimizer':<40} {'r':>3} {'best_lr':>10} {'step':>5} {'eval':>9}")
    print("-" * 70)
    for (opt, r), (lr, step, loss) in sorted(best_v1.items()):
        print(f"{opt:<40} {r:>3} {lr:>10.0e} {step:>5} {loss:>9.4f}")
    print()

    # Phase 1.5: state rebalance
    print("=== Phase 1.5: state_rebalanced_2k group ===")
    rows_sr = collect_sweep("state_rebalanced_2k", args.step)
    best_sr = best_per_rank(rows_sr)
    print(f"{'optimizer':<45} {'r':>3} {'best_lr':>10} {'step':>5} {'eval':>9}")
    print("-" * 75)
    for (opt, r), (lr, step, loss) in sorted(best_sr.items()):
        print(f"{opt:<45} {r:>3} {lr:>10.0e} {step:>5} {loss:>9.4f}")
    print()

    # Phase 2 (A): wider lr scan
    print("=== Phase 2 (A): polar_core_wide_lr_2k group ===")
    rows_wl = collect_sweep("polar_core_wide_lr_2k", args.step)
    best_wl = best_per_rank(rows_wl)
    print(f"{'optimizer':<40} {'r':>3} {'best_lr':>10} {'step':>5} {'eval':>9}")
    print("-" * 70)
    for (opt, r), (lr, step, loss) in sorted(best_wl.items()):
        print(f"{opt:<40} {r:>3} {lr:>10.0e} {step:>5} {loss:>9.4f}")
    print("ALL CELLS:")
    for r in sorted(rows_wl, key=lambda x: (x["lora_r"], x["lr"])):
        print(f"  r={r['lora_r']:>3} lr={r['lr']:>10.0e} step={r['step']:>5} eval={r['eval_loss']:.4f}")
    print()

    # Phase 2 (B): sign norm
    print("=== Phase 2 (B): polar_core_sign_2k group ===")
    rows_sn = collect_sweep("polar_core_sign_2k", args.step)
    if not rows_sn:
        print("  (no data yet — sweep may still be queued or just started)")
    else:
        best_sn = best_per_rank(rows_sn)
        print(f"{'optimizer':<40} {'r':>3} {'best_lr':>10} {'step':>5} {'eval':>9}")
        print("-" * 70)
        for (opt, r), (lr, step, loss) in sorted(best_sn.items()):
            print(f"{opt:<40} {r:>3} {lr:>10.0e} {step:>5} {loss:>9.4f}")
        print("ALL CELLS:")
        for r in sorted(rows_sn, key=lambda x: (x["lora_r"], x["lr"])):
            print(f"  r={r['lora_r']:>3} lr={r['lr']:>10.0e} step={r['step']:>5} eval={r['eval_loss']:.4f}")
    print()

    # Verdict comparison vs baselines
    print("=== Verdict (each new optimizer's best vs hybrid Picard) ===")
    pi = bl.get(("adam-polar-product-lora-coupled", 16))
    if pi:
        pi16 = pi[2]
        print(f"  hybrid Picard r=16: {pi16:.4f}  (target to beat)")
    pi = bl.get(("adam-polar-product-lora-coupled", 64))
    if pi:
        pi64 = pi[2]
        print(f"  hybrid Picard r=64: {pi64:.4f}")
    print()

    def verdict(loss, picard):
        if loss is None or picard is None:
            return "n/a"
        gap = loss - picard
        if loss < picard:
            return f"BEATS PICARD by {-gap:.4f}"
        if loss < picard + 0.02:
            return f"WITHIN 0.02 (Δ=+{gap:.4f}) — competitive"
        if loss < picard + 0.04:
            return f"PARTIAL (Δ=+{gap:.4f})"
        return f"NO HELP (Δ=+{gap:.4f})"

    for name, best_dict in [
        ("Phase 1 vanilla", best_v1),
        ("Phase 1.5 state-rebal", best_sr),
        ("Phase 2 (A) wide-lr", best_wl),
        ("Phase 2 (B) sign", best_per_rank(rows_sn) if rows_sn else {}),
    ]:
        for r_size in (16, 64):
            picard_loss = bl.get(("adam-polar-product-lora-coupled", r_size), [None]*3)[2]
            best = None
            for (opt, r), (lr, step, loss) in best_dict.items():
                if r != r_size: continue
                if best is None or loss < best[2]:
                    best = (opt, lr, loss)
            if best is None:
                print(f"  {name:<25} r={r_size}: (no data)")
                continue
            print(f"  {name:<25} r={r_size}: {best[0][:35]:<35} lr={best[1]:>10.0e} loss={best[2]:.4f}  → {verdict(best[2], picard_loss)}")
    print()


if __name__ == "__main__":
    main()
