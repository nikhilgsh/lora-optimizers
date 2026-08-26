"""E2 derivation ablations scored the way the paper scores optimizers:
steps-to-Adam speedup, not sigma-units of final loss.

A 0.003 loss gap is hard to interpret on its own. The question the paper asks is
how many steps each arm needs to reach tuned Adam's FINAL loss, and the speedup is
Adam's horizon over that. So each ablation says how much of the protagonist's
speedup that piece of structure is responsible for.

Uses lora_playground.leaderboard.reach_fraction / speedup_from_frac -- the same
crossing-interpolation the leaderboard and paper figures use -- rather than a
hand-rolled crossing, which would round up to the eval grid and understate.

One loader pass; arm predicates match scripts/ablation_table.py.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from lora_playground.leaderboard import reach_fraction, speedup_from_frac
from lora_playground.loader import load_runs
from ablation_table import ARMS, CELL, matches  # single source of truth for the arms

HORIZON = 9000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=HORIZON)
    a = ap.parse_args()

    # arm -> lr -> history (list of eval events)
    hist = {lab: {} for lab in ARMS}
    for cfg, events in load_runs(where=CELL, warn_cross_commit=False):
        evs = [e for e in events
               if e.get("event") == "eval" and e.get("eval_loss") is not None]
        if not evs:
            continue
        for lab, pred in ARMS.items():
            if matches(cfg, pred):
                hist[lab].setdefault(cfg["lr"], []).extend(evs)

    def final(h):
        fin = [e for e in h if e.get("step") == a.horizon]
        return min((e["eval_loss"] for e in fin), default=None)

    # Adam's tuned final loss is the target every arm is timed against.
    adam = hist.get("AdamW", {})
    adam_best = {lr: final(h) for lr, h in adam.items()}
    adam_best = {lr: v for lr, v in adam_best.items() if v is not None}
    if not adam_best:
        print("no completed AdamW run at the horizon; nothing to time against")
        return
    target = min(adam_best.values())
    print(f"target = tuned Adam final loss at step {a.horizon}: {target:.4f} "
          f"(lr={min(adam_best, key=adam_best.get)})\n")

    print(f"{'structure removed':28s} {'best lr':>8s} {'final':>7s} {'steps-to-Adam':>14s} "
          f"{'speedup':>8s} {'% of PoLoRA gain':>17s}")
    rows = []
    for lab in ARMS:
        best = None
        for lr, h in hist[lab].items():
            f = final(h)
            if f is None:
                continue
            frac = reach_fraction(h, target, a.horizon)
            if best is None or f < best[1]:
                best = (lr, f, frac)
        rows.append((lab, best))

    proto = next((b for l, b in rows if l == "PoLoRA (protagonist)" and b), None)
    proto_speed = speedup_from_frac(proto[2]) if proto else None
    for lab, best in rows:
        if best is None:
            print(f"{lab:28s} {'-':>8s} {'-':>7s} {'-':>14s} {'-':>8s} {'-':>17s}")
            continue
        lr, f, frac = best
        sp = speedup_from_frac(frac)
        steps = frac * a.horizon
        # fraction of the protagonist's excess speedup over Adam (1.0x) retained
        if proto_speed and proto_speed > 1.0:
            share = 100.0 * (sp - 1.0) / (proto_speed - 1.0)
            share_s = f"{share:.0f}%"
        else:
            share_s = "-"
        print(f"{lab:28s} {float(lr):8g} {f:7.4f} {steps:14.0f} {sp:7.2f}x {share_s:>17s}")


if __name__ == "__main__":
    main()
