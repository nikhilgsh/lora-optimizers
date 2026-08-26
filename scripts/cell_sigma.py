"""Across-seed spread measured AT this cell (Llama-3.2-1B / openmath / r256 / 9000).

Groups by the SAME explicit arm predicates ablation_table.py uses, not by a loose
key. A loose key is how the first version of this script went wrong: keying on
(optimizer, lr, cw_no_rr_precond, cw_no_diag_curv) collapsed a cw_solved_rho=True
run -- a different arm -- into the protagonist's seed-0 slot and reported
sigma=0.0158 instead of ~0.001. Many kl-diag-polar-lora runs at lr=1e-2 exist that
differ in cw_solved_rho / cw_metric_init / cw_unpinned / rdinv_variant, so the
predicate has to pin every one of them.

sigma is step-specific: an early-step spread does not transfer to step 9000, so the
step is printed with every row and must be quoted alongside the number.
"""
import argparse
import statistics
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from lora_playground.loader import load_runs

CELL = dict(model_name="meta-llama/Llama-3.2-1B", lora_r=256,
            data_dir=(lambda d: "openmath" in str(d)), max_steps=9000)
# Everything that distinguishes the locked protagonist config from its neighbours.
LOCKED = dict(cw_nesterov=True, polar_method="polar_express", beta1=0.9,
              precond_method="gram_ns", precond_delta=1e-4,
              cw_metric_init="1e-12", cw_solved_rho=False, cw_unpinned=False,
              cw_no_radius=False, cw_no_diag_curv=False, rdinv_variant="A",
              cw_factor_a=0.0, cw_factor_b=0.0, curvature_beta=0.99)

ARMS = {
    "PoLoRA (protagonist)":    dict(optimizer="kl-diag-polar-lora", lr=0.01,
                                    cw_no_rr_precond=False, **LOCKED),
    "w/o rxr metric contents": dict(optimizer="kl-shampoo-polar-lora", lr=0.01, **LOCKED),
    "w/o rxr preconditioner":  dict(optimizer="kl-diag-polar-lora", lr=0.003,
                                    cw_no_rr_precond=True, **LOCKED),
}
BORROWED = 0.0017


def matches(cfg, pred):
    for k, v in pred.items():
        if k not in cfg:
            return False
        c = cfg[k]
        if callable(v):
            if not v(c):
                return False
        elif isinstance(v, (list, set, tuple)):
            if c not in v:
                return False
        elif c != v:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()

    per_arm = defaultdict(dict)  # arm -> {seed: {step: loss}}
    for cfg, events in load_runs(where=CELL, warn_cross_commit=False):
        tr = {e["step"]: e["eval_loss"] for e in events
              if e.get("event") == "eval" and e.get("eval_loss") is not None}
        if not tr:
            continue
        for lab, pred in ARMS.items():
            if not matches(cfg, pred):
                continue
            seed = cfg.get("seed")
            prev = per_arm[lab].get(seed)
            if prev is None or max(tr) > max(prev):
                per_arm[lab][seed] = tr

    print(f"{'arm':26s} {'lr':>7s} {'seeds':>14s} {'step':>6s} {'mean':>8s} "
          f"{'sigma':>9s} {'2sigma':>8s}")
    sigmas = {}
    for lab, pred in ARMS.items():
        seeds = per_arm.get(lab, {})
        if len(seeds) < 2:
            print(f"{lab:26s} {float(pred['lr']):7g} {str(sorted(seeds)):>14s} "
                  f"{'-':>6s} {'-':>8s} {'-':>9s} {'-':>8s}")
            continue
        common = set.intersection(*(set(t) for t in seeds.values()))
        step = a.step if (a.step is not None and a.step in common) else max(common)
        vals = [seeds[s][step] for s in sorted(seeds)]
        m, s = statistics.mean(vals), statistics.stdev(vals)
        sigmas[lab] = (s, step)
        print(f"{lab:26s} {float(pred['lr']):7g} {str(sorted(seeds)):>14s} {step:6d} "
              f"{m:8.4f} {s:9.5f} {2*s:8.5f}")
        print(f"{'':26s} values: {['%.4f' % v for v in vals]}")

    print()
    print(f"borrowed anchor in use by ablation_table.py: {BORROWED}")
    if sigmas:
        worst = max(s for s, _ in sigmas.values())
        print(f"largest measured cell sigma so far: {worst:.5f} "
              f"({'above' if worst > BORROWED else 'below'} the borrowed anchor)")
    print("sigma is step-specific -- quote the step with it; these are EARLY steps")
    print("and the final-step spread can differ.")


if __name__ == "__main__":
    main()
