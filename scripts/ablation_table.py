"""E2 derivation-ablation table: step-matched eval loss per arm, in sigma units.

ONE loader pass. The obvious way to write this -- load_runs(where={**CELL, **arm})
once per arm -- rescans the whole logs/ tree per arm (544 groups x 6 arms), which
took minutes. Load the cell once, then match each arm's extra predicates in memory.

Usage:  python scripts/ablation_table.py [--step N] [--all-steps]
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from lora_playground.loader import load_runs

SIGMA = 0.00050  # MEASURED at this cell: protagonist across seeds 0-3 at lr=1e-2,
                 # step 9000 (scripts/cell_sigma.py). The borrowed packed_v1 anchor
                 # 0.0017 (OLMo/magicoder, r=16/64) is 3.4x too large here and made
                 # the two r x r arms look like noise when they are 5-6 sigma.

CELL = dict(model_name="meta-llama/Llama-3.2-1B", lora_r=256,
            data_dir=(lambda d: "openmath" in str(d)), max_steps=9000)
# Every field that distinguishes the locked protagonist from a neighbouring sweep.
# Without the last four this matched 13 runs across 7 configs at lr=1e-2 alone
# (cw_solved_rho=True, rdinv_variant VN/B, cw_metric_init zero/delta/ones, the
# curvature_beta grid) and silently kept whichever had the lowest loss -- which
# moved the lr=3e-3 point by 0.0067.
PIN = dict(cw_nesterov=True, polar_method="polar_express",
           beta1=0.9, precond_method="gram_ns",
           cw_solved_rho=False, rdinv_variant="A",
           cw_metric_init="1e-12", curvature_beta=0.99)

# label -> extra predicates on top of CELL. Order is display order.
ARMS = {
    "PoLoRA (protagonist)":      dict(optimizer="kl-diag-polar-lora", cw_unpinned=False,
                                      cw_no_diag_curv=False, cw_no_rr_precond=False, **PIN),
    "w/o msign (metric^-1)":     dict(optimizer="kl-diag-lora", **PIN),
    "w/o msign (metric^-1/2)":   dict(optimizer="kl-diag-flatout-lora", **PIN),
    "w/o outer un-whiten":       dict(optimizer="kl-diag-polar-flatout-lora", **PIN),
    "w/o rxr metric CONTENTS":   dict(optimizer="kl-shampoo-polar-lora", **PIN),
    "w/o rxr preconditioner":    dict(optimizer="kl-diag-polar-lora",
                                      cw_no_rr_precond=True, **PIN),
    "w/o diagonal P,Q":          dict(optimizer="kl-diag-polar-lora",
                                      cw_no_diag_curv=True, **PIN),
    "AdamW":                     dict(optimizer="adamw"),
}
REF = "PoLoRA (protagonist)"


def matches(cfg, pred):
    """Same predicate semantics as loader._matches: literal / membership / callable.
    A run missing a referenced field does not match."""
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
    ap.add_argument("--step", type=int, default=None,
                    help="report at this step; default = deepest step every arm has reached")
    ap.add_argument("--all-steps", action="store_true",
                    help="also print how deep each arm currently is")
    a = ap.parse_args()

    runs = load_runs(where=CELL, warn_cross_commit=False)   # <- the single pass
    series = {lab: {} for lab in ARMS}
    for cfg, events in runs:
        tr = {e["step"]: e["eval_loss"] for e in events
              if e.get("event") == "eval" and e.get("eval_loss") is not None}
        if not tr:
            continue
        for lab, pred in ARMS.items():
            if matches(cfg, pred):
                series[lab].setdefault(cfg["lr"], {}).update(tr)

    have = {lab: d for lab, d in series.items() if d}
    depth = {lab: max(max(t) for t in d.values()) for lab, d in have.items()}
    if a.all_steps:
        for lab in ARMS:
            print(f"  {lab:26s} deepest step {depth.get(lab, 0)}")
        print()
    if REF not in have:
        print("no protagonist data; nothing to anchor on")
        return
    step = a.step if a.step is not None else min(depth.values())

    at = {}
    for lab, d in have.items():
        vals = {lr: t[step] for lr, t in d.items() if step in t}
        if vals:
            blr = min(vals, key=vals.get)
            at[lab] = (blr, vals[blr])
    if REF not in at:
        print(f"protagonist has no eval at step {step}")
        return
    base = at[REF][1]

    print(f"step-matched at {step}/9000   sigma={SIGMA} (AdamW multiseed, packed_v1)\n")
    print(f"{'structure removed':28s} {'best lr':>8s} {'eval':>8s} {'delta':>9s} {'sigma':>7s}  verdict")
    for lab in ARMS:
        if lab not in at:
            d = depth.get(lab, 0)
            note = f"only to step {d}" if d else "no data yet"
            print(f"{lab:28s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>7s}  {note}")
            continue
        blr, v = at[lab]
        dl = v - base
        su = dl / SIGMA
        if lab == REF:
            verdict = "reference"
        elif abs(su) < 1:
            verdict = "within noise"
        else:
            verdict = f"{su:.1f} sigma"
        print(f"{lab:28s} {float(blr):8g} {v:8.4f} {dl:+9.4f} {su:7.1f}  {verdict}")


if __name__ == "__main__":
    main()
