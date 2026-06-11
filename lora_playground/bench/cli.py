"""``python -m lora_playground.bench <subcommand>`` — the single standardized
bench entry point.

  step-wall      full-step wall for a (model, rank, optimizer), optionally
                 sweeping precond_method (eigh/higham) and K, → optionally writes
                 timing_registry.json.
  inverse-sqrt   Higham-vs-eigh accuracy on real snapshot Grams.
  record         measure a training log and upsert a (wall-inclusive) registry
                 entry — thin wrapper over ``lora_playground.timing record``.
"""
from __future__ import annotations

import argparse
import json

from .config import BenchConfig, protagonist_config


# short alias → SNAP_ROOTS key (see lora_playground/snapshot_analysis/snapshots.py)
_SNAP_ALIASES = {
    "r256k1": ("3e-2", 256, 5, "chord-tight-k1"),
    "r64k1": ("3e-2", 64, 5, "chord-tight-k1"),
}


def _cmd_step_wall(a):
    from .harness import run_step_wall
    from .registry import record_bench
    if a.preset == "protagonist":
        cfg = protagonist_config(a.model, a.lora_r, hardware=a.hardware,
                                 optimizer=a.optimizer)
    else:
        cfg = BenchConfig(model_name=a.model, lora_r=a.lora_r, optimizer=a.optimizer,
                          hardware=a.hardware)
    if a.batch_size is not None:
        cfg.batch_size = a.batch_size
    if a.grad_accum_steps is not None:
        cfg.grad_accum_steps = a.grad_accum_steps
    if a.no_compile:
        cfg.compile_mode = None
    results = run_step_wall(cfg, methods=tuple(a.methods), ks=a.ks,
                            n_cycles=a.n_cycles)
    if a.write_registry:
        for r in results:
            entry = record_bench(r, a.hardware, force=a.force)
            print(f"# registry <- {entry['key']}  {entry['s_per_step']} s/step  "
                  f"(source={entry.get('source')})", flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            for r in results:
                fh.write(json.dumps(r, default=str) + "\n")


def _cmd_inverse_sqrt(a):
    from .inverse_sqrt import run_inverse_sqrt_accuracy
    key = _SNAP_ALIASES.get(a.snapshot, a.snapshot)
    run_inverse_sqrt_accuracy(key, n_iters_list=tuple(a.n_iters), eps=a.eps,
                              max_pairs=a.max_pairs)


def _cmd_profile(a):
    from .profile import profile_step
    if a.preset == "protagonist":
        cfg = protagonist_config(a.model, a.lora_r, optimizer=a.optimizer)
    else:
        cfg = BenchConfig(model_name=a.model, lora_r=a.lora_r, optimizer=a.optimizer)
    if a.no_compile:
        cfg.compile_mode = None
    profile_step(cfg, precond_method=(a.precond_method or None),
                 refresh_every=a.refresh_every, warmup=a.warmup, steps=a.steps)


def _cmd_record(a):
    from .. import timing
    entry = timing.record(a.log_path, hardware=a.hardware)
    print(json.dumps(entry, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lora_playground.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("step-wall", help="full-step wall (eigh vs higham, K sweep)")
    sw.add_argument("--model", required=True)
    sw.add_argument("--lora_r", type=int, required=True)
    sw.add_argument("--optimizer", default="kl-diag-polar-lora")
    sw.add_argument("--preset", default="protagonist", choices=["protagonist", "none"])
    sw.add_argument("--methods", nargs="+", default=["eigh"], choices=["eigh", "higham"])
    sw.add_argument("--ks", nargs="+", type=int, default=[10])
    sw.add_argument("--batch_size", type=int, default=None)
    sw.add_argument("--grad_accum_steps", type=int, default=None)
    sw.add_argument("--n_cycles", type=int, default=4)
    sw.add_argument("--hardware", default="blackwell")
    sw.add_argument("--no-compile", action="store_true", help="eager (smoke only)")
    sw.add_argument("--write-registry", action="store_true")
    sw.add_argument("--force", action="store_true", help="overwrite a train-log entry")
    sw.add_argument("--out", default=None, help="append result JSONL")
    sw.set_defaults(fn=_cmd_step_wall)

    iv = sub.add_parser("inverse-sqrt", help="Higham-vs-eigh accuracy on real snapshots")
    iv.add_argument("--snapshot", default="r256k1", help="alias (r256k1/r64k1) or SNAP_ROOTS key")
    iv.add_argument("--n_iters", nargs="+", type=int, default=[5, 8, 10, 16])
    iv.add_argument("--eps", type=float, default=1e-4)
    iv.add_argument("--max_pairs", type=int, default=None)
    iv.set_defaults(fn=_cmd_inverse_sqrt)

    pf = sub.add_parser("profile", help="op-level profile of optimizer.step()")
    pf.add_argument("--model", required=True)
    pf.add_argument("--lora_r", type=int, required=True)
    pf.add_argument("--optimizer", default="kl-diag-polar-lora")
    pf.add_argument("--preset", default="protagonist", choices=["protagonist", "none"])
    pf.add_argument("--precond_method", default=None, choices=[None, "eigh", "higham"])
    pf.add_argument("--refresh_every", type=int, default=None)
    pf.add_argument("--warmup", type=int, default=15)
    pf.add_argument("--steps", type=int, default=12)
    pf.add_argument("--no-compile", action="store_true")
    pf.set_defaults(fn=_cmd_profile)

    rc = sub.add_parser("record", help="upsert a wall-inclusive registry entry from a log")
    rc.add_argument("log_path")
    rc.add_argument("--hardware", required=True)
    rc.set_defaults(fn=_cmd_record)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
