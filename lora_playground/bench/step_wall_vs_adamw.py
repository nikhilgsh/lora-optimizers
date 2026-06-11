"""Full-step wall ×AdamW for the protagonist, sweeping precond_method and global
batch — the analog of `docs/notes/timing_benchmarks_r256_blackwell.md` §3a and the
paper's per-step-overhead table (skeleton.tex §sec:exp-walltime), now including the
`gram_ns` (Polar-Express Gram NS inverse-sqrt) upgrade alongside the eigh/QR default.

AdamW is the optimizer-invariant fwd+bwd baseline; the cw cells reuse the SAME
compiled model, so the only thing that moves between rows is `optimizer.step()`.
The reported ratio is mean full-step wall / AdamW mean full-step wall.

Run (Blackwell): python -m lora_playground.bench.step_wall_vs_adamw \
    --model allenai/OLMo-2-0425-1B --lora_r 256 --ga 1 4 16
"""
from __future__ import annotations

import argparse

import torch

from .config import protagonist_config
from .harness import build_model, build_protagonist_direct, make_batch, time_full_step
from ..optim import build_optimizer
from ..utils import collect_lora_pairs


def _time_optimizer(model, optimizer, batch, *, K, ga, device, n_warmup, n_cycles):
    n_reps = n_cycles * K
    if device.type == "cuda":
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    per_step, _is_refresh, phase = time_full_step(
        model, optimizer, batch, n_warmup, n_reps, K, ga, device)
    mean = sum(per_step) / len(per_step)
    opt = sum(phase["opt"]) / len(phase["opt"])
    fb = (sum(phase["fwd"]) + sum(phase["bwd"])) / len(phase["fwd"])
    peak = (torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0)
    return {"mean": mean, "opt": opt, "fwd_bwd": fb, "peak_mb": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--ga", nargs="+", type=int, default=[4],
                    help="grad_accum_steps to sweep (global batch = bs*ga; bs=4)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["eigh", "gram_ns"],
                    choices=["eigh", "gram_ns", "higham"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--higham_iters", type=int, default=8,
                    help="iter count for the NS methods (gram_ns sweet spot = 8)")
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_cycles", type=int, default=4)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = protagonist_config(args.model, args.lora_r)
    cfg.batch_size = args.batch_size
    cfg.higham_iters = args.higham_iters
    if args.no_compile:
        cfg.compile_mode = None
    print(f"# {cfg.banner()}", flush=True)
    print(f"# device={device} K={args.K} higham_iters={args.higham_iters} "
          f"n_warmup={args.n_warmup} n_cycles={args.n_cycles}", flush=True)

    peft = build_model(cfg, device)
    bare, model = peft.bare_model, peft.train_model
    gen = torch.Generator(device=device).manual_seed(0)
    batch = make_batch(bare, cfg.batch_size, cfg.max_seq_length, device, gen,
                       data_pipeline_version=cfg.data_pipeline_version)

    print(f"\n{'global_batch':>12} {'optimizer':>22} {'step ms':>9} {'opt ms':>8} "
          f"{'fwd+bwd ms':>11} {'xAdamW':>8} {'peak MB':>9}", flush=True)
    for ga in args.ga:
        gb = args.batch_size * ga
        # AdamW baseline (optimizer-invariant fwd+bwd; minimal kwargs).
        # AdamW HPs don't affect timing (step is a trivial add_); use build defaults.
        adamw = build_optimizer(bare, optimizer_type="adamw", lr=1e-3)
        a = _time_optimizer(model, adamw, batch, K=1, ga=ga, device=device,
                            n_warmup=args.n_warmup, n_cycles=args.n_cycles)
        adamw_mean = a["mean"]
        print(f"{gb:>12} {'adamw':>22} {a['mean']*1e3:>9.1f} {a['opt']*1e3:>8.2f} "
              f"{a['fwd_bwd']*1e3:>11.1f} {1.0:>8.3f} {a['peak_mb']:>9.0f}", flush=True)
        del adamw
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for method in args.methods:
            # Direct build: the factory silently drops precond_method for the cw
            # protagonist spec (see build_protagonist_direct), so go around it.
            opt = build_protagonist_direct(bare, cfg, precond_method=method, K=args.K)
            assert opt.precond_method == method, \
                f"precond_method not in effect: wanted {method}, got {opt.precond_method}"
            r = _time_optimizer(model, opt, batch, K=args.K, ga=ga, device=device,
                                n_warmup=args.n_warmup, n_cycles=args.n_cycles)
            label = f"{cfg.optimizer}/{method}"
            print(f"{gb:>12} {label:>22} {r['mean']*1e3:>9.1f} {r['opt']*1e3:>8.2f} "
                  f"{r['fwd_bwd']*1e3:>11.1f} {r['mean']/adamw_mean:>8.3f} "
                  f"{r['peak_mb']:>9.0f}", flush=True)
            del opt
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
