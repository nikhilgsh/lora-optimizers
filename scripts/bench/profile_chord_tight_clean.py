"""Per-component profile of the chord-tight-clean optimizer step.

Attaches CudaTimer to AdamPolarProductLoRA and prints per-scope wall time
across a few steps. Lets us see where opt_ms actually goes — Higham
refresh, σ_max power iters, polar map (NS), unwhiten, etc.

Usage (on a GPU node):
    python scripts/bench/profile_chord_tight_clean.py \\
        --lora_r 64 --picard_iters 2 --ns_form rect --n_steps 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_playground._step_timer import CudaTimer
from lora_playground.optim import build_optimizer
from lora_playground.training_kernel import build_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--ns_form", default="rect", choices=["rect", "gram"])
    parser.add_argument("--picard_iters", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--precond_method", default="higham")
    parser.add_argument("--higham_iters", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    peft_model = build_peft_model(
        model_name=args.model_name,
        target_modules=args.target_modules,
        training_mode="lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        dtype=dtype,
        device=device,
        attn_implementation="sdpa",
    )
    # The optimizer walks named_modules() to find lora_A/lora_B params; pass
    # the bare PEFT model (no DDP/compile wrappers). For training, also use it
    # since we didn't wrap in DDP or compile.
    bare_model = peft_model.bare_model
    model = peft_model.train_model

    optimizer = build_optimizer(
        bare_model,
        optimizer_type="adam-polar-product-lora-coupled-spectral-chord-tight-clean",
        lr=3e-3,
        precond_refresh_every=1,
        precond_method=args.precond_method,
        higham_iters=args.higham_iters,
        ns_form=args.ns_form,
        picard_iters_override=args.picard_iters,
    )

    # Attach timer
    timer = CudaTimer(device)
    optimizer._step_timer = timer

    vocab_size = bare_model.config.vocab_size
    g = torch.Generator(device=device).manual_seed(args.seed)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len),
                              device=device, generator=g)
    batch = {"input_ids": input_ids, "labels": input_ids}

    def one_step():
        for _ in range(args.grad_accum_steps):
            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

    # Warmup (untimed)
    for _ in range(args.n_warmup):
        one_step()
    torch.cuda.synchronize(device)

    # Reset timer state after warmup; time n_steps cleanly.
    timer.reset()
    t0 = time.perf_counter()
    for _ in range(args.n_steps):
        one_step()
    torch.cuda.synchronize(device)
    wall = (time.perf_counter() - t0) / args.n_steps
    summ = timer.summary()

    print(f"\n# profile: ns_form={args.ns_form} picard_iters={args.picard_iters} "
          f"r={args.lora_r} batch={args.batch_size} seq={args.seq_len} "
          f"grad_accum={args.grad_accum_steps} bf16={args.bf16}")
    print(f"# step wall ({args.n_steps} timed): {wall*1000:.1f} ms")
    print(f"# fwd+bwd+zero is OUTSIDE the timer scopes; only optimizer.step internals are below.\n")
    rows = sorted(summ.items(), key=lambda x: -x[1]["total_ms"])
    total = sum(v["total_ms"] for _, v in rows)
    print(f"  {'scope':<42} {'n':>6} {'mean_ms':>9} {'total_ms':>10} {'pct':>6}")
    for name, v in rows:
        # Total counts cumulatively across n_steps timed; per-step total = v[total]/n_steps.
        per_step = v["total_ms"] / args.n_steps
        print(f"  {name:<42} {int(v['n']):>6} {v['mean_ms']:>9.2f} {per_step:>10.2f} "
              f"{100*v['total_ms']/total:>5.1f}%")
    print(f"\n  {'(sum of timed scopes per step)':<42} {'':>6} {'':>9} {total/args.n_steps:>10.2f}")


if __name__ == "__main__":
    main()
