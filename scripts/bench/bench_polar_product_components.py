"""Component-level profile of AdamPolarProductLoRA.step().

Where bench_optimizer_step.py reports whole-step cost (fwd+bwd+opt+zero),
this script breaks the optimizer.step() itself into named components via
CUDA-event timers threaded through the optimizer:

  adam_direction         — per-coord Adam on (gA, gB) → (u_A, u_B)
  precond_refresh        — S_A^{-1/2}, S_B^{-1/2} via eigh or Higham
                           (only fires on refresh steps)
  picard_cross_coupling  — u_A_eff, u_B_eff formation per Picard iterate
  picard_polar_pipeline  — full _polar_pipeline call per iterate; subdivided:
    polar_whiten              — X_A, X_B = S^{-1/2} u
    polar_NS_A / polar_NS_B   — Newton-Schulz polar
    polar_unwhiten_rescale    — geo_{A,B} + Frobenius rescale to ‖u‖_F
  apply                  — A.add_(dA), B.add_(dB), grad.zero_()

Each cell runs a fixed number of steps (n_reps = n_cycles*K) and splits
recorded events into refresh-step vs stale-step phases by swapping the
optimizer's `_step_timer` attribute before each call.

Output: JSONL rows (one per (optimizer, r, K, phase)) with per-component
totals and per-step means. Sanity check: Σ component mean_ms ≈ optimizer
step time as measured by an outer CUDA-event pair (also recorded).

Usage:
    # Quick smoke (CPU-or-GPU, 1 cell):
    python scripts/bench/bench_polar_product_components.py --quick

    # A100 sweep (canonical):
    python scripts/bench/bench_polar_product_components.py \\
        --lora_r 16 64 \\
        --optimizers adam-polar-product-lora adam-polar-product-lora-coupled \\
        --precond_refresh_every 1 16 \\
        --bf16 \\
        --out logs/profile_polar_components/results.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from lora_playground._step_timer import CudaTimer
from lora_playground.optim import build_optimizer, OPTIMIZER_CHOICES
from lora_playground.utils import collect_lora_pairs


# Optimizers this harness instruments. AdamPolarProductLoRA-derived only —
# the timer hooks live in that class. AdamW is included as a no-instrumentation
# baseline (its CUDA cost is the floor anything polar-side must beat).
DEFAULT_OPTIMIZERS = [
    "adamw",
    "adam-polar-product-lora",
    "adam-polar-product-lora-coupled",
]


def build_model(model_name, lora_r, lora_alpha, target_modules, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    return model


def make_batch(model, batch_size, seq_len, device, generator):
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                              device=device, generator=generator)
    return {"input_ids": input_ids, "labels": input_ids}


def run_cell(model, optimizer, batch, n_warmup, n_cycles, K,
             grad_accum_steps, device):
    """Returns (refresh_summary, stale_summary, outer_step_ms_list).

    Refreshes vs stale are split by swapping optimizer._step_timer before
    each step based on the optimizer's internal step counter.
    """
    is_cuda = (device.type == "cuda")
    has_pair_state = hasattr(optimizer, "pair_state") and 0 in optimizer.pair_state

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    def fwd_bwd():
        for _ in range(grad_accum_steps):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

    # Warmup (no timer attached). Set the attribute explicitly so the
    # subsequent timed phase can swap it; the optimizer reads via getattr
    # with a None default so this is a no-op when it's None.
    optimizer._step_timer = None
    for _ in range(n_warmup):
        fwd_bwd()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
    sync()

    refresh_timer = CudaTimer(device) if is_cuda else None
    stale_timer = CudaTimer(device) if is_cuda else None
    outer_step_ms = {"refresh": [], "stale": []}

    n_reps = n_cycles * K
    for rep in range(n_reps):
        fwd_bwd()
        # Phase classification: a step is a "refresh" if the optimizer's
        # next step() call will hit the refresh branch.
        if has_pair_state:
            pre_step = optimizer.pair_state[0]['step']
            is_refresh = (pre_step % K == 0)
        else:
            is_refresh = False
        # Attach the right timer; AdamW reads no _step_timer so this is a
        # benign attribute set there.
        optimizer._step_timer = refresh_timer if is_refresh else stale_timer

        # Outer envelope around step() for sum-vs-total sanity.
        if is_cuda:
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
        else:
            t0 = time.perf_counter()
        optimizer.step()
        if is_cuda:
            ev_e.record()
        else:
            t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)

        if is_cuda:
            torch.cuda.synchronize(device)
            ms = ev_s.elapsed_time(ev_e)
        else:
            ms = (t1 - t0) * 1000.0
        outer_step_ms["refresh" if is_refresh else "stale"].append(ms)

    refresh_sum = refresh_timer.summary() if refresh_timer is not None else {}
    stale_sum = stale_timer.summary() if stale_timer is not None else {}
    return refresh_sum, stale_sum, outer_step_ms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    p.add_argument("--lora_r", nargs="+", type=int, default=[16, 64])
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default="all-linear")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMIZERS)
    p.add_argument("--precond_refresh_every", nargs="+", type=int, default=[1, 16])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_cycles", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSONL output path. If omitted, prints to stdout only.")
    p.add_argument("--quick", action="store_true",
                   help="Single-cell smoke: r=16, K=1, 1 cycle, 1 warmup, "
                        "AdamPolarProduct only. Ignores other size flags.")
    return p.parse_args()


def emit(out_fh, rec):
    line = json.dumps(rec, sort_keys=True)
    print(line, flush=True)
    if out_fh is not None:
        out_fh.write(line + "\n")
        out_fh.flush()


def main():
    args = parse_args()
    if args.quick:
        args.lora_r = [16]
        args.precond_refresh_every = [1]
        args.optimizers = ["adam-polar-product-lora"]
        args.n_cycles = 1
        args.n_warmup = 1

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("a")
    else:
        out_fh = None

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    try:
        for r in args.lora_r:
            print(f"# === lora_r={r} ===", flush=True)
            model = build_model(args.model_name, r, args.lora_alpha,
                                args.target_modules, dtype, device)
            pairs = collect_lora_pairs(model)
            n_pairs = len(pairs)
            n_lora_params = sum(A.numel() + B.numel() for A, B in pairs)
            gen = torch.Generator(device=device).manual_seed(args.seed)
            batch = make_batch(model, args.batch_size, args.seq_len, device, gen)

            for opt_name in args.optimizers:
                if opt_name not in OPTIMIZER_CHOICES:
                    print(f"# skip unknown optimizer: {opt_name}", flush=True)
                    continue
                Ks = (args.precond_refresh_every
                      if opt_name != "adamw" else [1])
                for K in Ks:
                    optimizer = build_optimizer(
                        model, optimizer_type=opt_name, lr=args.lr,
                        precond_refresh_every=K,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    refresh_sum, stale_sum, outer = run_cell(
                        model, optimizer, batch, args.n_warmup, args.n_cycles,
                        K, args.grad_accum_steps, device,
                    )
                    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                               if device.type == "cuda" else 0.0)

                    base_rec = {
                        "event": "profile_polar_components",
                        "optimizer": opt_name,
                        "lora_r": r,
                        "precond_refresh_every": K,
                        "n_pairs": n_pairs,
                        "n_lora_params": n_lora_params,
                        "model_name": args.model_name,
                        "target_modules": args.target_modules,
                        "dtype": str(dtype),
                        "device": str(device),
                        "gpu_name": gpu_name,
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "grad_accum_steps": args.grad_accum_steps,
                        "n_warmup": args.n_warmup,
                        "n_cycles": args.n_cycles,
                        "peak_memory_mb": peak_mb,
                    }

                    # Components form a 1-level tree: PARENT_SCOPES wrap the
                    # listed children. For coverage % use leaves only (else
                    # parent + children double-count).
                    PARENT_SCOPES = {
                        "picard_polar_pipeline": (
                            "polar_whiten", "polar_NS_A", "polar_NS_B",
                            "polar_unwhiten_rescale",
                        ),
                    }
                    parent_set = set(PARENT_SCOPES)

                    for phase, comp_sum, outer_list in (
                        ("refresh", refresh_sum, outer["refresh"]),
                        ("stale", stale_sum, outer["stale"]),
                    ):
                        if not outer_list:
                            continue
                        n_steps_phase = len(outer_list)
                        outer_mean = sum(outer_list) / n_steps_phase
                        # Per-step component mean: total_ms across all calls
                        # of that scope, divided by n_steps_phase. Captures
                        # "ms of polar_NS_A per optimizer step" even though
                        # the scope fires once per pair per Picard iter.
                        per_step_components = {
                            name: stats["total_ms"] / n_steps_phase
                            for name, stats in comp_sum.items()
                        }
                        leaf_sum = sum(v for n, v in per_step_components.items()
                                       if n not in parent_set)
                        rec = dict(base_rec)
                        rec.update({
                            "phase": phase,
                            "n_steps_phase": n_steps_phase,
                            "outer_step_ms_mean": outer_mean,
                            "outer_step_ms_min": min(outer_list),
                            "outer_step_ms_max": max(outer_list),
                            "leaf_sum_ms_per_step": leaf_sum,
                            "leaf_coverage_pct": (
                                100.0 * leaf_sum / outer_mean
                                if outer_mean > 0 else float("nan")),
                            "components_ms_per_step": per_step_components,
                            "components_n_calls": {
                                name: int(stats["n"])
                                for name, stats in comp_sum.items()
                            },
                            "parent_scopes": {
                                p: list(c) for p, c in PARENT_SCOPES.items()
                            },
                        })
                        emit(out_fh, rec)
                        # Human-readable line.
                        print(f"  {opt_name:<40} r={r:<3} K={K:<3} {phase:<7} "
                              f"outer={outer_mean:7.2f} ms  "
                              f"leaves_sum={leaf_sum:7.2f} ms  "
                              f"({rec['leaf_coverage_pct']:5.1f}% covered)",
                              flush=True)

                    del optimizer
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if out_fh is not None:
            out_fh.close()


if __name__ == "__main__":
    main()
