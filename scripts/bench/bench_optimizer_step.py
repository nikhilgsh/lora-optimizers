"""Benchmark FULL training-step wall time (forward + backward + optimizer.step)
as a function of (optimizer, lora_r, precond_refresh_every).

Why end-to-end: optimizer.step() in isolation gives a misleading ×AdamW ratio
because AdamW's step is a trivial param.add_() while the Gram optimizers do
substantial per-step work. The number that matters in production is
(fwd + bwd + step), where fwd+bwd is identical across optimizers and the
optimizer cost shows up as an additive overhead.

Per cell (optimizer, K):
  1. Build a fresh optimizer with precond_refresh_every=K.
  2. Warmup: n_warmup full forward+backward+step calls.
  3. Time loop: n_cycles*K full training steps (so each cell sees exactly
     n_cycles refresh-step cycles). Each step is timed individually with
     torch.cuda.synchronize() before/after; refresh vs stale steps are
     classified from the optimizer's internal step counter.
  4. Report:
     - mean_sec_per_step: amortized end-to-end step cost
     - refresh_sec_per_step / stale_sec_per_step: per-phase breakdown
     - ratio_vs_adamw: mean_sec_per_step / AdamW's mean_sec_per_step
       (production-relevant — small numbers like 1.1× are expected)
     - speedup_vs_k1: mean_K1 / mean_K within the same optimizer

Inputs are random token IDs at (batch_size, max_seq_length). The forward pass
work matches a real training step at this shape; only the data is fake. The
SDPA path used here is the implicit causal mask — NOT packed_v1's explicit 4D
block-diagonal mask. For shape-controlled relative optimizer timing this is
fine; for projecting an end-to-end production wall at a new training config
(new seq, batch, dataset, packing setting), prefer a short 20-step train_lora.py
smoke at that exact config and read `peak_memory_mb` + `tokens_per_sec` from
the eval log line.

Hardware: A100 is canonical; A6000 is acceptable for relative ratios since
they apply uniformly across rows.

Usage:
    # Smoke
    python scripts/bench/bench_optimizer_step.py --lora_r 16 --n_cycles 2 --bf16

    # Full bench
    python scripts/bench/bench_optimizer_step.py --lora_r 256 \\
        --precond_refresh_every 1 5 10 20 --bf16 \\
        --out logs/bench/precond_stale.jsonl
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lora_playground.data import build_block_diagonal_causal_mask
from lora_playground.distributed import (
    cleanup as dist_cleanup,
    get_world_size,
    init_distributed,
    is_main,
)
from lora_playground.mfu import (
    compute_mfu,
    count_total_params,
    device_peak_tflops,
    flops_per_token_for_mode,
)
from lora_playground.optim import build_optimizer, OPTIMIZER_CHOICES
from lora_playground.training_kernel import build_peft_model
from lora_playground.utils import collect_lora_pairs

# Optimizers that take a precond_refresh_every kwarg.
GRAM_PRECOND_OPTIMIZERS = [
    "adam-scaled-lora",
    "adam-lin-lora",
    "adam-polar-product-lora",
    "adamuon-polar-product-lora",
]
# Coupled-core solver variants (no precond_refresh; per-step QR + small SVDs).
# Included in default bench list at K=1 only.
COUPLED_CORE_OPTIMIZERS = [
    "polar-coupled-core-lora",
    "muon-coupled-core-lora",
]
DEFAULT_OPTIMIZERS = ["adamw"] + GRAM_PRECOND_OPTIMIZERS + COUPLED_CORE_OPTIMIZERS


def make_batch(model, batch_size: int, seq_len: int, device: torch.device,
               generator: torch.Generator,
               *, data_pipeline_version: str = "unpacked_v0",
               n_docs_per_slot: int = 3):
    """Random token-ID batch. Vocab size from model config.

    `data_pipeline_version="packed_v1"` simulates a production packed batch:
    `n_docs_per_slot` equal-length docs per slot, per-doc position_ids reset,
    4D block-diagonal causal SDPA mask. The 4D mask path is what the
    production train.py actually runs under packed_v1, so timing it here
    captures the additional kernel cost (vs the implicit 2D causal mask
    `unpacked_v0` would use).
    """
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                              device=device, generator=generator)
    if data_pipeline_version == "unpacked_v0":
        return {"input_ids": input_ids, "labels": input_ids}
    if data_pipeline_version != "packed_v1":
        raise ValueError(f"unknown data_pipeline_version: {data_pipeline_version}")
    # Synthetic packed slot: split seq_len into n_docs_per_slot equal chunks.
    n_docs = max(1, n_docs_per_slot)
    base = seq_len // n_docs
    rem = seq_len - base * n_docs
    doc_lens = [base + (1 if i < rem else 0) for i in range(n_docs)]
    pos: list[int] = []
    for L in doc_lens:
        pos.extend(range(L))
    position_ids = (torch.tensor(pos, device=device, dtype=torch.long)
                    .unsqueeze(0).expand(batch_size, -1).contiguous())
    # Mask dtype must match SDPA's expectation. bf16 model → bf16 mask.
    param_dtype = next(model.parameters()).dtype
    mask_dtype = param_dtype if param_dtype.is_floating_point else torch.float32
    mask_1 = build_block_diagonal_causal_mask(doc_lens, seq_len, mask_dtype, device)
    attention_mask = mask_1.expand(batch_size, 1, seq_len, seq_len).contiguous()
    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def time_full_step(model, optimizer, batch, n_warmup: int, n_reps: int,
                   K: int, grad_accum_steps: int, device: torch.device):
    """Time `n_reps` full training steps (grad_accum_steps × (fwd + bwd) +
    optimizer.step + zero_grad). Returns (per_step_times, is_refresh, phase_times)
    where phase_times is dict of lists: {'fwd': [...], 'bwd': [...], 'opt': [...],
    'zero': [...]}. is_refresh is False for optimizers without pair_state (AdamW)."""
    has_pair_state = hasattr(optimizer, "pair_state") and 0 in optimizer.pair_state
    is_cuda = (device.type == "cuda")

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    def one_step_timed():
        fwd_ms = bwd_ms = 0.0
        for _ in range(grad_accum_steps):
            sync()
            t0 = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            sync()
            fwd_ms += time.perf_counter() - t0
            t0 = time.perf_counter()
            loss.backward()
            sync()
            bwd_ms += time.perf_counter() - t0
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        opt_ms = time.perf_counter() - t0
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        sync()
        zero_ms = time.perf_counter() - t0
        return fwd_ms, bwd_ms, opt_ms, zero_ms

    def one_step_untimed():
        for _ in range(grad_accum_steps):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

    # Warmup
    for _ in range(n_warmup):
        sync()
        one_step_untimed()
        sync()

    # Time
    per_step = []
    is_refresh = []
    phase_times = {"fwd": [], "bwd": [], "opt": [], "zero": []}
    for _ in range(n_reps):
        sync()
        if has_pair_state:
            pre_step = optimizer.pair_state[0]['step']
            this_is_refresh = (pre_step % K == 0)
        else:
            this_is_refresh = False
        fwd_ms, bwd_ms, opt_ms, zero_ms = one_step_timed()
        per_step.append(fwd_ms + bwd_ms + opt_ms + zero_ms)
        phase_times["fwd"].append(fwd_ms)
        phase_times["bwd"].append(bwd_ms)
        phase_times["opt"].append(opt_ms)
        phase_times["zero"].append(zero_ms)
        is_refresh.append(this_is_refresh)
    return per_step, is_refresh, phase_times


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 for the model (matches production sweep dtype).")
    parser.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMIZERS,
                        help="Which optimizers to bench. AdamW serves as a baseline.")
    parser.add_argument("--precond_refresh_every", nargs="+", type=int, default=[1, 5, 10, 20],
                        help="K values to sweep. Only applied to Gram-preconditioned optimizers; "
                             "AdamW etc. are timed once at K=1.")
    parser.add_argument("--precond_method", nargs="+", default=["eigh"],
                        choices=["eigh", "higham"],
                        help="precond_method values to sweep over. Only applies to "
                             "adam-polar-product-lora and adamuon-polar-product-lora. Pass "
                             "'eigh higham' to compare both.")
    parser.add_argument("--higham_iters", type=int, default=5,
                        help="Newton-Schulz iterations when precond_method=higham.")
    parser.add_argument("--higham_compute_dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Inner-iteration dtype for Higham. fp32 (default) "
                             "preserves current behavior; fp16 opts into variant B "
                             "(n_iters-1 in fp16 + 1 fp32 polish iter). Bench numbers: "
                             "scripts/bench/bench_higham_variants.py.")
    parser.add_argument("--ns_form", type=str, default="rect",
                        choices=["rect", "gram", "gram-norestart"],
                        help="Newton-Schulz kernel form for the clean polar pipeline. "
                             "Only consulted by the chord-tight-clean variant.")
    parser.add_argument("--picard_iters_override", type=int, default=None,
                        help="Override picard_iters. Used for the chord-tight-clean "
                             "variant to compare k=2 vs k=3 at fixed ns_form.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (immaterial for timing; passed through).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-microbatch size for the forward pass. Match production.")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length for the forward pass. Match production.")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Number of forward+backward passes per optimizer.step(). "
                             "Match production (default 8) so the optimizer cost is "
                             "amortized over the right amount of fwd+bwd work.")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_cycles", type=int, default=4,
                        help="Number of full K-step refresh cycles to time per cell. "
                             "Total reps per cell = n_cycles * K, so each cell sees "
                             "exactly n_cycles refresh steps and n_cycles*(K-1) stale "
                             "steps. The reported 'mean_sec_per_step' is the amortized "
                             "cost over these whole cycles — the production-relevant "
                             "metric. For non-Gram-preconditioned optimizers (AdamW), "
                             "n_cycles is interpreted as a flat n_reps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn_implementation", default=None,
                        choices=[None, "eager", "sdpa", "flash_attention_2",
                                 "flash_attention_4"],
                        help="Pass-through to AutoModelForCausalLM.from_pretrained. "
                             "None = HF default. 'flash_attention_2' requires the "
                             "flash-attn package and a supported model. "
                             "'flash_attention_4' requires Hopper/Blackwell, "
                             "transformers >=5, and flash-attn-4 (or kernels-hub).")
    parser.add_argument("--compile", dest="compile_mode", nargs="?",
                        const="default", default=None,
                        choices=[None, "default", "reduce-overhead", "max-autotune"],
                        help="Apply torch.compile after PEFT wrap. Bare --compile "
                             "uses mode='default'. Omit for eager.")
    parser.add_argument("--use_liger", action="store_true",
                        help="Apply Liger Kernel patches (RMSNorm/MLP/RoPE/CE fusion) "
                             "to the base model BEFORE from_pretrained. Family-dispatched "
                             "from --model_name. Requires liger-kernel package.")
    parser.add_argument("--liger_flce", action="store_true",
                        help="With --use_liger, also enable FusedLinearCrossEntropy "
                             "(off in Liger by default). Bigger memory savings; requires "
                             "the model's loss path to use the standard HF labels-in-forward.")
    parser.add_argument("--out", default=None,
                        help="JSONL output path. If omitted, prints to stdout only.")
    parser.add_argument(
        "--data_pipeline_version",
        choices=["packed_v1", "unpacked_v0"],
        default="packed_v1",
        help="Synthetic batch shape. packed_v1 emits a 4D block-diagonal "
             "SDPA mask + per-doc position_ids reset; unpacked_v0 emits "
             "the legacy implicit-causal 2D path. Default packed_v1 (the "
             "new production path).",
    )
    parser.add_argument(
        "--n_docs_per_slot", type=int, default=3,
        help="packed_v1 only: number of equal-length docs per packed slot. "
             "3 reflects Magicoder ~663-token avg at seq=2048.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    # Idempotent DDP init — no-op when launched as a single process; under
    # torchrun (WORLD_SIZE>1) it sets up NCCL and pins the device to
    # LOCAL_RANK so per-rank metadata is correct downstream.
    rank, world_size, local_rank = init_distributed()
    if args.device == "cuda" and world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_main():
        print(f"# Building model {args.model_name} with lora_r={args.lora_r}, "
              f"target_modules={args.target_modules}, dtype={dtype}, "
              f"attn={args.attn_implementation}, compile={args.compile_mode}, "
              f"liger={args.use_liger}, world_size={world_size}", flush=True)
    peft = build_peft_model(
        model_name=args.model_name,
        training_mode="lora",
        target_modules=args.target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        use_liger=args.use_liger,
        liger_flce=args.liger_flce,
        gradient_checkpointing=False,
        compile_mode=args.compile_mode,
        device=device,
        world_size=world_size,
        local_rank=local_rank,
    )
    bare_model = peft.bare_model
    model = peft.train_model
    liger_family = peft.liger_applied
    resolved_attn = peft.resolved_attn or (args.attn_implementation or "default")
    if is_main():
        print(f"# resolved attn_implementation={resolved_attn}", flush=True)
    pairs = collect_lora_pairs(bare_model)
    n_pairs = len(pairs)
    n_lora_params = sum(A.numel() + B.numel() for A, B in pairs)
    if is_main():
        print(f"# {n_pairs} LoRA pairs, {n_lora_params:,} LoRA params", flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    batch = make_batch(
        bare_model, args.batch_size, args.seq_len, device, gen,
        data_pipeline_version=args.data_pipeline_version,
        n_docs_per_slot=args.n_docs_per_slot,
    )
    if is_main():
        print(f"# batch_size={args.batch_size}, seq_len={args.seq_len}, "
              f"grad_accum_steps={args.grad_accum_steps} "
              f"(effective batch = {args.batch_size * args.grad_accum_steps * world_size}, "
              f"world_size={world_size}) "
              f"data_pipeline_version={args.data_pipeline_version}", flush=True)

    # MFU scaffold. c = 4 for LoRA (frozen base, no param-grad pass).
    # Under DDP: each rank does the same flops on its own micro-batch in
    # parallel, so MFU = per-rank achieved / per-rank peak — the world_size
    # factor cancels. We report MFU as per-GPU utilization. Global
    # tokens_per_step (B × S × accum × WS) is the throughput-relevant
    # number for wall-budget math.
    n_total_params = count_total_params(bare_model)
    tokens_per_rank_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    tokens_per_step_global = tokens_per_rank_per_step * world_size
    peak_tflops = device_peak_tflops() if device.type == "cuda" else None
    flops_per_token_per_param = flops_per_token_for_mode(
        "lora", gradient_checkpointing=False,
    )
    if is_main():
        print(f"# n_total_params={n_total_params:,} "
              f"tokens_per_step_global={tokens_per_step_global:,} "
              f"peak_tflops_dense={peak_tflops} "
              f"flops_per_token_per_param={flops_per_token_per_param} (lora, no ckpt)",
              flush=True)

    # Only rank 0 writes the JSONL log (all ranks measure the same numbers
    # under DDP since the model + batch + dtype are identical per rank).
    out_path = Path(args.out) if (args.out and is_main()) else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("a")
    else:
        out_fh = None

    POLAR_OPTIMIZERS = {
        "adam-polar-product-lora",
        "adam-polar-product-lora-coupled",
        "adam-polar-product-lora-coupled-endrms",
        "adam-polar-product-lora-coupled-exact-chord",
        "adam-polar-product-lora-coupled-spectral-chord",
        "adam-polar-product-lora-coupled-spectral-chord-tight",
        "adam-polar-product-lora-coupled-spectral-chord-tight-clean",
        "adamuon-polar-product-lora",
    }
    if is_main():
        print(f"# {'optimizer':<32} {'method':>7} {'K':>4} {'fwd_ms':>8} {'bwd_ms':>8} "
              f"{'opt_ms':>8} {'zero_ms':>8} {'total_ms':>9} {'×AdamW':>8} "
              f"{'MFU':>6}",
              flush=True)
        print(f"#   fwd/bwd are summed across grad_accum_steps microbatches.", flush=True)
        print(f"#   opt is optimizer.step() only. zero is optimizer.zero_grad().", flush=True)
        print(f"#   total = fwd + bwd + opt + zero. ×AdamW = total / AdamW(K=1) total.", flush=True)
        print(f"#   MFU = c·N·tokens_per_rank_per_step / (step_sec × peak_TFLOPS). "
              f"c={flops_per_token_per_param} (lora). None on unknown GPU.",
              flush=True)
    adamw_mean = None
    per_optimizer_k1_mean = {}
    try:
        for opt_name in args.optimizers:
            if opt_name not in OPTIMIZER_CHOICES:
                if is_main():
                    print(f"# skip unknown optimizer: {opt_name}", flush=True)
                continue
            ks = args.precond_refresh_every if opt_name in GRAM_PRECOND_OPTIMIZERS else [1]
            methods = args.precond_method if opt_name in POLAR_OPTIMIZERS else ["eigh"]
            for method in methods:
                for K in ks:
                    # Construct a fresh optimizer per cell so K-stale caches don't
                    # leak across cells. Pass the BARE peft model (no DDP wrap)
                    # so collect_lora_pairs can find lora_A/lora_B without
                    # tripping on the `module.` prefix DDP adds.
                    optimizer = build_optimizer(
                        bare_model,
                        optimizer_type=opt_name,
                        lr=args.lr,
                        precond_refresh_every=K,
                        precond_method=method,
                        higham_iters=args.higham_iters,
                        higham_compute_dtype=args.higham_compute_dtype,
                        ns_form=args.ns_form,
                        picard_iters_override=args.picard_iters_override,
                    )
                    n_reps = args.n_cycles * K
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    per_step, is_refresh, phase_times = time_full_step(
                        model, optimizer, batch, args.n_warmup, n_reps, K,
                        args.grad_accum_steps, device,
                    )
                    fwd_mean = sum(phase_times["fwd"]) / len(phase_times["fwd"])
                    bwd_mean = sum(phase_times["bwd"]) / len(phase_times["bwd"])
                    opt_mean_phase = sum(phase_times["opt"]) / len(phase_times["opt"])
                    zero_mean = sum(phase_times["zero"]) / len(phase_times["zero"])
                    refresh_times = [t for t, r in zip(per_step, is_refresh) if r]
                    stale_times = [t for t, r in zip(per_step, is_refresh) if not r]
                    mean_amortized = sum(per_step) / len(per_step)
                    refresh_mean = (sum(refresh_times) / len(refresh_times)
                                    if refresh_times else float("nan"))
                    stale_mean = (sum(stale_times) / len(stale_times)
                                  if stale_times else float("nan"))
                    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                               if device.type == "cuda" else 0.0)
                    cell_key = (opt_name, method)
                    if opt_name == "adamw" and K == 1:
                        adamw_mean = mean_amortized
                    if K == 1:
                        per_optimizer_k1_mean[cell_key] = mean_amortized
                    k1_mean = per_optimizer_k1_mean.get(cell_key)
                    speedup_vs_k1 = (k1_mean / mean_amortized) if k1_mean else float("nan")
                    ratio_vs_adamw = (mean_amortized / adamw_mean) if adamw_mean else float("nan")

                    mfu = compute_mfu(
                        n_params=n_total_params,
                        tokens_per_step=tokens_per_rank_per_step,
                        step_time_sec=mean_amortized,
                        peak_tflops=peak_tflops,
                        flops_per_token_per_param=flops_per_token_per_param,
                    ) if peak_tflops is not None else None
                    mfu_str = f"{mfu*100:>5.1f}%" if mfu is not None else "  --  "
                    if is_main():
                        print(f"  {opt_name:<32} {method:>7} {K:>4} "
                              f"{fwd_mean*1000:>8.1f} {bwd_mean*1000:>8.1f} "
                              f"{opt_mean_phase*1000:>8.1f} {zero_mean*1000:>8.1f} "
                              f"{mean_amortized*1000:>9.1f} {ratio_vs_adamw:>7.2f}× "
                              f"{mfu_str:>6}",
                              flush=True)

                    rec = {
                        "event": "bench_step",
                        "optimizer": opt_name,
                        "precond_method": method,
                        "lora_r": args.lora_r,
                        "precond_refresh_every": K,
                        "mean_sec_per_step": mean_amortized,
                        "refresh_sec_per_step": refresh_mean,
                        "stale_sec_per_step": stale_mean,
                        "n_refresh_steps": len(refresh_times),
                        "n_stale_steps": len(stale_times),
                        "n_reps": n_reps,
                        "n_warmup": args.n_warmup,
                        "n_cycles": args.n_cycles,
                        "n_pairs": n_pairs,
                        "n_lora_params": n_lora_params,
                        "model_name": args.model_name,
                        "target_modules": args.target_modules,
                        "dtype": str(dtype),
                        "device": str(device),
                        "speedup_vs_k1": speedup_vs_k1,
                        "ratio_vs_adamw_step_only": ratio_vs_adamw,
                        "fwd_sec_per_step": fwd_mean,
                        "bwd_sec_per_step": bwd_mean,
                        "opt_sec_per_step": opt_mean_phase,
                        "zero_sec_per_step": zero_mean,
                        "peak_memory_mb": peak_mb,
                        "attn_implementation": resolved_attn,
                        "compile_mode": args.compile_mode,
                        "liger_family": liger_family,
                        "liger_flce": args.liger_flce if args.use_liger else None,
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "grad_accum_steps": args.grad_accum_steps,
                        "data_pipeline_version": args.data_pipeline_version,
                        "n_docs_per_slot": (
                            args.n_docs_per_slot
                            if args.data_pipeline_version == "packed_v1" else None
                        ),
                        "mfu": mfu,
                        "mfu_peak_tflops": peak_tflops,
                        "mfu_n_params": n_total_params,
                        "mfu_flops_per_token_per_param": flops_per_token_per_param,
                        "tokens_per_rank_per_step": tokens_per_rank_per_step,
                        "tokens_per_step_global": tokens_per_step_global,
                        "world_size": world_size,
                    }
                    if out_fh is not None:
                        out_fh.write(json.dumps(rec, sort_keys=True) + "\n")
                        out_fh.flush()

                    # Free optimizer state before the next cell.
                    del optimizer
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        if out_fh is not None:
            out_fh.close()
        # Tear down the distributed process group if we initialized one
        # (no-op in single-process mode).
        dist_cleanup()


if __name__ == "__main__":
    main()
