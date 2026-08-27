"""Semantic per-operation breakdown of the protagonist `optimizer.step()` —
the appendix table behind the ×AdamW wall numbers (main.tex §sec:exp-walltime).

Attaches a `CudaTimer` so the `cw_*` scopes in `_cw_apply_grouped` emit per-section
CUDA time, swept over precond_method ∈ {eigh, gram_ns}. The step cost is batch/seq-
independent, so a tiny fwd/bwd (eager, seq=128) suffices to make gradients; runs a
full refresh cycle so the 1-in-K `cw_refresh` (eigh) is amortized correctly.

Scopes:
  cw_refresh     batched QR eigenbasis refresh (eigh only; absent for gram_ns)
  cw_basis_proj  the inverse-sqrt itself (eigh: Rayleigh+QᵀsandwiCH; gram_ns: gram NS)
                 + SOAP/diagonal projections + Adam m,v EMAs
  cw_picard      whiten → polar (Gram NS msign) → unwhiten → σ_max operator-norm rescale
  cw_curv_grams  curvature gram / diagonal EMAs (P_A, Q_B, D_in, D_out)

Run (Blackwell): python -m lora_playground.bench.section_breakdown \
    --model allenai/OLMo-2-0425-1B --lora_r 256 --methods eigh gram_ns
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch

from .config import protagonist_config
from .harness import build_model, build_protagonist_direct
from .._step_timer import CudaTimer
from ..utils import collect_lora_pairs


def _breakdown_one(cfg, bare, model, ids, *, precond_method, K, higham_iters,
                   warmup, steps, device):
    cfg.higham_iters = higham_iters
    opt = build_protagonist_direct(bare, cfg, precond_method=precond_method, K=K)
    assert opt.precond_method == precond_method, \
        f"precond_method not in effect: wanted {precond_method}, got {opt.precond_method}"

    def make_grads():
        opt.zero_grad(set_to_none=False)
        model(input_ids=ids, labels=ids).loss.backward()

    for _ in range(warmup):
        make_grads(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        make_grads()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean_step = st.mean(times)

    opt._step_timer = CudaTimer(device)
    for _ in range(steps):
        make_grads(); opt.step()
    summ = opt._step_timer.summary()

    print(f"\n## precond_method={precond_method}  opt.step mean={mean_step:.1f} ms "
          f"(min={min(times):.1f} steady / max={max(times):.1f} refresh) over {steps} steps",
          flush=True)
    grand = sum(v["total_ms"] for k, v in summ.items() if k != "cw_pairstep")
    print(f"   {'scope':<16}{'per_step ms':>12}{'share':>8}{'n/step':>8}", flush=True)
    for name, s in sorted(summ.items(), key=lambda kv: -kv[1]["total_ms"]):
        if name == "cw_pairstep":
            continue
        per_step = s["total_ms"] / steps
        print(f"   {name:<16}{per_step:>12.2f}{100*s['total_ms']/grand:>7.1f}%"
              f"{s['n']/steps:>8.1f}", flush=True)
    del opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"precond_method": precond_method, "mean_step_ms": mean_step,
            "sections": {k: v["total_ms"] / steps for k, v in summ.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--methods", nargs="+", default=["eigh", "gram_ns"],
                    choices=["eigh", "gram_ns", "higham"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--higham_iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = protagonist_config(args.model, args.lora_r)
    cfg.compile_mode = None   # section profiling isolates opt.step; eager fwd/bwd is fine
    print(f"# {cfg.banner()}", flush=True)
    print(f"# section breakdown: K={args.K} higham_iters={args.higham_iters} "
          f"steps={args.steps}", flush=True)
    peft = build_model(cfg, device)
    bare, model = peft.bare_model, peft.train_model
    print(f"# n_pairs={len(collect_lora_pairs(bare))}", flush=True)
    ids = torch.randint(0, bare.config.vocab_size, (1, args.seq_len), device=device)

    for method in args.methods:
        _breakdown_one(cfg, bare, model, ids, precond_method=method, K=args.K,
                       higham_iters=args.higham_iters, warmup=args.warmup,
                       steps=args.steps, device=device)


if __name__ == "__main__":
    main()
