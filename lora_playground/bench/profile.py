"""Op-level profiling of ``optimizer.step()`` for a ``BenchConfig`` — locate the
cost (which op / per-pair Python loop dominates) and whether the step is
launch-overhead-bound vs compute-bound.

Consolidates ``scripts/bench/profile_curvature_whiten.py``. Uses the harness model
build + production optimizer kwargs so the profiled path is the real one (the
CLAUDE.md trap: build_optimizer defaults precond_method=eigh, a slow per-pair
fallback; the protagonist/chord-tight production path is different).
"""
from __future__ import annotations

import statistics as st
import time

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile

from .config import BenchConfig
from .harness import build_model
from ..optim import build_optimizer


def profile_step(cfg: BenchConfig, *, precond_method=None, refresh_every=None,
                 warmup=15, steps=12, device=None, seq_len=128, row_limit=15) -> dict:
    """Time ``optimizer.step()`` over a refresh cycle (mean/min≈steady/max≈refresh)
    and print a torch.profiler op table. Step cost is dim/rank-bound, not
    batch/seq-bound, so a cheap fwd/bwd at ``seq_len`` is enough to make grads."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = refresh_every if refresh_every is not None else cfg.precond_refresh_every
    print("# " + cfg.banner(), flush=True)
    peft = build_model(cfg, device)
    bare, model = peft.bare_model, peft.train_model

    kw = cfg.build_optimizer_kwargs(precond_method=precond_method)
    kw["precond_refresh_every"] = K
    opt = build_optimizer(bare, optimizer_type=cfg.optimizer, lr=1e-3, **kw)
    print(f"# precond_method={kw['precond_method']} polar_method={kw['polar_method']} "
          f"ns_steps={kw['muon_ns_steps']} refresh_every={K}", flush=True)

    ids = torch.randint(0, bare.config.vocab_size, (1, seq_len), device=device)

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
    print(f"# opt.step() ms over {len(times)}: mean={st.mean(times):.1f}  "
          f"min={min(times):.1f}(steady)  max={max(times):.1f}(refresh)", flush=True)

    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            make_grads(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    ka = prof.key_averages()

    def cuda_t(k):
        return getattr(k, "self_device_time_total", None) or getattr(k, "self_cuda_time_total", 0)
    tot_cpu = sum(k.self_cpu_time_total for k in ka) / 1000.0
    tot_cuda = sum(cuda_t(k) for k in ka) / 1000.0
    print(f"# self-CPU(launch)={tot_cpu:.1f}ms  self-CUDA={tot_cuda:.1f}ms  "
          f"{'LAUNCH-BOUND' if tot_cpu > tot_cuda else 'compute-bound'}", flush=True)
    try:
        print(ka.table(sort_by="self_cuda_time_total", row_limit=row_limit))
    except Exception:  # noqa: BLE001
        print(ka.table(sort_by="self_cpu_time_total", row_limit=row_limit))
    return {"mean_ms": st.mean(times), "min_ms": min(times), "max_ms": max(times),
            "self_cpu_ms": tot_cpu, "self_cuda_ms": tot_cuda}
