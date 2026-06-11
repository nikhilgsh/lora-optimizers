"""Full-step wall-time harness: time (fwd + bwd + optimizer.step + zero_grad)
for a ``BenchConfig``, at the production config, with timing hygiene enforced.

Why end-to-end (not opt.step in isolation): AdamW's step is a trivial add_ while
the Gram optimizers do real per-step work, so an isolated opt-step ratio
overstates the production overhead. The number that matters is the additive
overhead on top of the (optimizer-invariant) fwd+bwd.

``make_batch`` / ``time_full_step`` are migrated verbatim from the original
``scripts/bench/bench_optimizer_step.py`` (now a thin shim over this module) so
the measured numbers are continuous with the historical bench logs.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import torch

from .config import BenchConfig
from ..data import build_block_diagonal_causal_mask
from ..mfu import (
    compute_mfu, count_total_params, device_peak_tflops, flops_per_token_for_mode,
)
from ..optim import build_optimizer
from ..training_kernel import build_peft_model
from ..utils import collect_lora_pairs


def make_batch(model, batch_size, seq_len, device, generator, *,
               data_pipeline_version="packed_v1", n_docs_per_slot=3):
    """Random token-ID batch. ``packed_v1`` emits the production 4D
    block-diagonal SDPA mask + per-doc position_ids reset (captures the extra
    kernel cost vs the legacy implicit-causal path)."""
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                              device=device, generator=generator)
    if data_pipeline_version == "unpacked_v0":
        return {"input_ids": input_ids, "labels": input_ids}
    if data_pipeline_version != "packed_v1":
        raise ValueError(f"unknown data_pipeline_version: {data_pipeline_version}")
    n_docs = max(1, n_docs_per_slot)
    base = seq_len // n_docs
    rem = seq_len - base * n_docs
    doc_lens = [base + (1 if i < rem else 0) for i in range(n_docs)]
    pos: list[int] = []
    for L in doc_lens:
        pos.extend(range(L))
    position_ids = (torch.tensor(pos, device=device, dtype=torch.long)
                    .unsqueeze(0).expand(batch_size, -1).contiguous())
    param_dtype = next(model.parameters()).dtype
    mask_dtype = param_dtype if param_dtype.is_floating_point else torch.float32
    mask_1 = build_block_diagonal_causal_mask(doc_lens, seq_len, mask_dtype, device)
    attention_mask = mask_1.expand(batch_size, 1, seq_len, seq_len).contiguous()
    return {"input_ids": input_ids, "labels": input_ids,
            "position_ids": position_ids, "attention_mask": attention_mask}


def time_full_step(model, optimizer, batch, n_warmup, n_reps, K, grad_accum_steps, device):
    """Time ``n_reps`` full training steps. Returns (per_step_times, is_refresh,
    phase_times) — phase_times has lists for fwd/bwd/opt/zero, each step timed
    with cuda.synchronize() around it; refresh-vs-stale classified from the
    optimizer's internal step counter."""
    has_pair_state = hasattr(optimizer, "pair_state") and 0 in optimizer.pair_state
    is_cuda = (device.type == "cuda")

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    def one_step_timed():
        fwd_ms = bwd_ms = 0.0
        for _ in range(grad_accum_steps):
            sync(); t0 = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            sync(); fwd_ms += time.perf_counter() - t0
            t0 = time.perf_counter()
            loss.backward()
            sync(); bwd_ms += time.perf_counter() - t0
        t0 = time.perf_counter(); optimizer.step(); sync()
        opt_ms = time.perf_counter() - t0
        t0 = time.perf_counter(); optimizer.zero_grad(set_to_none=False); sync()
        zero_ms = time.perf_counter() - t0
        return fwd_ms, bwd_ms, opt_ms, zero_ms

    def one_step_untimed():
        for _ in range(grad_accum_steps):
            outputs = model(**batch)
            (outputs.loss / grad_accum_steps).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=False)

    for _ in range(n_warmup):
        sync(); one_step_untimed(); sync()

    per_step, is_refresh = [], []
    phase_times = {"fwd": [], "bwd": [], "opt": [], "zero": []}
    for _ in range(n_reps):
        sync()
        this_is_refresh = (optimizer.pair_state[0]['step'] % K == 0) if has_pair_state else False
        fwd_ms, bwd_ms, opt_ms, zero_ms = one_step_timed()
        per_step.append(fwd_ms + bwd_ms + opt_ms + zero_ms)
        for k, v in zip(("fwd", "bwd", "opt", "zero"), (fwd_ms, bwd_ms, opt_ms, zero_ms)):
            phase_times[k].append(v)
        is_refresh.append(this_is_refresh)
    return per_step, is_refresh, phase_times


def build_protagonist_direct(bare, cfg: BenchConfig, *, precond_method, K):
    """Build ``CurvatureWhitenLoRA`` directly with an EXPLICIT precond_method.

    The factory cannot do this today: the cw protagonist specs skip
    ``precond_method`` (``_CW_PRECOND_SKIP``) because ``build_optimizer``'s default
    is ``"higham"`` — wrong for cw — so an explicit ``precond_method="gram_ns"`` is
    silently dropped to the class default ``eigh`` (verified). This builds the
    optimizer exactly as the spec would once gram_ns is adopted via ``fixed``
    (identity flags pulled from the spec so they cannot drift), with the chosen
    inverse-sqrt method actually in effect — the faithful retiming measurement.
    """
    from ..optim import CurvatureWhitenLoRA
    from ..optim_specs import REGISTRY
    identity = dict(REGISTRY[cfg.optimizer].fixed)  # kl_coupled/soap_v/diag_metric/use_polar
    return CurvatureWhitenLoRA(
        bare, lr=1e-3, betas=(cfg.beta1, cfg.beta2), delta=cfg.precond_delta,
        curvature_beta=cfg.curvature_beta, ns_steps=cfg.muon_ns_steps,
        polar_method=cfg.polar_method, cw_picard_iters=cfg.cw_picard_iters,
        cw_nesterov=cfg.cw_nesterov, precond_refresh_every=K,
        precond_method=precond_method, higham_iters=cfg.higham_iters, **identity)


def build_model(cfg: BenchConfig, device):
    """Build the PEFT model for ``cfg`` (production dtype/compile/target_modules)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    peft = build_peft_model(
        model_name=cfg.model_name, training_mode="lora",
        target_modules=cfg.target_modules, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0, dtype=dtype, attn_implementation=None,
        use_liger=False, liger_flce=False, gradient_checkpointing=False,
        compile_mode=cfg.compile_mode, device=device, world_size=1, local_rank=0,
    )
    return peft


def measure_step_wall(peft, cfg: BenchConfig, *, precond_method: str, K: int,
                      device, n_warmup=3, n_cycles=4, lr=1e-3,
                      adamw_mean: Optional[float] = None) -> dict:
    """One cell: build the optimizer (cfg's production flags + the given
    precond_method/K), time the full step, return a result dict ready for
    ``registry.record_bench`` and the walltime-profile docs."""
    bare = peft.bare_model
    model = peft.train_model
    pairs = collect_lora_pairs(bare)
    gen = torch.Generator(device=device).manual_seed(0)
    batch = make_batch(bare, cfg.batch_size, cfg.max_seq_length, device, gen,
                       data_pipeline_version=cfg.data_pipeline_version)

    kw = cfg.build_optimizer_kwargs(precond_method=precond_method)
    kw["precond_refresh_every"] = K
    optimizer = build_optimizer(bare, optimizer_type=cfg.optimizer, lr=lr, **kw)

    n_reps = n_cycles * K
    if device.type == "cuda":
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    per_step, is_refresh, phase = time_full_step(
        model, optimizer, batch, n_warmup, n_reps, K, cfg.grad_accum_steps, device)

    mean = sum(per_step) / len(per_step)
    refresh = [t for t, r in zip(per_step, is_refresh) if r]
    stale = [t for t, r in zip(per_step, is_refresh) if not r]
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == "cuda" else 0.0)
    tokens_per_step = cfg.batch_size * cfg.max_seq_length * cfg.grad_accum_steps
    peak_tflops = device_peak_tflops() if device.type == "cuda" else None
    mfu = (compute_mfu(n_params=count_total_params(bare), tokens_per_step=tokens_per_step,
                       step_time_sec=mean, peak_tflops=peak_tflops,
                       flops_per_token_per_param=flops_per_token_for_mode("lora"))
           if peak_tflops else None)
    res = {
        **{k: v for k, v in asdict(cfg).items()
           if k in ("model_name", "lora_r", "optimizer", "muon_ns_steps",
                    "cw_picard_iters", "batch_size", "grad_accum_steps",
                    "max_seq_length", "compile_mode", "bf16")},
        "precond_method": precond_method,
        "precond_refresh_every": K,
        "mean_sec_per_step": mean,
        "refresh_sec_per_step": (sum(refresh) / len(refresh)) if refresh else float("nan"),
        "stale_sec_per_step": (sum(stale) / len(stale)) if stale else float("nan"),
        "opt_sec_per_step": sum(phase["opt"]) / len(phase["opt"]),
        "fwd_sec_per_step": sum(phase["fwd"]) / len(phase["fwd"]),
        "bwd_sec_per_step": sum(phase["bwd"]) / len(phase["bwd"]),
        "n_reps": n_reps, "n_pairs": len(pairs), "peak_memory_mb": peak_mb, "mfu": mfu,
        "ratio_vs_adamw": (mean / adamw_mean) if adamw_mean else None,
    }
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def run_step_wall(cfg: BenchConfig, *, methods=("eigh",), ks=None,
                  device=None, n_warmup=3, n_cycles=4) -> list[dict]:
    """Build the model ONCE, then time every (precond_method, K) cell. The
    canonical entry for the QR-vs-Higham retiming: ``methods=('eigh','higham')``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = ks or [cfg.precond_refresh_every]
    print("# " + cfg.banner(), flush=True)
    peft = build_model(cfg, device)
    results = []
    for method in methods:
        for K in ks:
            r = measure_step_wall(peft, cfg, precond_method=method, K=K,
                                  device=device, n_warmup=n_warmup, n_cycles=n_cycles)
            print(f"#   method={method:>7} K={K:>3}  "
                  f"step={r['mean_sec_per_step']*1000:8.1f} ms  "
                  f"opt={r['opt_sec_per_step']*1000:7.1f} ms  "
                  f"peak={r['peak_memory_mb']:.0f} MB", flush=True)
            results.append(r)
    return results
