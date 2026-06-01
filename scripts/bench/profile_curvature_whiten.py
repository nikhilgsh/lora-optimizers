"""Profile CurvatureWhitenLoRA(use_polar=True).step to locate the cost.

The optimizer step cost depends only on lora_r and the per-layer dims (not on
batch/seq), so we make grads with a cheap fwd/bwd and profile the real step.

Outputs concrete numbers:
  - opt.step() wall ms per step (mean/min/max over a full refresh cycle;
    max ≈ a refresh step, min ≈ steady step)
  - torch.profiler op table: Self CUDA, Self CPU (launch), and #Calls
    (112-ish call counts reveal the per-pair Python loop; CPU≫CUDA ⇒ the step
    is launch-overhead-bound, which is what batching fixes).

Usage (on a GPU): python scripts/bench/profile_curvature_whiten.py
"""
import argparse, statistics as st, time
import torch
from torch.profiler import profile, ProfilerActivity

from lora_playground.training_kernel import build_peft_model
from lora_playground.optim import build_optimizer
from lora_playground.utils import collect_lora_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="curvature-whiten-polar-lora")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--refresh_every", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    dev = torch.device("cuda")
    peft = build_peft_model(
        model_name="allenai/OLMo-2-0425-1B", training_mode="lora",
        target_modules="all-linear", lora_r=args.lora_r, lora_alpha=args.lora_r,
        lora_dropout=0.0, dtype=torch.bfloat16, attn_implementation="sdpa",
        use_liger=False, liger_flce=False, gradient_checkpointing=False,
        compile_mode=None, device=dev, world_size=1, local_rank=0,
    )
    model, bare = peft.train_model, peft.bare_model
    pairs = collect_lora_pairs(bare)
    print(f"# {args.optimizer}  lora_r={args.lora_r}  {len(pairs)} pairs  refresh_every={args.refresh_every}", flush=True)
    opt = build_optimizer(
        bare, optimizer_type=args.optimizer, lr=1e-3,
        precond_refresh_every=args.refresh_every, precond_delta=1e-3, curvature_beta=0.99,
    )
    ids = torch.randint(0, bare.config.vocab_size, (1, 128), device=dev)

    def make_grads():
        opt.zero_grad(set_to_none=False)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()

    for _ in range(args.warmup):
        make_grads(); opt.step()
    torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        make_grads(); torch.cuda.synchronize()
        t0 = time.perf_counter(); opt.step(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"# opt.step() ms over {len(times)} steps: mean={st.mean(times):.1f}  "
          f"min={min(times):.1f} (steady)  max={max(times):.1f} (refresh)", flush=True)

    make_grads()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        opt.step(); torch.cuda.synchronize()
    ka = prof.key_averages()
    def cuda_t(k):
        return getattr(k, "self_device_time_total", None) or getattr(k, "self_cuda_time_total", 0)
    tot_cpu = sum(k.self_cpu_time_total for k in ka) / 1000.0
    tot_cuda = sum(cuda_t(k) for k in ka) / 1000.0
    print(f"# ONE step: self-CPU(launch) total={tot_cpu:.1f}ms   self-CUDA total={tot_cuda:.1f}ms   "
          f"{'LAUNCH-BOUND' if tot_cpu > tot_cuda else 'compute-bound'}", flush=True)
    try:
        print(ka.table(sort_by="self_cuda_time_total", row_limit=15))
    except Exception:
        print(ka.table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
