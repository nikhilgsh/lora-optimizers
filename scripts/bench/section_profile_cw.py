"""Section-level breakdown of CurvatureWhitenLoRA(use_polar).step at r256.

Attaches a CudaTimer (`opt._step_timer`) so the `cw_*` scopes inside
`_cw_apply_grouped` / the refresh emit per-section CUDA times. Runs a full
refresh cycle so `cw_refresh` (1-in-K) is captured alongside the steady
sections. Answers: of the ~124 ms steady step, which algebraic block dominates?

Usage (GPU): python scripts/bench/section_profile_cw.py [--refresh_every 10]
"""
import argparse, statistics as st, time
import torch

from lora_playground.training_kernel import build_peft_model
from lora_playground.optim import build_optimizer
from lora_playground._step_timer import CudaTimer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="curvature-whiten-polar-lora")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--refresh_every", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=30)  # ≥ refresh_every*2 to amortize
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
    opt = build_optimizer(
        bare, optimizer_type=args.optimizer, lr=1e-3,
        precond_refresh_every=args.refresh_every, precond_delta=1e-3, curvature_beta=0.99,
    )
    n_pairs = len(opt.pairs)
    print(f"# {args.optimizer}  lora_r={args.lora_r}  {n_pairs} pairs  refresh_every={args.refresh_every}", flush=True)
    ids = torch.randint(0, bare.config.vocab_size, (1, 128), device=dev)

    def make_grads():
        opt.zero_grad(set_to_none=False)
        model(input_ids=ids, labels=ids).loss.backward()

    for _ in range(args.warmup):
        make_grads(); opt.step()
    torch.cuda.synchronize()

    # Whole-step wall (for cross-check that sections sum to ~the step).
    times = []
    for _ in range(args.steps):
        make_grads(); torch.cuda.synchronize()
        t0 = time.perf_counter(); opt.step(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"# opt.step() ms over {len(times)}: mean={st.mean(times):.1f} "
          f"min={min(times):.1f} max={max(times):.1f}", flush=True)

    # Section pass with the CudaTimer attached.
    opt._step_timer = CudaTimer(dev)
    for _ in range(args.steps):
        make_grads(); opt.step()
    summ = opt._step_timer.summary()
    print(f"# --- section breakdown over {args.steps} steps (total_ms / n_calls = per-occurrence) ---", flush=True)
    rows = sorted(summ.items(), key=lambda kv: -kv[1]["total_ms"])
    grand = sum(v["total_ms"] for k, v in summ.items() if k != "cw_pairstep")
    for name, s in rows:
        per_step = s["total_ms"] / args.steps
        share = "" if name == "cw_pairstep" else f"  {100*s['total_ms']/grand:5.1f}% of pairstep-sections"
        print(f"  {name:18s} total={s['total_ms']:8.1f}ms  n={int(s['n']):3d}  "
              f"mean/call={s['mean_ms']:6.2f}ms  per_step={per_step:6.2f}ms{share}", flush=True)


if __name__ == "__main__":
    main()
