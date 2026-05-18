"""CUDA-graphs prototype: measure speedup of capturing
`spd_inv_sqrt_higham_batched` and `_newton_schulz_gram_batched` as
CUDA graphs vs the default per-kernel-launch path.

Both functions are good graph-capture candidates: fixed shapes, no
host-side branches, no `.item()` / sync points in the inner loop. The
question is how much launch overhead is hiding in those scopes at
production (N, r, r) shapes on Blackwell.

Method:
  1. Time the function N times in the normal eager/scripted way.
  2. Allocate static input + output buffers.
  3. Warm up on a side stream (kernel selection + memory pool primed).
  4. Capture with `torch.cuda.graph()`.
  5. Replay N times (copy_ new input in, replay, read output out).
  6. Compare ms per call.
"""
import sys, time
sys.path.insert(0, "/mnt/home/nghosh/lora")

import torch
from lora_playground.utils import spd_inv_sqrt_higham_batched
from lora_playground.optim import _newton_schulz_gram_batched

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")


def bench_eager(fn_call, n_repeats=50, warmup=10):
    for _ in range(warmup):
        out = fn_call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        out = fn_call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats * 1000


def bench_graphed(static_in, static_out, replay_graph, refresh_input,
                  n_repeats=50, warmup=10):
    for _ in range(warmup):
        refresh_input()
        replay_graph()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        refresh_input()
        replay_graph()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats * 1000


def make_spd(N, r, log_cond=2.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sing = torch.logspace(0, -log_cond / 2, r).pow(2)
    H_list = []
    for _ in range(N):
        M = torch.randn(r, r, generator=g)
        U, _ = torch.linalg.qr(M)
        H_list.append(U @ torch.diag(sing) @ U.transpose(-2, -1))
    return torch.stack(H_list).to(device).float()


def make_rect(N, r, d, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(N, r, d, generator=g).to(device).float()


# ─── Higham ─────────────────────────────────────────────────────────────────


def bench_higham(N, r, compute_dtype):
    H = make_spd(N, r)
    eager_call = lambda: spd_inv_sqrt_higham_batched(
        H, n_iters=10, eps=1e-2, eps_relative=True, compute_dtype=compute_dtype,
    )
    t_eager = bench_eager(eager_call)

    # Graph capture
    static_in = torch.zeros_like(H)
    static_in.copy_(H)

    # Warmup on a side stream so allocator and kernel selection happen
    # before capture (otherwise capture errors on first-time kernel select).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = spd_inv_sqrt_higham_batched(
                static_in, n_iters=10, eps=1e-2, eps_relative=True,
                compute_dtype=compute_dtype,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = spd_inv_sqrt_higham_batched(
            static_in, n_iters=10, eps=1e-2, eps_relative=True,
            compute_dtype=compute_dtype,
        )

    def refresh():
        static_in.copy_(H)
    t_graph = bench_graphed(static_in, static_out, g.replay, refresh)
    return t_eager, t_graph


# ─── Gram-NS polar map ──────────────────────────────────────────────────────


def bench_gram_ns(N, r, d):
    X = make_rect(N, r, d)
    eager_call = lambda: _newton_schulz_gram_batched(
        X, nsteps=5, restart_at=2,
    )
    t_eager = bench_eager(eager_call)

    static_in = torch.zeros_like(X)
    static_in.copy_(X)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = _newton_schulz_gram_batched(static_in, nsteps=5, restart_at=2)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = _newton_schulz_gram_batched(
            static_in, nsteps=5, restart_at=2,
        )

    def refresh():
        static_in.copy_(X)
    t_graph = bench_graphed(static_in, static_out, g.replay, refresh)
    return t_eager, t_graph


# ─── Run ────────────────────────────────────────────────────────────────────


print(f"\n=== spd_inv_sqrt_higham_batched (eps_rel=1e-2 production setting) ===")
print(f"{'shape (N,r,r)':>15s} | {'dtype':>10s} | {'eager ms':>10s} | "
      f"{'graph ms':>10s} | {'speedup':>10s}")
print("-" * 70)
for N, r in [(112, 16), (112, 64), (112, 128), (224, 256)]:
    for dtype, label in [(None, "fp32"), (torch.float16, "fp16+pol")]:
        t_e, t_g = bench_higham(N, r, dtype)
        print(f"({N:4d},{r:4d},{r:4d}) | {label:>10s} | {t_e:10.3f} | "
              f"{t_g:10.3f} | {t_e/t_g:9.2f}x")

print(f"\n=== _newton_schulz_gram_batched (Dao Algorithm 3, fp16+restart) ===")
print(f"{'shape (N,r,d)':>20s} | {'eager ms':>10s} | {'graph ms':>10s} | "
      f"{'speedup':>10s}")
print("-" * 65)
# Production shapes for the polar map: (N, r, d_in or d_out) where d ≈ 2048 for OLMo MLP.
for N, r, d in [(112, 16, 2048), (112, 64, 2048), (112, 128, 2048), (224, 256, 2048)]:
    t_e, t_g = bench_gram_ns(N, r, d)
    print(f"({N:4d},{r:4d},{d:4d}) | {t_e:10.3f} | {t_g:10.3f} | "
          f"{t_e/t_g:9.2f}x")
