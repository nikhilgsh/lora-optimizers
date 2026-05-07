"""Microbench: per-pair Adam moment update / apply, looped vs torch._foreach_*.

`AdamPolarProductLoRA._adam_direction` runs per-pair in a Python loop:

    state['m_A'].mul_(beta1).add_(gA, alpha=1.0 - beta1)
    state['v_A'].mul_(beta2).addcmul_(gA, gA, value=1.0 - beta2)
    ... etc

Each in-place op launches a kernel. Across 112 pairs × ~6 ops per side ×
2 sides = ~1300 launches just for the moment update. PyTorch's foreach
ops (`torch._foreach_mul_`, `torch._foreach_add_`, `torch._foreach_addcmul_`)
take a list of tensors and do all updates in one launch.

Hypothesis: the 17 ms `adam_direction` measured in the profile is launch-
bound just like NS was. Foreach should drop it to ~2 ms, matching the
launch-cost ratio of stock `torch.optim.AdamW` (which uses foreach by
default and step-times at 1.5 ms).
"""
import argparse
import time

import torch


def adam_direction_loop(states, gA_list, gB_list, beta1, beta2, eps, step):
    """Mirrors AdamPolarProductLoRA._adam_direction at the per-pair level."""
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    u_As, u_Bs = [], []
    for state, gA, gB in zip(states, gA_list, gB_list):
        state['m_A'].mul_(beta1).add_(gA, alpha=1.0 - beta1)
        state['m_B'].mul_(beta1).add_(gB, alpha=1.0 - beta1)
        state['v_A'].mul_(beta2).addcmul_(gA, gA, value=1.0 - beta2)
        state['v_B'].mul_(beta2).addcmul_(gB, gB, value=1.0 - beta2)
        u_A = (state['m_A'] / bc1) / ((state['v_A'] / bc2).sqrt() + eps)
        u_B = (state['m_B'] / bc1) / ((state['v_B'] / bc2).sqrt() + eps)
        u_As.append(u_A)
        u_Bs.append(u_B)
    return u_As, u_Bs


def adam_direction_foreach(m_As, m_Bs, v_As, v_Bs, gA_list, gB_list,
                           beta1, beta2, eps, step):
    """Same math, foreach-batched across pairs.

    State is held as flat lists of tensors per kind, not as 112 dict lookups
    — the optimizer would maintain these list-of-tensors structures alongside
    or instead of pair_state for the foreach path.
    """
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    # In-place EMA updates. _foreach_mul_(list, scalar); _foreach_add_(list, list, alpha).
    torch._foreach_mul_(m_As, beta1)
    torch._foreach_add_(m_As, gA_list, alpha=1.0 - beta1)
    torch._foreach_mul_(m_Bs, beta1)
    torch._foreach_add_(m_Bs, gB_list, alpha=1.0 - beta1)
    torch._foreach_mul_(v_As, beta2)
    torch._foreach_addcmul_(v_As, gA_list, gA_list, value=1.0 - beta2)
    torch._foreach_mul_(v_Bs, beta2)
    torch._foreach_addcmul_(v_Bs, gB_list, gB_list, value=1.0 - beta2)
    # Bias-corrected output: u = (m / bc1) / (sqrt(v / bc2) + eps).
    # No in-place foreach for div+sqrt+add chain, so we build it via the
    # functional foreach ops; each call is one kernel launch over the list.
    m_hat_A = torch._foreach_div(m_As, bc1)
    m_hat_B = torch._foreach_div(m_Bs, bc1)
    v_hat_A = torch._foreach_div(v_As, bc2)
    v_hat_B = torch._foreach_div(v_Bs, bc2)
    denom_A = torch._foreach_sqrt(v_hat_A)
    denom_B = torch._foreach_sqrt(v_hat_B)
    torch._foreach_add_(denom_A, eps)
    torch._foreach_add_(denom_B, eps)
    u_A = torch._foreach_div(m_hat_A, denom_A)
    u_B = torch._foreach_div(m_hat_B, denom_B)
    return u_A, u_B


def equivalence_check(N_pairs, shapes, device, atol=1e-5):
    torch.manual_seed(0)
    states = []
    m_As, m_Bs, v_As, v_Bs = [], [], [], []
    gA_list, gB_list = [], []
    for (r, d_in, d_out) in shapes:
        m_A = torch.randn(r, d_in, device=device, dtype=torch.float32) * 0.01
        m_B = torch.randn(d_out, r, device=device, dtype=torch.float32) * 0.01
        v_A = torch.rand(r, d_in, device=device, dtype=torch.float32) * 0.001
        v_B = torch.rand(d_out, r, device=device, dtype=torch.float32) * 0.001
        states.append({"m_A": m_A.clone(), "m_B": m_B.clone(),
                       "v_A": v_A.clone(), "v_B": v_B.clone()})
        m_As.append(m_A); m_Bs.append(m_B); v_As.append(v_A); v_Bs.append(v_B)
        gA_list.append(torch.randn(r, d_in, device=device))
        gB_list.append(torch.randn(d_out, r, device=device))
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 5
    uA_loop, uB_loop = adam_direction_loop(
        [{"m_A": s["m_A"].clone(), "m_B": s["m_B"].clone(),
          "v_A": s["v_A"].clone(), "v_B": s["v_B"].clone()} for s in states],
        gA_list, gB_list, beta1, beta2, eps, step)
    uA_fe, uB_fe = adam_direction_foreach(
        m_As, m_Bs, v_As, v_Bs, gA_list, gB_list, beta1, beta2, eps, step)
    eA = max((a - b).abs().max().item() for a, b in zip(uA_loop, uA_fe))
    eB = max((a - b).abs().max().item() for a, b in zip(uB_loop, uB_fe))
    print(f"  equivalence (N={N_pairs}): u_A_max_err={eA:.2e}, u_B_max_err={eB:.2e}")
    assert eA < atol and eB < atol


def time_call(fn, n_warmup, n_reps, device):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device)
        times.append(s.elapsed_time(e))
    return times


def make_state_lists(shapes, device, seed=0):
    torch.manual_seed(seed)
    states, m_As, m_Bs, v_As, v_Bs, gA_list, gB_list = [], [], [], [], [], [], []
    for (r, d_in, d_out) in shapes:
        m_A = torch.zeros(r, d_in, device=device, dtype=torch.float32)
        m_B = torch.zeros(d_out, r, device=device, dtype=torch.float32)
        v_A = torch.zeros(r, d_in, device=device, dtype=torch.float32)
        v_B = torch.zeros(d_out, r, device=device, dtype=torch.float32)
        states.append({"m_A": m_A.clone(), "m_B": m_B.clone(),
                       "v_A": v_A.clone(), "v_B": v_B.clone()})
        m_As.append(m_A); m_Bs.append(m_B); v_As.append(v_A); v_Bs.append(v_B)
        gA_list.append(torch.randn(r, d_in, device=device))
        gB_list.append(torch.randn(d_out, r, device=device))
    return states, m_As, m_Bs, v_As, v_Bs, gA_list, gB_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_reps", type=int, default=20)
    args = p.parse_args()
    device = torch.device(args.device)

    # Real OLMo-2-1B all-linear pair shapes at r=16:
    SHAPES = (
        [(16, 2048, 2048)] * 64 +
        [(16, 2048, 8192)] * 32 +
        [(16, 8192, 2048)] * 16
    )
    print(f"# device={device}, n_pairs={len(SHAPES)}")
    if device.type == "cuda":
        print(f"# gpu={torch.cuda.get_device_name(device)}")

    print("\n# === Equivalence ===")
    equivalence_check(len(SHAPES), SHAPES, device)

    print("\n# === Timing (ms) ===")
    print(f"  {'impl':<12} {'median':>9} {'min':>9} {'max':>9} {'speedup':>8}")
    states, m_As, m_Bs, v_As, v_Bs, gA, gB = make_state_lists(SHAPES, device)
    states_clone = [{k: v.clone() for k, v in s.items()} for s in states]

    loop_t = time_call(
        lambda: adam_direction_loop(states_clone, gA, gB, 0.9, 0.999, 1e-8, 5),
        args.n_warmup, args.n_reps, device)
    fe_t = time_call(
        lambda: adam_direction_foreach(m_As, m_Bs, v_As, v_Bs, gA, gB,
                                       0.9, 0.999, 1e-8, 5),
        args.n_warmup, args.n_reps, device)
    loop_med = sorted(loop_t)[len(loop_t) // 2]
    fe_med = sorted(fe_t)[len(fe_t) // 2]
    print(f"  {'loop':<12} {loop_med:>9.3f} {min(loop_t):>9.3f} {max(loop_t):>9.3f} "
          f"{1.0:>7.2f}x")
    print(f"  {'foreach':<12} {fe_med:>9.3f} {min(fe_t):>9.3f} {max(fe_t):>9.3f} "
          f"{loop_med/fe_med:>7.2f}x")


if __name__ == "__main__":
    main()
