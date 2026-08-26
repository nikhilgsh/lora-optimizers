"""Does the per-coordinate gradient-energy SHAPE drift across steps?

PoLoRA's diagonal metric is max-normalized (`_rdinv` uses x/x_max), so only the
SHAPE of the accumulated energy matters, never its scale. beta2 sets how long an
EMA window that shape is averaged over. If the shape is near-static across steps,
a 5-step window (beta2=0.81) and a 100-step window (beta2=0.99) produce nearly the
same metric and beta2 is inert; if it drifts, beta2 matters.

So measure the shape directly rather than inferring it from loss curves. At each
step record the per-coordinate energy the metric is actually built from --
diag(G_A^T C_B^-1 G_A) for the input side and diag(G_B C_A^-1 G_B^T) for the output
side, i.e. the exact quantities optim.py accumulates into D_in / D_out -- and
report cos(d_t, d_{t+k}) against lag k, averaged over t and over LoRA pairs.

cos near 1 at large lag  => shape static  => beta2 inert
cos decaying with lag    => shape drifts  => beta2 should matter, with a
                            characteristic lag that predicts the useful window.

Usage:
  python scripts/grad_shape_autocorr.py --data_dir data/<...> --model_name <...>
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from lora_playground.data import PackingCollator
from lora_playground.optim import CurvatureWhitenLoRA, gram_ns_inv_sqrt
from lora_playground.optim_specs import REGISTRY
from lora_playground.training_kernel import build_peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True, help="label for the output row")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--max_pairs", type=int, default=16,
                    help="record this many LoRA pairs (all 112 is needless memory)")
    ap.add_argument("--out", default="bench_abl_out/grad_shape_autocorr.jsonl")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    peft = build_peft_model(
        model_name=a.model_name, training_mode="lora", target_modules="all-linear",
        lora_r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
        dtype=torch.bfloat16, attn_implementation="sdpa", use_liger=False,
        liger_flce=False, gradient_checkpointing=False, compile_mode=None,
        device=dev, world_size=1, local_rank=0)
    bare, model = peft.bare_model, peft.train_model
    model.train()

    ds = load_from_disk(os.path.join(a.data_dir, "train"))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    collate_fn=PackingCollator(seq_length=a.max_seq_length,
                                               mask_dtype=torch.bfloat16))
    opt = CurvatureWhitenLoRA(
        bare, lr=a.lr, betas=(0.9, 0.999), delta=1e-4, curvature_beta=0.99,
        ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
        cw_nesterov=True, precond_refresh_every=10, precond_method="gram_ns",
        higham_iters=8, cw_metric_init="1e-12",
        **dict(REGISTRY["kl-diag-polar-lora"].fixed))

    npairs = min(a.max_pairs, len(opt.pairs))
    din_hist = [[] for _ in range(npairs)]   # per pair: list over t of the d_in energy
    it = iter(dl)
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum_steps):
            batch = {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}
            out = model(**batch)
            (out.loss / a.grad_accum_steps).backward()
        # Record the per-step energy BEFORE opt.step(), using the same expression
        # optim.py accumulates: diag(G_A^T C_B^-1 G_A). C_B is rebuilt here from the
        # optimizer's current P exactly as the grouped step does.
        with torch.no_grad():
            for i in range(npairs):
                A, B = opt.pairs[i]
                st = opt.pair_state[i]
                gA = A.grad.float()
                doutB = opt._rdinv(st['D_out'].unsqueeze(0)).squeeze(0)
                P = (doutB * doutB).reciprocal()
                Bw = B.detach().float()
                CB = Bw.T @ (P.unsqueeze(-1) * Bw)
                CB = 0.5 * (CB + CB.T)
                CBh = gram_ns_inv_sqrt(CB.unsqueeze(0), nsteps=opt.higham_iters,
                                       eps=opt.delta, eps_relative=True).squeeze(0)
                CBinv = CBh @ CBh
                d = (gA * (CBinv @ gA)).sum(dim=0)         # length d_in
                din_hist[i].append(d.detach().cpu())
        opt.step()
        if step % 10 == 0:
            print(f"step {step}: loss {out.loss.item():.4f}", flush=True)

    # cos(d_t, d_{t+k}) averaged over t and pairs, per lag k
    lags = [1, 2, 5, 10, 20, 40]
    res = {}
    for k in lags:
        vals = []
        for i in range(npairs):
            H = din_hist[i]
            for t in range(len(H) - k):
                x, y = H[t], H[t + k]
                nx, ny = x.norm(), y.norm()
                if nx > 0 and ny > 0:
                    vals.append(float((x @ y) / (nx * ny)))
        res[k] = sum(vals) / len(vals) if vals else float("nan")

    print(f"\n=== gradient-energy shape autocorrelation: {a.tag} ===")
    print(f"{a.model_name}  r={a.lora_r}  {a.steps} steps  {npairs} pairs")
    print("lag k   mean cos(d_t, d_t+k)")
    for k in lags:
        print(f"{k:5d}   {res[k]:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(tag=a.tag, model=a.model_name, data_dir=a.data_dir,
                                lora_r=a.lora_r, steps=a.steps, pairs=npairs,
                                autocorr={str(k): res[k] for k in lags})) + "\n")
    print("appended to", a.out)


if __name__ == "__main__":
    main()
