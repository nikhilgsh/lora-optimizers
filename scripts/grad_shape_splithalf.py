"""Is the gradient-energy shape static because it is STRUCTURAL, or because the
batch is large enough that one step already estimates it precisely?

PoLoRA's diagonal metric is max-normalized, so only the SHAPE of the accumulated
per-coordinate energy enters the update, and beta2 sets the EMA window that shape is
averaged over. scripts/grad_shape_autocorr.py measured that shape barely drifts
across steps (cos ~0.95-0.99 at lag 40), which is why beta2 looks inert. Two
explanations:

  (1) structural -- the shape is set by the pretrained checkpoint (per-input-channel
      activation energy is strongly heterogeneous and fixed by the weights), so it is
      the same every step regardless of which tokens are in the batch;
  (2) batch-size -- the per-step estimate is already low-variance at 32768 tokens, so
      there is no noise left for the EMA to average away.

Split-half separates them. Compute the shape from the first half of a batch and from
the second half, at the SAME step, and correlate:

  split-half cos ~= across-step cos  => one step is already a precise estimate, so
      the across-step stability is explained by (2), and beta2 should come alive at
      small batch.
  split-half cos <  across-step cos  => the per-step estimate is genuinely noisy yet
      the shape still persists across steps, i.e. (1): the persistent part is
      structural, and averaging is recovering it.

Also sweeps the half-batch size, so the split-half correlation can be read as a
function of tokens per estimate -- that curve says directly how small a batch would
have to be before beta2 has something to do.

Usage:
  python scripts/grad_shape_splithalf.py --data_dir data/<...> --tag <label>
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


def energy_shape(opt, model, batches, dev, npairs):
    """diag(G_A^T C_B^-1 G_A) per pair, accumulated over `batches` -- the exact
    quantity optim.py folds into D_in."""
    opt.zero_grad(set_to_none=True)
    for b in batches:
        out = model(**b)
        (out.loss / len(batches)).backward()
    shapes = []
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
            shapes.append((gA * ((CBh @ CBh) @ gA)).sum(dim=0).detach().cpu())
    opt.zero_grad(set_to_none=True)
    return shapes


def cos_list(xs, ys):
    vals = []
    for x, y in zip(xs, ys):
        nx, ny = x.norm(), y.norm()
        if nx > 0 and ny > 0:
            vals.append(float((x @ y) / (nx * ny)))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warm_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--max_pairs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5, help="independent split-half trials")
    ap.add_argument("--out", default="bench_abl_out/grad_shape_splithalf.jsonl")
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
    it = iter(dl)

    def nextb():
        return {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}

    for step in range(a.warm_steps):          # warm P,Q so C_B is realistic
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum_steps):
            out = model(**nextb())
            (out.loss / a.grad_accum_steps).backward()
        opt.step()
    print(f"warmed {a.warm_steps} steps; loss {out.loss.item():.4f}", flush=True)

    tokens_per_micro = a.batch_size * a.max_seq_length
    # half-batch sizes in micro-batches: 1, 2, 4 -> 8192, 16384, 32768 tokens per half
    halves = [h for h in (1, 2, a.grad_accum_steps) if h >= 1]
    halves = sorted(set(halves))
    prev = None
    res = {}
    for h in halves:
        vals = []
        for _ in range(a.reps):
            A_b = [nextb() for _ in range(h)]
            B_b = [nextb() for _ in range(h)]
            sa = energy_shape(opt, model, A_b, dev, npairs)
            sb = energy_shape(opt, model, B_b, dev, npairs)
            vals.append(cos_list(sa, sb))
        res[h] = sum(vals) / len(vals)
        print(f"  half = {h} micro-batch(es) = {h*tokens_per_micro:6d} tokens: "
              f"split-half cos = {res[h]:.4f}", flush=True)

    print(f"\n=== split-half vs across-step: {a.tag} ===")
    print(f"{a.model_name}  r={a.lora_r}  {npairs} pairs  {a.reps} trials per point")
    print("tokens/estimate   split-half cos")
    for h in halves:
        print(f"{h*tokens_per_micro:15d}   {res[h]:.4f}")
    full = res[max(halves)]
    print()
    print(f"production step = {a.grad_accum_steps*tokens_per_micro} tokens; the "
          f"largest half here is {max(halves)*tokens_per_micro}.")
    print("Compare the largest split-half cos against the across-step cos from")
    print("scripts/grad_shape_autocorr.py: similar => the per-step estimate is already")
    print("precise (batch-size explanation); markedly lower => the persistent part is")
    print("structural.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(tag=a.tag, model=a.model_name, data_dir=a.data_dir,
                                lora_r=a.lora_r, pairs=npairs, reps=a.reps,
                                tokens_per_micro=tokens_per_micro,
                                split_half={str(h*tokens_per_micro): res[h]
                                            for h in halves})) + "\n")
    print("appended to", a.out)


if __name__ == "__main__":
    main()
