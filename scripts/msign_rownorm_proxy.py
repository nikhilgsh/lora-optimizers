"""Can msign be replaced by elementwise/reduction-only row normalization?

PoLoRA feeds a whitened momentum to the matrix sign. For the A factor that matrix is

    H_A = C_B^{-1/2} Mhat_A Q^{-1/2}          (r x d_in)

and for the B factor

    H_B = P^{-1/2} Mhat_B C_A^{-1/2}          (d_out x r)

and the update direction is msign(H) = (H H^T)^{-1/2} H, which makes the rows exactly
orthonormal. The cheap systems-friendly proxy is to normalize each row to unit norm,

    Stilde = D^{-1/2} H,      D = diag(H H^T),

i.e. a rowwise sum-of-squares, an rsqrt and a multiply -- no matrix inverse square root.

Row normalization fixes unequal row LENGTHS but cannot decorrelate rows. Writing

    R = D^{-1/2} H H^T D^{-1/2}

gives R_ii = 1 and R_ij = <H_i, H_j> / (|H_i| |H_j|), so R is exactly the matrix of
pairwise cosine similarities between rows, while the true msign delivers S S^T = I. So
"is row normalization a good substitute" is "after fixing their lengths, are the rows of
the real H already near-orthogonal", measured by how far R is from I.

    H H^T = [[100, 0], [0, 1]]      -> R = I; row norm IS msign, despite sigma 10 vs 1.
    H H^T = [[1, .9], [.9, 1]]      -> R unchanged; the bad spectrum is correlation, and
                                       no elementwise rule can fix it.

Reported per factor side (the three statistics are fixed; do not add more mid-study):

    eps_corr    = |R - I|_F / sqrt(r)
    max_offdiag = max_{i != j} |R_ij|
    kappa(R)    = lambda_max(R) / lambda_min(R)

eps_corr NEEDS ITS NULL and is meaningless without it. For an H whose rows are iid
Gaussian in R^d, E[R_ij^2] ~ 1/d, so |R - I|_F ~ r/sqrt(d) and eps_corr ~ sqrt(r/d) --
0.35 at r=256, d=2048, not 0. The script therefore computes the same statistic on a
random Gaussian of each H's exact shape and prints them side by side. Read eps_corr
against that column, never against zero.

And the statistic that actually decides it, from the same capture:

    cos(msign(H), D^{-1/2} H)

the Frobenius cosine between the true msign output and the row-normalized proxy. That is
the direct answer; the R statistics say why it came out the way it did.

H is captured by wrapping CurvatureWhitenLoRA._polar_ns_guarded on the instance, so both
the input and the msign output are the real ones the optimizer used -- no reimplementation
of the whitening, which would risk measuring a second copy that has drifted from the first.

Usage:
  python scripts/msign_rownorm_proxy.py --data_dir data/<...> --tag <label>
"""
import argparse
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from lora_playground.data import PackingCollator
from lora_playground.optim import CurvatureWhitenLoRA
from lora_playground.optim_specs import REGISTRY
from lora_playground.training_kernel import build_peft_model


def corr_stats(H):
    """(eps_corr, max_offdiag, kappa) for one batched H.

    Contracts on the SMALL side: msign makes the small side's rows/cols orthonormal, so
    the r x r Gram is the object to look at whichever way H is oriented. H_A is
    (n, r, d_in) and H_B is (n, d_out, r); pick the smaller of the last two dims.
    """
    H = H.float()
    if H.shape[-2] <= H.shape[-1]:          # rows are the small side (H_A)
        G = H @ H.transpose(-2, -1)
    else:                                    # cols are the small side (H_B)
        G = H.transpose(-2, -1) @ H
    r = G.shape[-1]
    d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(1e-30).rsqrt()
    R = G * d.unsqueeze(-1) * d.unsqueeze(-2)
    eye = torch.eye(r, device=R.device, dtype=R.dtype).expand_as(R)
    off = R - eye
    eps = (off.flatten(1).norm(dim=1) / math.sqrt(r))
    mx = off.abs().flatten(1).max(dim=1).values
    ev = torch.linalg.eigvalsh(0.5 * (R + R.transpose(-2, -1)))
    kap = ev[..., -1] / ev[..., 0].clamp_min(1e-12)
    return eps, mx, kap


def rownorm(H):
    """D^{-1/2} H on the small side -- the proxy. Same orientation convention as
    corr_stats: normalize whichever of rows/cols msign orthonormalizes."""
    H = H.float()
    if H.shape[-2] <= H.shape[-1]:
        n = H.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return H / n
    n = H.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    return H / n


def fro_cos(X, Y):
    X, Y = X.float().flatten(1), Y.float().flatten(1)
    return ((X * Y).sum(dim=1) / (X.norm(dim=1) * Y.norm(dim=1)).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warm_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--probe_steps", type=int, default=3)
    ap.add_argument("--out", default="bench_abl_out/msign_rownorm_proxy.jsonl")
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
    it = iter(dl)

    def nextb():
        return {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}

    opt = CurvatureWhitenLoRA(
        bare, lr=a.lr, betas=(0.9, 0.999), delta=1e-4, curvature_beta=0.99,
        ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
        cw_nesterov=True, precond_refresh_every=10, precond_method="gram_ns",
        higham_iters=8, cw_metric_init="1e-12",
        **dict(REGISTRY["kl-diag-polar-lora"].fixed))

    def grads():
        opt.zero_grad(set_to_none=False)
        for _ in range(a.grad_accum_steps):
            out = model(**nextb())
            (out.loss / a.grad_accum_steps).backward()
        return out.loss.item()

    for _ in range(a.warm_steps):
        grads(); opt.step()
    print(f"warmed {a.warm_steps} steps", flush=True)

    # Capture the REAL pre-msign matrix and the REAL msign output.
    captured = []
    _orig = opt._polar_ns_guarded

    def spy(z, grp, key):
        out = _orig(z, grp, key)
        captured.append((key, z.detach().clone(), out.detach().clone()))
        return out
    opt._polar_ns_guarded = spy

    for _ in range(a.probe_steps):
        grads(); opt.step()
    opt._polar_ns_guarded = _orig

    agg = {}
    for key, H, S in captured:
        side = "H_A" if key.endswith("zA") else "H_B"
        eps, mx, kap = corr_stats(H)
        cos = fro_cos(S, rownorm(H))
        # Null: same shape, iid Gaussian. eps_corr ~ sqrt(r/d) for random rows, so the
        # measured eps_corr means nothing except relative to this.
        eps0, _, _ = corr_stats(torch.randn_like(H.float()))
        b = agg.setdefault(side, dict(eps=[], mx=[], kap=[], cos=[], eps0=[], shapes=set()))
        b["eps"] += eps.tolist(); b["mx"] += mx.tolist(); b["kap"] += kap.tolist()
        b["cos"] += cos.tolist(); b["eps0"] += eps0.tolist()
        b["shapes"].add(tuple(H.shape[-2:]))

    def m(v):
        return sum(v) / len(v) if v else float("nan")

    print(f"\n=== msign vs row normalization: {a.tag} ===")
    print(f"{a.model_name}  r={a.lora_r}  {a.batch_size*a.grad_accum_steps*a.max_seq_length}"
          f" tokens/step  {a.probe_steps} probe steps")
    print(f"{'side':5} {'eps_corr':>9} {'(random)':>9} {'max|Rij|':>9} {'kappa(R)':>10} "
          f"{'cos(msign, rownorm)':>21}")
    out = {}
    for side, b in sorted(agg.items()):
        out[side] = dict(eps_corr=m(b["eps"]), eps_corr_random=m(b["eps0"]),
                         max_offdiag=m(b["mx"]), kappa=m(b["kap"]), cos=m(b["cos"]),
                         n=len(b["eps"]), shapes=sorted(b["shapes"]))
        print(f"{side:5} {m(b['eps']):9.4f} {m(b['eps0']):9.4f} {m(b['mx']):9.4f} "
              f"{m(b['kap']):10.2f} {m(b['cos']):21.4f}")
    print()
    print("cos near 1 AND eps_corr near its random column => the rows are already")
    print("   near-orthogonal once rescaled; row normalization can stand in for msign.")
    print("cos well below 1, or eps_corr well above random => the spectrum is set by")
    print("   cross-rank CORRELATION, not by unequal row norms, and no elementwise or")
    print("   reduction-only rule reproduces msign.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(tag=a.tag, model=a.model_name, lora_r=a.lora_r,
                                lr=a.lr, warm_steps=a.warm_steps,
                                probe_steps=a.probe_steps, sides=out), default=str) + "\n")
    print("appended to", a.out)


if __name__ == "__main__":
    main()
