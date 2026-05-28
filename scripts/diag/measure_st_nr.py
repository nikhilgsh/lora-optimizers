#!/usr/bin/env python
"""Measure the Davis-Drusvyatskiy (arXiv:2512.04299) spectral-advantage
quantities per adapted layer, on a real prompt-masked opc batch:

  st(A)  = ||A||_F^2 / sigma_max(A)^2     (incoming-activation stable rank)
  nr(G)  = ||G||_*^2 / ||G||_F^2          (gradient nuclear rank)
  ratio  = nr(G) / st(A)                  (predicted SpecGD-over-Euclidean speedup)

A = input activations to each target Linear (forward hook, non-pad tokens only).
G = dense weight gradient dL/dW (we unfreeze ONLY the target Linear weights;
    at LoRA init B=0 so the base-model gradient == the gradient the LoRA run
    sees at step 0). No LoRA / no optimizer needed — this is the layer-block
    gradient the paper's condition is stated on.

Outputs a JSON with per-layer rows + aggregates grouped by projection type.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/home/nghosh/lora")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from lora_playground.data import PadToMaxCollator


def sigma_max(M, iters=20):
    """Top singular value of (m,n) M via power iteration on M^T M. fp32."""
    M = M.float()
    n = M.shape[1]
    v = torch.randn(n, device=M.device, dtype=M.dtype)
    v /= v.norm() + 1e-30
    for _ in range(iters):
        u = M @ v
        v = M.transpose(0, 1) @ u
        v /= v.norm() + 1e-30
    return (M @ v).norm().item()  # = sigma_max


def proj_type(name):
    for k in ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj",
              "qkv_proj", "wi", "wo"]:
        if k in name:
            return k
    return name.split(".")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            mod.weight.requires_grad_(True)
            targets[name] = mod
    print(f"# {args.model_name}: {len(targets)} target Linear layers", flush=True)

    cur_mask = {}  # set per-batch: flattened attention mask (n_tokens,)
    act_F2 = {n: 0.0 for n in targets}     # accumulated ||A||_F^2 over batches
    act_smax2 = {n: [] for n in targets}   # per-batch sigma_max(A)^2

    hooks = []
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach()                       # (B,S,d_in)
            x = x.reshape(-1, x.shape[-1]).float()    # (B*S, d_in)
            m = cur_mask["m"]
            xv = x[m]                                  # non-pad tokens
            act_F2[name] += float(xv.pow(2).sum().item())
            act_smax2[name].append(sigma_max(xv) ** 2)
        return hook
    for name, mod in targets.items():
        hooks.append(mod.register_forward_hook(mk(name)))

    ds = load_from_disk(str(Path(args.data_dir) / "eval"))
    coll = PadToMaxCollator(seq_length=args.seq_length, pad_token_id=tok.pad_token_id)

    model.zero_grad(set_to_none=True)
    for b in range(args.n_batches):
        feats = [ds[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size)]
        batch = coll(feats)
        batch = {k: v.to(dev) for k, v in batch.items()}
        cur_mask["m"] = batch["attention_mask"].reshape(-1).bool()
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"])
        out.loss.backward()   # grads ACCUMULATE across batches (sum) → nr(G) on summed grad
        print(f"#  batch {b}: loss={out.loss.item():.4f}", flush=True)
    for h in hooks:
        h.remove()

    rows = []
    for name, mod in targets.items():
        st_A = act_F2[name] / (max(act_smax2[name]) + 1e-30)  # worst-case (largest) sigma over batches
        # actually use mean sigma_max^2 across batches for a representative st
        smax2_mean = sum(act_smax2[name]) / len(act_smax2[name])
        st_A = act_F2[name] / (smax2_mean + 1e-30)
        G = mod.weight.grad.float()
        sv = torch.linalg.svdvals(G)
        nuc = float(sv.sum().item())
        fro2 = float((sv ** 2).sum().item())
        nr_G = (nuc ** 2) / (fro2 + 1e-30)
        rows.append({
            "layer": name, "proj": proj_type(name),
            "d_out": G.shape[0], "d_in": G.shape[1],
            "st_A": st_A, "nr_G": nr_G, "ratio_nr_over_st": nr_G / (st_A + 1e-30),
        })

    import statistics as st
    def agg(key):
        vals = [r[key] for r in rows]
        return {"median": st.median(vals), "min": min(vals), "max": max(vals)}
    by_proj = {}
    for r in rows:
        by_proj.setdefault(r["proj"], []).append(r)
    proj_summary = {p: {
        "n": len(rs),
        "st_A_med": st.median([r["st_A"] for r in rs]),
        "nr_G_med": st.median([r["nr_G"] for r in rs]),
        "ratio_med": st.median([r["ratio_nr_over_st"] for r in rs]),
    } for p, rs in by_proj.items()}

    result = {
        "model_name": args.model_name, "data_dir": args.data_dir,
        "n_layers": len(rows), "n_batches": args.n_batches,
        "batch_size": args.batch_size, "seq_length": args.seq_length,
        "overall": {"st_A": agg("st_A"), "nr_G": agg("nr_G"),
                    "ratio_nr_over_st": agg("ratio_nr_over_st")},
        "by_proj": proj_summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"# wrote {args.out}", flush=True)
    print(f"# OVERALL median: st(A)={result['overall']['st_A']['median']:.2f}  "
          f"nr(G)={result['overall']['nr_G']['median']:.2f}  "
          f"ratio={result['overall']['ratio_nr_over_st']['median']:.2f}", flush=True)


if __name__ == "__main__":
    main()
