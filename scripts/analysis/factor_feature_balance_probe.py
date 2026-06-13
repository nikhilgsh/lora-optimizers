#!/usr/bin/env python
"""Measure the two output-feature contributions delta1, delta2 directly.

delta1 = B dA x   (update to A, seen through B)   -- d_out vector
delta2 = dB A x   (update to B, on current A)     -- d_out vector

on REAL activations x (inputs to lora_A) at a saved snapshot. Reports per-module
RMS(delta1), RMS(delta2), their ratio, and the c_A:c_B factor that would balance
them. delta2/delta1 = 1 means equal-rho already balances the two factor feature
contributions; otherwise c_A/c_B = RMS(delta2)/RMS(delta1) balances them.

Replay is copied verbatim from muon_activation_isotropy_probe.replay_* (verified
chord-tight path) but returns full dA, dB instead of only dA's rowspace.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lora_playground.data import PadToMaxCollator
from lora_playground.train import make_parser, parse_target_modules
from lora_playground.training_kernel import build_peft_model
from lora_playground.utils import collect_lora_pairs_named


def _load_run_args(ckpt_dir: Path):
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    toks = shlex.split(meta["cfg_snapshot"]["command"])
    return make_parser().parse_args(toks[1:]), meta


def _sorted_pairs(pair_state):
    return [pair_state[k] for k in sorted(pair_state)]


def _polar_batched(X):
    U, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    return U @ Vh


def _opnorm_batched(X):
    return torch.linalg.svdvals(X.float()).amax(dim=-1)


@torch.no_grad()
def replay_dA_dB(pair_state, group_state, *, lr, picard_iters, lora_plus_multiplier):
    """Return {global_pair_idx: (dA, dB)} for the chord-tight update."""
    pairs = _sorted_pairs(pair_state)
    out = {}
    for gid, gs in enumerate(group_state):
        items = [(gi, p) for gi, p in enumerate(pairs) if int(p.get("_group", -1)) == gid]
        items.sort(key=lambda it: int(it[1].get("_local_idx", it[0])))
        if not items:
            continue
        A_f = torch.stack([p["A"].float() for _, p in items])
        B_f = torch.stack([p["B"].float() for _, p in items])
        u_A = torch.stack([p["u_A"].float() for _, p in items])
        u_B = torch.stack([p["u_B"].float() for _, p in items])
        SA_half_inv = gs["SA_half_inv"].float()
        SB_half_inv = gs["SB_half_inv"].float()

        X_A_pre = SB_half_inv @ u_A
        X_B_pre = u_B @ SA_half_inv
        u_A = u_A / _opnorm_batched(X_A_pre).clamp_min(1e-30)[:, None, None]
        u_B = u_B / _opnorm_batched(X_B_pre).clamp_min(1e-30)[:, None, None]

        sigma_A = _opnorm_batched(A_f)
        sigma_B = _opnorm_batched(B_f)
        s_AB = sigma_A + sigma_B
        rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * float(lr))) / 2.0
        picard_coeff = (2.0 / (rho * s_AB).clamp_min(1e-30))[:, None, None]

        dA = torch.zeros_like(u_A)
        dB = torch.zeros_like(u_B)
        for k_iter in range(int(picard_iters)):
            if k_iter > 0:
                u_A_eff = u_A + picard_coeff * (B_f.transpose(-2, -1) @ dB @ A_f)
                u_B_eff = u_B + picard_coeff * (B_f @ dA @ A_f.transpose(-2, -1))
            else:
                u_A_eff, u_B_eff = u_A, u_B
            P_A = _polar_batched(SB_half_inv @ u_A_eff)
            P_B = _polar_batched(u_B_eff @ SA_half_inv)
            geo_A = SB_half_inv @ P_A
            geo_B = P_B @ SA_half_inv
            dA = -(rho[:, None, None] / _opnorm_batched(geo_A).clamp_min(1e-30)[:, None, None]) * geo_A
            dB = -(float(lora_plus_multiplier) * rho[:, None, None]
                   / _opnorm_batched(geo_B).clamp_min(1e-30)[:, None, None]) * geo_B
        for local_idx, (global_idx, _) in enumerate(items):
            out[global_idx] = (dA[local_idx].contiguous(), dB[local_idx].contiguous())
    return out


def _copy_weights(model, pair_state):
    pairs = _sorted_pairs(pair_state)
    named = collect_lora_pairs_named(model, adapter_name="default")
    names = []
    with torch.no_grad():
        for snap, (A, B, name) in zip(pairs, named):
            A.copy_(snap["A"].to(A.device, A.dtype))
            B.copy_(snap["B"].to(B.device, B.dtype))
            names.append(name)
    return names


def _module_kind(name):
    for sep in (".self_attn.", ".mlp."):
        if sep in name:
            return name.split(sep, 1)[1].split(".", 1)[0]
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="notebooks/snapshot_analysis/_data/factor_feature_balance_rows.csv")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    run_args, meta = _load_run_args(ckpt_dir)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (run_args.bf16 and device.type == "cuda") else None

    sd = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    updates = replay_dA_dB(sd["pair_state"], sd["group_state"],
                           lr=float(run_args.lr),
                           picard_iters=int(run_args.picard_iters_override or 3),
                           lora_plus_multiplier=float(run_args.lora_plus_multiplier))

    peft = build_peft_model(
        model_name=run_args.model_name, training_mode=run_args.training_mode,
        target_modules=parse_target_modules(run_args.target_modules),
        lora_r=run_args.lora_r, lora_alpha=run_args.lora_alpha,
        lora_dropout=run_args.lora_dropout, dtype=dtype,
        attn_implementation=run_args.attn_implementation, use_liger=False,
        liger_flce=False, gradient_checkpointing=False, compile_mode=None, device=device)
    model = peft.bare_model
    model.eval()
    names = _copy_weights(model, sd["pair_state"])
    pairs = _sorted_pairs(sd["pair_state"])

    tok = AutoTokenizer.from_pretrained(run_args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_from_disk(str(Path(run_args.data_dir) / "eval"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=PadToMaxCollator(seq_length=run_args.max_seq_length,
                                                    pad_token_id=tok.pad_token_id), num_workers=0)

    # accumulators per pair: sum of squared norms of delta1, delta2, and token count
    acc = {i: {"d1": 0.0, "d2": 0.0, "n": 0} for i in range(len(names))}
    mask_holder = {}

    mod_by_name = {}
    for mn, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            mod_by_name[f"{mn}[default]"] = mod.lora_A["default"]

    handles = []
    for idx, name in enumerate(names):
        dA, dB = updates[idx]
        A = pairs[idx]["A"].float()
        B = pairs[idx]["B"].float()

        def make_hook(i=idx, A=A, B=B, dA=dA, dB=dB):
            def hook(_m, inputs):
                x = inputs[0].detach()
                mask = mask_holder["mask"].to(x.device)
                x = x[mask].float()              # (tokens, d_in)
                if x.numel() == 0:
                    return
                dev = x.device
                A_, B_, dA_, dB_ = A.to(dev), B.to(dev), dA.to(dev), dB.to(dev)
                d1 = (x @ dA_.T) @ B_.T           # delta1 = B dA x  -> (tokens, d_out)
                d2 = (x @ A_.T) @ dB_.T           # delta2 = dB A x  -> (tokens, d_out)
                acc[i]["d1"] += float(d1.square().sum().cpu())
                acc[i]["d2"] += float(d2.square().sum().cpu())
                acc[i]["n"] += int(x.shape[0])
            return hook
        handles.append(mod_by_name[name].register_forward_pre_hook(make_hook()))

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches:
                break
            mask_holder["mask"] = batch["attention_mask"].bool()
            model(**{k: v.to(device) for k, v in batch.items()})
    for h in handles:
        h.remove()

    rows = []
    for idx, name in enumerate(names):
        a = acc[idx]
        if a["n"] == 0:
            continue
        d_out = int(pairs[idx]["B"].shape[0]); r = int(pairs[idx]["A"].shape[0]); d_in = int(pairs[idx]["A"].shape[1])
        rms1 = (a["d1"] / (a["n"] * d_out)) ** 0.5
        rms2 = (a["d2"] / (a["n"] * d_out)) ** 0.5
        rows.append({"name": name, "kind": _module_kind(name), "d_in": d_in, "d_out": d_out, "r": r,
                     "rms_delta1_BdA": rms1, "rms_delta2_dBA": rms2,
                     "ratio_d2_over_d1": rms2 / max(rms1, 1e-30)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # summary by kind
    import statistics as st
    print(f"step={meta['step']}  modules={len(rows)}")
    print(f"{'kind':12s} {'d_in':>5} {'d_out':>6} {'RMS(d1=BdA)':>12} {'RMS(d2=dBA)':>12} {'d2/d1':>7}")
    bykind = {}
    for row in rows:
        bykind.setdefault(row["kind"], []).append(row)
    for kind, rs in sorted(bykind.items()):
        med = lambda k: st.median([x[k] for x in rs])
        print(f"{kind:12s} {rs[0]['d_in']:>5} {rs[0]['d_out']:>6} "
              f"{med('rms_delta1_BdA'):>12.4e} {med('rms_delta2_dBA'):>12.4e} {med('ratio_d2_over_d1'):>7.3f}")
    allr = [row["ratio_d2_over_d1"] for row in rows]
    print(f"\nOverall median d2/d1 = {st.median(allr):.3f}   (1.0 = equal-rho already balances)")
    print(f"  if !=1, to balance set c_A/c_B = median(d2/d1) = {st.median(allr):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
