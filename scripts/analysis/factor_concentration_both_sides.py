#!/usr/bin/env python
"""Measure BOTH per-factor concentrations on real data:

  R_A = sqrt(d_in/r  * ||x   @ V_A||^2 / ||x||^2)     V_A = right-sing basis of dA (input side)
  R_B = sqrt(d_out/r * ||g   @ U_B||^2 / ||g||^2)     U_B = left-sing  basis of dB (output side)

where x = inputs to lora_A (forward), g = grad wrt lora_B output (backward).
R_A is Codex's A-side concentration; R_B is its B-side mirror (loss-gradient
concentration in the B-update column space). Principled step ratio:

  c_B / c_A = sqrt(d_out/r) * R_A / R_B

Replay copied from the verified chord-tight path; returns dA, dB.
"""
from __future__ import annotations
import argparse, csv, json, shlex
from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lora_playground.data import PadToMaxCollator
from lora_playground.train import make_parser, parse_target_modules
from lora_playground.training_kernel import build_peft_model
from lora_playground.utils import collect_lora_pairs_named


def _load_run_args(ckpt_dir):
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    return make_parser().parse_args(shlex.split(meta["cfg_snapshot"]["command"])[1:]), meta

def _sorted_pairs(ps): return [ps[k] for k in sorted(ps)]
def _polar(X):
    U, _, Vh = torch.linalg.svd(X.float(), full_matrices=False); return U @ Vh
def _op(X): return torch.linalg.svdvals(X.float()).amax(dim=-1)


@torch.no_grad()
def replay_dA_dB(pair_state, group_state, *, lr, picard_iters, lora_plus_multiplier):
    pairs = _sorted_pairs(pair_state); out = {}
    for gid, gs in enumerate(group_state):
        items = [(gi, p) for gi, p in enumerate(pairs) if int(p.get("_group", -1)) == gid]
        items.sort(key=lambda it: int(it[1].get("_local_idx", it[0])))
        if not items: continue
        A_f = torch.stack([p["A"].float() for _, p in items])
        B_f = torch.stack([p["B"].float() for _, p in items])
        u_A = torch.stack([p["u_A"].float() for _, p in items])
        u_B = torch.stack([p["u_B"].float() for _, p in items])
        SAi = gs["SA_half_inv"].float(); SBi = gs["SB_half_inv"].float()
        u_A = u_A / _op(SBi @ u_A).clamp_min(1e-30)[:, None, None]
        u_B = u_B / _op(u_B @ SAi).clamp_min(1e-30)[:, None, None]
        sA = _op(A_f); sB = _op(B_f); s_AB = sA + sB
        rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * float(lr))) / 2.0
        pc = (2.0 / (rho * s_AB).clamp_min(1e-30))[:, None, None]
        dA = torch.zeros_like(u_A); dB = torch.zeros_like(u_B)
        for k in range(int(picard_iters)):
            if k > 0:
                uAe = u_A + pc * (B_f.transpose(-2, -1) @ dB @ A_f)
                uBe = u_B + pc * (B_f @ dA @ A_f.transpose(-2, -1))
            else:
                uAe, uBe = u_A, u_B
            gA = SBi @ _polar(SBi @ uAe); gB = _polar(uBe @ SAi) @ SAi
            dA = -(rho[:, None, None] / _op(gA).clamp_min(1e-30)[:, None, None]) * gA
            dB = -(float(lora_plus_multiplier) * rho[:, None, None]
                   / _op(gB).clamp_min(1e-30)[:, None, None]) * gB
        for li, (gi, _) in enumerate(items):
            out[gi] = (dA[li].contiguous(), dB[li].contiguous())
    return out


def _kind(name):
    for s in (".self_attn.", ".mlp."):
        if s in name: return name.split(s, 1)[1].split(".", 1)[0]
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt = Path(args.ckpt_dir)
    run_args, meta = _load_run_args(ckpt)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (run_args.bf16 and device.type == "cuda") else None
    sd = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False)
    upd = replay_dA_dB(sd["pair_state"], sd["group_state"], lr=float(run_args.lr),
                       picard_iters=int(run_args.picard_iters_override or 3),
                       lora_plus_multiplier=float(run_args.lora_plus_multiplier))
    peft = build_peft_model(model_name=run_args.model_name, training_mode=run_args.training_mode,
        target_modules=parse_target_modules(run_args.target_modules), lora_r=run_args.lora_r,
        lora_alpha=run_args.lora_alpha, lora_dropout=run_args.lora_dropout, dtype=dtype,
        attn_implementation=run_args.attn_implementation, use_liger=False, liger_flce=False,
        gradient_checkpointing=False, compile_mode=None, device=device)
    model = peft.bare_model
    named = collect_lora_pairs_named(model, adapter_name="default")
    with torch.no_grad():
        for snap, (A, B, _) in zip(_sorted_pairs(sd["pair_state"]), named):
            A.copy_(snap["A"].to(A.device, A.dtype)); B.copy_(snap["B"].to(B.device, B.dtype))
    names = [n for _, _, n in named]
    pairs = _sorted_pairs(sd["pair_state"])

    tok = AutoTokenizer.from_pretrained(run_args.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = load_from_disk(str(Path(run_args.data_dir) / "eval"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=PadToMaxCollator(seq_length=run_args.max_seq_length, pad_token_id=tok.pad_token_id))

    # precompute V_A (d_in,r) and U_B (d_out,r) per pair
    VA = {}; UB = {}
    for i in range(len(names)):
        dA, dB = upd[i]
        _, _, VhA = torch.linalg.svd(dA.float(), full_matrices=False); VA[i] = VhA.T.contiguous()
        UfB, _, _ = torch.linalg.svd(dB.float(), full_matrices=False); UB[i] = UfB.contiguous()

    acc = {i: {"x2": 0.0, "xV2": 0.0, "g2": 0.0, "gU2": 0.0, "n": 0} for i in range(len(names))}
    mask_holder = {}
    a_mods = {}; b_mods = {}
    for mn, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            a_mods[f"{mn}[default]"] = mod.lora_A["default"]; b_mods[f"{mn}[default]"] = mod.lora_B["default"]

    handles = []
    for i, name in enumerate(names):
        def fhook(i=i, V=VA[i]):
            def h(_m, inp):
                x = inp[0].detach(); m = mask_holder["mask"].to(x.device)
                x = x[m].float()
                if x.numel() == 0: return
                Vd = V.to(x.device)
                acc[i]["x2"] += float(x.square().sum().cpu())
                acc[i]["xV2"] += float((x @ Vd).square().sum().cpu())
                acc[i]["n"] += int(x.shape[0])
            return h
        def bhook(i=i, U=UB[i]):
            def h(_m, gin, gout):
                g = gout[0]
                if g is None: return
                g = g.detach(); m = mask_holder["mask"].to(g.device)
                g = g[m].float()
                if g.numel() == 0: return
                Ud = U.to(g.device)
                acc[i]["g2"] += float(g.square().sum().cpu())
                acc[i]["gU2"] += float((g @ Ud).square().sum().cpu())
            return h
        handles.append(a_mods[name].register_forward_pre_hook(fhook()))
        handles.append(b_mods[name].register_full_backward_hook(bhook()))

    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= args.num_batches: break
        mask_holder["mask"] = batch["attention_mask"].bool()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
    for h in handles: h.remove()

    rows = []
    for i, name in enumerate(names):
        a = acc[i]
        if a["n"] == 0 or a["x2"] <= 0 or a["g2"] <= 0: continue
        r = int(pairs[i]["A"].shape[0]); d_in = int(pairs[i]["A"].shape[1]); d_out = int(pairs[i]["B"].shape[0])
        R_A = (d_in / r * a["xV2"] / a["x2"]) ** 0.5
        R_B = (d_out / r * a["gU2"] / a["g2"]) ** 0.5
        ratio = (d_out / r) ** 0.5 * R_A / R_B
        rows.append({"name": name, "kind": _kind(name), "d_in": d_in, "d_out": d_out, "r": r,
                     "R_A": R_A, "R_B": R_B, "cB_over_cA": ratio})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import statistics as st
    print(f"step={meta['step']} r={pairs[0]['A'].shape[0]} modules={len(rows)}")
    print(f"{'kind':12s} {'d_in':>5} {'d_out':>6} {'R_A':>6} {'R_B':>6} {'cB/cA':>7}")
    by = {}
    for row in rows: by.setdefault(row["kind"], []).append(row)
    for k, rs in sorted(by.items()):
        med = lambda f: st.median([x[f] for x in rs])
        print(f"{k:12s} {rs[0]['d_in']:>5} {rs[0]['d_out']:>6} {med('R_A'):>6.2f} {med('R_B'):>6.2f} {med('cB_over_cA'):>7.2f}")
    print(f"\noverall median: R_A={st.median([r['R_A'] for r in rows]):.2f}  "
          f"R_B={st.median([r['R_B'] for r in rows]):.2f}  "
          f"cB/cA={st.median([r['cB_over_cA'] for r in rows]):.2f}   (current=1)")


if __name__ == "__main__":
    main()
