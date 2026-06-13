#!/usr/bin/env python
"""Activation-space test for Keller-vs-muP scaling on LoRA A factors.

For a LoRA A factor with shape (r, d_in), the blog's compression-case
question is whether the input activations are isotropic with respect to the
right singular subspace of the current A update.

This script loads a diagnostic checkpoint, replays the next A-update subspace
from the saved optimizer snapshot, captures real inputs to each lora_A module
on eval batches, and reports:

    actual_ratio = RMS(x @ V) / RMS(x)

where V is the d_in x r right-singular basis of the replayed A update.
Keller corresponds to actual_ratio near 1.  The worst-case/muP end is
sqrt(d_in / r), whose corrective factor is sqrt(r / d_in).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    command = meta["cfg_snapshot"]["command"]
    toks = shlex.split(command)
    parser = make_parser()
    return parser.parse_args(toks[1:]), meta


def _sorted_pairs(pair_state: dict) -> list[dict]:
    return [pair_state[k] for k in sorted(pair_state)]


def _polar_batched(X: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
    return U @ Vh


def _opnorm_batched(X: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(X.float()).amax(dim=-1)


@torch.no_grad()
def replay_chord_tight_a_update_rowspaces(
    pair_state: dict,
    group_state: list[dict],
    *,
    lr: float,
    picard_iters: int,
    lora_plus_multiplier: float,
) -> dict[str, dict[int, torch.Tensor]]:
    """Replay the non-clean chord-tight update enough to get update subspaces.

    The saved r=64 snapshots in this repo used
    adam-polar-product-lora-coupled-spectral-chord-tight. This mirrors that
    path with exact SVD polars for analysis. Returned tensors are keyed by
    global pair index:
      - a_right: right basis of dA, shape (d_in, r)
      - b_left: left basis of dB, shape (d_out, r)
      - b_right: right basis of dB, shape (r, r)
    """
    pairs = _sorted_pairs(pair_state)
    out_a: dict[int, torch.Tensor] = {}
    out_b_left: dict[int, torch.Tensor] = {}
    out_b_right: dict[int, torch.Tensor] = {}

    for gid, gs in enumerate(group_state):
        group_items = [
            (gi, p)
            for gi, p in enumerate(pairs)
            if int(p.get("_group", -1)) == gid
        ]
        group_items.sort(key=lambda item: int(item[1].get("_local_idx", item[0])))
        if not group_items:
            continue

        A_f = torch.stack([p["A"].float() for _, p in group_items])
        B_f = torch.stack([p["B"].float() for _, p in group_items])
        u_A = torch.stack([p["u_A"].float() for _, p in group_items])
        u_B = torch.stack([p["u_B"].float() for _, p in group_items])
        SA_half_inv = gs["SA_half_inv"].float()
        SB_half_inv = gs["SB_half_inv"].float()

        # Unit-polar normalization used by the old chord-tight family.
        X_A_pre = SB_half_inv @ u_A
        X_B_pre = u_B @ SA_half_inv
        sigma_XA = _opnorm_batched(X_A_pre).clamp_min(1e-30)
        sigma_XB = _opnorm_batched(X_B_pre).clamp_min(1e-30)
        u_A = u_A / sigma_XA[:, None, None]
        u_B = u_B / sigma_XB[:, None, None]

        sigma_A = _opnorm_batched(A_f)
        sigma_B = _opnorm_batched(B_f)
        s_AB = sigma_A + sigma_B
        rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * float(lr))) / 2.0
        picard_coeff = (2.0 / (rho * s_AB).clamp_min(1e-30))[:, None, None]

        dA = torch.zeros_like(u_A)
        dB = torch.zeros_like(u_B)
        u_A_eff = u_A
        for k_iter in range(int(picard_iters)):
            if k_iter > 0:
                BT_dB_A = B_f.transpose(-2, -1) @ dB @ A_f
                B_dA_AT = B_f @ dA @ A_f.transpose(-2, -1)
                u_A_eff = u_A + picard_coeff * BT_dB_A
                u_B_eff = u_B + picard_coeff * B_dA_AT
            else:
                u_A_eff = u_A
                u_B_eff = u_B

            X_A = SB_half_inv @ u_A_eff
            X_B = u_B_eff @ SA_half_inv
            P_A = _polar_batched(X_A)
            P_B = _polar_batched(X_B)
            geo_A = SB_half_inv @ P_A
            geo_B = P_B @ SA_half_inv
            op_geoA = _opnorm_batched(geo_A).clamp_min(1e-30)
            op_geoB = _opnorm_batched(geo_B).clamp_min(1e-30)
            dA = -(rho[:, None, None] / op_geoA[:, None, None]) * geo_A
            dB = -(
                float(lora_plus_multiplier)
                * rho[:, None, None]
                / op_geoB[:, None, None]
            ) * geo_B

        for local_idx, (global_idx, _) in enumerate(group_items):
            _, _, Vh = torch.linalg.svd(dA[local_idx].float(), full_matrices=False)
            out_a[global_idx] = Vh.transpose(0, 1).contiguous()
            U_B, _, Vh_B = torch.linalg.svd(
                dB[local_idx].float(),
                full_matrices=False,
            )
            out_b_left[global_idx] = U_B.contiguous()
            out_b_right[global_idx] = Vh_B.transpose(0, 1).contiguous()

    return {"a_right": out_a, "b_left": out_b_left, "b_right": out_b_right}


@torch.no_grad()
def replay_chord_tight_clean_a_update_rowspaces(
    pair_state: dict,
    group_state: list[dict],
    *,
    lr: float,
    picard_iters: int,
    lora_plus_multiplier: float,
) -> dict[str, dict[int, torch.Tensor]]:
    """Replay the clean chord-tight update enough to get update subspaces.

    Clean chord-tight differs from the older path in three load-bearing ways:
    rho = lr / (sigma_A + sigma_B), the Adam directions are pre-rescaled in the
    whitened polar-input space, and Picard coupling uses coefficient 1/lr.
    Exact SVD polar is used here for analysis; it preserves the same rowspace
    as the production Newton-Schulz polar map while avoiding GPU-only replay.
    """
    pairs = _sorted_pairs(pair_state)
    out_a: dict[int, torch.Tensor] = {}
    out_b_left: dict[int, torch.Tensor] = {}
    out_b_right: dict[int, torch.Tensor] = {}

    for gid, gs in enumerate(group_state):
        group_items = [
            (gi, p)
            for gi, p in enumerate(pairs)
            if int(p.get("_group", -1)) == gid
        ]
        group_items.sort(key=lambda item: int(item[1].get("_local_idx", item[0])))
        if not group_items:
            continue

        A_f = torch.stack([p["A"].float() for _, p in group_items])
        B_f = torch.stack([p["B"].float() for _, p in group_items])
        u_A = torch.stack([p["u_A"].float() for _, p in group_items])
        u_B = torch.stack([p["u_B"].float() for _, p in group_items])
        SA_half_inv = gs["SA_half_inv"].float()
        SB_half_inv = gs["SB_half_inv"].float()

        sigma_A = _opnorm_batched(A_f)
        sigma_B = _opnorm_batched(B_f)
        rho = float(lr) / (sigma_A + sigma_B).clamp_min(1e-30)

        X_A = SB_half_inv @ u_A
        X_B = u_B @ SA_half_inv
        sigma_XA = _opnorm_batched(X_A).clamp_min(1e-30)
        sigma_XB = _opnorm_batched(X_B).clamp_min(1e-30)
        u_A = u_A / sigma_XA[:, None, None]
        u_B = u_B / sigma_XB[:, None, None]
        X_A = X_A / sigma_XA[:, None, None]
        X_B = X_B / sigma_XB[:, None, None]

        dA = torch.zeros_like(u_A)
        dB = torch.zeros_like(u_B)
        u_A_eff = u_A
        for k_iter in range(int(picard_iters)):
            if k_iter > 0:
                BT_dB_A = (B_f.transpose(-2, -1) @ dB) @ A_f
                B_dA_AT = B_f @ (dA @ A_f.transpose(-2, -1))
                u_A_eff = u_A + (1.0 / float(lr)) * BT_dB_A
                u_B_eff = u_B + (1.0 / float(lr)) * B_dA_AT
                X_A_eff = SB_half_inv @ u_A_eff
                X_B_eff = u_B_eff @ SA_half_inv
            else:
                X_A_eff = X_A
                X_B_eff = X_B

            P_A = _polar_batched(X_A_eff)
            P_B = _polar_batched(X_B_eff)
            geo_A = SB_half_inv @ P_A
            geo_B = P_B @ SA_half_inv
            op_geoA = _opnorm_batched(geo_A).clamp_min(1e-30)
            op_geoB = _opnorm_batched(geo_B).clamp_min(1e-30)
            dA = -(rho[:, None, None] / op_geoA[:, None, None]) * geo_A
            dB = -(
                float(lora_plus_multiplier)
                * rho[:, None, None]
                / op_geoB[:, None, None]
            ) * geo_B

        for local_idx, (global_idx, _) in enumerate(group_items):
            _, _, Vh = torch.linalg.svd(dA[local_idx].float(), full_matrices=False)
            out_a[global_idx] = Vh.transpose(0, 1).contiguous()
            U_B, _, Vh_B = torch.linalg.svd(
                dB[local_idx].float(),
                full_matrices=False,
            )
            out_b_left[global_idx] = U_B.contiguous()
            out_b_right[global_idx] = Vh_B.transpose(0, 1).contiguous()

    return {"a_right": out_a, "b_left": out_b_left, "b_right": out_b_right}


def _copy_snapshot_weights_into_model(model, pair_state: dict) -> list[str]:
    pairs = _sorted_pairs(pair_state)
    named = collect_lora_pairs_named(model, adapter_name="default")
    if len(named) != len(pairs):
        raise RuntimeError(f"model has {len(named)} LoRA pairs, snapshot has {len(pairs)}")
    names = []
    with torch.no_grad():
        for snap_pair, (A, B, name) in zip(pairs, named):
            A.copy_(snap_pair["A"].to(device=A.device, dtype=A.dtype))
            B.copy_(snap_pair["B"].to(device=B.device, dtype=B.dtype))
            names.append(name)
    return names


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> list[dict]:
    keys = ["module_kind", "d_in", "r"]
    fields = [
        "actual_ratio",
        "best_alpha",
        "mup_alpha",
        "quarter_alpha",
        "p_fit",
        "actual_over_isotropic",
        "actual_over_worst",
        "energy_concentration_vs_isotropic",
        "best_alpha_over_quarter",
        "best_alpha_over_keller",
        "best_alpha_over_mup",
        "b_input_actual_ratio",
        "b_cotangent_actual_ratio",
        "b_needed_expansion",
        "b_needed_expansion_over_mup",
        "b_expansion_q_fit",
    ]
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = dict(zip(keys, key))
        rec["n_modules"] = len(bucket)
        for field in fields:
            vals = torch.tensor([
                float(b[field])
                for b in bucket
                if field in b and math.isfinite(float(b[field]))
            ])
            if vals.numel() == 0:
                continue
            rec[field + "_median"] = float(vals.median())
            rec[field + "_p10"] = float(vals.quantile(0.10))
            rec[field + "_p90"] = float(vals.quantile(0.90))
        out.append(rec)
    return out


def _module_kind(name: str) -> str:
    if ".self_attn." in name:
        return name.split(".self_attn.", 1)[1].split(".", 1)[0]
    if ".mlp." in name:
        return name.split(".mlp.", 1)[1].split(".", 1)[0]
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="notebooks/snapshot_analysis/_data/muon_activation_isotropy_rows.csv")
    ap.add_argument("--summary-out", default="notebooks/snapshot_analysis/_data/muon_activation_isotropy_summary.csv")
    ap.add_argument(
        "--include-b-cotangent",
        action="store_true",
        help=(
            "Also run backward passes and measure projection of the cotangent "
            "signal onto the left singular subspace of dB."
        ),
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    run_args, meta = _load_run_args(ckpt_dir)
    device = torch.device(args.device)
    use_bf16 = bool(run_args.bf16 and device.type == "cuda")
    dtype = torch.bfloat16 if use_bf16 else None

    sd = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    pair_state = sd["pair_state"]
    group_state = sd["group_state"]

    optimizer_name = str(run_args.optimizer)
    picard_iters = (
        int(run_args.picard_iters_override)
        if run_args.picard_iters_override is not None
        else 1
    )
    if "chord-tight-clean" in optimizer_name:
        replay_variant = "chord-tight-clean"
        subspaces = replay_chord_tight_clean_a_update_rowspaces(
            pair_state,
            group_state,
            lr=float(run_args.lr),
            picard_iters=picard_iters,
            lora_plus_multiplier=float(run_args.lora_plus_multiplier),
        )
    else:
        replay_variant = "chord-tight"
        subspaces = replay_chord_tight_a_update_rowspaces(
            pair_state,
            group_state,
            lr=float(run_args.lr),
            picard_iters=picard_iters,
            lora_plus_multiplier=float(run_args.lora_plus_multiplier),
        )

    peft = build_peft_model(
        model_name=run_args.model_name,
        training_mode=run_args.training_mode,
        target_modules=parse_target_modules(run_args.target_modules),
        lora_r=run_args.lora_r,
        lora_alpha=run_args.lora_alpha,
        lora_dropout=run_args.lora_dropout,
        dtype=dtype,
        attn_implementation=run_args.attn_implementation,
        use_liger=False,
        liger_flce=False,
        gradient_checkpointing=False,
        compile_mode=None,
        device=device,
    )
    model = peft.bare_model
    model.eval()
    pair_names = _copy_snapshot_weights_into_model(model, pair_state)

    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_dataset = load_from_disk(str(Path(run_args.data_dir) / "eval"))
    collator = PadToMaxCollator(
        seq_length=run_args.max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    stats = {
        name: {"tokens": 0, "x_norm_sq": 0.0, "proj_norm_sq": 0.0}
        for name in pair_names
    }
    b_input_stats = {
        name: {"tokens": 0, "h_norm_sq": 0.0, "proj_norm_sq": 0.0}
        for name in pair_names
    }
    b_cotangent_stats = {
        name: {"tokens": 0, "g_norm_sq": 0.0, "proj_norm_sq": 0.0}
        for name in pair_names
    }
    handles = []
    mask_holder: dict[str, torch.Tensor] = {}

    lora_a_by_pair_name = {}
    lora_b_by_pair_name = {}
    for mod_name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            lora_a_by_pair_name[f"{mod_name}[default]"] = mod.lora_A["default"]
        if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
            lora_b_by_pair_name[f"{mod_name}[default]"] = mod.lora_B["default"]

    for idx, name in enumerate(pair_names):
        module = lora_a_by_pair_name[name]
        V_cpu = subspaces["a_right"][idx]

        def make_hook(pair_name=name, V=V_cpu):
            def hook(_module, inputs):
                x = inputs[0].detach()
                mask = mask_holder["mask"].to(x.device)
                x = x[mask].float()
                if x.numel() == 0:
                    return
                V_dev = V.to(device=x.device, dtype=torch.float32)
                proj = x @ V_dev
                stats[pair_name]["tokens"] += int(x.shape[0])
                stats[pair_name]["x_norm_sq"] += float(x.square().sum().cpu())
                stats[pair_name]["proj_norm_sq"] += float(proj.square().sum().cpu())

            return hook

        handles.append(module.register_forward_pre_hook(make_hook()))

        module_b = lora_b_by_pair_name[name]
        Vb_cpu = subspaces["b_right"][idx]
        Ub_cpu = subspaces["b_left"][idx]

        def make_b_input_hook(pair_name=name, V=Vb_cpu):
            def hook(_module, inputs):
                h = inputs[0].detach()
                mask = mask_holder["mask"].to(h.device)
                h = h[mask].float()
                if h.numel() == 0:
                    return
                V_dev = V.to(device=h.device, dtype=torch.float32)
                proj = h @ V_dev
                b_input_stats[pair_name]["tokens"] += int(h.shape[0])
                b_input_stats[pair_name]["h_norm_sq"] += float(h.square().sum().cpu())
                b_input_stats[pair_name]["proj_norm_sq"] += float(
                    proj.square().sum().cpu()
                )

            return hook

        handles.append(module_b.register_forward_pre_hook(make_b_input_hook()))

        if args.include_b_cotangent:
            def make_b_backward_hook(pair_name=name, U=Ub_cpu):
                def hook(_module, _grad_input, grad_output):
                    if not grad_output or grad_output[0] is None:
                        return
                    g = grad_output[0].detach()
                    mask = mask_holder["mask"].to(g.device)
                    g = g[mask].float()
                    if g.numel() == 0:
                        return
                    U_dev = U.to(device=g.device, dtype=torch.float32)
                    proj = g @ U_dev
                    b_cotangent_stats[pair_name]["tokens"] += int(g.shape[0])
                    b_cotangent_stats[pair_name]["g_norm_sq"] += float(
                        g.square().sum().cpu()
                    )
                    b_cotangent_stats[pair_name]["proj_norm_sq"] += float(
                        proj.square().sum().cpu()
                    )

                return hook

            handles.append(module_b.register_full_backward_hook(make_b_backward_hook()))

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.num_batches):
            break
        mask_holder["mask"] = batch["attention_mask"].bool()
        batch = {k: v.to(device) for k, v in batch.items()}
        if args.include_b_cotangent:
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("model output did not include loss")
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                _ = model(**batch)

    for h in handles:
        h.remove()

    rows = []
    pairs = _sorted_pairs(pair_state)
    for idx, name in enumerate(pair_names):
        st = stats[name]
        if st["tokens"] == 0 or st["x_norm_sq"] <= 0.0:
            continue
        A = pairs[idx]["A"]
        B = pairs[idx]["B"]
        r = int(A.shape[0])
        d_in = int(A.shape[1])
        d_out = int(B.shape[0])
        actual = math.sqrt((d_in / r) * st["proj_norm_sq"] / st["x_norm_sq"])
        worst = math.sqrt(d_in / r)
        mup_alpha = math.sqrt(r / d_in)
        best_alpha = 1.0 / max(actual, 1e-30)
        quarter_alpha = (r / d_in) ** 0.25
        p_fit = math.log(max(best_alpha, 1e-30)) / math.log(r / d_in)
        b_in = b_input_stats[name]
        b_in_rank = int(subspaces["b_right"][idx].shape[1])
        if b_in["tokens"] > 0 and b_in["h_norm_sq"] > 0.0:
            b_input_actual = math.sqrt(
                (r / b_in_rank) * b_in["proj_norm_sq"] / b_in["h_norm_sq"]
            )
        else:
            b_input_actual = float("nan")
        b_grad = b_cotangent_stats[name]
        b_left_rank = int(subspaces["b_left"][idx].shape[1])
        b_cot_actual = float("nan")
        b_needed_expansion = float("nan")
        b_needed_over_mup = float("nan")
        b_expansion_q = float("nan")
        b_mup_expansion = math.sqrt(d_out / b_left_rank)
        if b_grad["tokens"] > 0 and b_grad["g_norm_sq"] > 0.0:
            b_cot_actual = math.sqrt(
                (d_out / b_left_rank)
                * b_grad["proj_norm_sq"]
                / b_grad["g_norm_sq"]
            )
            b_needed_expansion = b_mup_expansion / max(b_cot_actual, 1e-30)
            b_needed_over_mup = b_needed_expansion / b_mup_expansion
            b_expansion_q = (
                math.log(max(b_needed_expansion, 1e-30))
                / math.log(d_out / b_left_rank)
                if d_out != b_left_rank
                else float("nan")
            )
        rows.append({
            "checkpoint": str(ckpt_dir),
            "checkpoint_step": int(meta["step"]),
            "optimizer": optimizer_name,
            "replay_variant": replay_variant,
            "lr": float(run_args.lr),
            "picard_iters": picard_iters,
            "model_name": run_args.model_name,
            "data_dir": run_args.data_dir,
            "data_pipeline_version": run_args.data_pipeline_version,
            "max_seq_length": int(run_args.max_seq_length),
            "pair": idx,
            "name": name,
            "module_kind": _module_kind(name),
            "tokens": st["tokens"],
            "r": r,
            "d_in": d_in,
            "d_out": d_out,
            "actual_ratio": actual,
            "isotropic_ratio": 1.0,
            "worst_ratio": worst,
            "actual_over_isotropic": actual,
            "actual_over_worst": actual / worst,
            "best_alpha": best_alpha,
            "keller_alpha": 1.0,
            "mup_alpha": mup_alpha,
            "quarter_alpha": quarter_alpha,
            "p_fit": p_fit,
            "best_alpha_over_keller": best_alpha,
            "best_alpha_over_quarter": best_alpha / quarter_alpha,
            "best_alpha_over_mup": best_alpha / mup_alpha,
            "energy_concentration_vs_isotropic": actual * actual,
            "b_input_tokens": b_in["tokens"],
            "b_input_rank": b_in_rank,
            "b_input_actual_ratio": b_input_actual,
            "b_cotangent_tokens": b_grad["tokens"],
            "b_cotangent_rank": b_left_rank,
            "b_cotangent_actual_ratio": b_cot_actual,
            "b_mup_expansion": b_mup_expansion,
            "b_needed_expansion": b_needed_expansion,
            "b_needed_expansion_over_mup": b_needed_over_mup,
            "b_expansion_q_fit": b_expansion_q,
        })

    summary = _summary(rows)
    _write_csv(Path(args.out), rows)
    _write_csv(Path(args.summary_out), summary)

    print(f"rows={len(rows)} batches={args.num_batches} batch_size={args.batch_size}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    for rec in summary:
        print(
            " ".join(
                [
                    f"{rec['module_kind']} d={rec['d_in']} r={rec['r']}",
                    f"actual={rec['actual_ratio_median']:.3f}",
                    f"best_alpha={rec['best_alpha_median']:.3f}",
                    f"muP={rec['mup_alpha_median']:.3f}",
                    f"best/muP={rec['best_alpha_over_mup_median']:.2f}",
                ]
            )
        )
        if "b_cotangent_actual_ratio_median" in rec:
            print(
                " ".join(
                    [
                        f"  Bleft={rec['b_cotangent_actual_ratio_median']:.3f}",
                        f"Bneed={rec['b_needed_expansion_median']:.3f}",
                        f"Bneed/muP={rec['b_needed_expansion_over_mup_median']:.2f}",
                        f"Bq={rec['b_expansion_q_fit_median']:.3f}",
                    ]
                )
            )


if __name__ == "__main__":
    main()
