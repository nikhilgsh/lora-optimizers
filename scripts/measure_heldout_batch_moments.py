#!/usr/bin/env python3
"""Measure centered and uncentered held-out-batch P_A/Q_B moments.

This is a read-only checkpoint diagnostic.  It loads the same first N eval
batches used by ``--optim_heldout_probe``, reconstructs one factor-gradient
pair per batch, and never applies an optimizer step.

The observation unit is a held-out *batch*, not an example.  For batch i,
``g_i`` is the gradient of that batch's mean supervised-token NLL and
``w_i`` is its supervised-token count divided by the total across batches.
With the checkpoint's stored diagonal P/Q metric held fixed, the measured
small-side targets are

    P_A = sum_i w_i (gA_i Q^-1 gA_i^T) / d_in
    Q_B = sum_i w_i (gB_i^T P^-1 gB_i) / d_out.

The artifact keeps the eight individual float32 factor gradients, the raw
stored P_A/Q_B state, the three decomposed moment tensors, weak-factor-mode
projectors, and per-pair statistics.  The decomposition is computed twice
(direct centered covariance and uncentered minus mean outer) and must pass a
relative-residual check before anything is written.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from lora_playground.checkpoint import load_checkpoint
from lora_playground.data import PadToMaxCollator
from lora_playground.optim import (
    CurvatureWhitenLoRA,
    build_optimizer,
    gram_ns_inv_sqrt,
)
from lora_playground.spectral import (
    lambda_max_power_iter_psd_batched,
    sigma_max_power_iter_batched,
)
from lora_playground.train import parse_target_modules
from lora_playground.training_kernel import (
    batch_to_device,
    build_peft_model,
    count_tokens,
)
from lora_playground.utils import collect_lora_pairs_named


SCHEMA_VERSION = 1
PULLBACK_SCHEMA_VERSION = 1
MOMENT_IDENTITY_RTOL = 5e-10


def _read_source_config(path: Path) -> dict:
    """Read the config event that generated the named prior probe."""
    with path.open() as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "config" and isinstance(event.get("_cli_args"), dict):
                return event
    raise ValueError(f"no config event with _cli_args found in {path}")


def _git_provenance() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            args, check=True, text=True, stdout=subprocess.PIPE
        ).stdout.strip()

    return {
        "repo_root": run("git", "rev-parse", "--show-toplevel"),
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_status": run("git", "status", "--short").splitlines(),
    }


def _build_optimizer_from_cli(bare_model, cfg: dict):
    """Forward the recorded training config through the canonical factory."""
    signature = inspect.signature(build_optimizer)
    kwargs = {}
    for name in signature.parameters:
        if name == "model":
            continue
        if name == "optimizer_type":
            kwargs[name] = cfg["optimizer"]
        elif name == "muon_alpha":
            kwargs[name] = cfg["lora_alpha"]
        elif name == "muon_rank":
            kwargs[name] = cfg["lora_r"]
        elif name in cfg:
            kwargs[name] = cfg[name]
    return build_optimizer(bare_model, **kwargs)


def _frob_cos(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.double().flatten()
    y = y.double().flatten()
    denom = x.norm() * y.norm()
    if not bool(torch.isfinite(denom)) or float(denom) <= 1e-30:
        return None
    return float((x @ y / denom).clamp(-1.0, 1.0))


def _trace(x: torch.Tensor) -> float:
    return float(torch.diagonal(x.double()).sum())


def _trace_share(numer: torch.Tensor, denom: torch.Tensor) -> float | None:
    den = _trace(denom)
    if not math.isfinite(den) or abs(den) <= 1e-30:
        return None
    return _trace(numer) / den


def _weak_projector(factor: torch.Tensor, *, side: str) -> torch.Tensor:
    """Projector onto the bottom half of A's row or B's column modes."""
    factor = factor.double()
    gram = factor @ factor.T if side == "A" else factor.T @ factor
    _, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    n_weak = max(1, gram.shape[0] // 2)
    weak = vectors[:, :n_weak]
    return weak @ weak.T


def _weak_alignment(moment: torch.Tensor, projector: torch.Tensor) -> dict:
    moment = 0.5 * (moment.double() + moment.double().T)
    weak_trace = float(torch.sum(moment * projector))
    total_trace = _trace(moment)
    return {
        "weak_trace_share": (
            weak_trace / total_trace
            if math.isfinite(total_trace) and abs(total_trace) > 1e-30
            else None
        ),
        "weak_projector_frob_cos": _frob_cos(moment, projector),
    }


def _side_moments(
    grads: torch.Tensor,
    weights: torch.Tensor,
    diag_inv: torch.Tensor,
    *,
    side: str,
) -> dict:
    """Direct weighted moment decomposition in float64 for one LoRA pair."""
    grads = grads.double()
    weights = weights.double()
    diag_inv = diag_inv.double()
    mean = torch.einsum("g,gij->ij", weights, grads)
    centered_grads = grads - mean.unsqueeze(0)

    if side == "A":
        dim = grads.shape[-1]

        def outer(x: torch.Tensor) -> torch.Tensor:
            return (x * diag_inv[None, None, :]) @ x.transpose(-2, -1) / dim

        mean_outer = (mean * diag_inv[None, :]) @ mean.T / dim
    elif side == "B":
        dim = grads.shape[-2]

        def outer(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(-2, -1) @ (x * diag_inv[None, :, None]) / dim

        mean_outer = mean.T @ (mean * diag_inv[:, None]) / dim
    else:
        raise ValueError(f"unknown side {side!r}")

    uncentered = torch.einsum("g,gij->ij", weights, outer(grads))
    centered = torch.einsum("g,gij->ij", weights, outer(centered_grads))
    residual = uncentered - centered - mean_outer
    residual_abs = float(residual.norm())
    residual_rel = residual_abs / max(float(uncentered.norm()), 1e-30)
    return {
        "mean_grad": mean,
        "uncentered": uncentered,
        "centered": centered,
        "mean_outer": mean_outer,
        "identity_abs_residual": residual_abs,
        "identity_rel_residual": residual_rel,
    }


def _pair_stats(
    stored: torch.Tensor,
    moments: dict,
    weak_projector: torch.Tensor,
) -> dict:
    stored_sym = 0.5 * (stored.double() + stored.double().T)
    stats = {
        "mean_outer_trace_share": _trace_share(
            moments["mean_outer"], moments["uncentered"]
        ),
        "identity_abs_residual": moments["identity_abs_residual"],
        "identity_rel_residual": moments["identity_rel_residual"],
        "stored_frob_cos_uncentered": _frob_cos(stored_sym, moments["uncentered"]),
        "stored_frob_cos_centered": _frob_cos(stored_sym, moments["centered"]),
        "stored_frob_cos_mean_outer": _frob_cos(stored_sym, moments["mean_outer"]),
    }
    for label, matrix in {
        "stored": stored_sym,
        "uncentered": moments["uncentered"],
        "centered": moments["centered"],
        "mean_outer": moments["mean_outer"],
    }.items():
        stats.update({
            f"{label}_{key}": value
            for key, value in _weak_alignment(matrix, weak_projector).items()
        })
    return stats


def _distribution(values: list[float | None]) -> dict:
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _aggregate(pair_payloads: list[dict]) -> dict:
    summary = {}
    for side in ("P_A", "Q_B"):
        keys = sorted(pair_payloads[0]["stats"][side])
        summary[side] = {
            key: _distribution([pair["stats"][side][key] for pair in pair_payloads])
            for key in keys
        }
    return summary


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _quantiles(values: list[float]) -> dict:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    last = len(finite) - 1
    return {
        "n": len(finite),
        "median": statistics.median(finite),
        "q25": finite[round(0.25 * last)],
        "q75": finite[round(0.75 * last)],
    }


def _resolve_optimizer_state_path(checkpoint: Path) -> Path:
    path = checkpoint / "optimizer.pt" if checkpoint.is_dir() else checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"optimizer state not found: {path}")
    return path


def _pullback_stub(cfg: dict) -> CurvatureWhitenLoRA:
    """Minimal owner for the exact production direction helpers."""
    unsupported = {
        "cw_picard_iters": cfg.get("cw_picard_iters", 1) != 1,
        "cw_solved_rho": bool(cfg.get("cw_solved_rho", False)),
        "cw_unpinned": bool(cfg.get("cw_unpinned", False)),
        "flat_outer": bool(cfg.get("flat_outer", False)),
        "use_polar": not bool(cfg.get("use_polar", True)),
    }
    active = [name for name, enabled in unsupported.items() if enabled]
    if active:
        raise ValueError(
            "offline pullback reconstruction does not implement: " + ", ".join(active)
        )
    if os.environ.get("LORA_MULTIMOMENT_RESCALE", "0") == "1":
        raise ValueError("offline pullback reconstruction requires LORA_MULTIMOMENT_RESCALE=0")

    stub = CurvatureWhitenLoRA.__new__(CurvatureWhitenLoRA)
    stub.eps = float(cfg.get("eps", 1e-8))
    stub.delta = float(cfg["delta"])
    stub.rdinv_delta = cfg.get("rdinv_delta")
    stub.rdinv_variant = cfg.get("rdinv_variant", "A")
    stub.ns_steps = int(cfg["ns_steps"])
    stub.polar_method = cfg["polar_method"]
    stub.higham_iters = int(cfg["higham_iters"])
    stub.alg1_magnitude_floor = 1e-12
    stub.lora_plus_multiplier = float(cfg.get("lora_plus_multiplier", 1.0))
    stub.cw_factor_a = float(cfg.get("cw_factor_a", 0.0))
    stub.cw_factor_b = float(cfg.get("cw_factor_b", 0.0))
    return stub


def _spectrum_stats(eigenvalues: torch.Tensor, n_weak: int) -> dict:
    eigenvalues = eigenvalues.clamp_min(0)
    normalized = eigenvalues / eigenvalues.max().clamp_min(1e-30)
    return {
        "min_over_max": float(normalized.min()),
        "median_over_max": float(normalized.median()),
        "bottom_half_trace_share": float(
            normalized[:n_weak].sum() / normalized.sum().clamp_min(1e-30)
        ),
        "trace_over_max": float(normalized.sum()),
        "stable_rank": float(normalized.square().sum()),
    }


def _slot_comparison(
    factorwise: torch.Tensor,
    product: torch.Tensor,
    partner: torch.Tensor,
    n_weak: int,
) -> dict:
    """Exact-eigendecomposition analysis; direction construction stays production-exact."""
    factorwise = _sym(factorwise.double())
    product = _sym(product.double())
    partner = _sym(partner.double())
    ef, uf = torch.linalg.eigh(factorwise)
    ep, up = torch.linalg.eigh(product)
    eh, uh = torch.linalg.eigh(partner)
    fn = factorwise / ef.max().clamp_min(1e-30)
    pn = product / ep.max().clamp_min(1e-30)

    def overlap(u: torch.Tensor, v: torch.Tensor) -> float:
        return float((u[:, :n_weak].T @ v[:, :n_weak]).square().sum() / n_weak)

    return {
        "frob_cos": float(
            (fn * pn).sum() / (fn.norm() * pn.norm()).clamp_min(1e-30)
        ),
        "commutator_rel": float(
            (fn @ pn - pn @ fn).norm()
            / (fn.norm() * pn.norm()).clamp_min(1e-30)
        ),
        "weak_overlap_factorwise_product": overlap(uf, up),
        "weak_overlap_factorwise_partner": overlap(uf, uh),
        "weak_overlap_product_partner": overlap(up, uh),
        "factorwise_spectrum": _spectrum_stats(ef, n_weak),
        "product_spectrum": _spectrum_stats(ep, n_weak),
        "partner_spectrum": _spectrum_stats(eh, n_weak),
    }


def _log_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / x.square().sum().clamp_min(1e-30))


def _mapped_inner(
    side_x: str,
    x: torch.Tensor,
    side_z: str,
    z: torch.Tensor,
    geometry: dict,
    *,
    weighted: bool,
) -> float:
    """Inner product after the skinny factors are mapped into product space."""
    A, B = geometry["A"], geometry["B"]
    if weighted:
        q, p = geometry["Q_eff"], geometry["P_eff"]
        cb, ca = geometry["C_B"], geometry["C_A"]
    else:
        q, p = torch.ones_like(geometry["Q_eff"]), torch.ones_like(geometry["P_eff"])
        cb, ca = B.T @ B, A @ A.T
    if side_x == "A" and side_z == "A":
        return float((x * (cb @ z) * q.unsqueeze(0)).sum())
    if side_x == "B" and side_z == "B":
        return float((x * ((p.unsqueeze(1) * z) @ ca)).sum())
    if side_x == "A" and side_z == "B":
        mapped = (B.T @ (p.unsqueeze(1) * z)) @ (A * q.unsqueeze(0))
        return float((x * mapped).sum())
    return _mapped_inner(side_z, z, side_x, x, geometry, weighted=weighted)


def _direction_pair_metrics(
    dA: torch.Tensor,
    dB: torch.Tensor,
    gA: torch.Tensor,
    gB: torch.Tensor,
    geometry: dict,
) -> dict:
    """Factor-mode, mapped-product, and first-order metrics for one pair."""
    k = geometry["n_weak"]
    ub, ua = geometry["U_B"], geometry["U_A"]
    proj_b = ub[:, :k] @ ub[:, :k].T
    proj_a = ua[:, :k] @ ua[:, :k].T
    components = {
        "A_weak": ("A", proj_b @ dA),
        "A_strong": ("A", dA - proj_b @ dA),
        "B_weak": ("B", dB @ proj_a),
        "B_strong": ("B", dB - dB @ proj_a),
    }
    factor = {
        "A_weak_energy": float(components["A_weak"][1].square().sum()),
        "A_total_energy": float(dA.square().sum()),
        "B_weak_energy": float(components["B_weak"][1].square().sum()),
        "B_total_energy": float(dB.square().sum()),
    }
    first_order = {
        name: float(((gA if side == "A" else gB) * value).sum())
        for name, (side, value) in components.items()
    }

    modewise = {}
    for side, update, basis, eigenvalues in (
        ("A", dA, ub, geometry["eig_B"]),
        ("B", dB, ua, geometry["eig_A"]),
    ):
        modes = basis.T @ update if side == "A" else update @ basis
        mode_energy = modes.square().sum(dim=1 if side == "A" else 0)
        singular = eigenvalues.clamp_min(1e-30).sqrt()
        amplitude = mode_energy.clamp_min(1e-30).sqrt()
        post_energy = mode_energy * eigenvalues
        modewise.update({
            f"{side}_factor_logamp_vs_logsing_slope": _log_slope(
                singular.log(), amplitude.log()
            ),
            f"{side}_post_logamp_vs_logsing_slope": _log_slope(
                singular.log(), (amplitude * singular).log()
            ),
            f"{side}_factor_weak_strong_permode_energy_ratio": float(
                mode_energy[:k].mean() / mode_energy[k:].mean().clamp_min(1e-30)
            ),
            f"{side}_post_weak_strong_permode_energy_ratio": float(
                post_energy[:k].mean() / post_energy[k:].mean().clamp_min(1e-30)
            ),
        })

    mapped = {}
    for metric, weighted in (("euclidean", False), ("pq_weighted", True)):
        def ip(left: str, right: str) -> float:
            sl, xl = components[left]
            sr, xr = components[right]
            return _mapped_inner(sl, xl, sr, xr, geometry, weighted=weighted)

        energies = {name: ip(name, name) for name in components}
        a_cross = 2.0 * ip("A_weak", "A_strong")
        b_cross = 2.0 * ip("B_weak", "B_strong")
        e_a = energies["A_weak"] + energies["A_strong"] + a_cross
        e_b = energies["B_weak"] + energies["B_strong"] + b_cross
        weak_ab_cross = 2.0 * ip("A_weak", "B_weak")
        strong_ab_cross = 2.0 * ip("A_strong", "B_strong")
        e_weak = energies["A_weak"] + energies["B_weak"] + weak_ab_cross
        e_strong = energies["A_strong"] + energies["B_strong"] + strong_ab_cross
        weak_strong_cross = 2.0 * (
            ip("A_weak", "A_strong")
            + ip("A_weak", "B_strong")
            + ip("B_weak", "A_strong")
            + ip("B_weak", "B_strong")
        )
        mapped[metric] = {
            **{f"{name}_energy": value for name, value in energies.items()},
            "A_weak_strong_cross": a_cross,
            "B_weak_strong_cross": b_cross,
            "A_branch_energy": e_a,
            "B_branch_energy": e_b,
            "weak_AB_cross": weak_ab_cross,
            "strong_AB_cross": strong_ab_cross,
            "weak_energy": e_weak,
            "strong_energy": e_strong,
            "weak_strong_cross": weak_strong_cross,
            "tangent_energy": e_weak + e_strong + weak_strong_cross,
        }
    return {
        "factor": factor,
        "first_order": first_order,
        "modewise": modewise,
        "mapped": mapped,
    }


def _summarize_pullback(raw: dict) -> dict:
    pairs = raw["pairs"]
    summary = {
        "schema_version": PULLBACK_SCHEMA_VERSION,
        "raw_json": raw["raw_json"],
        "semantics": raw["semantics"],
        "rank": raw["rank"],
        "pair_count": len(pairs),
        "timing": raw["timing"],
    }

    demand = {}
    for side in ("A", "B"):
        for kind in ("mean", "uncentered_batch"):
            numer = sum(p["heldout_demand"][f"{side}_{kind}_weak_energy"] for p in pairs)
            denom = sum(p["heldout_demand"][f"{side}_{kind}_total_energy"] for p in pairs)
            demand[f"{side}_{kind}_weak_global"] = numer / denom
    summary["heldout_partner_weak_demand"] = demand

    slot_summary = {}
    scalar_slot_keys = (
        "frob_cos",
        "commutator_rel",
        "weak_overlap_factorwise_product",
        "weak_overlap_factorwise_partner",
        "weak_overlap_product_partner",
    )
    for side in ("A", "B"):
        records = [p["slot_comparison"][side] for p in pairs]
        slot_summary[side] = {
            key: _quantiles([record[key] for record in records])
            for key in scalar_slot_keys
        }
        for spectrum in ("factorwise_spectrum", "product_spectrum", "partner_spectrum"):
            for key in records[0][spectrum]:
                slot_summary[side][f"{spectrum}_{key}"] = _quantiles(
                    [record[spectrum][key] for record in records]
                )
    summary["slot_comparison"] = slot_summary

    directions = {}
    for input_mode in ("frozen_momentum", "heldout_nesterov"):
        directions[input_mode] = {}
        for label in ("factorwise", "product", "identity"):
            records = [p["directions"][input_mode][label] for p in pairs]
            factor = {}
            for side in ("A", "B"):
                weak = sum(r["factor"][f"{side}_weak_energy"] for r in records)
                total = sum(r["factor"][f"{side}_total_energy"] for r in records)
                factor[f"{side}_weak_global"] = weak / total
            first_order = {
                key: sum(r["first_order"][key] for r in records)
                for key in records[0]["first_order"]
            }
            first_order["weak"] = first_order["A_weak"] + first_order["B_weak"]
            first_order["strong"] = first_order["A_strong"] + first_order["B_strong"]
            first_order["total"] = first_order["weak"] + first_order["strong"]
            modewise = {
                key: _quantiles([r["modewise"][key] for r in records])
                for key in records[0]["modewise"]
            }
            mapped = {}
            for metric in ("euclidean", "pq_weighted"):
                total = {
                    key: sum(r["mapped"][metric][key] for r in records)
                    for key in records[0]["mapped"][metric]
                }
                mapped[metric] = {
                    "A_branch_weak_gross_global": total["A_weak_energy"]
                    / (total["A_weak_energy"] + total["A_strong_energy"]),
                    "A_branch_weak_net_global": (
                        total["A_weak_energy"] + 0.5 * total["A_weak_strong_cross"]
                    ) / total["A_branch_energy"],
                    "B_branch_weak_gross_global": total["B_weak_energy"]
                    / (total["B_weak_energy"] + total["B_strong_energy"]),
                    "B_branch_weak_net_global": (
                        total["B_weak_energy"] + 0.5 * total["B_weak_strong_cross"]
                    ) / total["B_branch_energy"],
                    "tangent_weak_gross_global": total["weak_energy"]
                    / (total["weak_energy"] + total["strong_energy"]),
                    "tangent_weak_net_global": (
                        total["weak_energy"] + 0.5 * total["weak_strong_cross"]
                    ) / total["tangent_energy"],
                    "weak_AB_cancellation_ratio_global": total["weak_energy"]
                    / (
                        total["A_weak_energy"] + total["B_weak_energy"]
                    ),
                    "strong_AB_cancellation_ratio_global": total["strong_energy"]
                    / (
                        total["A_strong_energy"] + total["B_strong_energy"]
                    ),
                    "full_AB_cancellation_ratio_global": total["tangent_energy"]
                    / (total["A_branch_energy"] + total["B_branch_energy"]),
                    "weak_strong_cross_over_total_global": total["weak_strong_cross"]
                    / total["tangent_energy"],
                }
            directions[input_mode][label] = {
                "factor": factor,
                "first_order": first_order,
                "modewise": modewise,
                "mapped": mapped,
            }
    summary["directions"] = directions
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def analyze_pullback(args: argparse.Namespace) -> None:
    """Offline partner-mode, slot-mismatch, and pullback-cancellation audit."""
    started = time.perf_counter()
    artifact_path = args.analyze_artifact.resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    source = artifact["source_config_event"]
    cli = source["_cli_args"]
    opt_cfg = source["optimizer_config"]
    expected_rank = args.rank if args.rank is not None else int(cli["lora_r"])
    checkpoint = args.checkpoint or Path(artifact["checkpoint"])
    optimizer_path = _resolve_optimizer_state_path(checkpoint.resolve())
    checkpoint_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    states = checkpoint_state["pair_state"]
    pairs = artifact["pairs"]
    if len(pairs) != len(states):
        raise ValueError(f"artifact/checkpoint pair-count mismatch: {len(pairs)} != {len(states)}")
    if int(cli["lora_r"]) != expected_rank:
        raise ValueError(f"source rank {cli['lora_r']} != requested rank {expected_rank}")
    steps = {int(state["step"]) for state in states.values()}
    if steps != {int(args.expected_step)}:
        raise ValueError(f"checkpoint pair-state steps {steps} != {args.expected_step}")
    if opt_cfg["precond"] != "factorwise":
        raise ValueError("slot-mismatch audit requires a factorwise checkpoint")
    stub = _pullback_stub(opt_cfg)
    load_seconds = time.perf_counter() - started

    raw_path = args.analysis_out
    if raw_path is None:
        raw_path = artifact_path.with_name(
            artifact_path.name.removesuffix(".pt") + ".pullback.raw.json"
        )
    raw_path = raw_path.resolve()
    summary_path = raw_path.with_name(
        raw_path.name.removesuffix(".raw.json") + ".summary.json"
    )
    weights = artifact["heldout_batch_weights"].float()
    geometry = []
    raw_pairs = []
    geometry_started = time.perf_counter()
    max_state_error = {"P_A": 0.0, "Q_B": 0.0, "Q_inv": 0.0, "P_inv": 0.0}
    expected_state_keys = set(range(len(pairs)))
    if set(states) != expected_state_keys:
        raise ValueError(
            "checkpoint pair-state keys are not the expected contiguous indices: "
            f"got {sorted(states)[:8]}..."
        )
    for index, pair in enumerate(pairs):
        state = states[index]
        A, B = pair["factor_A"].float(), pair["factor_B"].float()
        if A.shape[0] != expected_rank or B.shape[1] != expected_rank:
            raise ValueError(
                f"pair {index} rank mismatch: A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
        max_state_error["P_A"] = max(
            max_state_error["P_A"],
            float((pair["P_A_stored_raw"].float() - state["P_A"].float()).abs().max()),
        )
        max_state_error["Q_B"] = max(
            max_state_error["Q_B"],
            float((pair["Q_B_stored_raw"].float() - state["Q_B"].float()).abs().max()),
        )
        q_isqrt = stub._rdinv(pair["Q_stored"].float())
        p_isqrt = stub._rdinv(pair["P_stored"].float())
        max_state_error["Q_inv"] = max(
            max_state_error["Q_inv"],
            float((pair["Q_inv_used"].float() - q_isqrt.square()).abs().max()),
        )
        max_state_error["P_inv"] = max(
            max_state_error["P_inv"],
            float((pair["P_inv_used"].float() - p_isqrt.square()).abs().max()),
        )
        q_eff, p_eff = q_isqrt.square().reciprocal(), p_isqrt.square().reciprocal()
        eig_b, u_b = torch.linalg.eigh(_sym(B.T @ B))
        eig_a, u_a = torch.linalg.eigh(_sym(A @ A.T))
        n_weak = expected_rank // 2
        c_b = B.T @ (p_eff.unsqueeze(1) * B)
        c_a = (A * q_eff.unsqueeze(0)) @ A.T
        item_geometry = {
            "A": A,
            "B": B,
            "eig_B": eig_b.clamp_min(0),
            "U_B": u_b,
            "eig_A": eig_a.clamp_min(0),
            "U_A": u_a,
            "n_weak": n_weak,
            "Q_isqrt": q_isqrt,
            "P_isqrt": p_isqrt,
            "Q_eff": q_eff,
            "P_eff": p_eff,
            "C_B": c_b,
            "C_A": c_a,
        }
        geometry.append(item_geometry)

        gA = pair["heldout_batch_mean_gA"].float()
        gB = pair["heldout_batch_mean_gB"].float()
        gA_modes = u_b.T @ gA
        gB_modes = gB @ u_a
        batch_gA = pair["heldout_batch_gA"].float()
        batch_gB = pair["heldout_batch_gB"].float()
        batch_gA_modes = torch.einsum("rk,brd->bkd", u_b[:, :n_weak], batch_gA)
        batch_gB_modes = torch.einsum("bdr,rk->bdk", batch_gB, u_a[:, :n_weak])
        raw_pairs.append({
            "pair_index": int(pair["pair_index"]),
            "pair_name": pair["pair_name"],
            "A_shape": list(A.shape),
            "B_shape": list(B.shape),
            "heldout_demand": {
                "A_mean_weak_energy": float(gA_modes[:n_weak].square().sum()),
                "A_mean_total_energy": float(gA.square().sum()),
                "B_mean_weak_energy": float(gB_modes[:, :n_weak].square().sum()),
                "B_mean_total_energy": float(gB.square().sum()),
                "A_uncentered_batch_weak_energy": float(
                    (weights * batch_gA_modes.square().flatten(1).sum(1)).sum()
                ),
                "A_uncentered_batch_total_energy": float(
                    (weights * batch_gA.square().flatten(1).sum(1)).sum()
                ),
                "B_uncentered_batch_weak_energy": float(
                    (weights * batch_gB_modes.square().flatten(1).sum(1)).sum()
                ),
                "B_uncentered_batch_total_energy": float(
                    (weights * batch_gB.square().flatten(1).sum(1)).sum()
                ),
            },
            "slot_comparison": {
                "A": _slot_comparison(
                    pair["P_A_stored_raw"].float(), c_b, B.T @ B, n_weak
                ),
                "B": _slot_comparison(
                    pair["Q_B_stored_raw"].float(), c_a, A @ A.T, n_weak
                ),
            },
            "directions": {"frozen_momentum": {}, "heldout_nesterov": {}},
        })
    geometry_seconds = time.perf_counter() - geometry_started

    direction_started = time.perf_counter()
    groups: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(pairs):
        key = (pair["factor_A"].shape[1], pair["factor_B"].shape[0])
        groups.setdefault(key, []).append(index)
    beta1 = float(opt_cfg["betas"][0])
    lr = float(opt_cfg["lr"])
    for indices in groups.values():
        Aw = torch.stack([geometry[i]["A"] for i in indices])
        Bw = torch.stack([geometry[i]["B"] for i in indices])
        q_isqrt = torch.stack([geometry[i]["Q_isqrt"] for i in indices])
        p_isqrt = torch.stack([geometry[i]["P_isqrt"] for i in indices])
        pa = _sym(torch.stack([pairs[i]["P_A_stored_raw"].float() for i in indices]))
        qb = _sym(torch.stack([pairs[i]["Q_B_stored_raw"].float() for i in indices]))
        lam_a, _ = lambda_max_power_iter_psd_batched(pa, n_iters=8)
        lam_b, _ = lambda_max_power_iter_psd_batched(qb, n_iters=8)
        pa_half = gram_ns_inv_sqrt(
            pa / lam_a.clamp_min(1e-30).view(-1, 1, 1),
            nsteps=stub.higham_iters,
            eps=stub.delta,
            eps_relative=True,
        )
        qb_half = gram_ns_inv_sqrt(
            qb / lam_b.clamp_min(1e-30).view(-1, 1, 1),
            nsteps=stub.higham_iters,
            eps=stub.delta,
            eps_relative=True,
        )
        c_b = torch.stack([geometry[i]["C_B"] for i in indices])
        c_a = torch.stack([geometry[i]["C_A"] for i in indices])
        c_b_half = gram_ns_inv_sqrt(
            _sym(c_b), nsteps=stub.higham_iters, eps=stub.delta, eps_relative=True
        )
        c_a_half = gram_ns_inv_sqrt(
            _sym(c_a), nsteps=stub.higham_iters, eps=stub.delta, eps_relative=True
        )
        eye = torch.eye(expected_rank).expand(len(indices), -1, -1).clone()
        v_a = torch.stack([states[i]["v_sigma_A"].float() for i in indices])
        v_b = torch.stack([states[i]["v_sigma_B"].float() for i in indices])
        sigma_a, _ = sigma_max_power_iter_batched(Aw, v_init=v_a, n_iters=8)
        sigma_b, _ = sigma_max_power_iter_batched(Bw, v_init=v_b, n_iters=8)
        cA, cB = stub._factor_scales(expected_rank, Aw.shape[-1], Bw.shape[-2])
        rho = (
            torch.full_like(sigma_b, lr)
            if bool(opt_cfg.get("cw_no_radius", False))
            else lr / (cA * sigma_b + cB * sigma_a).clamp_min(1e-12)
        )
        v_wa = torch.stack([states[i]["v_sigma_WA"].float() for i in indices])
        v_wb = torch.stack([states[i]["v_sigma_WB"].float() for i in indices])
        old_a = torch.stack([states[i]["m_A"].float() for i in indices])
        old_b = torch.stack([states[i]["m_B"].float() for i in indices])
        held_a = torch.stack([pairs[i]["heldout_batch_mean_gA"].float() for i in indices])
        held_b = torch.stack([pairs[i]["heldout_batch_mean_gB"].float() for i in indices])
        new_a = beta1 * old_a + (1.0 - beta1) * held_a
        new_b = beta1 * old_b + (1.0 - beta1) * held_b
        inputs = {
            "frozen_momentum": (old_a, old_b),
            "heldout_nesterov": (
                beta1 * new_a + (1.0 - beta1) * held_a,
                beta1 * new_b + (1.0 - beta1) * held_b,
            ),
        }
        slots = {
            "factorwise": (pa_half, qb_half),
            "product": (c_b_half, c_a_half),
            "identity": (eye, eye),
        }
        for input_mode, (m_a, m_b) in inputs.items():
            for label, (half_a, half_b) in slots.items():
                dA, dB = stub._cw_shadow_direction(
                    PAh=half_a,
                    QBh=half_b,
                    mhatA=m_a,
                    mhatB=m_b,
                    Q_isqrt=q_isqrt,
                    P_isqrt=p_isqrt,
                    cA=cA,
                    cB=cB,
                    rho=rho,
                    v_sigma_WA=v_wa,
                    v_sigma_WB=v_wb,
                )
                for local, index in enumerate(indices):
                    raw_pairs[index]["directions"][input_mode][label] = (
                        _direction_pair_metrics(
                            dA[local],
                            dB[local],
                            held_a[local],
                            held_b[local],
                            geometry[index],
                        )
                    )
    direction_seconds = time.perf_counter() - direction_started

    raw = {
        "schema_version": PULLBACK_SCHEMA_VERSION,
        "raw_json": str(raw_path),
        "command": shlex.join(sys.argv),
        "provenance": _git_provenance(),
        "artifact": str(artifact_path),
        "checkpoint": str(checkpoint.resolve()),
        "optimizer_state": str(optimizer_path),
        "rank": expected_rank,
        "pair_count": len(pairs),
        "state_validation_max_abs": max_state_error,
        "semantics": {
            "direction": (
                "same-checkpoint slot-only counterfactual using exact production "
                "whitening, PolarExpress, sigma-max pinning, and radius; not an "
                "observed optimizer step"
            ),
            "weak_modes": (
                "bottom half of B^T B for A-shaped tensors and bottom half of "
                "A A^T for B-shaped tensors"
            ),
            "product_slots": (
                "C_B=B^T P_eff B and C_A=A Q_eff A^T with "
                "P_eff/Q_eff=_rdinv(stored P/Q)^-2"
            ),
            "pq_weighted_energy": "||P_eff^(1/2) Y Q_eff^(1/2)||_F^2",
            "gross_weak": "E_weak/(E_weak+E_strong)",
            "net_weak": "(E_weak + half weak/strong cross)/E_total",
            "cancellation_ratio": (
                "combined mapped energy / sum of constituent mapped energies; "
                "values below one mean cancellation"
            ),
            "heldout_nesterov": (
                "heldout aggregate gradient inserted as a hypothetical current "
                "gradient; in-sample sensitivity, not causal loss evidence"
            ),
            "frozen_momentum": (
                "checkpoint EMA momentum is the common direction input; heldout "
                "gradient is used only for first-order evaluation"
            ),
            "lora_scale": float(cli["lora_alpha"]) / float(cli["lora_r"]),
        },
        "timing": {
            "load_seconds": load_seconds,
            "geometry_and_slot_seconds": geometry_seconds,
            "direction_and_mapping_seconds": direction_seconds,
            "seconds_before_raw_write": time.perf_counter() - started,
        },
        "pairs": raw_pairs,
    }
    # Raw evidence is durably written before deriving or writing the summary.
    _write_json(raw_path, raw)
    summary = _summarize_pullback(raw)
    summary["timing"]["total_seconds"] = time.perf_counter() - started
    _write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    print(f"saved pullback raw JSON: {raw_path}", flush=True)
    print(f"saved pullback summary: {summary_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-log",
        type=Path,
        default=Path(
            "scripts/results/"
            "cw_shadow_r16_b999_step7001_8batch_aggregate_grad_repeat.log"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--analyze-artifact",
        type=Path,
        default=None,
        help=(
            "run the CPU-only partner-mode/slot/pullback audit on an existing "
            "raw moment artifact instead of reconstructing gradients"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="assert this LoRA rank in either measurement or offline analysis mode",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=None,
        help="raw JSON destination for --analyze-artifact; summary is written beside it",
    )
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--expected-step", type=int, default=7000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "scripts/results/"
            "cw_shadow_r16_b999_step7000_heldout_batch_moments.pt"
        ),
    )
    return parser


def main() -> None:
    started = time.perf_counter()
    args = make_parser().parse_args()
    if args.analyze_artifact is not None:
        analyze_pullback(args)
        return
    if args.analysis_out is not None:
        raise ValueError("--analysis-out requires --analyze-artifact")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_event = _read_source_config(args.source_log)
    cfg = source_event["_cli_args"]
    if args.rank is not None and int(cfg["lora_r"]) != args.rank:
        raise ValueError(f"source rank {cfg['lora_r']} != requested rank {args.rank}")
    checkpoint = args.checkpoint or Path(cfg["resume_from"])
    n_batches = args.n_batches or int(cfg["optim_heldout_probe_batches"])
    if n_batches != int(cfg["optim_heldout_probe_batches"]):
        raise ValueError(
            f"n_batches={n_batches} would not reproduce the source probe's "
            f"{cfg['optim_heldout_probe_batches']} held-out batches"
        )
    if cfg["data_pipeline_version"] != "packed_v1.1":
        raise ValueError("this checkpoint probe expects packed_v1.1 eval semantics")
    if cfg["precond"] != "factorwise":
        raise ValueError("P_A/Q_B checkpoint comparison requires precond=factorwise")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the 1B checkpoint probe is GPU-only; use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not bool(cfg["no_tf32"]):
        # Match train.py's CUDA math policy before reconstructing gradients.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(int(cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_dataset = load_from_disk(os.path.join(cfg["data_dir"], "eval"))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=PadToMaxCollator(
            seq_length=int(cfg["max_seq_length"]),
            pad_token_id=tokenizer.pad_token_id,
        ),
        num_workers=0,
        pin_memory=True,
    )
    heldout_batches = []
    heldout_iter = iter(eval_loader)
    for _ in range(n_batches):
        try:
            heldout_batches.append(next(heldout_iter))
        except StopIteration as exc:
            raise ValueError(f"eval dataset has fewer than {n_batches} batches") from exc
    token_counts = [count_tokens(batch) for batch in heldout_batches]
    if any(count <= 0 for count in token_counts):
        raise ValueError(f"held-out batch has no supervised tokens: {token_counts}")
    weights = torch.tensor(token_counts, dtype=torch.float64)
    weights /= weights.sum()

    peft = build_peft_model(
        model_name=cfg["model_name"],
        training_mode=cfg["training_mode"],
        target_modules=parse_target_modules(cfg["target_modules"]),
        lora_r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        dtype=torch.bfloat16 if cfg["bf16"] else None,
        attn_implementation=cfg["attn_implementation"],
        use_liger=bool(cfg["use_liger"]),
        liger_flce=False,
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        compile_mode=None,
        device=device,
        world_size=1,
        local_rank=0,
    )
    bare_model, model = peft.bare_model, peft.train_model
    optimizer = _build_optimizer_from_cli(bare_model, cfg)
    resume = load_checkpoint(
        checkpoint, bare_model=bare_model, optimizer=optimizer, scheduler=None
    )
    if resume is None:
        raise FileNotFoundError(f"no checkpoint found at {checkpoint}")
    if int(resume["step"]) != args.expected_step:
        raise ValueError(
            f"loaded step {resume['step']}, expected step {args.expected_step}"
        )
    # Normal train.py resume reseeds to seed + checkpoint step before creating
    # the held-out iterator.  Eval is deterministic here, but preserve the exact
    # source protocol in case a future model has stochastic eval-time behavior.
    set_seed(int(cfg["seed"]) + int(resume["step"]))
    steps = {int(state["step"]) for state in optimizer.pair_state.values()}
    if steps != {args.expected_step}:
        raise ValueError(f"pair_state steps do not all equal {args.expected_step}: {steps}")

    named_pairs = collect_lora_pairs_named(bare_model)
    if len(named_pairs) != len(optimizer.pairs):
        raise ValueError("named-pair and optimizer-pair counts differ")
    for i, ((A, B), (named_A, named_B, _)) in enumerate(
        zip(optimizer.pairs, named_pairs)
    ):
        if A is not named_A or B is not named_B:
            raise ValueError(f"LoRA pair order mismatch at index {i}")
        if A.shape[0] != int(cfg["lora_r"]):
            raise ValueError(f"rank mismatch at pair {i}: {tuple(A.shape)}")

    grads_by_pair = [dict(A=[], B=[]) for _ in optimizer.pairs]
    batch_losses = []
    model.eval()
    for batch_index, batch in enumerate(heldout_batches):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch_to_device(batch, device))
        outputs.loss.backward()
        batch_losses.append(float(outputs.loss.detach()))
        for pair_index, (A, B) in enumerate(optimizer.pairs):
            if A.grad is None or B.grad is None:
                raise RuntimeError(
                    f"missing factor gradient for batch {batch_index}, pair {pair_index}"
                )
            if not bool(torch.isfinite(A.grad).all() and torch.isfinite(B.grad).all()):
                raise RuntimeError(
                    f"nonfinite factor gradient for batch {batch_index}, pair {pair_index}"
                )
            grads_by_pair[pair_index]["A"].append(A.grad.detach().float().cpu())
            grads_by_pair[pair_index]["B"].append(B.grad.detach().float().cpu())
    optimizer.zero_grad(set_to_none=True)

    pair_payloads = []
    max_identity_rel = 0.0
    with torch.no_grad():
        for i, ((A, B), (_, _, name), pair_grads) in enumerate(
            zip(optimizer.pairs, named_pairs, grads_by_pair)
        ):
            state = optimizer.pair_state[i]
            required = {"P_A", "Q_B", "Q", "P"}
            missing = required - set(state)
            if missing:
                raise KeyError(f"pair {i} checkpoint state missing {sorted(missing)}")
            Q_inv = optimizer._rdinv(state["Q"].unsqueeze(0)).square().squeeze(0)
            P_inv = optimizer._rdinv(state["P"].unsqueeze(0)).square().squeeze(0)
            gA = torch.stack(pair_grads["A"])
            gB = torch.stack(pair_grads["B"])
            moments_A = _side_moments(gA, weights, Q_inv.cpu(), side="A")
            moments_B = _side_moments(gB, weights, P_inv.cpu(), side="B")
            max_identity_rel = max(
                max_identity_rel,
                moments_A["identity_rel_residual"],
                moments_B["identity_rel_residual"],
            )
            weak_A = _weak_projector(A.detach().cpu(), side="A")
            weak_B = _weak_projector(B.detach().cpu(), side="B")
            stored_A = state["P_A"].detach().cpu().double()
            stored_B = state["Q_B"].detach().cpu().double()
            pair_payloads.append({
                "pair_index": i,
                "pair_name": name,
                "A_shape": tuple(A.shape),
                "B_shape": tuple(B.shape),
                "factor_A": A.detach().cpu(),
                "factor_B": B.detach().cpu(),
                "heldout_batch_gA": gA,
                "heldout_batch_gB": gB,
                "Q_stored": state["Q"].detach().cpu(),
                "P_stored": state["P"].detach().cpu(),
                "Q_inv_used": Q_inv.detach().cpu(),
                "P_inv_used": P_inv.detach().cpu(),
                "P_A_stored_raw": stored_A,
                "Q_B_stored_raw": stored_B,
                "P_A_heldout_batch_uncentered": moments_A["uncentered"],
                "P_A_heldout_batch_centered": moments_A["centered"],
                "P_A_heldout_batch_mean_outer": moments_A["mean_outer"],
                "Q_B_heldout_batch_uncentered": moments_B["uncentered"],
                "Q_B_heldout_batch_centered": moments_B["centered"],
                "Q_B_heldout_batch_mean_outer": moments_B["mean_outer"],
                "heldout_batch_mean_gA": moments_A["mean_grad"],
                "heldout_batch_mean_gB": moments_B["mean_grad"],
                "A_weak_mode_projector": weak_A,
                "B_weak_mode_projector": weak_B,
                "stats": {
                    "P_A": _pair_stats(stored_A, moments_A, weak_A),
                    "Q_B": _pair_stats(stored_B, moments_B, weak_B),
                },
            })
    if max_identity_rel > MOMENT_IDENTITY_RTOL:
        raise RuntimeError(
            "uncentered = centered + mean outer verification failed: "
            f"max relative residual {max_identity_rel:.3e} > {MOMENT_IDENTITY_RTOL:.3e}"
        )

    aggregate = _aggregate(pair_payloads)
    elapsed_pre_save = time.perf_counter() - started
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "measurement_semantics": {
            "observation_unit": "held-out batch (not per-example)",
            "batch_gradient": "gradient of batch-mean supervised-token NLL",
            "batch_weight": "supervised-token count / total supervised tokens",
            "P_A_target": "sum_i w_i gA_i Q^-1 gA_i^T / d_in",
            "Q_B_target": "sum_i w_i gB_i^T P^-1 gB_i / d_out",
            "diagonal_metric": "stored checkpoint Q/P held fixed",
            "weak_modes": "bottom half of current factor singular modes",
            "optimizer_steps_applied": 0,
        },
        "command": shlex.join(sys.argv),
        "args": _jsonable(vars(args)),
        "provenance": _git_provenance(),
        "source_log": str(args.source_log.resolve()),
        "source_config_event": source_event,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_resume_metadata": resume,
        "eval_batch_dataset_indices": [
            list(range(i * int(cfg["batch_size"]), (i + 1) * int(cfg["batch_size"])))
            for i in range(n_batches)
        ],
        "heldout_batch_token_counts": token_counts,
        "heldout_batch_weights": weights,
        "heldout_batch_losses": batch_losses,
        "pair_count": len(pair_payloads),
        "max_moment_identity_rel_residual": max_identity_rel,
        "aggregate": aggregate,
        "pairs": pair_payloads,
        "elapsed_seconds_before_save": elapsed_pre_save,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_artifact": str(args.out.resolve()),
        "measurement_semantics": artifact["measurement_semantics"],
        "checkpoint_step": resume["step"],
        "heldout_batch_token_counts": token_counts,
        "heldout_batch_weights": weights.tolist(),
        "heldout_batch_losses": batch_losses,
        "pair_count": len(pair_payloads),
        "max_moment_identity_rel_residual": max_identity_rel,
        "aggregate": aggregate,
        "elapsed_seconds_before_save": elapsed_pre_save,
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    summary_path = args.out.with_suffix(".summary.json")
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    print(f"saved raw tensors: {args.out}", flush=True)
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
