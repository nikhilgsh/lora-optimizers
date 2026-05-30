"""Heavy diagnostic helpers: SSC / κ(c) sweeps and solver-convergence trajectories.

These are the four hottest functions per the profiling audit. Extracted so the
profile script and the analysis notebooks both import the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lora_playground.optim import (
    _newton_schulz as newton_schulz_polar,
    _solve_c_from_kappa_batched,
)
from lora_playground.spectral import lambda_max_power_iter_psd_batched

from .snapshots import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    load_snapshot,
)
from .ssc import _prerescale_unit_op, _ssc_svd
from .whitening import DELTA_ABS, spd_half_inv, whitened_NS_input


def _device(device: str | None = None) -> str:
    if device is not None:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------
# SSC (single-run) calibration — F-norm / op-norm / stable-rank vs c
# ----------------------------------------------------------------------------
def ssc_calibration(steps: tuple[int, ...] = (200, 1000, 2000, 4000),
                    cs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
                                             2.0, 5.0, 10.0),
                    n_pairs: int = 12,
                    side: str = 'A',
                    device: str | None = None) -> pd.DataFrame:
    """SSC clip ratios on the default run (lr=3e-2, r=64)."""
    dev = _device(device)
    rows = []
    for step in steps:
        try:
            snap = load_snapshot(step)
        except FileNotFoundError:
            continue
        n_total = len(snap['pair_state'])
        pair_indices = np.linspace(0, n_total - 1, n_pairs, dtype=int)
        for pi in pair_indices:
            p = snap['pair_state'][int(pi)]
            X = whitened_NS_input(p, side=side).to(dev)
            X_rs = _prerescale_unit_op(X)
            op_X = torch.linalg.matrix_norm(X_rs, ord=2).item()
            f_X = X_rs.norm().item()
            for c in cs:
                Hc = _ssc_svd(X_rs, c).float()
                op_H = torch.linalg.matrix_norm(Hc, ord=2).item()
                f_H = Hc.norm().item()
                rows.append({
                    'step': step, 'pair': int(pi), 'c': c,
                    'op_ratio': op_H / op_X,
                    'f_ratio': f_H / f_X,
                    'stable_rank_X': (f_X / op_X) ** 2,
                    'stable_rank_H': (f_H / op_H) ** 2,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# κ(c) — cross-run rank-normalized spectral energy of the SSC output
# ----------------------------------------------------------------------------
def kappa_calibration(runs: list[tuple] | None = None,
                      steps: tuple[int, ...] = (4000,),
                      cs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5,
                                               0.7, 1.0, 2.0, 5.0, 10.0),
                      n_pairs: int = 2,
                      side: str = 'A',
                      device: str | None = None,
                      delta_abs: float = DELTA_ABS,
                      verbose: bool = False) -> pd.DataFrame:
    """κ(c) = stable_rank(H_c(X)) / r on pre-rescaled whitened snapshots.

    r = min(X.shape) = lora_r for side='A' input X = S_B^{-1/2} u_A.
    """
    dev = _device(device)
    runs = runs if runs is not None else RUNS
    rows = []
    for run_key in runs:
        root = SNAP_ROOTS[run_key]
        available = set(STEPS_BY_ROOT.get(run_key, []))
        steps_here = [s for s in steps if s in available]
        for step in steps_here:
            try:
                snap = load_snapshot(step, root=root)
            except FileNotFoundError:
                continue
            if verbose:
                print(f'  {run_key} step={step}', flush=True)
            n_total = len(snap['pair_state'])
            pair_indices = np.linspace(0, n_total - 1, n_pairs, dtype=int)
            for pi in pair_indices:
                p = snap['pair_state'][int(pi)]
                A = p['A'].float()
                B = p['B'].float()
                u = p[f'u_{side}'].float()
                S = (B.T @ B) if side == 'A' else (A @ A.T)
                evals, evecs = torch.linalg.eigh(
                    S + delta_abs * torch.eye(S.shape[0], dtype=S.dtype))
                S_half_inv = evecs @ torch.diag(evals.clamp_min(1e-30).rsqrt()) @ evecs.T
                X_raw = (S_half_inv @ u) if side == 'A' else (u @ S_half_inv)
                X = _prerescale_unit_op(X_raw.to(dev))
                r = min(X.shape)
                if X.shape[-2] <= X.shape[-1]:
                    gram_X = X @ X.transpose(-2, -1)
                else:
                    gram_X = X.transpose(-2, -1) @ X
                lam = torch.linalg.eigvalsh(gram_X).clamp_min(0.0)
                lam_max = lam.max().clamp_min(1e-30)
                if float(lam_max) == 0.0:
                    continue
                # X is pre-rescaled to σmax≈1. For SSC
                # h_c(s)=s/sqrt(1+(s/c)^2), so
                # stable_rank(H_c(X))/r = mean((h_c(s_i)/h_c(1))^2).
                # This avoids applying SSC/SVD for every c.
                s_sq = lam / lam_max
                kappa_in = float(s_sq.mean())
                cs_t = torch.as_tensor(cs, device=s_sq.device, dtype=s_sq.dtype)
                inv_c2 = 1.0 / cs_t.square()
                kappa_vals = (
                    s_sq.unsqueeze(0)
                    * (1.0 + inv_c2).unsqueeze(-1)
                    / (1.0 + s_sq.unsqueeze(0) * inv_c2.unsqueeze(-1))
                ).mean(dim=-1)
                for c, kappa_at_c in zip(cs, kappa_vals.detach().cpu().tolist()):
                    rows.append({
                        'lr': run_key[0], 'lora_r': run_key[1],
                        'ns': run_key[2], 'variant': run_key[3],
                        'r': r, 'step': step, 'pair': int(pi), 'c': c,
                        'kappa': float(kappa_at_c),
                        'kappa_input': kappa_in,
                    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Agreement-derived adaptive κ diagnostic
# ----------------------------------------------------------------------------
def _normalized_s_sq_small_side(X: torch.Tensor) -> torch.Tensor:
    """Squared singular values normalized by σ_max², using the small Gram."""
    Xf = X.float()
    if Xf.shape[-2] <= Xf.shape[-1]:
        gram = Xf @ Xf.transpose(-2, -1)
    else:
        gram = Xf.transpose(-2, -1) @ Xf
    lam = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    return lam / lam.max().clamp_min(1e-30)


def _stable_rank_from_sq_energy(e_sq: torch.Tensor) -> torch.Tensor:
    """Stable rank of a spectrum represented by nonnegative squared energies."""
    e = e_sq.clamp_min(0.0)
    return e.sum() / e.max().clamp_min(1e-30)


def _solve_c_for_target_kappa(s_sq: torch.Tensor, kappa: float) -> float:
    """Map a side-specific target κ to the SSC c that realizes it."""
    target = float(max(float(s_sq.mean()), min(1.0, float(kappa))))
    c = _solve_c_from_kappa_batched(s_sq.unsqueeze(0), target)
    return float(c.squeeze(0))


def _agreement_kappa_for_pair(
    pair: dict,
    *,
    lr: str | float,
    lora_r: int,
    ns: int,
    variant: str,
    step: int,
    pair_index: int,
    delta_abs: float,
    device: str | None,
) -> dict:
    dev = torch.device(device) if device is not None else None
    A = pair["A"].float().to(dev)
    B = pair["B"].float().to(dev)
    u_A = pair["u_A"].float().to(dev)
    u_B = pair["u_B"].float().to(dev)

    W_A = spd_half_inv(A @ A.T, delta_abs=delta_abs)
    W_B = spd_half_inv(B.T @ B, delta_abs=delta_abs)

    # Two r x r compatibility views. Raw LoRA gradients from one dense
    # gradient G obey g_A A^T = B^T g_B; here u_A/u_B are Adam directions,
    # so C_A - C_B only measures factor-view incompatibility. The historical
    # "signal"/"noise" field names below are diagnostic labels, not a
    # validated denoising model.
    C_A = W_B @ (u_A @ A.T) @ W_A
    C_B = W_B @ (B.T @ u_B) @ W_A
    C_plus = 0.5 * (C_A + C_B)
    C_minus = 0.5 * (C_A - C_B)

    s_plus_sq = torch.linalg.svdvals(C_plus).square()
    s_minus_sq = torch.linalg.svdvals(C_minus).square()
    signal_f2 = s_plus_sq.sum()
    noise_f2 = s_minus_sq.sum()
    q_agree = signal_f2 / (signal_f2 + noise_f2).clamp_min(1e-30)

    noise_op_sq = s_minus_sq.max().clamp_min(0.0)
    noise_mean_sq = s_minus_sq.mean().clamp_min(0.0)
    rel_op_sq = (s_plus_sq - noise_op_sq).clamp_min(0.0)
    rel_mean_sq = (s_plus_sq - noise_mean_sq).clamp_min(0.0)
    r = int(min(A.shape[0], B.shape[1]))
    k_rel_op = _stable_rank_from_sq_energy(rel_op_sq)
    k_rel_mean = _stable_rank_from_sq_energy(rel_mean_sq)
    kappa_agree_op = float(k_rel_op / max(r, 1))
    kappa_agree_mean = float(k_rel_mean / max(r, 1))

    X_A = _prerescale_unit_op(W_B @ u_A)
    X_B = _prerescale_unit_op(u_B @ W_A)
    s_sq_A = _normalized_s_sq_small_side(X_A)
    s_sq_B = _normalized_s_sq_small_side(X_B)
    kappa_input_A = float(s_sq_A.mean())
    kappa_input_B = float(s_sq_B.mean())

    target_op_A = max(kappa_input_A, min(1.0, kappa_agree_op))
    target_op_B = max(kappa_input_B, min(1.0, kappa_agree_op))
    target_mean_A = max(kappa_input_A, min(1.0, kappa_agree_mean))
    target_mean_B = max(kappa_input_B, min(1.0, kappa_agree_mean))

    return {
        "lr": lr,
        "lora_r": int(lora_r),
        "ns": int(ns),
        "variant": variant,
        "step": int(step),
        "pair": int(pair_index),
        "shape": repr((A.shape[0], A.shape[1], B.shape[0])),
        "q_agree": float(q_agree),
        "snr_op": float(
            s_plus_sq.max().sqrt() / s_minus_sq.max().sqrt().clamp_min(1e-30)
        ),
        "signal_stable_rank": float(_stable_rank_from_sq_energy(s_plus_sq)),
        "noise_stable_rank": float(_stable_rank_from_sq_energy(s_minus_sq)),
        "reliable_rank_opfloor": float(k_rel_op),
        "reliable_rank_meanfloor": float(k_rel_mean),
        "n_reliable_opfloor": int((s_plus_sq > noise_op_sq).sum()),
        "n_reliable_meanfloor": int((s_plus_sq > noise_mean_sq).sum()),
        "kappa_agree_opfloor": kappa_agree_op,
        "kappa_agree_meanfloor": kappa_agree_mean,
        "kappa_input_A": kappa_input_A,
        "kappa_input_B": kappa_input_B,
        "kappa_target_opfloor_A": target_op_A,
        "kappa_target_opfloor_B": target_op_B,
        "kappa_target_meanfloor_A": target_mean_A,
        "kappa_target_meanfloor_B": target_mean_B,
        "c_opfloor_A": _solve_c_for_target_kappa(s_sq_A, target_op_A),
        "c_opfloor_B": _solve_c_for_target_kappa(s_sq_B, target_op_B),
        "c_meanfloor_A": _solve_c_for_target_kappa(s_sq_A, target_mean_A),
        "c_meanfloor_B": _solve_c_for_target_kappa(s_sq_B, target_mean_B),
    }


def agreement_kappa_diagnostic(
    runs: list[tuple] | None = None,
    steps: tuple[int, ...] = (4000,),
    n_pairs: int = 4,
    delta_abs: float = DELTA_ABS,
    device: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute an exploratory κ target from factor-view compatibility.

    The symmetric and antisymmetric compatibility summaries are

        C_A = S_B^{-1/2} (u_A A^T) S_A^{-1/2}
        C_B = S_B^{-1/2} (B^T u_B) S_A^{-1/2}.
        C+ = (C_A + C_B)/2,    C- = (C_A - C_B)/2.

    We report two heuristic floors for σ_i(C+)²:
      * opfloor: subtract ||C-||_op² from every mode.
      * meanfloor: subtract mean_i σ_i(C-)².

    The helper only measures what target κ and implied c would be; it does not
    change optimizer behavior or validate C- as harmful update energy.
    """
    dev = _device(device)
    runs = runs if runs is not None else RUNS
    rows = []
    for run_key in runs:
        root = SNAP_ROOTS[run_key]
        available = set(STEPS_BY_ROOT.get(run_key, []))
        steps_here = [s for s in steps if s in available]
        for step in steps_here:
            try:
                snap = load_snapshot(step, root=root)
            except FileNotFoundError:
                continue
            if verbose:
                print(f"agreement-kappa: {run_key} step={step}", flush=True)
            n_total = len(snap["pair_state"])
            pair_indices = np.linspace(0, n_total - 1, n_pairs, dtype=int)
            for pi in pair_indices:
                rows.append(_agreement_kappa_for_pair(
                    snap["pair_state"][int(pi)],
                    lr=run_key[0],
                    lora_r=run_key[1],
                    ns=run_key[2],
                    variant=run_key[3],
                    step=step,
                    pair_index=int(pi),
                    delta_abs=delta_abs,
                    device=dev,
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Local objective sweep over c (snapshot-only, no dense d_out x d_in materialize)
# ----------------------------------------------------------------------------
def _opnorm_small_side(X: torch.Tensor) -> torch.Tensor:
    """Operator norm via the smaller-side Gram matrix."""
    Xf = X.float()
    if Xf.shape[-2] <= Xf.shape[-1]:
        gram = Xf @ Xf.transpose(-2, -1)
    else:
        gram = Xf.transpose(-2, -1) @ Xf
    return torch.linalg.eigvalsh(gram).clamp_min(0.0).max().sqrt()


def _linear_residual_norm_sq(
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
) -> torch.Tensor:
    """||B dA + dB A||_F^2 using only r x r Gram products."""
    SB = B.T @ B
    SA = A @ A.T
    dA_dAT = dA @ dA.T
    dBT_dB = dB.T @ dB
    B_T_dB = B.T @ dB
    A_dAT = A @ dA.T
    norm_BdA = (SB * dA_dAT).sum()
    norm_dBA = (dBT_dB * SA).sum()
    cross = (B_T_dB * A_dAT.T).sum()
    return norm_BdA + norm_dBA + 2.0 * cross


def _chord_residual_norm_sq(
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
) -> torch.Tensor:
    """||B dA + dB A + dB dA||_F^2 using only r x r Gram products."""
    linear = _linear_residual_norm_sq(A, B, dA, dB)
    dA_dAT = dA @ dA.T
    dBT_dB = dB.T @ dB
    B_T_dB = B.T @ dB
    A_dAT = A @ dA.T
    norm_dBdA = (dBT_dB * dA_dAT).sum()
    cross_BdA_dBdA = (B_T_dB * dA_dAT.T).sum()
    cross_dBA_dBdA = (A_dAT * dBT_dB).sum()
    return linear + norm_dBdA + 2.0 * (cross_BdA_dBdA + cross_dBA_dBdA)


def _simulate_chord_tight_ssc_update(
    pair: dict,
    *,
    lr: float,
    c: float,
    picard_iters: int,
    delta_abs: float,
    device: str | None = None,
) -> dict:
    """Replay the chord-tight-clean SSC update on one snapshot pair.

    This mirrors the production pipeline's whitening, pre-rescale, Picard
    cross-coupling, SSC map, unwhitening, and rho/opnorm rescale. It returns
    both raw and post-pre-rescale Adam covectors so objective sweeps can test
    which surrogate is being optimized by the code path.
    """
    dev = torch.device(device) if device is not None else None
    A = pair["A"].float().to(dev)
    B = pair["B"].float().to(dev)
    u_A_raw = pair["u_A"].float().to(dev)
    u_B_raw = pair["u_B"].float().to(dev)

    S_A = A @ A.T
    S_B = B.T @ B
    W_A = spd_half_inv(S_A, delta_abs=delta_abs)
    W_B = spd_half_inv(S_B, delta_abs=delta_abs)

    sigma_A = _opnorm_small_side(A)
    sigma_B = _opnorm_small_side(B)
    rho = float(lr) / (sigma_A + sigma_B).clamp_min(1e-30)

    X_A = W_B @ u_A_raw
    X_B = u_B_raw @ W_A
    sigma_XA = _opnorm_small_side(X_A).clamp_min(1e-30)
    sigma_XB = _opnorm_small_side(X_B).clamp_min(1e-30)
    u_A = u_A_raw / sigma_XA
    u_B = u_B_raw / sigma_XB
    X_A = X_A / sigma_XA
    X_B = X_B / sigma_XB

    dA = torch.zeros_like(u_A)
    dB = torch.zeros_like(u_B)
    X_A_eff = X_A
    X_B_eff = X_B
    for n in range(int(picard_iters)):
        if n == 0:
            X_A_eff = X_A
            X_B_eff = X_B
        else:
            u_A_eff = u_A + (1.0 / float(lr)) * (B.T @ dB @ A)
            u_B_eff = u_B + (1.0 / float(lr)) * (B @ dA @ A.T)
            X_A_eff = W_B @ u_A_eff
            X_B_eff = u_B_eff @ W_A

        P_A = _ssc_svd(X_A_eff, c).float()
        P_B = _ssc_svd(X_B_eff, c).float()
        geo_A = W_B @ P_A
        geo_B = P_B @ W_A
        dA = -(rho / _opnorm_small_side(geo_A).clamp_min(1e-30)) * geo_A
        dB = -(rho / _opnorm_small_side(geo_B).clamp_min(1e-30)) * geo_B

    linear_scaled = (u_A * dA).sum() + (u_B * dB).sum()
    linear_raw = (u_A_raw * dA).sum() + (u_B_raw * dB).sum()
    quad_tangent = _linear_residual_norm_sq(A, B, dA, dB) / (2.0 * float(lr))
    quad_chord = _chord_residual_norm_sq(A, B, dA, dB) / (2.0 * float(lr))
    op_XA = _opnorm_small_side(X_A_eff).clamp_min(1e-30)
    op_XB = _opnorm_small_side(X_B_eff).clamp_min(1e-30)
    return {
        "linear_scaled": float(linear_scaled),
        "linear_raw": float(linear_raw),
        "quad_tangent": float(quad_tangent),
        "quad_chord": float(quad_chord),
        "obj_scaled_tangent": float(linear_scaled + quad_tangent),
        "obj_scaled_chord": float(linear_scaled + quad_chord),
        "obj_raw_tangent": float(linear_raw + quad_tangent),
        "obj_raw_chord": float(linear_raw + quad_chord),
        "sr_XA_eff": float(X_A_eff.square().sum() / op_XA.pow(2)),
        "sr_XB_eff": float(X_B_eff.square().sum() / op_XB.pow(2)),
        "rho": float(rho),
        "sigma_A": float(sigma_A),
        "sigma_B": float(sigma_B),
    }


def local_objective_c_grid(
    runs: list[tuple] | None = None,
    steps: tuple[int, ...] = (2000, 4000),
    cs: tuple[float, ...] = (
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0,
        1.5, 2.0, 3.0, 5.0, 10.0,
    ),
    n_pairs: int = 6,
    picard_iters: int = 3,
    delta_abs: float = DELTA_ABS,
    device: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Sweep SSC c on saved snapshots and score local surrogate objectives.

    Columns include four objective variants:
      * obj_scaled_tangent: production pre-rescaled Adam covectors, tangent J.
      * obj_scaled_chord:   production pre-rescaled Adam covectors, chord J.
      * obj_raw_tangent:    raw Adam covectors, tangent J.
      * obj_raw_chord:      raw Adam covectors, chord J.

    Lower is better because this is the minimization form of the local
    quadratic surrogate. The computation stays on r x r Grams for residual
    norms, avoiding dense d_out x d_in products.
    """
    dev = _device(device)
    runs = runs if runs is not None else RUNS
    rows = []
    for run_key in runs:
        root = SNAP_ROOTS[run_key]
        lr = float(run_key[0])
        available = set(STEPS_BY_ROOT.get(run_key, []))
        steps_here = [s for s in steps if s in available]
        for step in steps_here:
            snap = load_snapshot(step, root=root)
            n_total = len(snap["pair_state"])
            pair_indices = np.linspace(0, n_total - 1, n_pairs, dtype=int)
            if verbose:
                print(f"local c-grid: {run_key} step={step} device={dev}", flush=True)
            for pi in pair_indices:
                pair = snap["pair_state"][int(pi)]
                A = pair["A"]
                B = pair["B"]
                shape = (A.shape[0], A.shape[1], B.shape[0])
                for c in cs:
                    metrics = _simulate_chord_tight_ssc_update(
                        pair,
                        lr=lr,
                        c=float(c),
                        picard_iters=picard_iters,
                        delta_abs=delta_abs,
                        device=dev,
                    )
                    rows.append({
                        "run": repr(run_key),
                        "lr": run_key[0],
                        "lora_r": run_key[1],
                        "ns": run_key[2],
                        "variant": run_key[3],
                        "step": int(step),
                        "pair": int(pi),
                        "shape": repr(shape),
                        "c": float(c),
                        "picard_iters": int(picard_iters),
                        **metrics,
                    })
    return pd.DataFrame(rows)


def best_c_by_snapshot_objective(
    df: pd.DataFrame,
    objective_col: str = "obj_scaled_tangent",
) -> pd.DataFrame:
    """Return one best-c row per (run, step, pair) for an objective column."""
    if df.empty:
        return df.copy()
    idx = (
        df.sort_values(["run", "step", "pair", objective_col, "c"])
        .groupby(["run", "step", "pair"], sort=False)
        .head(1)
        .index
    )
    out = df.loc[idx].copy()
    out = out.rename(columns={"c": "best_c", objective_col: "best_obj"})
    return out


# ----------------------------------------------------------------------------
# BJ inner-loop convergence (chord-tight)
# ----------------------------------------------------------------------------
def simulate_bcd(p: dict, eta: float = 3e-2, delta_abs: float = 1e-6,
                 K: int = 10, ns_iters: int = 10,
                 device: str | None = None):
    """One BJ inner-loop simulation under chord-tight magnitude (rho = eta/s).

    All op-norms via Gram-matrix tricks on r×r or 2r×2r matrices — no svdvals
    on (r, d_in) or (d_out, d_in).
    """
    dev = _device(device)
    A = p['A'].float().to(dev)
    B = p['B'].float().to(dev)
    u_A = p['u_A'].float().to(dev)
    u_B = p['u_B'].float().to(dev)
    r = A.shape[0]
    I_r = torch.eye(r, dtype=A.dtype, device=A.device)

    S_B = B.T @ B
    eB_raw = torch.linalg.eigvalsh(S_B).clamp_min(0)
    e2B, UB = torch.linalg.eigh(S_B + delta_abs * I_r)
    W_B = UB @ torch.diag(e2B.clamp_min(1e-30).rsqrt()) @ UB.T
    sB_max = eB_raw.max().sqrt().item()

    S_A = A @ A.T
    eA_raw = torch.linalg.eigvalsh(S_A).clamp_min(0)
    e2A, UA = torch.linalg.eigh(S_A + delta_abs * I_r)
    W_A = UA @ torch.diag(e2A.clamp_min(1e-30).rsqrt()) @ UA.T
    sA_max = eA_raw.max().sqrt().item()

    rho = eta / (sA_max + sB_max)

    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    traj_dA, traj_dB, traj_J, traj_phi = [], [], [], []
    for _ in range(K):
        u_A_tilde = u_A + (1.0 / eta) * (B.T @ dB @ A)
        u_B_tilde = u_B + (1.0 / eta) * (B @ dA @ A.T)
        D_A = W_B @ newton_schulz_polar(W_B @ u_A_tilde, ns_iters)
        D_B = newton_schulz_polar(u_B_tilde @ W_A, ns_iters) @ W_A
        D_A_op = torch.linalg.eigvalsh(D_A @ D_A.T).clamp_min(0).max().sqrt().item()
        D_B_op = torch.linalg.eigvalsh(D_B.T @ D_B).clamp_min(0).max().sqrt().item()
        dA = -rho * D_A / D_A_op
        dB = -rho * D_B / D_B_op
        # ||J||_2 via rank-2r Gram trick
        U_2 = torch.cat([B, dB], dim=1)
        V_2 = torch.cat([dA, A], dim=0)
        J_op = (torch.linalg.eigvals((U_2.T @ U_2) @ (V_2 @ V_2.T))
                .real.clamp_min(0).max().sqrt().item())
        J = B @ dA + dB @ A
        phi = ((u_A * dA).sum().item()
               + (u_B * dB).sum().item()
               + (1.0 / (2 * eta)) * J.pow(2).sum().item())
        traj_dA.append(dA.detach().cpu())
        traj_dB.append(dB.detach().cpu())
        traj_J.append(J_op)
        traj_phi.append(phi)
    return traj_dA, traj_dB, traj_J, traj_phi


def collect_bcd_convergence(step: int, K: int = 10,
                            n_pairs: int | None = None,
                            eta: float = 3e-2,
                            ns_iters: int = 10,
                            device: str | None = None) -> list[dict]:
    snap = load_snapshot(step)
    items = list(snap['pair_state'].items())
    if n_pairs is not None:
        items = items[:n_pairs]
    results = []
    for pi, p in items:
        traj_dA, traj_dB, traj_J, traj_phi = simulate_bcd(
            p, eta=eta, K=K, ns_iters=ns_iters, device=device)
        final_dA, final_dB = traj_dA[-1], traj_dB[-1]
        rel_dA = [(traj_dA[n] - final_dA).norm().item()
                  / (final_dA.norm().item() + 1e-30) for n in range(K)]
        rel_dB = [(traj_dB[n] - final_dB).norm().item()
                  / (final_dB.norm().item() + 1e-30) for n in range(K)]
        rel_consec_A = [(traj_dA[n] - traj_dA[n - 1]).norm().item()
                        / (traj_dA[n].norm().item() + 1e-30) for n in range(1, K)]
        rel_consec_B = [(traj_dB[n] - traj_dB[n - 1]).norm().item()
                        / (traj_dB[n].norm().item() + 1e-30) for n in range(1, K)]
        results.append(dict(pi=pi, rel_dA=rel_dA, rel_dB=rel_dB,
                            rel_consec_A=rel_consec_A, rel_consec_B=rel_consec_B,
                            traj_J=traj_J, traj_phi=traj_phi))
    return results


# ----------------------------------------------------------------------------
# Higham (matrix inverse-sqrt) coupled-NS convergence
# ----------------------------------------------------------------------------
def higham_residual_traj(H: torch.Tensor, K_max: int = 20,
                         delta_abs: float = 1e-6,
                         device: str | None = None):
    """Production-style Higham: λ_max via 8-iter power iter (NOT eigh).

    Returns (residuals, diffs):
      residuals[k] = ‖Z_k H Z_k − I‖_F
      diffs[k]     = ‖Z_k − H^{−1/2}‖_F / ‖H^{−1/2}‖_F (vs eigh ground truth)
    """
    dev = _device(device)
    H = H.to(dev).float()
    H = 0.5 * (H + H.T)
    n = H.shape[-1]
    eps_eff = delta_abs
    H_damped = H + eps_eff * torch.eye(n, device=dev)
    # eigh ground truth for the diff plot only (NOT used in iteration).
    evals_d, U_d = torch.linalg.eigh(H_damped)
    Z_true = U_d @ torch.diag(evals_d.clamp_min(1e-30).rsqrt()) @ U_d.T
    lam_max, _ = lambda_max_power_iter_psd_batched(H_damped.unsqueeze(0), n_iters=8)
    s = lam_max.item()
    eye = torch.eye(n, dtype=H.dtype, device=dev)
    Y = H_damped / s
    Z = eye.clone()
    three_eye = 3.0 * eye
    residuals, diffs = [], []
    for _ in range(1, K_max + 1):
        T = three_eye - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
        Z_phys = Z / (s ** 0.5)
        R = Z_phys @ H_damped @ Z_phys - eye
        residuals.append(R.norm().item())
        diffs.append((Z_phys - Z_true).norm().item() / Z_true.norm().item())
    return residuals, diffs
