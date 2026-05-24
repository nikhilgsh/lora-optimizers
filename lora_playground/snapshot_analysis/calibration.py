"""Heavy diagnostic helpers: SSC / κ(c) sweeps and solver-convergence trajectories.

These are the four hottest functions per the profiling audit. Extracted so the
profile script and the analysis notebooks both import the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lora_playground.optim import _newton_schulz as newton_schulz_polar
from lora_playground.spectral import lambda_max_power_iter_psd_batched

from .snapshots import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    load_snapshot,
)
from .ssc import _prerescale_unit_op, _ssc_svd
from .whitening import DELTA_ABS, whitened_NS_input


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
                      steps: tuple[int, ...] = (200, 1000, 2000, 4000),
                      cs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5,
                                               0.7, 1.0, 2.0, 5.0, 10.0),
                      n_pairs: int = 12,
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
                op_X = torch.linalg.matrix_norm(X, ord=2).item()
                f_X = X.norm().item()
                if op_X == 0.0:
                    continue
                kappa_in = (f_X / op_X) ** 2 / r
                for c in cs:
                    Hc = _ssc_svd(X, c).float()
                    op_H = torch.linalg.matrix_norm(Hc, ord=2).item()
                    f_H = Hc.norm().item()
                    if op_H == 0.0:
                        continue
                    rows.append({
                        'lr': run_key[0], 'lora_r': run_key[1],
                        'ns': run_key[2], 'variant': run_key[3],
                        'r': r, 'step': step, 'pair': int(pi), 'c': c,
                        'kappa': (f_H / op_H) ** 2 / r,
                        'kappa_input': kappa_in,
                    })
    return pd.DataFrame(rows)


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
