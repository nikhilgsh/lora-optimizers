import json
import statistics

import torch
from torch.optim import AdamW, Optimizer, SGD

from .utils import (
    collect_lora_pairs,
    f_lorsum,
    lorsum,
    solve_spd,
    solve_sylvester,
    spd_frac_power_inv,
    spd_inv_sqrt_higham,
    spdify,
    truncated_svd,
)


def _spd_inv_half(H, eps, method="eigh", higham_iters=10):
    """Dispatch (H + eps·I)^{-1/2}: 'eigh' uses spd_frac_power_inv; 'higham' uses
    Newton-Schulz (no eigh, ~10× faster on (r×r) at r=256 due to no kernel-launch
    storm). Caller can swap between them via precond_method=... at construction."""
    if method == "eigh":
        return spd_frac_power_inv(H, gamma=0.5, eps=eps)
    if method == "higham":
        return spd_inv_sqrt_higham(H, n_iters=higham_iters, eps=eps)
    raise ValueError(f"Unknown precond_method '{method}' (expected 'eigh' or 'higham').")


def _adamw_side_step(grad, m_raw, v_raw, beta1, beta2, eps, step, lr):
    """Side-channel AdamW step on raw `grad`. Mutates `m_raw`, `v_raw` in place
    and returns -lr * m̂/(√v̂+ε) — i.e. what plain AdamW would apply at this
    step given the same lr. Used by H1 diagnostics to compare against the
    geometrically-preconditioned step that the host optimizer applies.
    """
    g32 = grad.detach().to(dtype=torch.float32)
    m_raw.mul_(beta1).add_(g32, alpha=1.0 - beta1)
    v_raw.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = m_raw / bc1
    v_hat = v_raw / bc2
    return -lr * m_hat / (v_hat.sqrt() + eps)


def _frob_cos(a, b):
    af = a.detach().to(torch.float32).flatten()
    bf = b.detach().to(torch.float32).flatten()
    na = float(af.norm())
    nb = float(bf.norm())
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float((af @ bf) / (na * nb))


def _spd_eig_extremes(M):
    # eigvalsh can fail (LinAlgError, code 257) on near-degenerate Gram matrices;
    # this is a side-channel diagnostic, so degrade to NaN instead of killing the
    # training run.
    try:
        ev = torch.linalg.eigvalsh(M.to(torch.float32))
    except torch._C._LinAlgError:
        return float("nan"), float("nan")
    return float(ev[0]), float(ev[-1])


def _gram_eig_extremes_from_factor(X):
    # λ(XᵀX) = σ(X)². Routing through svdvals(X) avoids forming the Gram
    # (which squares condition number) and is robust where eigvalsh(XᵀX)
    # fails on near-degenerate spectra.
    try:
        sv = torch.linalg.svdvals(X.to(torch.float32))
    except torch._C._LinAlgError:
        return float("nan"), float("nan")
    return float(sv[-1] ** 2), float(sv[0] ** 2)


def _emit_optim_diagnostics(step_count, per_pair_records):
    """Aggregate per-pair diagnostic records and emit one JSONL `optim_step` event.

    Each record is a flat dict of float-valued stats; we report median/min/max
    across pairs to keep log size bounded.
    """
    if not per_pair_records:
        return
    keys = list(per_pair_records[0].keys())
    payload = {"event": "optim_step", "step": int(step_count), "n_pairs": len(per_pair_records)}
    for k in keys:
        vals = [r[k] for r in per_pair_records if r[k] == r[k]]  # drop NaN
        if not vals:
            payload[k + "_median"] = float("nan")
            continue
        payload[k + "_median"] = statistics.median(vals)
        payload[k + "_min"] = min(vals)
        payload[k + "_max"] = max(vals)
    print(json.dumps(payload, sort_keys=True), flush=True)

OPTIMIZER_CHOICES = {
    "adamw",
    "lin-lora",
    "scaled-lora",
    "adam-scaled-lora",
    "adam-lin-lora",
    "adam-scaled-lora-post",
    "adam-lin-lora-post",
    "adam-scaled-lora-matrix",
    "adam-lin-lora-matrix",
    "polar-product-lora",
    "adam-polar-product-lora",
    "adam-polar-product-lora-coupled",
    "adamuon-polar-product-lora",
    "adamuon-lora",
    "muon-lora",
    "product-muon-lora",
    "adam-muon-lora",
    "adam-product-muon-lora",
    "muon-adam-lora",
    "diag-scaled-lora",
    "kron-grad-lora",
    "psi-lora",
    "galore-adamw",
    "sgd",
    "sgd-m",
    "svd-step-adamw",
    "svd-cumulative-adamw",
}

class ScaledLoRA(Optimizer):
    """
    ScaledLoRA: Preconditioned gradient descent for LoRA (A, B) tensors.

    Applies gradient descent with preconditioning matrices derived from the current
    LoRA factors. The update is:
        ΔA = -lr * S_B^{-1} ∇_A
        ΔB = -lr * ∇_B S_A^{-1}
    where S_A = A A^T + δ I and S_B = B^T B + δ I.
    """
    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta

    @torch.no_grad()
    def step(self, closure=None):
        """
        ScaledLoRA update step.

        For each pair:
            A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
        Gradients:
            ∇_A ∈ ℝ^{r×d_in}, ∇_B ∈ ℝ^{d_out×r}

        Computes preconditioned updates using S_A and S_B.
        Everything is cast to float32 for numerical stability.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]

        for A, B in self.pairs:  # A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for ScaledLoRA update.")

            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Compute preconditioning matrices: S_A = A A^T + δ I, S_B = B^T B + δ I
            SB = spdify(B.T @ B, self.delta)       # S_B ∈ ℝ^{r×r}
            SA = spdify(A @ A.T, self.delta)       # S_A ∈ ℝ^{r×r}

            # Preconditioned gradient updates: ΔB = -lr * ∇_B S_A^{-1}
            dB = -lr * solve_spd(SA, gB.T).T       # ΔB ∈ ℝ^{d_out×r}

            # ΔA = -lr * S_B^{-1} ∇_A
            dA = -lr * solve_spd(SB, gA)           # ΔA ∈ ℝ^{r×d_in}

            # Apply the update, cast back to parameter dtype/device
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            A.grad.zero_()
            B.grad.zero_()

class LinLoRA(Optimizer):
    """
    LinLoRA: Linearized least-squares update for LoRA (A, B) tensors.

    Linearizes the LoRA product BA around the current factors to compute a
    coupled update for both matrices. Solves for a correction matrix K ∈ ℝ^{r×r}
    via the Sylvester equation:
        S_B K + K S_A = -lr * (∇_A A^T),
    where S_A = A A^T + δ I and S_B = B^T B + δ I. Then applies:
        ΔA = -S_B^{-1} (lr * ∇_A + K A)
        ΔB = -(lr * ∇_B + B K) S_A^{-1}
    """
    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta

    @torch.no_grad()
    def step(self, closure=None):
        """
        LinLoRA update step via Sylvester equation.

        For each pair:
            A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
        Gradients:
            ∇_A ∈ ℝ^{r×d_in}, ∇_B ∈ ℝ^{d_out×r}

        Solves: S_B K + K S_A = -lr * (∇_A A^T) for K ∈ ℝ^{r×r},
        then updates A and B using the linearized corrections.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]

        for A, B in self.pairs:  # A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for LinLoRA update.")

            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Regularized Gram matrices: S_A = A A^T + δ I, S_B = B^T B + δ I
            SB = spdify(B.T @ B, self.delta)       # S_B ∈ ℝ^{r×r}
            SA = spdify(A @ A.T, self.delta)       # S_A ∈ ℝ^{r×r}
            RHS = -lr * (gA @ A.T).float()         # RHS = -lr * (∇_A A^T) ∈ ℝ^{r×r}

            # Solve Sylvester equation: S_B K + K S_A = RHS for K ∈ ℝ^{r×r}
            K = solve_sylvester(SB, SA, RHS)       # K ∈ ℝ^{r×r}

            # Update B: ΔB = -(lr * ∇_B + B K) S_A^{-1}
            termB = (lr * gB + B @ K.to(dtype=B.dtype)).float()   # ℝ^{d_out×r}
            dB = -solve_spd(SA, termB.T).T                        # ΔB ∈ ℝ^{d_out×r}

            # Update A: ΔA = -S_B^{-1} (lr * ∇_A + K A)
            termA = (lr * gA + K.to(dtype=A.dtype) @ A).float()   # ℝ^{r×d_in}
            dA = -solve_spd(SB, termA)                            # ΔA ∈ ℝ^{r×d_in}

            # Apply the update, cast back to parameter dtype/device
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            A.grad.zero_()
            B.grad.zero_()


class AdamLinLoRA(Optimizer):
    """
    AdamLinLoRA: Adam-preconditioned version of LinLoRA.

    Applies Adam optimization to the linearized preconditioned gradients computed by LinLoRA.
    For each LoRA pair (A, B):
        1. Solves Sylvester equation: S_B K + K S_A = -(∇_A A^T) for K ∈ ℝ^{r×r}
        2. Computes preconditioned gradients:
            v_A = S_B^{-1} (∇_A + K A)
            v_B = (∇_B + B K) S_A^{-1}
        3. Applies Adam updates using exponential moving averages (m, v):
            m_t = β₁ m_{t-1} + (1-β₁) v_t
            v_t = β₂ v_{t-1} + (1-β₂) v_t²
            Δθ = -lr * m̂_t / (√v̂_t + ε)
    where S_A = A A^T + δ I and S_B = B^T B + δ I.
    """
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None, scaled_metric=False, lora_plus_multiplier=1.0, log_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every

        # Initialize state: first and second moments for each (A, B) pair
        # Use pair_state to avoid conflicts with PyTorch's Optimizer.state
        self.pair_state = {}
        self.gammas = []
        for i, (A, B) in enumerate(pairs):
            entry = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }
            if log_diagnostics:
                # Side-channel raw-grad Adam state for cosine comparison vs the
                # geometrically-preconditioned step actually applied. Only
                # allocated when diagnostics are enabled.
                entry['m_A_raw'] = torch.zeros_like(A, dtype=torch.float32)
                entry['v_A_raw'] = torch.zeros_like(A, dtype=torch.float32)
                entry['m_B_raw'] = torch.zeros_like(B, dtype=torch.float32)
                entry['v_B_raw'] = torch.zeros_like(B, dtype=torch.float32)
            self.pair_state[i] = entry

            r, d_in = A.shape
            if scaled_metric:
                self.gammas.append((d_in / r) ** 0.5)
            else:
                self.gammas.append(1.0)

    @torch.no_grad()
    def step(self, closure=None):
        """
        AdamLinLoRA update step.

        For each pair (A, B):
            1. Solve Sylvester equation for correction matrix K
            2. Compute linearized preconditioned gradients
            3. Update first moments: m ← β₁ m + (1-β₁) v
            4. Update second moments: v ← β₂ v + (1-β₂) v²
            5. Bias-correct and apply: Δθ = -lr * m̂ / (√v̂ + ε)
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, ((A, B), gamma) in enumerate(zip(self.pairs, self.gammas)):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamLinLoRA update.")

            state = self.pair_state[i]
            state['step'] += 1
            
            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Regularized Gram matrices: S_A = A A^T + δ I, S_B = B^T B + δ I.
            # Cache eigendecomp (for Sylvester solve) and Cholesky factor (for
            # the two solve_spd calls); refresh every K steps. K=1 reproduces
            # the original per-step behavior.
            need_refresh = (state['step'] - 1) % self.precond_refresh_every == 0
            if need_refresh:
                SB = spdify(B.T @ B, self.delta)
                SA = spdify(A @ A.T, self.delta)
                state['evalA'], state['QA'] = torch.linalg.eigh(SA)
                state['evalB'], state['QB'] = torch.linalg.eigh(SB)
                state['LA'] = torch.linalg.cholesky(SA)
                state['LB'] = torch.linalg.cholesky(SB)
                if self.log_diagnostics:
                    state['SA_eig_extremes'] = (float(state['evalA'][0]), float(state['evalA'][-1]))
                    state['SB_eig_extremes'] = (float(state['evalB'][0]), float(state['evalB'][-1]))
            evalA, QA = state['evalA'], state['QA']
            evalB, QB = state['evalB'], state['QB']
            LA, LB = state['LA'], state['LB']

            RHS = -gamma * (gA @ A.T).float()              # RHS = -(∇_A A^T) ∈ ℝ^{r×r} [no lr]

            # Solve Sylvester equation: S_B K + (γ² S_A) K^T = RHS via cached eigendecomps.
            # eigh(c·M) shares Q with eigh(M); eigenvalues scale by c, so multiply evalA by γ².
            T_syl = QB.T @ RHS @ QA                                         # (r, r)
            denom = evalB[:, None] + (gamma ** 2) * evalA[None, :]          # (r, r)
            K = QB @ (T_syl / denom) @ QA.T                                 # (r, r)

            # Compute preconditioned gradients (without lr factor)
            # precond_B = (∇_B + B K) S_A^{-1}
            termB = (gB + (1. / gamma) * B @ K.to(dtype=B.dtype)).float()   # ℝ^{d_out×r}
            precond_B = torch.cholesky_solve(termB.T, LA).T                 # ∈ ℝ^{d_out×r}

            # precond_A = S_B^{-1} (∇_A + K A)
            termA = (gA + gamma * K.to(dtype=A.dtype) @ A).float()          # ℝ^{r×d_in}
            precond_A = torch.cholesky_solve(termA, LB)                     # ∈ ℝ^{r×d_in}

            # Update first moment: m_t = β₁ m_{t-1} + (1-β₁) v_t
            state['m_A'].mul_(self.beta1).add_(precond_A, alpha=1 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(precond_B, alpha=1 - self.beta1)

            # Update second moment: v_t = β₂ v_{t-1} + (1-β₂) v_t²
            state['v_A'].mul_(self.beta2).addcmul_(precond_A, precond_A, value=1 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(precond_B, precond_B, value=1 - self.beta2)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            m_hat_A = state['m_A'] / bias_correction1
            m_hat_B = state['m_B'] / bias_correction1
            v_hat_A = state['v_A'] / bias_correction2
            v_hat_B = state['v_B'] / bias_correction2

            # Adam update: Δθ = -m̂ / (√v̂ + ε)
            dA = -lr * m_hat_A / (v_hat_A.sqrt() + self.eps)
            dB = -self.lora_plus_multiplier * lr * m_hat_B / (v_hat_B.sqrt() + self.eps)

            if self.log_diagnostics:
                # H1 probe: side-compute the plain-AdamW step on the same raw
                # gradient (independent m,v state), then compare directions.
                dA_raw = _adamw_side_step(
                    gA, state['m_A_raw'], state['v_A_raw'],
                    self.beta1, self.beta2, self.eps, state['step'], lr,
                )
                dB_raw = _adamw_side_step(
                    gB, state['m_B_raw'], state['v_B_raw'],
                    self.beta1, self.beta2, self.eps, state['step'], lr,
                )
                # SA/SB extremes from cached eigenvalues (free; computed at refresh).
                sa_min, sa_max = state['SA_eig_extremes']
                sb_min, sb_max = state['SB_eig_extremes']
                # cos_pre_*: cos(geometric direction, raw gradient) — measures
                # how much the geometric solve rotates the gradient BEFORE
                # Adam touches it. Distinguishes "weak geometry" (cos_pre ≈ 1
                # → S^{-1} near-identity on this gradient) from "geometry
                # erased by Adam's √v̂" (cos_pre < 1, cos_post ≈ 1).
                diag_records.append({
                    "cos_A": _frob_cos(dA, dA_raw),
                    "cos_B": _frob_cos(dB, dB_raw),
                    "cos_pre_A": _frob_cos(precond_A, gA),
                    "cos_pre_B": _frob_cos(precond_B, gB),
                    "norm_dA_lin": float(dA.detach().to(torch.float32).norm()),
                    "norm_dA_raw": float(dA_raw.detach().norm()),
                    "norm_dB_lin": float(dB.detach().to(torch.float32).norm()),
                    "norm_dB_raw": float(dB_raw.detach().norm()),
                    "norm_gA": float(gA.detach().to(torch.float32).norm()),
                    "norm_gB": float(gB.detach().to(torch.float32).norm()),
                    "norm_precond_A": float(precond_A.detach().to(torch.float32).norm()),
                    "norm_precond_B": float(precond_B.detach().to(torch.float32).norm()),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                })

            # Apply the update, cast back to parameter dtype/device
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamScaledLoRA(Optimizer):
    """
    AdamScaledLoRA: Adam-preconditioned version of ScaledLoRA.

    Applies Adam optimization to the preconditioned gradients computed by ScaledLoRA.
    For each LoRA pair (A, B), computes:
        v_A = S_B^{-1} ∇_A,  v_B = ∇_B S_A^{-1}
    where S_A = A A^T + δ I and S_B = B^T B + δ I. Then applies Adam
    updates using exponential moving averages (m, v) of v_A and v_B:
        m_t = β₁ m_{t-1} + (1-β₁) v_t
        v_t = β₂ v_{t-1} + (1-β₂) v_t²
        Δθ = -lr * m_t / (√v_t + ε)
    """
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None, log_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every

        # Initialize state: first and second moments for each (A, B) pair
        # Use pair_state instead of state to avoid conflicts with PyTorch's Optimizer.state
        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            entry = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }
            if log_diagnostics:
                entry['m_A_raw'] = torch.zeros_like(A, dtype=torch.float32)
                entry['v_A_raw'] = torch.zeros_like(A, dtype=torch.float32)
                entry['m_B_raw'] = torch.zeros_like(B, dtype=torch.float32)
                entry['v_B_raw'] = torch.zeros_like(B, dtype=torch.float32)
            self.pair_state[i] = entry

    @torch.no_grad()
    def step(self, closure=None):
        """
        AdamScaledLoRA update step.

        For each pair (A, B):
            1. Compute preconditioned gradients: v_A = S_B^{-1} ∇_A, v_B = ∇_B S_A^{-1}
            2. Update first moments: m_A ← β₁ m_A + (1-β₁) v_A
            3. Update second moments: v_A ← β₂ v_A + (1-β₂) v_A²
            4. Bias-correct and apply: Δθ = -lr * m̂ / (√v̂ + ε)
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamScaledLoRA update.")

            state = self.pair_state[i]
            state['step'] += 1
            
            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Preconditioning matrices: S_A = A A^T + δ I, S_B = B^T B + δ I.
            # Cached as Cholesky factors and refreshed every K steps; cholesky_solve
            # against the cached factor between refreshes. K=1 ⇒ refresh every step
            # (original behavior). For diagnostics we still need SA/SB on refresh
            # steps; on stale steps we skip the materialization.
            need_refresh = (state['step'] - 1) % self.precond_refresh_every == 0
            if need_refresh:
                SB = spdify(B.T @ B, self.delta)       # S_B ∈ ℝ^{r×r}
                SA = spdify(A @ A.T, self.delta)       # S_A ∈ ℝ^{r×r}
                state['LA'] = torch.linalg.cholesky(SA)
                state['LB'] = torch.linalg.cholesky(SB)
                if self.log_diagnostics:
                    state['SA_for_diag'] = SA
                    state['SB_for_diag'] = SB
            LA = state['LA']
            LB = state['LB']

            # Compute preconditioned gradients (not scaled by lr yet). Match
            # solve_spd's dtype handling — gA/gB may be bf16; cholesky_solve
            # requires matching dtype with the cached float32 L, so cast here.
            precond_B = torch.cholesky_solve(gB.T.to(LA.dtype), LA).T   # ∇_B S_A^{-1} ∈ ℝ^{d_out×r}
            precond_A = torch.cholesky_solve(gA.to(LB.dtype), LB)       # S_B^{-1} ∇_A ∈ ℝ^{r×d_in}

            # Update first moment: m_t = β₁ m_{t-1} + (1-β₁) v_t
            state['m_A'].mul_(self.beta1).add_(precond_A, alpha=1 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(precond_B, alpha=1 - self.beta1)

            # Update second moment: v_t = β₂ v_{t-1} + (1-β₂) v_t²
            state['v_A'].mul_(self.beta2).addcmul_(precond_A, precond_A, value=1 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(precond_B, precond_B, value=1 - self.beta2)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            m_hat_A = state['m_A'] / bias_correction1
            m_hat_B = state['m_B'] / bias_correction1
            v_hat_A = state['v_A'] / bias_correction2
            v_hat_B = state['v_B'] / bias_correction2

            # Adam update: Δθ = -lr * m̂ / (√v̂ + ε)
            dA = -lr * m_hat_A / (v_hat_A.sqrt() + self.eps)
            dB = -lr * m_hat_B / (v_hat_B.sqrt() + self.eps)

            if self.log_diagnostics:
                dA_raw = _adamw_side_step(
                    gA, state['m_A_raw'], state['v_A_raw'],
                    self.beta1, self.beta2, self.eps, state['step'], lr,
                )
                dB_raw = _adamw_side_step(
                    gB, state['m_B_raw'], state['v_B_raw'],
                    self.beta1, self.beta2, self.eps, state['step'], lr,
                )
                # SA/SB only re-materialize on refresh; otherwise read from cache.
                SA_diag = state['SA_for_diag'] if not need_refresh else SA
                SB_diag = state['SB_for_diag'] if not need_refresh else SB
                sa_min, sa_max = _spd_eig_extremes(SA_diag)
                sb_min, sb_max = _spd_eig_extremes(SB_diag)
                # cos_pre_*: cos(geometric direction, raw gradient) — measures
                # how much the geometric solve rotates the gradient BEFORE
                # Adam touches it. Distinguishes "weak geometry" (cos_pre ≈ 1
                # → S^{-1} near-identity on this gradient) from "geometry
                # erased by Adam's √v̂" (cos_pre < 1, cos_post ≈ 1).
                diag_records.append({
                    "cos_A": _frob_cos(dA, dA_raw),
                    "cos_B": _frob_cos(dB, dB_raw),
                    "cos_pre_A": _frob_cos(precond_A, gA),
                    "cos_pre_B": _frob_cos(precond_B, gB),
                    "norm_dA_lin": float(dA.detach().to(torch.float32).norm()),
                    "norm_dA_raw": float(dA_raw.detach().norm()),
                    "norm_dB_lin": float(dB.detach().to(torch.float32).norm()),
                    "norm_dB_raw": float(dB_raw.detach().norm()),
                    "norm_gA": float(gA.detach().to(torch.float32).norm()),
                    "norm_gB": float(gB.detach().to(torch.float32).norm()),
                    "norm_precond_A": float(precond_A.detach().to(torch.float32).norm()),
                    "norm_precond_B": float(precond_B.detach().to(torch.float32).norm()),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                })

            # Apply the update, cast back to parameter dtype/device
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamScaledLoRAPost(Optimizer):
    """AdamScaledLoRA with composition order swapped (H4).

    The original AdamScaledLoRA applies Adam (m,v) to the *geometrically
    preconditioned* gradient v = S⁻¹∇. Adam's per-coord √v̂ then re-normalizes
    away the cross-coordinate scale structure that the Gram solve installed.

    This variant maintains Adam state on the *raw* gradient (∇A, ∇B), produces
    the unitless Adam direction u = m̂/(√v̂+ε), then applies the Gram solve
    *after*: ΔA = −lr · S_B⁻¹ u_A, ΔB = −lr · u_B S_A⁻¹.

    v̂ adapts to the natural gradient distribution (its strength); the geometry
    installs the (A,B) coupling on the Adam *step*, not the gradient.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, adapter_name=None, log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamScaledLoRAPost update.")

            state = self.pair_state[i]
            state['step'] += 1

            gA = A.grad.float()    # raw ∇_A
            gB = B.grad.float()    # raw ∇_B

            # Adam state on the RAW gradient (the key reordering vs AdamScaledLoRA).
            state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
            state['v_A'].mul_(self.beta2).addcmul_(gA, gA, value=1.0 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(gB, gB, value=1.0 - self.beta2)

            bc1 = 1.0 - self.beta1 ** state['step']
            bc2 = 1.0 - self.beta2 ** state['step']
            u_A = (state['m_A'] / bc1) / ((state['v_A'] / bc2).sqrt() + self.eps)
            u_B = (state['m_B'] / bc1) / ((state['v_B'] / bc2).sqrt() + self.eps)

            # Gram solve applied to the Adam direction, NOT the raw gradient.
            SB = spdify(B.T @ B, self.delta)
            SA = spdify(A @ A.T, self.delta)
            geo_A = solve_spd(SB, u_A)
            geo_B = solve_spd(SA, u_B.T).T

            # RMS-align: rescale geometric step so its Frobenius norm matches
            # the bare Adam step's. Without this, ‖S^{-1} u‖_F varies as
            # 1/σ_min(S) which moves by ~100× over training (σ_min ~0.01 → 1
            # as ‖B‖ grows). Effective lr drifts; learning rate sweeps
            # become uninterpretable. Cribbed from AdaMuon (arxiv 2507.11005)
            # γ_t = ‖target‖_F / ‖raw‖_F rescaling.
            uA_norm = u_A.float().norm()
            uB_norm = u_B.float().norm()
            gA_norm = geo_A.float().norm() + 1e-30
            gB_norm = geo_B.float().norm() + 1e-30
            dA = -lr * (uA_norm / gA_norm) * geo_A
            dB = -lr * (uB_norm / gB_norm) * geo_B

            if self.log_diagnostics:
                # cos(applied_step, plain-AdamW-direction). Plain AdamW would
                # apply -lr·u_A; we compute cos(dA, -u_A) so the sign is
                # consistent across Pre and Post variants regardless of
                # whether the negative is baked into geo_X or the lr factor.
                sa_min, sa_max = _spd_eig_extremes(SA)
                sb_min, sb_max = _spd_eig_extremes(SB)
                diag_records.append({
                    "cos_A": _frob_cos(dA, -u_A),
                    "cos_B": _frob_cos(dB, -u_B),
                    "norm_dA_post": float(dA.detach().to(torch.float32).norm()),
                    "norm_dA_adamw_eq": float(lr * uA_norm),
                    "norm_dB_post": float(dB.detach().to(torch.float32).norm()),
                    "norm_dB_adamw_eq": float(lr * uB_norm),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                    "rms_scale_A": float(uA_norm / gA_norm),
                    "rms_scale_B": float(uB_norm / gB_norm),
                })

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamLinLoRAPost(Optimizer):
    """AdamLinLoRA with composition order swapped (H4).

    Adam state runs on raw (∇A, ∇B). The unitless Adam direction
    u_A = m̂_A/(√v̂_A+ε), u_B analogous, is then fed through the LinLoRA
    Sylvester step *as if it were the gradient*: solve

        S_B K + γ² K S_A = -γ · lr · (u_A A^T)

    and apply

        ΔA = -S_B⁻¹ (lr · u_A + γ K A)
        ΔB = -(lr · u_B + (1/γ) B K) S_A⁻¹.

    Sylvester coupling is preserved on the Adam step rather than the gradient.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, adapter_name=None, scaled_metric=False,
                 lora_plus_multiplier=1.0, log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every

        self.pair_state = {}
        self.gammas = []
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }
            r, d_in = A.shape
            self.gammas.append((d_in / r) ** 0.5 if scaled_metric else 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, ((A, B), gamma) in enumerate(zip(self.pairs, self.gammas)):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamLinLoRAPost update.")

            state = self.pair_state[i]
            state['step'] += 1

            gA = A.grad.float()
            gB = B.grad.float()

            state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
            state['v_A'].mul_(self.beta2).addcmul_(gA, gA, value=1.0 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(gB, gB, value=1.0 - self.beta2)

            bc1 = 1.0 - self.beta1 ** state['step']
            bc2 = 1.0 - self.beta2 ** state['step']
            u_A = (state['m_A'] / bc1) / ((state['v_A'] / bc2).sqrt() + self.eps)
            u_B = (state['m_B'] / bc1) / ((state['v_B'] / bc2).sqrt() + self.eps)

            # Substitute u_A, u_B for ∇_A, ∇_B in the LinLoRA derivation.
            # Factor lr out of the linear system so we can RMS-align the
            # resulting direction afterwards (the original formulation bakes
            # lr into K and termA, making ‖ΔA‖_F vary as lr/σ_min(S_B), which
            # in our setup drifts by ~100× over training as ‖B‖ grows).
            SB = spdify(B.T @ B, self.delta)
            SA = spdify(A @ A.T, self.delta)
            RHS = -gamma * (u_A @ A.T.float())
            K = solve_sylvester(SB, (gamma ** 2) * SA, RHS)

            termA = (u_A + gamma * K.to(u_A.dtype) @ A.float())
            geo_A = -solve_spd(SB, termA)               # lr-free direction
            termB = (u_B + (1.0 / gamma) * B.float() @ K.to(u_B.dtype))
            geo_B = -solve_spd(SA, termB.T).T

            # RMS-align (cribbed from AdaMuon, arxiv 2507.11005): rescale so
            # ‖ΔA‖_F = lr·‖u_A‖_F, decoupling step magnitude from S_B
            # conditioning. lr now controls magnitude only; geometry controls
            # direction.
            uA_norm = u_A.float().norm()
            uB_norm = u_B.float().norm()
            gA_norm = geo_A.float().norm() + 1e-30
            gB_norm = geo_B.float().norm() + 1e-30
            dA = lr * (uA_norm / gA_norm) * geo_A
            dB = lr * (uB_norm / gB_norm) * geo_B

            # lora+ multiplier scales the B-side step (matching AdamLinLoRA semantics).
            if self.lora_plus_multiplier != 1.0:
                dB = self.lora_plus_multiplier * dB

            if self.log_diagnostics:
                # cos(applied_step, plain-AdamW-direction). See AdamScaledLoRAPost
                # for sign-convention rationale.
                sa_min, sa_max = _spd_eig_extremes(SA)
                sb_min, sb_max = _spd_eig_extremes(SB)
                diag_records.append({
                    "cos_A": _frob_cos(dA, -u_A),
                    "cos_B": _frob_cos(dB, -u_B),
                    "norm_dA_post": float(dA.detach().to(torch.float32).norm()),
                    "norm_dA_adamw_eq": float(lr * uA_norm),
                    "norm_dB_post": float(dB.detach().to(torch.float32).norm()),
                    "norm_dB_adamw_eq": float(lr * uB_norm),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                    "rms_scale_A": float(uA_norm / gA_norm),
                    "rms_scale_B": float(uB_norm / gB_norm),
                })

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamScaledLoRAMatrix(Optimizer):
    """AdamScaledLoRA with a per-pair *scalar* second moment (H5).

    Original AdamScaledLoRA uses per-coordinate v̂ on the preconditioned
    gradient v = S⁻¹∇. The √v̂ rescaling acts coordinate-by-coordinate,
    re-shredding the cross-coordinate scale structure that the Gram solve
    just installed. This variant keeps per-element first-moment m, but
    replaces v̂ with a single scalar EMA per (A,B) pair tracking the
    mean-square ‖precond‖² / N_total over the joint pair. The pair gets an
    adaptive learning rate (Adam's stability) without per-coord directional
    shredding.

    NOTE on normalization: v_pair is the *mean* of squared elements across
    the pair, not the sum. With sum-of-squares, √v̂ ≈ √N · RMS(g) and the
    effective per-coord lr is lr/√N — for typical LoRA shapes that scales
    the η range by ~1/700 and the optimizer fails to learn at the η values
    that work for AdamLinLoRA / AdamScaledLoRA. With mean-of-squares, √v̂
    has units of |g| (same as per-coord Adam), so the same η transfers.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_pair': 0.0,
                'n_total': float(A.numel() + B.numel()),
                'step': 0,
            }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamScaledLoRAMatrix update.")
            state = self.pair_state[i]
            state['step'] += 1
            gA = A.grad
            gB = B.grad
            SB = spdify(B.T @ B, self.delta)
            SA = spdify(A @ A.T, self.delta)
            precond_A = solve_spd(SB, gA)
            precond_B = solve_spd(SA, gB.T).T

            state['m_A'].mul_(self.beta1).add_(precond_A, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(precond_B, alpha=1.0 - self.beta1)
            # Mean-square (not sum-of-squares) so √v̂ has units of |g|.
            sqmean = float((precond_A.float() ** 2).sum() + (precond_B.float() ** 2).sum()) / state['n_total']
            state['v_pair'] = self.beta2 * state['v_pair'] + (1.0 - self.beta2) * sqmean

            bc1 = 1.0 - self.beta1 ** state['step']
            bc2 = 1.0 - self.beta2 ** state['step']
            denom = (state['v_pair'] / bc2) ** 0.5 + self.eps
            scale = -lr / denom

            dA = scale * (state['m_A'] / bc1)
            dB = scale * (state['m_B'] / bc1)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class AdamLinLoRAMatrix(Optimizer):
    """AdamLinLoRA with per-pair scalar second moment (H5).

    Same as AdamLinLoRA but Adam's per-coordinate v̂ is replaced by a scalar
    EMA per (A,B) pair tracking the *mean square* ‖precond‖² / N_total over
    the joint pair (where precond is the Sylvester-corrected step). Direction
    comes from m̂ per-element; only magnitude is adaptively rescaled per pair.

    See AdamScaledLoRAMatrix docstring for the mean-vs-sum normalization
    rationale — without it the optimizer fails to learn at the standard η.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, adapter_name=None, scaled_metric=False,
                 lora_plus_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.lora_plus_multiplier = lora_plus_multiplier

        self.pair_state = {}
        self.gammas = []
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_pair': 0.0,
                'n_total': float(A.numel() + B.numel()),
                'step': 0,
            }
            r, d_in = A.shape
            self.gammas.append((d_in / r) ** 0.5 if scaled_metric else 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, ((A, B), gamma) in enumerate(zip(self.pairs, self.gammas)):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamLinLoRAMatrix update.")
            state = self.pair_state[i]
            state['step'] += 1
            gA = A.grad
            gB = B.grad
            SB = spdify(B.T @ B, self.delta)
            SA = spdify(A @ A.T, self.delta)
            RHS = -gamma * (gA @ A.T).float()
            K = solve_sylvester(SB, (gamma ** 2) * SA, RHS)
            termB = (gB + (1.0 / gamma) * B @ K.to(dtype=B.dtype)).float()
            precond_B = solve_spd(SA, termB.T).T
            termA = (gA + gamma * K.to(dtype=A.dtype) @ A).float()
            precond_A = solve_spd(SB, termA)

            state['m_A'].mul_(self.beta1).add_(precond_A, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(precond_B, alpha=1.0 - self.beta1)
            # Mean-square (not sum-of-squares) so √v̂ has units of |g|.
            sqmean = float((precond_A.float() ** 2).sum() + (precond_B.float() ** 2).sum()) / state['n_total']
            state['v_pair'] = self.beta2 * state['v_pair'] + (1.0 - self.beta2) * sqmean

            bc1 = 1.0 - self.beta1 ** state['step']
            bc2 = 1.0 - self.beta2 ** state['step']
            denom = (state['v_pair'] / bc2) ** 0.5 + self.eps
            scale = -lr / denom

            dA = scale * (state['m_A'] / bc1)
            dB = self.lora_plus_multiplier * scale * (state['m_B'] / bc1)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class LoRAPlusAdamW(AdamW):
    """
    LoRA+ AdamW: AdamW with different learning rates for lora_A and lora_B.
    
    Applies lora_plus_multiplier to lora_B learning rate. Reduces to standard AdamW when multiplier=1.0.
    """
    def __init__(self, model, lr=2e-4, lora_plus_multiplier=1.0, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.0, adapter_name=None):
        lora_A_params = []
        lora_B_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                lora_A_params.append(param)
            elif "lora_B" in name:
                lora_B_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        if lora_A_params:
            param_groups.append({"params": lora_A_params, "lr": lr})
        if lora_B_params:
            param_groups.append({"params": lora_B_params, "lr": lr * lora_plus_multiplier})
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        
        super().__init__(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class SVDStepAdamW(AdamW):
    """
    AdamW in dense target-weight space with rank-r projection of each step.

    The dense AdamW proposal defines Delta W_t = W_tilde - W_t, then the applied
    update is Pi_r(Delta W_t). The cumulative displacement from initialization
    is not constrained to rank r.
    """
    def __init__(self, targets, rank, lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, svd_niter=4):
        if not targets:
            raise ValueError("SVDStepAdamW requires at least one dense target weight.")
        if weight_decay != 0.0:
            raise ValueError("SVDStepAdamW currently requires weight_decay=0.0.")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        self.targets = list(targets)
        self.rank = rank
        self.svd_niter = svd_niter
        super().__init__(
            [target.weight for target in self.targets],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure=None):
        before = {
            target.name: target.weight.detach().float().clone()
            for target in self.targets
        }
        loss = super().step(closure)
        for target in self.targets:
            raw_delta = target.weight.detach().float() - before[target.name]
            step_delta = truncated_svd(raw_delta, self.rank, niter=self.svd_niter)
            target.weight.copy_(
                (before[target.name] + step_delta).to(
                    dtype=target.weight.dtype,
                    device=target.weight.device,
                )
            )
        return loss


class SVDCumulativeAdamW(AdamW):
    """
    AdamW in dense target-weight space with rank-r cumulative displacement.

    The dense AdamW proposal defines Delta W_t = W_tilde - W_t. These proposals
    are accumulated in full-rank float32 buffers C_t, but the live model weight
    is W0 + Pi_r(C_t), so each target displacement from W0 remains rank r.
    """
    def __init__(self, targets, rank, lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, svd_niter=4):
        if not targets:
            raise ValueError("SVDCumulativeAdamW requires at least one dense target weight.")
        if weight_decay != 0.0:
            raise ValueError("SVDCumulativeAdamW currently requires weight_decay=0.0.")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        self.targets = list(targets)
        self.rank = rank
        self.svd_niter = svd_niter
        self.accumulators = {
            target.name: torch.zeros_like(target.base_weight, dtype=torch.float32)
            for target in self.targets
        }
        super().__init__(
            [target.weight for target in self.targets],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure=None):
        before = {
            target.name: target.weight.detach().float().clone()
            for target in self.targets
        }
        loss = super().step(closure)
        for target in self.targets:
            raw_delta = target.weight.detach().float() - before[target.name]
            self.accumulators[target.name].add_(raw_delta)
            projected = truncated_svd(self.accumulators[target.name], self.rank,
                                      niter=self.svd_niter)
            target.weight.copy_(
                (target.base_weight + projected).to(
                    dtype=target.weight.dtype,
                    device=target.weight.device,
                )
            )
        return loss


def _newton_schulz(X, nsteps=5, eps=1e-7):
    """
    Newton-Schulz orthogonalization of X (float32). Canonical Muon: returns a
    matrix with approximately orthonormal rows (or cols, for tall X), Frobenius
    norm ≈ √min(r, d), INDEPENDENT of the input magnitude. This is the whole
    point of Muon — the optimizer's step size is set by lr alone, not by the
    (highly variable, especially early-training) gradient magnitude.

    Pre-normalize by spectral-norm proxy ||X||_F (only to bring singular values
    into NS's basin of attraction near 1). Do NOT multiply back by the norm.
    """
    X = X.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    # X is now (r, d) with r ≤ d
    norm = X.norm() + eps
    X = X / norm
    for _ in range(nsteps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X.T if tall else X


class MuonLoRA(Optimizer):
    """
    MuonLoRA: momentum + Newton-Schulz orthogonalization for LoRA factors.

    Update rule per LoRA pair A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}:
        m_A ← β m_A + (1−β) G_A
        m_B ← β m_B + (1−β) G_B
        A   ← A − lr · D(m_A)
        B   ← B − m·lr · D(m_B)
    where D = NS when ns_steps > 0 (orthogonalizing direction; canonical Muon),
    and D = identity when ns_steps == 0 (raw momentum SGD; Tier-2 sanity check
    isolating the contribution of the orthogonalization itself).

    The lr_b_multiplier (m above) plays the LoRA+ role for Muon: PEFT inits
    B=0, so B's update needs to grow faster than A's to make the linearized
    weight update ΔW ≈ (α/r)(B·δA + δB·A) effective early in training. m=1
    recovers vanilla Muon.
    """
    def __init__(self, model, lr=3e-4, beta=0.95, ns_steps=5, adapter_name=None,
                 lr_b_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta = beta
        self.ns_steps = ns_steps
        self.lr_b_multiplier = lr_b_multiplier
        self.pair_state = {
            i: {
                "m_A": torch.zeros_like(A, dtype=torch.float32),
                "m_B": torch.zeros_like(B, dtype=torch.float32),
            }
            for i, (A, B) in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("MuonLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            gA = A.grad.float()
            gB = B.grad.float()
            state["m_A"].mul_(self.beta).add_(gA, alpha=1 - self.beta)
            state["m_B"].mul_(self.beta).add_(gB, alpha=1 - self.beta)
            if self.ns_steps > 0:
                dA = _newton_schulz(state["m_A"], self.ns_steps)
                dB = _newton_schulz(state["m_B"], self.ns_steps)
            else:
                dA = state["m_A"]
                dB = state["m_B"]
            A.add_((-lr * dA).to(dtype=A.dtype, device=A.device))
            B.add_((-self.lr_b_multiplier * lr * dB).to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class ProductMuonLoRA(Optimizer):
    """
    ProductMuonLoRA: Muon on the merged-weight gradient projected onto the LoRA
    subspace, then Sylvester-recovered into factor updates.

    Mathematically (theory doc lemma at line 622, identity at line 660):
        ΔW = polar(Ĝ) · V_A V_A^T
    where Ĝ ∈ ℝ^{d_out × d_in} is the merged-weight gradient and V_A V_A^T is
    the orthogonal projector onto the row-space of A. Equivalent rank-r
    representation in terms of available factor gradients:
        Ĝ V_A V_A^T = (r/α) · ∇_B · (A A^T + δI)^{-1} · A
    (gauge-invariant under A → R A, B → B R^{-1} for any invertible R, since
    A A^T transforms as R A A^T R^T and the R cancels).

    Per pair, per step:
      1. EMA-momentum the *merged-direction* proxy
             D_t = (1/scale) · m_B · (A A^T + δI)^{-1} · A      (rank ≤ r, gauge-invariant)
      2. Apply NS to D_t in factored form (the rank-r thin SVD: QR(m_B) and
         QR((solve_spd(SA, A))^T), then NS the small r × r core).
      3. Sylvester-recover (δA, δB) such that B δA + δB A = -lr · NS(D_t).
         Reuses the exact path from AdamLinLoRA.

    The ∇_A-channel is *not* used to construct D — it would re-introduce
    gauge-dependence (B^T G doesn't have a clean gauge transform when
    composed with B m_A). At B=0 init, ∇_A is also zero on the relevant
    component, so dropping it costs nothing early; later, ∇_B carries the
    full merged-gradient signal.

    The lr_b_multiplier provides LoRA+ asymmetry — orthogonal to the
    geometric correctness above, addresses the H1 (B=0 init) hypothesis.
    """
    def __init__(self, model, lr=3e-4, beta=0.95, ns_steps=5, alpha=16, rank=16,
                 delta=1e-6, adapter_name=None, lr_b_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta = beta
        self.ns_steps = ns_steps
        self.scale = alpha / rank
        self.delta = delta
        self.lr_b_multiplier = lr_b_multiplier
        self.pair_state = {
            i: {
                # Momentum tracks the gauge-invariant merged-direction proxy D
                # rather than raw factor gradients — this keeps the EMA itself
                # gauge-invariant under A → R A.
                "m_D": None,  # lazy-init at first step (need d_out, d_in)
            }
            for i, _ in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("ProductMuonLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            gB = B.grad.float()                                # (d_out, r)
            A_f = A.detach().float()                           # (r, d_in)
            B_f = B.detach().float()                           # (d_out, r)

            # Build the gauge-invariant rank-r merged direction in factored form:
            #     D = m_B_left · Z   where Z = (A A^T + δI)^{-1} A ∈ (r, d_in).
            # Stash m_B_left = gB; momentum is on (m_B_left, Z) jointly via
            # tracking the rank-r D itself in factored form. For simplicity and
            # correctness, we EMA-update the d_out × r left factor (gB) while
            # recomputing Z each step (it depends on current A, which moves).
            SA = spdify(A_f @ A_f.T, self.delta)               # (r, r)
            Z = solve_spd(SA, A_f)                             # (r, d_in)

            # Track the merged-direction proxy D = (1/scale) gB Z. Momentum on
            # gB (a gauge-invariant quantity in the column-space-of-B sense).
            # NB: at B=0 init, gB is the only signal — exactly right.
            inv_scale = 1.0 / self.scale
            m_left = state.get("m_left")
            if m_left is None:
                m_left = torch.zeros_like(gB, dtype=torch.float32)
                state["m_left"] = m_left
            m_left.mul_(self.beta).add_(gB, alpha=1 - self.beta)
            left = inv_scale * m_left                          # (d_out, r)
            right = Z                                          # (r, d_in)

            # NS on the rank-r product `left @ right` via thin QR + small NS.
            Q_L, R_L = torch.linalg.qr(left, mode="reduced")          # (d_out, r), (r, r)
            Q_R, R_R = torch.linalg.qr(right.T, mode="reduced")       # (d_in, r),  (r, r)
            C = R_L @ R_R.T                                            # (r, r)
            if self.ns_steps > 0:
                C_ns = _newton_schulz(C, self.ns_steps)
            else:
                C_ns = C
            # Target merged direction U = Q_L · C_ns · Q_R^T (rank ≤ r).

            # Recover (δA, δB) by AdamLinLoRA's Sylvester least-squares:
            #   want B δA + δB A = -lr · U  (linearized merged-weight prox).
            # Compute grad-equivalents without forming U as a (d_out × d_in) matrix:
            BtQ_L = B_f.T @ Q_L                                # (r, r)
            QR_tAt = Q_R.T @ A_f.T                             # (r, r)
            grad_A_eq = (BtQ_L @ C_ns) @ Q_R.T                 # (r, d_in)
            grad_B_eq = Q_L @ (C_ns @ QR_tAt)                  # (d_out, r)

            SB = spdify(B_f.T @ B_f, self.delta)               # (r, r)
            RHS = -(grad_A_eq @ A_f.T)                         # (r, r)
            K = solve_sylvester(SB, SA, RHS)                   # (r, r)
            termB = grad_B_eq + B_f @ K                        # (d_out, r)
            precond_B = solve_spd(SA, termB.T).T               # (d_out, r)
            termA = grad_A_eq + K @ A_f                        # (r, d_in)
            precond_A = solve_spd(SB, termA)                   # (r, d_in)

            A.add_((-lr * precond_A).to(dtype=A.dtype, device=A.device))
            B.add_((-self.lr_b_multiplier * lr * precond_B).to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class AdamMuonLoRA(Optimizer):
    """
    AdamMuonLoRA (Tier 4): Newton-Schulz applied to Adam's preconditioned
    direction m̂/(√v̂+ε) instead of raw momentum. Decouples diagonal
    preconditioning (Adam) from spectral capping (Muon). Per LoRA factor
    independently — this is the cheap analog of AdamLinLoRA in Muon space.
    """
    def __init__(self, model, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, ns_steps=5,
                 adapter_name=None, lr_b_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        self.lr_b_multiplier = lr_b_multiplier
        self.pair_state = {
            i: {
                "m_A": torch.zeros_like(A, dtype=torch.float32),
                "v_A": torch.zeros_like(A, dtype=torch.float32),
                "m_B": torch.zeros_like(B, dtype=torch.float32),
                "v_B": torch.zeros_like(B, dtype=torch.float32),
                "step": 0,
            }
            for i, (A, B) in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("AdamMuonLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            state["step"] += 1
            t = state["step"]
            gA = A.grad.float()
            gB = B.grad.float()
            state["m_A"].mul_(self.beta1).add_(gA, alpha=1 - self.beta1)
            state["m_B"].mul_(self.beta1).add_(gB, alpha=1 - self.beta1)
            state["v_A"].mul_(self.beta2).addcmul_(gA, gA, value=1 - self.beta2)
            state["v_B"].mul_(self.beta2).addcmul_(gB, gB, value=1 - self.beta2)
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
            adam_A = (state["m_A"] / bc1) / ((state["v_A"] / bc2).sqrt() + self.eps)
            adam_B = (state["m_B"] / bc1) / ((state["v_B"] / bc2).sqrt() + self.eps)
            if self.ns_steps > 0:
                dA = _newton_schulz(adam_A, self.ns_steps)
                dB = _newton_schulz(adam_B, self.ns_steps)
            else:
                dA = adam_A
                dB = adam_B
            A.add_((-lr * dA).to(dtype=A.dtype, device=A.device))
            B.add_((-self.lr_b_multiplier * lr * dB).to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class AdamProductMuonLoRA(Optimizer):
    """
    H2 ⊗ H4 hybrid: ProductMuonLoRA's gauge-invariant geometry + Adam EMA on
    the recovered factor updates.

    Pipeline per pair, per step:
      1. Z = (A Aᵀ + δI)⁻¹ A                       (rank-r right factor, gauge-inv)
      2. left = (1/scale) · gB                     (no momentum yet — Adam acts later)
      3. Build rank-r merged-direction proxy D = left @ Z, NS via factored form
         (QR on left and Z.T, NS the small r×r core).
      4. Sylvester-recover (precond_A, precond_B) such that B precond_A + precond_B A
         ≈ NS-direction. Mirrors AdamLinLoRA.
      5. Adam on (precond_A, precond_B): EMA m, v with bias correction; final
         update is -lr · m̂/(√v̂ + ε).

    Combines H2 (correct geometry) with H4 (Adam preconditioning, AFTER the
    geometric solve a la AdamLinLoRA — not before).

    Note: at B=0 init, Sylvester min-Frobenius sets precond_A = 0, so A doesn't
    update until B grows. The lr_b_multiplier knob handles this.
    """
    def __init__(self, model, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, ns_steps=5,
                 alpha=16, rank=16, delta=1e-6, adapter_name=None, lr_b_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        self.scale = alpha / rank
        self.delta = delta
        self.lr_b_multiplier = lr_b_multiplier
        self.pair_state = {
            i: {
                "m_A": torch.zeros_like(A, dtype=torch.float32),
                "v_A": torch.zeros_like(A, dtype=torch.float32),
                "m_B": torch.zeros_like(B, dtype=torch.float32),
                "v_B": torch.zeros_like(B, dtype=torch.float32),
                "step": 0,
            }
            for i, (A, B) in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("AdamProductMuonLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            state["step"] += 1
            t = state["step"]

            gB = B.grad.float()
            A_f = A.detach().float()
            B_f = B.detach().float()
            inv_scale = 1.0 / self.scale

            SA = spdify(A_f @ A_f.T, self.delta)
            Z = solve_spd(SA, A_f)
            left = inv_scale * gB
            right = Z

            Q_L, R_L = torch.linalg.qr(left, mode="reduced")
            Q_R, R_R = torch.linalg.qr(right.T, mode="reduced")
            C = R_L @ R_R.T
            if self.ns_steps > 0:
                C_ns = _newton_schulz(C, self.ns_steps)
            else:
                C_ns = C

            BtQ_L = B_f.T @ Q_L
            QR_tAt = Q_R.T @ A_f.T
            grad_A_eq = (BtQ_L @ C_ns) @ Q_R.T
            grad_B_eq = Q_L @ (C_ns @ QR_tAt)

            SB = spdify(B_f.T @ B_f, self.delta)
            RHS = -(grad_A_eq @ A_f.T)
            K = solve_sylvester(SB, SA, RHS)
            termB = grad_B_eq + B_f @ K
            precond_B = solve_spd(SA, termB.T).T
            termA = grad_A_eq + K @ A_f
            precond_A = solve_spd(SB, termA)

            state["m_A"].mul_(self.beta1).add_(precond_A, alpha=1 - self.beta1)
            state["m_B"].mul_(self.beta1).add_(precond_B, alpha=1 - self.beta1)
            state["v_A"].mul_(self.beta2).addcmul_(precond_A, precond_A, value=1 - self.beta2)
            state["v_B"].mul_(self.beta2).addcmul_(precond_B, precond_B, value=1 - self.beta2)
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
            m_hat_A = state["m_A"] / bc1
            m_hat_B = state["m_B"] / bc1
            v_hat_A = state["v_A"] / bc2
            v_hat_B = state["v_B"] / bc2
            dA = -lr * m_hat_A / (v_hat_A.sqrt() + self.eps)
            dB = -self.lr_b_multiplier * lr * m_hat_B / (v_hat_B.sqrt() + self.eps)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class PolarProductLoRA(Optimizer):
    """Closed-form polar update under spectral-product norm (theory line 622-660).

    Solves   min_{ΔA, ΔB}  ⟨∇_A F, ΔA⟩ + ⟨∇_B F, ΔB⟩ + (1/2λ)·(‖ΔB·A‖²₂ + ‖B·ΔA‖²₂)

    Closed-form per the theory's lemma:
        ΔB = -lr · polar(∇B · S_A⁻¹ᐟ²) · S_A⁻¹ᐟ²        where S_A = AAᵀ + δI
        ΔA = -lr · S_B⁻¹ᐟ² · polar(S_B⁻¹ᐟ² · ∇A)        where S_B = BᵀB + δI

    "polar" here is approximated by Newton-Schulz orthogonalization (canonical
    Muon, Frobenius-norm-preserving variant). This sandwiches the spectral cap
    between two factors of a *spectral square root* preconditioner — gentler
    than the full S⁻¹ used in scaled-lora (which is what blew up in the
    unfixed *-Post variants). The composition uses BOTH the LoRA product
    structure (∇A's update depends on B; ∇B's update depends on A) AND a
    spectral correction (polar). Plain SGD-style (no Adam).

    Cost per pair per step: 2× r×r eigendecomposition (for S^{-1/2}, cheap at
    r=16/64), 2× Newton-Schulz polar on (m×r) and (d_out×r) matrices.
    """

    def __init__(self, model, lr=2e-4, delta=1e-6, ns_steps=5, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.ns_steps = ns_steps

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]

        for A, B in self.pairs:
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for PolarProductLoRA update.")
            gA = A.grad.float()
            gB = B.grad.float()

            # r × r SPD square-root inverses
            SA_half_inv = spd_frac_power_inv(A.float() @ A.float().T, gamma=0.5, eps=self.delta)
            SB_half_inv = spd_frac_power_inv(B.float().T @ B.float(), gamma=0.5, eps=self.delta)

            # ΔB = -lr · polar(∇B · S_A^{-1/2}) · S_A^{-1/2}
            X_B = gB @ SA_half_inv                        # (d_out, r)
            P_B = _newton_schulz(X_B, nsteps=self.ns_steps)
            dB = -lr * (P_B @ SA_half_inv)                # (d_out, r)

            # ΔA = -lr · S_B^{-1/2} · polar(S_B^{-1/2} · ∇A)
            X_A = SB_half_inv @ gA                        # (r, d_in)
            P_A = _newton_schulz(X_A, nsteps=self.ns_steps)
            dA = -lr * (SB_half_inv @ P_A)                # (r, d_in)

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class AdamPolarProductLoRA(Optimizer):
    """Polar-product update applied to Adam's denoised direction.

    Composition: Adam runs on raw (∇A, ∇B); the resulting Adam direction
    u_A = m̂_A/(√v̂_A+ε), u_B analogous, is then fed through the same
    polar-product update as PolarProductLoRA (substituting u for ∇).

    By H1 measurements, plain pre-Adam preconditioning is erased by Adam's
    per-coord √v̂ on a sign-like input. Here the geometric correction is
    polar (matrix-structural — invariant to per-coord rescaling), so it
    survives Adam by construction. RMS-aligned step magnitude (cribbed from
    AdaMuon arxiv 2507.11005 and our H4 fix) prevents the σ_min(S)-driven
    magnitude drift that plagued unfixed *-Post variants.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, ns_steps=5, adapter_name=None,
                 lora_plus_multiplier=1.0,
                 log_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=1,
                 precond_method="eigh", higham_iters=10,
                 picard_iters=1):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.ns_steps = ns_steps
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every
        self.precond_method = precond_method
        self.higham_iters = higham_iters
        self.picard_iters = picard_iters
        if picard_iters < 1:
            raise ValueError("picard_iters must be >= 1")

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }

    def _polar_pipeline(self, u_A, u_B, SA_half_inv, SB_half_inv, lr):
        """One pass of the polar-product update + RMS-align.

        Returns (dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm,
        P_A, P_B). P_A, P_B are the polar (Newton-Schulz) outputs used
        for the H3 polar-sensitivity diagnostic.
        """
        X_B = u_B @ SA_half_inv
        P_B = _newton_schulz(X_B, nsteps=self.ns_steps)
        geo_B = P_B @ SA_half_inv

        X_A = SB_half_inv @ u_A
        P_A = _newton_schulz(X_A, nsteps=self.ns_steps)
        geo_A = SB_half_inv @ P_A

        uA_norm = u_A.norm()
        uB_norm = u_B.norm()
        gA_norm = geo_A.norm() + 1e-30
        gB_norm = geo_B.norm() + 1e-30
        dA = -lr * (uA_norm / gA_norm) * geo_A
        dB = -self.lora_plus_multiplier * lr * (uB_norm / gB_norm) * geo_B
        return dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm, P_A, P_B

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamPolarProductLoRA update.")
            state = self.pair_state[i]
            state['step'] += 1

            gA = A.grad.float()
            gB = B.grad.float()

            # Adam state on the RAW gradient.
            state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
            state['v_A'].mul_(self.beta2).addcmul_(gA, gA, value=1.0 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(gB, gB, value=1.0 - self.beta2)

            bc1 = 1.0 - self.beta1 ** state['step']
            bc2 = 1.0 - self.beta2 ** state['step']
            u_A = (state['m_A'] / bc1) / ((state['v_A'] / bc2).sqrt() + self.eps)
            u_B = (state['m_B'] / bc1) / ((state['v_B'] / bc2).sqrt() + self.eps)

            # Spectral square-root preconditioners. Refresh every K steps; reuse
            # cached value otherwise. K=1 ⇒ refresh every step (original behavior).
            # precond_method='higham' uses Newton-Schulz iteration instead of eigh
            # — ~10× faster at r=256 by avoiding the eigh kernel-launch storm.
            if (state['step'] - 1) % self.precond_refresh_every == 0:
                state['SA_half_inv'] = _spd_inv_half(
                    A.float() @ A.float().T, eps=self.delta,
                    method=self.precond_method, higham_iters=self.higham_iters,
                )
                state['SB_half_inv'] = _spd_inv_half(
                    B.float().T @ B.float(), eps=self.delta,
                    method=self.precond_method, higham_iters=self.higham_iters,
                )
            SA_half_inv = state['SA_half_inv']
            SB_half_inv = state['SB_half_inv']

            # Picard fixed-point iteration on the joint natural-gradient
            # equations. picard_iters=1 ⇒ block-diagonal (Adam direction fed
            # in as-is); picard_iters≥2 ⇒ feed cross-coupling correction
            # (1/η)·Bᵀ·dB_prev·A and (1/η)·B·dA_prev·Aᵀ into the polar pipeline.
            # Joint normal eqs (under spectral-product metric):
            #   S_B·ΔA + Bᵀ·ΔB·A = -η·u_A
            #   ΔB·S_A + B·ΔA·Aᵀ = -η·u_B
            # Block-diagonal drops the cross-terms; Picard restores them.
            A_f = A.float()
            B_f = B.float()
            dA_prev = torch.zeros_like(A_f)
            dB_prev = torch.zeros_like(B_f)
            for k in range(self.picard_iters):
                if k == 0:
                    u_A_eff = u_A
                    u_B_eff = u_B
                else:
                    u_A_eff = u_A + (B_f.T @ dB_prev @ A_f) / lr
                    u_B_eff = u_B + (B_f @ dA_prev @ A_f.T) / lr
                dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm, _, _ = \
                    self._polar_pipeline(u_A_eff, u_B_eff, SA_half_inv, SB_half_inv, lr)
                dA_prev = dA
                dB_prev = dB

            if self.log_diagnostics:
                step_count_local = state['step']
                is_probe_step = (step_count_local % self.diagnostics_every == 0)
                # cos(applied_step, plain-AdamW-direction). See AdamScaledLoRAPost
                # for sign-convention rationale.
                sa_min, sa_max = _gram_eig_extremes_from_factor(A)
                sb_min, sb_max = _gram_eig_extremes_from_factor(B)
                rec = {
                    "cos_A": _frob_cos(dA, -u_A),
                    "cos_B": _frob_cos(dB, -u_B),
                    "norm_dA": float(dA.detach().norm()),
                    "norm_dA_adamw_eq": float(lr * uA_norm),
                    "norm_dB": float(dB.detach().norm()),
                    "norm_dB_adamw_eq": float(lr * uB_norm),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                    "rms_scale_A": float(uA_norm / gA_norm),
                    "rms_scale_B": float(uB_norm / gB_norm),
                }

                # H1 — cross-term ratios γ_A, γ_B. Always cheap; uses the
                # APPLIED step (dA, dB) as the dB_prev/dA_prev surrogate for
                # the would-be iter-2 correction. Defined as the relative
                # magnitude of the perturbation that iter-2 would inject:
                #     γ_A = ‖Bᵀ dB A / lr‖_F / ‖u_A‖_F
                #     γ_B = ‖B  dA Aᵀ / lr‖_F / ‖u_B‖_F
                cross_A = (B_f.T @ dB.float() @ A_f) / lr
                cross_B = (B_f @ dA.float() @ A_f.T) / lr
                rec["gamma_A"] = float(cross_A.norm() / (u_A.norm() + 1e-30))
                rec["gamma_B"] = float(cross_B.norm() / (u_B.norm() + 1e-30))

                # H4 — numerical and stable rank of S_A, S_B (r×r, cheap).
                # nrank_τ = #{σᵢ > τ·σ_max}; stable rank = sum(σ²)/σ_max².
                # eigvalsh(SA) returns σ²(A) directly (S_A = A Aᵀ has eigs σᵢ²).
                try:
                    eigA = torch.linalg.eigvalsh(A_f @ A_f.T).clamp_min(0.0)
                    eigB = torch.linalg.eigvalsh(B_f.T @ B_f).clamp_min(0.0)
                    smax_A = float(eigA.max())
                    smax_B = float(eigB.max())
                    rec["nrank_A_1e3"] = int((eigA > 1e-3 * smax_A).sum())
                    rec["nrank_A_1e2"] = int((eigA > 1e-2 * smax_A).sum())
                    rec["nrank_B_1e3"] = int((eigB > 1e-3 * smax_B).sum())
                    rec["nrank_B_1e2"] = int((eigB > 1e-2 * smax_B).sum())
                    rec["stable_rank_A"] = float(eigA.sum() / (smax_A + 1e-30))
                    rec["stable_rank_B"] = float(eigB.sum() / (smax_B + 1e-30))
                except torch._C._LinAlgError:
                    for k in ("nrank_A_1e3", "nrank_A_1e2", "nrank_B_1e3",
                              "nrank_B_1e2", "stable_rank_A", "stable_rank_B"):
                        rec[k] = float("nan")

                # H2/H3 — Picard contraction + polar sensitivity.
                # Probe-step only (every diagnostics_every) since it costs 3
                # extra polar-pipeline calls per pair. Independent of the
                # applied step (self.picard_iters); always runs 3 iters from
                # zero so we can compare uncoupled and coupled symmetrically.
                if is_probe_step:
                    dA1, dB1, _, _, _, _, _, _, P_A1, P_B1 = self._polar_pipeline(
                        u_A, u_B, SA_half_inv, SB_half_inv, lr)
                    u_A_2 = u_A + (B_f.T @ dB1 @ A_f) / lr
                    u_B_2 = u_B + (B_f @ dA1 @ A_f.T) / lr
                    dA2, dB2, _, _, _, _, _, _, P_A2, P_B2 = self._polar_pipeline(
                        u_A_2, u_B_2, SA_half_inv, SB_half_inv, lr)
                    u_A_3 = u_A + (B_f.T @ dB2 @ A_f) / lr
                    u_B_3 = u_B + (B_f @ dA2 @ A_f.T) / lr
                    dA3, dB3, _, _, _, _, _, _, _, _ = self._polar_pipeline(
                        u_A_3, u_B_3, SA_half_inv, SB_half_inv, lr)
                    nA1 = float(dA1.norm()) + 1e-30
                    nA2 = float(dA2.norm()) + 1e-30
                    nB1 = float(dB1.norm()) + 1e-30
                    nB2 = float(dB2.norm()) + 1e-30
                    rec["picard_contract_A_12"] = float((dA2 - dA1).norm()) / nA1
                    rec["picard_contract_A_23"] = float((dA3 - dA2).norm()) / nA2
                    rec["picard_contract_B_12"] = float((dB2 - dB1).norm()) / nB1
                    rec["picard_contract_B_23"] = float((dB3 - dB2).norm()) / nB2
                    rec["polar_cos_A_12"] = _frob_cos(P_A1, P_A2)
                    rec["polar_cos_B_12"] = _frob_cos(P_B1, P_B2)

                diag_records.append(rec)

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamuonPolarProductLoRA(Optimizer):
    """AdaMuon-style polar-first composition of the spectral-product update.

    Contrast with AdamPolarProductLoRA (which does Adam → polar):
      • Adam(m̂, v̂) on raw (∇A, ∇B), then polar-product geometry on the
        Adam direction. v̂ is built from raw-gradient statistics.
    This optimizer does polar → variance-on-polar-output:
      • Plain momentum M on raw grads, optional sign-stabilization
        (AdaMuon Thm 1), polar-product geometry on M, then accumulate V
        elementwise on the polar output, normalize, RMS-align.

    Per pair (A, B):
        Mₐ ← β₁·Mₐ + ∇A,                M_B ← β₁·M_B + ∇B
        Sₐ = AAᵀ + δI,                   S_B = BᵀB + δI            (cached as fractional powers)
        signed Mₐ ← sign(Mₐ) if sign_stabilize else Mₐ              (analog for B)
        P_B = polar(signed M_B · S_A⁻¹ᐟ²),  D_B = P_B · S_A⁻¹ᐟ²
        P_A = polar(S_B⁻¹ᐟ² · signed Mₐ),   D_A = S_B⁻¹ᐟ² · P_A
        Vₐ ← β₂·Vₐ + (1−β₂)·Dₐ⊙Dₐ        (V_B analog)
        D̃ₐ = Dₐ / (√Vₐ + ε),             D̃_B analog
        γₐ = 0.2·√(rₐ·dₐ) / ‖D̃ₐ‖_F        (RMS-align to AdamW magnitude)
        ΔA = −lr·γₐ·D̃ₐ,                  ΔB = −m·lr·γ_B·D̃_B

    AdaMuon (arxiv 2507.11005) §3.1 argues variance estimation belongs on
    the polar output Oₜ rather than on raw G or momentum M, because the
    raw gradient carries ill-conditioned scaling that polar is designed
    to eliminate, making it unsuitable for stable variance tracking. This
    class tests whether that argument transfers to LoRA fine-tune scale
    when the polar operator is the spectral-product (S_A, S_B) form
    rather than vanilla NS.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, ns_steps=5, sign_stabilize=True,
                 adapter_name=None, lora_plus_multiplier=1.0,
                 log_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=1,
                 precond_method="eigh", higham_iters=10):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.ns_steps = ns_steps
        self.sign_stabilize = sign_stabilize
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every
        self.precond_method = precond_method
        self.higham_iters = higham_iters

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamuonPolarProductLoRA update.")
            state = self.pair_state[i]
            state['step'] += 1

            gA = A.grad.float()
            gB = B.grad.float()

            # Plain momentum on raw grads (no second moment yet).
            state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
            state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
            mA = state['m_A']
            mB = state['m_B']

            # AdaMuon Thm 1: sign(·) is the unique admissible elementwise transform
            # before the polar operator.
            sA = mA.sign() if self.sign_stabilize else mA
            sB = mB.sign() if self.sign_stabilize else mB

            # Spectral preconditioners (same as AdamPolarProductLoRA). Cached
            # and refreshed every K steps; K=1 reproduces the original per-step
            # behavior. precond_method='higham' swaps eigh for NS iteration.
            if (state['step'] - 1) % self.precond_refresh_every == 0:
                state['SA_half_inv'] = _spd_inv_half(
                    A.float() @ A.float().T, eps=self.delta,
                    method=self.precond_method, higham_iters=self.higham_iters,
                )
                state['SB_half_inv'] = _spd_inv_half(
                    B.float().T @ B.float(), eps=self.delta,
                    method=self.precond_method, higham_iters=self.higham_iters,
                )
            SA_half_inv = state['SA_half_inv']
            SB_half_inv = state['SB_half_inv']

            # Polar-product on (signed) momentum, NOT on Adam direction.
            X_B = sB @ SA_half_inv
            P_B = _newton_schulz(X_B, nsteps=self.ns_steps)
            D_B = P_B @ SA_half_inv

            X_A = SB_half_inv @ sA
            P_A = _newton_schulz(X_A, nsteps=self.ns_steps)
            D_A = SB_half_inv @ P_A

            # Variance accumulated on the polar output (AdaMuon §3.1).
            state['v_A'].mul_(self.beta2).addcmul_(D_A, D_A, value=1.0 - self.beta2)
            state['v_B'].mul_(self.beta2).addcmul_(D_B, D_B, value=1.0 - self.beta2)
            bc2 = 1.0 - self.beta2 ** state['step']
            tildaA = D_A / ((state['v_A'] / bc2).sqrt() + self.eps)
            tildaB = D_B / ((state['v_B'] / bc2).sqrt() + self.eps)

            # RMS-align: target ‖step‖_F = lr · 0.2·√(rows·cols) (AdaMuon paper §3.3
            # constant, matched to Adam's empirical RMS ≈ 0.2). Decouples step
            # magnitude from V's scale.
            rA, dA_in = A.shape
            dB_out, rB = B.shape
            target_A = 0.2 * (rA * dA_in) ** 0.5
            target_B = 0.2 * (dB_out * rB) ** 0.5
            tA_norm = tildaA.norm() + 1e-30
            tB_norm = tildaB.norm() + 1e-30
            gammaA = target_A / tA_norm
            gammaB = target_B / tB_norm
            dA = -lr * gammaA * tildaA
            dB = -self.lora_plus_multiplier * lr * gammaB * tildaB

            if self.log_diagnostics:
                # Reference plain-AdamW step direction for cos comparisons.
                # Cheap side-channel: m̂/(√v̂+ε) on raw grads, no extra state.
                # We don't maintain Adam first/second moments here, so use the
                # current-step gradient as a proxy reference (signed). This is
                # consistent across diagnostic-emitting optimizers as a
                # "would plain Adam go this way" signal.
                ref_A = -gA.sign()
                ref_B = -gB.sign()
                sa_min, sa_max = _gram_eig_extremes_from_factor(A)
                sb_min, sb_max = _gram_eig_extremes_from_factor(B)
                diag_records.append({
                    "cos_A": _frob_cos(dA, ref_A),
                    "cos_B": _frob_cos(dB, ref_B),
                    "norm_dA": float(dA.detach().norm()),
                    "norm_dA_target": float(lr * target_A),
                    "norm_dB": float(dB.detach().norm()),
                    "norm_dB_target": float(self.lora_plus_multiplier * lr * target_B),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "SA_min": sa_min, "SA_max": sa_max,
                    "SB_min": sb_min, "SB_max": sb_max,
                    "gammaA": float(gammaA),
                    "gammaB": float(gammaB),
                })

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdaMuonLoRA(Optimizer):
    """Faithful port of AdaMuon (arxiv 2507.11005, Algorithm 1) for LoRA factors.

    Per LoRA factor independently:
        Mₜ ← β·Mₜ₋₁ + Gₜ                            (plain SGD momentum on raw grad)
        Oₜ = NewtonSchulz(sign(Mₜ), T)              (sign-stabilized polar)
        Vₜ ← β·Vₜ₋₁ + (1−β)·Oₜ⊙Oₜ                   (variance on the polar output)
        Õₜ = Oₜ ⊘ (√Vₜ + ε)                         (elementwise normalize)
        γₜ = 0.2·√(rows·cols) / ‖Õₜ‖_F              (RMS-align to Adam magnitude)
        ΔW = −lr·γₜ·Õₜ

    Diff vs the older `MuonAdamLoRA` (which the project earlier called "naive
    NS→Adam" and reported as ~0.78 at 2k):
      1. **sign(Mₜ) before NS.** Old impl fed NS the raw gradient, producing
         step-to-step uncorrelated NS outputs. AdaMuon stabilizes by tracking
         momentum first and applying sign() before NS so NS sees a stationary
         input distribution (paper Theorem 1: sign is the unique admissible
         elementwise transform).
      2. **Only Vₜ on the polar output, no Mₜ on it.** Old impl ran full
         Adam(m, v) on the NS output — double smoothing.
      3. **RMS-align step magnitude.** Old impl applied lr directly to m̂/√v̂,
         leaving step magnitude unbounded.

    Vanilla-NS counterpart of `AdamuonPolarProductLoRA`. Comparing the two
    isolates "spectral-product geometry helps" from "AdaMuon stabilizers
    help" when both are layered on a polar-first composition.

    LoRA+ via `lr_b_multiplier` on B's lr.
    """
    def __init__(self, model, lr=3e-4, beta=0.95, eps=1e-8, ns_steps=5,
                 adapter_name=None, lr_b_multiplier=1.0,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta = beta
        self.eps = eps
        self.ns_steps = ns_steps
        self.lr_b_multiplier = lr_b_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.pair_state = {
            i: {
                "M_A": torch.zeros_like(A, dtype=torch.float32),
                "V_A": torch.zeros_like(A, dtype=torch.float32),
                "M_B": torch.zeros_like(B, dtype=torch.float32),
                "V_B": torch.zeros_like(B, dtype=torch.float32),
                "step": 0,
            }
            for i, (A, B) in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("AdaMuonLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            state["step"] += 1
            gA = A.grad.float()
            gB = B.grad.float()

            # (1) Plain momentum on raw gradient. Per AdaMuon paper Eq. (1)
            # the recursion is M ← β·M + G (no (1−β) factor on G).
            state["M_A"].mul_(self.beta).add_(gA)
            state["M_B"].mul_(self.beta).add_(gB)

            # (2) sign-stabilize, then NS. Per paper Eq. (7) and Theorem 1.
            sA = state["M_A"].sign()
            sB = state["M_B"].sign()
            O_A = _newton_schulz(sA, nsteps=self.ns_steps) if self.ns_steps > 0 else sA
            O_B = _newton_schulz(sB, nsteps=self.ns_steps) if self.ns_steps > 0 else sB

            # (3) Variance on the polar output. Per paper Eq. (5).
            state["V_A"].mul_(self.beta).addcmul_(O_A, O_A, value=1.0 - self.beta)
            state["V_B"].mul_(self.beta).addcmul_(O_B, O_B, value=1.0 - self.beta)

            # (4) Elementwise normalize. Per paper Eq. (6). No bias correction
            # — paper Appendix B explicitly omits it.
            tildaA = O_A / (state["V_A"].sqrt() + self.eps)
            tildaB = O_B / (state["V_B"].sqrt() + self.eps)

            # (5) RMS-align. Per paper Eq. (8): γ = 0.2·√(mn) / ‖Õ‖_F.
            rA, dA_in = A.shape
            dB_out, rB = B.shape
            target_A = 0.2 * (rA * dA_in) ** 0.5
            target_B = 0.2 * (dB_out * rB) ** 0.5
            tA_norm = tildaA.norm() + 1e-30
            tB_norm = tildaB.norm() + 1e-30
            gammaA = target_A / tA_norm
            gammaB = target_B / tB_norm
            dA = -lr * gammaA * tildaA
            dB = -self.lr_b_multiplier * lr * gammaB * tildaB

            if self.log_diagnostics:
                ref_A = -gA.sign()
                ref_B = -gB.sign()
                diag_records.append({
                    "cos_A": _frob_cos(dA, ref_A),
                    "cos_B": _frob_cos(dB, ref_B),
                    "norm_dA": float(dA.detach().norm()),
                    "norm_dA_target": float(lr * target_A),
                    "norm_dB": float(dB.detach().norm()),
                    "norm_dB_target": float(self.lr_b_multiplier * lr * target_B),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                    "gammaA": float(gammaA),
                    "gammaB": float(gammaB),
                })

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class MuonAdamLoRA(Optimizer):
    """
    Reverse of AdamMuonLoRA: NS first, then Adam (instead of Adam then NS).

    Per LoRA factor independently:
      ns_g = NS(g_raw)                       # orthogonalized gradient direction
      m  ← β₁ m  + (1−β₁) ns_g                # EMA on the NS direction
      v  ← β₂ v  + (1−β₂) ns_g²               # second-moment of NS direction
      Δθ = -lr · m̂/(√v̂ + ε)                  # Adam step on top of NS

    Mechanism contrast with AdamMuonLoRA:
      • AdamMuonLoRA: Adam tames raw-gradient magnitude variation, NS then
        spectrally caps the result. NS sees a per-element-normalized direction.
      • MuonAdamLoRA: NS first orthogonalizes the raw gradient (so all rows
        have unit-spec-norm magnitude), then Adam EMAs and per-element
        rescales that normalized direction over time. Adam's v on NS output
        is informative because NS rows have non-uniform per-element entries
        even though spectral norm = 1.

    LoRA+ via lr_b_multiplier (m on B's lr) preserved.
    """
    def __init__(self, model, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, ns_steps=5,
                 adapter_name=None, lr_b_multiplier=1.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        self.lr_b_multiplier = lr_b_multiplier
        self.pair_state = {
            i: {
                "m_A": torch.zeros_like(A, dtype=torch.float32),
                "v_A": torch.zeros_like(A, dtype=torch.float32),
                "m_B": torch.zeros_like(B, dtype=torch.float32),
                "v_B": torch.zeros_like(B, dtype=torch.float32),
                "step": 0,
            }
            for i, (A, B) in enumerate(pairs)
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("MuonAdamLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            state["step"] += 1
            t = state["step"]
            gA = A.grad.float()
            gB = B.grad.float()
            if self.ns_steps > 0:
                ns_A = _newton_schulz(gA, self.ns_steps)
                ns_B = _newton_schulz(gB, self.ns_steps)
            else:
                ns_A = gA
                ns_B = gB
            state["m_A"].mul_(self.beta1).add_(ns_A, alpha=1 - self.beta1)
            state["m_B"].mul_(self.beta1).add_(ns_B, alpha=1 - self.beta1)
            state["v_A"].mul_(self.beta2).addcmul_(ns_A, ns_A, value=1 - self.beta2)
            state["v_B"].mul_(self.beta2).addcmul_(ns_B, ns_B, value=1 - self.beta2)
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
            m_hat_A = state["m_A"] / bc1
            m_hat_B = state["m_B"] / bc1
            v_hat_A = state["v_A"] / bc2
            v_hat_B = state["v_B"] / bc2
            dA = -lr * m_hat_A / (v_hat_A.sqrt() + self.eps)
            dB = -self.lr_b_multiplier * lr * m_hat_B / (v_hat_B.sqrt() + self.eps)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class _HookLoRAOptimizer(Optimizer):
    """
    Base class for DiagScaledLoRA and KronGradLoRA.

    Registers forward and full-backward hooks on each LoRA module to capture:
      - cached_X[i]: layer inputs  → used to maintain D_V ∈ ℝ^{d_in} (EMA of diag(XᵀX/n))
      - cached_S[i]: output grads  → used to maintain D_U ∈ ℝ^{d_out} (EMA of diag(SᵀS/n))

    Paper convention (PSI-LoRA 2602.16456):
      D_V ← β₂ D_V + (1−β₂)(1/B) diag(XᵀX)   [Algorithm 3, line 4]
      D_U ← β₂ D_U + (1−β₂)(1/B) diag(SᵀS)   [Algorithm 3, line 5]

    Codebase PEFT convention: A ∈ ℝ^{r×d_in} (= paper Vᵀ), B ∈ ℝ^{d_out×r} (= paper U).
    """
    def __init__(self, model, params, param_groups, pairs, lora_modules,
                 ema_beta, delta, gamma, adapter_name):
        super().__init__(param_groups, {})
        self.pairs = pairs
        self.ema_beta = ema_beta
        self.delta = delta
        self.gamma = gamma
        self.cached_X = {}
        self.cached_S = {}
        self._hook_handles = []

        for i, (mod, (A, B)) in enumerate(zip(lora_modules, pairs)):
            d_in = A.shape[1]
            d_out = B.shape[0]

            def make_fwd(idx):
                def fwd_hook(module, inp, out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                    self.cached_X[idx] = x
                return fwd_hook

            def make_bwd(idx):
                def bwd_hook(module, grad_in, grad_out):
                    if grad_out[0] is not None:
                        s = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1])
                        self.cached_S[idx] = s
                return bwd_hook

            self._hook_handles.append(mod.register_forward_hook(make_fwd(i)))
            self._hook_handles.append(mod.register_full_backward_hook(make_bwd(i)))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _update_diag_stats(self, state, i):
        if i in self.cached_X:
            x = self.cached_X[i].float()
            state["D_V"].mul_(self.ema_beta).add_(x.pow(2).mean(0), alpha=1 - self.ema_beta)
        if i in self.cached_S:
            s = self.cached_S[i].float()
            state["D_U"].mul_(self.ema_beta).add_(s.pow(2).mean(0), alpha=1 - self.ema_beta)


def _collect_lora_modules(model, adapter_name=None):
    """Return (module, A_param, B_param) triples matching collect_lora_pairs order."""
    result = []
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            try:
                keys = [adapter_name] if adapter_name else list(mod.lora_A.keys())
                for k in keys:
                    if k in mod.lora_A and k in mod.lora_B:
                        A = mod.lora_A[k].weight
                        B = mod.lora_B[k].weight
                        result.append((mod, A, B))
                continue
            except Exception:
                if hasattr(mod.lora_A, "weight") and hasattr(mod.lora_B, "weight"):
                    result.append((mod, mod.lora_A.weight, mod.lora_B.weight))
    return result


class DiagScaledLoRA(_HookLoRAOptimizer):
    """
    DiagScaledLoRA: diagonal K-FAC scaling on independent (A, B) gradients.

    NOT the PSI-LoRA paper. The paper's Algorithm 3 has F-LoRSUM proximal subspace
    iteration, full-weight low-rank momentum, and U/V coupling — none of which are
    present here. This is a deliberate ablation isolating the effect of the
    layer-level D_V/D_U diagonal statistics alone.

    Update per pair (A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}):
        D_V ← ema_beta · D_V + (1−ema_beta) · diag(XᵀX/n)    [d_in]
        D_U ← ema_beta · D_U + (1−ema_beta) · diag(SᵀS/n)    [d_out]
        ΔA  = G_A · (D_V + δ)^{−γ}
        ΔB  = (D_U + δ)^{−γ} · G_B
        A ← A − lr · ΔA,  B ← B − lr · ΔB
    """
    def __init__(self, model, lr=3e-4, gamma=0.5, ema_beta=0.99,
                 delta=1e-5, adapter_name=None):
        triples = _collect_lora_modules(model, adapter_name)
        if not triples:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        lora_modules = [m for m, _, _ in triples]
        pairs = [(A, B) for _, A, B in triples]
        params = [p for A, B in pairs for p in (A, B)]
        self.pair_state = {
            i: {
                "D_V": torch.ones(A.shape[1], dtype=torch.float32, device=A.device),
                "D_U": torch.ones(B.shape[0], dtype=torch.float32, device=B.device),
            }
            for i, (A, B) in enumerate(pairs)
        }
        super().__init__(
            model=model,
            params=params,
            param_groups=[{"params": params, "lr": lr}],
            pairs=pairs,
            lora_modules=lora_modules,
            ema_beta=ema_beta,
            delta=delta,
            gamma=gamma,
            adapter_name=adapter_name,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("DiagScaledLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            self._update_diag_stats(state, i)
            gA = A.grad.float()
            gB = B.grad.float()
            sv = (state["D_V"] + self.delta).pow(-self.gamma)  # (d_in,)
            su = (state["D_U"] + self.delta).pow(-self.gamma)  # (d_out,)
            dA = gA * sv                        # (r, d_in) * (d_in,)
            dB = su.unsqueeze(1) * gB           # (d_out, 1) * (d_out, r)
            A.add_((-lr * dA).to(dtype=A.dtype, device=A.device))
            B.add_((-lr * dB).to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class KronGradLoRA(_HookLoRAOptimizer):
    """
    KronGradLoRA: DiagScaledLoRA + r×r Kronecker factors from gradient outer products.

    Custom variant — NOT from any paper. Extends DiagScaledLoRA with H_A ∈ ℝ^{r×r} and
    H_B ∈ ℝ^{r×r} from gradient outer products, capturing within-LoRA-subspace curvature.

    Update per pair:
        H_A ← ema_beta · H_A + (1−ema_beta) · G_A G_Aᵀ    [r×r]
        H_B ← ema_beta · H_B + (1−ema_beta) · G_Bᵀ G_B    [r×r]
        ΔA  = (H_A + δI)^{−γ} · G_A · (D_V + δ)^{−γ}
        ΔB  = (D_U + δ)^{−γ} · G_B · (H_B + δI)^{−γ}
    """
    def __init__(self, model, lr=3e-4, gamma=0.5, ema_beta=0.99,
                 delta=1e-5, adapter_name=None):
        triples = _collect_lora_modules(model, adapter_name)
        if not triples:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        lora_modules = [m for m, _, _ in triples]
        pairs = [(A, B) for _, A, B in triples]
        params = [p for A, B in pairs for p in (A, B)]
        self.pair_state = {
            i: {
                "D_V": torch.ones(A.shape[1], dtype=torch.float32, device=A.device),
                "D_U": torch.ones(B.shape[0], dtype=torch.float32, device=B.device),
                "H_A": torch.zeros(A.shape[0], A.shape[0], dtype=torch.float32, device=A.device),
                "H_B": torch.zeros(B.shape[1], B.shape[1], dtype=torch.float32, device=B.device),
            }
            for i, (A, B) in enumerate(pairs)
        }
        super().__init__(
            model=model,
            params=params,
            param_groups=[{"params": params, "lr": lr}],
            pairs=pairs,
            lora_modules=lora_modules,
            ema_beta=ema_beta,
            delta=delta,
            gamma=gamma,
            adapter_name=adapter_name,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("KronGradLoRA requires gradients on both A and B.")
            state = self.pair_state[i]
            self._update_diag_stats(state, i)
            gA = A.grad.float()
            gB = B.grad.float()
            # Update r×r Kronecker factors from gradient outer products
            state["H_A"].mul_(self.ema_beta).add_(gA @ gA.T, alpha=1 - self.ema_beta)
            state["H_B"].mul_(self.ema_beta).add_(gB.T @ gB, alpha=1 - self.ema_beta)
            # Diagonal K-FAC scaling
            sv = (state["D_V"] + self.delta).pow(-self.gamma)   # (d_in,)
            su = (state["D_U"] + self.delta).pow(-self.gamma)   # (d_out,)
            # r×r inverse fractional powers
            HA_inv = spd_frac_power_inv(state["H_A"], self.gamma, self.delta)  # (r, r)
            HB_inv = spd_frac_power_inv(state["H_B"], self.gamma, self.delta)  # (r, r)
            dA = HA_inv @ gA * sv                           # (r,r)@(r,d_in) * (d_in,)
            dB = su.unsqueeze(1) * gB @ HB_inv              # (d_out,1)*(d_out,r)@(r,r)
            A.add_((-lr * dA).to(dtype=A.dtype, device=A.device))
            B.add_((-lr * dB).to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


class PSILoRA(_HookLoRAOptimizer):
    """
    PSI-LoRA: Proximal Subspace Iteration LoRA, ported from Algorithm 3 of
    Almansoori et al. 2026 (arXiv 2602.16456). Reference impl:
    ~/PSI-LoRA/src/oplora/{utils.py:scaled_low_rank_sum, optimizer.py:OPLoraOptimizer}.

    Per LoRA pair (A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}; paper's V = Aᵀ, U = B):

      1. Hook captures X ∈ ℝ^{B×d_in} (forward input) and S ∈ ℝ^{B×d_out} (output grad).
      2. Update diagonal K-FAC stats (β₂=ema_beta):
         D_V ← β₂ D_V + (1−β₂)·(1/B)·diag(XᵀX)         shape (d_in,)
         D_U ← β₂ D_U + (1−β₂)·(1/B)·diag(SᵀS)         shape (d_out,)
      3. Compose dense step proposal as a SUM OF LOW-RANK FACTORS (no dense expansion):
         Ŵ = B@A − η·SᵀX − η·α₁·(M_B @ M_A)
         (Aᵀ = paper V, M_A and M_B are paper's r×n / m×r momentum factors.)
      4. F-LoRSUM (eq. 14): K alternating ALS iterations of the proximal projection
         under K-FAC metrics M_U = (D_U+δ)^γ, M_V = (D_V+δ)^γ to obtain (A_new, B_new).
      5. LoRSUM the momentum buffer in full weight space:
         (M_A, M_B) ← LoRSUM([(M_A, M_B), (X, Sᵀ)], (α₁, 1−α₁), K, ρ)
      6. Copy A_new, B_new back into the LoRA params.

    Hyperparameters:
      gamma=0.5, ema_beta=0.99, delta=1e-5  — diagonal K-FAC metric controls
      momentum=0.9                          — α₁ in paper Algorithm 3
      inner_iters=1                         — K, paper recommends K=1
      proximal_rho=0.01                     — ρ in eq. 9 / 14
    """
    def __init__(self, model, lr=3e-4, gamma=0.5, ema_beta=0.99, delta=1e-5,
                 momentum=0.9, inner_iters=1, proximal_rho=0.01,
                 momentum_rank=None, adapter_name=None):
        triples = _collect_lora_modules(model, adapter_name)
        if not triples:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        lora_modules = [m for m, _, _ in triples]
        pairs = [(A, B) for _, A, B in triples]
        params = [p for A, B in pairs for p in (A, B)]

        self.momentum = momentum
        self.inner_iters = inner_iters
        self.proximal_rho = proximal_rho

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            r = A.shape[0]
            r_m = momentum_rank if momentum_rank is not None else r
            self.pair_state[i] = {
                "D_V": torch.ones(A.shape[1], dtype=torch.float32, device=A.device),
                "D_U": torch.ones(B.shape[0], dtype=torch.float32, device=B.device),
                # Low-rank momentum factors: M_B @ M_A ≈ EMA of full grad.
                # M_A: (r_m, d_in), M_B: (d_out, r_m). Init to small noise on M_A, zero on M_B
                # (so their product is zero; matches paper's "M_t initialized to zero"
                # while keeping M_A non-degenerate so the LoRSUM ALS doesn't collapse).
                "M_A": torch.randn(r_m, A.shape[1], dtype=torch.float32, device=A.device),
                "M_B": torch.zeros(B.shape[0], r_m, dtype=torch.float32, device=B.device),
            }
        super().__init__(
            model=model,
            params=params,
            param_groups=[{"params": params, "lr": lr}],
            pairs=pairs,
            lora_modules=lora_modules,
            ema_beta=ema_beta,
            delta=delta,
            gamma=gamma,
            adapter_name=adapter_name,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        # CRITICAL: the proximal regularizer must be lr-scaled to match the
        # reference (~/PSI-LoRA/src/oplora/optimizer.py LR_LMBD=True at line 27;
        # every call to low_rank_sum/scaled_low_rank_sum passes
        # `lr * self.defaults["lmbd"]`). Without this, at small lr the prox term
        # dominates the gradient term and updates collapse — produces the
        # lr-insensitive pathology where all small η give identical loss.
        # Also clamp from below at 1e-5 (ref line 973: `lmbd = max(lmbd, 1e-5)`).
        rho = max(lr * self.proximal_rho, 1e-5)
        K = self.inner_iters
        alpha1 = self.momentum

        for i, (A, B) in enumerate(self.pairs):
            state = self.pair_state[i]
            self._update_diag_stats(state, i)

            X = self.cached_X.get(i)
            S = self.cached_S.get(i)
            if X is None or S is None:
                # Hook didn't fire (e.g. eval-mode); skip without erroring.
                continue
            X = X.float()                           # (B, d_in)
            S = S.float()                           # (B, d_out)

            A_curr = A.float()                      # (r, d_in)
            B_curr = B.float()                      # (d_out, r)
            M_A = state["M_A"]                      # (r_m, d_in)
            M_B = state["M_B"]                      # (d_out, r_m)

            # F-LoRSUM eq. 14 over factors:
            #   factors[0] = (A, B) prox center, coeff = 1
            #   factors[1] = (X, Sᵀ), coeff = -η
            #   factors[2] = (M_A, M_B), coeff = -η · α₁
            # Note: factor_in shape (k, d_in), factor_out shape (d_out, k).
            #   For (X, Sᵀ): X is (B_size, d_in)=factor_in shape (k=B_size, d_in) ✓;
            #                Sᵀ is (d_out, B_size)=factor_out shape (d_out, k=B_size) ✓.
            factors = [
                (A_curr, B_curr),
                (X, S.T),
                (M_A, M_B),
            ]
            # Convex combination form (matches reference, NOT paper Algorithm 3):
            # The reference uses [1.0, -lr·(1-α₁), -lr·α₁] for [weight, gradient,
            # momentum]. Paper Algorithm 3 box says [1.0, -lr, -lr·α₁] (sum form),
            # but the reference's ScaledOPLoraOptimizer.step() at
            # ~/PSI-LoRA/src/oplora/optimizer.py:1053 uses convex combination —
            # this is what produces the published numbers. With α₁=0 both forms
            # collapse to [1.0, -lr, 0] and agree.
            coeffs = [1.0, -lr * (1.0 - alpha1), -lr * alpha1]

            A_new, B_new = f_lorsum(
                factors=factors,
                coefficients=coeffs,
                D_U=state["D_U"], D_V=state["D_V"],
                num_iters=K, lmbd=rho,
                gamma=self.gamma, delta=self.delta,
            )

            # Update low-rank momentum only when α₁ > 0 (matches reference
            # ~/PSI-LoRA/src/oplora/optimizer.py:1069 guard "if beta1 > 0.0"):
            # M ← LoRSUM([(M, M), (X, Sᵀ)], (α₁, 1-α₁); K, ρ).
            if alpha1 > 0.0:
                M_A_new, M_B_new = lorsum(
                    factors=[(M_A, M_B), (X, S.T)],
                    coefficients=[alpha1, 1.0 - alpha1],
                    num_iters=K, lmbd=rho,
                )
                state["M_A"].copy_(M_A_new)
                state["M_B"].copy_(M_B_new)

            # Write new factors back to the LoRA params.
            A.copy_(A_new.to(dtype=A.dtype, device=A.device))
            B.copy_(B_new.to(dtype=B.dtype, device=B.device))
            if A.grad is not None:
                A.grad.zero_()
            if B.grad is not None:
                B.grad.zero_()


class GaLoreAdamW(Optimizer):
    """
    GaLore: Gradient Low-Rank Projection AdamW for dense target weights.
    Faithful port of the official implementation
    (https://github.com/jiaweizzhao/GaLore, galore_torch/{adamw,galore_projector}.py).

    For each weight W ∈ ℝ^{d_out×d_in} with gradient G, proj_type="std":
      - If d_out ≥ d_in (tall):  pick top-r right singular vectors V_r ∈ ℝ^{r×d_in}
            R = G @ V_rᵀ ∈ ℝ^{d_out×r}     (project on input side)
            Adam state shape: (d_out, r)
            ΔW = scale · R_hat @ V_r ∈ ℝ^{d_out×d_in}
      - If d_out < d_in (wide):   pick top-r left singular vectors U_r ∈ ℝ^{d_out×r}
            R = U_rᵀ @ G ∈ ℝ^{r×d_in}      (project on output side)
            Adam state shape: (r, d_in)
            ΔW = scale · U_r @ R_hat ∈ ℝ^{d_out×d_in}

    Important fidelity points (vs prior buggy version):
      • Adam moments PERSIST across projection updates. Official does NOT reset
        them when V_r/U_r refresh. Our prior version reset every 200 steps,
        destroying β₂=0.999 statistics that need ~1k steps to converge.
      • Projection axis chosen by shape (proj_type="std"), not always left.
      • Exact torch.linalg.svd for the projection, not randomized.

    Ref: GaLore (Zhao et al., 2024), arXiv 2403.03507.
    """
    def __init__(self, targets, rank, lr=3e-4, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, update_proj_gap=200, scale=1.0, proj_type="std"):
        if not targets:
            raise ValueError("GaLoreAdamW requires at least one dense target weight.")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if proj_type != "std":
            raise NotImplementedError(f"Only proj_type='std' is implemented (got '{proj_type}').")
        self.targets = list(targets)
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        super().__init__(
            [{"params": [t.weight for t in self.targets], "lr": lr,
              "betas": betas, "eps": eps, "weight_decay": weight_decay}],
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        # Per-weight GaLore state (separate from Adam state in self.state).
        # ortho_matrix is V_r (tall: r×d_in) or U_r (wide: d_out×r).
        # side ∈ {"right","left"} indicates how we project.
        # m, v are persistent and never reset — match official.
        self.galore_state = {
            t.name: {"ortho": None, "side": None, "m": None, "v": None, "step": 0}
            for t in self.targets
        }

    def _update_projection(self, G_f, r):
        """Compute fresh ortho matrix; returns (ortho_matrix, side)."""
        U, _, Vh = torch.linalg.svd(G_f, full_matrices=False)
        d_out, d_in = G_f.shape
        if d_out >= d_in:
            return Vh[:r, :], "right"  # (r, d_in)
        else:
            return U[:, :r], "left"    # (d_out, r)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for target in self.targets:
            W = target.weight
            G = W.grad
            if G is None:
                continue

            gs = self.galore_state[target.name]
            # Match official: project BEFORE incrementing step, so the iter
            # passed to projection is 0,1,2,... and refresh fires at iter
            # ∈ {0, gap, 2·gap, ...}. Bias correction below uses the
            # post-increment step ∈ {1, 2, ...}.
            iter_idx = gs["step"]

            G_f = G.float()
            d_out, d_in = G_f.shape
            r = min(self.rank, d_out, d_in)

            if gs["ortho"] is None or (iter_idx % self.update_proj_gap == 0):
                gs["ortho"], gs["side"] = self._update_projection(G_f, r)
                # NOTE: do NOT reset m/v here (matches official GaLore).

            gs["step"] += 1
            t = gs["step"]

            ortho = gs["ortho"]
            if gs["side"] == "right":
                # ortho: (r, d_in); R = G @ orthoᵀ ∈ (d_out, r)
                R = G_f @ ortho.T
            else:
                # ortho: (d_out, r); R = orthoᵀ @ G ∈ (r, d_in)
                R = ortho.T @ G_f

            # Lazy moment init now that we know R's shape.
            if gs["m"] is None:
                gs["m"] = torch.zeros_like(R)
                gs["v"] = torch.zeros_like(R)

            # Adam in projected space; moments persist across projection updates.
            gs["m"].mul_(beta1).add_(R, alpha=1 - beta1)
            gs["v"].mul_(beta2).addcmul_(R, R, value=1 - beta2)
            denom = gs["v"].sqrt().add_(eps)
            bc1 = 1 - beta1 ** t
            bc2 = 1 - beta2 ** t
            step_size = lr * (bc2 ** 0.5) / bc1
            R_hat = gs["m"] / denom  # element-wise; matches official norm_grad

            # Project back and apply with GaLore scale.
            if gs["side"] == "right":
                norm_grad = R_hat @ ortho                # (d_out, d_in)
            else:
                norm_grad = ortho @ R_hat                # (d_out, d_in)
            norm_grad = norm_grad * self.scale

            W.add_((-step_size * norm_grad).to(dtype=W.dtype, device=W.device))
            if weight_decay > 0.0:
                W.add_(W, alpha=-lr * weight_decay)
            W.grad = None

        return loss


def build_optimizer(
    model,
    optimizer_type: str,
    lr: float,
    weight_decay: float = 0.0,
    scaled_metric: bool = False,
    lora_plus_multiplier: float = 1.0,
    targets=None,
    svd_rank: int | None = None,
    svd_niter: int = 4,
    precond_gamma: float = 0.5,
    precond_ema_beta: float = 0.99,
    precond_delta: float = 1e-5,
    psi_inner_iters: int = 1,
    psi_momentum: float = 0.9,
    psi_rho: float = 0.01,
    psi_momentum_rank: int | None = None,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
    muon_ns_steps: int = 5,
    muon_alpha: int = 16,
    muon_rank: int = 16,
    log_optim_diagnostics: bool = False,
    optim_diagnostics_every: int = 20,
    precond_refresh_every: int = 1,
    precond_method: str = "eigh",
    higham_iters: int = 10,
):
    if optimizer_type not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"Unsupported optimizer_type '{optimizer_type}'. "
            f"Expected one of: {', '.join(sorted(OPTIMIZER_CHOICES))}."
        )

    if optimizer_type == "galore-adamw":
        if targets is None:
            raise ValueError("galore-adamw requires dense target weights (use --training_mode galore).")
        if svd_rank is None:
            raise ValueError("galore-adamw requires svd_rank.")
        return GaLoreAdamW(
            targets,
            rank=svd_rank,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=weight_decay,
            update_proj_gap=galore_update_proj_gap,
            scale=galore_scale,
        )

    if optimizer_type in {"svd-step-adamw", "svd-cumulative-adamw"}:
        if targets is None:
            raise ValueError(f"{optimizer_type} requires dense target weights.")
        if svd_rank is None:
            raise ValueError(f"{optimizer_type} requires svd_rank.")
        cls = SVDStepAdamW if optimizer_type == "svd-step-adamw" else SVDCumulativeAdamW
        return cls(
            targets,
            rank=svd_rank,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            svd_niter=svd_niter,
        )

    if optimizer_type == "adamw":
        return LoRAPlusAdamW(
            model,
            lr=lr,
            lora_plus_multiplier=lora_plus_multiplier,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    if optimizer_type == "lin-lora":
        return LinLoRA(model, lr=lr, delta=1e-6)
    if optimizer_type == "scaled-lora":
        return ScaledLoRA(model, lr=lr, delta=1e-6)
    if optimizer_type == "adam-scaled-lora":
        return AdamScaledLoRA(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
        )
    if optimizer_type == "adam-lin-lora":
        return AdamLinLoRA(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
        )
    if optimizer_type == "adam-scaled-lora-post":
        return AdamScaledLoRAPost(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-lin-lora-post":
        return AdamLinLoRAPost(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-scaled-lora-matrix":
        return AdamScaledLoRAMatrix(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
        )
    if optimizer_type == "adam-lin-lora-matrix":
        return AdamLinLoRAMatrix(
            model,
            lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "polar-product-lora":
        return PolarProductLoRA(
            model, lr=lr, delta=1e-6, ns_steps=muon_ns_steps,
        )
    if optimizer_type == "adam-polar-product-lora":
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=1,
        )
    if optimizer_type == "adam-polar-product-lora-coupled":
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=2,
        )
    if optimizer_type == "adamuon-polar-product-lora":
        return AdamuonPolarProductLoRA(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            sign_stabilize=True,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
        )
    if optimizer_type == "adamuon-lora":
        return AdaMuonLoRA(
            model, lr=lr,
            beta=0.95,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-lora":
        return MuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "product-muon-lora":
        return ProductMuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            alpha=muon_alpha, rank=muon_rank,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "adam-muon-lora":
        return AdamMuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "adam-product-muon-lora":
        return AdamProductMuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            alpha=muon_alpha, rank=muon_rank,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "muon-adam-lora":
        return MuonAdamLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "diag-scaled-lora":
        return DiagScaledLoRA(
            model, lr=lr,
            gamma=precond_gamma,
            ema_beta=precond_ema_beta,
            delta=precond_delta,
        )
    if optimizer_type == "kron-grad-lora":
        return KronGradLoRA(
            model, lr=lr,
            gamma=precond_gamma,
            ema_beta=precond_ema_beta,
            delta=precond_delta,
        )
    if optimizer_type == "psi-lora":
        return PSILoRA(
            model, lr=lr,
            gamma=precond_gamma,
            ema_beta=precond_ema_beta,
            delta=precond_delta,
            momentum=psi_momentum,
            inner_iters=psi_inner_iters,
            proximal_rho=psi_rho,
            momentum_rank=psi_momentum_rank,
        )
    if optimizer_type == "sgd":
        params = [p for p in model.parameters() if p.requires_grad]
        return SGD(params, lr=lr, momentum=0.0)
    if optimizer_type == "sgd-m":
        params = [p for p in model.parameters() if p.requires_grad]
        return SGD(params, lr=lr, momentum=0.9)

    raise ValueError(f"Optimizer type '{optimizer_type}' is not implemented.")
