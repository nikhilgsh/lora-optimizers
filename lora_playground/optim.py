import torch
from torch.optim import AdamW, Optimizer, SGD

from .utils import (
    collect_lora_pairs,
    f_lorsum,
    lorsum,
    solve_spd,
    solve_sylvester,
    spd_frac_power_inv,
    spdify,
    truncated_svd,
)

OPTIMIZER_CHOICES = {
    "adamw",
    "lin-lora",
    "scaled-lora",
    "adam-scaled-lora",
    "adam-lin-lora",
    "muon-lora",
    "product-muon-lora",
    "adam-muon-lora",
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
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None, scaled_metric=False, lora_plus_multiplier=1.0):
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
        
        # Initialize state: first and second moments for each (A, B) pair
        # Use pair_state to avoid conflicts with PyTorch's Optimizer.state
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

        for i, ((A, B), gamma) in enumerate(zip(self.pairs, self.gammas)):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamLinLoRA update.")

            state = self.pair_state[i]
            state['step'] += 1
            
            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Regularized Gram matrices: S_A = A A^T + δ I, S_B = B^T B + δ I
            SB = spdify(B.T @ B, self.delta)       # S_B ∈ ℝ^{r×r}
            SA = spdify(A @ A.T, self.delta)       # S_A ∈ ℝ^{r×r}
            RHS = -gamma * (gA @ A.T).float()              # RHS = -(∇_A A^T) ∈ ℝ^{r×r} [no lr]

            # Solve Sylvester equation: S_B K + K S_A = RHS for K ∈ ℝ^{r×r}
            K = solve_sylvester(SB, (gamma ** 2) * SA, RHS)       # K ∈ ℝ^{r×r}

            # Compute preconditioned gradients (without lr factor)
            # precond_B = (∇_B + B K) S_A^{-1}
            termB = (gB + (1. / gamma) * B @ K.to(dtype=B.dtype)).float()   # ℝ^{d_out×r}
            precond_B = solve_spd(SA, termB.T).T             # ∈ ℝ^{d_out×r}

            # precond_A = S_B^{-1} (∇_A + K A)
            termA = (gA + gamma * K.to(dtype=A.dtype) @ A).float()   # ℝ^{r×d_in}
            precond_A = solve_spd(SB, termA)                 # ∈ ℝ^{r×d_in}

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

            # Apply the update, cast back to parameter dtype/device
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()


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
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        
        # Initialize state: first and second moments for each (A, B) pair
        # Use pair_state instead of state to avoid conflicts with PyTorch's Optimizer.state
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

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamScaledLoRA update.")

            state = self.pair_state[i]
            state['step'] += 1
            
            gA = A.grad          # ∇_A ∈ ℝ^{r×d_in}
            gB = B.grad          # ∇_B ∈ ℝ^{d_out×r}

            # Compute preconditioning matrices: S_A = A A^T + δ I, S_B = B^T B + δ I
            SB = spdify(B.T @ B, self.delta)       # S_B ∈ ℝ^{r×r}
            SA = spdify(A @ A.T, self.delta)       # S_A ∈ ℝ^{r×r}

            # Compute preconditioned gradients (not scaled by lr yet)
            precond_B = solve_spd(SA, gB.T).T      # ∇_B S_A^{-1} ∈ ℝ^{d_out×r}
            precond_A = solve_spd(SB, gA)          # S_B^{-1} ∇_A ∈ ℝ^{r×d_in}

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

            # Apply the update, cast back to parameter dtype/device
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
        A   ← A − lr · NS(m_A)
        B   ← B − lr · NS(m_B)

    NS(X) gives orthonormal rows (or cols for tall X), preventing rank collapse
    where a few singular values dominate and effective rank drops below r.
    """
    def __init__(self, model, lr=3e-4, beta=0.95, ns_steps=5, adapter_name=None):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.beta = beta
        self.ns_steps = ns_steps
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
            dA = _newton_schulz(state["m_A"], self.ns_steps)
            dB = _newton_schulz(state["m_B"], self.ns_steps)
            A.add_((-lr * dA).to(dtype=A.dtype, device=A.device))
            B.add_((-lr * dB).to(dtype=B.dtype, device=B.device))
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
                "M_A": torch.randn(r_m, A.shape[1], dtype=torch.float32, device=A.device).mul(0.01),
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
        rho = self.proximal_rho
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
            coeffs = [1.0, -lr, -lr * alpha1]

            A_new, B_new = f_lorsum(
                factors=factors,
                coefficients=coeffs,
                D_U=state["D_U"], D_V=state["D_V"],
                num_iters=K, lmbd=rho,
                gamma=self.gamma, delta=self.delta,
            )

            # Update low-rank momentum: M ← LoRSUM([(M, M), (X, Sᵀ)], (α₁, 1-α₁); K, ρ)
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
            gs["step"] += 1
            t = gs["step"]

            G_f = G.float()
            d_out, d_in = G_f.shape
            r = min(self.rank, d_out, d_in)

            if gs["ortho"] is None or (t % self.update_proj_gap == 0):
                gs["ortho"], gs["side"] = self._update_projection(G_f, r)
                # NOTE: do NOT reset m/v here (matches official GaLore).

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
        )
    if optimizer_type == "muon-lora":
        return MuonLoRA(model, lr=lr)
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
