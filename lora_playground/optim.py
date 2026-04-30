import torch
from torch.optim import AdamW, Optimizer, SGD

from .utils import collect_lora_pairs, solve_spd, solve_sylvester, spdify, truncated_svd

OPTIMIZER_CHOICES = {
    "adamw",
    "lin-lora",
    "scaled-lora",
    "adam-scaled-lora",
    "adam-lin-lora",
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
):
    if optimizer_type not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"Unsupported optimizer_type '{optimizer_type}'. "
            f"Expected one of: {', '.join(sorted(OPTIMIZER_CHOICES))}."
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
    if optimizer_type == "sgd":
        params = [p for p in model.parameters() if p.requires_grad]
        return SGD(params, lr=lr, momentum=0.0)
    if optimizer_type == "sgd-m":
        params = [p for p in model.parameters() if p.requires_grad]
        return SGD(params, lr=lr, momentum=0.9)

    raise ValueError(f"Optimizer type '{optimizer_type}' is not implemented.")
