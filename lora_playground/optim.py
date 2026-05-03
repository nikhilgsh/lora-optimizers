import inspect
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


# Init parameters that are construction inputs, not algorithmic state — the
# model itself, parameter collections, adapter selectors. Excluded from
# optimizer_config_dict because they don't distinguish algorithmic behavior
# across runs. Add a name here only if it's a constructor wiring input, not
# a hyperparameter.
_CONFIG_DICT_SKIP = frozenset({
    "self", "model", "targets", "pairs", "lora_modules",
    "params", "param_groups", "adapter_name",
})

# Aliases for params whose attribute is stored under a different name (e.g.
# `betas` tuple stored as separate `beta1, beta2` attrs). Each entry maps
# the __init__ param name to a callable extracting the value from `self`.
def _extract_lr(opt):
    """torch.optim.Optimizer stores lr inside param_groups, not as self.lr."""
    groups = getattr(opt, "param_groups", None)
    if groups:
        return groups[0].get("lr")
    return getattr(opt, "lr", None)


_CONFIG_DICT_ALIASES = {
    "betas": lambda opt: (
        getattr(opt, "beta1", None),
        getattr(opt, "beta2", None),
    ),
    "lr": _extract_lr,
}


def _is_json_safe(v) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_json_safe(x) for x in v)
    return False


def optimizer_config_dict(opt) -> dict:
    """Resolved-hyperparameter snapshot of `opt` via __init__ introspection.

    Walks `inspect.signature(type(opt).__init__)`; for each parameter not in
    `_CONFIG_DICT_SKIP`, looks up the matching attribute on `opt` (with
    `_CONFIG_DICT_ALIASES` overriding for split-storage cases). Returns a flat
    dict of JSON-safe values plus `_optim_class` for traceability.

    Convention enforcement: every optimizer in OPTIMIZER_CHOICES must store
    each non-skipped __init__ param as an attribute of the same name (or via
    an alias). The unit test in `tests/test_optimizer_config_dict.py` walks
    OPTIMIZER_CHOICES and fails if any param is unrecorded — that's how we
    keep this from going stale as new optimizers ship.
    """
    out = {"_optim_class": type(opt).__name__}
    sig = inspect.signature(type(opt).__init__)
    for name, param in sig.parameters.items():
        if name in _CONFIG_DICT_SKIP:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in _CONFIG_DICT_ALIASES:
            value = _CONFIG_DICT_ALIASES[name](opt)
        elif hasattr(opt, name):
            value = getattr(opt, name)
        else:
            # Fallback for torch/HF optimizer wrappers (LoRAPlusAdamW, the HF
            # Adafactor wrapper, plain SGD) that store kwargs in param_groups
            # rather than as instance attrs. param_groups[0] holds the default
            # values for that group; defaults dict on the base Optimizer also
            # mirrors them.
            groups = getattr(opt, "param_groups", None)
            defaults = getattr(opt, "defaults", None)
            if groups and name in groups[0]:
                value = groups[0][name]
            elif defaults and name in defaults:
                value = defaults[name]
            else:
                # Surface as missing rather than silently dropping. The CI test
                # asserts no value lands here for any optimizer in
                # OPTIMIZER_CHOICES.
                value = "<unrecorded>"
        if _is_json_safe(value) or value == "<unrecorded>":
            out[name] = value
    return out


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


def _svd_col_basis(X, rel_tol=1e-6):
    try:
        U, S, _ = torch.linalg.svd(X.to(torch.float32), full_matrices=False)
    except torch._C._LinAlgError:
        return None
    if S.numel() == 0:
        return U[:, :0]
    threshold = rel_tol * float(S[0])
    keep = S > threshold
    return U[:, keep]


def _finite_step_product_diagnostics(A_f, B_f, dA, dB, eps=1e-30):
    """Rank-r diagnostics for the finite LoRA product dB @ dA.

    All norms are computed from skinny factors and r x r cores; this avoids
    materializing a dense d_out x d_in update.
    """
    A_f = A_f.detach().to(torch.float32)
    B_f = B_f.detach().to(torch.float32)
    dA = dA.detach().to(torch.float32)
    dB = dB.detach().to(torch.float32)

    dA_gram = dA @ dA.T
    dB_gram = dB.T @ dB
    A_gram = A_f @ A_f.T
    B_gram = B_f.T @ B_f

    second_sq = torch.sum(dB_gram * dA_gram).clamp_min(0.0)
    second_norm = second_sq.sqrt()

    bdA_sq = torch.sum(B_gram * dA_gram).clamp_min(0.0)
    dBA_sq = torch.sum(dB_gram * A_gram).clamp_min(0.0)
    cross = torch.trace((B_f.T @ dB) @ (A_f @ dA.T))
    tangent_sq = (bdA_sq + dBA_sq + 2.0 * cross).clamp_min(0.0)
    tangent_norm = tangent_sq.sqrt()

    out = {
        "finite_step_norm": float(second_norm),
        "tangent_step_norm": float(tangent_norm),
        "finite_step_to_tangent": float(second_norm / (tangent_norm + eps)),
    }

    try:
        _, rB = torch.linalg.qr(dB, mode="reduced")
        _, rA = torch.linalg.qr(dA.T, mode="reduced")
        sv = torch.linalg.svdvals(rB @ rA.T)
        if sv.numel() == 0 or float(sv[0]) <= eps:
            out["finite_step_spectral_norm"] = 0.0
            out["finite_step_stable_rank"] = 0.0
        else:
            out["finite_step_spectral_norm"] = float(sv[0])
            out["finite_step_stable_rank"] = float((sv * sv).sum() / (sv[0] * sv[0] + eps))
    except torch._C._LinAlgError:
        out["finite_step_spectral_norm"] = float("nan")
        out["finite_step_stable_rank"] = float("nan")

    qB = _svd_col_basis(B_f)
    qA = _svd_col_basis(A_f.T)
    if qB is None or qA is None:
        out["finite_step_new_new_frac"] = float("nan")
    else:
        if qB.numel() == 0:
            dB_perp_gram = dB_gram
        else:
            proj_dB = qB.T @ dB
            dB_perp_gram = dB_gram - proj_dB.T @ proj_dB
        if qA.numel() == 0:
            dA_perp_gram = dA_gram
        else:
            proj_dA = dA @ qA
            dA_perp_gram = dA_gram - proj_dA @ proj_dA.T
        new_new_sq = torch.sum(dB_perp_gram * dA_perp_gram).clamp_min(0.0)
        new_new_frac = (new_new_sq / (second_sq + eps)).clamp(0.0, 1.0)
        out["finite_step_new_new_frac"] = float(new_new_frac)
    return out


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
    "adafactor",
    "lin-lora",
    "scaled-lora",
    "adam-scaled-lora",
    "adam-lin-lora",
    "adam-lin-core-lora",
    "adam-scaled-lora-post",
    "adam-lin-lora-post",
    "adam-scaled-lora-matrix",
    "adam-lin-lora-matrix",
    "polar-product-lora",
    "adam-polar-product-lora",
    "adam-polar-product-lora-coupled",
    "adam-polar-product-lora-coupled-endrms",
    "adam-soap-polar-product-lora",
    "adafactor-polar-product-lora",
    "sign-momentum-polar-product-lora",
    "adam-clip-product-lora",
    "adam-clip-product-lora-coupled",
    "adam-clip-product-lora-coupled-endrms",
    "adam-polar-product-lora-gauge",
    "adam-polar-product-lora-gauge-coupled",
    "adam-polar-product-lora-clip-gauge",
    "adam-polar-product-lora-clip-gauge-coupled",
    "polar-coupled-core-lora",
    "polar-coupled-core-imbalance-scalar-lora",
    "polar-coupled-core-imbalance-lora",
    "polar-coupled-core-imbalance-restore-lora",
    "polar-coupled-core-balanced-scalar-lora",
    "polar-coupled-core-state-rebalanced-lora",
    "polar-coupled-core-sign-lora",
    "polar-coupled-core-sign-rebalanced-lora",
    "polar-coupled-core-factor-adam-lora",
    "polar-coupled-core-factor-adam-rebalanced-lora",
    "muon-coupled-core-lora",
    "muon-coupled-core-imbalance-scalar-lora",
    "muon-coupled-core-imbalance-lora",
    "muon-coupled-core-balanced-scalar-lora",
    "muon-coupled-core-state-rebalanced-lora",
    "muon-coupled-core-sign-lora",
    "muon-coupled-core-sign-rebalanced-lora",
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

            # δ is a numerical regularizer for near-singular grams, not the
            # minimizer of a damped factor-space surrogate — see
            # docs/notes/polar_product/theory.md (Case 1, "Damped surrogate").
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

    NOTE — implementation does not match the principled "Adam version of LinLoRA".
    The Sylvester step here runs on raw gradients (compatibility holds, RHS is
    well-defined), but downstream Adam is applied independently to (v_A, v_B) as
    factor tensors. Independent factor-Adam is gauge-dependent — the LoRA
    variational problem is invariant under (ΔA, ΔB) → (ΔA + S A, ΔB - B S), but
    independent (m_A, v_A, m_B, v_B) state is not. The principled "Adam of
    LinLoRA" maintains momentum/RMS in core/tangent space (Frobenius restriction
    of variant 2 in docs/notes/polar_product/theory.md §6) and solves the
    Sylvester on the EMA core covector. This implementation is kept as an
    empirical baseline; reinterpret leaderboard standing accordingly.
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
        self.scaled_metric = scaled_metric
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


class AdamLinCoreLoRA(Optimizer):
    """Cross-check on core-space momentum.

    Same Sylvester-based solver as AdamLinLoRA, but EMA-Adam lives in the
    core/Sylvester-quotient space (the r×r rotation matrix K) instead of on
    the factor preconditioned gradients. Tests the hypothesis that core-space
    momentum is generally broken (independent of our coupled-polar solver),
    by comparing apples-to-apples against AdamLinLoRA which has the same
    base solver but factor-space Adam.

    Per step:
      1. Compute core covector M = -gA · A^T (r×r), the Sylvester RHS.
         By gradient compatibility this equals B^T · gB up to sign/scale.
      2. EMA-Adam on M: m_M, v_M (r×r).
      3. Solve Sylvester with M_hat (the EMA-Adam'd core covector) as RHS,
         giving K_hat.
      4. Lift to (dA, dB) using K_hat and raw factor gradients (the
         outside-rotation terms still use gA, gB).

    Eval below adam-lin-lora at matched lr → core-space momentum is degraded
    in this solver too → general failure mode (confirms variant 2 diagnosis).
    Eval at-or-above adam-lin-lora → core-space momentum is fine here, our
    coupled-polar solver has a specific issue.

    EMPIRICAL FINDING (do not ship): a 5-step smoke at OLMo-2-1B r=4 lr=1e-3
    diverges — eval jumps 2.58 → 12.67 at step 2, Cholesky fails at step 3.
    The /sqrt(v_M) normalization on a small r×r matrix degenerates to
    ≈ 3·sign(M) (homogeneous coordinate scales mean Adam's per-coord
    rescaling doesn't help, just inflates magnitude). This independently
    confirms variant 2's align_mom diagnosis: core-space Adam-style momentum
    is structurally broken because the core object lacks the heterogeneous
    coordinate scales Adam exists to normalize. Kept as documented evidence
    against core-space momentum; do not include in production sweeps.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
                 adapter_name=None, scaled_metric=False, lora_plus_multiplier=1.0,
                 bias_correction=False,
                 log_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.scaled_metric = scaled_metric
        self.bias_correction = bias_correction
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every

        self.pair_state = {}
        self.gammas = []
        for i, (A, B) in enumerate(pairs):
            r = A.shape[0]
            self.pair_state[i] = {
                "m_M": torch.zeros((r, r), dtype=torch.float32, device=A.device),
                "v_M": torch.zeros((r, r), dtype=torch.float32, device=A.device),
                "step": 0,
            }
            r, d_in = A.shape
            if scaled_metric:
                self.gammas.append((d_in / r) ** 0.5)
            else:
                self.gammas.append(1.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, ((A, B), gamma) in enumerate(zip(self.pairs, self.gammas)):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamLinCoreLoRA update.")
            state = self.pair_state[i]
            state["step"] += 1

            gA = A.grad
            gB = B.grad

            need_refresh = (state["step"] - 1) % self.precond_refresh_every == 0
            if need_refresh:
                SB = spdify(B.T @ B, self.delta)
                SA = spdify(A @ A.T, self.delta)
                state["evalA"], state["QA"] = torch.linalg.eigh(SA)
                state["evalB"], state["QB"] = torch.linalg.eigh(SB)
                state["LA"] = torch.linalg.cholesky(SA)
                state["LB"] = torch.linalg.cholesky(SB)
            evalA, QA = state["evalA"], state["QA"]
            evalB, QB = state["evalB"], state["QB"]
            LA, LB = state["LA"], state["LB"]

            # Core covector M = -γ (gA · A^T) (r×r). This is the Sylvester
            # RHS — the projected dense gradient onto the active rank-r
            # tangent expressed in the (B, A) basis. By compatibility,
            # gA · A^T = B^T · gB, so M is the unique core-space cost gradient.
            M = -gamma * (gA @ A.T).float()

            # EMA-Adam on M (core space).
            state["m_M"].mul_(self.beta1).add_(M, alpha=1 - self.beta1)
            state["v_M"].mul_(self.beta2).addcmul_(M, M, value=1 - self.beta2)
            if self.bias_correction:
                bc1 = 1 - self.beta1 ** state["step"]
                bc2 = 1 - self.beta2 ** state["step"]
                m_hat = state["m_M"] / bc1
                v_hat = state["v_M"] / bc2
            else:
                m_hat = state["m_M"]
                v_hat = state["v_M"]
            M_hat = m_hat / (v_hat.sqrt() + self.eps)

            # Solve Sylvester with EMA-Adam'd core covector as RHS.
            T_syl = QB.T @ M_hat @ QA
            denom = evalB[:, None] + (gamma ** 2) * evalA[None, :]
            K_hat = QB @ (T_syl / denom) @ QA.T

            # Lift to (dA, dB) using K_hat (core-Adam'd) in place of raw K.
            # Raw factor gradients still appear (the "outside the rotation" terms);
            # only the K-rotation gets EMA-Adam.
            termB = (gB + (1. / gamma) * B @ K_hat.to(dtype=B.dtype)).float()
            precond_B = torch.cholesky_solve(termB.T, LA).T
            termA = (gA + gamma * K_hat.to(dtype=A.dtype) @ A).float()
            precond_A = torch.cholesky_solve(termA, LB)

            dA = -lr * precond_A
            dB = -self.lora_plus_multiplier * lr * precond_B

            if self.log_diagnostics:
                diag_records.append({
                    "norm_M": float(M.norm()),
                    "norm_M_hat": float(M_hat.norm()),
                    "cos_M_Mhat": _frob_cos(M, M_hat),
                    "norm_K_hat": float(K_hat.norm()),
                    "norm_dA": float(dA.norm()),
                    "norm_dB": float(dB.norm()),
                    "norm_gA": float(gA.detach().to(torch.float32).norm()),
                    "norm_gB": float(gB.detach().to(torch.float32).norm()),
                    "norm_A": float(A.detach().to(torch.float32).norm()),
                    "norm_B": float(B.detach().to(torch.float32).norm()),
                })

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
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

    NOTE — implementation realizes its intent incoherently. The Sylvester RHS
    `-γ·lr·(u_A A^T)` is one of two algebraically-equal forms under gradient
    compatibility: ∇_A A^T = B^T ∇_B for raw gradients, so either RHS gives
    the same K. Once Adam runs independently on each factor, u_A A^T ≠ B^T u_B
    in general, and the Sylvester silently picks one side of an inconsistent
    pair of normal equations. The principled "Adam version" maintains the core
    covector Ĥ in tangent space, EMAs Ĥ across steps with basis transport, and
    solves the Sylvester on the EMA — see variant 2 (Frobenius restriction) in
    docs/notes/polar_product/theory.md §6. This implementation is kept as
    an empirical baseline; results should be read as "factor-Adam-then-
    Sylvester-on-one-side" rather than "Adam-preconditioned LinLoRA".
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
        self.scaled_metric = scaled_metric
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
        self.scaled_metric = scaled_metric
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
        self.lora_plus_multiplier = lora_plus_multiplier
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


def _build_lora_adafactor(model, lr, lora_plus_multiplier, weight_decay):
    """HuggingFace transformers.optimization.Adafactor restricted to LoRA
    params (mirrors LoRAPlusAdamW layout). Pure baseline — no polar pipeline.
    Compares head-to-head with AdamW on plain LoRA.

    HuggingFace's Adafactor is the de-facto reference impl (T5 default). We
    configure it to consume an explicit ``lr`` (``relative_step=False,
    scale_parameter=False, warmup_init=False``) so the sweep grid mirrors
    Adam's — otherwise Adafactor would compute its own intrinsic step size
    and ignore ``lr``. ``beta1=0.9`` enables Adam-style momentum (HF
    supports it as an optional knob); set to None for the canonical
    no-momentum Adafactor.
    """
    from transformers.optimization import Adafactor as HFAdafactor

    lora_A_params, lora_B_params, other_params = [], [], []
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
    return HFAdafactor(
        param_groups,
        lr=lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        beta1=0.9,                  # Adam-style momentum on top of rank-1 v
        weight_decay=weight_decay,
        eps=(1e-30, 1e-3),          # paper defaults (eps1, eps2)
        clip_threshold=1.0,         # paper default ``d``
        decay_rate=-0.8,            # paper β₂ schedule exponent
    )


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


def _newton_schulz_hybrid_deepseek(X, total_steps=10, eps=1e-7):
    """
    DeepSeek-V4 hybrid Newton-Schulz (DeepSeek-V4 §2.4 Algorithm 1).

    Two-stage degree-5 polynomial iteration. First 8 of 10 steps use the
    aggressive Bernstein-style coefficients (3.4445, -4.7750, 2.0315) to
    drive σ rapidly toward 1 even from wide spread. Last 2 steps switch to
    conservative (2, -1.5, 0.5) which "stabilize singular values precisely
    at 1" once already close.

    For total_steps != 10, the split is 80%/20%.
    """
    X = X.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    norm = X.norm() + eps
    X = X / norm
    n_aggressive = max(1, int(round(0.8 * total_steps)))
    n_refine = max(1, total_steps - n_aggressive)
    for coeffs in [(3.4445, -4.7750, 2.0315)] * n_aggressive + [(2.0, -1.5, 0.5)] * n_refine:
        a, b, c = coeffs
        # X_{k+1} = a·X + b·(X X^T) X + c·(X X^T)² X
        XX = X @ X.T
        X = a * X + (b * XX + c * XX @ XX) @ X
    return X.T if tall else X


def _polar_express_quintic_remez(l, u, max_iter=64, tol=1e-15):
    """Optimal odd-quintic minimax approximation to the constant function
    x → 1 over [l, u]. Returns (a, b, c) for p(x) = a·x + b·x³ + c·x⁵.
    Ported from Amsel et al. PolarExpress (arXiv:2505.16932) repo
    polar_express.py:optimal_quintic.
    """
    import numpy as np
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = float('inf'), None
    for _ in range(max_iter):
        old_E = E
        LHS = np.array([
            [l, l ** 3, l ** 5, 1],
            [q, q ** 3, q ** 5, -1],
            [r, r ** 3, r ** 5, 1],
            [u, u ** 3, u ** 5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        disc = 9 * b * b - 20 * a * c
        if disc < 0:
            break
        roots = (-3 * b + np.array([-1, 1]) * disc ** 0.5) / (10 * c)
        if np.any(roots < 0):
            break
        q, r = np.sqrt(roots)
        if old_E is not None and abs(old_E - E) < tol:
            break
    return float(a), float(b), float(c)


def _polar_express_compose_coeffs(l_init=1e-3, num_iters=10, safety_factor_eps=1e-2, cushion=0.02):
    """Generate per-iteration optimal coefficients for PolarExpress
    (arXiv:2505.16932). Coefficients map [l_init, 1] toward [1, 1] across
    num_iters degree-5 NS iterations. Computed once at module load."""
    u = 1.0
    l = float(l_init)
    safety_factor = 1 + safety_factor_eps
    coeffs = []
    for it in range(num_iters):
        a, b, c = _polar_express_quintic_remez(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l ** 3 + c * l ** 5
            pu = a * u + b * u ** 3 + c * u ** 5
            rescaler = 2 / (pl + pu)
            a *= rescaler; b *= rescaler; c *= rescaler
        if it < num_iters - 1:
            a /= safety_factor; b /= safety_factor ** 3; c /= safety_factor ** 5
        coeffs.append((float(a), float(b), float(c)))
        l = a * l + b * l ** 3 + c * l ** 5
        u = 2 - l
    return coeffs


# Pre-computed PolarExpress coefficient series. Computed at import for the
# default worst-case σ_min/σ_max ratio of 1e-3.
_POLAR_EXPRESS_COEFFS = _polar_express_compose_coeffs(l_init=1e-3, num_iters=10)


def _polar_express(X, nsteps=5, eps=1e-7):
    """PolarExpress orthogonalization (Amsel et al., arXiv:2505.16932).
    Like Newton-Schulz but with per-iteration optimal degree-5 coefficients
    pre-generated by _polar_express_compose_coeffs. Pre-normalizes by
    Frobenius norm with safety factor 1.01 (matching reference impl).
    """
    X = X.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    norm = X.norm() * 1.01 + eps
    X = X / norm
    coeffs = _POLAR_EXPRESS_COEFFS[:nsteps]
    if len(coeffs) < nsteps:
        # Repeat the last (most refined) coefficient for any extra steps.
        coeffs = coeffs + [_POLAR_EXPRESS_COEFFS[-1]] * (nsteps - len(coeffs))
    for a, b, c in coeffs:
        XX = X @ X.T
        X = a * X + (b * XX + c * XX @ XX) @ X
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
        self.alpha = alpha
        self.rank = rank
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
        self.alpha = alpha
        self.rank = rank
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

    NOTE — at picard_iters=1 (the uncoupled default) the polar-product
    correction is per-factor and intent matches implementation. At
    picard_iters≥2 (the `-coupled` and `-coupled-endrms` build_optimizer
    entries) the cross-coupling step uses one of two compatibility-equivalent
    expressions — (1/η)·B^T·dB_prev·A and (1/η)·B·dA_prev·A^T — that diverge
    once Adam has run independently on each factor. The Picard fixed point is
    therefore the KKT point of an incoherent objective, not a principled
    approximation to the joint tangent operator-norm problem. The principled
    "Adam-preconditioned hybrid Picard replacement" is core-momentum on the
    forbidden-corner core Ĥ followed by projected quotient polar — variant 2
    in docs/notes/polar_product/theory.md §6, with the variational
    backdrop in docs/notes/polar_product/theory.md. Existing leaderboard
    standing of -coupled / -coupled-endrms reflects which incoherent fixed
    point lands well empirically, not which preconditioner is principled.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, ns_steps=5, adapter_name=None,
                 lora_plus_multiplier=1.0,
                 log_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=1,
                 precond_method="eigh", higham_iters=10,
                 picard_iters=1, end_rms_align=False, picard_alpha=1.0,
                 operator_type="polar",
                 polar_norm_dir="frob",
                 polar_sigma_power=None,
                 polar_method="ns",
                 anderson_m=0, anderson_reg=1e-10):
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
        # Damping factor on the iter-2+ cross-coupling term. α=1 reproduces
        # standard Picard; α=0 zeros the cross-term (equivalent to picard_iters=1
        # except in passing through the diagnostic instrumentation). Continuous
        # probe of cross-term magnitude.
        self.picard_alpha = picard_alpha
        # Muon+ (arXiv:2602.21545) — replace Frobenius RMS-align with per-row
        # or per-column ℓ₂ normalization of the orthogonalized output, then
        # rescale to the original ‖u‖_F. "frob" = current behavior. "row" /
        # "col" / "row_col" / "col_row" = Muon+ Norm_(d) directions applied
        # to geo_A and geo_B. The rescale-to-‖u‖_F preserves overall step
        # magnitude regardless of direction; the only effect is per-row /
        # per-column homogenization of update sizes.
        if polar_norm_dir not in {"frob", "row", "col", "row_col", "col_row"}:
            raise ValueError(f"polar_norm_dir must be one of frob/row/col/row_col/col_row, got {polar_norm_dir!r}")
        self.polar_norm_dir = polar_norm_dir
        # HTMuon (arXiv:2603.10067) — replace NS polar (σ → 1) with SVD-based
        # σ → σ^p generalization. None = use NS (default Muon-polar). 0 =
        # exact polar via SVD (σ → 1, equivalent to NS at high iter count).
        # p ∈ (0, 1) = HT-SR-motivated heavier-tailed update. 1 = no
        # orthogonalization (σ → σ, identity-on-input). The HTMuon paper
        # default is p=0.125. Trade-off: σ^p with p>0 preserves more of the
        # gradient's natural heavy-tailedness, at the cost of NS's
        # implicit dead-zone suppression of noise directions.
        if polar_sigma_power is not None and not (0.0 <= polar_sigma_power <= 1.0):
            raise ValueError(f"polar_sigma_power must be in [0,1] or None, got {polar_sigma_power!r}")
        self.polar_sigma_power = polar_sigma_power
        # polar_method selects the polynomial polar approximation: "ns" =
        # standard degree-3 (1.5x − 0.5x³, current Muon default); "ns_hybrid"
        # = DeepSeek-V4 §2.4 two-stage degree-5 (8 aggressive + 2 refine);
        # "polar_express" = Amsel et al. 2505.16932 per-iteration optimal
        # degree-5 minimax. Higher-quality methods better orthogonalize wide
        # σ-range inputs (where standard NS-5 leaves residual variation).
        if polar_method not in {"ns", "ns_hybrid", "polar_express"}:
            raise ValueError(f"polar_method must be one of ns/ns_hybrid/polar_express, got {polar_method!r}")
        self.polar_method = polar_method
        # When True, override the polar pipeline's per-iterate RMS-align so
        # the step magnitude is rescaled to the ORIGINAL ‖u_A‖, ‖u_B‖
        # (Adam direction norms before any cross-term correction). The
        # default (False) reproduces the original behavior where the
        # pipeline rescales to ‖u_A_eff‖ — which inflates the step when
        # the cross-term is large. No-op at picard_iters=1 (cross-term=0).
        self.end_rms_align = end_rms_align
        if picard_iters < 1:
            raise ValueError("picard_iters must be >= 1")
        if operator_type not in {"polar", "clip"}:
            raise ValueError(f"operator_type must be 'polar' or 'clip', got {operator_type!r}")
        self.operator_type = operator_type
        # Anderson(m) acceleration of the picard fixed-point iteration on (dA, dB).
        # m=0 disables; m>=1 keeps the last m (input, output) pairs and mixes
        # the next iterate as G(x_k) - ΔG · γ where γ = argmin ‖r_k - ΔR γ‖².
        # Damped-oscillation regime (osc_cos ≈ -0.85 at r=16) is Anderson's
        # sweet spot; per closeout doc, expect 3-5 effective iters to converge
        # vs 16 for plain Picard. anderson_reg adds Tikhonov to the LSQ solve
        # for numerical stability.
        if anderson_m < 0:
            raise ValueError("anderson_m must be >= 0")
        self.anderson_m = int(anderson_m)
        self.anderson_reg = float(anderson_reg)

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                'step': 0,
            }

    @staticmethod
    def _sigma_power_polar(M, p, eps=1e-30):
        """HTMuon (arXiv:2603.10067) generalized polar: SVD-based σ → σ^p.

        p=0 ⇒ exact polar (UV^T, all σ collapsed to 1). p=1 ⇒ identity on M
        (σ unchanged). p ∈ (0, 1) ⇒ heavier-tailed than polar but still
        compresses range. SVD cost is O(min(m,n)² · max(m,n)) which is
        manageable when one side is r (~16-64).
        """
        # SVD on float32 for stability; cast back to input dtype.
        in_dtype = M.dtype
        Mf = M.float()
        U, S, Vh = torch.linalg.svd(Mf, full_matrices=False)
        S_pow = S.clamp_min(eps).pow(p)
        out = (U * S_pow.unsqueeze(0)) @ Vh
        return out.to(in_dtype)

    @staticmethod
    def _muon_plus_norm(M, direction, eps=1e-30):
        """Muon+ Norm_(d) operator (arXiv:2602.21545 §3, Eq. 3-8). Divides
        each row/col by its ℓ₂ norm (or both, composed). For direction='frob'
        returns the input unchanged. Caller is expected to rescale to a
        target Frobenius norm afterwards."""
        if direction == "frob":
            return M
        if direction == "row":
            denom = M.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
            return M / denom
        if direction == "col":
            denom = M.pow(2).sum(dim=-2, keepdim=True).sqrt().clamp_min(eps)
            return M / denom
        if direction == "row_col":
            return AdamPolarProductLoRA._muon_plus_norm(
                AdamPolarProductLoRA._muon_plus_norm(M, "row", eps), "col", eps)
        if direction == "col_row":
            return AdamPolarProductLoRA._muon_plus_norm(
                AdamPolarProductLoRA._muon_plus_norm(M, "col", eps), "row", eps)
        raise ValueError(direction)

    def _polar_pipeline(self, u_A, u_B, SA_half_inv, SB_half_inv, lr):
        """One pass of the polar-product update + RMS-align.

        Returns (dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm,
        P_A, P_B). P_A, P_B are the polar (Newton-Schulz) outputs used
        for the H3 polar-sensitivity diagnostic.

        operator_type='polar' (default): Newton-Schulz polar, saturates all
        singular values to 1. Direction-only.
        operator_type='clip': singular-value clip with R-equal τ rule
        (τ = ‖X‖_F/√r). Preserves sub-bulk spectrum, caps top modes.
        Variationally exact prox of per-block constrained Frobenius
        program; tested as the missing fourth quadrant (clip without
        gauge) — the gauge would absorb cross-coupling, which we don't
        want here.
        """
        op = getattr(self, "operator_type", "polar")
        psp = getattr(self, "polar_sigma_power", None)
        pm = getattr(self, "polar_method", "ns")

        def _polar_op(X):
            if op == "clip":
                return _clip_R_equal(X)
            if psp is not None:
                return self._sigma_power_polar(X, psp)
            if pm == "ns_hybrid":
                return _newton_schulz_hybrid_deepseek(X, total_steps=max(self.ns_steps, 10))
            if pm == "polar_express":
                return _polar_express(X, nsteps=self.ns_steps)
            return _newton_schulz(X, nsteps=self.ns_steps)

        X_B = u_B @ SA_half_inv
        P_B = _polar_op(X_B)
        geo_B = P_B @ SA_half_inv

        X_A = SB_half_inv @ u_A
        P_A = _polar_op(X_A)
        geo_A = SB_half_inv @ P_A

        uA_norm = u_A.norm()
        uB_norm = u_B.norm()
        # Muon+ row/col normalization on the orthogonalized geo_{A,B} BEFORE
        # the Frobenius rescale. polar_norm_dir='frob' = original behavior
        # (no row/col reshaping); other values normalize rows/cols of geo to
        # unit ℓ₂ then rescale to ‖u‖_F so the total step magnitude is the
        # same. The only effect is per-row/per-col homogenization.
        geo_A_n = self._muon_plus_norm(geo_A, self.polar_norm_dir)
        geo_B_n = self._muon_plus_norm(geo_B, self.polar_norm_dir)
        gA_norm = geo_A_n.norm() + 1e-30
        gB_norm = geo_B_n.norm() + 1e-30
        dA = -lr * (uA_norm / gA_norm) * geo_A_n
        dB = -self.lora_plus_multiplier * lr * (uB_norm / gB_norm) * geo_B_n
        return dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm, P_A, P_B

    def _adam_direction(self, state, gA, gB):
        """Per-coord Adam on raw (gA, gB) → (u_A, u_B) for the polar pipeline.

        Hook point: subclasses may override to run Adam in a different basis
        (e.g. SOAP rotates by data-derived eigenbases of gA gA^T and gB^T gB
        before applying per-coord Adam, then rotates back). State for momentum
        and variance lives in self.pair_state[i] under keys the override
        controls; the parent contract is just that this returns Adam-style
        denoised directions of the same shape as gA, gB.
        """
        state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
        state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
        state['v_A'].mul_(self.beta2).addcmul_(gA, gA, value=1.0 - self.beta2)
        state['v_B'].mul_(self.beta2).addcmul_(gB, gB, value=1.0 - self.beta2)
        bc1 = 1.0 - self.beta1 ** state['step']
        bc2 = 1.0 - self.beta2 ** state['step']
        u_A = (state['m_A'] / bc1) / ((state['v_A'] / bc2).sqrt() + self.eps)
        u_B = (state['m_B'] / bc1) / ((state['v_B'] / bc2).sqrt() + self.eps)
        return u_A, u_B

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

            # Adam direction in the polar pipeline's input frame. The default
            # implementation runs per-coord Adam on raw (gA, gB); subclasses
            # (e.g. AdamSOAPPolarProductLoRA) may override this hook to run
            # Adam in a data-derived eigenbasis.
            u_A, u_B = self._adam_direction(state, gA, gB)

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
            # Saved BEFORE the loop — these are the AdaMuon RMS-align targets
            # in end_rms_align mode (vs ‖u_A_eff‖ in the original mode).
            uA_norm_orig = u_A.norm()
            uB_norm_orig = u_B.norm()
            # Anderson history: list of (x_flat, g_flat) where x is the input
            # to G and g = G(x) is the output. Only used when anderson_m > 0.
            and_xs = [] if self.anderson_m > 0 else None
            and_gs = [] if self.anderson_m > 0 else None
            shapeA = A_f.shape
            shapeB = B_f.shape
            nA_el = A_f.numel()
            for k in range(self.picard_iters):
                if k == 0:
                    u_A_eff = u_A
                    u_B_eff = u_B
                else:
                    u_A_eff = u_A + self.picard_alpha * (B_f.T @ dB_prev @ A_f) / lr
                    u_B_eff = u_B + self.picard_alpha * (B_f @ dA_prev @ A_f.T) / lr
                dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm, _, _ = \
                    self._polar_pipeline(u_A_eff, u_B_eff, SA_half_inv, SB_half_inv, lr)
                if self.end_rms_align:
                    # Override the pipeline's RMS-align: rescale to the
                    # ORIGINAL Adam-direction norm rather than ‖u_A_eff‖.
                    # Re-expose uA_norm / gA_norm / rms_scale_A consistently
                    # so the diagnostics block below still reflects what
                    # was actually applied.
                    gA_norm = geo_A.norm() + 1e-30
                    gB_norm = geo_B.norm() + 1e-30
                    uA_norm = uA_norm_orig
                    uB_norm = uB_norm_orig
                    dA = -lr * (uA_norm / gA_norm) * geo_A
                    dB = -self.lora_plus_multiplier * lr * (uB_norm / gB_norm) * geo_B
                if self.anderson_m > 0:
                    # Type-II Anderson on the (dA, dB) joint iterate.
                    # x_curr is the input to G this iter; g_curr is G(x_curr).
                    x_curr = torch.cat([dA_prev.reshape(-1), dB_prev.reshape(-1)])
                    g_curr = torch.cat([dA.reshape(-1), dB.reshape(-1)])
                    r_curr = g_curr - x_curr
                    m_use = min(self.anderson_m, len(and_xs))
                    if m_use >= 1:
                        # ΔR[:, j] = r_curr - r_{k-1-j}; ΔG[:, j] = g_curr - g_{k-1-j}
                        DR_cols = []
                        DG_cols = []
                        for j in range(m_use):
                            x_old = and_xs[-1 - j]
                            g_old = and_gs[-1 - j]
                            r_old = g_old - x_old
                            DR_cols.append(r_curr - r_old)
                            DG_cols.append(g_curr - g_old)
                        DR = torch.stack(DR_cols, dim=1)
                        DG = torch.stack(DG_cols, dim=1)
                        gram = DR.T @ DR
                        rhs = DR.T @ r_curr
                        reg = self.anderson_reg * gram.diagonal().mean().clamp_min(1e-30)
                        gram_reg = gram + reg * torch.eye(
                            m_use, dtype=gram.dtype, device=gram.device)
                        try:
                            gamma = torch.linalg.solve(gram_reg, rhs)
                            g_mixed = g_curr - DG @ gamma
                            dA = g_mixed[:nA_el].reshape(shapeA)
                            dB = g_mixed[nA_el:].reshape(shapeB)
                        except torch._C._LinAlgError:
                            pass  # fall through to plain Picard iterate
                    and_xs.append(x_curr)
                    and_gs.append(g_curr)
                    if len(and_xs) > self.anderson_m + 1:
                        and_xs.pop(0)
                        and_gs.pop(0)
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

                # Muon+ premise probe (arXiv:2602.21545 §2): after orthogonal-
                # ization, per-row/per-col ℓ₂ norms of geo_{A,B} have high
                # variance even though the matrix is well-conditioned σ-wise.
                # Logged from the PRE-normalization geo_A so the std reflects
                # what the Muon+ normalization step is actually correcting.
                # std/mean ratio ≫ 0 ⇒ Muon+ has something to fix; ≈ 0 ⇒ rows
                # are already balanced and Muon+ is a no-op.
                with torch.no_grad():
                    geo_A_f = geo_A.float()
                    geo_B_f = geo_B.float()
                    rows_A = geo_A_f.pow(2).sum(dim=-1).sqrt()
                    cols_A = geo_A_f.pow(2).sum(dim=-2).sqrt()
                    rows_B = geo_B_f.pow(2).sum(dim=-1).sqrt()
                    cols_B = geo_B_f.pow(2).sum(dim=-2).sqrt()
                    rec["geoA_row_norm_cv"] = float(
                        rows_A.std() / rows_A.mean().clamp_min(1e-30))
                    rec["geoA_col_norm_cv"] = float(
                        cols_A.std() / cols_A.mean().clamp_min(1e-30))
                    rec["geoB_row_norm_cv"] = float(
                        rows_B.std() / rows_B.mean().clamp_min(1e-30))
                    rec["geoB_col_norm_cv"] = float(
                        cols_B.std() / cols_B.mean().clamp_min(1e-30))

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

                    # X_unc spectrum probe (Step 3 prep — clip τ-rule
                    # design). The spectrum of the whitened Adam direction
                    # X_unc = S_B^{-1/2} u_A is the input to the per-block
                    # operator (polar in this baseline; clip in the
                    # candidate). Its singular values determine whether
                    # clip would do real work. SVD is r×n / m×r with r
                    # small — cheap.
                    Xunc_A = SB_half_inv @ u_A           # (r, n)
                    Xunc_B = u_B @ SA_half_inv           # (m, r)
                    sv_A = torch.linalg.svdvals(Xunc_A.float())
                    sv_B = torch.linalg.svdvals(Xunc_B.float())
                    sv_A_sorted, _ = torch.sort(sv_A, descending=True)
                    sv_B_sorted, _ = torch.sort(sv_B, descending=True)
                    sv_A_max = float(sv_A_sorted[0])
                    sv_B_max = float(sv_B_sorted[0])
                    sv_A_min = float(sv_A_sorted[-1])
                    sv_B_min = float(sv_B_sorted[-1])
                    sv_A_med = float(sv_A_sorted[len(sv_A_sorted) // 2])
                    sv_B_med = float(sv_B_sorted[len(sv_B_sorted) // 2])
                    # p90 (= clip-threshold candidate)
                    p90_idx_A = max(0, int(0.1 * len(sv_A_sorted)) - 1)
                    p90_idx_B = max(0, int(0.1 * len(sv_B_sorted)) - 1)
                    sv_A_p90 = float(sv_A_sorted[p90_idx_A])
                    sv_B_p90 = float(sv_B_sorted[p90_idx_B])
                    rec["xunc_A_smax"] = sv_A_max
                    rec["xunc_A_smin"] = sv_A_min
                    rec["xunc_A_smedian"] = sv_A_med
                    rec["xunc_A_sp90"] = sv_A_p90
                    rec["xunc_A_frob"] = float(sv_A.pow(2).sum().sqrt())
                    rec["xunc_A_stable_rank"] = float(
                        sv_A.pow(2).sum() / (sv_A_max ** 2 + 1e-30))
                    rec["xunc_A_participation"] = float(
                        sv_A.sum().pow(2) / (sv_A.pow(2).sum() + 1e-30))
                    rec["xunc_B_smax"] = sv_B_max
                    rec["xunc_B_smin"] = sv_B_min
                    rec["xunc_B_smedian"] = sv_B_med
                    rec["xunc_B_sp90"] = sv_B_p90
                    rec["xunc_B_frob"] = float(sv_B.pow(2).sum().sqrt())
                    rec["xunc_B_stable_rank"] = float(
                        sv_B.pow(2).sum() / (sv_B_max ** 2 + 1e-30))
                    rec["xunc_B_participation"] = float(
                        sv_B.sum().pow(2) / (sv_B.pow(2).sum() + 1e-30))
                    # Reference magnitude: would-be clip-τ from R-equal rule.
                    # τ_R-equal = ‖X_unc‖_F / √r so ratio (smax / τ_R-equal)
                    # = smax · √r / ‖X_unc‖_F tells us how peaky vs flat
                    # the spectrum is in the units that matter for clip.
                    r = u_A.shape[0]
                    rec["xunc_A_smax_over_tau_equal"] = sv_A_max * (r ** 0.5) / (rec["xunc_A_frob"] + 1e-30)

                    # cos(polar, clip) diagnostic — measures how much the
                    # operator choice affects step direction. Both operators
                    # preserve U, V; only the singular values differ. Compute
                    # cos analytically without re-doing SVDs.
                    # polar: σ → 1; clip: σ → min(σ, τ_R-equal).
                    # cos(polar, clip) = Σ min(σ, τ) / (√r · √Σ min(σ, τ)²)
                    tau_A = float(rec["xunc_A_frob"]) / max(1.0, r ** 0.5)
                    tau_B = float(rec["xunc_B_frob"]) / max(1.0, r ** 0.5)
                    sv_A_clipped = sv_A.clamp_max(tau_A)
                    sv_B_clipped = sv_B.clamp_max(tau_B)
                    sumA = float(sv_A_clipped.sum())
                    fnA = float(sv_A_clipped.pow(2).sum().sqrt() + 1e-30)
                    sumB = float(sv_B_clipped.sum())
                    fnB = float(sv_B_clipped.pow(2).sum().sqrt() + 1e-30)
                    sqrt_r = max(1.0, r ** 0.5)
                    rec["cos_polar_clip_A"] = sumA / (sqrt_r * fnA)
                    rec["cos_polar_clip_B"] = sumB / (sqrt_r * fnB)
                    rec["xunc_B_smax_over_tau_equal"] = sv_B_max * (r ** 0.5) / (rec["xunc_B_frob"] + 1e-30)
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
                    # Oscillation detector — cos between successive Picard
                    # iterate displacements δ_0 = dA2 - dA1 and δ_1 = dA3 - dA2.
                    # Positive ⇒ monotone contraction toward fixed point.
                    # Negative ⇒ iterates oscillating across the fixed point;
                    # signature of negative-eigenvalue Picard Jacobian.
                    # Per-pair scalar; aggregated to {min, median, max} via
                    # _emit_optim_diagnostics.
                    delta_A_0 = dA2 - dA1
                    delta_A_1 = dA3 - dA2
                    delta_B_0 = dB2 - dB1
                    delta_B_1 = dB3 - dB2
                    rec["picard_osc_cos_A"] = _frob_cos(delta_A_1, delta_A_0)
                    rec["picard_osc_cos_B"] = _frob_cos(delta_B_1, delta_B_0)
                    rec.update(_finite_step_product_diagnostics(A_f, B_f, dA, dB))
                    # Cross-term direction probe — does the iter-1→iter-2
                    # displacement (the part the cross-term injects) point
                    # along the gradient direction (-u_A)? Positive ⇒ cross-
                    # term is descent-aligned refinement. Negative or near-
                    # zero ⇒ cross-term is rotation orthogonal to gradient
                    # (the noise hypothesis at small r).
                    rec["cross_cos_A"] = _frob_cos(delta_A_0, -u_A)
                    rec["cross_cos_B"] = _frob_cos(delta_B_0, -u_B)
                    rec["cross_norm_A_rel"] = float(delta_A_0.norm() / nA1)
                    rec["cross_norm_B_rel"] = float(delta_B_0.norm() / nB1)

                    # Cautious-mask diagnostic — per-coordinate sign agreement of
                    # the iter-2 correction (δ = dA2−dA1) with the descent
                    # direction (−grad_A). Two reductions:
                    #   *_count_frac: fraction of coordinates where signs agree
                    #     (= what fraction of the correction the cautious mask
                    #     would preserve if applied uniformly).
                    #   *_norm_frac: fraction of ‖δ‖² coming from agreeing
                    #     coordinates (= what fraction of the correction's
                    #     energy survives the cautious mask). More directly
                    #     predicts how cautious-coupled would behave.
                    # gA, gB are the raw gradients on A, B (computed earlier
                    # in this step from A.grad / B.grad).
                    neg_gA = -gA
                    neg_gB = -gB
                    mask_A = (torch.sign(delta_A_0) == torch.sign(neg_gA))
                    mask_B = (torch.sign(delta_B_0) == torch.sign(neg_gB))
                    rec["cautious_mask_A_count_frac"] = float(mask_A.float().mean())
                    rec["cautious_mask_B_count_frac"] = float(mask_B.float().mean())
                    delta_A_sq = delta_A_0 * delta_A_0
                    delta_B_sq = delta_B_0 * delta_B_0
                    den_A = float(delta_A_sq.sum()) + 1e-30
                    den_B = float(delta_B_sq.sum()) + 1e-30
                    rec["cautious_mask_A_norm_frac"] = float(
                        (delta_A_sq * mask_A.float()).sum() / den_A)
                    rec["cautious_mask_B_norm_frac"] = float(
                        (delta_B_sq * mask_B.float()).sum() / den_B)
                    # Baseline: agreement of iter-1 step itself with −grad.
                    # If iter-1 already disagrees on many coordinates, the
                    # mask isn't isolating "iter-2 is bad" from "every step
                    # has noise vs raw grad."
                    iter1_mask_A = (torch.sign(dA1) == torch.sign(neg_gA))
                    iter1_mask_B = (torch.sign(dB1) == torch.sign(neg_gB))
                    rec["cautious_iter1_A_count_frac"] = float(iter1_mask_A.float().mean())
                    rec["cautious_iter1_B_count_frac"] = float(iter1_mask_B.float().mean())

                    # Descent alignment + tangent magnitude per iter (mechanism
                    # diagnostic — k=2's r=64 benefit comes from JOINT NE
                    # convergence, not col(B) refinement).
                    # descent_iter = ⟨G_A, dA⟩ + ⟨G_B, dB⟩ (negative ⇒
                    # descent direction). If iter 2 < iter 1, k=2 buys
                    # more linear descent per step.
                    # frob_J_iter = ‖B dA + dB A‖_F (joint tangent norm).
                    for tag, dA_var, dB_var in [
                        ("iter1", dA1, dB1),
                        ("iter2", dA2, dB2),
                    ]:
                        dA32 = dA_var.float(); dB32 = dB_var.float()
                        rec[f"descent_{tag}"] = float(
                            (gA * dA32).sum() + (gB * dB32).sum()
                        )
                        Jt = B_f @ dA32 + dB32 @ A_f
                        rec[f"frob_J_{tag}"] = float(Jt.norm())

                    # col(B) decomposition of dB at iter 1 vs iter 2
                    # (mechanism diagnostic for r=64 cross-coupling).
                    # Hypothesis: at r=64 with k=2, baseline polar's win
                    # comes from iter 2 GROWING the col(B) component of
                    # dB (refining the existing preconditioner). Gauge
                    # k=2 should NOT show this growth (extension-forced).
                    # P_col(B) = Q_B Q_B^T (projector onto col(B)).
                    # Decompose: dB_iter = P_col(B) · dB_iter + (I − P_col(B)) · dB_iter.
                    Q_B_proj, _ = torch.linalg.qr(B_f, mode='reduced')  # (m, r)
                    for tag, dB_var in [("iter1", dB1), ("iter2", dB2)]:
                        dB32 = dB_var.float()
                        proj = Q_B_proj @ (Q_B_proj.T @ dB32)
                        perp = dB32 - proj
                        rec[f"colB_frac_{tag}"] = float(
                            (proj.norm() / (dB32.norm() + 1e-30)).pow(2)
                        )
                        rec[f"colB_norm_{tag}"] = float(proj.norm())
                        rec[f"perp_norm_{tag}"] = float(perp.norm())

                    # Local-model variational score for k=1 and k=2 candidates
                    # (see plan i-read-all-three-quizzical-plum, Step 1).
                    # score(ΔA, ΔB) = ⟨u_A, ΔA⟩ + ⟨u_B, ΔB⟩
                    #                 + (1/(2·lr))·‖B·ΔA + ΔB·A‖_F²
                    # Negative (k2 − k1) ⇒ the per-step variational program
                    # prefers k=2 over k=1 on this pair. If the sign of
                    # (k2 − k1) consistently matches the empirical winner per
                    # rank, a deterministic local-model selector is a no-HP
                    # rule that picks k from local geometry.
                    def _local_score(dA_, dB_):
                        dA32 = dA_.float()
                        dB32 = dB_.float()
                        lin = float((u_A * dA32).sum() + (u_B * dB32).sum())
                        J = B_f @ dA32 + dB32 @ A_f
                        coupling = 0.5 * float((J * J).sum()) / lr
                        return lin + coupling
                    rec["local_score_k1"] = _local_score(dA1, dB1)
                    rec["local_score_k2"] = _local_score(dA2, dB2)
                    rec["local_score_k2_minus_k1"] = (
                        rec["local_score_k2"] - rec["local_score_k1"]
                    )

                diag_records.append(rec)

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class AdamSOAPPolarProductLoRA(AdamPolarProductLoRA):
    """SOAP-style preconditioning of (u_A, u_B) before the polar pipeline.

    Motivated by docs/notes/polar_product/closeout_2026_05_02.md §3 candidate 1:
    at r=16, full per-coord Adam on raw G beats no-Adam by +0.04 and matrix-Adam
    sits at +0.02, so well-conditioned polar inputs are doing real work. SOAP
    runs Adam in a data-derived eigenbasis (rather than the coordinate basis),
    which captures correlated-direction structure per-coord Adam misses.

    For LoRA factor gradients gA: (r, d_in), gB: (d_out, r) only the r-side is
    small enough to admit a cheap r×r preconditioner — so we maintain
        L_A: (r,r) ≈ EMA[gA gA^T]   (left covariance of A-grad)
        R_B: (r,r) ≈ EMA[gB^T gB]   (right covariance of B-grad)
    and refresh eigenbases Q_A, Q_B every soap_refresh_every steps. Per step:
        gA_rot = Q_A^T @ gA;     gB_rot = gB @ Q_B
        run per-coord Adam on (gA_rot, gB_rot)  → (û_A_rot, û_B_rot)
        u_A = Q_A @ û_A_rot;     u_B = û_B_rot @ Q_B^T
    The downstream polar pipeline (Newton-Schulz with S^{-1/2} preconditioners
    on the Gram side, RMS-align, Picard if enabled, diagnostics) is unchanged.

    Until the first refresh, Q_A = Q_B = I and this reduces exactly to
    AdamPolarProductLoRA. The L_A/R_B EMAs use no bias correction (Muon-canonical
    convention; SOAP paper consistent). Adam on the rotated grads keeps standard
    bias correction on m, v.

    Implementation notes (v2, faithful to Vyas et al. arXiv:2409.11321 Alg 3):
      * Momentum M is stored in the COORDINATE basis (line 3 of Alg 3) and
        rotated into the current eigenbasis EACH STEP for the Adam computation
        (line 4). The variance V is stored in the rotated basis (line 7); its
        bounded staleness across refreshes is the standard SOAP tradeoff.
      * Eigenvectors are refreshed via one step of power iteration + QR seeded
        with the previous Q (Algorithm 4: ``Q_new = qr(L @ Q_old).Q``). This
        is cheaper than full eigh and, more importantly, makes Q evolve
        smoothly across refreshes — full eigh has arbitrary sign/permutation
        per eigenvector and would discontinuously flip Q between adjacent
        refreshes (especially harmful at soap_refresh_every=1).
    """

    def __init__(self, model, *, soap_beta=0.95, soap_refresh_every=1, **kwargs):
        super().__init__(model, **kwargs)
        if soap_refresh_every < 1:
            raise ValueError("soap_refresh_every must be >= 1")
        if not (0.0 <= soap_beta < 1.0):
            raise ValueError("soap_beta must be in [0, 1)")
        self.soap_beta = float(soap_beta)
        self.soap_refresh_every = int(soap_refresh_every)
        for i, (A, B) in enumerate(self.pairs):
            r = A.shape[0]
            assert B.shape[1] == r, "LoRA factor convention: A is (r, d_in), B is (d_out, r)"
            dev = A.device
            eye = torch.eye(r, dtype=torch.float32, device=dev)
            st = self.pair_state[i]
            st['L_A'] = torch.zeros((r, r), dtype=torch.float32, device=dev)
            st['R_B'] = torch.zeros((r, r), dtype=torch.float32, device=dev)
            # Q_A, Q_B initialize to identity so the first soap_refresh_every-1
            # steps are exactly equivalent to AdamPolarProductLoRA.
            st['Q_A'] = eye.clone()
            st['Q_B'] = eye.clone()

    def _adam_direction(self, state, gA, gB):
        # Use the PRIOR eigenbasis to project this step's gradient (paper Alg 3
        # line 2: G_t' = Q_L^T G_t Q_R, where Q is the basis from step t-1).
        Q_A = state['Q_A']
        Q_B = state['Q_B']

        # Momentum in COORD basis (paper Alg 3 line 3); rotated into the prior
        # eigenbasis EACH STEP for the Adam computation (line 4). Keeping M in
        # coord basis is what makes a basis change cheap — only V picks up the
        # bounded staleness, not M.
        state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
        state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
        m_A_rot = Q_A.T @ state['m_A']
        m_B_rot = state['m_B'] @ Q_B

        # Variance in ROTATED basis (paper Alg 3 line 7).
        gA_rot = Q_A.T @ gA
        gB_rot = gB @ Q_B
        state['v_A'].mul_(self.beta2).addcmul_(gA_rot, gA_rot, value=1.0 - self.beta2)
        state['v_B'].mul_(self.beta2).addcmul_(gB_rot, gB_rot, value=1.0 - self.beta2)

        bc1 = 1.0 - self.beta1 ** state['step']
        bc2 = 1.0 - self.beta2 ** state['step']
        uA_rot = (m_A_rot / bc1) / ((state['v_A'] / bc2).sqrt() + self.eps)
        uB_rot = (m_B_rot / bc1) / ((state['v_B'] / bc2).sqrt() + self.eps)

        # Rotate back into the coordinate frame the polar pipeline expects.
        u_A = Q_A @ uA_rot
        u_B = uB_rot @ Q_B.T

        # AFTER computing the update, advance the preconditioner state for
        # step t+1 (paper Alg 3 lines 13-17).
        beta_p = self.soap_beta
        state['L_A'].mul_(beta_p).add_(gA @ gA.T, alpha=1.0 - beta_p)
        state['R_B'].mul_(beta_p).add_(gB.T @ gB, alpha=1.0 - beta_p)

        # Refresh eigenbasis via Algorithm 4 (power iteration + QR seeded with
        # prior Q). Before the QR step, sort prior Q's columns by descending
        # estimated eigenvalue and permute V's rotated axis accordingly so V's
        # per-slot index continues to track Q's column index across refreshes.
        # Matches official SOAP (https://github.com/nikhilvyas/SOAP) function
        # ``get_orthogonal_matrix_QR``: QR seeding alone makes Q evolve smoothly
        # but eigenvalue rank can swap between refreshes, and without this
        # permutation V's slots silently de-align from Q's columns.
        if state['step'] % self.soap_refresh_every == 0:
            try:
                est_eig_A = torch.diag(state['Q_A'].T @ state['L_A'] @ state['Q_A'])
                sort_A = torch.argsort(est_eig_A, descending=True)
                state['Q_A'] = state['Q_A'][:, sort_A].contiguous()
                state['v_A'] = state['v_A'].index_select(0, sort_A).contiguous()
                Q_A_new, _ = torch.linalg.qr(state['L_A'] @ state['Q_A'])
                state['Q_A'] = Q_A_new

                est_eig_B = torch.diag(state['Q_B'].T @ state['R_B'] @ state['Q_B'])
                sort_B = torch.argsort(est_eig_B, descending=True)
                state['Q_B'] = state['Q_B'][:, sort_B].contiguous()
                state['v_B'] = state['v_B'].index_select(1, sort_B).contiguous()
                Q_B_new, _ = torch.linalg.qr(state['R_B'] @ state['Q_B'])
                state['Q_B'] = Q_B_new
            except torch._C._LinAlgError:
                # Degenerate covariance (zero grad streak) — keep prior basis.
                pass

        # Per-pair SOAP spectrum probe — answers "is L_A actually isotropic?"
        # Computed at the diagnostics cadence (cheap: r×r eigh, r=16).
        if self.log_diagnostics and state['step'] % self.diagnostics_every == 0:
            with torch.no_grad():
                try:
                    eA = torch.linalg.eigvalsh(state['L_A']).clamp_min(0.0)
                    eB = torch.linalg.eigvalsh(state['R_B']).clamp_min(0.0)
                    state['_soap_spectrum'] = {
                        'L_A_cond': float(eA[-1] / (eA[0] + 1e-30)),
                        'L_A_top_frac': float(eA[-1] / (eA.sum() + 1e-30)),
                        'L_A_pr': float(eA.sum().pow(2) / (eA.pow(2).sum() + 1e-30)),
                        'R_B_cond': float(eB[-1] / (eB[0] + 1e-30)),
                        'R_B_top_frac': float(eB[-1] / (eB.sum() + 1e-30)),
                        'R_B_pr': float(eB.sum().pow(2) / (eB.pow(2).sum() + 1e-30)),
                    }
                except Exception:
                    state['_soap_spectrum'] = None
        return u_A, u_B

    @torch.no_grad()
    def step(self, closure=None):
        out = super().step(closure)
        if self.log_diagnostics:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                recs = [
                    s['_soap_spectrum'] for s in self.pair_state.values()
                    if s.get('_soap_spectrum')
                ]
                if recs:
                    payload = {
                        'event': 'optim_soap_step',
                        'step': int(step_count),
                        'n_pairs': len(recs),
                    }
                    for k in recs[0].keys():
                        vals = [r[k] for r in recs if r[k] == r[k]]
                        if vals:
                            payload[k + '_median'] = statistics.median(vals)
                            payload[k + '_min'] = min(vals)
                            payload[k + '_max'] = max(vals)
                    print(json.dumps(payload, sort_keys=True), flush=True)
        return out


class AdaFactorPolarProductLoRA(AdamPolarProductLoRA):
    """Adafactor-style rank-1 v factorization fed into the polar pipeline.

    Designed as a probe for "how much preconditioning precision the polar
    pipeline actually needs." Identical to AdamPolarProductLoRA except the
    second-moment EMA v ∈ ℝ^{m×n} is replaced by its rank-1 outer-product
    approximation: row sums R ∈ ℝ^m and column sums C ∈ ℝ^n, with
    ``v_approx[i, j] = R[i] · C[j] / sum(R)`` (Shazeer & Stern 2018,
    arXiv:1804.04235 Algorithm 2). Momentum, bias correction, eps, and the
    full polar pipeline are unchanged. This isolates the diagonal-precision
    axis from every other design choice — SOAP, basis rotation, RMS-align,
    polar — so the comparison vs AdamPolarProductLoRA cleanly answers
    whether full per-coord ``v`` is load-bearing for our setup.

    Three-way reading with SOAP:
      * Adafactor ≈ Adam: per-coord ``v`` is overkill; preconditioning is
        saturated by even rank-1 estimates. SOAP/low-rank d-side dead.
      * Adafactor < Adam by ~SOAP gap: full v is real but rotation isn't.
        SOAP ≈ Adam means rotation specifically adds nothing.
      * Adafactor << Adam: per-coord v is load-bearing, SOAP fails for
        another reason (polar saturation specific to rotation, etc.).
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Replace v_A, v_B (full element-wise) with row/col sums.
        for i, (A, B) in enumerate(self.pairs):
            r, d_in = A.shape
            d_out, _ = B.shape
            dev = A.device
            st = self.pair_state[i]
            # v_A: (r, d_in) → R_A: (r,), C_A: (d_in,)
            st['R_A'] = torch.zeros(r, dtype=torch.float32, device=dev)
            st['C_A'] = torch.zeros(d_in, dtype=torch.float32, device=dev)
            # v_B: (d_out, r) → R_B_af: (d_out,), C_B_af: (r,)
            st['R_B_af'] = torch.zeros(d_out, dtype=torch.float32, device=dev)
            st['C_B_af'] = torch.zeros(r, dtype=torch.float32, device=dev)
            # The parent's v_A, v_B slots are unused in this variant; we leave
            # them allocated to keep state shape compatible with diagnostics.

    def _adam_direction(self, state, gA, gB):
        # Adam-style momentum (unchanged from parent).
        state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
        state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)

        # Rank-1 second-moment factorization.
        gA_sq = gA * gA
        gB_sq = gB * gB
        state['R_A'].mul_(self.beta2).add_(gA_sq.sum(dim=1), alpha=1.0 - self.beta2)
        state['C_A'].mul_(self.beta2).add_(gA_sq.sum(dim=0), alpha=1.0 - self.beta2)
        state['R_B_af'].mul_(self.beta2).add_(gB_sq.sum(dim=1), alpha=1.0 - self.beta2)
        state['C_B_af'].mul_(self.beta2).add_(gB_sq.sum(dim=0), alpha=1.0 - self.beta2)

        bc1 = 1.0 - self.beta1 ** state['step']
        bc2 = 1.0 - self.beta2 ** state['step']

        # Adafactor reconstruction: v_approx[i, j] = R[i] · C[j] / sum(R).
        # Note sum(R) == sum(C) in expectation (both sum the same gradient²).
        sumR_A = state['R_A'].sum().clamp_min(1e-30)
        v_A_approx = state['R_A'].unsqueeze(1) * state['C_A'].unsqueeze(0) / sumR_A
        sumR_B = state['R_B_af'].sum().clamp_min(1e-30)
        v_B_approx = state['R_B_af'].unsqueeze(1) * state['C_B_af'].unsqueeze(0) / sumR_B

        u_A = (state['m_A'] / bc1) / ((v_A_approx / bc2).sqrt() + self.eps)
        u_B = (state['m_B'] / bc1) / ((v_B_approx / bc2).sqrt() + self.eps)

        # Stable-rank-of-g² probe: predicts whether the rank-1 approximation
        # is exact (stable_rank=1 ⇔ g² is rank-1 ⇔ Adafactor = Adam) or lossy
        # (stable_rank ≫ 1 ⇔ rank-1 throws away real info). Cheap: 2 power-
        # iter steps for top singular value, no full SVD.
        if self.log_diagnostics and state['step'] % self.diagnostics_every == 0:
            with torch.no_grad():
                def _stable_rank(M_sq):
                    fro_sq = (M_sq * M_sq).sum()
                    m, n = M_sq.shape
                    v = torch.randn(n, device=M_sq.device, dtype=M_sq.dtype)
                    for _ in range(2):
                        v = M_sq.T @ (M_sq @ v)
                        v = v / v.norm().clamp_min(1e-30)
                    top_sq = (M_sq @ v).pow(2).sum()
                    return float(fro_sq / top_sq.clamp_min(1e-30))
                try:
                    state['_adafactor_diag'] = {
                        'stable_rank_gA_sq': _stable_rank(gA_sq),
                        'stable_rank_gB_sq': _stable_rank(gB_sq),
                        'gA_sq_min_fullrank': float(min(gA_sq.shape)),
                        'gB_sq_min_fullrank': float(min(gB_sq.shape)),
                    }
                except Exception:
                    state['_adafactor_diag'] = None
        return u_A, u_B

    @torch.no_grad()
    def step(self, closure=None):
        out = AdamPolarProductLoRA.step(self, closure)
        if self.log_diagnostics:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                recs = [
                    s['_adafactor_diag'] for s in self.pair_state.values()
                    if s.get('_adafactor_diag')
                ]
                if recs:
                    payload = {
                        'event': 'optim_adafactor_step',
                        'step': int(step_count),
                        'n_pairs': len(recs),
                    }
                    for k in recs[0].keys():
                        vals = [r[k] for r in recs if r[k] == r[k]]
                        if vals:
                            payload[k + '_median'] = statistics.median(vals)
                            payload[k + '_min'] = min(vals)
                            payload[k + '_max'] = max(vals)
                    print(json.dumps(payload, sort_keys=True), flush=True)
        return out


class SignMomentumPolarProductLoRA(AdamPolarProductLoRA):
    """sign(m) fed into polar pipeline. No v tracking at all.

    Cleanest test of "do we need v?" without the m/|g| magnitude instability
    that breaks naive instant-Adam (β₂=0). sign(m) has unit-magnitude entries
    by construction, so the polar pipeline's RMS-align downstream sees a
    well-behaved input regardless of gradient magnitudes.

    Connection to effective-rank lifting: sign(m) is FULL RANK generically
    (every entry is ±1), while Adam's m/√v has effective rank ≈ 1/(1−β₁) ≈
    10 due to LoRA's per-example rank-1 gradient structure. So sign(m)
    feeds polar a higher-rank input. Polar then orthogonalizes more
    directions — IF those directions carry signal, this helps; if they're
    noise, it hurts.

    Inspired by LION (Chen et al. 2023, arXiv:2302.06675), which uses
    sign(m) for the parameter update directly.
    """

    def _adam_direction(self, state, gA, gB):
        # Momentum EMA (β₁) — same as parent.
        state['m_A'].mul_(self.beta1).add_(gA, alpha=1.0 - self.beta1)
        state['m_B'].mul_(self.beta1).add_(gB, alpha=1.0 - self.beta1)
        # No v EMA, no bias correction (sign is invariant to positive scaling).
        # state['v_A'], state['v_B'] are allocated by the parent constructor
        # but unused here — kept for state-shape compatibility with the
        # parent's diagnostic block that reads v.
        u_A = state['m_A'].sign()
        u_B = state['m_B'].sign()
        return u_A, u_B


class AdamPolarProductLoRAGauge(Optimizer):
    """Per-block polar + Sylvester min-Frob gauge lift, no postscale.

    The ``adam-polar-product-lora`` baseline produces an update pair that does
    not satisfy the min-Frobenius gauge ``B^T ΔB = ΔA A^T`` because each factor
    is independently RMS-aligned to its own Adam-direction Frobenius norm. The
    proposal in ``docs/notes/polar_product/proposal.md`` adds a Sylvester lift
    to project onto the gauge surface, then applies separate per-factor RMS
    scalars — which destroys the gauge it just enforced.

    This optimizer takes the clean stance (option 1 in the plan): enforce the
    gauge via the lift, apply NO postscale. Magnitude is whatever the lift
    produces times ``lr``. Tests whether the gauge alone — with no magnitude
    band-aid — does useful work under the polar operator, before any
    consideration of the variationally-correct clip operator (which is gated
    on the unresolved ``τ``-rule blocker, Q2 of the proposal).

    Per pair (A ∈ R^{r×n}, B ∈ R^{m×r}) per step:
      1. Adam EMA on raw factor gradients → (u_A, u_B).
      2. Thin QR: B = Q_B R_B, A^T = Q_A R_AT (so A = R_AT^T Q_A^T;
         define R_A := R_AT^T so A = R_A Q_A^T per proposal §0.3).
      3. Whitening: ũ_A = R_B^{-T} u_A, ũ_B = u_B R_AT^{-1} = u_B R_A^{-T}.
      4. Per-block polar via Newton–Schulz: P_A = polar(ũ_A) ∈ R^{r×n},
         P_B = polar(ũ_B) ∈ R^{m×r}.
      5. Joint tangent target J_target = -lr · (Q_B P_A + P_B Q_A^T).
         This sits in col(B) + row(A) by construction.
      6. Min-Frob lift via Sylvester (theory.md §"Sylvester gauge lift"):
            S_B K + K S_A = B^T J_target A^T,    K ∈ R^{r×r}
            ΔA = S_B^{-1} (B^T J_target − K A)
            ΔB = (J_target A^T − B K) S_A^{-1}
         With S_A = A A^T + δ I, S_B = B^T B + δ I.
         The lift solves min ‖ΔA‖_F² + ‖ΔB‖_F² s.t. B ΔA + ΔB A = J_target,
         which automatically gives B^T ΔB = ΔA A^T = K (KKT condition).
      7. Apply ΔA, ΔB. No RMS-align; no separate per-factor scalar.

    PEFT init B=0 fallback. At step 1, B is exactly zero and the QR + lift
    is ill-conditioned (Q_B undefined; ΔA = 0 forced by the gauge since J
    can only depend on B-induced contributions). Match the existing
    AdamPolarProductLoRA fallback: at step 1 (or whenever ‖B‖_F is below a
    tiny threshold), do a per-block whitened polar without the lift, with
    SB_half_inv = δ^{-1/2} I — equivalent to a plain Adam-direction Muon
    step on each factor independently. From step 2 onward the full lift
    runs unchanged.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, ns_steps=5, adapter_name=None,
                 lora_plus_multiplier=1.0,
                 picard_iters=1,
                 log_diagnostics=False, diagnostics_every=20):
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
        self.picard_iters = int(picard_iters)
        if self.picard_iters < 1:
            raise ValueError("picard_iters must be >= 1")
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
                raise ValueError("Gradients are required for AdamPolarProductLoRAGauge update.")
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

            A_f = A.float()
            B_f = B.float()
            r = A_f.shape[0]

            b_norm = float(B_f.norm())
            use_fallback = b_norm < 1e-8

            if use_fallback:
                # Per-block whitened polar without lift (matches the
                # adam-polar-product-lora step-1 path; at B=0 only ΔB carries
                # signal, but we update both for symmetry — A's update is
                # tiny and goes through the well-conditioned S_A^{-1/2}).
                S_A = spdify(A_f @ A_f.T, self.delta)
                SA_half_inv = _spd_inv_half(S_A, eps=self.delta, method="eigh")
                if b_norm < 1e-8:
                    SB_half_inv = (self.delta ** -0.5) * torch.eye(
                        r, dtype=torch.float32, device=A_f.device,
                    )
                else:
                    S_B = spdify(B_f.T @ B_f, self.delta)
                    SB_half_inv = _spd_inv_half(S_B, eps=self.delta, method="eigh")
                X_B = u_B @ SA_half_inv
                P_B = _newton_schulz(X_B, nsteps=self.ns_steps)
                geo_B = P_B @ SA_half_inv
                X_A = SB_half_inv @ u_A
                P_A = _newton_schulz(X_A, nsteps=self.ns_steps)
                geo_A = SB_half_inv @ P_A
                dA = -lr * geo_A
                dB = -self.lora_plus_multiplier * lr * geo_B
            else:
                # Thin QR. torch.linalg.qr(M, mode='reduced') returns Q (cols
                # orthonormal) and R upper-triangular with M = QR.
                Q_B, R_B = torch.linalg.qr(B_f, mode='reduced')   # (m,r), (r,r)
                Q_A, R_AT = torch.linalg.qr(A_f.T, mode='reduced')  # (n,r), (r,r)
                # A^T = Q_A R_AT, so A = R_AT^T Q_A^T = R_A Q_A^T with R_A = R_AT^T.

                S_B = spdify(B_f.T @ B_f, self.delta)              # (r,r)
                S_A = spdify(A_f @ A_f.T, self.delta)              # (r,r)

                # Picard inner iteration. At iter 0 use raw u; at iter k≥1
                # add cross-coupling correction to u via the standard
                # AdamPolarProductLoRA-coupled pattern:
                #   u_A_eff = u_A + (B^T · ΔB_prev · A) / lr
                #   u_B_eff = u_B + (B  · ΔA_prev · A^T) / lr
                # This is algebraically equivalent to the proposal §0.3
                # T_A formulation (X_unc = T_A − lr·R_B^{-T}·u_A) but
                # leaves polar's "wipe magnitude, restore via -lr factor"
                # convention intact at each iter.
                dA = torch.zeros_like(A_f)
                dB = torch.zeros_like(B_f)
                for k in range(self.picard_iters):
                    if k == 0:
                        u_A_eff = u_A
                        u_B_eff = u_B
                    else:
                        u_A_eff = u_A + (B_f.T @ dB @ A_f) / lr
                        u_B_eff = u_B + (B_f @ dA @ A_f.T) / lr

                    # Whitening via triangular solves.
                    u_A_white = torch.linalg.solve_triangular(R_B.T, u_A_eff, upper=False)
                    u_B_white = torch.linalg.solve_triangular(R_AT.T, u_B_eff.T, upper=False).T

                    # Per-block polar (NS); unit singular values; sign comes
                    # from the -lr factor in J_target below.
                    P_A = _newton_schulz(u_A_white, nsteps=self.ns_steps)  # (r,n)
                    P_B = _newton_schulz(u_B_white, nsteps=self.ns_steps)  # (m,r)

                    # Min-Frob lift on J_target = -lr (Q_B P_A + P_B Q_A^T).
                    core_11 = P_A @ Q_A + Q_B.T @ P_B                # (r,r)
                    RHS_K = -lr * (R_B.T @ core_11 @ R_AT)           # (r,r)
                    K = solve_sylvester(S_B, S_A, RHS_K)             # (r,r)

                    BTJ = -lr * (R_B.T @ (P_A + Q_B.T @ P_B @ Q_A.T))  # (r,n)
                    rhs_dA = BTJ - K @ A_f                              # (r,n)
                    dA = solve_spd(S_B, rhs_dA)                         # (r,n)

                    JAT = -lr * (Q_B @ P_A @ Q_A @ R_AT + P_B @ R_AT)  # (m,r)
                    rhs_dB = JAT - B_f @ K                              # (m,r)
                    dB = solve_spd(S_A, rhs_dB.T).T                     # (m,r)

                # LoRA+ multiplier on ΔB only (post-picard).
                dB = self.lora_plus_multiplier * dB

            if self.log_diagnostics:
                step_count_local = state['step']
                if step_count_local % self.diagnostics_every == 0:
                    rec = {
                        "norm_dA": float(dA.detach().norm()),
                        "norm_dB": float(dB.detach().norm()),
                        "norm_A": float(A_f.norm()),
                        "norm_B": float(B_f.norm()),
                        "fallback": int(use_fallback),
                    }
                    # Gauge-deviation invariant: should be ≈ 0 in the lift path,
                    # nonzero in the fallback path.
                    BTdB = B_f.T @ dB.float()
                    dAAT = dA.float() @ A_f.T
                    gauge_resid = float((BTdB - dAAT).norm())
                    gauge_denom = float(dAAT.norm()) + 1e-30
                    rec["gauge_residual_abs"] = gauge_resid
                    rec["gauge_residual_rel"] = gauge_resid / gauge_denom

                    # B-spectrum trajectory diagnostic (stable rank, condition).
                    # Mirrors the diag in baseline AdamPolarProductLoRA so we
                    # can compare longitudinal B-evolution across variants.
                    try:
                        eigB = torch.linalg.eigvalsh(B_f.T @ B_f).clamp_min(0.0)
                        smax_B = float(eigB.max())
                        rec["stable_rank_B"] = float(eigB.sum() / (smax_B + 1e-30))
                        rec["nrank_B_1e2"] = int((eigB > 1e-2 * smax_B).sum())
                        # Frob norm of K (the gauge variable from the lift).
                        # Under min-Frob KKT: K = B^T dB = dA A^T. Predicted
                        # to be small when the gauge constraint absorbs the
                        # cross-coupling input.
                        rec["K_frob"] = float(BTdB.norm())
                    except torch._C._LinAlgError:
                        rec["stable_rank_B"] = float("nan")
                        rec["nrank_B_1e2"] = -1
                        rec["K_frob"] = float("nan")

                    diag_records.append(rec)

            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


def _clip_R_equal(X):
    """Clip the singular values of X at τ = ‖X‖_F / √r_act.

    R-equal τ rule (LEGACY — failed at startup; see _clip_adamw_capped).
    Inherits ‖X‖_F as the magnitude, which is unbounded at startup when
    R_B^{-T} blows up (B small). Kept for diagnostic comparison only.
    """
    Xf = X.float()
    U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
    r_act = S.numel()
    tau = float(S.pow(2).sum().sqrt()) / max(1.0, r_act ** 0.5)
    S_clipped = S.clamp_max(tau)
    return (U * S_clipped.unsqueeze(0)) @ Vh


def _clip_adamw_capped(X, M_cap):
    """Clip X's singular values so ‖clip(X)‖_F ≤ M_cap (R-AdamW-cap rule).

    If ‖X‖_F ≤ M_cap, return X unchanged (natural prox already under
    the AdamW magnitude budget). Otherwise water-fill τ so that
        Σ min(σ_i, τ)² = M_cap².
    M_cap = lr · ‖u_A‖_F (or ‖u_B‖_F) supplies the validated AdamW
    magnitude convention as a CAP; clip's role is spectrum reshape
    within that budget, not magnitude-from-spectrum-shape.

    Algorithm: for each k = 1 .. r, hypothesize that the top k singular
    values are clipped to a common τ_k = √((M² − Σ_{i>k} σ_i²) / k).
    The valid k is the smallest one for which τ_k ≤ σ_k (= the k-th
    largest singular value), at which point the result is consistent.
    """
    Xf = X.float()
    U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
    norm_sq = float(S.pow(2).sum().item())
    M2 = float(M_cap) ** 2
    if norm_sq <= M2 or M2 <= 0.0:
        return Xf
    S_sorted, _ = torch.sort(S, descending=True)
    n = S_sorted.numel()
    # cum_sq_below[k] = sum of σ_i² for i > k (1-indexed).
    cum_sq_top = torch.cumsum(S_sorted.pow(2), 0)
    total_sq = float(cum_sq_top[-1].item())
    tau = None
    for k in range(1, n + 1):
        sum_below = total_sq - float(cum_sq_top[k - 1].item())
        target_sq = (M2 - sum_below) / k
        if target_sq < 0.0:
            continue
        tau_k = target_sq ** 0.5
        sigma_k = float(S_sorted[k - 1].item())
        # τ_k must lie in (σ_{k+1}, σ_k] for the "exactly top k modes
        # clipped" decomposition to be self-consistent. Without the lower
        # bound check, S.clamp_max(τ_k) clips MORE modes than k accounts
        # for and the resulting Frob norm is below M_cap (over-clip bug).
        sigma_k_plus_1 = float(S_sorted[k].item()) if k < n else 0.0
        if sigma_k_plus_1 <= tau_k <= sigma_k:
            tau = tau_k
            break
    if tau is None:
        return Xf  # degenerate; should not happen if norm_sq > M²
    S_clipped = S.clamp_max(tau)
    return (U * S_clipped.unsqueeze(0)) @ Vh


class AdamPolarProductLoRAClipGauge(Optimizer):
    """Per-block singular-value CLIP + Sylvester min-Frob gauge lift.

    Variationally clean implementation of theory.md's adjacent formulation:
        min ⟨u_A,ΔA⟩ + ⟨u_B,ΔB⟩ + (1/(2η))‖J‖_F²
        s.t.  ‖B ΔA‖_op ≤ τ_A,  ‖ΔB A‖_op ≤ τ_B
    Clip is the EXACT proximal map of each per-block subproblem (Frobenius
    projection onto the spectral-norm ball), unlike polar (which is the
    op-norm-only direction maximizer with magnitude wiped). The gauge
    lift recovers (ΔA, ΔB) from per-block (X^star, Y^star) on the
    min-Frob surface B^T ΔB = ΔA A^T.

    τ rule: R-equal — τ_A = ‖X_unc,A‖_F / √r per pair per step (no swept
    hyperparameter, no external reference). Spectrum probe at r=16/r=64
    shows σ_max / τ_R-equal ≈ 1.8–3.7 across pairs (peaky enough that
    clip does real work, with r=64 noticeably peakier).

    Skeleton (per pair, per step):
      1. Adam EMA on raw factor gradients → (u_A, u_B).
      2. Thin QR: B = Q_B R_B, A^T = Q_A R_AT.
      3. Whitening: ũ_A = R_B^{-T} u_A, ũ_B = u_B R_AT^{-1}.
      4. Per-block clip with R-equal τ:
            P_A = clip(ũ_A; τ_A), τ_A = ‖ũ_A‖_F / √r
            P_B = clip(ũ_B; τ_B), τ_B = ‖ũ_B‖_F / √r
      5. J_target = -lr · (Q_B P_A + P_B Q_A^T)
      6. Min-Frob lift via solve_sylvester (same as gauge variant):
            S_B K + K S_A = B^T J_target A^T
            ΔA = S_B^{-1} (B^T J_target − K A)
            ΔB = (J_target A^T − B K) S_A^{-1}
      7. Apply ΔA, ΔB. NO RMS-align — clip's τ supplies magnitude
         internally through ‖X_unc‖_F.

    PEFT init B=0 fallback. Same as AdamPolarProductLoRAGauge: at step 1
    or B near zero, do per-block independent whitened polar (with
    SB_half_inv = δ^{-1/2} I), no lift. From step 2 onward, full clip+lift.
    """

    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6,
                 eps=1e-8, ns_steps=5, adapter_name=None,
                 lora_plus_multiplier=1.0,
                 picard_iters=1,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.ns_steps = ns_steps  # only used in the B≈0 fallback
        self.lora_plus_multiplier = lora_plus_multiplier
        self.picard_iters = int(picard_iters)
        if self.picard_iters < 1:
            raise ValueError("picard_iters must be >= 1")
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
                raise ValueError("Gradients are required for AdamPolarProductLoRAClipGauge.")
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

            A_f = A.float()
            B_f = B.float()
            r = A_f.shape[0]

            b_norm = float(B_f.norm())
            use_fallback = b_norm < 1e-8

            if use_fallback:
                # Per-block independent whitened polar — same as the
                # AdamPolarProductLoRAGauge fallback. Picard iteration
                # is skipped at fallback (cross-coupling target undefined
                # when B≈0).
                S_A = spdify(A_f @ A_f.T, self.delta)
                SA_half_inv = _spd_inv_half(S_A, eps=self.delta, method="eigh")
                if b_norm < 1e-8:
                    SB_half_inv = (self.delta ** -0.5) * torch.eye(
                        r, dtype=torch.float32, device=A_f.device,
                    )
                else:
                    S_B = spdify(B_f.T @ B_f, self.delta)
                    SB_half_inv = _spd_inv_half(S_B, eps=self.delta, method="eigh")
                X_B = u_B @ SA_half_inv
                P_B = _newton_schulz(X_B, nsteps=self.ns_steps)
                geo_B = P_B @ SA_half_inv
                X_A = SB_half_inv @ u_A
                P_A = _newton_schulz(X_A, nsteps=self.ns_steps)
                geo_A = SB_half_inv @ P_A
                dA = -lr * geo_A
                dB = -lr * geo_B
                tau_A_log = float("nan")
                tau_B_log = float("nan")
            else:
                Q_B, R_B = torch.linalg.qr(B_f, mode='reduced')   # (m,r), (r,r)
                Q_A, R_AT = torch.linalg.qr(A_f.T, mode='reduced')  # (n,r), (r,r)

                # Whitened linear cost (constant across picard iters).
                # ũ_A = R_B^{-T} u_A; ũ_B = u_B R_AT^{-1} = u_B R_A^{-T}.
                L0_A = torch.linalg.solve_triangular(R_B.T, u_A, upper=False)         # (r, n)
                L0_B = torch.linalg.solve_triangular(R_AT.T, u_B.T, upper=False).T    # (m, r)

                S_B = spdify(B_f.T @ B_f, self.delta)
                S_A = spdify(A_f @ A_f.T, self.delta)

                # Picard inner iteration on the per-block prox + lift.
                # iter 0: T_A = T_B = 0 (no cross-coupling target).
                # iter k ≥ 1: T_A = -Q_B^T · ΔB_prev · A; T_B = -B · ΔA_prev · Q_A.
                # X_unc has sign and magnitude (lr factor) baked in via -lr·L0.
                # clip is sign-equivariant (clip(-X) = -clip(X)), so X^star
                # also carries sign and magnitude — the lift's J_target uses
                # X^star directly with NO additional -lr factor (contrast with
                # the polar variant where polar wipes magnitude).
                dA = torch.zeros_like(A_f)
                dB = torch.zeros_like(B_f)
                for k in range(self.picard_iters):
                    if k == 0:
                        T_A = torch.zeros_like(L0_A)
                        T_B = torch.zeros_like(L0_B)
                    else:
                        T_A = -Q_B.T @ dB @ A_f          # (r, n)
                        T_B = -B_f @ dA @ Q_A             # (m, r)

                    X_unc = T_A - lr * L0_A               # (r, n) — sign + mag inside
                    Y_unc = T_B - lr * L0_B               # (m, r)

                    # R-polar-equivalent cap: M = lr · √r (matches polar's
                    # natural step magnitude in the QR-whitened basis).
                    # The earlier R-AdamW-cap (M = lr·‖u_A‖_F) was ~√d_in
                    # times bigger than polar's effective magnitude,
                    # producing apparently "blowing up" steps at high lr.
                    # With M = lr·√r, clip's magnitude matches polar's and
                    # the only difference is spectrum shape (clip preserves
                    # sub-bulk modes; polar flattens to 1).
                    sqrt_r = float(r) ** 0.5
                    M_A = lr * sqrt_r
                    M_B = lr * sqrt_r
                    P_A = _clip_adamw_capped(X_unc, M_A)  # (r, n)
                    P_B = _clip_adamw_capped(Y_unc, M_B)  # (m, r)

                    # Lift: J_target = Q_B P_A + P_B Q_A^T (no -lr; sign in P).
                    # core_11 = (1,1) block of J_target in (Q_B, Q_A) basis
                    #         = P_A Q_A + Q_B^T P_B
                    core_11 = P_A @ Q_A + Q_B.T @ P_B    # (r, r)
                    RHS_K = R_B.T @ core_11 @ R_AT        # (r, r)
                    K = solve_sylvester(S_B, S_A, RHS_K)  # (r, r)

                    # ΔA = S_B^{-1} (B^T J_target − K A);
                    # B^T J_target = R_B^T (P_A + Q_B^T P_B Q_A^T)
                    BTJ = R_B.T @ (P_A + Q_B.T @ P_B @ Q_A.T)  # (r, n)
                    rhs_dA = BTJ - K @ A_f                      # (r, n)
                    dA = solve_spd(S_B, rhs_dA)                 # (r, n)

                    # ΔB = (J_target A^T − B K) S_A^{-1};
                    # J_target A^T = Q_B P_A Q_A R_AT + P_B R_AT
                    JAT = Q_B @ P_A @ Q_A @ R_AT + P_B @ R_AT  # (m, r)
                    rhs_dB = JAT - B_f @ K                      # (m, r)
                    dB = solve_spd(S_A, rhs_dB.T).T             # (m, r)

                # LoRA+ multiplier on ΔB only (post-picard).
                dB = self.lora_plus_multiplier * dB

                # Log τ_A, τ_B from the LAST picard iter for diagnostics.
                with torch.no_grad():
                    sv_A = torch.linalg.svdvals(X_unc.float())
                    sv_B = torch.linalg.svdvals(Y_unc.float())
                    tau_A_log = float(sv_A.pow(2).sum().sqrt() / max(1.0, sv_A.numel() ** 0.5))
                    tau_B_log = float(sv_B.pow(2).sum().sqrt() / max(1.0, sv_B.numel() ** 0.5))

            if self.log_diagnostics:
                if state['step'] % self.diagnostics_every == 0:
                    rec = {
                        "norm_dA": float(dA.detach().norm()),
                        "norm_dB": float(dB.detach().norm()),
                        "norm_A": float(A_f.norm()),
                        "norm_B": float(B_f.norm()),
                        "fallback": int(use_fallback),
                        "tau_A": tau_A_log,
                        "tau_B": tau_B_log,
                    }
                    BTdB = B_f.T @ dB.float()
                    dAAT = dA.float() @ A_f.T
                    gauge_resid = float((BTdB - dAAT).norm())
                    gauge_denom = float(dAAT.norm()) + 1e-30
                    rec["gauge_residual_abs"] = gauge_resid
                    rec["gauge_residual_rel"] = gauge_resid / gauge_denom

                    # B-spectrum trajectory diagnostic (stable rank, condition).
                    # Mirrors the diag in baseline AdamPolarProductLoRA so we
                    # can compare longitudinal B-evolution across variants.
                    try:
                        eigB = torch.linalg.eigvalsh(B_f.T @ B_f).clamp_min(0.0)
                        smax_B = float(eigB.max())
                        rec["stable_rank_B"] = float(eigB.sum() / (smax_B + 1e-30))
                        rec["nrank_B_1e2"] = int((eigB > 1e-2 * smax_B).sum())
                        # Frob norm of K (the gauge variable from the lift).
                        # Under min-Frob KKT: K = B^T dB = dA A^T. Predicted
                        # to be small when the gauge constraint absorbs the
                        # cross-coupling input.
                        rec["K_frob"] = float(BTdB.norm())
                    except torch._C._LinAlgError:
                        rec["stable_rank_B"] = float("nan")
                        rec["nrank_B_1e2"] = -1
                        rec["K_frob"] = float("nan")

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
        self.momentum_rank = momentum_rank

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


# ---------------------------------------------------------------------------
# Projected-quotient-polar core solver for joint operator-norm LoRA updates.
# Implements docs/notes/polar_product/theory.md (variants 1 + 2 of the
# Section 6 ladder). Variant 1 = pure projected polar on raw factor gradients;
# variant 2 = adds transported core-space momentum (Muon-style LoRA tangent
# optimizer). Factor-Adam is intentionally omitted — it breaks gradient
# compatibility (Section 6 of that doc).
# ---------------------------------------------------------------------------


def _build_active_core(A, B, G_A, G_B, delta=1e-6, sv_tol=1e-5):
    """Steps 1-5 of Section 7 of the core-solver doc.

    Builds the active tangent-gradient core Ĥ = [[C, E], [F, 0]] of size
    (r+t) × (r+s), together with the orthonormal bases (Q_L, U) and (Q_R, V)
    needed to lift core updates back to (ΔA, ΔB).

    Returns dict with Q_L, R_L, Q_R, R_R, U, V, C, E, F, H_hat, compat, s, t.
    """
    r = A.shape[0]
    n = A.shape[1]
    m = B.shape[0]

    # Step 1: thin QRs.
    Q_L, R_L = torch.linalg.qr(B, mode="reduced")          # (m, r), (r, r)
    Q_R, R_R_T = torch.linalg.qr(A.T, mode="reduced")       # (n, r), (r, r)
    R_R = R_R_T.T                                           # (r, r) lower triangular

    # Step 2: triangular solves for the accessible gradient core.
    L0 = torch.linalg.solve_triangular(R_L.T, G_A, upper=False)  # (r, n)
    R0 = torch.linalg.solve_triangular(R_R, G_B.T, upper=False).T  # (m, r)

    # Step 3: shared core block, averaged.
    C_L = L0 @ Q_R       # (r, r)
    C_R = Q_L.T @ R0     # (r, r)
    C = 0.5 * (C_L + C_R)
    compat_num = (C_L - C_R).norm().item()
    compat_den = C_L.norm().item() + C_R.norm().item() + 1e-30
    compat = compat_num / compat_den

    # Step 4: residuals.  Math-equivalent under grad-compat:
    #   L_perp = L0 - C Q_R^T = L0 (I - Q_R Q_R^T)
    #   R_perp = R0 - Q_L C   = (I - Q_L Q_L^T) R0
    # The right-hand forms don't depend on the (averaged) C, so they're
    # numerically robust to float32 noise in C — avoiding spurious tiny
    # singular components that would otherwise leak into the active rank.
    L_perp = L0 - (L0 @ Q_R) @ Q_R.T   # (r, n), rows in Q_R^⊥ to roundoff
    R_perp = R0 - Q_L @ (Q_L.T @ R0)   # (m, r), cols in Q_L^⊥ to roundoff

    # Step 5: thin SVDs. Structural rank is bounded: L_perp has rows in Q_R^⊥
    # (dim n - r) so rank(L_perp) ≤ min(r, n - r); similarly rank(R_perp) ≤
    # min(r, m - r). SVDs of rank-deficient matrices return arbitrary
    # singular vectors at noise-level singular values; those vectors are
    # not constrained to lie in the orthogonal complement, which breaks
    # the V^T Q_R = 0 / U^T Q_L = 0 identity that the lift-back gauge
    # depends on. Cap the active rank at the structural maximum.
    s_max = max(0, min(r, n - r))
    t_max = max(0, min(r, m - r))
    Ul, Sl, Vhl = torch.linalg.svd(L_perp, full_matrices=False)
    sv_max_l = float(Sl.max().item()) if Sl.numel() > 0 else 0.0
    s_raw = int((Sl > max(sv_tol * sv_max_l, sv_tol)).sum().item())
    s = min(s_raw, s_max)
    if s > 0:
        E = Ul[:, :s] * Sl[:s].unsqueeze(0)   # (r, s)
        V = Vhl[:s, :].T                       # (n, s)
    else:
        E = L_perp.new_zeros(r, 0)
        V = L_perp.new_zeros(n, 0)

    Ur, Sr, Vhr = torch.linalg.svd(R_perp, full_matrices=False)
    sv_max_r = float(Sr.max().item()) if Sr.numel() > 0 else 0.0
    t_raw = int((Sr > max(sv_tol * sv_max_r, sv_tol)).sum().item())
    t = min(t_raw, t_max)
    if t > 0:
        U = Ur[:, :t]                          # (m, t)
        F = Sr[:t].unsqueeze(1) * Vhr[:t, :]   # (t, r)
    else:
        U = R_perp.new_zeros(m, 0)
        F = R_perp.new_zeros(0, r)

    # Assemble Ĥ.
    if s > 0:
        top = torch.cat([C, E], dim=1)
    else:
        top = C
    if t > 0:
        if s > 0:
            bot = torch.cat([F, F.new_zeros(t, s)], dim=1)
        else:
            bot = F
        H_hat = torch.cat([top, bot], dim=0)
    else:
        H_hat = top

    return {
        "Q_L": Q_L, "R_L": R_L, "Q_R": Q_R, "R_R": R_R,
        "U": U, "V": V, "C": C, "E": E, "F": F,
        "H_hat": H_hat, "compat": compat, "s": s, "t": t,
    }


def _polar_via_svd(M, eps=1e-12, sv_tol=1e-5):
    """Compact polar factor of M via SVD.

    Returns (P, sv) where P = U_k @ Vh_k truncated at numerical rank
    (singular values above max(sv_tol·σ_max, 1e-12)). The full-matrix
    polar U @ Vh is arbitrary in the null-direction completions, which
    leaks into the projected polar's (22) block when M is rank-deficient
    (e.g. one-factor case where C=E=0). The compact polar avoids this
    leakage and matches the doc's "if Ĥ = 0, return zero" robustness.
    """
    if M.numel() == 0 or M.norm().item() < eps:
        return M.new_zeros(M.shape), M.new_zeros(0)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    sv_max = float(S.max().item())
    keep = S > max(sv_tol * sv_max, 1e-12)
    k = int(keep.sum().item())
    if k == 0:
        return M.new_zeros(M.shape), S
    return U[:, :k] @ Vh[:k, :], S


def _project_zero_22(M, r):
    """Π: zero the (rows ≥ r, cols ≥ r) block on a clone."""
    R = M.clone()
    if R.shape[0] > r and R.shape[1] > r:
        R[r:, r:] = 0.0
    return R


def _imbalance_gauge_shift(dA_0, dB_0, A, B, *, mode, rho=None, delta=1e-6):
    """Compute kernel-direction shift S that adjusts (ΔA_0, ΔB_0) toward an
    iLoRA-style imbalance target. See docs/notes/polar_product/theory.md
    "Recommended gauge: imbalance-preserving (iLoRA-style)".

    The kernel of J_t is {(SA, -BS) : S ∈ R^(r×r)}, so the lifts
        ΔA(S) = ΔA_0 + S A
        ΔB(S) = ΔB_0 - B S
    all give the same tangent J_t[ΔA, ΔB]. Use this freedom to control the
    imbalance functional I(A, B) = AA^T - ρ B^T B (ρ = r / d_out, after
    iLoRA Corollary 1: forward-variance preservation gives μ_1 = O(r/m)).

    Linearization in the lifted update:
        I(A + ΔA, B + ΔB) ≈ I_t + δI_0 + L(S)
    with
        δI_0 = ΔA_0 A^T + A ΔA_0^T - ρ(ΔB_0^T B + B^T ΔB_0)
        L(S) = S S_R + S_R S^T + ρ(S^T S_L + S_L S),  S_R = AA^T, S_L = B^T B.

    Restricting to symmetric S (the antisymmetric part is range-redundant
    when T = S_R + ρ S_L is positive definite), L(S) reduces to the
    symmetric Sylvester operator
        L_sym(S_s) = S_s T + T S_s,  T = S_R + ρ S_L.

    Modes:
      - "preserve": solve L_sym(S) = -δI_0 ⇒ post-step ΔI ≈ 0 (preserve I_t).
        Conservative; safe at PEFT init since δI_0 ≈ 0 there.
      - "restore":  solve L_sym(S) = -(I_t + δI_0) ⇒ post-step I ≈ 0.
        Aggressive; can blow up A at PEFT init (I_t = AA^T large) — boundary
        case must be handled separately by the caller (e.g., Case-2 fallback).

    Returns (dA_new, dB_new, S, gauge_diag_dict).
    """
    r = A.shape[0]
    m = B.shape[0]
    if rho is None:
        rho = r / max(m, 1)
    AAT = A @ A.T
    BTB = B.T @ B
    I_t = AAT - rho * BTB
    # δI_0 from the base lift (symmetric).
    dA0_AT = dA_0 @ A.T
    dB0T_B = dB_0.T @ B
    delta_I_0 = (dA0_AT + dA0_AT.T) - rho * (dB0T_B + dB0T_B.T)
    # T = AA^T + ρ B^T B is the operator whose Sylvester L_sym(S) = ST + TS
    # acts on symmetric S. Used by both the scalar and full-S branches and
    # by the linearized-residual diagnostic below.
    T = spdify(AAT + rho * BTB, eps=delta)
    if mode == "preserve-scalar":
        # Restrict S = s · I (single scalar). Closed form, no Sylvester solve.
        # δI(s) = δI_0 + 2 s M where M = AA^T + ρ B^T B.
        # Minimizing ‖δI(s)‖_F² gives s = -⟨δI_0, M⟩ / (2 ‖M‖_F²).
        # This is the smallest-complexity gauge fix per the doc's
        # "Recommended gauge: imbalance-preserving" guidance — fixes only
        # the scalar rescaling gauge, no hyperparameters.
        M = AAT + rho * BTB
        denom = 2.0 * float((M * M).sum().item()) + 1e-30
        s_scalar = -float((delta_I_0 * M).sum().item()) / denom
        S = s_scalar * torch.eye(r, device=A.device, dtype=A.dtype)
        dA_new = dA_0 + s_scalar * A
        dB_new = dB_0 - s_scalar * B
    elif mode == "balanced-scalar":
        # S = s · I targeting ‖dA(s)‖_F = ‖dB(s)‖_F (matches AdamW / hybrid
        # Picard empirical ratio of ~1, NOT iLoRA's √(m/r) — Picard wins at
        # ratio 1, so balanced-Frobenius is the empirically-validated target).
        # Solve f(s) = ‖dA_0 + sA‖² − ‖dB_0 − sB‖² = 0 for s. Quadratic:
        # f(s) = (‖A‖² − ‖B‖²) s² + 2(⟨dA_0,A⟩ + ⟨dB_0,B⟩) s + (‖dA_0‖² − ‖dB_0‖²).
        # Two roots; pick min |s| (smallest kernel motion).
        nA2 = float((A * A).sum().item())
        nB2 = float((B * B).sum().item())
        ndA02 = float((dA_0 * dA_0).sum().item())
        ndB02 = float((dB_0 * dB_0).sum().item())
        cross = float((dA_0 * A).sum().item()) + float((dB_0 * B).sum().item())
        a = nA2 - nB2
        b = 2.0 * cross
        c = ndA02 - ndB02
        if abs(a) < 1e-30:
            s_scalar = -c / (b + 1e-30) if abs(b) > 1e-30 else 0.0
        else:
            disc = b * b - 4.0 * a * c
            if disc < 0:
                # No real solution; fall back to s minimizing |f(s)| via vertex.
                s_scalar = -b / (2.0 * a)
            else:
                sd = disc ** 0.5
                s1 = (-b + sd) / (2.0 * a)
                s2 = (-b - sd) / (2.0 * a)
                s_scalar = s1 if abs(s1) < abs(s2) else s2
        S = s_scalar * torch.eye(r, device=A.device, dtype=A.dtype)
        dA_new = dA_0 + s_scalar * A
        dB_new = dB_0 - s_scalar * B
    else:
        if mode == "preserve":
            target = -delta_I_0
        elif mode == "restore":
            target = -(I_t + delta_I_0)
        else:
            raise ValueError(f"Unknown imbalance gauge mode '{mode}'.")
        # Symmetric Sylvester: S T + T S = target.
        S = solve_sylvester(T, T, target)
        # Symmetrize numerically (the analytic solution is symmetric).
        S = 0.5 * (S + S.T)
        dA_new = dA_0 + S @ A
        dB_new = dB_0 - B @ S
    # Diagnostics: imbalance residual before/after gauge shift (linearized).
    nan = float("nan")
    den_pre = float(I_t.norm().item()) + 1e-30
    pre = float(I_t.norm().item()) / (float(AAT.norm().item()) + rho * float(BTB.norm().item()) + 1e-30)
    # Linearized post-step imbalance (we don't apply the step here; just for telemetry).
    L_S = S @ T + T @ S
    post_lin_residual = (I_t + delta_I_0 + L_S).norm().item()
    base_lift_residual = (I_t + delta_I_0).norm().item()
    gauge_diag = {
        "gauge_S_norm": float(S.norm().item()),
        "gauge_dA_shift_norm": float((S @ A).norm().item()),
        "gauge_dB_shift_norm": float((B @ S).norm().item()),
        "imbalance_residual_lin_pre": float(base_lift_residual),
        "imbalance_residual_lin_post": float(post_lin_residual),
    }
    return dA_new, dB_new, S, gauge_diag


def _polar_coupled_core_lift(
    core_obj, bases, lr, r, *,
    core_scale="squared_penalty",
    core_norm="operator",
    delta=1e-6,
    H_hat_for_align=None,
    gauge="min-frobenius",
    A_for_gauge=None,
    B_for_gauge=None,
    rho=None,
    pre_polar_normalize=None,
):
    """Sections 2 + 4: polar of `core_obj` (Ĥ for variant 1, M_hat for variant
    2), project (22), renormalize, scale, lift back via Sylvester gauge.
    Returns (dA, dB, certs).
    """
    Q_L, R_L = bases["Q_L"], bases["R_L"]
    Q_R, R_R = bases["Q_R"], bases["R_R"]
    U, V = bases["U"], bases["V"]
    s, t = bases["s"], bases["t"]
    n = Q_R.shape[0]
    m = Q_L.shape[0]

    nan = float("nan")
    if core_obj.numel() == 0 or core_obj.norm().item() < 1e-20:
        dA = R_L.new_zeros(r, n)
        dB = R_L.new_zeros(m, r)
        return dA, dB, {"gamma": nan, "nuc": 0.0, "LB": 0.0, "UB": 0.0,
                        "relgap": nan, "align_inst": 0.0,
                        "s_active": s, "t_active": t}

    # Optional pre-polar elementwise normalization (rung 5-lite, no EMA).
    # `pre_polar_normalize="sign"` divides core elementwise by |.|+ε before
    # polar — gives Adam-like per-coord adaptivity in core space without
    # the basis-rotation EMA-transport issue. See plan Phase 2 (B).
    if pre_polar_normalize == "sign":
        eps_norm = 1e-6 * float(core_obj.abs().max().clamp_min(1e-30).item())
        core_obj = core_obj / (core_obj.abs() + eps_norm)
    elif pre_polar_normalize is not None and pre_polar_normalize != "none":
        raise ValueError(f"Unknown pre_polar_normalize '{pre_polar_normalize}'.")

    if core_norm == "operator":
        P, sv = _polar_via_svd(core_obj)
        nuc = float(sv.sum().item())
        R = _project_zero_22(P, r)
        gamma_t = torch.linalg.svdvals(R)
        gamma = float(gamma_t[0].item()) if gamma_t.numel() > 0 else 0.0
        if gamma < 1e-12:
            dA = R_L.new_zeros(r, n)
            dB = R_L.new_zeros(m, r)
            return dA, dB, {"gamma": nan, "nuc": nuc, "LB": 0.0, "UB": nuc,
                            "relgap": nan, "align_inst": 0.0,
                            "s_active": s, "t_active": t}
        Z_plus = R / gamma
        if core_scale == "squared_penalty":
            tau_hat = nuc / gamma
            Z_upd = (-lr * tau_hat) * Z_plus
        elif core_scale == "constrained":
            Z_upd = (-lr) * Z_plus
        else:
            raise ValueError(f"Unknown core_scale '{core_scale}'.")
        LB = nuc / gamma
        UB = nuc
        relgap = 1.0 - 1.0 / gamma
        if H_hat_for_align is not None:
            align_inst = float((H_hat_for_align * Z_plus).sum().item())
        else:
            align_inst = nan
    elif core_norm == "frobenius":
        Z_upd = (-lr) * core_obj
        gamma = nan
        nuc = float(torch.linalg.svdvals(core_obj).sum().item())
        LB = nan
        UB = nuc
        relgap = nan
        align_inst = nan
    else:
        raise ValueError(f"Unknown core_norm '{core_norm}'.")

    # Steps 9-10: lift back.
    X = Z_upd[:r, :r]
    Y = Z_upd[:r, r:] if s > 0 else Z_upd.new_zeros(r, 0)
    W = Z_upd[r:, :r] if t > 0 else Z_upd.new_zeros(0, r)

    S_L = spdify(R_L.T @ R_L, delta)
    S_R = spdify(R_R @ R_R.T, delta)

    RHS_K = R_L.T @ X @ R_R.T
    K = solve_sylvester(S_L, S_R, RHS_K)

    RHS_A = (R_L.T @ X - K @ R_R) @ Q_R.T
    if s > 0:
        RHS_A = RHS_A + (R_L.T @ Y) @ V.T
    dA = solve_spd(S_L, RHS_A)

    RHS_B = Q_L @ (X @ R_R.T - R_L @ K)
    if t > 0:
        RHS_B = RHS_B + (U @ W) @ R_R.T
    dB = solve_spd(S_R, RHS_B.T).T

    # Gauge shift (Section "Open sub-problem" of polar_product/theory.md).
    # min-frobenius is the base lift above; imbalance modes shift through the
    # kernel of J_t to control the iLoRA imbalance functional.
    gauge_diag = {}
    if gauge in ("imbalance-preserve", "imbalance-restore",
                 "imbalance-preserve-scalar", "balanced-scalar"):
        if A_for_gauge is None or B_for_gauge is None:
            raise ValueError(f"gauge '{gauge}' requires A_for_gauge and B_for_gauge.")
        mode = {
            "imbalance-preserve": "preserve",
            "imbalance-restore": "restore",
            "imbalance-preserve-scalar": "preserve-scalar",
            "balanced-scalar": "balanced-scalar",
        }[gauge]
        dA, dB, _S, gauge_diag = _imbalance_gauge_shift(
            dA, dB, A_for_gauge, B_for_gauge, mode=mode, rho=rho, delta=delta,
        )
    elif gauge != "min-frobenius":
        raise ValueError(f"Unknown gauge '{gauge}'.")

    certs = {
        "gamma": float(gamma) if gamma == gamma else nan,
        "nuc": nuc,
        "LB": LB if LB == LB else nan,
        "UB": UB,
        "relgap": relgap if relgap == relgap else nan,
        "align_inst": align_inst,
        "s_active": s,
        "t_active": t,
        "gauge": gauge,
    }
    certs.update(gauge_diag)
    return dA, dB, certs


def _rebalance_state(A, B, rho=None, eps=1e-6):
    """Post-step state-gauge reparameterization: choose R such that after
        A ← R^{-1} A,   B ← B R
    we have A A^T ≈ ρ B^T B (the iLoRA-stable invariant; ρ = r / d_out).

    Crucially: B A is invariant under this transformation:
        (B R)(R^{-1} A) = B A
    so the model's adapter contribution is unchanged. Only the *coordinates*
    of the factor representation change. This avoids inflating the dropped
    second-order term Δ B Δ A (which an update-gauge kernel motion would).

    Math: with S_A = A A^T, S_B = B^T B, want
        R^{-1} S_A R^{-T} = ρ R^T S_B R.
    Let P = R R^T (symmetric PSD). Then S_A = ρ P S_B P, with SPD solution
        P = ρ^{-1/2} S_B^{-1/2} (S_B^{1/2} S_A S_B^{1/2})^{1/2} S_B^{-1/2}
    and R = P^{1/2} (or any factor with R R^T = P).

    Boundary case: if S_B is essentially singular (B ≈ 0), rebalancing is
    not well-defined — the caller should ensure B has positive rank before
    invoking. Returns (A_new, B_new) without mutating inputs.
    """
    r = A.shape[0]
    m = B.shape[0]
    if rho is None:
        rho = r / max(m, 1)
    A_f = A.float()
    B_f = B.float()
    # Skip if B is too small for rebalancing to be well-defined.
    if _factor_essentially_zero(B_f):
        return A, B  # unchanged
    S_A = A_f @ A_f.T
    S_B = B_f.T @ B_f
    # Damp Grams for numerical safety. Scale ε to S_B's magnitude — for tiny
    # B the absolute eps would dominate; scale-relative keeps the conditioning
    # under control without becoming a tunable hyperparameter at the
    # algorithm level (it's just float32 numerical safety).
    sB_scale = float(S_B.diag().mean().clamp_min(1e-30).item())
    eps_B = max(eps * sB_scale, 1e-12)
    S_A_d = spdify(S_A, eps * float(S_A.diag().mean().clamp_min(1e-30).item()))
    S_B_d = spdify(S_B, eps_B)
    # S_B^{1/2} and S_B^{-1/2} via eigh.
    evals_B, Q_B = torch.linalg.eigh(S_B_d)
    evals_B = evals_B.clamp_min(eps_B)
    SB_half = Q_B @ torch.diag(evals_B.sqrt()) @ Q_B.T
    SB_neg_half = Q_B @ torch.diag(evals_B.rsqrt()) @ Q_B.T
    # Inner: SB^{1/2} S_A SB^{1/2}, then square root.
    inner = SB_half @ S_A_d @ SB_half
    inner = 0.5 * (inner + inner.T)
    evals_in, Q_in = torch.linalg.eigh(inner)
    evals_in = evals_in.clamp_min(0.0)
    inner_half = Q_in @ torch.diag(evals_in.sqrt()) @ Q_in.T
    # P = ρ^{-1/2} · SB^{-1/2} · inner_half · SB^{-1/2}.
    P = (rho ** -0.5) * (SB_neg_half @ inner_half @ SB_neg_half)
    P = 0.5 * (P + P.T)  # symmetrize
    # R = Cholesky factor of P (R R^T = P).
    P_reg = P + (eps * float(P.diag().mean().clamp_min(1e-30).item())) * torch.eye(r, device=P.device, dtype=P.dtype)
    try:
        R = torch.linalg.cholesky(P_reg)
    except torch._C._LinAlgError:
        # Fallback: matrix square root via eigh.
        evals_P, Q_P = torch.linalg.eigh(P_reg)
        evals_P = evals_P.clamp_min(eps_B)
        R = Q_P @ torch.diag(evals_P.sqrt()) @ Q_P.T
    # Apply: B' = B R,  A' = R^{-1} A.
    B_new = B_f @ R
    A_new = torch.linalg.solve_triangular(R, A_f, upper=False) if R.is_contiguous() else torch.linalg.solve(R, A_f)
    # Cast back to original dtypes / devices.
    return A_new.to(dtype=A.dtype, device=A.device), B_new.to(dtype=B.dtype, device=B.device)


def _factor_essentially_zero(X, eps=1e-12):
    """Detect the *initialization-zero* boundary case (PEFT default zeros B).

    NOT a general "poorly-conditioned" check — that is what the small δI
    regularization in `spdify` already handles inside the joint solver. We
    only branch out of the joint solver at the genuine boundary: a factor
    that is literally zero (or near machine epsilon thereof). Mid-training
    spectral collapse, where a real singular value drifts toward zero, is
    NOT this case — the ½-approximation guarantee for the joint Case 3
    problem still holds for those steps; only the QR for B = 0 is undefined.
    """
    if X.numel() == 0:
        return True
    return float(X.norm().item()) < eps * (X.numel() ** 0.5)


def _zero_B_fallback(A, B, G_A, G_B, lr, *, delta=1e-6, core_scale="squared_penalty"):
    """Bootstrap fallback when B is (near-)zero / rank-deficient (PEFT's
    default LoRA init), violating the Section 1 standing assumption.

    At B = 0: the tangent constraint ‖B ΔA + ΔB A‖_2 ≤ λ collapses to
    ‖ΔB A‖_2 ≤ λ, and the linear cost reduces to ⟨G_B, ΔB⟩ (since
    G_A = B^T (·) = 0). This is exactly Case 2 (one-factor restriction)
    of polar_product/theory.md, with closed-form
        ΔB = -λ · polar(G_B U_R Σ_R^{-1}) · Σ_R^{-1} U_R^T,
    where A = U_R Σ_R V_R^T (compact SVD). Plain Muon polar(G_B) misses
    the A-spectrum preconditioner; this form preserves it.

    Symmetric case (A near-zero, B full-rank) is unusual under standard
    PEFT init but handled by the (A ↔ B^T) transpose symmetry of the
    construction — we apply Case 2 on B if A is degenerate, else
    Case 2 on A.
    """
    A_f = A.float()
    B_f = B.float()
    GA_f = G_A.float()
    GB_f = G_B.float()
    nan = float("nan")
    # Pick which side is the boundary case.
    A_def = _factor_essentially_zero(A_f)
    B_def = _factor_essentially_zero(B_f)

    if B_def and not A_def:
        # B = 0, A full-rank: Case 2 on B.  A = U_R Σ_R V_R^T (compact SVD).
        U_R, S_R, _ = torch.linalg.svd(A_f, full_matrices=False)
        S_clamp = S_R.clamp_min(delta)
        X = (GB_f @ U_R) / S_clamp.unsqueeze(0)        # G_B U_R Σ_R^{-1}, (m, r)
        Up, _, Vhp = torch.linalg.svd(X, full_matrices=False)
        P = Up @ Vhp                                    # polar(G_B U_R Σ_R^{-1})
        # ΔB = -lr · P · Σ_R^{-1} · U_R^T
        scale = (-lr) if core_scale == "constrained" else (-lr * float(X.norm().item() + 1e-30))
        # squared_penalty form: scale by ‖X‖_*. Approximate via Frobenius
        # is not exact; use nuclear via svdvals.
        if core_scale == "squared_penalty":
            scale = -lr * float(torch.linalg.svdvals(X).sum().item())
        dB = scale * (P / S_clamp.unsqueeze(0)) @ U_R.T
        dA = A_f.new_zeros(A_f.shape)
    elif A_def and not B_def:
        # A = 0, B full-rank: Case 2 on A by symmetry. B = U_L Σ_L V_L^T.
        U_L, S_L, Vh_L = torch.linalg.svd(B_f, full_matrices=False)
        S_clamp = S_L.clamp_min(delta)
        # ΔA = -lr · Σ_L^{-1} V_L^T · polar(V_L Σ_L^{-1} V_L^T G_A)
        # Equivalently use (B^T B)^{-1/2} preconditioner symmetric to Case 2.
        # Y = V_L Σ_L^{-1} V_L^T G_A, (r, n).
        VL = Vh_L.T
        Y = VL @ ((VL.T @ GA_f) / S_clamp.unsqueeze(1))
        Up, _, Vhp = torch.linalg.svd(Y, full_matrices=False)
        P = Up @ Vhp
        if core_scale == "squared_penalty":
            scale = -lr * float(torch.linalg.svdvals(Y).sum().item())
        else:
            scale = -lr
        dA = scale * VL @ ((VL.T @ P) / S_clamp.unsqueeze(1))
        dB = B_f.new_zeros(B_f.shape)
    else:
        # Both degenerate (A=0 AND B=0) or other pathological case.
        # No useful gradient signal — leave both unchanged this step.
        dA = A_f.new_zeros(A_f.shape)
        dB = B_f.new_zeros(B_f.shape)

    certs = {"gamma": nan, "nuc": float(GB_f.norm().item()),
             "LB": nan, "UB": nan, "relgap": nan,
             "align_inst": nan, "s_active": 0, "t_active": 0,
             "compat": nan, "fallback": 1.0}
    # Attach factor diagnostics in fallback path too — the gauge-imbalance
    # story starts at step 0, and tracking ‖B‖_F's bootstrap from zero is
    # important for the open gauge sub-problem.
    _attach_factor_diagnostics(certs, A_f, B_f, None, dA, dB)
    return dA, dB, certs, None


def _polar_coupled_core_step(
    A, B, G_A, G_B, lr, *,
    delta=1e-6,
    core_scale="squared_penalty",
    core_norm="operator",
    sv_tol=1e-5,
    gauge="min-frobenius",
    rho=None,
    pre_polar_normalize=None,
):
    """Variant 1 entry point: build active core, polar+lift on Ĥ.
    Returns (dA, dB, certs, bases) — bases enables variant-2 reuse.
    """
    A_f = A.float()
    B_f = B.float()
    GA_f = G_A.float()
    GB_f = G_B.float()
    if _factor_essentially_zero(B_f) or _factor_essentially_zero(A_f):
        return _zero_B_fallback(A, B, G_A, G_B, lr, delta=delta, core_scale=core_scale)
    bases = _build_active_core(A_f, B_f, GA_f, GB_f, delta=delta, sv_tol=sv_tol)
    H_hat = bases["H_hat"]
    r = A_f.shape[0]
    dA, dB, certs = _polar_coupled_core_lift(
        H_hat, bases, lr, r,
        core_scale=core_scale, core_norm=core_norm, delta=delta,
        H_hat_for_align=H_hat,
        gauge=gauge, A_for_gauge=A_f, B_for_gauge=B_f, rho=rho,
        pre_polar_normalize=pre_polar_normalize,
    )
    certs["compat"] = bases["compat"]
    _attach_factor_diagnostics(certs, A_f, B_f, bases, dA, dB)
    return dA, dB, certs, bases


def _attach_factor_diagnostics(certs, A_f, B_f, bases, dA, dB):
    """Add factor-shape diagnostics to certs dict for gauge-imbalance analysis.

    See docs/notes/polar_product/theory.md "Open sub-problem: gauge choice
    under asymmetric LoRA initialization" — the empirical anchors for any
    gauge candidate are:
      - imbalance residual ‖AA^T − ρ B^T B‖_F / (‖AA^T‖_F + ρ‖B^T B‖_F + ε)
        with ρ = r/m (iLoRA's invariant; primary metric)
      - ratio_dA_dB = ‖dA‖_F / ‖dB‖_F (secondary; min-Frobenius gives 50-100)
      - σ_min(A), σ_min(B), ‖A‖_F, ‖B‖_F trajectories (B-growth check)
    """
    try:
        sv_A = torch.linalg.svdvals(A_f)
        sv_B = torch.linalg.svdvals(B_f)
        nA = float(A_f.norm().item())
        nB = float(B_f.norm().item())
        sigmin_A = float(sv_A.min().item()) if sv_A.numel() > 0 else float('nan')
        sigmin_B = float(sv_B.min().item()) if sv_B.numel() > 0 else float('nan')
        sigmax_A = float(sv_A.max().item()) if sv_A.numel() > 0 else float('nan')
        sigmax_B = float(sv_B.max().item()) if sv_B.numel() > 0 else float('nan')
        certs["norm_A"] = nA
        certs["norm_B"] = nB
        certs["sigmin_A"] = sigmin_A
        certs["sigmin_B"] = sigmin_B
        certs["sigmin_BA_ratio"] = (sigmin_B / sigmin_A) if sigmin_A > 1e-30 else float('nan')
        certs["sigmax_BA_ratio"] = (sigmax_B / sigmax_A) if sigmax_A > 1e-30 else float('nan')
        certs["cond_A"] = (sigmax_A / sigmin_A) if sigmin_A > 1e-30 else float('nan')
        certs["cond_B"] = (sigmax_B / sigmin_B) if sigmin_B > 1e-30 else float('nan')
        nda = float(dA.norm().item())
        ndb = float(dB.norm().item())
        certs["ratio_dA_dB"] = (nda / ndb) if ndb > 1e-30 else float('nan')
        # iLoRA imbalance residual (primary diagnostic; Gu et al. NeurIPS 2024,
        # Corollary 1: μ_1 = O(r/m) where m = d_out).
        # I = AA^T - ρ B^T B, ρ = r/m.
        r, n = A_f.shape
        m = B_f.shape[0]
        rho = r / m if m > 0 else 1.0
        AAT = A_f @ A_f.T          # (r, r)
        BTB = B_f.T @ B_f          # (r, r)
        I = AAT - rho * BTB
        num = float(I.norm().item())
        den = float(AAT.norm().item()) + rho * float(BTB.norm().item()) + 1e-30
        certs["imbalance_residual"] = num / den
        certs["rho_target"] = rho
    except (torch._C._LinAlgError, RuntimeError):
        for k in ("norm_A", "norm_B", "sigmin_A", "sigmin_B",
                  "sigmin_BA_ratio", "sigmax_BA_ratio", "cond_A", "cond_B",
                  "ratio_dA_dB", "imbalance_residual", "rho_target"):
            certs[k] = float('nan')


class PolarCoupledCoreLoRA(Optimizer):
    """Variant 1 of docs/notes/polar_product/theory.md Section 6 ladder.

    Pure projected-quotient-polar update on raw factor gradients. No Adam,
    no momentum: this is the clean variational baseline for the joint
    operator-norm LoRA problem (Case 3 of polar_product/theory.md).

    Certificates logged: γ ∈ [1, 2], ‖Ĥ‖_*, LB = ‖Ĥ‖_*/γ, UB = ‖Ĥ‖_*,
    relgap = 1 − 1/γ ∈ [0, 0.5], compat (gradient-compatibility violation;
    near machine epsilon for raw factor gradients).
    """

    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None,
                 core_scale="squared_penalty",
                 gauge="min-frobenius", rho=None,
                 state_rebalance=False, rebalance_every=1,
                 pre_polar_normalize=None,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.core_scale = core_scale
        self.gauge = gauge
        self.rho = rho
        self.state_rebalance = state_rebalance
        self.rebalance_every = rebalance_every
        self.pre_polar_normalize = pre_polar_normalize
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.pair_state = {i: {"step": 0} for i in range(len(pairs))}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for PolarCoupledCoreLoRA update.")
            state = self.pair_state[i]
            state["step"] += 1

            dA, dB, certs, _ = _polar_coupled_core_step(
                A, B, A.grad, B.grad, lr,
                delta=self.delta,
                core_scale=self.core_scale,
                core_norm="operator",
                gauge=self.gauge,
                rho=self.rho,
                pre_polar_normalize=self.pre_polar_normalize,
            )
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

            # Post-step state-gauge rebalance: (A, B) ← (R^{-1} A, B R) such
            # that A A^T ≈ ρ B^T B. Preserves B A exactly. See
            # docs/notes/polar_product/theory.md "Open sub-problem" section.
            if (self.state_rebalance
                    and state["step"] % self.rebalance_every == 0
                    and not _factor_essentially_zero(B.detach().float())):
                A_new, B_new = _rebalance_state(A.detach(), B.detach(),
                                                 rho=self.rho, eps=self.delta)
                A.data.copy_(A_new)
                B.data.copy_(B_new)

            if self.log_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                diag_records.append(rec)

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class PolarCoupledCoreFactorAdamLoRA(Optimizer):
    """Rung-6 ablation: factor-space Adam preconditioning + projected-quotient-polar core solver.

    Picard's adam-polar-product-lora-coupled runs Adam-EMA on factor gradients
    G_A, G_B before its spectral-product step. This class does the SAME factor-Adam
    preconditioning but feeds u_A, u_B into our exact-KKT projected-quotient-polar
    core solver instead of Picard iteration.

    Tests whether (a) factor-space adaptivity is the ingredient Picard wins on,
    and (b) given factor-Adam, our solver outperforms Picard's iteration.

    Compatibility note: factor-Adam breaks the gradient-compatibility identity
    B^T G_B = G_A A^T (u_A and u_B are normalized in different feature frames).
    The core construction's (C_L + C_R)/2 averaging projects back to the
    compatible subspace; cost shows up as compat > 0 in diagnostics. Per
    docs/notes/polar_product/theory.md Section 6 this is "ablation only";
    we run the ablation because Picard's empirical win contradicts the doc's
    a priori reasoning.
    """

    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None,
                 betas=(0.9, 0.999), eps=1e-8,
                 core_scale="squared_penalty",
                 gauge="min-frobenius", rho=None,
                 state_rebalance=False, rebalance_every=1,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.core_scale = core_scale
        self.gauge = gauge
        self.rho = rho
        self.state_rebalance = state_rebalance
        self.rebalance_every = rebalance_every
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            self.pair_state[i] = {
                "step": 0,
                "m_A": torch.zeros_like(A, dtype=torch.float32),
                "v_A": torch.zeros_like(A, dtype=torch.float32),
                "m_B": torch.zeros_like(B, dtype=torch.float32),
                "v_B": torch.zeros_like(B, dtype=torch.float32),
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
                raise ValueError("Gradients are required for PolarCoupledCoreFactorAdamLoRA update.")
            state = self.pair_state[i]
            state["step"] += 1
            t = state["step"]

            gA = A.grad.float()
            gB = B.grad.float()
            state["m_A"].mul_(self.beta1).add_(gA, alpha=1 - self.beta1)
            state["v_A"].mul_(self.beta2).addcmul_(gA, gA, value=1 - self.beta2)
            state["m_B"].mul_(self.beta1).add_(gB, alpha=1 - self.beta1)
            state["v_B"].mul_(self.beta2).addcmul_(gB, gB, value=1 - self.beta2)

            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
            m_hat_A = state["m_A"] / bc1
            v_hat_A = state["v_A"] / bc2
            m_hat_B = state["m_B"] / bc1
            v_hat_B = state["v_B"] / bc2
            u_A = m_hat_A / (v_hat_A.sqrt() + self.eps)
            u_B = m_hat_B / (v_hat_B.sqrt() + self.eps)

            dA, dB, certs, _ = _polar_coupled_core_step(
                A, B, u_A, u_B, lr,
                delta=self.delta,
                core_scale=self.core_scale,
                core_norm="operator",
                gauge=self.gauge,
                rho=self.rho,
                pre_polar_normalize=None,
            )
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

            if (self.state_rebalance
                    and t % self.rebalance_every == 0
                    and not _factor_essentially_zero(B.detach().float())):
                A_new, B_new = _rebalance_state(A.detach(), B.detach(),
                                                 rho=self.rho, eps=self.delta)
                A.data.copy_(A_new)
                B.data.copy_(B_new)

            if self.log_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                rec["norm_gA"] = float(gA.norm().item())
                rec["norm_gB"] = float(gB.norm().item())
                rec["norm_uA"] = float(u_A.norm().item())
                rec["norm_uB"] = float(u_B.norm().item())
                diag_records.append(rec)

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class MuonCoupledCoreLoRA(Optimizer):
    """Variant 2 of the Section 6 ladder: variant 1 + transported core
    momentum, mirroring canonical Muon (~/modded-nanogpt/train_gpt.py:170+).

    Each step:
      1. Build the active core Ĥ_t and bases (Q_L, U), (Q_R, V).
      2. Transport the previous EMA M_{t-1} onto current bases:
             M_transported = T_L · M_{t-1} · T_R^T
         where T_L = U_cur^T U_prev, T_R = V_cur^T V_prev.
      3. EMA update (no bias correction — matches canonical Muon):
             M_t = β · Π(M_transported) + (1 − β) · Ĥ_t
      4. Nesterov lookahead in core space:
             M_step = (1 − β) · Ĥ_t + β · M_t
      5. Run projected quotient polar on M_step, lift back via Sylvester.
      6. Save M_t (EMA, NOT M_step) for next-step transport.

    The non-bias-corrected EMA gives a small step-1 magnitude
    (~(1−β)·grad with the lookahead, ~0.10·grad at β=0.95), which is the
    Muon-typical "build up momentum gradually" behavior. Adam-style bc
    would inflate step 1 by 1/(1−β)=20×, which causes divergence at
    moderate lr — empirically observed and removed here.
    """

    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None,
                 beta1=0.95,
                 core_scale="squared_penalty",
                 gauge="min-frobenius", rho=None,
                 state_rebalance=False, rebalance_every=1,
                 pre_polar_normalize=None,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.beta1 = beta1
        self.core_scale = core_scale
        self.gauge = gauge
        self.rho = rho
        self.state_rebalance = state_rebalance
        self.rebalance_every = rebalance_every
        self.pre_polar_normalize = pre_polar_normalize
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.pair_state = {i: {"step": 0,
                               "M_prev": None,
                               "U_prev": None,
                               "V_prev": None}
                           for i in range(len(pairs))}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for MuonCoupledCoreLoRA update.")
            state = self.pair_state[i]
            state["step"] += 1
            t_step = state["step"]

            A_f = A.float()
            B_f = B.float()
            GA_f = A.grad.float()
            GB_f = B.grad.float()
            # Initialization-zero boundary case (Section 1 standing
            # assumption); see _polar_coupled_core_step for context.
            if _factor_essentially_zero(B_f) or _factor_essentially_zero(A_f):
                dA_fb, dB_fb, certs_fb, _ = _zero_B_fallback(
                    A, B, A.grad, B.grad, lr, delta=self.delta,
                    core_scale=self.core_scale,
                )
                A.add_(dA_fb.to(dtype=A.dtype, device=A.device))
                B.add_(dB_fb.to(dtype=B.dtype, device=B.device))
                A.grad.zero_()
                B.grad.zero_()
                if self.log_diagnostics:
                    rec = {k: float(v) for k, v in certs_fb.items() if isinstance(v, (int, float))}
                    rec["norm_dA"] = float(dA_fb.norm().item())
                    rec["norm_dB"] = float(dB_fb.norm().item())
                    rec["transport_residual"] = float("nan")
                    rec["align_mom"] = float("nan")
                    diag_records.append(rec)
                continue
            bases = _build_active_core(A_f, B_f, GA_f, GB_f, delta=self.delta)
            H_hat = bases["H_hat"]
            r = A_f.shape[0]
            tt = bases["t"]
            ss = bases["s"]
            U_cur = torch.cat([bases["Q_L"], bases["U"]], dim=1) if tt > 0 else bases["Q_L"]
            V_cur = torch.cat([bases["Q_R"], bases["V"]], dim=1) if ss > 0 else bases["Q_R"]

            # EMA in core space, mirroring canonical Muon (no bias correction).
            transport_residual = float("nan")
            if state["M_prev"] is None or t_step == 1:
                # First-ever non-fallback step: M_prev = 0 ⇒ M_t = (1-β)·H_hat.
                M_t = (1.0 - self.beta1) * H_hat
            else:
                T_L = U_cur.T @ state["U_prev"]
                T_R = V_cur.T @ state["V_prev"]
                M_transported = T_L @ state["M_prev"] @ T_R.T
                M_transported = _project_zero_22(M_transported, r)
                M_t = self.beta1 * M_transported + (1.0 - self.beta1) * H_hat
                den = M_transported.norm().item() + 1e-30
                transport_residual = float((M_t - M_transported).norm().item() / den)

            # Nesterov lookahead in core space (canonical Muon: g = (1-β)·grad + β·mb).
            M_step = (1.0 - self.beta1) * H_hat + self.beta1 * M_t

            dA, dB, certs = _polar_coupled_core_lift(
                M_step, bases, lr, r,
                core_scale=self.core_scale, core_norm="operator",
                delta=self.delta, H_hat_for_align=H_hat,
                gauge=self.gauge, A_for_gauge=A_f, B_for_gauge=B_f, rho=self.rho,
                pre_polar_normalize=self.pre_polar_normalize,
            )
            certs["compat"] = bases["compat"]
            _attach_factor_diagnostics(certs, A_f, B_f, bases, dA, dB)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

            # Save core EMA state for next-step transport. May be invalidated
            # by post-step rebalance below.
            state["M_prev"] = M_t
            state["U_prev"] = U_cur
            state["V_prev"] = V_cur

            # Post-step state-gauge rebalance (variant 2: bases change
            # discontinuously, so reset transported core EMA after rebalance).
            if (self.state_rebalance
                    and state["step"] % self.rebalance_every == 0
                    and not _factor_essentially_zero(B.detach().float())):
                A_new, B_new = _rebalance_state(A.detach(), B.detach(),
                                                 rho=self.rho, eps=self.delta)
                A.data.copy_(A_new)
                B.data.copy_(B_new)
                state["M_prev"] = None
                state["U_prev"] = None
                state["V_prev"] = None

            if self.log_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                rec["transport_residual"] = transport_residual
                rec["align_mom"] = certs["LB"]
                diag_records.append(rec)

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


class MuonRMSCoupledCoreLoRA(Optimizer):
    """Variant 3 of the Section 6 ladder: variant 2 + scalar RMS magnitude
    normalization. Adam-style "step magnitude ≈ lr regardless of grad state"
    behavior, without the per-coordinate diagonal that breaks gradient
    compatibility.

    Per-pair scalar second moment of τ̂ (= ‖core_obj‖_*/γ — the achieved
    primal value, equivalent to ‖Ĥ_t‖_F² up to a constant per the doc):
        s_t = β₂ · s_{t-1} + (1 − β₂) · τ̂_t²
        s_hat = s_t / (1 − β₂^t)               (Adam-style bias correction)
        Z_upd = −(lr / (√s_hat + ε)) · τ̂_t · Z_+
    At equilibrium with `s_hat ≈ τ̂_t²`, the spectral norm of Z_upd is ≈ lr,
    matching AdamW's per-coordinate-magnitude invariance to grad scale.

    This addresses the magnitude collapse observed in variant 2: when
    momentum averages out the core covector, ‖M_step‖_* shrinks but s_t
    tracks it, so the outer scale grows to compensate.

    Built on top of variant 2's transported core EMA + Nesterov lookahead
    (canonical Muon, no bc on the EMA itself — bias correction is only on
    the scalar RMS, mirroring Adam's β₂ branch).
    """

    def __init__(self, model, lr=2e-4, delta=1e-6, adapter_name=None,
                 beta1=0.95, beta2=0.999, eps=1e-8,
                 log_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.log_diagnostics = log_diagnostics
        self.diagnostics_every = diagnostics_every
        self.pair_state = {i: {"step": 0,
                               "M_prev": None,
                               "U_prev": None,
                               "V_prev": None,
                               "s": 0.0}
                           for i in range(len(pairs))}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_diagnostics else None

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for MuonRMSCoupledCoreLoRA update.")
            state = self.pair_state[i]
            state["step"] += 1
            t_step = state["step"]

            A_f = A.float()
            B_f = B.float()
            GA_f = A.grad.float()
            GB_f = B.grad.float()
            # Initialization-zero boundary case (Section 1 standing assumption).
            if _factor_essentially_zero(B_f) or _factor_essentially_zero(A_f):
                dA_fb, dB_fb, certs_fb, _ = _zero_B_fallback(
                    A, B, A.grad, B.grad, lr, delta=self.delta,
                    core_scale="constrained",  # boundary case has no τ̂ history yet
                )
                A.add_(dA_fb.to(dtype=A.dtype, device=A.device))
                B.add_(dB_fb.to(dtype=B.dtype, device=B.device))
                A.grad.zero_()
                B.grad.zero_()
                if self.log_diagnostics:
                    rec = {k: float(v) for k, v in certs_fb.items() if isinstance(v, (int, float))}
                    rec["norm_dA"] = float(dA_fb.norm().item())
                    rec["norm_dB"] = float(dB_fb.norm().item())
                    rec["s_rms"] = float(state["s"])
                    rec["lr_eff"] = lr
                    diag_records.append(rec)
                continue

            bases = _build_active_core(A_f, B_f, GA_f, GB_f, delta=self.delta)
            H_hat = bases["H_hat"]
            r = A_f.shape[0]
            tt, ss = bases["t"], bases["s"]
            U_cur = torch.cat([bases["Q_L"], bases["U"]], dim=1) if tt > 0 else bases["Q_L"]
            V_cur = torch.cat([bases["Q_R"], bases["V"]], dim=1) if ss > 0 else bases["Q_R"]

            # Variant 2's EMA + Nesterov (canonical Muon, no bc on the EMA).
            transport_residual = float("nan")
            if state["M_prev"] is None or t_step == 1:
                M_t = (1.0 - self.beta1) * H_hat
            else:
                T_L = U_cur.T @ state["U_prev"]
                T_R = V_cur.T @ state["V_prev"]
                M_transported = T_L @ state["M_prev"] @ T_R.T
                M_transported = _project_zero_22(M_transported, r)
                M_t = self.beta1 * M_transported + (1.0 - self.beta1) * H_hat
                den = M_transported.norm().item() + 1e-30
                transport_residual = float((M_t - M_transported).norm().item() / den)
            M_step = (1.0 - self.beta1) * H_hat + self.beta1 * M_t

            # Compute τ̂ for THIS step's M_step (used both as the polar magnitude
            # and as the s_t accumulator input).
            P, sv = _polar_via_svd(M_step)
            nuc = float(sv.sum().item())
            R_proj = _project_zero_22(P, r)
            gamma_sv = torch.linalg.svdvals(R_proj)
            gamma = float(gamma_sv[0].item()) if gamma_sv.numel() > 0 else 0.0
            tau_hat = (nuc / gamma) if gamma > 1e-12 else 0.0

            # Scalar RMS update (Adam-style bias correction on β₂).
            state["s"] = self.beta2 * state["s"] + (1.0 - self.beta2) * (tau_hat * tau_hat)
            bc2 = 1.0 - self.beta2 ** t_step
            s_hat = state["s"] / bc2
            lr_eff = lr / (s_hat ** 0.5 + self.eps)

            # Apply step. Use the lift via _polar_coupled_core_lift with
            # the effective lr; squared-penalty form gives Z_upd = -lr_eff·τ̂·Z_+
            # whose spectral norm at equilibrium is ≈ lr (since lr_eff·τ̂ → lr).
            dA, dB, certs = _polar_coupled_core_lift(
                M_step, bases, lr_eff, r,
                core_scale="squared_penalty", core_norm="operator",
                delta=self.delta, H_hat_for_align=H_hat,
            )
            certs["compat"] = bases["compat"]
            _attach_factor_diagnostics(certs, A_f, B_f, bases, dA, dB)
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            A.grad.zero_()
            B.grad.zero_()

            state["M_prev"] = M_t
            state["U_prev"] = U_cur
            state["V_prev"] = V_cur

            if self.log_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["compat"] = bases["compat"]
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                rec["transport_residual"] = transport_residual
                rec["align_mom"] = certs["LB"]
                rec["s_rms"] = float(state["s"])
                rec["lr_eff"] = lr_eff
                rec["tau_step"] = tau_hat
                diag_records.append(rec)

        if self.log_diagnostics and diag_records:
            step_count = self.pair_state[0]["step"]
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)


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
    picard_alpha: float = 1.0,
    picard_iters_override: int | None = None,
    anderson_m: int = 0,
    anderson_reg: float = 1e-10,
    soap_beta: float = 0.95,
    soap_refresh_every: int = 1,
    polar_norm_dir: str = "frob",
    polar_sigma_power: float | None = None,
    polar_method: str = "ns",
    beta1: float = 0.9,
    beta2: float = 0.999,
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
    if optimizer_type == "adafactor":
        return _build_lora_adafactor(
            model,
            lr=lr,
            lora_plus_multiplier=lora_plus_multiplier,
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
    if optimizer_type == "adam-lin-core-lora":
        # Cross-check: same Sylvester solver as adam-lin-lora, but Adam-EMA
        # on the core-space K matrix instead of factor preconditioned grads.
        return AdamLinCoreLoRA(
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
            betas=(beta1, beta2),
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
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
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
            picard_iters=picard_iters_override if picard_iters_override is not None else 3,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
        )
    if optimizer_type == "adam-soap-polar-product-lora":
        return AdamSOAPPolarProductLoRA(
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
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            soap_beta=soap_beta,
            soap_refresh_every=soap_refresh_every,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
        )
    if optimizer_type == "adafactor-polar-product-lora":
        return AdaFactorPolarProductLoRA(
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
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
        )
    if optimizer_type == "sign-momentum-polar-product-lora":
        return SignMomentumPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-endrms":
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
            end_rms_align=True,
        )
    if optimizer_type == "adam-clip-product-lora":
        # Clip operator + RMS-align (no gauge/lift). Mirrors baseline
        # adam-polar-product-lora but uses clip instead of polar — tests
        # whether spectrum-preservation helps when cross-coupling is NOT
        # absorbed by a gauge constraint.
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            operator_type="clip",
        )
    if optimizer_type == "adam-clip-product-lora-coupled":
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            picard_alpha=picard_alpha,
            operator_type="clip",
        )
    if optimizer_type == "adam-clip-product-lora-coupled-endrms":
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=2, end_rms_align=True,
            operator_type="clip",
        )
    if optimizer_type == "adam-polar-product-lora-gauge":
        return AdamPolarProductLoRAGauge(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-gauge-coupled":
        return AdamPolarProductLoRAGauge(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-clip-gauge":
        return AdamPolarProductLoRAClipGauge(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-clip-gauge-coupled":
        return AdamPolarProductLoRAClipGauge(
            model, lr=lr,
            betas=(0.9, 0.999),
            delta=1e-6,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-scalar-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-preserve-scalar",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-preserve",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-restore-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-restore",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-balanced-scalar-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="balanced-scalar",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-state-rebalanced-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-sign-lora":
        # Phase 2 (B): per-step elementwise sign normalization of the
        # core covector before polar — Adam-like per-coord adaptivity in
        # core space, no EMA, no basis-transport issue.
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-factor-adam-lora":
        # Rung-6 ablation: Adam-EMA on factor gradients, then projected-quotient-polar.
        # Direct theoretical comparison to Picard's adam-polar-product-lora-coupled.
        return PolarCoupledCoreFactorAdamLoRA(
            model, lr=lr, delta=1e-6,
            betas=(0.9, 0.999), eps=1e-8,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-factor-adam-rebalanced-lora":
        return PolarCoupledCoreFactorAdamLoRA(
            model, lr=lr, delta=1e-6,
            betas=(0.9, 0.999), eps=1e-8,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-sign-rebalanced-lora":
        # variant 1 + sign + state rebalance (compound: per-coord adaptivity
        # AND iLoRA-target factor geometry).
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            state_rebalance=True, rebalance_every=1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-imbalance-scalar-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="imbalance-preserve-scalar",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-imbalance-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="imbalance-preserve",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-balanced-scalar-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="balanced-scalar",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-state-rebalanced-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-sign-lora":
        # variant 2 + sign norm: momentum AND per-coord adaptivity in core.
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-sign-rebalanced-lora":
        # variant 2 + sign norm + state rebalance: full stack.
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            state_rebalance=True, rebalance_every=1,
            log_diagnostics=log_optim_diagnostics,
            diagnostics_every=optim_diagnostics_every,
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
