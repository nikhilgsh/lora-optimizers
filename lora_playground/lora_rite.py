"""LoRA-RITE optimizer (Yen et al., ICLR 2025, arXiv:2410.20625).

Ported from the authors' official PyTorch reimplementation
(https://github.com/... ; local copy used for the port), adapted to this
codebase's conventions:

  * driven by ``collect_lora_pairs(model)`` so it does NOT depend on PEFT
    emitting the LoRA params in ``A_1, B_1, A_2, B_2, ...`` order (the
    reference assumes that ordering; we pair by module instead);
  * per-pair state in ``self.pair_state`` (tensor dicts) so the project's
    checkpoint/resume path (lora_playground/checkpoint.py) persists it;
  * optimizer math runs in float32 (the reference runs eigh/QR/SVD in the
    param dtype — unsafe in bf16) and the update is cast back to the param
    dtype at apply time, per this repo's optimizer convention.

The algorithm is unchanged. For a pair A:(r,d_in), B:(d_out,r) (PEFT
orientation), LoRA-RITE QR-factors each side, expresses the gradient in the
rotation-invariant ("unmagnified") basis, accumulates an r x r second moment
transported across the changing basis each step, preconditions by its inverse
square root, EMA-smooths the first moment, then re-magnifies by the rotation.
This is the recipe the authors validate in their README
(betas=(0.9, 0.999), clip_unmagnified_grad=1.0, escape-mass OFF).

The momentum-free / second-moment-free reduction of this update is the bare
decoupled spectral-LMO factor step (see paper appendix "Momentum-Free
LoRA-RITE"); the full method here keeps the adaptive second moment, which is
the part that distinguishes it from Muon/iMuon.
"""

import torch
from torch.optim.optimizer import Optimizer

from .utils import collect_lora_pairs


class _LoraRiteHelper:
    """Math helpers, vendored from the reference PyTorch reimplementation.

    All operations are framework-neutral and dtype-preserving; callers feed
    float32 tensors so eigh/QR/SVD run in float32.
    """

    def __init__(self, maybe_inf_to_nan: bool = True):
        self._maybe_inf_to_nan = maybe_inf_to_nan

    def inf_to_nan(self, array):
        if not self._maybe_inf_to_nan:
            return array
        return torch.nan_to_num(array, nan=torch.nan, posinf=torch.nan, neginf=torch.nan)

    def bias_corrected_decay(self, step, decay: float):
        # Section 7.1 of arXiv:1804.04235 (Adafactor): folds bias correction
        # into the decay so the moments need no separate correction.
        t = step + 1.0
        return decay * (1.0 - decay ** (t - 1.0)) / (1.0 - decay ** t)

    def move_lora_dim_to_last(self, x, dim):
        x = torch.moveaxis(x, dim, -1)
        return x.reshape(-1, x.shape[-1]), x.shape

    def restore_original_shape_and_dim(self, x, dim, shape):
        x = x.reshape(shape)
        return torch.moveaxis(x, -1, dim)

    def restore_param_shape(self, x, p, dim):
        _, shape = self.move_lora_dim_to_last(p, dim)
        return self.restore_original_shape_and_dim(x, dim, shape)

    def inverse_sqrt(self, x, esc, epsilon, epsilon_root, relative_epsilon=False, force_positive=False):
        if relative_epsilon:
            eps = torch.max(torch.linalg.eigvalsh(x)) * epsilon_root
        else:
            eps = epsilon_root
        w, v = torch.linalg.eigh(x)
        if force_positive:
            w = torch.maximum(w, torch.zeros_like(w))
        w = 1.0 / (torch.sqrt(w + esc) + epsilon)
        w = torch.unsqueeze(w, 0)
        return self.make_symmetric(v * w @ v.T).to(x.dtype)

    def make_symmetric(self, x):
        return (x + x.T) / 2

    def transform_first_moment_to_new_basis(self, m, p):
        return m @ p.T

    def transform_second_moment_to_new_basis(self, v, p):
        v_new = p @ v @ p.T
        v_new = self.make_symmetric(v_new)
        v_new = torch.nan_to_num(v_new)
        return v_new

    def get_unmagnified_rotate_second_escape(self, v_new, v_old):
        eig_old = torch.linalg.eigvalsh(v_old)
        eig_new = torch.linalg.eigvalsh(v_new)
        eigen_diff = torch.maximum(torch.max(eig_old - eig_new), torch.tensor(0).to(eig_old))
        trace_diff = torch.maximum(torch.trace(v_old) - torch.trace(v_new), torch.tensor(0).to(eig_old))
        return torch.minimum(eigen_diff, trace_diff)

    def get_unmagnified_grad(self, g, ri):
        return g @ ri

    def rotate_update(self, g, ri):
        return g @ ri.T

    def get_preconditioned_update(self, g, p, e, epsilon, epsilon_root, relative_epsilon, apply_escape):
        u = g
        q = self.make_symmetric(p)
        if not apply_escape:
            e = 0
        qir = self.make_symmetric(self.inverse_sqrt(q, e, epsilon, epsilon_root, relative_epsilon))
        u = u @ qir
        return torch.nan_to_num(u)

    def update_first_moments(self, step, update, moments, beta1: float):
        beta1_decay = self.bias_corrected_decay(step, beta1)
        return (1.0 - beta1_decay) * update + beta1_decay * moments

    def compute_second_moments(self, update):
        s = (update.T @ update) / update.shape[0]
        return self.make_symmetric(s)

    def update_second_moments(self, step, update, moments, beta2: float):
        beta2_decay = self.bias_corrected_decay(step, beta2)
        s = (1.0 - beta2_decay) * update + beta2_decay * moments
        return self.make_symmetric(s)

    def update_second_escape(self, step, update, moments, beta2: float):
        beta2_decay = self.bias_corrected_decay(step, beta2)
        return (1.0 - beta2_decay) * update + beta2_decay * moments

    def get_rotation_and_basis(self, w):
        return torch.linalg.qr(w)

    def reduce_rms(self, x):
        return torch.sqrt(torch.mean(torch.pow(x, 2)))

    def clip_update(self, update, clip_threshold: float):
        mean_update = self.inf_to_nan(self.reduce_rms(update))
        denom = torch.maximum(torch.ones_like(mean_update), mean_update / clip_threshold)
        return update / denom

    def skip_update(self, update, skip_threshold: float):
        mean_update = self.inf_to_nan(self.reduce_rms(update))
        if mean_update > skip_threshold:
            update = torch.zeros_like(update)
        return update


class LoRARite(Optimizer):
    """LoRA-RITE: transformation-invariant adaptive preconditioning for LoRA.

    Pairs are (A, B) with A:(r, d_in), B:(d_out, r). The reference's
    lora_l_dim=0 / lora_r_dim=-1 (r on the last axis after the move) is the
    correct mapping for this orientation; we keep it.
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        weight_decay: float = 0.0,
        apply_escape: bool = False,
        balance_param: bool = False,
        maybe_inf_to_nan: bool = True,
        adapter_name=None,
    ):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        super().__init__([{"params": params, "lr": lr}], {})

        self.pairs = pairs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.maybe_inf_to_nan = maybe_inf_to_nan
        self.epsilon = eps
        self.epsilon_root = eps ** 2
        self.relative_epsilon = False
        self.apply_escape = apply_escape
        self.clip_unmagnified_grad = clip_unmagnified_grad
        self.update_capping = update_capping
        self.update_skipping = update_skipping
        self.weight_decay = weight_decay
        self.balance_param = balance_param
        # LoRA factors are 2D in this repo: A:(r,d_in) -> move dim 0 to last,
        # B:(d_out,r) -> dim -1 already last.
        self.lora_l_dim = 0
        self.lora_r_dim = -1
        self.helper = _LoraRiteHelper(maybe_inf_to_nan)

        # Per-pair state (float32), persisted by checkpoint.py via pair_state.
        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            aL, _ = self.helper.move_lora_dim_to_last(A.data.float(), self.lora_l_dim)
            aR, _ = self.helper.move_lora_dim_to_last(B.data.float(), self.lora_r_dim)
            r = aL.shape[-1]
            self.pair_state[i] = {
                "step": 0,
                "v_l": torch.zeros((r, r), dtype=torch.float32, device=aL.device),
                "v_r": torch.zeros((r, r), dtype=torch.float32, device=aR.device),
                "m_l": torch.zeros_like(aL),
                "m_r": torch.zeros_like(aR),
                "basis_l": torch.zeros_like(aL),
                "basis_r": torch.zeros_like(aR),
                "escape_l": 0.0,
                "escape_r": 0.0,
            }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        h = self.helper
        beta1, beta2 = self.beta1, self.beta2
        eps, eps_root = self.epsilon, self.epsilon_root
        rel_eps, apply_escape = self.relative_epsilon, self.apply_escape

        # ---- Pass 1: unmagnified gradients + global gradient norm ----------
        tmp = {}
        g_norm_sq = torch.zeros((), dtype=torch.float32)
        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for LoRA-RITE update.")
            st = self.pair_state[i]

            param_l, _ = h.move_lora_dim_to_last(A.data.float(), self.lora_l_dim)
            param_r, _ = h.move_lora_dim_to_last(B.data.float(), self.lora_r_dim)

            dec_l = h.get_rotation_and_basis(param_l)   # QR -> (basis, rotate)
            dec_r = h.get_rotation_and_basis(param_r)
            basis_l, rotate_l = dec_l[0], dec_l[1]
            basis_r, rotate_r = dec_r[0], dec_r[1]
            rotate_inv_l = torch.linalg.pinv(rotate_l)
            rotate_inv_r = torch.linalg.pinv(rotate_r)
            p_l = basis_r.T @ st["basis_r"]
            p_r = basis_l.T @ st["basis_l"]

            update_l = h.inf_to_nan(A.grad.float())
            update_r = h.inf_to_nan(B.grad.float())
            update_l, _ = h.move_lora_dim_to_last(update_l, self.lora_l_dim)
            update_r, _ = h.move_lora_dim_to_last(update_r, self.lora_r_dim)
            update_l = h.get_unmagnified_grad(update_l, rotate_inv_r)
            update_r = h.get_unmagnified_grad(update_r, rotate_inv_l)

            if self.update_skipping > 0:
                update_l = h.skip_update(update_l, self.update_skipping)
                update_r = h.skip_update(update_r, self.update_skipping)

            st["basis_l"] = basis_l
            st["basis_r"] = basis_r

            tmp[i] = dict(
                rotate_inv_l=rotate_inv_l, rotate_inv_r=rotate_inv_r,
                update_l=update_l, update_r=update_r, p_l=p_l, p_r=p_r,
            )
            g_norm_sq = g_norm_sq + torch.linalg.norm(update_l) ** 2
            g_norm_sq = g_norm_sq + torch.linalg.norm(update_r) ** 2
        g_norm = torch.sqrt(g_norm_sq)

        # ---- Pass 2: precondition, momentum, re-magnify, apply -------------
        for i, (A, B) in enumerate(self.pairs):
            st = self.pair_state[i]
            t = tmp[i]
            count = st["step"]

            param_l, _ = h.move_lora_dim_to_last(A.data.float(), self.lora_l_dim)
            param_r, _ = h.move_lora_dim_to_last(B.data.float(), self.lora_r_dim)

            rotate_inv_l, rotate_inv_r = t["rotate_inv_l"], t["rotate_inv_r"]
            update_l, update_r = t["update_l"], t["update_r"]
            p_l, p_r = t["p_l"], t["p_r"]

            if self.clip_unmagnified_grad > 0 and g_norm > self.clip_unmagnified_grad:
                update_l = update_l / g_norm * self.clip_unmagnified_grad
                update_r = update_r / g_norm * self.clip_unmagnified_grad

            s_l = h.compute_second_moments(update_l)
            s_r = h.compute_second_moments(update_r)

            transformed_v_l = h.transform_second_moment_to_new_basis(st["v_l"], p_l)
            transformed_v_r = h.transform_second_moment_to_new_basis(st["v_r"], p_r)

            if apply_escape:
                escape_l = h.get_unmagnified_rotate_second_escape(transformed_v_l, st["v_l"])
                escape_r = h.get_unmagnified_rotate_second_escape(transformed_v_r, st["v_r"])
                escape_l = h.update_second_escape(count, 0, escape_l + st["escape_l"], beta2)
                escape_r = h.update_second_escape(count, 0, escape_r + st["escape_r"], beta2)
            else:
                escape_l = escape_r = 0

            v_l = h.update_second_moments(count, s_l, transformed_v_l, beta2)
            v_r = h.update_second_moments(count, s_r, transformed_v_r, beta2)

            update_l = h.get_preconditioned_update(update_l, v_l, escape_l, eps, eps_root, rel_eps, apply_escape)
            update_r = h.get_preconditioned_update(update_r, v_r, escape_r, eps, eps_root, rel_eps, apply_escape)

            m_l = h.transform_first_moment_to_new_basis(st["m_l"], p_l)
            m_r = h.transform_first_moment_to_new_basis(st["m_r"], p_r)
            m_l = h.update_first_moments(count, update_l, m_l, beta1)
            m_r = h.update_first_moments(count, update_r, m_r, beta1)
            update_l, update_r = m_l, m_r

            if self.update_capping > 0:
                update_l = h.clip_update(update_l, self.update_capping)
                update_r = h.clip_update(update_r, self.update_capping)

            update_l = h.rotate_update(update_l, rotate_inv_r)
            update_r = h.rotate_update(update_r, rotate_inv_l)

            if self.weight_decay > 0:
                update_l = update_l + self.weight_decay * param_l
                update_r = update_r + self.weight_decay * param_r

            step_size = -1.0 * self.param_groups[0]["lr"]
            update_l = step_size * update_l
            update_r = step_size * update_r

            if self.balance_param:
                l_norm = torch.linalg.norm(param_l + update_l) + 1e-6
                r_norm = torch.linalg.norm(param_r + update_r) + 1e-6
                balanced_norm = torch.sqrt(l_norm * r_norm)
                update_l = update_l * (balanced_norm / l_norm) + param_l * (balanced_norm / l_norm - 1)
                update_r = update_r * (balanced_norm / r_norm) + param_r * (balanced_norm / r_norm - 1)

            update_l = h.restore_param_shape(update_l, A.data.float(), self.lora_l_dim)
            update_r = h.restore_param_shape(update_r, B.data.float(), self.lora_r_dim)
            A.data.add_(update_l.to(A.data.dtype))
            B.data.add_(update_r.to(B.data.dtype))

            st["step"] = count + 1
            st["v_l"], st["v_r"] = v_l, v_r
            st["m_l"], st["m_r"] = m_l, m_r
            st["escape_l"], st["escape_r"] = escape_l, escape_r

        return loss
