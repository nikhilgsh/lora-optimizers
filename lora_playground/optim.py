import inspect
import json
import os
import statistics

import torch
from torch.optim import AdamW, Optimizer, SGD

from .utils import (
    collect_lora_pairs,
    collect_lora_pairs_named,
    collect_ucv_triples,
    f_lorsum,
    lorsum,
    solve_spd,
    solve_sylvester,
    spd_frac_power_inv,
    spd_inv_sqrt_higham,
    spdify,
    truncated_svd,
)
from ._step_timer import maybe_time
from .optim_diagnostics import factor_diagnostics


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
    # Stored as `self._higham_compute_dtype` (None or torch.float16) — invert
    # the conversion back to the "fp32"/"fp16" string the __init__ accepts.
    "higham_compute_dtype": lambda opt: (
        "fp16"
        if getattr(opt, "_higham_compute_dtype", None) is not None
        else "fp32"
    ),
}


def _is_json_safe(v) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_json_safe(x) for x in v)
    return False


# ─── shared effective-config resolvers ────────────────────────────────────────
#
# `_polar_pipeline._polar_op` short-circuits across `polar_sigma_power` and
# `polar_method`. Two consumers need the resolved truth: the runtime dispatch
# inside `_polar_op` (which method to actually call) and the per-run cfg event
# `optimizer_effective` block (which method to record on disk). Both call the
# resolver below — single source of truth, no drift surface.
def resolve_effective_inner_polar(
    polar_sigma_power,
    polar_method,
    *,
    optimizer_class_name=None,
):
    """Resolve the canonical name of the polar operator for given kwargs.

    Returns one of:
      {"method": "svd_exact",     "label": "svd_exact"}
      {"method": "sigma_power",   "sigma_power": float,
                                  "label": "sigma_power(p=<float>)"}
      {"method": "ns_hybrid",     "label": "ns_hybrid"}
      {"method": "polar_express", "label": "polar_express"}
      {"method": "ns",            "label": "ns"}
      None  — no polar pipeline applies for this optimizer

    Precedence (highest first):
      polar_sigma_power == 0.0   → svd_exact
      polar_sigma_power != None  → sigma_power(p=…)
      polar_method ∈ {ns, ns_hybrid, polar_express} → that string
      optimizer_class_name implies a polar-product variant → "ns" (legacy
        fallback for cfgs missing polar_method; the runtime always sets it)
      otherwise → None
    """
    psp = polar_sigma_power
    if psp is not None and psp != "None":
        try:
            psp_f = float(psp)
        except (TypeError, ValueError):
            psp_f = None
        if psp_f is not None:
            if psp_f == 0.0:
                return {"method": "svd_exact", "label": "svd_exact"}
            return {
                "method": "sigma_power",
                "sigma_power": psp_f,
                "label": f"sigma_power(p={psp_f})",
            }
    if polar_method in {"ns", "ns_hybrid", "polar_express", "ssc"}:
        return {"method": polar_method, "label": polar_method}
    if optimizer_class_name:
        norm = optimizer_class_name.lower().replace("_", "").replace("-", "")
        if "polarproduct" in norm:
            return {"method": "ns", "label": "ns"}
    return None


def optimizer_effective_config(opt) -> dict:
    """Call `opt.effective_config()` if the method exists, else return {}.

    Module-level helper rather than a base-class method — `torch.optim.Optimizer`
    is third-party and we don't monkey-patch it. Project-owned subclasses that
    define short-circuit precedence (currently the AdamPolarProductLoRA family
    and the Gauge variants) implement `effective_config(self) -> dict`.
    """
    fn = getattr(opt, "effective_config", None)
    return fn() if callable(fn) else {}


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


# σ_max primitives live in lora_playground.spectral now. These names are
# re-exported here as thin aliases so the hundreds of call sites below
# don't churn. See `docs/notes/sigma_max_estimation.md` for the design.
#
# IMPORTANT: keep `_sigma_max_chol_eigvalsh` pointing at `sigma_max_krylov_chol`
# even though the default eps changed (1e-12 → 1e-6) — call sites do not
# pass `eps` explicitly, so they pick up the new default. The old 1e-12 was
# below the bf16-accumulation noise floor and caused Cholesky failure on h100;
# the new default is principled (~1 decade above the noise floor).
from .spectral import (
    sigma_max_power_iter as _sigma_max_power_iter,
    sigma_max_power_iter_batched as _sigma_max_power_iter_batched,
    sigma_max_krylov_chol as _sigma_max_chol_eigvalsh,
    sigma_max_power_iter_nonsym as _sigma_max_power_iter_nonsym,
    sigma_max_warm_power_iter_unfactored as _sigma_max_warm_power_iter_unfactored,
)


def _spd_inv_half(H, eps, method="eigh", higham_iters=10, eps_relative=False):
    """Dispatch (H + eps_eff·I)^{-1/2}: 'eigh' uses spd_frac_power_inv; 'higham'
    uses Newton-Schulz. When eps_relative=True, the effective damping is
    eps·λ_max(H) (σ_max-relative) instead of absolute eps."""
    if method == "eigh":
        eps_eff = eps
        if eps_relative:
            # Cheap λ_max via 3 power iters (matches the higham path's
            # internal estimate); inexpensive at r ≤ 256.
            n = H.shape[-1]
            v = torch.ones(n, dtype=H.dtype, device=H.device)
            v = v / v.norm().clamp(min=1e-30)
            for _ in range(3):
                v = H @ v
                nv = v.norm().clamp(min=1e-30)
                v = v / nv
            eps_eff = float(eps) * float(nv)
        return spd_frac_power_inv(H, gamma=0.5, eps=eps_eff)
    if method == "higham":
        return spd_inv_sqrt_higham(H, n_iters=higham_iters, eps=eps,
                                    eps_relative=eps_relative)
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


def _chord_update_opnorm_power_iter(A_f, B_f, dA, dB, n_iters=8):
    """Estimate ‖(B+dB)(A+dA) - BA‖₂ without materializing the dense update.

    The chord update is represented as a single skinny product

        ΔW = [B+dB, -B] @ [A+dA; A],

    then estimated by plain power iteration on that product structure. This
    intentionally avoids QR, Cholesky, eig/SVD, and the dense d_out × d_in
    chord matrix; each iteration is only low-rank matvecs through the factors.
    """
    A_f = A_f.detach().to(torch.float32)
    B_f = B_f.detach().to(torch.float32)
    dA = dA.detach().to(torch.float32)
    dB = dB.detach().to(torch.float32)
    left = torch.cat((B_f + dB, -B_f), dim=-1)
    right = torch.cat((A_f + dA, A_f), dim=-2)
    sigma, _ = _sigma_max_warm_power_iter_unfactored(
        left, right, n_iters=n_iters,
    )
    return sigma


def _is_main_process() -> bool:
    """Return True on rank 0 (single-process always). Used to gate the
    JSONL emissions in this module so they don't duplicate under DDP."""
    try:
        from .distributed import is_main
        return is_main()
    except Exception:
        return True


def _emit_non_finite_chain(step_count, intermediates, pair_names_in_group,
                           group_global_indices):
    """Emit a single `non_finite_intermediate` event at the end of a step if
    ANY intermediate in the optimizer's computation chain went non-finite.
    For each affected intermediate, lists which pair indices (global) carry
    the non-finite values. First emission per run pinpoints where in the
    chain the NaN was born.

    `intermediates` is a dict {name: tensor_or_scalar_or_None}. Tensors are
    expected to be either:
      - shape (N, ...): batched per pair in the group; we reduce all-but-
        the-first dim with isfinite.all and report pair-local indices.
      - shape (N,): per-pair scalar.
      - shape (): scalar (group-wide).
    """
    if not _is_main_process():
        return None
    pair_local_bad = {}    # {name: [local_idx, ...]}
    scalar_bad = {}         # {name: True} for group-wide scalars
    for name, t in intermediates.items():
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.numel() == 0:
            continue
        finite = torch.isfinite(t)
        if t.dim() == 0:
            if not bool(finite):
                scalar_bad[name] = True
        elif t.dim() == 1:
            mask = ~finite
            if mask.any():
                pair_local_bad[name] = mask.nonzero(as_tuple=True)[0].tolist()
        else:
            mask = (~finite).flatten(1).any(dim=1)
            if mask.any():
                pair_local_bad[name] = mask.nonzero(as_tuple=True)[0].tolist()
    if not pair_local_bad and not scalar_bad:
        return None
    # Translate per-intermediate local indices → global pair info.
    where = {}
    for name, local_idxs in pair_local_bad.items():
        where[name] = [
            {"local": int(li),
             "global": int(group_global_indices[li]),
             "pair_name": pair_names_in_group[li]
                 if li < len(pair_names_in_group) else f"pair_{li}"}
            for li in local_idxs
        ]
    for name in scalar_bad:
        where[name] = "group_scalar"
    payload = {
        "event": "non_finite_intermediate",
        "step": int(step_count),
        "where": where,
    }
    print(json.dumps(payload, sort_keys=True, default=float), flush=True)
    return where


def _emit_sigma_guard_event(step_count, *, site, side, n, guard_info,
                            pair_names_in_group, group_global_indices):
    """Emit when a σmax safety guard activates in a debug run.

    The event is causal instrumentation: it distinguishes "the guarded helper
    ran" from "the guard actually changed a dangerous denominator." It is only
    called from debug-gated paths because the `.any().item()` checks synchronize.
    """
    if not _is_main_process() or not guard_info:
        return None
    hit_mask = guard_info.get("guard_hit")
    if hit_mask is None:
        return None
    hit_mask = hit_mask.detach()
    if not bool(hit_mask.any().item()):
        return None

    def _count(name):
        t = guard_info.get(name)
        return int(t.detach().sum().item()) if isinstance(t, torch.Tensor) else 0

    local_idxs = hit_mask.nonzero(as_tuple=True)[0].tolist()
    where = [
        {"local": int(li),
         "global": int(group_global_indices[li]),
         "pair_name": pair_names_in_group[li]
             if li < len(pair_names_in_group) else f"pair_{li}"}
        for li in local_idxs
    ]
    raw = guard_info.get("sigma_raw")
    floor = guard_info.get("sigma_floor")
    payload = {
        "event": "sigma_max_guard_hit",
        "step": int(step_count),
        "site": str(site),
        "side": str(side),
        "n": int(n) if n is not None else -1,
        "n_pairs": int(hit_mask.numel()),
        "n_hit": int(hit_mask.sum().item()),
        "ones_fallback_count": _count("ones_fallback"),
        "warm_start_fallback_count": _count("warm_start_fallback"),
        "iter_fallback_count": _count("iter_fallback"),
        "floor_count": _count("floor"),
        "where": where,
    }
    if isinstance(raw, torch.Tensor) and isinstance(floor, torch.Tensor):
        raw_hit = raw.detach()[hit_mask]
        floor_hit = floor.detach()[hit_mask]
        payload.update({
            "sigma_raw_min": float(raw_hit.min().item()),
            "sigma_raw_max": float(raw_hit.max().item()),
            "sigma_floor_min": float(floor_hit.min().item()),
            "sigma_floor_max": float(floor_hit.max().item()),
            "floor_over_raw_max": float(
                (floor_hit / raw_hit.clamp_min(1e-30)).max().item()
            ),
        })
    print(json.dumps(payload, sort_keys=True, default=float), flush=True)
    return payload


def _emit_non_finite_event(step_count, pair_index, pair_name,
                           where, last_diag):
    """Emit one JSONL `non_finite_detected` event identifying a LoRA pair
    that has non-finite entries in one of its tensors at the *start* of an
    optimizer step. `where` is a dict `{tensor_name: bool}` indicating
    which tensors went bad (A/B/grad_A/grad_B). `last_diag` is the
    previous step's per-pair diagnostic record (or None). Always emitted
    on rank 0 — this is a fault signal, not a probe."""
    if not _is_main_process():
        return
    payload = {
        "event": "non_finite_detected",
        "step": int(step_count),
        "pair_index": int(pair_index),
        "pair_name": pair_name,
        "where": where,
        "prev_diag": last_diag,
    }
    print(json.dumps(payload, sort_keys=True, default=float), flush=True)


def _tensor_absmax_by_pair(t):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return t.detach().abs().flatten(1).amax(dim=1)


def _tensor_norm_by_pair(t):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return t.detach().flatten(1).norm(dim=1)


def _tensor_finite_by_pair(t):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return torch.isfinite(t.detach()).flatten(1).all(dim=1)


def _json_list_from_tensor(t):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return t
    t_cpu = t.detach().cpu()
    if t_cpu.dtype == torch.bool:
        return [bool(x) for x in t_cpu.reshape(-1).tolist()]
    return [float(x) for x in t_cpu.reshape(-1).tolist()]


def _emit_optimizer_pair_stats(step_count, group_id, group_global_indices,
                               pair_names_in_group, stats):
    """Emit per-pair scalar debug telemetry for one shape group."""
    if not _is_main_process():
        return
    payload = {
        "event": "optimizer_pair_stats",
        "step": int(step_count),
        "group_id": int(group_id),
        "pair_indices": [int(i) for i in group_global_indices],
        "pair_names": list(pair_names_in_group),
        "stats": {
            k: _json_list_from_tensor(v)
            for k, v in stats.items()
            if v is not None
        },
    }
    print(json.dumps(payload, sort_keys=True, default=float), flush=True)


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
    if _is_main_process():
        print(json.dumps(payload, sort_keys=True), flush=True)

OPTIMIZER_CHOICES = {
    "adamw",
    "adafactor",
    "lin-lora",
    "scaled-lora",
    "adam-scaled-lora",
    "adam-lin-lora",
    "adam-lin-core-lora",
    "curvature-whiten-lora",
    "curvature-whiten-polar-lora",
    "kl-shampoo-lora",
    "kl-shampoo-polar-lora",
    "kl-diag-lora",
    "kl-diag-polar-lora",
    "kl-diag-polar-flatout-lora",
    "diag-shampoo-lora",
    "diag-shampoo-polar-lora",
    "adam-scaled-lora-post",
    "adam-lin-lora-post",
    "adam-scaled-lora-matrix",
    "adam-lin-lora-matrix",
    "polar-product-lora",
    "adam-polar-product-lora",
    "adam-polar-product-lora-coupled",
    "adam-polar-product-lora-coupled-endrms",
    "adam-polar-product-lora-coupled-exact-chord",
    "adam-polar-product-lora-coupled-spectral-chord",
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-rho",
    "adam-polar-product-lora-coupled-spectral-chord-tight-exact",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
    "adam-polar-product-lora-coupled-spectral-chord-direction",
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
    "imuon-lora",
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
    "adam-ucv-core-lora",
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
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None, scaled_metric=False, lora_plus_multiplier=1.0, log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
            if log_basic_diagnostics:
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
        diag_records = [] if self.log_basic_diagnostics else None

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
                if self.log_basic_diagnostics:
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

            if self.log_basic_diagnostics:
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
            if self.log_basic_diagnostics and st['step'] % self.diagnostics_every == 0:
                self._last_cw_diag_records.append(self._cw_diag_record(
                    A_post=A,
                    B_post=B,
                    A_pre=A.detach().float() - dA.detach().float(),
                    B_pre=B.detach().float() - dB.detach().float(),
                    dA=dA,
                    dB=dB,
                    lr=lr,
                    rho=rho,
                    sigma_A=sA,
                    sigma_B=sB,
                    sigma_WA=sWA,
                    sigma_WB=sWB,
                ))
            A.grad.zero_()
            B.grad.zero_()

        if self.log_basic_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)

class CurvatureWhitenLoRA(Optimizer):
    """SOAP-on-momentum in the affordable S⊗D curvature basis.

    The strongest curvature preconditioner one can AFFORD for a LoRA factor
    (r×d, r≪d) is a Kronecker product: a FULL r×r matrix on the small side (r²
    is affordable) and a DIAGONAL on the large side (d×d is not). Each index is
    preconditioned exactly once — "most powerful affordable, non-redundant":

        z_A  =  SOAP(m_A; S_rA ⊗ D_in)               (A: r×d_in)
        z_B  =  SOAP(m_B; D_out ⊗ S_rB)              (B: d_out×r)
        ΔA   ∝  S_rA^{-1/2} · z_A · D_in^{-1/2}
        ΔB   ∝  D_out^{-1/2} · z_B · S_rB^{-1/2}

    where m̂ is bias-corrected momentum and the curvature factors are EMAs
    (decay ``curvature_beta``):
        S_rA = EMA(g_A g_Aᵀ) ∈ ℝ^{r×r}     D_in  = EMA(diag(g_Aᵀ g_A)) ∈ ℝ^{d_in}
        S_rB = EMA(g_Bᵀ g_B) ∈ ℝ^{r×r}     D_out = EMA(diag(g_B g_Bᵀ)) ∈ ℝ^{d_out}

    The small-side inverse-sqrt ``S_r^{-1/2} = Q Λ^{-1/2} Qᵀ`` is formed from the
    eigenbasis Q of S_r, maintained CHEAPLY by SOAP's warm-started power-iteration
    + QR (Vyas et al. 2024, Alg 4; first refresh = full eigh), refreshed every
    ``precond_refresh_every`` steps, batched across pairs — with eigenvalues
    λ_i = qᵢᵀ S_r qᵢ (Rayleigh quotient). So the SOAP QR machinery is reused purely
    as the efficient route to S_r^{-1/2}.

    Both inverse-sqrts use RELATIVE damping ``(λ/λ_max + δ)^{-1/2}`` with
    δ=``delta`` (caps conditioning at ~1/δ, controlling weak-direction
    amplification); a still-zero factor (early steps) falls back to identity.
    S_r and D set only the SHAPE (anisotropy); the step scale follows the
    chord-tight-clean convention so lr matches the curvature baselines exactly.
    lr is a spectral trust-region budget: ρ = lr / (σ_max(A) + σ_max(B)), and
    each per-factor update is rescaled to σ_max(ΔA) = ρ (σ_max via warm-started
    power iteration, NOT eigh). This is the same lr meaning as the chord-tight
    A/B arms, so the harmonious curve drops into the same lr grid.

    The SOAP part is Adam in the eigenbasis of the Kronecker curvature estimate:
    Q_r on the small side and the coordinate basis on the large diagonal side.
    ``v_A`` / ``v_B`` are elementwise EMAs of the rotated gradients, not inverse
    eigenvalues. ``use_polar`` optionally replaces z by polar(z) before the
    outer curvature whitening, giving a direct "SOAP inner, chord-tight outer"
    ablation. ``precond_delta_relative`` is accepted for factory compatibility
    (damping here is always relative).

    ── CANONICAL SPEC: paper/skeleton.tex Algorithm 1 (`alg:ours`). ─────────────
    That algorithm is the source of truth for the protagonist's update math
    (whiten → polar → unwhiten + operator-norm radius ρ=η/(σmax(A)+σmax(B))).
    THIS class is the IMPLEMENTATION; it realizes that math via an EIGENBASIS
    (Q diagonalises the small-side curvature S_r), so the in-code form looks like
    ``Q @ (m / √λ) @ Qᵀ …`` — that is the eigenbasis realization of Alg 1's
    ``C_A^{-1/2} φ(C_A^{-1/2} M̂_A Q^{-1/2}) Q^{-1/2}``, NOT a different update.
    Branches: the protagonist (`diag-shampoo-polar` + `cw_nesterov`) runs the
    ``soap_v=False`` closed-form-Shampoo path; ``soap_v=True`` is the SOAP-v̂ EMA
    variant (a different arm). When describing the METHOD or its ablations, cite
    Alg 1 — do NOT paraphrase from these eigenbasis fragments.
    """
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-3, eps=1e-8,
                 curvature_beta=0.99, use_polar=False, ns_steps=5,
                 polar_method="ns",
                 precond_delta_relative=False, adapter_name=None,
                 lora_plus_multiplier=1.0, log_basic_diagnostics=False,
                 log_heavy_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=10, kl_coupled=False, soap_v=True,
                 diag_metric=False, cw_picard_iters=1, flat_outer=False,
                 cw_nesterov=False, cw_no_radius=False, cw_no_diag_curv=False,
                 cw_factor_a=0.0, cw_factor_b=0.0):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.eps = eps
        self.delta = float(delta)            # relative damping for the inverse-sqrts
        self.beta1, self.beta2 = betas
        self.curvature_beta = float(curvature_beta)
        self.use_polar = bool(use_polar)
        self.ns_steps = int(ns_steps)
        if polar_method not in {"ns", "polar_express"}:
            raise ValueError(
                f"CurvatureWhitenLoRA polar_method must be 'ns' or 'polar_express', "
                f"got {polar_method!r}")
        self.polar_method = str(polar_method)
        self.precond_delta_relative = bool(precond_delta_relative)
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = int(precond_refresh_every)
        # KL-Shampoo-LoRA mode (soap_curvature_whitening.md exp 5). When
        # kl_coupled, the curvature factors are accumulated by the coupled KL
        # fixed point (Prop 4) — each Gram whitens g by the OTHER factor's
        # inverse before forming, with 1/d normalizers — instead of the
        # one-sided EMA(gg^T)/diag(g^T g). When not soap_v, the SOAP elementwise
        # v̂ is dropped and the inner core is the closed-form Shampoo whitening
        # S^{-1/2} m̂ D^{-1/2} (same operator as the outer sandwich, polar in
        # between). The two together are KL-Shampoo-LoRA; defaults reproduce the
        # SOAP-curvature class unchanged.
        self.kl_coupled = bool(kl_coupled)
        self.soap_v = bool(soap_v)
        # Option (b) of kl_shampoo_polar_derivation.md §"Cross-coupling": commit to
        # the single global diagonal metric (P,Q)=(D_out, D_in). The small-side
        # curvature is then the conjugate-diagonal-weighted geometric Gram,
        # M_A = Bᵀ diag(D_out) B and M_B = A diag(D_in) Aᵀ (recomputed each step),
        # in place of the dense KL S_curv. The diagonal KL coupling is unchanged
        # but now whitens by M⁻¹, so the whole step is a consistent single-metric
        # two-sided program with an exact cross term (vs the mixed-metric option a).
        # Requires kl_coupled=True (it reuses the D_in/D_out EMAs).
        self.diag_metric = bool(diag_metric)
        # diag_metric reuses the D_in/D_out EMAs as the single global diagonal metric.
        # With kl_coupled=True those diagonals are the KL coupled fixed point (Prop 4);
        # with kl_coupled=False they are plain grad-energy EMAs (the diag-shampoo arm).
        # Both are valid — the small side M_A = Bᵀ diag(D_out) B is recomputed each step
        # either way; only the diagonal accumulation rule differs.
        # Picard block-coordinate depth (cross-coupling). k=1 is the single-block
        # step (no cross-term). k>=2 corrects the simultaneous-step coupling via the
        # diagonal cross-term (kl_shampoo_polar_derivation.md §Cross-coupling) and is
        # only defined for the kl path (the cross-term uses the D_in/D_out diagonals).
        # flat_outer: skip the un-whiten so dX ∝ φ(z) (flat-spectrum, chord-tight
        # basin, curvature-chosen frame). Heuristic robustness probe — NOT the
        # curvature-metric LMO. Only meaningful with polar (it IS the polar step).
        self.flat_outer = bool(flat_outer)
        if self.flat_outer and not use_polar:
            raise ValueError("flat_outer=True requires use_polar=True (it is the polar-without-unwhiten step).")
        self.cw_picard_iters = int(cw_picard_iters)
        if self.cw_picard_iters < 1:
            raise ValueError("cw_picard_iters must be >= 1.")
        if self.cw_picard_iters > 1 and self.soap_v:
            raise ValueError("cw_picard_iters>1 is only defined for the kl path (soap_v=False).")
        # Nesterov lookahead on the momentum fed to the whiten→polar→unwhiten core
        # (Muon convention: orthogonalize ĝ + β₁·m rather than m̂). Ablation only —
        # the Shampoo lineage this curvature whitening inherits from uses plain EMA
        # momentum (~/SOAP, Meta distributed_shampoo); Muon uses Nesterov. The
        # operator-norm rescale ρ/σmax(W) normalizes magnitude, so this changes only
        # the momentum DIRECTION (no lr recalibration). Defined for the closed-form
        # Shampoo core (soap_v=False); the SOAP-v̂ arm keeps plain EMA.
        self.cw_nesterov = bool(cw_nesterov)
        if self.cw_nesterov and self.soap_v:
            raise ValueError("cw_nesterov is only defined for the soap_v=False (closed-form Shampoo) path.")
        # ABLATION (−radius / Spectron): use ρ=lr (plain) instead of the operator-norm
        # radius ρ=lr/(σmax(A)+σmax(B)). Tests whether the Spectron-style spectral
        # trust-region scaling helps. Default False = protagonist (radius on).
        self.cw_no_radius = bool(cw_no_radius)
        # ABLATION (−Shampoo / no diagonal curvature): force the relative-damped input/
        # output diagonals to identity (dinA=doutB=1). On the diag_metric path this makes
        # C_A=BᵀB, C_B=AAᵀ (the bare partner-Grams, i.e. P=Q=I in skeleton Alg 1) and drops
        # the input/output diagonal whitening — a partner-Gram-whitened polar (iMuon-like).
        # Tests whether the two-sided diagonal Shampoo curvature helps. Requires diag_metric.
        self.cw_no_diag_curv = bool(cw_no_diag_curv)
        if self.cw_no_diag_curv and not self.diag_metric:
            raise ValueError("cw_no_diag_curv requires diag_metric=True (the protagonist path).")
        # Per-factor shape scaling (sweep knob). c_A = (r/d_in)^a, c_B = (d_out/r)^b,
        # folded into the operator-norm radius so the merged product cap stays η:
        #   ρ = η / (c_A·σmax(B) + c_B·σmax(A)),  σmax(dA)=c_A·ρ,  σmax(dB)=c_B·ρ.
        # (0,0) is bit-identical to the protagonist (equal radius). Named rules are
        # points in (a,b): Keller (0, 1/2), Codex-rowspace (1/4, 1/2), MuP (1/2, 1/2).
        # The max(1,·) of Keller/MuP is inert for LoRA shapes (r<d_in, d_out>r always),
        # so it collapses to a fixed a — no separate clamp needed. Only the ratio
        # c_A/c_B matters (the product cap removes the common scale).
        self.cw_factor_a = float(cw_factor_a)
        self.cw_factor_b = float(cw_factor_b)

        # Q starts as identity → step 1 is plain Adam (no rotation), since the
        # eigenbasis must be built from gradients STRICTLY BEFORE the step it is
        # used in (Alg 3 updates L/R/Q after the weight step). First eigh seed
        # happens after step 1. Grams are zero-initialized (Alg 3 EMA).
        self._q_initialized = False
        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            r, d_in = A.shape
            d_out = B.shape[0]
            eye = torch.eye(r, dtype=torch.float32, device=A.device)
            self.pair_state[i] = {
                'm_A': torch.zeros_like(A, dtype=torch.float32),   # momentum
                'm_B': torch.zeros_like(B, dtype=torch.float32),
                # SOAP second moments in the S⊗D eigenbasis. The D side is
                # diagonal, so its eigenbasis is the coordinate basis.
                'v_A': torch.zeros_like(A, dtype=torch.float32),
                'v_B': torch.zeros_like(B, dtype=torch.float32),
                # small (r) side: full r×r curvature, whitened via its eigenbasis.
                'L_A': torch.zeros((r, r), dtype=torch.float32, device=A.device),
                'R_B': torch.zeros((r, r), dtype=torch.float32, device=A.device),
                # large side: diagonal curvature = EMA of per-column / per-row
                # gradient energy.
                'D_in': torch.zeros(d_in, dtype=torch.float32, device=A.device),
                'D_out': torch.zeros(d_out, dtype=torch.float32, device=B.device),
                'Q_A': eye.clone(),
                'Q_B': eye.clone(),
                # exact eigenvalues of L_A / R_B from the last eigh refresh
                # (zero until the first refresh → _rdinv → identity → plain
                # momentum on step 1).
                'lam_A': torch.zeros(r, dtype=torch.float32, device=A.device),
                'lam_B': torch.zeros(r, dtype=torch.float32, device=A.device),
                'step': 0,
            }

    @staticmethod
    def _sym(M):
        return 0.5 * (M + M.transpose(-2, -1))

    def _rdinv(self, x):
        """Relative-damped inverse-sqrt of a nonneg tensor along its last dim:
        (x/x_max + δ)^{-1/2}. Returns ones where x_max≈0 (uninitialized factor →
        no whitening; the Frobenius rescale then yields a plain momentum step)."""
        xmax = x.amax(dim=-1, keepdim=True)
        out = (x / xmax.clamp_min(1e-30) + self.delta).rsqrt()
        return torch.where(xmax < 1e-30, torch.ones_like(out), out)

    def _cw_diag_record(self, *, A_post, B_post, A_pre, B_pre, dA, dB,
                        lr, rho, sigma_A, sigma_B, sigma_WA, sigma_WB):
        """Diagnostics for one curvature-whiten step.

        Factor diagnostics are computed after applying the update, matching the
        historical KL logging location. Product-step diagnostics use the
        pre-step factors because they describe the actual just-applied update
        map: B dA + dB A + dB dA.
        """
        rec = factor_diagnostics(A_post, B_post)
        rec.update(_finite_step_product_diagnostics(A_pre, B_pre, dA, dB))
        lr_f = float(lr)
        tangent = rec.get("tangent_step_norm", float("nan"))
        finite = rec.get("finite_step_norm", float("nan"))
        rec.update({
            "norm_dA": float(dA.detach().to(torch.float32).norm()),
            "norm_dB": float(dB.detach().to(torch.float32).norm()),
            "cw_rho": float(rho),
            "cw_sigma_A": float(sigma_A),
            "cw_sigma_B": float(sigma_B),
            "cw_sigma_WA": float(sigma_WA),
            "cw_sigma_WB": float(sigma_WB),
            "product_tangent_over_lr": float(tangent / max(lr_f, 1e-30)),
            "product_finite_over_lr": float(finite / max(lr_f, 1e-30)),
        })
        return rec

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        cb = self.curvature_beta
        b1, eps = self.beta1, self.eps
        pairs = self.pairs
        n = len(pairs)

        # Require grads, then bump the (uniform) step counter for every pair.
        for A, B in pairs:
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for CurvatureWhitenLoRA update.")
        for i in range(n):
            self.pair_state[i]['step'] += 1

        # The per-pair Kronecker/eigenbasis update, run either as a grouped
        # batched pass (default — one set of bmm/batched-NS/batched-σ_max per
        # shape group) or a reference per-pair loop. Both use the SAME blessed
        # primitives (spectral.sigma_max_power_iter_batched + _newton_schulz_batched),
        # so they agree to batched-vs-looped float reduction order
        # (tests/test_curvature_whiten_batched.py); the grouped pass is the fast one.
        self._last_cw_diag_records = []
        _timer = getattr(self, "_step_timer", None)
        if _timer: _timer.start("cw_pairstep")
        if getattr(self, "_batched_step", True):
            self._cw_apply_grouped(lr, cb, b1, eps)
        else:
            self._cw_apply_per_pair(lr, cb, b1, eps)
        if _timer: _timer.stop()

        # ── Refresh the eigenbasis for the NEXT step. First refresh = ONE eigh
        # to seed Q (Alg 4 init); thereafter ONE warm-started power-iteration +
        # QR step (Alg 4: Q ← QR(L@Q)), every `precond_refresh_every`, batched
        # across pairs. eigh is too slow to run per refresh in production (~8× the
        # QR cost at r=256); the step's eigenvalues come from the cheap Rayleigh
        # diagonal above, not a fresh eigh. ──────────────────────────────────
        step_count = self.pair_state[0]['step']
        if (not self._q_initialized) or (step_count % self.precond_refresh_every == 0):
            if _timer: _timer.start("cw_refresh")
            LA_stack = torch.stack([self.pair_state[i]['L_A'] for i in range(n)])
            RB_stack = torch.stack([self.pair_state[i]['R_B'] for i in range(n)])
            if not self._q_initialized:
                QA = torch.linalg.eigh(self._sym(LA_stack))[1]
                QB = torch.linalg.eigh(self._sym(RB_stack))[1]
                self._q_initialized = True
            else:
                QA_prev = torch.stack([self.pair_state[i]['Q_A'] for i in range(n)])
                QB_prev = torch.stack([self.pair_state[i]['Q_B'] for i in range(n)])
                QA = torch.linalg.qr(LA_stack @ QA_prev)[0]
                QB = torch.linalg.qr(RB_stack @ QB_prev)[0]
            for i in range(n):
                self.pair_state[i]['Q_A'] = QA[i]
                self.pair_state[i]['Q_B'] = QB[i]
            if _timer: _timer.stop()

        # Tier-1 factor diagnostics (shared library): balance_resid /
        # stable_rank / σ_max, emitted as one `optim_step` event at the
        # diagnostics cadence. kl-shampoo (this class) previously logged no
        # optimizer-internal diagnostics, so the balance ↔ σ_max-fragility link
        # was un-instrumented on exactly the optimizer that NaNs via σ_max
        # under-estimation. Pure (A, B) function — see lora_playground.optim_diagnostics.
        if self.log_basic_diagnostics and step_count % self.diagnostics_every == 0:
            records = self._last_cw_diag_records
            if not records:
                records = [factor_diagnostics(A, B) for (A, B) in self.pairs]
            _emit_optim_diagnostics(step_count, records)

    def _factor_scales(self, r, d_in, d_out):
        """Per-factor shape coefficients (c_A, c_B) from the (a, b) exponents.

        c_A = (r/d_in)^a (compression side), c_B = (d_out/r)^b (expansion side).
        (0,0) → (1, 1). Returns python floats (shape-derived, constant per group).
        """
        a, b = self.cw_factor_a, self.cw_factor_b
        c_A = 1.0 if a == 0.0 else (r / d_in) ** a
        c_B = 1.0 if b == 0.0 else (d_out / r) ** b
        return c_A, c_B

    def _smax_warm(self, M, states, key, n_warm=3):
        """Warm-started batched σ_max with per-pair v_init caching.

        M: (N, p, q); `states`: list of N pair-state dicts. Caches the converged
        top singular vector under `key` and reuses it as v_init next step. The
        top direction is near-stationary across a step (weights move slowly; the
        update direction is correlated step-to-step), so warm n_iters=3 ≈ cold
        n_iters=8 (tests/test_sigma_max_power_iter.py::
        test_warm_start_beats_cold_start_at_same_n_iters). First call per key
        cold-starts (8 iters). The library's row-norm floor guards against any
        warm-start under-estimation overscaling the chord rescale."""
        from .spectral import sigma_max_power_iter_batched as _smax_b
        cached = [st.get(key) for st in states]
        vi = torch.stack(cached) if all(c is not None for c in cached) else None
        # Always run the full iteration count (the warm v_init only accelerates
        # convergence). The prior warm n_iters=3 chronically UNDER-estimated σ_max
        # by ~10-25% across every call site (zA/zB/WA/WB/A/B; SMAXDBG diagnostic),
        # which overscales the rescale (dA = ρ/σ̂·WA) and compounds into a NaN —
        # worst on the polar arm, whose orthogonalized WA has a flat spectrum that
        # single-vector power iteration tracks poorly.
        s, v = _smax_b(M, v_init=vi, n_iters=8)
        for j, st in enumerate(states):
            st[key] = v[j].detach()
        # Lower-bound floor: σ_max ≥ max(row L2, col L2) for ANY matrix, so this
        # never exceeds the true σ_max — it only lifts a stale/cold estimate that
        # missed the top direction. Deterministic in M, so batched and per-pair
        # agree (the NS scale-invariance + this shared floor preserve the oracle).
        with torch.no_grad():
            Mf = M.detach().float()
            rn = Mf.pow(2).sum(dim=-1).amax(dim=-1).sqrt()
            cn = Mf.pow(2).sum(dim=-2).amax(dim=-1).sqrt()
            s = torch.maximum(s, torch.maximum(rn, cn).reshape(s.shape).to(s.dtype))
        if os.environ.get("LORA_SMAX_DEBUG") == "1":
            # Diagnostic only: compare the warm estimate against the true σ_max
            # (full SVD) and flag gross under-estimates per call site, to locate
            # which _smax_warm site overscales before a NaN. Inert without the env.
            with torch.no_grad():
                true = torch.linalg.matrix_norm(M.detach().float(), ord=2).flatten()
                sf = s.detach().flatten()
                ratio = sf / (true + 1e-30)
                i = int(ratio.argmin()); mn = float(ratio[i])
                if mn < 0.9:
                    step = states[0].get("step", -1) if states else -1
                    print(f"SMAXDBG step={step} key={key} min_ratio={mn:.4f} "
                          f"warm={float(sf[i]):.5g} true={float(true[i]):.5g}", flush=True)
        return s

    def _polar_poly_batched(self, X):
        """Dispatch the spectral nonlinearity on a σ_max-normalized batch ``X``.

        ``polar_method='ns'`` → cubic-Muon Newton–Schulz (the legacy hardcoded
        path); ``'polar_express'`` → Amsel et al. (2505.16932) quintic-Remez
        polar. ``X`` is already normalized by ``_polar_ns_guarded`` so both use
        ``pre_norm='none'``. NOTE ns=5 is a PARTIAL polar; a true full polar
        needs ns≥8 or PolarExpress≥6 (see reference_full_polar_iteration_floor).
        """
        if self.polar_method == "polar_express":
            return _polar_express_gram_batched(
                X, nsteps=self.ns_steps, dtype=torch.float32, pre_norm="none")
        return _newton_schulz_gram_batched(
            X, nsteps=self.ns_steps, dtype=torch.float32, pre_norm="none",
            safety_factor=1.0)

    def _polar_ns_guarded(self, Z, states, key):
        """Scale-invariant polar of a batch ``Z`` (N,p,q) via gram-NS, guarded
        against a σ_max UNDER-estimate.

        The spectral-norm-law failure (CLAUDE.md): a stale/cold warm start vector
        can miss ``Z``'s top singular direction, so the warm σ_max under-estimates;
        ``Z / σ̂`` then enters the Newton–Schulz far above its convergence basin
        and diverges to NaN (the KL-coupled curvature shifts the spectrum enough
        to trigger this; CPU repro in tests/test_kl_shampoo_lora.py). Two layers:

          1. Floor the warm σ_max at the largest row / column L2 norm — both are
             valid LOWER bounds on σ_max, so a stale estimate can't be grossly
             small. In the healthy case the warm estimate already dominates the
             floor, so the denominator equals the old ``_smax_warm`` value exactly
             (preserving the batched↔per-pair equivalence the NS scale-invariance
             gives); the floor only binds when the warm estimate is pathological.
          2. Frobenius finiteness fallback: any non-finite NS output is recomputed
             from the Frobenius-normalized input. σ_max ≤ ‖·‖_F, so that input is
             guaranteed in-basin. φ is scale-invariant, so the fallback changes
             only NS convergence speed, never the polar target.

        PolarExpress exception (Amsel et al. 2505.16932 §3.3, Algorithm 1 ln 10):
        the quintic-Remez polynomials are derived for an input rescaled so its
        singular values lie in [ℓ, u] with u=1. Per Thm 3.3 / §3.3 the lower
        bound is forgiving — "the method converges for any ℓ∈(0,u], and even an
        order of magnitude error only delays convergence by a few iterations" —
        but VIOLATING the upper bound (σ>1) detonates the iteration. Offline on
        real snapshot momenta this is a CLIFF: σ_out=1 exactly when input σ_max
        ≤ u, fully non-finite at a ≥3% σ_max UNDER-estimate (no finite-but-
        overscaled window). The warm σ_max floor under-estimates by ~3.3× median
        at r=256, so an estimate-based denominator would detonate most pairs every
        step. The paper's own prescription is the fix: "a trivial upper bound is
        given by ‖M‖_F ... we therefore rescale M by ‖M‖_F and set u=1." So
        normalize the PolarExpress path by the Frobenius norm (a GUARANTEED upper
        bound, σ_max ≤ ‖·‖_F), removing the σ_max-accuracy dependence entirely;
        the loose lower bound only costs a couple iters (PE6 already gives σ_out=1,
        verified). The strict-below-u margin (§3.4's x→p_t(x/1.01) round-off trick)
        is supplied downstream by ``_polar_express_gram_batched``'s default
        safety_factor=1.05, so the polynomial sees σ_max ≤ 1/1.05 < 1. Cubic
        Newton–Schulz keeps the tight σ_max denominator (it needs σ≈1 to converge
        at ns=5/8 and degrades, not detonates, on over-compressed input). The
        finiteness fallback stays as defense-in-depth for both. ``_smax_warm`` is
        still called to keep the warm-start cache fresh for the other σ_max sites.
        """
        eps = self.eps
        s = self._smax_warm(Z, states, key)                          # (N,) warm σ_max
        rn = Z.pow(2).sum(dim=-1).amax(dim=-1).clamp_min(0).sqrt()    # max row L2  ≤ σ_max
        cn = Z.pow(2).sum(dim=-2).amax(dim=-1).clamp_min(0).sqrt()    # max col L2  ≤ σ_max
        s = torch.maximum(s, torch.maximum(rn, cn))
        if self.polar_method == "polar_express":
            s = Z.flatten(1).norm(dim=1)                              # ‖·‖_F ≥ σ_max (no detonation)
        Xn = Z / (s.view(-1, 1, 1) + eps)
        out = self._polar_poly_batched(Xn)
        bad = ~torch.isfinite(out.flatten(1)).all(dim=1)             # (N,) per-matrix
        if bool(bad.any()):
            fro = Z[bad].flatten(1).norm(dim=1).view(-1, 1, 1)
            Xf = Z[bad] / (fro + eps)                                # σ_max ≤ 1 guaranteed
            out = out.clone()
            out[bad] = self._polar_poly_batched(Xf)
        return out

    def _cw_apply_per_pair(self, lr, cb, b1, eps):
        """Reference per-pair update. Independent code from the grouped path but
        the SAME blessed primitives (sigma_max_power_iter_batched on a 1-batch +
        _newton_schulz_batched), so it is the equivalence oracle for
        _cw_apply_grouped and the n==1 fallback. Step counters already bumped."""
        from .spectral import sigma_max_power_iter_batched as _smax_b
        beta2 = self.beta2
        for i, (A, B) in enumerate(self.pairs):
            st = self.pair_state[i]
            gA = A.grad.float(); gB = B.grad.float()
            st['m_A'].mul_(b1).add_(gA, alpha=1.0 - b1)
            st['m_B'].mul_(b1).add_(gB, alpha=1.0 - b1)
            bc1 = 1.0 - b1 ** st['step']
            bc2 = 1.0 - beta2 ** st['step']
            QA, QB, LA, RB = st['Q_A'], st['Q_B'], st['L_A'], st['R_B']
            QAt = QA.transpose(-2, -1); QBt = QB.transpose(-2, -1)
            # Relative-damped diagonals M_in = dinA^(-2), M_out = doutB^(-2): the
            # SINGLE metric the diag arm commits to — used by the self small-side
            # Gram, the whitening, and the Picard cross, coherently.
            if self.cw_no_diag_curv:  # −Shampoo: M_in=M_out=I → C_A=BᵀB, C_B=AAᵀ
                dinA = torch.ones_like(st['D_in']); doutB = torch.ones_like(st['D_out'])
            else:
                dinA = self._rdinv(st['D_in']); doutB = self._rdinv(st['D_out'])
            Din_m = (dinA * dinA).reciprocal()
            Dout_m = (doutB * doutB).reciprocal()
            if self.diag_metric:
                # Option (b): M_A = Bᵀ diag(M_out) B, M_B = A diag(M_in) Aᵀ from the
                # SAME damped diagonals the cross uses (metric coherence), recomputed
                # and stored so the eigenbasis refresh tracks it.
                Bf = B.detach().float(); Af = A.detach().float()
                LA = Bf.transpose(-2, -1) @ (Dout_m.unsqueeze(-1) * Bf)
                RB = (Af * Din_m.unsqueeze(0)) @ Af.transpose(-2, -1)
                st['L_A'].copy_(LA); st['R_B'].copy_(RB)
            evA = (QA * (LA @ QA)).sum(dim=0)
            evB = (QB * (RB @ QB)).sum(dim=0)
            lamA = self._rdinv(evA); lamB = self._rdinv(evB)
            # SOAP v̂ EMA (state) once; mhat for the kl path.
            if self.soap_v:
                gA_basis = QAt @ gA
                gB_basis = gB @ QB
                st['v_A'].mul_(beta2).addcmul_(gA_basis, gA_basis, value=1.0 - beta2)
                st['v_B'].mul_(beta2).addcmul_(gB_basis, gB_basis, value=1.0 - beta2)
            else:
                if self.cw_nesterov:
                    # Lookahead: β₁·m + (1−β₁)·g (m already updated this step) — the
                    # EMA analog of Muon's ĝ + β·buf. Only direction matters downstream.
                    mhatA = (st['m_A'].mul(b1).add(gA, alpha=1.0 - b1)) / bc1
                    mhatB = (st['m_B'].mul(b1).add(gB, alpha=1.0 - b1)) / bc1
                else:
                    mhatA = st['m_A'] / bc1; mhatB = st['m_B'] / bc1
            Af = A.detach().float(); Bf = B.detach().float()
            sA = self._smax_warm(Af.unsqueeze(0), [st], 'v_sigma_A')[0]
            sB = self._smax_warm(Bf.unsqueeze(0), [st], 'v_sigma_B')[0]
            # c_A, c_B fold the per-factor shape scaling into ρ; merged cap
            # ρ(c_A·σmax(B) + c_B·σmax(A)) = η preserved (=current when c=1).
            cA, cB = self._factor_scales(Af.shape[0], Af.shape[1], Bf.shape[0])
            rho = lr if self.cw_no_radius else lr / (cA * sB + cB * sA + self.eps)
            # No §2.5 pre-rescale: σ_max momentum-normalization diluted the cross by
            # √(stable_rank) of the whitened momentum, making k≥2 a no-op. Cross is
            # added to the raw momentum (mirror of _cw_apply_grouped).
            # Picard block-coordinate loop (mirror of _cw_apply_grouped). k=1 ⇒
            # cross-term never formed ⇒ bit-identical to the pre-Picard step.
            dA = torch.zeros_like(gA)
            dB = torch.zeros_like(gB)
            for _pic in range(self.cw_picard_iters):
                if self.soap_v:
                    zA = QA @ ((QAt @ (st['m_A'] / bc1)) / ((st['v_A'] / bc2).sqrt() + self.eps))
                    zB = (((st['m_B'] / bc1) @ QB) / ((st['v_B'] / bc2).sqrt() + self.eps)) @ QBt
                else:
                    if _pic == 0:
                        inA = mhatA; inB = mhatB
                    else:
                        cross_A = ((Bf.transpose(-2, -1) @ (Dout_m.unsqueeze(-1) * dB)) @ Af) * Din_m.unsqueeze(0)
                        cross_B = Dout_m.unsqueeze(-1) * (Bf @ ((dA * Din_m.unsqueeze(0)) @ Af.transpose(-2, -1)))
                        inA = mhatA + (1.0 / lr) * cross_A
                        inB = mhatB + (1.0 / lr) * cross_B
                    zA = (QA @ ((QAt @ inA) * lamA.unsqueeze(-1))) * dinA.unsqueeze(0)
                    zB = (((inB @ QB) * lamB.unsqueeze(0)) @ QBt) * doutB.unsqueeze(-1)
                if self.use_polar:
                    zA = self._polar_ns_guarded(zA.unsqueeze(0), [st], 'v_sigma_zA')[0]
                    zB = self._polar_ns_guarded(zB.unsqueeze(0), [st], 'v_sigma_zB')[0]
                if self.flat_outer:
                    # Robustness probe (heuristic, NOT the curvature-metric LMO):
                    # skip the un-whiten. dX ∝ φ(z) is flat-spectrum (chord-tight
                    # basin) with a curvature-chosen frame — trust curvature for
                    # direction, not magnitude. σmax(φ(z))≈1, so the rescale → ρ.
                    WA, WB = zA, zB
                else:
                    WA = (QA @ ((QAt @ zA) * lamA.unsqueeze(-1))) * dinA.unsqueeze(0)
                    WB = (((zB @ QB) * lamB.unsqueeze(0)) @ QBt) * doutB.unsqueeze(-1)
                sWA = self._smax_warm(WA.unsqueeze(0), [st], 'v_sigma_WA')[0]
                sWB = self._smax_warm(WB.unsqueeze(0), [st], 'v_sigma_WB')[0]
                dA = -(cA * rho / (sWA + self.eps)) * WA
                dB = -self.lora_plus_multiplier * (cB * rho / (sWB + self.eps)) * WB
            emit_diag = self.log_basic_diagnostics and st['step'] % self.diagnostics_every == 0
            if emit_diag:
                A_pre = A.detach().float()
                B_pre = B.detach().float()
            A.add_(dA.to(dtype=A.dtype, device=A.device))
            B.add_(dB.to(dtype=B.dtype, device=B.device))
            if emit_diag:
                self._last_cw_diag_records.append(self._cw_diag_record(
                    A_post=A,
                    B_post=B,
                    A_pre=A_pre,
                    B_pre=B_pre,
                    dA=dA,
                    dB=dB,
                    lr=lr,
                    rho=rho,
                    sigma_A=sA,
                    sigma_B=sB,
                    sigma_WA=sWA,
                    sigma_WB=sWB,
                ))
            if self.kl_coupled:
                # Coupled KL fixed point (Prop 4); mirror of _cw_apply_grouped.
                r = gA.shape[0]; d_in = gA.shape[1]; d_out = gB.shape[0]
                Din_inv = dinA * dinA
                Dout_inv = doutB * doutB
                SAinv = QA @ ((lamA * lamA).unsqueeze(-1) * QAt)
                RBinv = QB @ ((lamB * lamB).unsqueeze(-1) * QBt)
                if not self.diag_metric:
                    st['L_A'].mul_(cb).add_((gA * Din_inv.unsqueeze(0)) @ gA.transpose(-2, -1),
                                            alpha=(1.0 - cb) / d_in)
                    st['R_B'].mul_(cb).add_(gB.transpose(-2, -1) @ (gB * Dout_inv.unsqueeze(-1)),
                                            alpha=(1.0 - cb) / d_out)
                st['D_in'].mul_(cb).add_((gA * (SAinv @ gA)).sum(dim=0), alpha=(1.0 - cb) / r)
                st['D_out'].mul_(cb).add_((gB * (gB @ RBinv)).sum(dim=1), alpha=(1.0 - cb) / r)
            else:
                # diag_metric recomputes L_A/R_B from the diagonals each step (above), so
                # do NOT clobber them with a Gram EMA — only accumulate the plain diagonals.
                if not self.diag_metric:
                    st['L_A'].mul_(cb).add_(gA @ gA.transpose(-2, -1), alpha=1.0 - cb)
                    st['R_B'].mul_(cb).add_(gB.transpose(-2, -1) @ gB, alpha=1.0 - cb)
                st['D_in'].mul_(cb).add_((gA * gA).sum(dim=0), alpha=1.0 - cb)
                st['D_out'].mul_(cb).add_((gB * gB).sum(dim=1), alpha=1.0 - cb)
            A.grad.zero_(); B.grad.zero_()

    def _cw_apply_grouped(self, lr, cb, b1, eps):
        """Grouped batched update: one set of bmm / batched-NS / batched-σ_max
        per (d_in, d_out) shape group. Same math as _cw_apply_per_pair (Q/L/R are
        r×r for every pair, so the eigenbasis refresh in step() stays global)."""
        from collections import defaultdict
        from .spectral import sigma_max_power_iter_batched as _smax_b
        beta2 = self.beta2
        S, pairs = self.pair_state, self.pairs
        timer = getattr(self, "_step_timer", None)
        groups = defaultdict(list)
        for i, (A, B) in enumerate(pairs):
            groups[(A.shape[1], B.shape[0])].append(i)
        for idxs in groups.values():
            t = S[idxs[0]]['step']
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - beta2 ** t
            gA = torch.stack([pairs[i][0].grad.float() for i in idxs])
            gB = torch.stack([pairs[i][1].grad.float() for i in idxs])
            Aw = torch.stack([pairs[i][0].detach().float() for i in idxs])
            Bw = torch.stack([pairs[i][1].detach().float() for i in idxs])
            mA = torch.stack([S[i]['m_A'] for i in idxs]).mul_(b1).add_(gA, alpha=1.0 - b1)
            mB = torch.stack([S[i]['m_B'] for i in idxs]).mul_(b1).add_(gB, alpha=1.0 - b1)
            QA = torch.stack([S[i]['Q_A'] for i in idxs])
            QB = torch.stack([S[i]['Q_B'] for i in idxs])
            LA = torch.stack([S[i]['L_A'] for i in idxs])
            RB = torch.stack([S[i]['R_B'] for i in idxs])
            vA = torch.stack([S[i]['v_A'] for i in idxs])
            vB = torch.stack([S[i]['v_B'] for i in idxs])
            Din = torch.stack([S[i]['D_in'] for i in idxs])
            Dout = torch.stack([S[i]['D_out'] for i in idxs])
            QAt = QA.transpose(-2, -1); QBt = QB.transpose(-2, -1)
            # Relative-damped diagonals M_in = dinA^(-2), M_out = doutB^(-2): the
            # SINGLE metric the diag arm commits to — used by the self small-side
            # Gram, the whitening, and the Picard cross, coherently.
            if self.cw_no_diag_curv:  # −Shampoo: M_in=M_out=I → C_A=BᵀB, C_B=AAᵀ
                dinA = torch.ones_like(Din); doutB = torch.ones_like(Dout)
            else:
                dinA = self._rdinv(Din); doutB = self._rdinv(Dout)
            Din_m = (dinA * dinA).reciprocal()
            Dout_m = (doutB * doutB).reciprocal()
            if self.diag_metric:
                # Option (b): small-side curvature = conjugate-diagonal-weighted
                # geometric Gram, recomputed each step (replaces the dense S_curv
                # EMA). M_A = Bᵀ diag(M_out) B, M_B = A diag(M_in) Aᵀ from the SAME
                # damped diagonals the cross uses (metric coherence). Downstream
                # Rayleigh eigenvalues / QR refresh / whitening and the diagonal KL
                # coupling are unchanged; the write-back of L_A/R_B below stores M so
                # the eigenbasis refresh tracks it.
                LA = Bw.transpose(-2, -1) @ (Dout_m.unsqueeze(-1) * Bw)
                RB = (Aw * Din_m.unsqueeze(1)) @ Aw.transpose(-2, -1)
            if timer: timer.start("cw_basis_proj")
            # Relative-damped inverse-sqrt factors (-1/2 power) for the small (λ)
            # and large (D) sides; needed by both the outer sandwich and the
            # KL-Shampoo inner core. evA/evB are the Rayleigh eigenvalues of the
            # curvature in the current eigenbasis.
            evA = (QA * (LA @ QA)).sum(dim=1)
            evB = (QB * (RB @ QB)).sum(dim=1)
            lamA = self._rdinv(evA); lamB = self._rdinv(evB)
            # SOAP v̂ EMA (state update, once) — only the SOAP-curvature arm.
            if self.soap_v:
                gA_basis = QAt @ gA
                gB_basis = gB @ QB
                vA.mul_(beta2).addcmul_(gA_basis, gA_basis, value=1.0 - beta2)
                vB.mul_(beta2).addcmul_(gB_basis, gB_basis, value=1.0 - beta2)
            else:
                if self.cw_nesterov:
                    # Lookahead (mA/mB already updated this step); mirror of per-pair.
                    mhatA = (mA.mul(b1).add(gA, alpha=1.0 - b1)) / bc1
                    mhatB = (mB.mul(b1).add(gB, alpha=1.0 - b1)) / bc1
                else:
                    mhatA = mA / bc1; mhatB = mB / bc1
            grp = [S[i] for i in idxs]
            # σ_max(A), σ_max(B) and ρ are loop-invariant (factors don't move until
            # the update is applied after the loop), so compute them once.
            sA = self._smax_warm(Aw, grp, 'v_sigma_A')
            sB = self._smax_warm(Bw, grp, 'v_sigma_B')
            # c_A, c_B: per-factor shape scaling folded into ρ (shape-constant within
            # the group); merged cap ρ(c_A·σmax(B)+c_B·σmax(A))=η preserved.
            cA, cB = self._factor_scales(Aw.shape[-2], Aw.shape[-1], Bw.shape[-2])
            # cw_no_radius: plain η per group. Keep ρ a (ngroups,) tensor (not a
            # python float) so the per-group ρ[j] in the diagnostic record below and
            # the broadcast cA·ρ/σ rescale both stay shape-correct.
            rho = (torch.full_like(sB, float(lr)) if self.cw_no_radius
                   else lr / (cA * sB + cB * sA + self.eps))
            # No §2.5 pre-rescale: the σ_max momentum-normalization diluted the cross
            # by √(stable_rank) of the whitened momentum (σ_max=1 base has Frobenius
            # √sr ≫ 1), so k≥2 collapsed onto k=1. The cross is added to the raw
            # momentum. (The raw cross/momentum ratio ∝ 1/‖G‖ — loss-scale-dependent,
            # but far larger than the σ_max-diluted version on high-rank factors.)
            # ── Picard block-coordinate loop (cw_picard_iters). k=1 ⇒ the cross-term
            # is never formed (iter 0) ⇒ bit-identical to the pre-Picard step. For
            # k≥2 each iter adds the cross-coupling correction (Jacobi)
            #   g̃_A = m̂_A + (1/η)·Bᵀ diag(D_out) dB A diag(D_in)   (mirror for B)
            # from kl_shampoo_polar_derivation.md §Cross-coupling (Prop 3): the
            # full-space diagonals D_out,D_in at power 1 — exact for diag_metric
            # (option b), mixed-metric for dense kl (option a). Reassociated to stay
            # in the skinny r×d factors (no dense d_out×d_in). Magnitude is re-pinned
            # (ρ-rescale) every iter, so each dA/dB fed forward is at physical scale
            # and there is no cross-iter normalization drift to track.
            dA = torch.zeros_like(gA)
            dB = torch.zeros_like(gB)
            if timer: timer.start("cw_picard")
            for _pic in range(self.cw_picard_iters):
                if self.soap_v:
                    zA = QA @ ((QAt @ (mA / bc1)) / ((vA / bc2).sqrt() + self.eps))
                    zB = (((mB / bc1) @ QB) / ((vB / bc2).sqrt() + self.eps)) @ QBt
                else:
                    if _pic == 0:
                        inA = mhatA; inB = mhatB
                    else:
                        cross_A = ((Bw.transpose(-2, -1) @ (Dout_m.unsqueeze(-1) * dB)) @ Aw) * Din_m.unsqueeze(1)
                        cross_B = Dout_m.unsqueeze(-1) * (Bw @ ((dA * Din_m.unsqueeze(1)) @ Aw.transpose(-2, -1)))
                        inA = mhatA + (1.0 / lr) * cross_A
                        inB = mhatB + (1.0 / lr) * cross_B
                    zA = (QA @ ((QAt @ inA) * lamA.unsqueeze(-1))) * dinA.unsqueeze(1)
                    zB = (((inB @ QB) * lamB.unsqueeze(1)) @ QBt) * doutB.unsqueeze(-1)
                if self.use_polar:
                    zA = self._polar_ns_guarded(zA, grp, 'v_sigma_zA')
                    zB = self._polar_ns_guarded(zB, grp, 'v_sigma_zB')
                if self.flat_outer:
                    # See _cw_apply_per_pair: skip the un-whiten (flat-spectrum probe).
                    WA, WB = zA, zB
                else:
                    WA = (QA @ ((QAt @ zA) * lamA.unsqueeze(-1))) * dinA.unsqueeze(1)
                    WB = (((zB @ QB) * lamB.unsqueeze(1)) @ QBt) * doutB.unsqueeze(-1)
                sWA = self._smax_warm(WA, grp, 'v_sigma_WA')
                sWB = self._smax_warm(WB, grp, 'v_sigma_WB')
                dA = -(cA * rho / (sWA + self.eps)).view(-1, 1, 1) * WA
                dB = -self.lora_plus_multiplier * (cB * rho / (sWB + self.eps)).view(-1, 1, 1) * WB
            if timer: timer.stop()
            if timer: timer.start("cw_curv_grams")
            if self.kl_coupled:
                # Coupled KL fixed point (Prop 4): each Gram whitens g by the
                # OTHER factor's relative-damped inverse before forming, with 1/d
                # normalizers — the streaming flip-flop toward the matrix-normal
                # MLE. dinA²/lamA² are the power-(-1) factors; SAinv = Q diag(λ⁻¹) Qᵀ.
                # At warmup the factors are zero ⇒ _rdinv returns ones ⇒ identity
                # whitening ⇒ first alternation is one-sided Shampoo.
                r = gA.shape[1]; d_in = gA.shape[2]; d_out = gB.shape[1]
                Din_inv = dinA * dinA
                Dout_inv = doutB * doutB
                SAinv = QA @ ((lamA * lamA).unsqueeze(-1) * QAt)
                RBinv = QB @ ((lamB * lamB).unsqueeze(-1) * QBt)
                if not self.diag_metric:
                    LA.mul_(cb).add_((gA * Din_inv.unsqueeze(1)) @ gA.transpose(-2, -1),
                                     alpha=(1.0 - cb) / d_in)
                    RB.mul_(cb).add_(gB.transpose(-2, -1) @ (gB * Dout_inv.unsqueeze(-1)),
                                     alpha=(1.0 - cb) / d_out)
                Din.mul_(cb).add_((gA * (SAinv @ gA)).sum(dim=1), alpha=(1.0 - cb) / r)
                Dout.mul_(cb).add_((gB * (gB @ RBinv)).sum(dim=2), alpha=(1.0 - cb) / r)
            else:
                # diag_metric recomputes LA/RB from the diagonals each step (above), so
                # do NOT clobber them with a Gram EMA — only accumulate the plain diagonals.
                if not self.diag_metric:
                    LA.mul_(cb).add_(gA @ gA.transpose(-2, -1), alpha=1.0 - cb)
                    RB.mul_(cb).add_(gB.transpose(-2, -1) @ gB, alpha=1.0 - cb)
                Din.mul_(cb).add_((gA * gA).sum(dim=1), alpha=1.0 - cb)
                Dout.mul_(cb).add_((gB * gB).sum(dim=2), alpha=1.0 - cb)
            if timer: timer.stop()
            for j, i in enumerate(idxs):
                A_, B_ = pairs[i]
                A_.add_(dA[j].to(dtype=A_.dtype, device=A_.device))
                B_.add_(dB[j].to(dtype=B_.dtype, device=B_.device))
                if self.log_basic_diagnostics and t % self.diagnostics_every == 0:
                    self._last_cw_diag_records.append(self._cw_diag_record(
                        A_post=A_,
                        B_post=B_,
                        A_pre=Aw[j],
                        B_pre=Bw[j],
                        dA=dA[j],
                        dB=dB[j],
                        lr=lr,
                        rho=rho[j],
                        sigma_A=sA[j],
                        sigma_B=sB[j],
                        sigma_WA=sWA[j],
                        sigma_WB=sWB[j],
                    ))
                S[i]['m_A'].copy_(mA[j]); S[i]['m_B'].copy_(mB[j])
                S[i]['v_A'].copy_(vA[j]); S[i]['v_B'].copy_(vB[j])
                S[i]['L_A'].copy_(LA[j]); S[i]['R_B'].copy_(RB[j])
                S[i]['D_in'].copy_(Din[j]); S[i]['D_out'].copy_(Dout[j])
                A_.grad.zero_(); B_.grad.zero_()


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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
    def __init__(self, model, lr=2e-4, betas=(0.9, 0.999), delta=1e-6, eps=1e-8, adapter_name=None, log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20, precond_refresh_every=1):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
            if log_basic_diagnostics:
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
        diag_records = [] if self.log_basic_diagnostics else None

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
                if self.log_basic_diagnostics:
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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 eps=1e-8, adapter_name=None, log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
        pairs = collect_lora_pairs(model, adapter_name)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.delta = delta
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 lora_plus_multiplier=1.0, log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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


def _newton_schulz(X, nsteps=5, eps=1e-7, pre_norm="frob"):
    """
    Newton-Schulz orthogonalization of X (float32). Canonical Muon: returns a
    matrix with approximately orthonormal rows (or cols, for tall X), Frobenius
    norm ≈ √min(r, d), INDEPENDENT of the input magnitude. This is the whole
    point of Muon — the optimizer's step size is set by lr alone, not by the
    (highly variable, especially early-training) gradient magnitude.

    `pre_norm` controls the basin-bringing divisor applied to X before iteration:
      - "frob" (default): X ← X / ‖X‖_F. Scale-invariant, ~free (one reduction).
        Caveat: leaves σ_max(X) ≈ σ_max_orig / ‖X‖_F = 1/√(stable_rank), so
        for a wide spectrum the Schulz iteration starts FAR from σ=1 and 5
        iters do not fully converge (output spectrum is graded, not flat).
      - "spec":  X ← X / σ_max(X). Forces σ_max=1 exactly; iteration starts
        at the σ=1 fixed point for the top mode. Costs one power-iter.
      - "none":  no divide. Trust the caller (e.g. when upstream applied a
        spec-norm or §2.5-style pre-rescale).
    """
    X = X.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    # X is now (r, d) with r ≤ d
    if pre_norm == "frob":
        norm = X.norm() + eps
        X = X / norm
    elif pre_norm == "spec":
        # σ_max via the library power-iter util (matvec-based, no SVD); the
        # full-SVD `matrix_norm(ord=2)` here was ~80% of the curvature-whiten-
        # polar step at r256 (224 SVDs/step). NS only needs σ < √3 and the
        # caller renormalizes the update magnitude downstream, so a power-iter
        # estimate is safe. Uses the house n_iters=8 (≈5% p95 cold rel-error).
        smax, _ = _sigma_max_power_iter(X, n_iters=8)
        X = X / (smax + eps)
    elif pre_norm != "none":
        raise ValueError(f"pre_norm must be one of {{'frob','spec','none'}}, got {pre_norm!r}")
    for _ in range(nsteps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X.T if tall else X


def _newton_schulz_batched(X, nsteps=5, eps=1e-7, dtype=None, pre_norm="frob"):
    """Batched Newton-Schulz over leading dims. X: (..., m, n) -> (..., m, n).

    Mirrors `_newton_schulz` (Muon orthogonalization, per-matrix Frobenius
    pre-normalization, no post-multiply) but vectorizes across the batch.
    Used by the batched polar-pipeline path for shape-grouped pairs.

    `dtype` controls the iteration dtype:
    - None / torch.float32: fp32 throughout (default). Matches per-matrix
      `_newton_schulz` to fp32 noise.
    - torch.bfloat16: pre-norm in fp32 (small numbers), iterate in bf16.
      Tensor cores accumulate to fp32 internally on Ampere+; output is bf16
      cast back to fp32 by caller. Same pattern as modded-nanogpt's polar
      express (`train_gpt.py:187` — `X = g.bfloat16()` then iterate). 2×
      throughput on Ampere bf16 tensor cores vs fp32; orthogonality residual
      bottoms at ~bf16 precision (~1e-3) which is well within Algorithm 1's
      tolerance for the polar map.

    Equivalence at fp32: max-abs-err < 1e-7 vs per-matrix `_newton_schulz`
    on real LoRA shapes (`scripts/bench/bench_ns_batched.py`).
    """
    X = X.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)
    if pre_norm == "frob":
        norm = X.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1) + eps
        X = X / norm
    elif pre_norm == "spec":
        smax = torch.linalg.matrix_norm(X, ord=2).unsqueeze(-1).unsqueeze(-1) + eps
        X = X / smax
    elif pre_norm != "none":
        raise ValueError(f"pre_norm must be one of {{'frob','spec','none'}}, got {pre_norm!r}")
    if dtype is not None and dtype != X.dtype:
        X = X.to(dtype)
    for _ in range(nsteps):
        XXT = X @ X.transpose(-2, -1)
        X = 1.5 * X - 0.5 * XXT @ X
    return X.transpose(-2, -1) if tall else X


def _newton_schulz_gram_batched(
    X,
    nsteps=5,
    eps=1e-7,
    dtype=torch.float16,
    restart_at=2,
    safety_factor=1.05,
    pre_norm="frob",
):
    """Gram-form Newton-Schulz (Dao 2026, Algorithm 3 — Stabilized Gram NS).

    Mathematically equivalent to `_newton_schulz_batched` (cubic Muon polynomial
    1.5 X - 0.5 X X^T X), but iterates on the smaller-side r×r Gram matrix
    R = X X^T instead of on rectangular X. Only the initial R_0 = X X^T and
    the final X_final = Q_T X_0 require rectangular matmuls; everything in
    between is (r, r) x (r, r). For r ≪ d this is ~7× fewer FLOPs at K=5.

    Production path (dtype=torch.float16, restart_at=2): runs the iteration
    in fp16 on tensor cores, with one restart at iter τ=2 to reset the
    spurious-negative-eigenvalue compounding that breaks naive Gram NS
    in half precision (Dao §"Instability of Naive Gram NS"). The restart
    re-forms R from the current X iterate, capping the exponential blowup
    of any negative R eigenvalues at half-precision noise floor magnitude
    1e-6 over the remaining T-τ iters.

    Safety path (dtype=torch.float32, restart_at=None): inner loop wrapped
    with `allow_tf32 = False`. Noise floor drops to ~1e-7 so the blowup
    factor (15/8)^(2T) from any spurious negative never reaches the basin
    edge — no restart needed. Slower (no tensor cores) but bulletproof.
    Used by tests as the fp32 reference and as a fallback if fp16+restart
    is ever observed to blow up on real data.

    Future opt: at cubic Muon coefficients the per-iter blowup factor is
    only 2.25 (vs Polar Express 3.5), and Tier 1 evidence shows fp16
    WITHOUT restart tracks fp32 fine on real chord-tight r=64 X_eff up to
    cond(G) ≈ 1e4. Once that holds across more (r, optimizer) configs we
    could drop restart for a ~80% headroom gain. Defer until evidence is
    broader; one blown-up cell mid-sweep costs more than the headroom.

    The reconstruction matmul X_final = Q · X_0 does not compound and runs
    in the iteration dtype (bf16 internal accumulate on tensor cores).
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X
    X = X.float()
    # Tall-transpose so the second-to-last dim is the smaller (r ≤ d).
    # The Gram form assumes r ≤ d; flipping breaks the FLOP win and would
    # instantiate a (d, d) Q matrix. Always force this orientation.
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)
    # Flatten leading dims to exactly one batch dim so torch.baddbmm (used to
    # fuse the inner-iter `b·(R@Q) + a·Q` and `b²·R²·R + 2ab·R²` updates) is
    # well-defined (it requires 3-D inputs). Restore the original shape at
    # the end.
    orig_leading = X.shape[:-2]
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    # X is now (batch, r, d) with r ≤ d.
    r = X.shape[-2]

    # Pre-norm (per-matrix, fp32) with Dao's safety factor so the post-norm
    # σ_max ≤ 1/safety_factor < 1 — gives margin for half-precision roundoff
    # that can otherwise push singular values just above the NS basin.
    # pre_norm controls which divisor brings σ_max into the basin:
    #   "frob": divide by ‖X‖_F (scale-invariant). Caveat: σ_max post-divide
    #           = 1/√(stable_rank), so iteration starts far from σ=1 for wide
    #           spectra (incomplete after 5 iters).
    #   "spec": divide by σ_max(X) directly. Forces σ_max(X_normed) = 1; iteration
    #           starts at the σ=1 fixed point for the top mode. One extra
    #           power-iter cost; tightest output.
    #   "none": no divide beyond safety_factor. Use when caller already
    #           spec-normed (e.g. chord-tight-clean §2.5 pre-rescale).
    if pre_norm == "frob":
        norm = X.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1) + eps
        X_normed = X / (norm * safety_factor)
    elif pre_norm == "spec":
        smax = torch.linalg.matrix_norm(X, ord=2).unsqueeze(-1).unsqueeze(-1) + eps
        X_normed = X / (smax * safety_factor)
    elif pre_norm == "none":
        X_normed = X / safety_factor
    else:
        raise ValueError(f"pre_norm must be one of {{'frob','spec','none'}}, got {pre_norm!r}")

    # State dtype for the iteration; reconstruction also runs in this dtype.
    iter_dtype = dtype
    X0_iter = X_normed.to(iter_dtype)

    # Cubic Muon polynomial X ← 1.5 X - 0.5 X X^T X, which in Gram form is
    # Z_t = a I + b R, M_t = Z_t (symmetric). a = 1.5, b = -0.5.
    a = 1.5
    b = -0.5

    # If running in fp32 (safety mode), force TF32 off to mirror spd_power_batched.
    use_tf32_guard = (iter_dtype == torch.float32) and X.is_cuda
    prev_tf32_matmul = None
    if use_tf32_guard:
        prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        eye = torch.eye(r, dtype=iter_dtype, device=X0_iter.device)
        eye_b = eye.expand(*X0_iter.shape[:-2], r, r)

        R = X0_iter @ X0_iter.transpose(-2, -1)  # (..., r, r), should be PSD
        Q = eye_b.clone()
        # First Z is M_1 = a I + b R; mathematically Q_1 = M_1, R_1 = M_1 R M_1.
        # We fold the first iter into the loop below; init Q = I.

        for t in range(1, nsteps + 1):
            # Restart at iter τ (Dao Algorithm 3 step 8): update X_0 ← Q_τ X_0,
            # reform R from the updated X, reset Q to I. This zeroes the
            # cumulative drift in Q and resets the negative-eigenvalue
            # magnitudes to the noise floor of the current X iterate. The
            # final Y = Q · X0_iter at function exit then applies only the
            # POST-restart Q to the post-restart X, matching Dao's algorithm.
            if restart_at is not None and t == restart_at + 1:
                X0_iter = Q @ X0_iter                     # (..., r, d)
                R = X0_iter @ X0_iter.transpose(-2, -1)   # (..., r, r)
                Q = eye_b.clone()

            # Matrix-quadratic ordering per Dao §"Computing Matrix Quadratics":
            # keep a·I out of the fused-add downcast by handling a·Q / a·R
            # separately. For cubic NS (c = 0), M = a·I + b·R, so
            #   Q ← b·R·Q  + a·Q     (one fused baddbmm)
            #   R ← M·R·M = a²·R + 2ab·R² + b²·R³
            new_Q = torch.baddbmm(Q, R, Q, beta=a, alpha=b)

            R2 = R @ R
            # new_R = a²·R + R²·(2ab·I + b²·R) — fuse second matmul with its add.
            #   torch.baddbmm(C=R2, A=R2, B=R, beta=2ab, alpha=b²)
            #     = 2ab·R² + b²·R²·R = 2ab·R² + b²·R³.
            # Then add a²·R as a separate elementwise (one launch).
            mid = torch.baddbmm(R2, R2, R, beta=(2.0 * a * b), alpha=(b * b))
            new_R = (a * a) * R + mid
            # No explicit symmetrize: in exact arithmetic M and R are symmetric
            # and M·R·M is symmetric (M = aI+bR commutes with R, so MR=RM).
            # In fp16 asymmetry compounds at ~1e-4/iter — well below NS=5
            # polar tolerance. Tested in tests/test_ns_gram.py.

            Q = new_Q
            R = new_R
    finally:
        if use_tf32_guard:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul

    # Final reconstruction: one rectangular matmul, no compounding downstream.
    # NS contract (matching _newton_schulz_batched): output is approximately
    # orthonormal (σ_out → 1 for surviving directions), Frobenius-norm
    # INDEPENDENT of input magnitude. Do not multiply by safety_factor or
    # norm at the end — the basin scaling is absorbed by the NS map.
    out_iter = Q @ X0_iter                                # (batch, r, d)
    out = out_iter.float()
    # Restore the original leading dimensions.
    out = out.reshape(*orig_leading, *out.shape[-2:])

    return out.transpose(-2, -1) if tall else out


# Opt-in: compile the leaf-numeric helpers with torch.compile when
# LORA_COMPILE_KERNELS=1. These functions have fixed-shape inner loops
# and no Python control flow inside the loop body (the restart_at branch
# is at a fixed iteration, predictable), so they compile cleanly into
# fused kernels. Trade-off: ~30s one-time compile cost per shape group.
# Worth it for sweeps; off by default to keep unit tests / smokes fast.
if os.environ.get("LORA_COMPILE_KERNELS", "0") == "1":
    _newton_schulz_gram_batched = torch.compile(
        _newton_schulz_gram_batched, dynamic=False, fullgraph=False,
    )


def _ssc_svd(M, c, eps=1e-30):
    """SPECTRA (arXiv:2603.14315) Soft Spectral Clipping via SVD — ground truth.

    H_c(X) = U · diag(σ / sqrt(1 + (σ/c)²)) · V^T,  where X = U diag(σ) V^T.

    Equivalent (operator form): H_c(X) = (I + X X^T / c²)^{-1/2} X. The MISR
    routine `_ssc_misr_batched` computes this without SVD via Newton-Schulz
    on the inverse-square-root of (I + X X^T / c²); use this function as
    the reference for validating MISR convergence on real snapshots.

    Operates on the last two dims; leading dims are broadcast through
    `torch.linalg.svd`. Returns same shape and dtype as input.
    """
    in_dtype = M.dtype
    Mf = M.float()
    U, S, Vh = torch.linalg.svd(Mf, full_matrices=False)
    # h_c(σ) = σ / sqrt(1 + (σ/c)²); denom ≥ 1 so eps clamp is only for safety.
    denom = torch.sqrt(1.0 + (S / c).pow(2)).clamp_min(eps)
    S_clip = S / denom
    out = (U * S_clip.unsqueeze(-2)) @ Vh
    return out.to(in_dtype)


def _ssc_misr_batched(X, c, nsteps=10, eps=1e-12):
    """SPECTRA (arXiv:2603.14315) Soft Spectral Clipping via Newton-Schulz MISR.

    Computes H_c(X) = (I + X X^T / c²)^{-1/2} · X without SVD, via the
    matrix-inverse-square-root Schulz iteration on the gram-side matrix
    G = I + X X^T / c²  (size r×r when X is (..., r, d) with r ≤ d).

    Iteration (Algorithm 2 of SPECTRA):
        α ≥ σ_max(G);  Y_0 = G / α;  Z_0 = I.
        for k = 0..K-1:
            T_k = (3 I − Z_k Y_k) / 2
            Y_{k+1} = Y_k T_k
            Z_{k+1} = T_k Z_k
        return (Z_K / sqrt(α)) · X.

    Z_K / sqrt(α) approximates G^{-1/2}; left-multiplying X recovers H_c.

    Assumes X has been pre-rescaled so σ_max(X) ≲ 1 (which holds inside
    `_chord_tight_clean_polar_pipeline` after §2.5 normalization). The
    Schulz iteration is contractive when σ(G/α) ∈ (0, 3], guaranteed by
    α = trace(G) + ε ≥ σ_max(G). Cubic convergence near σ=1; for c ≪ 1
    the gram has larger condition number and more iters (≥10) are needed.

    Always returns float32 (consistent with `_newton_schulz_gram_batched`
    output convention for downstream chord-tight-clean pipeline).
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X.float()
    # Tall-transpose so the second-to-last dim is the smaller side; the
    # gram (I + X X^T / c²) is then (r, r) with r ≤ d, matching the FLOP
    # win of `_newton_schulz_gram_batched`. SSC is left-multiplicative on
    # the smaller-side basis, so the transpose is undone at the end.
    Xf = X.float()
    tall = Xf.shape[-2] > Xf.shape[-1]
    if tall:
        Xf = Xf.transpose(-2, -1)
    orig_leading = Xf.shape[:-2]
    Xf = Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1])
    batch, r, d = Xf.shape
    device, dtype = Xf.device, Xf.dtype
    # `c` may be a scalar (fixed-c SSC) or a (batch,) tensor (κ-adaptive SSC,
    # one solved c per pair). Broadcast either to (batch, 1, 1) so the gram
    # G = I + X X^T / c² is correct per-batch.
    if torch.is_tensor(c):
        inv_c2 = (1.0 / c.to(device=device, dtype=dtype).pow(2)).view(-1, 1, 1)
        # baddbmm requires alpha to be a Python scalar — promote via expand
        # on the gram path below instead. Compute scaled outer product manually.
        I_r = torch.eye(r, device=device, dtype=dtype).expand(batch, r, r)
        G = I_r + inv_c2 * torch.bmm(Xf, Xf.transpose(-2, -1))
    else:
        inv_c2 = 1.0 / (float(c) ** 2)
        I_r = torch.eye(r, device=device, dtype=dtype).expand(batch, r, r)
        # G = I + X X^T / c² is symmetric PSD with eigenvalues ≥ 1.
        G = torch.baddbmm(I_r, Xf, Xf.transpose(-2, -1), alpha=inv_c2, beta=1.0)
    # α ≥ σ_max(G): use trace as a cheap upper bound (sum of eigvals ≥ max eigval).
    alpha = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)  # (batch,)
    alpha_b11 = alpha.view(-1, 1, 1)
    Y = G / alpha_b11
    Z = I_r.clone()
    half = 0.5
    three_I = 3.0 * I_r
    for _ in range(nsteps):
        T = half * (three_I - torch.bmm(Z, Y))
        Y = torch.bmm(Y, T)
        Z = torch.bmm(T, Z)
    # Z ≈ G^{-1/2} · sqrt(α);  so G^{-1/2} ≈ Z / sqrt(α).
    G_inv_half = Z / alpha_b11.sqrt()
    out = torch.bmm(G_inv_half, Xf)  # (batch, r, d)
    out = out.reshape(*orig_leading, *out.shape[-2:])
    return out.transpose(-2, -1) if tall else out


def _solve_c_from_kappa_batched(s_sq, kappa, c_lo=1e-3, c_hi=1e3, iters=40):
    """Solve per-batch for the SSC threshold c that realizes a target
    rank-normalized energy κ.

    Defining h_c(s) = s / sqrt(1 + (s/c)²), the rank-normalized energy is
        κ(c) := (1/r) Σ_i (h_c(s_i) / h_c(1))²
              = (1+1/c²)/r · Σ_i  s_i² / (1 + s_i²/c²).
    κ is monotone DECREASING in c on (0, ∞) for any spectrum with s_i ∈ [0, 1],
    s_max = 1: small c flattens every direction toward the same magnitude (κ →
    rank/r, near 1), large c approaches identity (κ → ‖s‖² / r, can be small
    when the input spectrum is concentrated). Bisection in log-c is well-posed.

    Inputs
    ------
    s_sq : (N, r) tensor — squared singular values normalized so each row's
        max is 1 (the §2.5-rescaled input spectrum, in squared form so we
        avoid an extra sqrt — eigvalsh gives σ² directly).
    kappa : float — target κ. Should lie in (mean(s_sq)/r, 1].

    Returns
    -------
    c : (N,) tensor — solved c per pair.
    """
    device, dtype = s_sq.device, s_sq.dtype
    N, r = s_sq.shape
    lo = torch.full((N,), float(c_lo), device=device, dtype=dtype)
    hi = torch.full((N,), float(c_hi), device=device, dtype=dtype)
    log_lo, log_hi = lo.log(), hi.log()
    target = torch.full((N,), float(kappa), device=device, dtype=dtype)
    for _ in range(iters):
        log_mid = 0.5 * (log_lo + log_hi)
        c = log_mid.exp()
        inv_c2 = (1.0 / c.pow(2)).unsqueeze(-1)            # (N, 1)
        # κ(c) = (1/r) Σ s² (1 + 1/c²) / (1 + s²/c²)
        num = s_sq * (1.0 + inv_c2)
        den = 1.0 + s_sq * inv_c2
        kappa_at_c = (num / den).mean(dim=-1)              # (N,)
        # κ is monotone DECREASING in c → if κ(c) < target, lower c (raise κ).
        too_low = kappa_at_c < target
        log_hi = torch.where(too_low, log_mid, log_hi)
        log_lo = torch.where(too_low, log_lo, log_mid)
    return (0.5 * (log_lo + log_hi)).exp()


def _ssc_c_out_of_solver_domain(c, *, device, dtype, c_lo=1e-3, c_hi=1e3):
    c = c.to(device=device, dtype=dtype).reshape(-1)
    return (
        ~torch.isfinite(c)
        | (c < float(c_lo))
        | (c > float(c_hi))
    )


def _sanitize_ssc_c_init(c_init, *, device, dtype, c_lo=1e-3, c_hi=1e3):
    c = c_init.to(device=device, dtype=dtype).reshape(-1)
    c = torch.nan_to_num(
        c,
        nan=1.0,
        posinf=float(c_hi),
        neginf=float(c_lo),
    )
    return c.clamp(min=float(c_lo), max=float(c_hi))


def _ssc_misr_bisect_batched(X, kappa, K=3, nsteps=10, eps=1e-12,
                              c_init=None, log_window=0.5,
                              c_lo=1e-3, c_hi=1e3):
    """Solve c via warm-started bisection in log-c using MISR forward as the
    κ evaluator. K MISR runs total; final run is the apply.

    For each candidate c, κ(c) = ‖H_c(X)‖²_F / (r · σ_max(H_c(X))²) =
    stable_rank(H_c(X)) / r. Bisection on this against the target κ.

    Warm-start: if `c_init` (N,) provided, initial bracket is [c_init/e^window,
    c_init · e^window] so K=3 gives a factor e^(window/4) ≈ 1.13× accuracy at
    window=0.5. Without c_init, full range [1e-3, 1e3] (K=3 too coarse — fall
    back to eigvalsh first step).

    Returns (out, c_solved).
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X.float(), torch.zeros(X.shape[:-2], device=X.device)
    Xf = X.float()
    tall = Xf.shape[-2] > Xf.shape[-1]
    if tall:
        Xf = Xf.transpose(-2, -1)
    orig_leading = Xf.shape[:-2]
    Xf_flat = Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1])
    N, r, _ = Xf_flat.shape
    device, dtype = Xf_flat.device, Xf_flat.dtype

    # Warm-started log-c bracket. Without init, fall through full range; the
    # first-step caller is expected to use eigvalsh (or warmup) to seed c_init.
    log_floor = torch.tensor(float(c_lo), device=device, dtype=dtype).log()
    log_ceil = torch.tensor(float(c_hi), device=device, dtype=dtype).log()
    if c_init is not None:
        c_init = _sanitize_ssc_c_init(
            c_init, device=device, dtype=dtype, c_lo=c_lo, c_hi=c_hi,
        )
        log_c_mid = c_init.log()
        log_lo = (log_c_mid - log_window).clamp(min=log_floor, max=log_ceil)
        log_hi = (log_c_mid + log_window).clamp(min=log_floor, max=log_ceil)
    else:
        log_lo = log_floor.expand(N).clone()
        log_hi = log_ceil.expand(N).clone()
    target = torch.full((N,), float(kappa), device=device, dtype=dtype)

    last_out = None
    last_c = None
    for k in range(K):
        log_mid = 0.5 * (log_lo + log_hi)
        c = log_mid.exp()                                            # (N,)
        H = _ssc_misr_batched(Xf_flat, c=c, nsteps=nsteps)           # (N, r, d)
        # κ(H) = ‖H‖²_F / (r · σ_max(H)²).
        # Closed form for σ_max(H_c(X)) when X is pre-rescaled (σ_max(X)=1):
        #     σ_max(H_c(X)) = h_c(1) = c / √(c²+1).
        # This assumes MISR has converged; the production sequential path is
        # kept for parity, while the parallel path can use a larger eval pass.
        F2 = (H ** 2).sum(dim=(-2, -1)).clamp_min(eps)               # (N,)
        sigma_max_H_sq = c.pow(2) / (c.pow(2) + 1.0)                 # (N,)
        kappa_current = F2 / (r * sigma_max_H_sq.clamp_min(eps))     # (N,)
        # κ is monotone DECREASING in c (small c flattens spectrum → high κ).
        too_high = kappa_current > target
        log_lo = torch.where(too_high, log_mid, log_lo)
        log_hi = torch.where(too_high, log_hi, log_mid)
        last_out = H
        last_c = c

    # last_out is from the Kth bisection step; that's our apply.
    out = last_out.reshape(*orig_leading, *last_out.shape[-2:])
    c_solved = last_c.reshape(*orig_leading) if orig_leading else last_c
    return (out.transpose(-2, -1) if tall else out), c_solved


def _ssc_misr_bisect_batched_kpar(X, kappa, K=3, nsteps=10, eps=1e-12,
                                   c_init=None, log_window=0.5,
                                   nsteps_eval=None, c_lo=1e-3,
                                   c_hi=1e3, return_info=False):
    """K-way batched parallel-grid variant of `_ssc_misr_bisect_batched`.

    Instead of K sequential bisection steps (each launching one MISR call on
    N pairs), place K candidate c values uniformly in log-c on the bracket
    [log(c_init) - log_window, log(c_init) + log_window] per pair, run ONE
    MISR call on a (K*N, r, d) batched input with c of shape (K*N,), and
    post-select the candidate whose κ_current is closest to the target.

    Tradeoff vs. sequential bisection
    ---------------------------------
    Sequential bisect halves the bracket each step → K=3 narrows to
    log_window/2^3 = log_window/8 (≈ ±0.0625 in log-c for window=0.5).
    Parallel-grid is a one-shot evaluation; K=3 candidates over a full
    ±log_window bracket land on a residual grid of log_window/(K-1) ≈
    log_window/2 (≈ ±0.25 in log-c for K=3, window=0.5). To match K=3
    sequential we need K=9 parallel (or shrink log_window 4×). The win is
    a single batched matmul launch instead of K — at small r, MISR is
    launch-bound and the parallel cost is roughly that of one sequential
    iteration, so for launch-bound regimes (e.g. r=256, small N) it pays
    even at K=5–9.

    Returns (out, c_solved). If return_info=True, returns
    (out, c_solved, info), where info includes whether the warm start had to
    be clamped to the solver domain and whether the selected candidate was on
    the parallel-grid edge. Same shapes/semantics as the sequential path.
    `c_init` is REQUIRED here (parallel grid only makes sense with a warm
    start — full-range parallel-3 would be ~10× coarser than parallel-3
    on a warm bracket).
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X.float(), torch.zeros(X.shape[:-2], device=X.device)
    if c_init is None:
        raise ValueError(
            "_ssc_misr_bisect_batched_kpar requires c_init (warm start); "
            "full-range parallel-grid is too coarse to be useful."
        )
    if K < 1:
        raise ValueError(f"K must be ≥ 1, got {K}")
    # The κ scorer uses the closed-form σmax of the converged SSC map. At
    # production apply nsteps=10 the small-c candidates at r=256 can be
    # under-converged, so use a more-converged eval pass for winner selection
    # while keeping the winner apply at `nsteps`.
    if nsteps_eval is None:
        nsteps_eval = 2 * nsteps

    Xf = X.float()
    tall = Xf.shape[-2] > Xf.shape[-1]
    if tall:
        Xf = Xf.transpose(-2, -1)
    orig_leading = Xf.shape[:-2]
    Xf_flat = Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1])
    N, r, d = Xf_flat.shape
    device, dtype = Xf_flat.device, Xf_flat.dtype

    c_init_raw = c_init.to(device=device, dtype=dtype).reshape(-1)
    c_init_clamped = _ssc_c_out_of_solver_domain(
        c_init_raw, device=device, dtype=dtype, c_lo=c_lo, c_hi=c_hi,
    )
    c_init = _sanitize_ssc_c_init(
        c_init_raw, device=device, dtype=dtype, c_lo=c_lo, c_hi=c_hi,
    )
    log_c_mid = c_init.log().reshape(-1)  # (N,)
    log_floor = torch.tensor(float(c_lo), device=device, dtype=dtype).log()
    log_ceil = torch.tensor(float(c_hi), device=device, dtype=dtype).log()
    if K == 1:
        offsets = torch.zeros(1, device=device, dtype=dtype)
    else:
        offsets = torch.linspace(-log_window, log_window, K,
                                  device=device, dtype=dtype)             # (K,)
    # (K, N) grid of log-c candidates.
    log_c_grid = log_c_mid.unsqueeze(0) + offsets.unsqueeze(1)            # (K, N)
    log_c_grid = log_c_grid.clamp(min=log_floor, max=log_ceil)
    c_grid = log_c_grid.exp()                                              # (K, N)

    # Stack X along leading dim K. We broadcast the SAME X across the K
    # candidates (X doesn't depend on c). Total batched MISR input is
    # (K*N, r, d) with c of shape (K*N,).
    X_rep = Xf_flat.unsqueeze(0).expand(K, N, r, d).reshape(K * N, r, d)
    c_flat = c_grid.reshape(K * N)
    # Eval pass: run MISR at nsteps_eval (usually 2× apply nsteps) so the
    # closed-form σmax used by the κ scorer is accurate enough at small c.
    H_eval = _ssc_misr_batched(X_rep, c=c_flat, nsteps=nsteps_eval, eps=eps)
    H_eval = H_eval.reshape(K, N, r, d)

    F2 = (H_eval ** 2).sum(dim=(-2, -1)).clamp_min(eps)                   # (K, N)
    sigma_max_H_sq = c_grid.pow(2) / (c_grid.pow(2) + 1.0)                # (K, N)
    kappa_grid = F2 / (r * sigma_max_H_sq.clamp_min(eps))                 # (K, N)

    target = torch.full((N,), float(kappa), device=device, dtype=dtype)
    # Pick the candidate index (per pair) minimizing |κ(c_k) - target|.
    diffs = (kappa_grid - target.unsqueeze(0)).abs()                       # (K, N)
    diffs = torch.where(torch.isfinite(diffs), diffs, torch.full_like(diffs, float("inf")))
    winner = diffs.argmin(dim=0)                                           # (N,)
    pair_idx = torch.arange(N, device=device)
    c_solved = c_grid[winner, pair_idx]                                    # (N,)
    if nsteps_eval == nsteps:
        out_flat = H_eval[winner, pair_idx]                                # (N, r, d)
    else:
        out_flat = _ssc_misr_batched(Xf_flat, c=c_solved, nsteps=nsteps, eps=eps)

    out = out_flat.reshape(*orig_leading, *out_flat.shape[-2:])
    c_out = c_solved.reshape(*orig_leading) if orig_leading else c_solved
    out = out.transpose(-2, -1) if tall else out
    if return_info:
        if K == 1:
            edge_hit = torch.zeros_like(winner, dtype=torch.bool)
        else:
            edge_hit = (winner == 0) | (winner == K - 1)

        def _info_shape(v):
            return v.reshape(*orig_leading) if orig_leading else v

        info = {
            "c_init_clamped": _info_shape(c_init_clamped),
            "edge_hit": _info_shape(edge_hit),
            "winner": _info_shape(winner),
        }
        return out, c_out, info
    return out, c_out


def _ssc_adaptive_kappa_batched(X, kappa, nsteps=10, eps=1e-12,
                                c_lo=1e-3, c_hi=1e3, bisect_iters=40):
    """SPECTRA Soft Spectral Clipping with the threshold c solved per-pair
    per-step from a target rank-normalized energy κ.

    Pipeline:
      1. G_X = X X^T (r×r per pair — one bmm).
      2. λ_i = eigvalsh(G_X) (eigenvalues only, no eigenvectors).
      3. s_sq = λ / λ.max — pre-rescaled squared spectrum, max = 1.
      4. c = _solve_c_from_kappa_batched(s_sq, κ) — log-bisection in c.
      5. Apply existing MISR kernel with that per-pair c.

    Returns (H_c(X), c) where c is (N,) for diagnostic logging. No SVD;
    eigvalsh is launch-bound on small r×r at production scale, so this is
    intended as the upper-bound-on-quality probe before designing a refresh-
    schedule amortization. See `algorithm_tight_chord.md` §C.5 for the
    state-dependent interpretation of c.
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X.float(), torch.zeros(X.shape[:-2], device=X.device)
    Xf = X.float()
    tall = Xf.shape[-2] > Xf.shape[-1]
    if tall:
        Xf = Xf.transpose(-2, -1)
    orig_leading = Xf.shape[:-2]
    Xf_flat = Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1])
    batch, r, d = Xf_flat.shape
    # Use the smaller-side gram (r×r) so eigvalsh stays on small matrices.
    G_X = torch.bmm(Xf_flat, Xf_flat.transpose(-2, -1))
    # PSD ⇒ eigvalsh returns ascending real eigenvalues. Add a tiny floor for
    # rank-deficient inputs (the post-§2.5-rescale path generally has a clean
    # top eigenvalue but the tail can be ≈ 0 in finite precision).
    lam = torch.linalg.eigvalsh(G_X).clamp_min(0.0)          # (batch, r)
    lam_max = lam.max(dim=-1, keepdim=True).values.clamp_min(eps)
    s_sq = lam / lam_max                                     # max-normalized to 1
    c = _solve_c_from_kappa_batched(s_sq, kappa,
                                    c_lo=c_lo, c_hi=c_hi, iters=bisect_iters)
    # Apply MISR with per-pair c. Untranspose was undone by reshape above;
    # _ssc_misr_batched handles its own tall-transpose, so pass Xf_flat as-is
    # (after our outer transpose).
    out_flat = _ssc_misr_batched(Xf_flat, c=c, nsteps=nsteps, eps=eps)
    out = out_flat.reshape(*orig_leading, *out_flat.shape[-2:])
    c_out = c.reshape(*orig_leading)
    return (out.transpose(-2, -1) if tall else out), c_out


def _stable_rank_c_from_kappa_batched(X, kappa, eps=1e-6,
                                      c_lo=1e-3, c_hi=1e3):
    """Cheap one-spike-plus-flat-tail SSC c estimate from stable rank.

    Assumes X is already in the chord-tight §2.5 convention with
    σ_max(X) ~= 1. The estimate replaces the full normalized spectrum by
    one spike at 1 plus a flat tail whose mean squared singular value is

        m = (r * μ - 1) / (r - 1),   μ = ||X||_F^2 / r.

    Solving the flat-tail κ equation gives

        c^2 = m * (1 - κ_tail) / (κ_tail - m),
        κ_tail = (r * κ - 1) / (r - 1).

    This is stateless and uses only a Frobenius-norm reduction. It deliberately
    does not warm-start or cache c, avoiding the kpar edge-ratchet failure mode.
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return torch.zeros(X.shape[:-2], device=X.device)
    Xf = X.float()
    tall = Xf.shape[-2] > Xf.shape[-1]
    if tall:
        Xf = Xf.transpose(-2, -1)
    orig_leading = Xf.shape[:-2]
    Xf_flat = Xf.reshape(-1, Xf.shape[-2], Xf.shape[-1])
    N, r, _ = Xf_flat.shape
    device, dtype = Xf_flat.device, Xf_flat.dtype

    if r <= 1:
        return torch.full((*orig_leading,), float(c_hi),
                          device=device, dtype=dtype)

    mu = Xf_flat.square().sum(dim=(-2, -1)) / float(r)
    m_tail = ((float(r) * mu - 1.0) / float(r - 1)).clamp(
        float(eps), 1.0 - float(eps),
    )
    k_tail_value = (float(r) * float(kappa) - 1.0) / float(r - 1)
    k_tail = torch.full_like(m_tail, k_tail_value).clamp(
        float(eps), 1.0 - float(eps),
    )

    denom = (k_tail - m_tail).clamp_min(float(eps))
    c2 = m_tail * (1.0 - k_tail) / denom
    c = c2.sqrt().clamp(float(c_lo), float(c_hi))
    c = torch.where(k_tail <= m_tail + float(eps),
                    torch.full_like(c, float(c_hi)), c)
    c = torch.where(k_tail >= 1.0 - float(eps),
                    torch.full_like(c, float(c_lo)), c)
    c = torch.nan_to_num(
        c, nan=float(c_hi), posinf=float(c_hi), neginf=float(c_lo),
    ).clamp(float(c_lo), float(c_hi))
    return c.reshape(*orig_leading)


def _ssc_adaptive_stable_rank_batched(X, kappa, nsteps=10, eps=1e-12,
                                      c_lo=1e-3, c_hi=1e3):
    """SSC with c chosen by `_stable_rank_c_from_kappa_batched`."""
    c = _stable_rank_c_from_kappa_batched(
        X, kappa=kappa, c_lo=c_lo, c_hi=c_hi,
    )
    out = _ssc_misr_batched(X, c=c, nsteps=nsteps, eps=eps)
    return out, c


def _polar_retract(X, nsteps=5, eps=1e-7):
    """Polar retraction for near-orthonormal X (singular values already ≈ 1).

    Unlike `_newton_schulz`, does NOT pre-normalize by ‖X‖_F. The Frobenius
    normalization there divides σ by √min(d, r), which for tall (d≫r) inputs
    pushes σ far below NS's basin of attraction near 1 — five iterations only
    recover σ ≈ 0.97, leaving 3% orthogonality error.

    For Stiefel retraction U ← polar(U + dU) where dU is a small Stiefel-tangent
    perturbation, σ(U + dU) ≈ √(1 + ε) with ε ≈ ‖dU‖_F² / r ≪ 1 — already in
    NS's basin. Five quintic iterations drive the error to fp32 precision.

    Caller must guarantee σ(X) is in (0, √3]. For arbitrary X (Adam direction,
    raw gradient), use `_newton_schulz` instead.
    """
    X = X.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
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
    """PolarExpress orthogonalization (Amsel et al., arXiv:2505.16932) —
    REFERENCE rectangular implementation. Kept as the numeric baseline for
    `_polar_express_gram` unit tests; production callers should use the
    Gram form, which is mathematically equivalent and ~Nsteps× cheaper at
    rank r ≪ d.
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


def _polar_express_gram_batched(
    X,
    nsteps=5,
    eps=1e-7,
    dtype=torch.float32,
    safety_factor=1.05,
    restart_at=2,
    pre_norm="frob",
):
    """Batched Stabilized Gram-form PolarExpress (Dao 2026, Algorithm 3, with
    Amsel quintic Remez coefficients arXiv:2505.16932 in place of cubic Muon).
    Mirrors `_newton_schulz_gram_batched` but uses the degree-5 polynomial
    M_t = a_t I + b_t R + c_t R² with per-iter (a, b, c) from
    `_POLAR_EXPRESS_COEFFS`.

    Per Dao 2026 §"When to Restart: Polar Express Coefficients for Muon":
    Polar Express coefficients have larger per-iter growth (≈3.5) than
    cubic Muon (2.25), so the negative-eigenvalue blowup is faster. The
    paper recommends restart_at=2 and safety_factor=1.02–1.05 (more
    conservative end). We default to safety_factor=1.05 and restart_at=2,
    matching the paper's stability recommendation. At nsteps=5 the restart
    sits at iter 2; at nsteps=10 it still fires once at iter 2 then runs
    the remaining 8 iters from the restart point.

    Per-iter (in Gram space):
        R2 = R @ R                          (r,r matmul)
        M  = a I + b R + c R2               (elementwise)
        Q  ← M Q                            (r,r matmul)
        R  ← M R M                          (two r,r matmuls)
    Restart at iter τ: X_iter ← Q_τ · X_normed; reform R; reset Q ← I.
    Final: X_out = Q · X (last applied X iterate). One initial r²d matmul
    (X X^T) and at most two r²d matmuls for X-reconstruction (one per
    restart + final). fp32 throughout: Polar Express is sensitive enough
    that fp16 Gram NS needs MORE restarts; defer fp16 path until tested.

    Input X is (..., r, d) (or (..., d, r) — auto-transposed). Returns
    Q · X_normed in the original orientation. Pre-normalizes by Frobenius
    norm × safety_factor.
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X
    X = X.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)
    orig_leading = X.shape[:-2]
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    r = X.shape[-2]
    if pre_norm == "frob":
        norm = X.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1) + eps
        X_normed = X / (norm * safety_factor)
    elif pre_norm == "spec":
        smax = torch.linalg.matrix_norm(X, ord=2).unsqueeze(-1).unsqueeze(-1) + eps
        X_normed = X / (smax * safety_factor)
    elif pre_norm == "none":
        X_normed = X / safety_factor
    else:
        raise ValueError(f"pre_norm must be one of {{'frob','spec','none'}}, got {pre_norm!r}")

    iter_dtype = dtype
    X0_iter = X_normed.to(iter_dtype)
    coeffs = _POLAR_EXPRESS_COEFFS[:nsteps]
    if len(coeffs) < nsteps:
        coeffs = coeffs + [_POLAR_EXPRESS_COEFFS[-1]] * (nsteps - len(coeffs))

    use_tf32_guard = (iter_dtype == torch.float32) and X.is_cuda
    prev_tf32_matmul = None
    if use_tf32_guard:
        prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        eye = torch.eye(r, dtype=iter_dtype, device=X0_iter.device)
        eye_b = eye.expand(*X0_iter.shape[:-2], r, r)
        R = X0_iter @ X0_iter.transpose(-2, -1)            # (..., r, r) PSD
        Q = eye_b.clone()
        for t, (a, b, c) in enumerate(coeffs, start=1):
            # Restart at iter τ (Dao Algorithm 3 step 8): re-form R from
            # the current X iterate, reset Q. Caps the exponential blowup
            # of any spurious negative eigenvalues of R at fp16/fp32 noise
            # floor. For polar_express the paper recommends τ=2.
            if restart_at is not None and t == restart_at + 1:
                X0_iter = Q @ X0_iter                      # (..., r, d)
                R = X0_iter @ X0_iter.transpose(-2, -1)    # (..., r, r)
                Q = eye_b.clone()
            R2 = R @ R                                     # (..., r, r)
            # M = a I + b R + c R2  (symmetric, commutes with R)
            M = (b * R) + (c * R2)
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            new_Q = M @ Q                                  # (..., r, r)
            MR = M @ R                                     # (..., r, r)
            new_R = MR @ M                                 # (..., r, r) PSD
            Q = new_Q
            R = new_R
    finally:
        if use_tf32_guard:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul

    out_iter = Q @ X0_iter                                 # (..., r, d)
    out = out_iter.float()
    out = out.reshape(*orig_leading, *out.shape[-2:])
    return out.transpose(-2, -1) if tall else out


def _polar_express_gram(X, nsteps=5, eps=1e-7):
    """Non-batched alias for `_polar_express_gram_batched` — handles 2D
    input by adding/removing a singleton batch dim. Kept for parity with
    the rectangular `_polar_express` signature."""
    if X.dim() == 2:
        return _polar_express_gram_batched(X.unsqueeze(0), nsteps=nsteps, eps=eps).squeeze(0)
    return _polar_express_gram_batched(X, nsteps=nsteps, eps=eps)


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


class AdamOrthogonalCoreLoRA(Optimizer):
    """
    Orthogonal-core LoRA optimizer (UCV^T).

    Spec: docs/notes/polar_product/orthogonal_core_lora_2026_05_03.md.

    Per training step on triples (U, C, V):
      1. Project subspace gradients to the Stiefel tangent at U, V:
             gU ← (I - U U^T) gU,   gV ← (I - V V^T) gV.
      2. Adam EMA on (m_U, v_U), (m_C, v_C), (m_V, v_V) with bias correction;
         get u_U, u_C, u_V.
      3. Polar (Newton-Schulz) on u_U, u_V; RMS-match scaling
             dU = -lr * (||u_U||_F / (||P_U||_F + eps)) * P_U.
         Plain Adam step on the core: dC = -lr * u_C.
      4. Retract via NS polar: U ← polar(U + dU), V ← polar(V + dV);
         additive on core: C ← C + dC.

    No Picard loop, no k hyperparameter, no core remix coefficient.

    Weight decay (decoupled, AdamW-style) applied to C only — U, V live on
    the Stiefel manifold and shouldn't be shrunk.
    """

    def __init__(self, model, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, ns_steps=5):
        triples = collect_ucv_triples(model)
        if not triples:
            raise ValueError(
                "No UCV (U, C, V) triples found on model. Did you call "
                "inject_ucv_adapters() before building the optimizer?"
            )
        params = [p for U, C, V in triples for p in (U, C, V)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.triples = triples
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.pair_state = {
            i: {
                "m_U": torch.zeros_like(U, dtype=torch.float32),
                "v_U": torch.zeros_like(U, dtype=torch.float32),
                "m_C": torch.zeros_like(C, dtype=torch.float32),
                "v_C": torch.zeros_like(C, dtype=torch.float32),
                "m_V": torch.zeros_like(V, dtype=torch.float32),
                "v_V": torch.zeros_like(V, dtype=torch.float32),
                "step": 0,
            }
            for i, (U, C, V) in enumerate(triples)
        }

    @staticmethod
    def _adam_dir(grad, m, v, beta1, beta2, t, eps):
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        return (m / bc1) / ((v / bc2).sqrt() + eps)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        for i, (U, C, V) in enumerate(self.triples):
            if U.grad is None or C.grad is None or V.grad is None:
                raise ValueError(
                    "AdamOrthogonalCoreLoRA requires gradients on U, C, and V."
                )
            state = self.pair_state[i]
            state["step"] += 1
            t = state["step"]

            U32 = U.float()
            V32 = V.float()
            gU = U.grad.float()
            gC = C.grad.float()
            gV = V.grad.float()

            # 1. Stiefel-tangent projection on U, V grads.
            gU = gU - U32 @ (U32.T @ gU)
            gV = gV - V32 @ (V32.T @ gV)

            # 2. Adam directions for all three.
            uU = self._adam_dir(gU, state["m_U"], state["v_U"], self.beta1, self.beta2, t, self.eps)
            uC = self._adam_dir(gC, state["m_C"], state["v_C"], self.beta1, self.beta2, t, self.eps)
            uV = self._adam_dir(gV, state["m_V"], state["v_V"], self.beta1, self.beta2, t, self.eps)

            # 3. Polar / RMS-match on U, V; plain Adam on C.
            if self.ns_steps > 0:
                P_U = _newton_schulz(uU, self.ns_steps)
                P_V = _newton_schulz(uV, self.ns_steps)
            else:
                P_U, P_V = uU, uV
            scale_U = uU.norm() / (P_U.norm() + self.eps)
            scale_V = uV.norm() / (P_V.norm() + self.eps)
            dU = -lr * scale_U * P_U
            dV = -lr * scale_V * P_V
            dC = -lr * uC

            # Decoupled weight decay on C only.
            if self.weight_decay != 0.0:
                dC = dC - lr * self.weight_decay * C.float()

            # 4. Retract subspaces; additive on core.
            # Use _polar_retract (not _newton_schulz) for the retraction step.
            # _newton_schulz pre-normalizes by ‖X‖_F, which divides σ by √r for
            # tall X — five iterations only reach σ ≈ 0.97 (3% error). For
            # near-orthonormal U+dU, _polar_retract skips the normalization and
            # quintic-converges to fp32 precision in 5 iters.
            U_new = _polar_retract(U32 + dU, self.ns_steps) if self.ns_steps > 0 else (U32 + dU)
            V_new = _polar_retract(V32 + dV, self.ns_steps) if self.ns_steps > 0 else (V32 + dV)
            U.copy_(U_new.to(dtype=U.dtype, device=U.device))
            V.copy_(V_new.to(dtype=V.dtype, device=V.device))
            C.add_(dC.to(dtype=C.dtype, device=C.device))

            U.grad.zero_()
            C.grad.zero_()
            V.grad.zero_()


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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=1,
                 precond_method="higham", higham_iters=10,
                 picard_iters=1, end_rms_align=False, picard_alpha=1.0,
                 htmuon_p=None,
                 operator_type="polar",
                 polar_norm_dir="frob",
                 polar_sigma_power=None,
                 polar_method="ns",
                 ssc_c=None, ssc_nsteps=10, ssc_kappa=None,
                 ssc_kappa_refresh_every=1, ssc_kappa_warmup_steps=5,
                 ssc_kappa_solver="eigvalsh", ssc_kappa_bisect_iters=3,
                 ssc_kappa_bisect_mode="sequential",
                 ssc_kappa_bisect_nsteps_eval=None,
                 ssc_kappa_cache_share_picard=False,
                 ssc_kappa_cache_ema_beta=None,
                 ssc_kappa_cross_group_eigvalsh=True,
                 ssc_kappa_diagnose_eigvalsh=False,
                 ssc_kappa_diagnose_start_step=1,
                 ssc_kappa_diag_ema_beta=None,
                 anderson_m=0, anderson_reg=1e-10,
                 core_remix_alpha=0.0,
                 exact_chord=False,
                 magnitude_rule="adam_frobenius",
                 disable_whitening=False,
                 precond_delta_relative=False,
                 log_non_finite=False,
                 log_non_finite_start_step=1,
                 debug_optimizer_state=False,
                 debug_optimizer_state_every=1,
                 debug_optimizer_state_start_step=1,
                 debug_snapshot_dir=None,
                 debug_snapshot_limit=8,
                 debug_abort_on_non_finite=False,
                 ns_form="gram",
                 higham_compute_dtype="fp32",
                 fw_linearization="anchored",
                 curvature_whitening=False, curvature_beta=0.99):
        named = collect_lora_pairs_named(model, adapter_name)
        if not named:
            raise ValueError("No LoRA (A,B) tensors found on model.")
        pairs = [(A, B) for A, B, _ in named]
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.pair_names = [n for _, _, n in named]
        self.delta = delta
        self.precond_delta_relative = bool(precond_delta_relative)
        # Curvature whitening (experimental): replace the geometric factor Grams
        # (S_B=BᵀB whitens A-update, S_A=AAᵀ whitens B-update) with EMAs of the
        # factor-gradient outer products (S_curv_A=EMA(g_A g_Aᵀ)=BᵀHB whitens the
        # A-update; S_curv_B=EMA(g_Bᵀ g_B) whitens the B-update). Everything else
        # (polar, κ/SSC, ρ, Picard) is unchanged — isolates metric vs geometry.
        self.curvature_whitening = bool(curvature_whitening)
        # β_c = 0.99 matches SOAP's tuned preconditioner (Kronecker-factor) EMA
        # β_shampoo (Vyas et al. 2024; their L←β₂L+(1-β₂)GGᵀ is our S_curv).
        # Distinct from this optimizer's Adam β2 (=0.999, elementwise v_A).
        self.curvature_beta = float(curvature_beta)
        # `log_non_finite`: when True, run the top-of-step per-pair
        # (A, B, grad_A, grad_B) isfinite check AND the end-of-step
        # chain-of-intermediates check. Both add measurable overhead at
        # high rank (~10% total wall at r=256, ~448 + ~20*N isfinite
        # kernel launches per step). Default OFF; turn on for
        # NaN-debugging runs.
        self.log_non_finite = bool(log_non_finite)
        self.log_non_finite_start_step = max(1, int(log_non_finite_start_step))
        self.debug_optimizer_state = bool(debug_optimizer_state)
        self.debug_optimizer_state_every = max(1, int(debug_optimizer_state_every))
        self.debug_optimizer_state_start_step = max(
            1, int(debug_optimizer_state_start_step)
        )
        self.debug_snapshot_dir = debug_snapshot_dir
        self.debug_snapshot_limit = int(debug_snapshot_limit)
        self.debug_abort_on_non_finite = bool(debug_abort_on_non_finite)
        self._debug_snapshots_written = 0
        # ns_form: "gram" (default, _newton_schulz_gram_batched — Dao 2026
        # Algorithm 3, fp16+restart, our reimplementation; production path),
        # "rect" (_newton_schulz_batched on rectangular (r,d) — legacy, kept
        # for trajectory comparisons against pre-gram sweeps), or
        # "gram-norestart" (same as gram, but restart_at=None — drops the
        # stability hedge for the FLOP headroom; validated to track rect-fp32
        # on Tier 1 corpus + tight-damping rebuild at cubic-Muon NS=5, see
        # tests/test_ns_gram.py).
        # Only consulted by `_chord_tight_clean_polar_pipeline`; other
        # magnitude_rules ignore it.
        if ns_form not in ("rect", "gram", "gram-norestart"):
            raise ValueError(
                f"ns_form must be 'rect', 'gram', or 'gram-norestart', got {ns_form!r}"
            )
        self.ns_form = ns_form
        if fw_linearization not in ("anchored", "full"):
            raise ValueError(
                f"fw_linearization must be 'anchored' or 'full', got {fw_linearization!r}"
            )
        self.fw_linearization = fw_linearization
        # Inner-iteration dtype for `spd_inv_sqrt_higham_batched`.
        # "fp32" (default) preserves the validated fp32-no-TF32 path;
        # "fp16" opts into the variant-B mixed-precision body (fp16
        # inner + 1 fp32 polish iter). See
        # `docs/notes/polar_product/algorithm_clean_implementation.md`
        # §7 and `scripts/bench/bench_higham_variants.py` for the
        # rationale and per-rank wins.
        if higham_compute_dtype not in ("fp32", "fp16"):
            raise ValueError(
                f"higham_compute_dtype must be 'fp32' or 'fp16', "
                f"got {higham_compute_dtype!r}"
            )
        self._higham_compute_dtype = (
            torch.float16 if higham_compute_dtype == "fp16" else None
        )
        # Per-step diagnostic stash flag. When True for a single step, the
        # batched path writes the §9 inputs into pair_state under keys
        # "A" / "B" (the factors fed INTO the step, before the in-place
        # update) and "u_A" / "u_B" (the Adam-RMS direction, before σ_max
        # re-normalization). Consumed by --snapshot_steps in train.py;
        # train.py clears the stash after save_checkpoint and resets the
        # flag to False.
        self.snapshot_pair_tensors = False
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.ns_steps = ns_steps
        self.lora_plus_multiplier = lora_plus_multiplier
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        # HTMuon σ → σ^p sub-mode of spectral_chord_tight_clean. None = use
        # NS5 polar output unchanged. When set ∈ (0, 1], the polar output P
        # is left-multiplied by (X X^T)^(p/2) (right-multiplied by (X^T X)^(p/2)
        # on the B-side) so the effective singular-value transfer becomes
        # σ → σ^p instead of σ → 1. p must be a power-of-two reciprocal
        # (0.5, 0.25, 0.125, 0.0625, ...) for the iterated-sqrt primitive
        # to land on it exactly. See docs/papers/htmuon_2603.10067.pdf and
        # scripts/bench/bench_htmuon_op.py for accuracy/timing.
        self.htmuon_p = htmuon_p
        # Experimental core-coordinate remix coefficient. alpha in [0, 1].
        # 0 = no remix (baseline). Nonzero values modify the components of
        # (u_A, u_B) visible through the r x r products u_A A^T and B^T u_B.
        # This is a hypothesis-probe knob, not a validated policy. Applied
        # before the Picard loop / polar pipeline:
        #   tilde_H_A = (1-alpha) H_A - alpha H_B,   H_A = u_A A^T  (r x r)
        # then the projections of (u_A, u_B) onto row(A) / col(B) are
        # replaced by the projections of the remixed core signals.
        if not (0.0 <= core_remix_alpha <= 1.0):
            raise ValueError(f"core_remix_alpha must be in [0, 1], got {core_remix_alpha!r}")
        self.core_remix_alpha = core_remix_alpha
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
        if polar_method not in {"ns", "ns_hybrid", "polar_express", "ssc"}:
            raise ValueError(f"polar_method must be one of ns/ns_hybrid/polar_express/ssc, got {polar_method!r}")
        self.polar_method = polar_method
        # SSC (SPECTRA arXiv:2603.14315): soft spectral clipping h_c(σ) = σ/√(1+(σ/c)²)
        # via Newton-Schulz MISR. c is in units of the §2.5-rescaled polar input
        # where σ_max(X)=1; active clipping range is c ∈ (0, 1]. c → large ≈ identity.
        # SSC has two interpretations of the shape parameter (Appendix C.5 of
        # algorithm_tight_chord.md): fixed ssc_c (η-independent knee location)
        # or state-dependent via ssc_kappa (target rank-normalized energy,
        # solved per-pair per-step for c). Exactly one must be set.
        if polar_method == "ssc":
            n_set = (ssc_c is not None) + (ssc_kappa is not None)
            if n_set != 1:
                raise ValueError(
                    "polar_method='ssc' requires exactly one of {ssc_c, ssc_kappa}; "
                    f"got ssc_c={ssc_c!r}, ssc_kappa={ssc_kappa!r}"
                )
        self.ssc_c = ssc_c
        self.ssc_kappa = ssc_kappa
        self.ssc_nsteps = ssc_nsteps
        # Amortize κ-adaptive eigvalsh+bisection: refresh per-pair cached c
        # every N steps; in between, apply _ssc_misr_batched directly with the
        # cached c. N=1 reproduces per-step solving (current default behavior).
        # Cache lives on the group state per (side, Picard-iter-index) so n=0
        # and n=1 calls have independent caches. Only consulted when
        # polar_method='ssc' and ssc_kappa is not None.
        if ssc_kappa_refresh_every < 1:
            raise ValueError(
                f"ssc_kappa_refresh_every must be ≥ 1, got {ssc_kappa_refresh_every!r}"
            )
        if ssc_kappa_refresh_every != 1 and ssc_kappa is None:
            raise ValueError(
                "ssc_kappa_refresh_every>1 requires --ssc_kappa (κ-adaptive SSC)"
            )
        self.ssc_kappa_refresh_every = int(ssc_kappa_refresh_every)
        # Warmup: refresh-every-step for the first M steps before honoring the
        # refresh-every-N cadence. Motivation: at LoRA init B=0 ⇒ the polar
        # input's spectrum is extremely concentrated; κ-target=0.6 is
        # unreachable and bisection saturates at c_lo, producing a degenerate
        # near-polar c. Holding that saturated c for N>1 steps applies a
        # qualitatively wrong operator. Refresh-every-step lets the spectrum
        # spread out before caching kicks in. M=0 = no warmup.
        if ssc_kappa_warmup_steps < 0:
            raise ValueError(
                f"ssc_kappa_warmup_steps must be ≥ 0, got {ssc_kappa_warmup_steps!r}"
            )
        self.ssc_kappa_warmup_steps = int(ssc_kappa_warmup_steps)
        # κ-solver dispatch: "eigvalsh" = exact bisection on full r×r spectrum
        # (production default); "misr_bisect" = warm-started K-candidate
        # bisection on MISR F-norm, no eigvalsh after warmup; "stable_rank" =
        # stateless one-spike-plus-flat-tail c from ||X||_F²/r.
        if ssc_kappa_solver not in ("eigvalsh", "misr_bisect", "stable_rank"):
            raise ValueError(
                "ssc_kappa_solver must be 'eigvalsh', 'misr_bisect', or "
                f"'stable_rank', got {ssc_kappa_solver!r}"
            )
        if ssc_kappa_solver == "stable_rank" and ssc_kappa_refresh_every != 1:
            raise ValueError(
                "ssc_kappa_solver='stable_rank' is stateless and requires "
                "ssc_kappa_refresh_every=1"
            )
        if ssc_kappa_solver == "stable_rank" and ssc_kappa_cache_share_picard:
            raise ValueError(
                "ssc_kappa_solver='stable_rank' does not use Picard cache sharing"
            )
        if ssc_kappa_solver == "stable_rank" and ssc_kappa_cache_ema_beta is not None:
            raise ValueError(
                "ssc_kappa_solver='stable_rank' does not use cache EMA"
            )
        self.ssc_kappa_solver = ssc_kappa_solver
        self.ssc_kappa_bisect_iters = int(ssc_kappa_bisect_iters)
        # Optimization B: amortize κ-adaptive eigvalsh across shape groups.
        # When True (default — pure speedup at uniform r across groups), a
        # pre-pass in `_step_batched` computes c per (side, Picard iter) by
        # stacking all groups' (Ng, r, r) grams into one (N_total, r, r)
        # eigvalsh + bisection call, then scatters c into each group's
        # `gs['_xgroup_c_pre']` so the per-group polar pipeline skips its
        # own eigvalsh and runs only the MISR apply. 12 launches at 3
        # groups × 2 sides × 2 Picard iters collapse to 4.
        # Requires polar_method='ssc' + ssc_kappa set + solver='eigvalsh' +
        # uniform r across groups; silently disables otherwise.
        self.ssc_kappa_cross_group_eigvalsh = bool(ssc_kappa_cross_group_eigvalsh)
        self.ssc_kappa_diagnose_eigvalsh = bool(ssc_kappa_diagnose_eigvalsh)
        self.ssc_kappa_diagnose_start_step = max(
            1, int(ssc_kappa_diagnose_start_step)
        )
        if ssc_kappa_diag_ema_beta is not None:
            if not (0.0 <= float(ssc_kappa_diag_ema_beta) < 1.0):
                raise ValueError(
                    "ssc_kappa_diag_ema_beta must be in [0, 1) or None, "
                    f"got {ssc_kappa_diag_ema_beta!r}"
                )
            if not self.ssc_kappa_diagnose_eigvalsh:
                raise ValueError(
                    "ssc_kappa_diag_ema_beta requires "
                    "--ssc_kappa_diagnose_eigvalsh"
                )
            self.ssc_kappa_diag_ema_beta = float(ssc_kappa_diag_ema_beta)
        else:
            self.ssc_kappa_diag_ema_beta = None
        # Optimization A: share the κ-adaptive cached c across Picard inner
        # iterations. At picard=2, the cross-coupling correction at n=1 nudges
        # the polar input X_*_eff slightly from the n=0 X_* — snapshot drift
        # analysis (docs/notes/polar_product/walltime_profile.md §Snapshot-
        # derived c-drift) shows |Δc|/c is p50<1.1%, p99<3.3% at production η.
        # Sharing the n=0 c for n=1 halves the eigvalsh / MISR-bisect calls per
        # step, but the production reference used independent per-(side, n)
        # cache slots. Default False preserves that behavior. Only consulted
        # when polar_method='ssc' and ssc_kappa is not None.
        self.ssc_kappa_cache_share_picard = bool(ssc_kappa_cache_share_picard)
        if ssc_kappa_cache_ema_beta is not None:
            if not (0.0 <= float(ssc_kappa_cache_ema_beta) < 1.0):
                raise ValueError(
                    "ssc_kappa_cache_ema_beta must be in [0, 1) or None, "
                    f"got {ssc_kappa_cache_ema_beta!r}"
                )
            if ssc_kappa is None:
                raise ValueError(
                    "ssc_kappa_cache_ema_beta requires --ssc_kappa "
                    "(κ-adaptive SSC)"
                )
            self.ssc_kappa_cache_ema_beta = float(ssc_kappa_cache_ema_beta)
        else:
            self.ssc_kappa_cache_ema_beta = None
        # 'sequential' = K MISR launches, classical bisection (default, backward
        # compat). 'parallel' = K log-spaced candidates evaluated in one
        # batched MISR launch of (K*N, r, d) input; argmin |κ-target| picks
        # the winner. Coarser per K (residual ~log_window/(K-1) vs.
        # log_window/2^K) but only one launch — wins when launch-bound at
        # small r. See `_ssc_misr_bisect_batched_kpar`.
        if ssc_kappa_bisect_mode not in ("sequential", "parallel"):
            raise ValueError(
                f"ssc_kappa_bisect_mode must be 'sequential' or 'parallel', "
                f"got {ssc_kappa_bisect_mode!r}"
            )
        self.ssc_kappa_bisect_mode = ssc_kappa_bisect_mode
        self.ssc_kappa_bisect_nsteps_eval = (
            None if ssc_kappa_bisect_nsteps_eval is None
            else int(ssc_kappa_bisect_nsteps_eval)
        )
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
        # Exact-chord variant: replace the linearization J = B·ΔA + ΔB·A with
        # the actual ΔW = (B+ΔB)·ΔA + ΔB·A = B·ΔA + ΔB·(A+ΔA), which keeps the
        # second-order ΔB·ΔA term. The block-coordinate decomposition still
        # holds: A-subproblem uses B_eff = B + dB_prev (so the A-side
        # preconditioner becomes S_{B+dB_prev}), B-subproblem uses A_eff =
        # A + dA_prev. Cross-coupling corrections: ũ_A = u_A + (1/η)·(B+dB)^T·dB·A,
        # ũ_B = u_B + (1/η)·B·dA·(A+dA)^T. Spectral preconditioners must be
        # recomputed each Picard iterate (k≥1) since A_eff, B_eff change.
        # See docs/notes/polar_product/algorithm.md §2 remark.
        self.exact_chord = bool(exact_chord)
        # Whitening ablation. When True, replace S_A^{-1/2}, S_B^{-1/2} with
        # the identity at every precond_refresh. The polar pipeline then
        # reduces to per-factor polar on the raw Adam direction
        # (≈ algorithm_tight_chord.md §2 program (W) — per-factor Muon-style
        # update). Tests the importance of whitening for chord-tight's
        # variational interpretation.
        self.disable_whitening = bool(disable_whitening)
        # Magnitude rule for the per-block update — Substitution 1 vs 1' in
        # docs/notes/polar_product/algorithm.md §6/§6.1.
        #   "adam_frobenius" (default): rescale per-block direction to
        #     ‖dA‖_F = lr·‖u_A‖_F (current Theorem 1, no variational source).
        #   "spectral_chord":  rescale per-block direction to ‖dA‖_op = ρ where
        #     ρ = lr/(σ_max(A) + σ_max(B) + 1) (Spectron-style chord trust
        #     region, ‖ΔW‖_op ≤ lr by submultiplicativity). Self-dampens at
        #     σ-drift failure mode; lr typically needs to be retuned (~10-30×
        #     larger than the Frobenius rule's lr).
        #   "spectral_chord_tight": same trust region but with the EXACT
        #     quadratic root ρ = (-s + √(s²+4lr))/2 where s = σ_A + σ_B.
        #     Solves ρ²+sρ-lr=0 (chord bound with no slack). Spectron's "+1"
        #     bakes in conservative slack (substitutes ρ²≤ρ); the tight rule
        #     gives ρ ≈ √lr at s→0 (early training, B≈0) and ρ ≈ lr/s at
        #     s→∞. Larger ρ at small s where Spectron under-steps.
        #   "spectral_chord_direction": variant 1 of algorithm_tight_chord.md.
        #     Replace the worst-case scalar s with the direction-aware
        #     a = ‖B·P‖_2 + ‖Q·A‖_2 and the cross-term 1 with b = ‖Q·P‖_2,
        #     where P = geo_A / ‖geo_A‖_2 and Q = geo_B / ‖geo_B‖_2 are the
        #     unit-norm polar directions. λ = (-a + √(a²+4·b·lr))/(2b) (or
        #     lr/a if b=0) is the largest λ such that the direction-aware
        #     bound ‖ΔW‖_2 ≤ a·λ + b·λ² ≤ lr holds — strictly tighter than
        #     spectral_chord_tight's worst-case ρ when P,Q misaligned with B,A
        #     top singular directions. Stage-0 diagnostics
        #     (docs/notes/polar_product/tight_chord_diagnostics_stage0.md)
        #     measured λ_dir_gain ≈ 1.3-1.4 at r=64, growing with training.
        if magnitude_rule not in {"adam_frobenius", "spectral_chord",
                                  "spectral_chord_tight",
                                  "spectral_chord_tight_clean",
                                  "spectral_chord_tight_no_rho",
                                  "spectral_chord_direction"}:
            raise ValueError(
                f"magnitude_rule must be one of {{adam_frobenius, spectral_chord, "
                f"spectral_chord_tight, spectral_chord_tight_clean, "
                f"spectral_chord_tight_no_rho, spectral_chord_direction}}, got "
                f"{magnitude_rule!r}")
        self.magnitude_rule = magnitude_rule

        # spectral_chord_tight_no_rho is the §8 ablation: drop the ρ-routed
        # rescale entirely, apply dA = -lr·D_A directly. The §8 derivation
        # doesn't pin a cross-coupling coefficient for k≥2, so disallow.
        if magnitude_rule == "spectral_chord_tight_no_rho" and picard_iters > 1:
            raise ValueError(
                f"spectral_chord_tight_no_rho requires picard_iters=1 "
                f"(§8 doesn't derive a cross-coupling coefficient without ρ); "
                f"got picard_iters={picard_iters}")

        # Refuse at construction if magnitude_rule is one that only the
        # batched path implements AND the config disqualifies from the
        # batched path. Without this guard, the optimizer silently falls
        # back to `_step_per_pair`, which runs the adam_frobenius rescale
        # for these rules (the post-`_polar_pipeline` chord-rescale at
        # line ~6225 doesn't cover them) — produces wrong-magnitude updates
        # and divergence. Better to fail loudly at construction than at
        # step 10 of training.
        if magnitude_rule in ("spectral_chord_tight_clean",
                              "spectral_chord_tight_no_rho"):
            disqualifiers = []
            if self.operator_type != "polar":
                disqualifiers.append(f"operator_type={self.operator_type!r}")
            if self.polar_method not in ("ns", "ssc", "polar_express"):
                disqualifiers.append(f"polar_method={self.polar_method!r}")
            if self.polar_norm_dir != "frob":
                disqualifiers.append(f"polar_norm_dir={self.polar_norm_dir!r}")
            if self.polar_sigma_power is not None:
                disqualifiers.append(
                    f"polar_sigma_power={self.polar_sigma_power!r}")
            if picard_iters > 1 and self.anderson_m > 0:
                disqualifiers.append(
                    f"anderson_m={self.anderson_m} with picard_iters>1")
            if picard_iters > 1 and self.end_rms_align:
                disqualifiers.append("end_rms_align=True with picard_iters>1")
            if disqualifiers:
                raise ValueError(
                    f"magnitude_rule={magnitude_rule!r} is only implemented "
                    f"in the batched optimizer path, but the following "
                    f"config disqualifies it from batched eligibility: "
                    f"{disqualifiers}. The per-pair fallback silently "
                    f"miscomputes the magnitude for these rules; refuse "
                    f"at construction instead of training a broken model."
                )

        # Shape-group bookkeeping for the batched hot path. Pairs with the
        # same (A.shape, B.shape) get stacked into 3-D buffers; per-pair Adam
        # state is a view into the buffer, so both per-pair and batched paths
        # see the same memory through views — behavioral equivalence preserved
        # by construction. The diagnostic path (`_step_per_pair`) and the
        # batched path (`_step_batched`) work on identical state.
        shape_to_pairs: dict[tuple, list[int]] = {}
        for i, (A, B) in enumerate(pairs):
            key = (tuple(A.shape), tuple(B.shape))
            shape_to_pairs.setdefault(key, []).append(i)

        self.group_state: list[dict] = []
        self.pair_state: dict[int, dict] = {}
        for gid, (key, indices) in enumerate(shape_to_pairs.items()):
            A_shape, B_shape = key
            r = A_shape[0]
            d_in = A_shape[1]
            d_out = B_shape[0]
            N = len(indices)
            anchor_A, _ = pairs[indices[0]]
            device = anchor_A.device
            # Group buffers (fp32). Adam moments live here; per-pair state is
            # a view slice. SA_half_inv / SB_half_inv populated by step().
            m_A_buf = torch.zeros(N, *A_shape, dtype=torch.float32, device=device)
            v_A_buf = torch.zeros(N, *A_shape, dtype=torch.float32, device=device)
            m_B_buf = torch.zeros(N, *B_shape, dtype=torch.float32, device=device)
            v_B_buf = torch.zeros(N, *B_shape, dtype=torch.float32, device=device)
            SA_half_buf = torch.zeros(N, r, r, dtype=torch.float32, device=device)
            SB_half_buf = torch.zeros(N, r, r, dtype=torch.float32, device=device)
            # Reusable scratch for stacked A/B/grad copies; refreshed each step.
            A_stack = torch.zeros(N, *A_shape, dtype=torch.float32, device=device)
            B_stack = torch.zeros(N, *B_shape, dtype=torch.float32, device=device)
            gA_stack = torch.zeros(N, *A_shape, dtype=torch.float32, device=device)
            gB_stack = torch.zeros(N, *B_shape, dtype=torch.float32, device=device)
            self.group_state.append({
                'gid': gid,
                'indices': indices,
                'A_shape': A_shape,
                'B_shape': B_shape,
                'r': r,
                'd_in': d_in,
                'd_out': d_out,
                'N': N,
                'm_A': m_A_buf,
                'v_A': v_A_buf,
                'm_B': m_B_buf,
                'v_B': v_B_buf,
                'SA_half_inv': SA_half_buf,
                'SB_half_inv': SB_half_buf,
                'A_stack': A_stack,
                'B_stack': B_stack,
                'gA_stack': gA_stack,
                'gB_stack': gB_stack,
            })
            # Per-pair state holds VIEWS into the group buffers, so legacy
            # accesses (`pair_state[i]['m_A'].mul_(...)`) write through to
            # the buffer that the batched path reads.
            for k, gi in enumerate(indices):
                self.pair_state[gi] = {
                    'm_A': m_A_buf[k],
                    'v_A': v_A_buf[k],
                    'm_B': m_B_buf[k],
                    'v_B': v_B_buf[k],
                    '_group': gid,
                    '_local_idx': k,
                    'step': 0,
                }
        self._n_groups = len(self.group_state)

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

    def effective_config(self) -> dict:
        """Resolved behavioral fields for the cfg event `optimizer_effective`
        block. Calls the shared `resolve_effective_inner_polar` so the loader
        and the runtime cannot drift. Inherited by subclasses; override only
        if a subclass changes the precedence (none currently do).
        """
        out: dict = {"effective_picard_iters": int(self.picard_iters)}
        eff = resolve_effective_inner_polar(
            getattr(self, "polar_sigma_power", None),
            getattr(self, "polar_method", "ns"),
            optimizer_class_name=type(self).__name__,
        )
        if eff is not None:
            out["effective_inner_polar"] = eff["label"]
        # Polar pre-norm regime that the NS/polar-express primitive actually
        # received. For chord-tight-clean (magnitude_rule="spectral_chord_tight_clean")
        # we now pass pre_norm="none" because §2.5 has already spec-normed
        # the polar input — passing "frob" would re-shrink σ_max by
        # 1/√(stable_rank) and leave the 5-iter Schulz incomplete. For the
        # plain chord-tight path (no §2.5), the default pre_norm="frob"
        # is correct. SSC's MISR path has no Frob pre-norm to disable;
        # tag it "ssc" so the loader can distinguish (vs None = not a
        # polar-product optimizer at all).
        polar_method = getattr(self, "polar_method", None)
        magnitude_rule = getattr(self, "magnitude_rule", None)
        if polar_method == "ssc":
            out["effective_polar_pre_norm"] = "ssc"
        elif polar_method in ("ns", "ns_hybrid", "polar_express"):
            if magnitude_rule == "spectral_chord_tight_clean":
                out["effective_polar_pre_norm"] = "none"
            else:
                out["effective_polar_pre_norm"] = "frob"
        return out

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
        timer = getattr(self, "_step_timer", None)
        eff = resolve_effective_inner_polar(
            psp, pm, optimizer_class_name=type(self).__name__,
        )
        eff_method = eff["method"] if eff is not None else "ns"

        def _polar_op(X):
            if op == "clip":
                return _clip_R_equal(X)
            if eff_method == "svd_exact":
                return self._sigma_power_polar(X, 0.0)
            if eff_method == "sigma_power":
                return self._sigma_power_polar(X, eff["sigma_power"])
            if eff_method == "ns_hybrid":
                return _newton_schulz_hybrid_deepseek(X, total_steps=max(self.ns_steps, 10))
            if eff_method == "polar_express":
                return _polar_express(X, nsteps=self.ns_steps)
            return _newton_schulz(X, nsteps=self.ns_steps)

        with maybe_time(timer, "polar_whiten"):
            X_B = u_B @ SA_half_inv
            X_A = SB_half_inv @ u_A
        with maybe_time(timer, "polar_NS_B"):
            P_B = _polar_op(X_B)
        with maybe_time(timer, "polar_NS_A"):
            P_A = _polar_op(X_A)
        with maybe_time(timer, "polar_unwhiten_rescale"):
            geo_B = P_B @ SA_half_inv
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

    # Class-level switch: subclasses that override `_adam_direction` (SOAP,
    # AdaFactor, etc.) should set this to False so they retain the per-pair
    # path. The batched path inlines the standard per-coord Adam moment update
    # on stacked buffers and would silently bypass the override.
    _BATCHED_PATH_SUPPORTED = True

    def _batched_path_eligible(self, is_probe_step: bool = False):
        """Production hot path. False whenever a feature flag would change
        per-pair behavior in a way the batched path doesn't reproduce.

        ``is_probe_step`` is accepted for back-compat with monkey-patch test
        helpers but is no longer consulted: as of Phase D, basic-diag
        records on probe steps are emitted by the batched path itself
        (see _step_batched), so probe steps stay on the batched path.
        The per-pair path is reached only when one of the ablation flags
        below is set — kept for reproducibility of past sweeps in params/."""
        if type(self) is not AdamPolarProductLoRA:
            if not getattr(self, "_BATCHED_PATH_SUPPORTED", False):
                return False
        if self.core_remix_alpha > 0.0:
            return False
        if self.polar_norm_dir != "frob":
            return False
        if self.polar_sigma_power is not None:
            return False
        if self.operator_type != "polar":
            return False
        if self.polar_method not in ("ns", "ssc", "polar_express"):
            return False
        # adam_frobenius, spectral_chord, spectral_chord_tight, and
        # spectral_chord_direction (variant 1) all implemented in batched
        # path. Future magnitude rules need an explicit branch.
        if getattr(self, "magnitude_rule", "adam_frobenius") not in (
                "adam_frobenius", "spectral_chord", "spectral_chord_tight",
                "spectral_chord_tight_clean", "spectral_chord_tight_no_rho",
                "spectral_chord_direction"):
            return False
        # anderson_m, end_rms_align only modify the cross-term path
        # (k_iter > 0). At picard_iters=1 each is mathematically a no-op,
        # so the batched path produces the same answer. At picard_iters > 1
        # they take effect and the per-pair path implements them.
        # exact_chord IS implemented in batched (refreshes precond per
        # Picard iter against effective factors via batched higham/eigh).
        if self.picard_iters > 1:
            if self.anderson_m > 0:
                return False
            if self.end_rms_align:
                return False
        return True

    def _save_debug_snapshot(self, *, reason, step_count, group_state,
                             local_idx, global_idx, tensors, scalars=None,
                             where=None):
        if not self.debug_snapshot_dir:
            return None
        if self._debug_snapshots_written >= self.debug_snapshot_limit:
            return None
        if not _is_main_process():
            return None

        os.makedirs(self.debug_snapshot_dir, exist_ok=True)
        pair_name = self.pair_names[global_idx] if global_idx < len(self.pair_names) else f"pair_{global_idx}"
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in pair_name)[-160:]
        path = os.path.join(
            self.debug_snapshot_dir,
            f"step{int(step_count):06d}_pair{int(global_idx):03d}_{reason}_{safe_name}.pt",
        )

        def _slice_value(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                td = v.detach()
                if td.dim() > 0 and td.shape[0] == group_state['N']:
                    td = td[local_idx]
                return td.to("cpu")
            if isinstance(v, (float, int, bool, str)):
                return v
            return str(v)

        A_f = group_state.get('A_stack')
        B_f = group_state.get('B_stack')
        payload_tensors = {}
        for name, value in tensors.items():
            sliced = _slice_value(value)
            if sliced is not None:
                payload_tensors[name] = sliced
        if A_f is not None:
            A_local = A_f[local_idx].detach()
            payload_tensors["SA_gram_recomputed"] = (A_local @ A_local.T).to("cpu")
        if B_f is not None:
            B_local = B_f[local_idx].detach()
            payload_tensors["SB_gram_recomputed"] = (B_local.T @ B_local).to("cpu")

        payload_scalars = {}
        for name, value in (scalars or {}).items():
            sliced = _slice_value(value)
            if isinstance(sliced, torch.Tensor):
                if sliced.numel() == 1:
                    payload_scalars[name] = float(sliced)
                else:
                    payload_tensors[f"scalar_tensor_{name}"] = sliced
            elif sliced is not None:
                payload_scalars[name] = sliced

        torch.save({
            "reason": reason,
            "step": int(step_count),
            "pair_index": int(global_idx),
            "local_index": int(local_idx),
            "group_id": int(group_state['gid']),
            "pair_name": pair_name,
            "where": where,
            "optimizer_hparams": {
                "lr": self.param_groups[0]["lr"],
                "delta": self.delta,
                "precond_delta_relative": self.precond_delta_relative,
                "precond_method": self.precond_method,
                "higham_iters": self.higham_iters,
                "picard_iters": self.picard_iters,
                "magnitude_rule": self.magnitude_rule,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
            },
            "scalars": payload_scalars,
            "tensors": payload_tensors,
        }, path)
        self._debug_snapshots_written += 1
        print(json.dumps({
            "event": "optimizer_debug_snapshot",
            "step": int(step_count),
            "pair_index": int(global_idx),
            "pair_name": pair_name,
            "reason": reason,
            "path": path,
        }, sort_keys=True), flush=True)
        return path

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        # When basic diagnostics are enabled, take per-pair on probe steps
        # (so the diag block has the per-pair intermediates it consumes)
        # and batched everywhere else. The optimizer step count is tracked
        # per LoRA pair; any pair's count works since all advance in lockstep.
        is_probe_step = False
        next_step = (
            next(iter(self.pair_state.values())).get('step', 0) + 1
            if self.pair_state else 0
        )
        if self.log_basic_diagnostics and self.pair_state:
            is_probe_step = (next_step % self.diagnostics_every == 0)
        if self.precond_method == "higham":
            try:
                from .utils import HIGHAM_DEBUG
                if HIGHAM_DEBUG["enabled"]:
                    HIGHAM_DEBUG["step"] = int(next_step)
                    HIGHAM_DEBUG["call"] = 0
            except Exception:
                pass
        eligible = self._batched_path_eligible(is_probe_step=is_probe_step)
        # Per-step timing probe (env-gated; off in production by default).
        # Emits one JSON event per step with path taken and wall time.
        if os.environ.get("LORA_TIME_STEP", "0") == "1":
            import time as _t
            import json as _j
            import sys as _s
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = _t.perf_counter()
            if eligible:
                self._step_batched()
            else:
                self._step_per_pair()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = _t.perf_counter() - t0
            step_n = (next(iter(self.pair_state.values())).get('step', -1)
                      if self.pair_state else -1)
            _s.stdout.write(_j.dumps({
                "event": "optim_path_timing",
                "step": int(step_n),
                "path": "batched" if eligible else "per_pair",
                "is_probe_step": bool(is_probe_step),
                "log_basic_diagnostics": bool(self.log_basic_diagnostics),
                "elapsed_sec": float(elapsed),
            }) + "\n")
            _s.stdout.flush()
            return None
        # Per-section CudaTimer profiling (env-gated). Attaches `_step_timer`
        # so all `maybe_time(timer, name)` scopes in _step_batched and
        # _chord_tight_clean_polar_pipeline accumulate timings. After the step,
        # dump per-section ms summary as JSONL and reset for the next step.
        if os.environ.get("LORA_PROFILE_OPTIM", "0") == "1":
            from ._step_timer import CudaTimer
            device = next(iter(self.pairs))[0].device if self.pairs else torch.device("cuda")
            if not hasattr(self, "_step_timer") or self._step_timer is None:
                self._step_timer = CudaTimer(device)
            self._step_timer.reset()
            if eligible:
                self._step_batched()
            else:
                self._step_per_pair()
            summary = self._step_timer.summary()
            step_n = (next(iter(self.pair_state.values())).get('step', -1)
                      if self.pair_state else -1)
            payload = {
                "event": "optim_step_timing",
                "step": int(step_n),
                "path": "batched" if eligible else "per_pair",
                "sections": {
                    name: {"ms": stats["total_ms"], "n": int(stats["n"]),
                           "mean_ms": stats["mean_ms"]}
                    for name, stats in summary.items()
                },
                "total_section_ms": float(sum(s["total_ms"] for s in summary.values())),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            return None
        if eligible:
            return self._step_batched()
        return self._step_per_pair()

    def _xgroup_section_a_dryrun(self, gs, lr):
        """Non-mutating prediction of the per-group prep stage (Adam,
        sigma_AB-hoist, precond refresh) used to feed the cross-group
        eigvalsh pre-flight. Bit-identical to what the main loop in
        `_step_batched` will compute moments later, because we use the
        same fp32 ops in the same order; just on temporary copies so the
        actual state buffers are left for the main loop to mutate.

        Assumes `gs['A_stack']`, `gs['B_stack']`, `gs['gA_stack']`,
        `gs['gB_stack']` are already populated (the main loop stages
        these unconditionally at the top of its iteration; we hoist that
        out to a separate top-level pass when xgroup_active).
        """
        indices = gs['indices']
        step_count = self.pair_state[indices[0]]['step']  # incremented upstream

        # Adam (non-mutating). out_of_place ops only.
        m_A = self.beta1 * gs['m_A'] + (1.0 - self.beta1) * gs['gA_stack']
        m_B = self.beta1 * gs['m_B'] + (1.0 - self.beta1) * gs['gB_stack']
        v_A = self.beta2 * gs['v_A'] + (1.0 - self.beta2) * gs['gA_stack'].pow(2)
        v_B = self.beta2 * gs['v_B'] + (1.0 - self.beta2) * gs['gB_stack'].pow(2)
        bc1 = 1.0 - self.beta1 ** step_count
        bc2 = 1.0 - self.beta2 ** step_count
        u_A = (m_A / bc1) / ((v_A / bc2).sqrt() + self.eps)
        u_B = (m_B / bc1) / ((v_B / bc2).sqrt() + self.eps)

        # sigma_AB hoist (read-only on A_stack/B_stack; v_init read-only).
        sigma_A_hoist, _ = _sigma_max_power_iter_batched(
            gs['A_stack'], v_init=gs.get('v_sigma_A'), n_iters=8)
        sigma_B_hoist, _ = _sigma_max_power_iter_batched(
            gs['B_stack'], v_init=gs.get('v_sigma_B'), n_iters=8)

        # Precond refresh prediction. Mirrors the main loop's gating.
        if (step_count - 1) % self.precond_refresh_every == 0:
            if self.disable_whitening:
                SA_half_inv = torch.eye(
                    gs['SA_half_inv'].shape[-1],
                    dtype=gs['SA_half_inv'].dtype,
                    device=gs['SA_half_inv'].device,
                ).expand_as(gs['SA_half_inv']).contiguous()
                SB_half_inv = torch.eye(
                    gs['SB_half_inv'].shape[-1],
                    dtype=gs['SB_half_inv'].dtype,
                    device=gs['SB_half_inv'].device,
                ).expand_as(gs['SB_half_inv']).contiguous()
            else:
                if self.curvature_whitening and gs.get('Scurv_A') is not None:
                    SA_grams = gs['Scurv_B']
                    SB_grams = gs['Scurv_A']
                    _lamA = _sigma_max_power_iter_batched(SA_grams, n_iters=8)[0]
                    _lamB = _sigma_max_power_iter_batched(SB_grams, n_iters=8)[0]
                else:
                    SA_grams = gs['A_stack'] @ gs['A_stack'].transpose(-2, -1)
                    SB_grams = gs['B_stack'].transpose(-2, -1) @ gs['B_stack']
                    _lamA = sigma_A_hoist.pow(2)
                    _lamB = sigma_B_hoist.pow(2)
                if self.precond_method == "higham":
                    from .utils import spd_inv_sqrt_higham_batched
                    SA_half_inv = spd_inv_sqrt_higham_batched(
                        SA_grams, n_iters=self.higham_iters, eps=self.delta,
                        eps_relative=self.precond_delta_relative,
                        lam_max=_lamA,
                        compute_dtype=self._higham_compute_dtype,
                    )
                    SB_half_inv = spd_inv_sqrt_higham_batched(
                        SB_grams, n_iters=self.higham_iters, eps=self.delta,
                        eps_relative=self.precond_delta_relative,
                        lam_max=_lamB,
                        compute_dtype=self._higham_compute_dtype,
                    )
                else:
                    # Per-pair eigh fallback rarely paired with the cross-group
                    # optimization at production scale; fall through to disable.
                    return None
        else:
            SA_half_inv = gs['SA_half_inv']
            SB_half_inv = gs['SB_half_inv']

        return {
            'u_A': u_A, 'u_B': u_B,
            'SA_half_inv': SA_half_inv, 'SB_half_inv': SB_half_inv,
            'sigma_A': sigma_A_hoist, 'sigma_B': sigma_B_hoist,
            'A_f': gs['A_stack'], 'B_f': gs['B_stack'],
        }

    def _xgroup_ssc_kappa_preflight(self, per_group_args, lr):
        """Optimization B: cross-shape-group eigvalsh batching for κ-adaptive SSC.

        Mirrors the picard math from `_chord_tight_clean_polar_pipeline` but
        stacks (X X^T) Grams across all shape groups into one eigvalsh +
        bisection call per (side, picard_iter). Each group has a different
        d_in / d_out but the SAME r (LoRA rank), so the Grams are uniformly
        (Ng, r, r) and stackable along the leading axis.

        Populates `gs['_xgroup_c_pre'][(side, n)] = c_per_pair` for each
        group; the subsequent per-group pipeline call sees this cache and
        skips its own eigvalsh.

        Cost: replaces 2 * picard_iters * G eigvalsh launches with
        2 * picard_iters launches. The duplicate MISR/matmul work in the
        pre-pass is small relative to eigvalsh launch overhead at r=256.

        Args
        ----
        per_group_args : list of dict, one per shape group, with keys:
            gs, A_f, B_f, u_A, u_B, SA_half_inv, SB_half_inv, sigma_A, sigma_B
        lr : float — learning rate (shared across groups).

        No return value — mutates each gs['_xgroup_c_pre'].
        """
        G = len(per_group_args)
        if G == 0:
            return
        # Uniform-r invariant. LoRA rank is the same across shape groups in
        # all currently-supported base models (OLMo-2-1B groups qkvo / gate-up
        # / down all share r=lora_r). Cross-group stacking relies on this:
        # grams are (Ng, r, r) and only stackable along the batch axis when r
        # matches. If a future base model violates this, the gating logic in
        # `_step_batched` must split the preflight per-r-bucket before
        # invocation rather than silently degrading.
        r_first = per_group_args[0]['A_f'].shape[-2]
        for entry in per_group_args[1:]:
            r_other = entry['A_f'].shape[-2]
            if r_other != r_first:
                raise RuntimeError(
                    f"_xgroup_ssc_kappa_preflight: heterogeneous LoRA rank "
                    f"across shape groups (r={r_first} vs r={r_other}); "
                    "cross-group eigvalsh batching requires uniform r. "
                    "Either set --ssc_kappa_cross_group_eigvalsh false to "
                    "fall back to per-group eigvalsh, or split per-r-bucket "
                    "in the caller."
                )
        r = r_first

        # Set up per-group state mirroring the pipeline §2.2–§2.5.
        states = []
        for entry in per_group_args:
            gs = entry['gs']
            A_f = entry['A_f']
            B_f = entry['B_f']
            u_A = entry['u_A']
            u_B = entry['u_B']
            SA_half_inv = entry['SA_half_inv']
            SB_half_inv = entry['SB_half_inv']
            sigma_A = entry['sigma_A']
            sigma_B = entry['sigma_B']
            N = A_f.shape[0]
            device, dtype = A_f.device, A_f.dtype
            # §2.2 σ_max already hoisted upstream (sigma_A, sigma_B).
            s_AB = sigma_A + sigma_B
            rho = lr / (s_AB + 1e-30)
            # §2.4 whitened Adam direction.
            X_A = SB_half_inv @ u_A
            X_B = u_B @ SA_half_inv
            # §2.5 pre-rescale via power-iter σ_max. Use cached warm-start
            # vectors from the gs (separate keys from the pipeline's, since
            # the pipeline will recompute its own σ_max(X_*) on the same X).
            # Sharing the warm-start vector across pre-pass and pipeline is
            # safe and slightly accelerates convergence.
            sigma_XA, v_sigma_XA = _sigma_max_power_iter_batched(
                X_A, v_init=gs.get('v_sigma_XA'), n_iters=8)
            sigma_XB, v_sigma_XB = _sigma_max_power_iter_batched(
                X_B, v_init=gs.get('v_sigma_XB'), n_iters=8)
            # NOTE: do NOT write these back to gs here — the pipeline below
            # will run the same power-iter starting from the same v_init and
            # write its own (identical) results back. Avoids ordering hazard.
            inv_XA = (1.0 / (sigma_XA + 1e-30)).unsqueeze(-1).unsqueeze(-1)
            inv_XB = (1.0 / (sigma_XB + 1e-30)).unsqueeze(-1).unsqueeze(-1)
            X_A = X_A * inv_XA
            X_B = X_B * inv_XB
            u_A_local = u_A * inv_XA
            u_B_local = u_B * inv_XB
            # Reset / install the cross-group c cache.
            gs['_xgroup_c_pre'] = {}
            states.append({
                'gs': gs, 'A_f': A_f, 'B_f': B_f,
                'u_A': u_A_local, 'u_B': u_B_local,
                'X_A': X_A, 'X_B': X_B,
                'SA_half_inv': SA_half_inv, 'SB_half_inv': SB_half_inv,
                'sigma_A': sigma_A, 'sigma_B': sigma_B,
                'rho': rho, 'N': N, 'device': device, 'dtype': dtype,
                'dA': torch.zeros_like(u_A_local),
                'dB': torch.zeros_like(u_B_local),
            })

        # Respect Optimization A (`ssc_kappa_cache_share_picard`): when
        # the per-group pipeline reuses n=0's solved c for n=1, the
        # pre-flight should only solve at n=0 and replicate c into the
        # (side, 1) cache slot. Otherwise the pipeline would see fresh
        # n=1 c from pre-flight while the share_picard semantics say
        # "reuse n=0 c" — a silent divergence vs the legacy path.
        share_picard = bool(self.ssc_kappa_cache_share_picard)
        # Picard outer loop: at each n, build X_A_eff/X_B_eff per group,
        # cross-group eigvalsh+bisect → scatter c → MISR per group →
        # update dA/dB per group for next iter.
        n_iters_to_solve = 1 if share_picard else self.picard_iters
        for n in range(n_iters_to_solve):
            # ---- A side ----
            X_A_eff_list = []
            for st in states:
                if n == 0:
                    X_A_eff_list.append(st['X_A'])
                else:
                    A_f, B_f = st['A_f'], st['B_f']
                    BT_dB_A = B_f.transpose(-2, -1) @ st['dB'] @ A_f
                    u_A_eff = st['u_A'] + (1.0 / lr) * BT_dB_A
                    if self.fw_linearization == "full":
                        # Full-FW self-terms (S_B_full @ dA). Skip here since
                        # cross-group with full-FW is not the production path;
                        # if both flags are set, disable the optimization.
                        return
                    X_A_eff_list.append(st['SB_half_inv'] @ u_A_eff)
            # d_in differs per group (e.g., OLMo qkvo d=2048 vs gate-up d=8192),
            # so we cannot cat X tensors along the batch axis. Compute grams
            # per group (each (Ng, r, r) — uniform shape) and cat those.
            G_X_list = [X @ X.transpose(-2, -1) for X in X_A_eff_list]
            G_X_stack = torch.cat(G_X_list, dim=0)  # (sum Ng, r, r)
            lam = torch.linalg.eigvalsh(G_X_stack).clamp_min(0.0)
            lam_max = lam.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
            s_sq = lam / lam_max
            c_all = _solve_c_from_kappa_batched(
                s_sq, self.ssc_kappa, c_lo=1e-3, c_hi=1e3, iters=40,
            )
            # Scatter c back per group; run MISR per group to get P_A for dA.
            offset = 0
            for st, X_eff in zip(states, X_A_eff_list):
                Ng = X_eff.shape[0]
                c_g = c_all[offset:offset + Ng]
                offset += Ng
                st['gs']['_xgroup_c_pre'][('A', n)] = c_g.detach()
                P_A = _ssc_misr_batched(X_eff, c=c_g, nsteps=self.ssc_nsteps).float()
                st['P_A'] = P_A
                st['X_A_eff'] = X_eff

            # ---- B side ----
            X_B_eff_list = []
            for st in states:
                if n == 0:
                    X_B_eff_list.append(st['X_B'])
                else:
                    A_f, B_f = st['A_f'], st['B_f']
                    B_dA_AT = B_f @ st['dA'] @ A_f.transpose(-2, -1)
                    u_B_eff = st['u_B'] + (1.0 / lr) * B_dA_AT
                    X_B_eff_list.append(u_B_eff @ st['SA_half_inv'])
            # Grams are X^T X for B (X_B is (Ng, d_out, r)).
            G_X_list = [X.transpose(-2, -1) @ X for X in X_B_eff_list]
            G_X_stack = torch.cat(G_X_list, dim=0)  # (sum Ng, r, r)
            lam = torch.linalg.eigvalsh(G_X_stack).clamp_min(0.0)
            lam_max = lam.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
            s_sq = lam / lam_max
            c_all = _solve_c_from_kappa_batched(
                s_sq, self.ssc_kappa, c_lo=1e-3, c_hi=1e3, iters=40,
            )
            offset = 0
            for st, X_eff in zip(states, X_B_eff_list):
                Ng = X_eff.shape[0]
                c_g = c_all[offset:offset + Ng]
                offset += Ng
                st['gs']['_xgroup_c_pre'][('B', n)] = c_g.detach()
                P_B = _ssc_misr_batched(X_eff, c=c_g, nsteps=self.ssc_nsteps).float()
                st['P_B'] = P_B
                st['X_B_eff'] = X_eff

            # ---- Compute dA, dB for n+1 (skip if last iter) ----
            if n + 1 < n_iters_to_solve:
                for st in states:
                    geo_A = st['SB_half_inv'] @ st['P_A']
                    geo_B = st['P_B'] @ st['SA_half_inv']
                    op_geoA_b, _ = _sigma_max_power_iter_batched(
                        geo_A, v_init=None, n_iters=8)
                    op_geoB_b, _ = _sigma_max_power_iter_batched(
                        geo_B, v_init=None, n_iters=8)
                    rho_unsq = st['rho'].unsqueeze(-1).unsqueeze(-1)
                    op_geoA = (op_geoA_b + 1e-30).unsqueeze(-1).unsqueeze(-1)
                    op_geoB = (op_geoB_b + 1e-30).unsqueeze(-1).unsqueeze(-1)
                    st['dA'] = -(rho_unsq / op_geoA) * geo_A
                    st['dB'] = -(self.lora_plus_multiplier * rho_unsq / op_geoB) * geo_B

        # When share_picard, replicate the n=0 c into the remaining
        # picard slots so the per-group `_polar` fast-path picks it up
        # for n>=1 and produces the same MISR output as the share_picard
        # cache would have.
        if share_picard:
            for st in states:
                c_pre = st['gs']['_xgroup_c_pre']
                c_A0 = c_pre.get(('A', 0))
                c_B0 = c_pre.get(('B', 0))
                for n_extra in range(1, self.picard_iters):
                    if c_A0 is not None:
                        c_pre[('A', n_extra)] = c_A0
                    if c_B0 is not None:
                        c_pre[('B', n_extra)] = c_B0
                # Also populate the share_picard cache key the per-group
                # pipeline will look up at refresh-due boundaries on the
                # next step. Mirrors what the per-group eigvalsh path
                # would have written via `gs[cache_key] = c_solved.detach()`.
                if c_A0 is not None:
                    st['gs']['ssc_c_cached_A'] = c_A0
                if c_B0 is not None:
                    st['gs']['ssc_c_cached_B'] = c_B0

    def _chord_tight_clean_polar_pipeline(
        self, gs, A_f, B_f, u_A, u_B, SA_half_inv, SB_half_inv, lr, timer,
        sigma_A=None, sigma_B=None, step_count=None,
    ):
        """Algorithm 2′ — chord-tight-clean polar pipeline. Walkthrough +
        FLOP budget: `docs/notes/polar_product/algorithm_clean_implementation.md`.
        Stage labels below (§2.2–§2.7) cross-reference that doc.

        Inputs `u_A`, `u_B` are the post-Adam-direction update (doc §2.1 done
        upstream); `SA_half_inv`, `SB_half_inv` are the refreshed Higham
        whiteners (doc §2.3 done upstream); `sigma_A`, `sigma_B` are the
        optional caller-hoisted σ_max values (per doc §7 "Done").

        Behavior knobs:
          - `self.ns_form ∈ {"gram", "gram-norestart", "rect"}`. Canonical = "gram"
            (Dao 2026 Alg 3, fp16+restart at τ=2 — doc §5).
          - `self.htmuon_p` (default None): when set, applies an extra
            σ→σ^p polishing P ← (X Xᵀ)^(p/2) · polar(X). None ⇒ bit-identical
            to clean NS, not described in the algorithm-clean doc.

        Returns a dict whose keys match the locals the caller's downstream
        diagnostic-emission block consumes.
        """
        N = A_f.shape[0]
        device, dtype = A_f.device, A_f.dtype

        def _b11(t):
            return t.unsqueeze(-1).unsqueeze(-1)

        def _polar(X, side=None, n=None):
            if self.polar_method == "ssc":
                # SPECTRA soft spectral clipping in place of NS polar map.
                # Input is post-§2.5-rescale so σ_max(X) ≈ 1; ssc_c is in
                # those units (c ≲ 1 produces meaningful clipping).
                if self.ssc_kappa is not None:
                    # Optimization B fast path: cross-group eigvalsh has
                    # already computed c for this (side, n) in a pre-pass
                    # in `_step_batched`. Skip eigvalsh + bisect; just MISR.
                    xg_pre = gs.get('_xgroup_c_pre')
                    if xg_pre is not None and (side, n) in xg_pre:
                        c_pre = xg_pre[(side, n)]
                        out = _ssc_misr_batched(X, c=c_pre, nsteps=self.ssc_nsteps)
                        if side is not None:
                            gs[f'ssc_c_last_{side}'] = c_pre.detach()
                        return out.float()
                    # κ-adaptive: solve c per-pair per-step from target κ
                    # (Appendix C.5 state-dependent interpretation).
                    # Refresh schedule: when ssc_kappa_refresh_every>1, solve
                    # c every N steps and reuse cached c between refreshes
                    # via the cheap _ssc_misr_batched (no eigvalsh+bisect).
                    # Cache key is per-(side, Picard iter n) so n=0/n=1 don't
                    # collide. Refresh boundary uses step_count (1-indexed
                    # matching precond_refresh_every).
                    N = self.ssc_kappa_refresh_every
                    M = self.ssc_kappa_warmup_steps
                    # Optimization A: when share_picard=True, use one
                    # cache slot per side, shared across Picard inner iters
                    # (n=0 solves; n=1 reuses). When False, fall back to per-n
                    # caches (pre-flag behavior). See __init__ for the drift
                    # justification.
                    share_picard = self.ssc_kappa_cache_share_picard
                    if share_picard:
                        cache_key = f'ssc_c_cached_{side}'
                        stamp_key = f'ssc_c_cached_{side}_step'
                    else:
                        cache_key = f'ssc_c_cached_{side}_n{n}'
                        stamp_key = None
                    # Picard-share short-circuit: at n>0, if n=0 already solved
                    # (and stamped) this step, reuse without re-solving — this
                    # is the optimization, independent of the cross-step
                    # refresh schedule (N).
                    picard_reuse = (
                        share_picard
                        and n is not None and n > 0
                        and cache_key in gs
                        and stamp_key is not None
                        and gs.get(stamp_key) == step_count
                    )
                    refresh_due = (not picard_reuse) and (
                        N == 1
                        or step_count is None
                        or step_count <= M
                        or (step_count - 1) % N == 0
                        or cache_key not in gs
                    )
                    in_warmup = (step_count is not None and step_count <= M)
                    if refresh_due:
                        if self.ssc_kappa_solver == "stable_rank":
                            out, c_solved = _ssc_adaptive_stable_rank_batched(
                                X, kappa=self.ssc_kappa, nsteps=self.ssc_nsteps,
                            )
                        elif self.ssc_kappa_solver == "misr_bisect":
                            # Warm-start from prior c if cached; else fall back
                            # to eigvalsh once to seed (K=3 with full-range
                            # bracket is too coarse for a fresh start).
                            # ALSO: during warmup (step_count <= M_warmup), the
                            # polar input X spectrum changes rapidly (B starts
                            # at zero with lora_init_b=zero; SB^{-1/2} is
                            # damping-only at step 1, then drastically different
                            # at step 2). Cached c from step 1 is meaningless
                            # for step 2's spectrum → kpar bracket misses
                            # true c → 100x errors observed in DIAG. Force
                            # eigvalsh during warmup to bypass.
                            c_init = gs.get(cache_key)
                            if c_init is None or in_warmup:
                                out, c_solved = _ssc_adaptive_kappa_batched(
                                    X, kappa=self.ssc_kappa, nsteps=self.ssc_nsteps,
                                )
                            else:
                                if self.ssc_kappa_bisect_mode == "parallel":
                                    out, c_solved = _ssc_misr_bisect_batched_kpar(
                                        X, kappa=self.ssc_kappa,
                                        K=self.ssc_kappa_bisect_iters,
                                        nsteps=self.ssc_nsteps, c_init=c_init,
                                        nsteps_eval=self.ssc_kappa_bisect_nsteps_eval,
                                    )
                                else:
                                    out, c_solved = _ssc_misr_bisect_batched(
                                        X, kappa=self.ssc_kappa,
                                        K=self.ssc_kappa_bisect_iters,
                                        nsteps=self.ssc_nsteps, c_init=c_init,
                                    )
                        else:
                            out, c_solved = _ssc_adaptive_kappa_batched(
                                X, kappa=self.ssc_kappa, nsteps=self.ssc_nsteps,
                            )
                        if N > 1 or share_picard:
                            # Stash for the next N-1 steps. Detach to break
                            # any spurious autograd ties (we run under
                            # @torch.no_grad anyway). When share_picard=True,
                            # also stash at N=1 so the n=1 Picard inner iter
                            # can reuse the c solved at n=0; stamp the step so
                            # the n>0 short-circuit only triggers within the
                            # same step.
                            c_cache = c_solved.detach()
                            beta = self.ssc_kappa_cache_ema_beta
                            if (
                                beta is not None
                                and cache_key in gs
                                and not in_warmup
                            ):
                                prev = gs[cache_key].to(
                                    device=c_cache.device, dtype=c_cache.dtype
                                )
                                # c is positive and diagnostics are log-error
                                # based, so smooth in log-c units.
                                c_cache = (
                                    beta * prev.clamp_min(1e-30).log()
                                    + (1.0 - beta)
                                    * c_cache.clamp_min(1e-30).log()
                                ).exp()
                            gs[cache_key] = c_cache
                            if share_picard and stamp_key is not None:
                                gs[stamp_key] = step_count
                    else:
                        cached_c = gs[cache_key]
                        out = _ssc_misr_batched(
                            X, c=cached_c, nsteps=self.ssc_nsteps,
                        )
                        c_solved = cached_c
                    if side is not None:
                        # Stash the realized per-pair c for diagnostic logging.
                        # Last polar call wins (n=k-1 of Picard); fine since
                        # the cross-coupling-corrected iterate is what gets
                        # applied to the weights.
                        gs[f'ssc_c_last_{side}'] = c_solved.detach()
                    # Diagnostic: compare the actually-used c against eigvalsh
                    # ground truth on the SAME X. Catches kpar-grid-misses or
                    # stale-cache drift. Doubles eigvalsh work — flag-gated.
                    if (
                        self.ssc_kappa_diagnose_eigvalsh
                        and side is not None
                        and step_count is not None
                        and step_count >= self.ssc_kappa_diagnose_start_step
                    ):
                        with torch.no_grad():
                            _, c_eigvalsh = _ssc_adaptive_kappa_batched(
                                X, kappa=self.ssc_kappa, nsteps=self.ssc_nsteps,
                            )
                            log_err = (c_solved.log() - c_eigvalsh.log()).abs()
                            abs_err = (c_solved - c_eigvalsh).abs()
                            rel_err = abs_err / c_eigvalsh.abs().clamp_min(1e-30)
                            c_ema_ref = None
                            ema_beta = self.ssc_kappa_diag_ema_beta
                            if ema_beta is not None:
                                ema_key = f'{cache_key}_diag_true_ema'
                                c_true_detached = c_eigvalsh.detach()
                                if ema_key in gs and not in_warmup:
                                    prev = gs[ema_key].to(
                                        device=c_true_detached.device,
                                        dtype=c_true_detached.dtype,
                                    )
                                    c_ema_ref = (
                                        ema_beta
                                        * prev.clamp_min(1e-30).log()
                                        + (1.0 - ema_beta)
                                        * c_true_detached.clamp_min(1e-30).log()
                                    ).exp()
                                else:
                                    c_ema_ref = c_true_detached
                                gs[ema_key] = c_ema_ref.detach()
                            # Boundary detector: when refresh_due via kpar, the
                            # winner is on the grid edge if |log(c_used/c_init)|
                            # is within eps of log_window. picard_reuse / non-
                            # kpar refresh paths have no bracket, so n_boundary=0.
                            n_boundary = 0
                            if (refresh_due and not picard_reuse
                                    and self.ssc_kappa_solver == "misr_bisect"
                                    and self.ssc_kappa_bisect_mode == "parallel"
                                    and 'c_init' in dir()):
                                pass  # c_init is in the enclosing scope
                            def _put_stats(payload, prefix, values):
                                payload[f"{prefix}_p50"] = float(values.median().item())
                                payload[f"{prefix}_p99"] = float(
                                    values.quantile(0.99).item()
                                    if values.numel() > 1
                                    else values.max().item()
                                )
                                payload[f"{prefix}_max"] = float(values.max().item())

                            payload = {
                                "event": "ssc_c_diag",
                                "step": int(step_count),
                                "side": str(side),
                                "n": int(n) if n is not None else -1,
                                "refresh_due": bool(refresh_due),
                                "picard_reuse": bool(picard_reuse),
                                "n_pairs": int(log_err.numel()),
                                "log_err_p50": float(log_err.median().item()),
                                "log_err_p99": float(
                                    log_err.quantile(0.99).item()
                                    if log_err.numel() > 1
                                    else log_err.max().item()
                                ),
                                "log_err_max": float(log_err.max().item()),
                                "abs_err_p50": float(abs_err.median().item()),
                                "abs_err_p99": float(
                                    abs_err.quantile(0.99).item()
                                    if abs_err.numel() > 1
                                    else abs_err.max().item()
                                ),
                                "abs_err_max": float(abs_err.max().item()),
                                "rel_err_p50": float(rel_err.median().item()),
                                "rel_err_p99": float(
                                    rel_err.quantile(0.99).item()
                                    if rel_err.numel() > 1
                                    else rel_err.max().item()
                                ),
                                "rel_err_max": float(rel_err.max().item()),
                                "c_used_p50": float(c_solved.median().item()),
                                "c_used_p99": float(
                                    c_solved.quantile(0.99).item()
                                    if c_solved.numel() > 1
                                    else c_solved.max().item()
                                ),
                                "c_used_max": float(c_solved.max().item()),
                                "c_true_p50": float(c_eigvalsh.median().item()),
                                "c_true_p99": float(
                                    c_eigvalsh.quantile(0.99).item()
                                    if c_eigvalsh.numel() > 1
                                    else c_eigvalsh.max().item()
                                ),
                                "c_true_max": float(c_eigvalsh.max().item()),
                                "kpar_log_window": float(0.5),
                                "kpar_K": int(self.ssc_kappa_bisect_iters)
                                          if self.ssc_kappa_solver == "misr_bisect" else -1,
                            }
                            if c_ema_ref is not None:
                                ema_log_err = (
                                    c_solved.log() - c_ema_ref.log()
                                ).abs()
                                ema_abs_err = (c_solved - c_ema_ref).abs()
                                ema_rel_err = (
                                    ema_abs_err
                                    / c_ema_ref.abs().clamp_min(1e-30)
                                )
                                true_ema_log_err = (
                                    c_eigvalsh.log() - c_ema_ref.log()
                                ).abs()
                                true_ema_abs_err = (c_eigvalsh - c_ema_ref).abs()
                                true_ema_rel_err = (
                                    true_ema_abs_err
                                    / c_ema_ref.abs().clamp_min(1e-30)
                                )
                                payload["diag_ema_beta"] = float(ema_beta)
                                _put_stats(payload, "ema_ref_log_err", ema_log_err)
                                _put_stats(payload, "ema_ref_abs_err", ema_abs_err)
                                _put_stats(payload, "ema_ref_rel_err", ema_rel_err)
                                _put_stats(payload, "true_ema_log_err", true_ema_log_err)
                                _put_stats(payload, "true_ema_abs_err", true_ema_abs_err)
                                _put_stats(payload, "true_ema_rel_err", true_ema_rel_err)
                                _put_stats(payload, "c_ema", c_ema_ref)
                            print(json.dumps(payload, sort_keys=True), flush=True)
                            if (
                                self.debug_optimizer_state
                                and step_count % self.debug_optimizer_state_every == 0
                                and step_count >= self.debug_optimizer_state_start_step
                            ):
                                pair_indices = list(gs.get('indices', []))
                                pair_names = [
                                    self.pair_names[gi]
                                    if gi < len(self.pair_names) else f"pair_{gi}"
                                    for gi in pair_indices
                                ]
                                pair_payload = {
                                    "event": "ssc_c_pair_diag",
                                    "step": int(step_count),
                                    "side": str(side),
                                    "n": int(n) if n is not None else -1,
                                    "refresh_due": bool(refresh_due),
                                    "picard_reuse": bool(picard_reuse),
                                    "pair_indices": [int(i) for i in pair_indices],
                                    "pair_names": pair_names,
                                    "c_used": _json_list_from_tensor(c_solved),
                                    "c_true": _json_list_from_tensor(c_eigvalsh),
                                    "log_err": _json_list_from_tensor(log_err),
                                    "abs_err": _json_list_from_tensor(abs_err),
                                    "rel_err": _json_list_from_tensor(rel_err),
                                }
                                if c_ema_ref is not None:
                                    pair_payload.update({
                                        "diag_ema_beta": float(ema_beta),
                                        "c_ema": _json_list_from_tensor(c_ema_ref),
                                        "ema_ref_log_err": _json_list_from_tensor(
                                            ema_log_err
                                        ),
                                        "ema_ref_abs_err": _json_list_from_tensor(
                                            ema_abs_err
                                        ),
                                        "ema_ref_rel_err": _json_list_from_tensor(
                                            ema_rel_err
                                        ),
                                        "true_ema_log_err": _json_list_from_tensor(
                                            true_ema_log_err
                                        ),
                                        "true_ema_abs_err": _json_list_from_tensor(
                                            true_ema_abs_err
                                        ),
                                        "true_ema_rel_err": _json_list_from_tensor(
                                            true_ema_rel_err
                                        ),
                                    })
                                print(json.dumps(pair_payload, sort_keys=True), flush=True)
                    return out.float()
                return _ssc_misr_batched(X, c=self.ssc_c, nsteps=self.ssc_nsteps).float()
            # §2.5 has already spec-normed X so σ_max(X) = 1. Pass
            # pre_norm='none' so the NS/polar-express functions don't
            # re-shrink the input via their default Frobenius pre-norm.
            # A redundant Frob pre-norm here would divide σ_max further
            # by ‖X‖_F = √(stable_rank), pushing the iterate far from
            # the σ=1 fixed point and leaving 5-iter Schulz incomplete
            # (whitening_fraction ≈ 0.72 instead of ≈ 1.0). The legacy
            # behavior is recoverable by setting pre_norm='frob' below.
            ctc_pre_norm = "none"
            if self.polar_method == "polar_express":
                # Amsel et al. 2505.16932 quintic-Remez polar in batched Gram
                # form. Mirrors the ns_form=gram path but with Polar Express
                # coefficients in place of cubic Muon.
                return _polar_express_gram_batched(
                    X, nsteps=self.ns_steps, pre_norm=ctc_pre_norm,
                ).float()
            if self.ns_form == "gram":
                return _newton_schulz_gram_batched(
                    X, nsteps=self.ns_steps, pre_norm=ctc_pre_norm,
                ).float()
            if self.ns_form == "gram-norestart":
                return _newton_schulz_gram_batched(
                    X, nsteps=self.ns_steps, restart_at=None,
                    pre_norm=ctc_pre_norm,
                ).float()
            return _newton_schulz_batched(
                X, nsteps=self.ns_steps, dtype=torch.bfloat16,
                pre_norm=ctc_pre_norm,
            ).float()

        # §2.2 σ_max(A), σ_max(B) → ρ = η/s. Warm-started power iter on raw
        #      factors. When caller passes sigma_A/sigma_B (λ_max hoist:
        #      `_step_batched` shares these with Higham's damping), reuse.
        with maybe_time(timer, "chord_tight_clean_sigma_AB"):
            if sigma_A is None or sigma_B is None:
                sigma_A, v_sigma_A = _sigma_max_power_iter_batched(
                    A_f, v_init=gs.get('v_sigma_A'), n_iters=8)
                sigma_B, v_sigma_B = _sigma_max_power_iter_batched(
                    B_f, v_init=gs.get('v_sigma_B'), n_iters=8)
                gs['v_sigma_A'] = v_sigma_A
                gs['v_sigma_B'] = v_sigma_B
            s_AB = sigma_A + sigma_B                           # doc-name: s
            rho = lr / (s_AB + 1e-30)                          # (N,)

        # §2.4 Whitened Adam direction. Computed once; reused for pre-rescale
        #      and for the n≥1 polar input.
        with maybe_time(timer, "chord_tight_clean_whiten_input"):
            X_A = SB_half_inv @ u_A                            # (N, r, d_in)
            X_B = u_B @ SA_half_inv                            # (N, d_out, r)

        # §2.5 Pre-rescale: divide X_A, X_B by σ_max(X_A), σ_max(X_B).
        #      After this, σ_max(X_A) = σ_max(X_B) = 1 by construction.
        #      The SAME divisors are applied to u_A, u_B so that the n≥1
        #      cross-coupling (which mixes u with dA, dB derived from the
        #      rescaled X) stays consistent — this is load-bearing.
        with maybe_time(timer, "chord_tight_clean_pre_rescale"):
            sigma_XA, v_sigma_XA = _sigma_max_power_iter_batched(
                X_A, v_init=gs.get('v_sigma_XA'), n_iters=8)
            sigma_XB, v_sigma_XB = _sigma_max_power_iter_batched(
                X_B, v_init=gs.get('v_sigma_XB'), n_iters=8)
            gs['v_sigma_XA'] = v_sigma_XA
            gs['v_sigma_XB'] = v_sigma_XB
            inv_XA = _b11(1.0 / (sigma_XA + 1e-30))
            inv_XB = _b11(1.0 / (sigma_XB + 1e-30))
            X_A = X_A * inv_XA
            X_B = X_B * inv_XB
            u_A = u_A * inv_XA
            u_B = u_B * inv_XB

        # §2.6 Picard outer loop. dA, dB initialized to zero so the n=0 iter
        #      sees no cross-coupling; the pre-loop sentinels for P_A, P_B,
        #      geo_A, geo_B, op_geoA_b, op_geoB_b guard the return dict in
        #      the degenerate picard_iters=0 config.
        dA = torch.zeros_like(u_A)
        dB = torch.zeros_like(u_B)
        # Picard coupling coefficient (1/η, Lemma 1). Stored as (N, 1, 1)
        # so downstream chain-tensors emission sees the expected shape.
        picard_coeff_s = torch.full((N, 1, 1), 1.0 / lr, device=device, dtype=dtype)

        u_A_eff, u_B_eff = u_A, u_B
        P_A = P_B = geo_A = geo_B = None
        op_geoA_b = op_geoB_b = None
        op_geoA = op_geoB = None
        BT_dB_A = B_dA_AT = None
        dA_prev_picard = dB_prev_picard = None
        X_A_eff = X_B_eff = None
        picard_trace = {}

        slots_A = gs.setdefault('v_op_geoA_slots', [None] * self.picard_iters)
        slots_B = gs.setdefault('v_op_geoB_slots', [None] * self.picard_iters)
        track_sigma_guard = (
            bool(self.log_non_finite)
            and step_count is not None
            and _is_main_process()
        )
        if track_sigma_guard:
            group_global_indices = list(gs.get('indices', range(N)))
            pair_names_in_group = [
                self.pair_names[gi] if gi < len(self.pair_names) else f"pair_{gi}"
                for gi in group_global_indices
            ]
        else:
            group_global_indices = None
            pair_names_in_group = None

        def _trace_picard(n, **items):
            if not track_sigma_guard:
                return
            for name, value in items.items():
                if value is not None:
                    picard_trace[f"{name}_n{n}"] = value

        # Full-FW self-term grams (algorithm_tight_chord.md §6). Materialized
        # once outside the Picard loop; both r×r and constant across iters.
        # Effective δ matches the Higham/eigh damping (relative ⇒ δ·σ_max²,
        # absolute ⇒ δ). σ_A, σ_B already in scope from §2.2.
        if self.fw_linearization == "full":
            r = A_f.shape[-2]
            eye_r = torch.eye(r, device=device, dtype=dtype).expand(N, r, r)
            if self.precond_delta_relative:
                delta_eff_A = self.delta * (sigma_A ** 2)        # (N,)
                delta_eff_B = self.delta * (sigma_B ** 2)        # (N,)
                S_A_full = (A_f @ A_f.transpose(-2, -1)
                            + delta_eff_A.view(N, 1, 1) * eye_r)
                S_B_full = (B_f.transpose(-2, -1) @ B_f
                            + delta_eff_B.view(N, 1, 1) * eye_r)
            else:
                S_A_full = A_f @ A_f.transpose(-2, -1) + self.delta * eye_r
                S_B_full = B_f.transpose(-2, -1) @ B_f + self.delta * eye_r
        else:
            S_A_full = S_B_full = None

        for n in range(self.picard_iters):
            with maybe_time(timer, "chord_tight_clean_picard"):
                if n == 0:
                    u_A_eff, u_B_eff = u_A, u_B
                    X_A_eff, X_B_eff = X_A, X_B
                else:
                    # 1/η cross-coupling on the whitened polar input:
                    # X_A^eff = S_B^{-1/2} (u_A + (1/η) Bᵀ dB A).
                    dA_prev_picard = dA
                    dB_prev_picard = dB
                    # Associate to keep the intermediate at (N, r, r), never the
                    # (N, d_out, d_in) outer-product shape. Matmul is associative
                    # so this is bit-for-bit the same map up to fp rounding, but
                    # avoids a ~d_out·d_in materialization that OOMs at 8B/r256.
                    BT_dB_A = (B_f.transpose(-2, -1) @ dB) @ A_f     # (N, r, d_in) via (N, r, r)
                    B_dA_AT = B_f @ (dA @ A_f.transpose(-2, -1))     # (N, d_out, r) via (N, r, r)
                    u_A_eff = u_A + (1.0 / lr) * BT_dB_A
                    u_B_eff = u_B + (1.0 / lr) * B_dA_AT
                    # Full-FW (§6) self-terms: keep block's own previous
                    # contribution in the polar input. Anchored path skips.
                    if self.fw_linearization == "full":
                        u_A_eff = u_A_eff + (1.0 / lr) * (S_B_full @ dA)
                        u_B_eff = u_B_eff + (1.0 / lr) * (dB @ S_A_full)
                    X_A_eff = SB_half_inv @ u_A_eff
                    X_B_eff = u_B_eff @ SA_half_inv

                # Polar map via Newton-Schulz (canonical: gram, fp16+restart).
                P_A = _polar(X_A_eff, side='A', n=n)
                P_B = _polar(X_B_eff, side='B', n=n)

                # HTMuon σ→σ^p sub-mode (silent when htmuon_p is None).
                # Applies U Σ^p V^T = (X Xᵀ)^(p/2) · polar(X).
                if self.htmuon_p is not None:
                    from .utils import spd_power_batched
                    G_A = X_A_eff @ X_A_eff.transpose(-2, -1)               # (N, r, r)
                    G_B = X_B_eff.transpose(-2, -1) @ X_B_eff               # (N, r, r)
                    G_A_phalf = spd_power_batched(G_A, self.htmuon_p / 2.0)
                    G_B_phalf = spd_power_batched(G_B, self.htmuon_p / 2.0)
                    P_A = G_A_phalf @ P_A
                    P_B = P_B @ G_B_phalf

                # Unwhiten back to factor space.
                geo_A = SB_half_inv @ P_A                              # (N, r, d_in)
                geo_B = P_B @ SA_half_inv                              # (N, d_out, r)

                # Site-C σ_max(geo). Warm-start keyed by Picard iter n via
                # pre-allocated slot lists (dynamic-string dict keys would
                # graph-break under torch.compile).
                if track_sigma_guard:
                    op_geoA_b, v_op_geoA, guard_A = _sigma_max_power_iter_batched(
                        geo_A, v_init=slots_A[n], n_iters=8, return_info=True)
                    op_geoB_b, v_op_geoB, guard_B = _sigma_max_power_iter_batched(
                        geo_B, v_init=slots_B[n], n_iters=8, return_info=True)
                    _emit_sigma_guard_event(
                        step_count, site="site_c_geo", side="A", n=n,
                        guard_info=guard_A,
                        pair_names_in_group=pair_names_in_group,
                        group_global_indices=group_global_indices,
                    )
                    _emit_sigma_guard_event(
                        step_count, site="site_c_geo", side="B", n=n,
                        guard_info=guard_B,
                        pair_names_in_group=pair_names_in_group,
                        group_global_indices=group_global_indices,
                    )
                else:
                    op_geoA_b, v_op_geoA = _sigma_max_power_iter_batched(
                        geo_A, v_init=slots_A[n], n_iters=8)
                    op_geoB_b, v_op_geoB = _sigma_max_power_iter_batched(
                        geo_B, v_init=slots_B[n], n_iters=8)
                slots_A[n] = v_op_geoA
                slots_B[n] = v_op_geoB

                rho_unsq = _b11(rho)
                op_geoA = _b11(op_geoA_b + 1e-30)
                op_geoB = _b11(op_geoB_b + 1e-30)
                # §2.7 Updates (caller applies A += dA, B += dB).
                dA = -(rho_unsq / op_geoA) * geo_A
                dB = -(self.lora_plus_multiplier * rho_unsq / op_geoB) * geo_B
                trace_items = {
                    "u_A_eff": u_A_eff,
                    "u_B_eff": u_B_eff,
                    "X_A_eff": X_A_eff,
                    "X_B_eff": X_B_eff,
                    "P_A": P_A,
                    "P_B": P_B,
                    "geo_A": geo_A,
                    "geo_B": geo_B,
                    "op_geoA_b": op_geoA_b,
                    "op_geoB_b": op_geoB_b,
                    "dA": dA,
                    "dB": dB,
                }
                if n > 0:
                    trace_items.update({
                        "BT_dB_A": BT_dB_A,
                        "B_dA_AT": B_dA_AT,
                    })
                if self.polar_method == "ssc" and self.ssc_kappa is not None:
                    trace_items.update({
                        "ssc_c_A": gs.get("ssc_c_last_A"),
                        "ssc_c_B": gs.get("ssc_c_last_B"),
                    })
                _trace_picard(n, **trace_items)

        return {
            "u_A": u_A, "u_B": u_B,
            "SA_half_inv_k": SA_half_inv, "SB_half_inv_k": SB_half_inv,
            "sigma_A": sigma_A, "sigma_B": sigma_B,
            "rho": rho, "picard_coeff_s": picard_coeff_s,
            "u_A_eff": u_A_eff, "u_B_eff": u_B_eff,
            "X_A": X_A, "X_B": X_B,
            "P_A": P_A, "P_B": P_B,
            "geo_A": geo_A, "geo_B": geo_B,
            "op_geoA_b": op_geoA_b, "op_geoB_b": op_geoB_b,
            "op_geoA": op_geoA, "op_geoB": op_geoB,
            "dA": dA, "dB": dB,
            "dA_prev_picard": dA_prev_picard,
            "dB_prev_picard": dB_prev_picard,
            "BT_dB_A": BT_dB_A,
            "B_dA_AT": B_dA_AT,
            "X_A_eff": X_A_eff,
            "X_B_eff": X_B_eff,
            "s_AB": s_AB,
            "picard_trace": picard_trace,
        }

    @torch.no_grad()
    def _step_batched(self):
        """Production hot path: shape-grouped 3D buffers + batched primitives.

        Behaviorally equivalent to `_step_per_pair` (verified by
        `tests/test_polar_product_batched_equivalence.py`) under the eligibility
        conditions in `_batched_path_eligible`. Keeps the same per-pair
        precond_refresh (eigh / higham per pair) since precond batching has its
        own algorithmic risk story; just removes the launch storm from Adam,
        polar NS, polar unwhiten/rescale, and apply.
        """
        lr = self.param_groups[0]["lr"]
        timer = getattr(self, "_step_timer", None)

        # NaN-trigger detector: scan every pair's (A, B, grad_A, grad_B) for
        # non-finite entries BEFORE consuming gradients. Identifies which
        # pair / which tensor went bad at the moment of failure, and dumps
        # the prior step's per-pair stats so the precursor state is
        # captured. Gated on `log_non_finite` because the kernel-launch
        # cost (~448 isfinite reductions at r=256, plus the end-of-step
        # chain check below) compounds to ~10% wall overhead. Default OFF;
        # turn on with --log_non_finite for NaN-debugging runs.
        step_now = (next(iter(self.pair_state.values())).get('step', 0) + 1
                    if self.pair_state else 1)
        log_non_finite_now = (
            self.log_non_finite
            and step_now >= self.log_non_finite_start_step
        )
        if log_non_finite_now:
            for i, (A, B) in enumerate(self.pairs):
                gA, gB = A.grad, B.grad
                checks = {
                    "A": bool(~torch.isfinite(A).all()) if A.numel() else False,
                    "B": bool(~torch.isfinite(B).all()) if B.numel() else False,
                    "grad_A": bool(~torch.isfinite(gA).all()) if gA is not None else False,
                    "grad_B": bool(~torch.isfinite(gB).all()) if gB is not None else False,
                }
                if any(checks.values()):
                    name = self.pair_names[i] if i < len(self.pair_names) else f"pair_{i}"
                    last_diag = self.pair_state.get(i, {}).get('last_diag')
                    _emit_non_finite_event(step_now, i, name, checks, last_diag)

        # Optimization B (cross-group eigvalsh batching for κ-adaptive SSC):
        # When active, run Adam + precond + sigma_AB-hoist for every shape
        # group up-front, then call `_xgroup_ssc_kappa_preflight` once to
        # compute c per (side, picard iter) via a single stacked eigvalsh.
        # The main per-group loop below skips its own eigvalsh calls
        # (intercepted in `_chord_tight_clean_polar_pipeline._polar`).
        xgroup_active = (
            self.ssc_kappa_cross_group_eigvalsh
            and self.magnitude_rule == "spectral_chord_tight_clean"
            and self.polar_method == "ssc"
            and self.ssc_kappa is not None
            and self.ssc_kappa_solver == "eigvalsh"
            and self.fw_linearization != "full"
            and len(self.group_state) > 1
        )
        # Skip preflight when no group needs a c-refresh this step. Without this
        # gate the preflight runs unconditionally, paying full eigvalsh + MISR +
        # power-iter cost every step and silently bypassing the refresh schedule
        # that the per-group fast path would otherwise honor.
        if xgroup_active:
            N_refresh = self.ssc_kappa_refresh_every
            M_warmup = self.ssc_kappa_warmup_steps
            share_picard = self.ssc_kappa_cache_share_picard
            any_refresh_needed = False
            for gs in self.group_state:
                indices = gs['indices']
                if not indices:
                    continue
                s_post = self.pair_state[indices[0]]['step'] + 1
                if share_picard:
                    cache_missing = (
                        'ssc_c_cached_A' not in gs
                        or 'ssc_c_cached_B' not in gs
                    )
                else:
                    cache_missing = any(
                        f'ssc_c_cached_{side}_n{n}' not in gs
                        for side in ('A', 'B')
                        for n in range(self.picard_iters)
                    )
                if (cache_missing or N_refresh == 1
                        or s_post <= M_warmup
                        or (s_post - 1) % N_refresh == 0):
                    any_refresh_needed = True
                    break
            if not any_refresh_needed:
                xgroup_active = False

        diag_records = []

        # Cross-group eigvalsh pre-flight (Optimization B). Runs ONCE
        # before the per-group main loop, stages each group's stacked
        # tensors and increments pair_state['step'], runs a non-mutating
        # dry-run of Adam+precond+sigma_hoist, then invokes the cross-
        # group eigvalsh that populates gs['_xgroup_c_pre']. The main
        # loop below detects '_xgroup_prestaged' and skips redundant
        # stacking/increment. Adam/precond/sigma_hoist re-run for real
        # (with mutations) — bit-identical to dry-run since we mirror
        # the exact op sequence in fp32.
        for gs in self.group_state:
            gs.pop('_xgroup_c_pre', None)
            gs.pop('_xgroup_prestaged', None)
        if xgroup_active:
            per_group_args = []
            for gs in self.group_state:
                indices = gs['indices']
                A_list = [self.pairs[gi][0] for gi in indices]
                B_list = [self.pairs[gi][1] for gi in indices]
                for j, gi in enumerate(indices):
                    if A_list[j].grad is None or B_list[j].grad is None:
                        raise ValueError(
                            "Gradients are required for AdamPolarProductLoRA update.")
                    self.pair_state[gi]['step'] += 1
                gs['A_stack'] = torch.stack(A_list).float()
                gs['B_stack'] = torch.stack(B_list).float()
                gs['gA_stack'] = torch.stack([A.grad for A in A_list]).float()
                gs['gB_stack'] = torch.stack([B.grad for B in B_list]).float()
                gs['_xgroup_prestaged'] = True
                prep = self._xgroup_section_a_dryrun(gs, lr)
                if prep is None:
                    # Dry-run unsupported (e.g., per-pair eigh) — disable
                    # the optimization for this step, fall back to the
                    # legacy per-group path.
                    for gs2 in self.group_state:
                        gs2.pop('_xgroup_prestaged', None)
                    xgroup_active = False
                    break
                per_group_args.append({'gs': gs, **prep})
            if xgroup_active:
                self._xgroup_ssc_kappa_preflight(per_group_args, lr)

        for gs in self.group_state:
            N = gs['N']
            indices = gs['indices']

            if gs.pop('_xgroup_prestaged', False):
                # Stacks + step increment already done in the pre-flight
                # phase above. Recover A_list/B_list for the apply step
                # at the end of this iteration.
                A_list = [self.pairs[gi][0] for gi in indices]
                B_list = [self.pairs[gi][1] for gi in indices]
            else:
                # Stack params + grads. `torch.stack` is one launch per buffer
                # (vs N copy_ launches for the per-pair pattern); for OLMo r=64
                # with N=64 in the largest group, that's ~3 ms saved per step.
                A_list = [self.pairs[gi][0] for gi in indices]
                B_list = [self.pairs[gi][1] for gi in indices]
                for j, gi in enumerate(indices):
                    if A_list[j].grad is None or B_list[j].grad is None:
                        raise ValueError("Gradients are required for AdamPolarProductLoRA update.")
                    self.pair_state[gi]['step'] += 1
                gs['A_stack'] = torch.stack(A_list).float()
                gs['B_stack'] = torch.stack(B_list).float()
                gs['gA_stack'] = torch.stack([A.grad for A in A_list]).float()
                gs['gB_stack'] = torch.stack([B.grad for B in B_list]).float()

            # Diagnostic snapshot stash (gated). A/B here are the values the
            # §9 step is about to consume — they're the algorithm inputs at
            # this step. Clone the per-pair float32 slices into pair_state
            # for save_checkpoint to pick up.
            if self.snapshot_pair_tensors:
                for j, gi in enumerate(indices):
                    self.pair_state[gi]['A'] = gs['A_stack'][j].detach().clone()
                    self.pair_state[gi]['B'] = gs['B_stack'][j].detach().clone()

            step_count = self.pair_state[indices[0]]['step']

            # Batched Adam: in-place on group buffers (one launch per op vs N).
            with maybe_time(timer, "adam_direction"):
                gs['m_A'].mul_(self.beta1).add_(gs['gA_stack'], alpha=1.0 - self.beta1)
                gs['m_B'].mul_(self.beta1).add_(gs['gB_stack'], alpha=1.0 - self.beta1)
                gs['v_A'].mul_(self.beta2).addcmul_(gs['gA_stack'], gs['gA_stack'], value=1.0 - self.beta2)
                gs['v_B'].mul_(self.beta2).addcmul_(gs['gB_stack'], gs['gB_stack'], value=1.0 - self.beta2)
                # Curvature-whitening EMA: matrix second moments of the factor
                # gradients (r×r). Updated every step; the whitener reads these
                # at its refresh cadence. S_curv_A whitens the A-update, S_curv_B
                # the B-update (see __init__ note).
                if self.curvature_whitening:
                    _ScA = gs['gA_stack'] @ gs['gA_stack'].transpose(-2, -1)
                    _ScB = gs['gB_stack'].transpose(-2, -1) @ gs['gB_stack']
                    if gs.get('Scurv_A') is None:
                        gs['Scurv_A'] = _ScA.clone()
                        gs['Scurv_B'] = _ScB.clone()
                    else:
                        _cb = self.curvature_beta
                        gs['Scurv_A'].mul_(_cb).add_(_ScA, alpha=1.0 - _cb)
                        gs['Scurv_B'].mul_(_cb).add_(_ScB, alpha=1.0 - _cb)
                bc1 = 1.0 - self.beta1 ** step_count
                bc2 = 1.0 - self.beta2 ** step_count
                u_A = (gs['m_A'] / bc1) / ((gs['v_A'] / bc2).sqrt() + self.eps)
                u_B = (gs['m_B'] / bc1) / ((gs['v_B'] / bc2).sqrt() + self.eps)

            # Diagnostic snapshot stash (gated): the Adam-RMS direction here
            # is the muon-squared-relevant object, before the σ_max normalization
            # at lines further down rescales u_A, u_B in place.
            if self.snapshot_pair_tensors:
                for j, gi in enumerate(indices):
                    self.pair_state[gi]['u_A'] = u_A[j].detach().clone()
                    self.pair_state[gi]['u_B'] = u_B[j].detach().clone()

            # λ_max hoist for the clean rule: σ_max(A), σ_max(B) are needed
            # downstream by `_chord_tight_clean_polar_pipeline` for ρ = η/s,
            # AND λ_max(S_A) = σ_max(A)² feeds Higham's damping (closed-form
            # `eps_relative` path). Compute them once here so a single
            # canonical σ_max(A) per step is used by both consumers; pass
            # `lam_max=σ_max.pow(2)` into Higham and `sigma_A` / `sigma_B`
            # into the pipeline. Non-clean rules are untouched (the variable
            # stays None and Higham falls back to its internal power iter).
            sigma_A_hoist = None
            sigma_B_hoist = None
            if self.magnitude_rule == "spectral_chord_tight_clean":
                with maybe_time(timer, "chord_tight_clean_sigma_AB"):
                    sigma_A_hoist, v_sigma_A = _sigma_max_power_iter_batched(
                        gs['A_stack'], v_init=gs.get('v_sigma_A'), n_iters=8)
                    sigma_B_hoist, v_sigma_B = _sigma_max_power_iter_batched(
                        gs['B_stack'], v_init=gs.get('v_sigma_B'), n_iters=8)
                    gs['v_sigma_A'] = v_sigma_A
                    gs['v_sigma_B'] = v_sigma_B

            # Precond refresh. precond_method='higham' uses batched
            # `spd_inv_sqrt_higham_batched` (one bmm sequence over all pairs in
            # the group, ~100× faster than per-pair eigh at r=256). 'eigh' stays
            # per-pair: batched_eigh only saves ~1.06× at r=256 (per-batch eigh
            # has real work; launch overhead dominates only at r=16 where eigh
            # is already trivial). Per-pair eigh is the algorithmic baseline.
            #
            # Validated by the r=256 K=1 integration test
            # (`logs/integration_higham_test/`, 2026-05-04): det-init higham
            # ran 1000 steps clean, 0 non_finite_Z events out of 224k probe
            # emits, trajectory within 0.6σ_AdamW peak / 0.07σ final vs the
            # eigh reference.
            if (step_count - 1) % self.precond_refresh_every == 0:
                with maybe_time(timer, "precond_refresh"):
                    if self.disable_whitening:
                        # Identity whitening: SA = SB = I broadcast across pairs.
                        I_A = torch.eye(gs['SA_half_inv'].shape[-1],
                                        dtype=gs['SA_half_inv'].dtype,
                                        device=gs['SA_half_inv'].device)
                        I_B = torch.eye(gs['SB_half_inv'].shape[-1],
                                        dtype=gs['SB_half_inv'].dtype,
                                        device=gs['SB_half_inv'].device)
                        gs['SA_half_inv'].copy_(I_A.expand_as(gs['SA_half_inv']))
                        gs['SB_half_inv'].copy_(I_B.expand_as(gs['SB_half_inv']))
                    else:
                        if self.curvature_whitening and gs.get('Scurv_A') is not None:
                            SA_grams = gs['Scurv_B']   # EMA(g_Bᵀg_B) whitens B-update
                            SB_grams = gs['Scurv_A']   # EMA(g_A g_Aᵀ) whitens A-update
                        else:
                            SA_grams = gs['A_stack'] @ gs['A_stack'].transpose(-2, -1)
                            SB_grams = gs['B_stack'].transpose(-2, -1) @ gs['B_stack']
                        if self.precond_method == "higham":
                            from .utils import spd_inv_sqrt_higham_batched
                            if self.curvature_whitening and gs.get('Scurv_A') is not None:
                                # PSD curvature grams: λ_max = σ_max(S_curv) directly.
                                _lam_A = _sigma_max_power_iter_batched(SA_grams, n_iters=8)[0]
                                _lam_B = _sigma_max_power_iter_batched(SB_grams, n_iters=8)[0]
                            else:
                                _lam_A = (sigma_A_hoist.pow(2)
                                          if sigma_A_hoist is not None else None)
                                _lam_B = (sigma_B_hoist.pow(2)
                                          if sigma_B_hoist is not None else None)
                            gs['SA_half_inv'].copy_(spd_inv_sqrt_higham_batched(
                                SA_grams, n_iters=self.higham_iters, eps=self.delta,
                                eps_relative=self.precond_delta_relative,
                                lam_max=_lam_A,
                                compute_dtype=self._higham_compute_dtype,
                            ))
                            gs['SB_half_inv'].copy_(spd_inv_sqrt_higham_batched(
                                SB_grams, n_iters=self.higham_iters, eps=self.delta,
                                eps_relative=self.precond_delta_relative,
                                lam_max=_lam_B,
                                compute_dtype=self._higham_compute_dtype,
                            ))
                        else:
                            for k in range(N):
                                gs['SA_half_inv'][k] = _spd_inv_half(
                                    SA_grams[k], eps=self.delta,
                                    method=self.precond_method,
                                    higham_iters=self.higham_iters,
                                    eps_relative=self.precond_delta_relative,
                                )
                                gs['SB_half_inv'][k] = _spd_inv_half(
                                    SB_grams[k], eps=self.delta,
                                    method=self.precond_method,
                                    higham_iters=self.higham_iters,
                                    eps_relative=self.precond_delta_relative,
                                )
            SA_half_inv = gs['SA_half_inv']
            SB_half_inv = gs['SB_half_inv']

            # §10-clean Algorithm 2′: focused dispatch. The helper at
            # `_chord_tight_clean_polar_pipeline` implements the doc-faithful
            # pipeline (single S_B^{-1/2} u_A matmul, 1/η cross-coupling,
            # linear ρ = η/s, Picard-iter-keyed warm-start at site C). The
            # legacy gated branches in the else-block below stay untouched
            # for non-clean rules. See docs/notes/polar_product/algorithm_clean_implementation.md.
            picard_trace = {}
            if self.magnitude_rule == "spectral_chord_tight_clean":
                A_f = gs['A_stack']
                B_f = gs['B_stack']
                _clean_result = self._chord_tight_clean_polar_pipeline(
                    gs, A_f, B_f, u_A, u_B, SA_half_inv, SB_half_inv, lr, timer,
                    sigma_A=sigma_A_hoist, sigma_B=sigma_B_hoist,
                    step_count=step_count,
                )
                u_A = _clean_result['u_A']
                u_B = _clean_result['u_B']
                SA_half_inv_k = _clean_result['SA_half_inv_k']
                SB_half_inv_k = _clean_result['SB_half_inv_k']
                sigma_A = _clean_result['sigma_A']
                sigma_B = _clean_result['sigma_B']
                rho = _clean_result['rho']
                picard_coeff_s = _clean_result['picard_coeff_s']
                u_A_eff = _clean_result['u_A_eff']
                u_B_eff = _clean_result['u_B_eff']
                X_A = _clean_result['X_A']
                X_B = _clean_result['X_B']
                P_A = _clean_result['P_A']
                P_B = _clean_result['P_B']
                geo_A = _clean_result['geo_A']
                geo_B = _clean_result['geo_B']
                op_geoA_b = _clean_result['op_geoA_b']
                op_geoB_b = _clean_result['op_geoB_b']
                op_geoA = _clean_result['op_geoA']
                op_geoB = _clean_result['op_geoB']
                s_AB = _clean_result['s_AB']
                dA = _clean_result['dA']
                dB = _clean_result['dB']
                dA_prev_picard = _clean_result['dA_prev_picard']
                dB_prev_picard = _clean_result['dB_prev_picard']
                BT_dB_A = _clean_result['BT_dB_A']
                B_dA_AT = _clean_result['B_dA_AT']
                X_A_eff = _clean_result['X_A_eff']
                X_B_eff = _clean_result['X_B_eff']
                picard_trace = _clean_result.get('picard_trace') or {}
            else:
                # Unit-polar normalization of u_A, u_B for chord-tight family.
                # The Picard coefficient 2/(ρ·s) was derived assuming the base
                # covector is unit-magnitude in the polar-input space (where the
                # polar map actually operates). Without this normalization, the
                # correction's effective scale is set by Adam's absolute u-magnitude
                # and not by the geometric formula — empirically rendering Picard
                # inert (γ ≈ 0.003 at lr=1e-2; see K vs L diagnostics 2026-05-12).
                #
                # Polar(c·X) = Polar(X) for c > 0, so this is a strict no-op at
                # picard_iters=1 (trajectory bit-identical). Affects only k ≥ 2.
                if self.magnitude_rule in ("spectral_chord_tight",
                                           "spectral_chord_direction"):
                    with maybe_time(timer, "picard_unit_polar_norm"):
                        X_A_pre = SB_half_inv @ u_A             # (N, r, d_in)
                        X_B_pre = u_B @ SA_half_inv             # (N, d_out, r)
                        # σ_max via batched power-iter (8 iters cold-start) —
                        # avoids the cuSOLVER batched-eigvalsh latency hit at
                        # r=256 (~15 ms per (256,256) matrix × 112 pairs = 1-2 s
                        # per call; 4-6 such calls per step at r=256 was the
                        # bottleneck behind chord-whiten's 12 s/step). Power-iter
                        # at n_iters=8 is the original implementation
                        # (replaced by eigvalsh in 57a932b for accuracy; the
                        # actual issue was cold-start 3 iters being too few).
                        # Warm-start from prior step's top singular vector;
                        # 4 iters suffice once warm (vs 8 cold). v_init=None
                        # on first call → helper uses deterministic M @ ones.
                        sigma_XA_b, v_XA = _sigma_max_power_iter_batched(
                            X_A_pre, v_init=gs.get('v_sigma_XA'), n_iters=8)
                        sigma_XB_b, v_XB = _sigma_max_power_iter_batched(
                            X_B_pre, v_init=gs.get('v_sigma_XB'), n_iters=8)
                        gs['v_sigma_XA'] = v_XA
                        gs['v_sigma_XB'] = v_XB
                        sigma_XA = sigma_XA_b + 1e-30
                        sigma_XB = sigma_XB_b + 1e-30
                        u_A = u_A / sigma_XA.unsqueeze(-1).unsqueeze(-1)
                        u_B = u_B / sigma_XB.unsqueeze(-1).unsqueeze(-1)

                # Picard outer loop (batched bmm chains).
                u_A_eff = u_A
                u_B_eff = u_B
                dA_prev = torch.zeros_like(u_A)
                dB_prev = torch.zeros_like(u_B)
                A_f = gs['A_stack']
                B_f = gs['B_stack']
                # In exact_chord mode the per-Picard-iter precond_refresh uses the
                # CURRENT iter's preconditioner. Initial values are the same SA/SB
                # we just computed; recomputed below at k_iter > 0.
                SA_half_inv_k = SA_half_inv
                SB_half_inv_k = SB_half_inv

                # spectral_chord: σ_max(A), σ_max(B) via batched power-iter (8
                # iters cold-start). Originally used 3-iter cold-start power-iter,
                # which under-estimated σ_max → switched to eigvalsh in 57a932b
                # for accuracy, but cuSOLVER batched syevd on (n_pairs, r, r) at
                # r=256 falls back to per-matrix calls with ~15 ms overhead each
                # → 1-2 s per call × 4 such calls per step was the dominant cost
                # in chord-whiten production. 8-iter power-iter is the original
                # method with sufficient iters for accuracy at any r.
                if self.magnitude_rule in ("spectral_chord",
                                           "spectral_chord_tight",
                                           "spectral_chord_direction"):
                    sigma_A, v_A = _sigma_max_power_iter_batched(
                        A_f, v_init=gs.get('v_sigma_A'), n_iters=8)
                    sigma_B, v_B = _sigma_max_power_iter_batched(
                        B_f, v_init=gs.get('v_sigma_B'), n_iters=8)
                    gs['v_sigma_A'] = v_A
                    gs['v_sigma_B'] = v_B
                    if self.magnitude_rule == "spectral_chord":
                        rho = lr / (sigma_A + sigma_B + 1.0)  # (N,)
                    elif self.magnitude_rule == "spectral_chord_tight":
                        s_AB = sigma_A + sigma_B
                        rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * lr)) / 2.0
                    else:  # spectral_chord_direction: dA/dB use λ_dir, but
                           # rho computed for diagnostic comparison consistency
                        s_AB = sigma_A + sigma_B
                        rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * lr)) / 2.0
                    # Cache batched Grams for direction-aware σ_max in the loop.
                    GBB_s = B_f.transpose(-1, -2) @ B_f                    # (N, r, r)
                    GAA_s = A_f @ A_f.transpose(-1, -2)                    # (N, r, r)
                    # Picard cross-coupling coefficient: `2/(ρ·s)`, derived from
                    # the normalized whitened objective (see user-proposed math
                    # +`docs/notes/polar_product/handoff_2026_05_11.md` §Picard).
                    # Shape (N,) — unsqueeze to (N, 1, 1) for broadcast against
                    # cross-coupling matrices (N, r, r). Replaces the legacy
                    # `picard_alpha/lr` scaling, which never matched the chord-tight
                    # bound's variational structure and produced ~null effect from
                    # Picard k≥2 under chord-tight.
                    if self.magnitude_rule in ("spectral_chord_tight",
                                               "spectral_chord_direction"):
                        s_AB = sigma_A + sigma_B
                        picard_coeff_s = (2.0 / (rho * s_AB + 1e-30)).unsqueeze(-1).unsqueeze(-1)
                    else:
                        picard_coeff_s = None
                else:
                    rho = None
                    GBB_s = None
                    GAA_s = None
                    picard_coeff_s = None
                for k_iter in range(self.picard_iters):
                    with maybe_time(timer, "picard_cross_coupling"):
                        if k_iter > 0:
                            if self.exact_chord:
                                # Exact-chord (algorithm.md §2 remark): use
                                # A_eff = A + dA_prev, B_eff = B + dB_prev for
                                # both cross-coupling and the spectral
                                # preconditioner. Refresh SA/SB against the
                                # effective factors via batched higham at every
                                # Picard iter (~1 ms/group at r=64; would be
                                # ~80 ms/group with eigh per-pair).
                                A_eff = A_f + dA_prev
                                B_eff = B_f + dB_prev
                                BT_dB_A = B_eff.transpose(-2, -1) @ dB_prev @ A_f
                                B_dA_AT = B_f @ dA_prev @ A_eff.transpose(-2, -1)
                                if picard_coeff_s is not None:
                                    u_A_eff = u_A + picard_coeff_s * BT_dB_A
                                    u_B_eff = u_B + picard_coeff_s * B_dA_AT
                                else:
                                    u_A_eff = u_A + (self.picard_alpha / lr) * BT_dB_A
                                    u_B_eff = u_B + (self.picard_alpha / lr) * B_dA_AT
                                SA_grams_k = A_eff @ A_eff.transpose(-2, -1)
                                SB_grams_k = B_eff.transpose(-2, -1) @ B_eff
                                if self.precond_method == "higham":
                                    from .utils import spd_inv_sqrt_higham_batched
                                    SA_half_inv_k = spd_inv_sqrt_higham_batched(
                                        SA_grams_k, n_iters=self.higham_iters,
                                        eps=self.delta,
                                        eps_relative=self.precond_delta_relative,
                                    )
                                    SB_half_inv_k = spd_inv_sqrt_higham_batched(
                                        SB_grams_k, n_iters=self.higham_iters,
                                        eps=self.delta,
                                        eps_relative=self.precond_delta_relative,
                                    )
                                else:
                                    SA_half_inv_k = torch.stack([
                                        _spd_inv_half(
                                            SA_grams_k[k_pair], eps=self.delta,
                                            method=self.precond_method,
                                            higham_iters=self.higham_iters,
                                            eps_relative=self.precond_delta_relative)
                                        for k_pair in range(N)])
                                    SB_half_inv_k = torch.stack([
                                        _spd_inv_half(
                                            SB_grams_k[k_pair], eps=self.delta,
                                            method=self.precond_method,
                                            higham_iters=self.higham_iters,
                                            eps_relative=self.precond_delta_relative)
                                        for k_pair in range(N)])
                            else:
                                # Compute-bound chain matmul; bmm vs per-pair was
                                # 0.97× in microbench (neutral). Prefer batched
                                # form for code uniformity.
                                BT_dB_A = B_f.transpose(-2, -1) @ dB_prev @ A_f
                                B_dA_AT = B_f @ dA_prev @ A_f.transpose(-2, -1)
                                if picard_coeff_s is not None:
                                    u_A_eff = u_A + picard_coeff_s * BT_dB_A
                                    u_B_eff = u_B + picard_coeff_s * B_dA_AT
                                else:
                                    u_A_eff = u_A + (self.picard_alpha / lr) * BT_dB_A
                                    u_B_eff = u_B + (self.picard_alpha / lr) * B_dA_AT
                    with maybe_time(timer, "picard_polar_pipeline"):
                        with maybe_time(timer, "polar_whiten"):
                            X_A = SB_half_inv_k @ u_A_eff
                            X_B = u_B_eff @ SA_half_inv_k
                        # Iterate NS in bf16 for ~3.6× throughput on Ampere
                        # tensor cores (microbench: scripts/bench/bench_ns_bf16.py).
                        # Pre-norm + Frobenius rescale stay fp32 (small-number
                        # robustness); only the matmul-heavy iterations run bf16.
                        # Output cast back to fp32 for downstream unwhiten/rescale
                        # which still operates in fp32. Pattern mirrors
                        # modded-nanogpt train_gpt.py:187.
                        with maybe_time(timer, "polar_NS_A"):
                            if self.polar_method == "polar_express":
                                P_A = _polar_express_gram_batched(
                                    X_A, nsteps=self.ns_steps,
                                ).float()
                            else:
                                P_A = _newton_schulz_batched(
                                    X_A, nsteps=self.ns_steps, dtype=torch.bfloat16
                                ).float()
                        with maybe_time(timer, "polar_NS_B"):
                            if self.polar_method == "polar_express":
                                P_B = _polar_express_gram_batched(
                                    X_B, nsteps=self.ns_steps,
                                ).float()
                            else:
                                P_B = _newton_schulz_batched(
                                    X_B, nsteps=self.ns_steps, dtype=torch.bfloat16
                                ).float()
                        with maybe_time(timer, "polar_unwhiten_rescale"):
                            if self.magnitude_rule == "spectral_chord_tight_no_rho":
                                # §8 no-ρ: dA = -lr · S_B^{-1/2} polar(S_B^{-1/2} u_A)
                                # No σ_max(geo) computation, no rescale by ρ/op-norm.
                                geo_A = SB_half_inv_k @ P_A
                                geo_B = P_B @ SA_half_inv_k
                                dA = -lr * geo_A
                                dB = -(self.lora_plus_multiplier * lr) * geo_B
                            elif self.magnitude_rule in ("spectral_chord",
                                                         "spectral_chord_tight",
                                                         "spectral_chord_direction"):
                                geo_A = SB_half_inv_k @ P_A
                                geo_B = P_B @ SA_half_inv_k
                                geoA_f = geo_A.float()
                                geoB_f = geo_B.float()
                                # σ_max via 8-iter batched power-iter; replaces
                                # eigvalsh on (n_pairs, r, r) Gram which has
                                # ~1.5 s/call cuSOLVER overhead at r=256.
                                op_geoA_b, v_geoA = _sigma_max_power_iter_batched(
                                    geoA_f, v_init=gs.get('v_op_geoA'), n_iters=8)
                                op_geoB_b, v_geoB = _sigma_max_power_iter_batched(
                                    geoB_f, v_init=gs.get('v_op_geoB'), n_iters=8)
                                gs['v_op_geoA'] = v_geoA
                                gs['v_op_geoB'] = v_geoB
                                op_geoA = (op_geoA_b + 1e-30).unsqueeze(-1).unsqueeze(-1)
                                op_geoB = (op_geoB_b + 1e-30).unsqueeze(-1).unsqueeze(-1)
                                if self.magnitude_rule == "spectral_chord_direction":
                                    # Variant 1 batched: λ_dir from
                                    # a·λ + b·λ² = lr per pair.
                                    P_dir = geoA_f / op_geoA              # (N, r, n)
                                    Q_dir = geoB_f / op_geoB              # (N, m, r)
                                    PPt_b = P_dir @ P_dir.transpose(-1, -2)
                                    QtQ_b = Q_dir.transpose(-1, -2) @ Q_dir
                                    # Use power-iter on N = G_outer·G_inner (r×r non-symmetric,
                                    # real-positive spectrum). Same per-iter cost as Krylov-chol
                                    # but no Cholesky → no chol-fails-when-rank-deficient bug
                                    # that crashed chord-direction on h100. See
                                    # `docs/notes/sigma_max_estimation.md` and bench data:
                                    # accuracy matches krylov-chol within 1e-3, timing
                                    # within ~5% on RTX A6000 across all regimes including
                                    # rank-deficient (where krylov-chol crashes outright).
                                    sigma_BP_b, _ = _sigma_max_power_iter_nonsym(GBB_s, PPt_b)
                                    sigma_QA_b, _ = _sigma_max_power_iter_nonsym(GAA_s, QtQ_b)
                                    sigma_QP_b, _ = _sigma_max_power_iter_nonsym(PPt_b, QtQ_b)
                                    a_b = (sigma_BP_b + sigma_QA_b).clamp_min(1e-30)
                                    b_b = sigma_QP_b
                                    # Per-pair quadratic: pick lr/a where b≈0,
                                    # else closed-form root. b_b is (N,) so just
                                    # use the closed form everywhere with safe
                                    # denominator clamping.
                                    disc = a_b * a_b + 4.0 * b_b * lr
                                    lam_quad = (-a_b + torch.sqrt(disc)) / (2.0 * b_b.clamp_min(1e-30))
                                    lam_lin = lr / a_b
                                    lam = torch.where(b_b > 1e-30, lam_quad, lam_lin)
                                    lam_unsq = lam.unsqueeze(-1).unsqueeze(-1)
                                    dA = -lam_unsq * P_dir
                                    dB = -(self.lora_plus_multiplier *
                                           lam_unsq) * Q_dir
                                else:
                                    rho_unsq = rho.unsqueeze(-1).unsqueeze(-1)
                                    dA = -(rho_unsq / op_geoA) * geo_A
                                    dB = -(self.lora_plus_multiplier *
                                           rho_unsq / op_geoB) * geo_B
                            else:
                                from ._batched_polar import unwhiten_rescale_frob_batched
                                dA, dB = unwhiten_rescale_frob_batched(
                                    P_A, P_B, SA_half_inv_k, SB_half_inv_k,
                                    u_A_eff, u_B_eff, lr,
                                    lora_plus_multiplier=self.lora_plus_multiplier,
                                )
                    dA_prev = dA
                    dB_prev = dB

            chain_tensors = {
                "u_A": u_A, "u_B": u_B,
                "SA_half_inv": SA_half_inv_k, "SB_half_inv": SB_half_inv_k,
                "sigma_A": locals().get('sigma_A'),
                "sigma_B": locals().get('sigma_B'),
                "rho": locals().get('rho'),
                "picard_coeff_s": locals().get('picard_coeff_s'),
                "u_A_eff": u_A_eff, "u_B_eff": u_B_eff,
                "X_A": locals().get('X_A'), "X_B": locals().get('X_B'),
                "X_A_eff": locals().get('X_A_eff'),
                "X_B_eff": locals().get('X_B_eff'),
                "BT_dB_A": locals().get('BT_dB_A'),
                "B_dA_AT": locals().get('B_dA_AT'),
                "dA_prev_picard": locals().get('dA_prev_picard'),
                "dB_prev_picard": locals().get('dB_prev_picard'),
                "P_A": locals().get('P_A'), "P_B": locals().get('P_B'),
                "geo_A": locals().get('geo_A'), "geo_B": locals().get('geo_B'),
                "op_geoA_b": locals().get('op_geoA_b'),
                "op_geoB_b": locals().get('op_geoB_b'),
                "dA": dA, "dB": dB,
            }
            if picard_trace:
                chain_tensors.update(picard_trace)

            stats = None
            if (
                self.debug_optimizer_state
                and step_count % self.debug_optimizer_state_every == 0
                and step_count >= self.debug_optimizer_state_start_step
            ):
                stats = {
                    "A_norm": _tensor_norm_by_pair(A_f),
                    "B_norm": _tensor_norm_by_pair(B_f),
                    "gA_absmax": _tensor_absmax_by_pair(gs['gA_stack']),
                    "gB_absmax": _tensor_absmax_by_pair(gs['gB_stack']),
                    "mA_absmax": _tensor_absmax_by_pair(gs['m_A']),
                    "mB_absmax": _tensor_absmax_by_pair(gs['m_B']),
                    "vA_absmax": _tensor_absmax_by_pair(gs['v_A']),
                    "vB_absmax": _tensor_absmax_by_pair(gs['v_B']),
                    "uA_absmax": _tensor_absmax_by_pair(u_A),
                    "uB_absmax": _tensor_absmax_by_pair(u_B),
                    "uA_eff_absmax": _tensor_absmax_by_pair(u_A_eff),
                    "uB_eff_absmax": _tensor_absmax_by_pair(u_B_eff),
                    "SA_half_inv_absmax": _tensor_absmax_by_pair(SA_half_inv_k),
                    "SB_half_inv_absmax": _tensor_absmax_by_pair(SB_half_inv_k),
                    "SA_half_inv_finite": _tensor_finite_by_pair(SA_half_inv_k),
                    "SB_half_inv_finite": _tensor_finite_by_pair(SB_half_inv_k),
                    "dA_absmax": _tensor_absmax_by_pair(dA),
                    "dB_absmax": _tensor_absmax_by_pair(dB),
                    "dA_norm": _tensor_norm_by_pair(dA),
                    "dB_norm": _tensor_norm_by_pair(dB),
                    "sigma_A": locals().get('sigma_A'),
                    "sigma_B": locals().get('sigma_B'),
                    "rho": locals().get('rho'),
                    "picard_coeff": (
                        picard_coeff_s.squeeze(-1).squeeze(-1)
                        if picard_coeff_s is not None else None
                    ),
                    "op_geoA": locals().get('op_geoA_b'),
                    "op_geoB": locals().get('op_geoB_b'),
                    "SA_lammax_from_sigma": (
                        sigma_A * sigma_A if locals().get('sigma_A') is not None else None
                    ),
                    "SB_lammax_from_sigma": (
                        sigma_B * sigma_B if locals().get('sigma_B') is not None else None
                    ),
                    "SA_diag_max": A_f.pow(2).sum(dim=-1).amax(dim=-1),
                    "SB_diag_max": B_f.pow(2).sum(dim=-2).amax(dim=-1),
                    "SA_trace": A_f.pow(2).sum(dim=(-1, -2)),
                    "SB_trace": B_f.pow(2).sum(dim=(-1, -2)),
                    "ssc_c_A": gs.get('ssc_c_last_A'),
                    "ssc_c_B": gs.get('ssc_c_last_B'),
                }
                _emit_optimizer_pair_stats(
                    step_count,
                    group_id=gs['gid'],
                    group_global_indices=list(indices),
                    pair_names_in_group=[self.pair_names[gi] for gi in indices],
                    stats=stats,
                )

            # Chain-of-intermediates non-finite check. Emits a single
            # `non_finite_intermediate` event if any intermediate at this
            # group went non-finite. ~20 isfinite reductions over batched
            # tensors. Catches the failure at the link where it was born,
            # not after it propagates to A/B at the start of the next step.
            # Gated on log_non_finite — adds ~5-10% wall on top of the
            # top-of-step check.
            bad_where = None
            if log_non_finite_now:
                bad_where = _emit_non_finite_chain(
                    step_count,
                    chain_tensors,
                    pair_names_in_group=[self.pair_names[gi] for gi in indices],
                    group_global_indices=list(indices),
                )
            if bad_where:
                bad_locals = set()
                for entries in bad_where.values():
                    if isinstance(entries, list):
                        bad_locals.update(int(e["local"]) for e in entries)
                for local_bad in sorted(bad_locals):
                    self._save_debug_snapshot(
                        reason="non_finite_intermediate",
                        step_count=step_count,
                        group_state=gs,
                        local_idx=local_bad,
                        global_idx=indices[local_bad],
                        tensors={
                            **chain_tensors,
                            "A": A_f,
                            "B": B_f,
                            "gA": gs['gA_stack'],
                            "gB": gs['gB_stack'],
                            "m_A": gs['m_A'],
                            "m_B": gs['m_B'],
                            "v_A": gs['v_A'],
                            "v_B": gs['v_B'],
                        },
                        scalars=stats or {},
                        where=bad_where,
                    )
                if self.debug_abort_on_non_finite:
                    raise RuntimeError(
                        f"Non-finite optimizer intermediate at step {step_count}; "
                        f"snapshot_dir={self.debug_snapshot_dir!r}"
                    )

            # Basic-tier diagnostics on probe steps: slice per-pair from the
            # batched 3D buffers and call the shared _emit_basic_diagnostics
            # helper. Lets the batched path stay on probe steps without
            # falling back to per-pair (case-(a) of _batched_path_eligible).
            is_probe_step = (
                self.log_basic_diagnostics
                and step_count % self.diagnostics_every == 0
            )
            if is_probe_step:
                # Recover per-pair geo_*, uA_norm, uB_norm, gA_norm, gB_norm
                # for the frob branch (not exposed by unwhiten_rescale_frob_batched).
                if self.magnitude_rule not in ("spectral_chord",
                                               "spectral_chord_tight",
                                               "spectral_chord_tight_clean",
                                               "spectral_chord_tight_no_rho",
                                               "spectral_chord_direction"):
                    geo_A_diag = SB_half_inv_k @ P_A
                    geo_B_diag = P_B @ SA_half_inv_k
                    uA_norm_b = u_A_eff.flatten(-2).norm(dim=-1)        # (N,)
                    uB_norm_b = u_B_eff.flatten(-2).norm(dim=-1)        # (N,)
                    gA_norm_b = geo_A_diag.flatten(-2).norm(dim=-1) + 1e-30
                    gB_norm_b = geo_B_diag.flatten(-2).norm(dim=-1) + 1e-30
                    op_geoA_b_diag = None
                    op_geoB_b_diag = None
                    sigma_A_b_diag = None
                    sigma_B_b_diag = None
                else:
                    geo_A_diag = geo_A
                    geo_B_diag = geo_B
                    # uA_norm/uB_norm follow per-pair convention: rho (or lam)
                    # for chord variants where the unit-polar-norm has
                    # absorbed the original Adam magnitude into the rescale.
                    # spectral_chord_tight_no_rho has neither rho nor lam —
                    # use raw Adam magnitude (Frobenius) as the fallback.
                    if self.magnitude_rule == "spectral_chord_direction":
                        uA_norm_b = lam.detach()
                        uB_norm_b = lam.detach()
                    elif self.magnitude_rule == "spectral_chord_tight_no_rho":
                        uA_norm_b = u_A_eff.flatten(-2).norm(dim=-1)        # (N,)
                        uB_norm_b = u_B_eff.flatten(-2).norm(dim=-1)        # (N,)
                    else:
                        uA_norm_b = rho.detach()
                        uB_norm_b = rho.detach()
                    # op_geoA/op_geoB are (N, 1, 1) for chord variants that
                    # rescale by σ_max(geo). no_rho skips the rescale entirely
                    # so it never computes them — use Frobenius of geo as the
                    # fallback gA_norm/gB_norm and pass None for the op-norm
                    # diagnostic (consumed downstream as `if not None`).
                    if self.magnitude_rule == "spectral_chord_tight_no_rho":
                        op_geoA_b_diag = None
                        op_geoB_b_diag = None
                        gA_norm_b = geo_A.flatten(-2).norm(dim=-1) + 1e-30
                        gB_norm_b = geo_B.flatten(-2).norm(dim=-1) + 1e-30
                        sigma_A_b_diag = None
                        sigma_B_b_diag = None
                    else:
                        op_geoA_b_diag = op_geoA.squeeze(-1).squeeze(-1)
                        op_geoB_b_diag = op_geoB.squeeze(-1).squeeze(-1)
                        gA_norm_b = op_geoA_b_diag
                        gB_norm_b = op_geoB_b_diag
                        sigma_A_b_diag = sigma_A
                        sigma_B_b_diag = sigma_B

                picard_coeff_b = (picard_coeff_s.squeeze(-1).squeeze(-1)
                                  if picard_coeff_s is not None else None)
                s_AB_b = s_AB if self.magnitude_rule in (
                    "spectral_chord_tight", "spectral_chord_tight_clean",
                    "spectral_chord_direction"
                ) else None
                # κ-adaptive SSC stashes the per-pair solved c into gs;
                # surface it for the per-pair diagnostic emitter.
                ssc_c_A_b = gs.get('ssc_c_last_A')
                ssc_c_B_b = gs.get('ssc_c_last_B')

                for k_pair in range(N):
                    gi = indices[k_pair]
                    state_k = self.pair_state[gi]
                    rec = self._emit_basic_diagnostics(
                        state=state_k,
                        A=A_list[k_pair], B=B_list[k_pair],
                        A_f=A_f[k_pair], B_f=B_f[k_pair],
                        u_A=u_A[k_pair], u_B=u_B[k_pair],
                        dA=dA[k_pair], dB=dB[k_pair],
                        gA=gs['gA_stack'][k_pair], gB=gs['gB_stack'][k_pair],
                        SA_half_inv=SA_half_inv_k[k_pair],
                        SB_half_inv=SB_half_inv_k[k_pair],
                        geo_A=geo_A_diag[k_pair], geo_B=geo_B_diag[k_pair],
                        sigma_A_t=(sigma_A_b_diag[k_pair]
                                   if sigma_A_b_diag is not None else None),
                        sigma_B_t=(sigma_B_b_diag[k_pair]
                                   if sigma_B_b_diag is not None else None),
                        op_geoA=(op_geoA_b_diag[k_pair]
                                 if op_geoA_b_diag is not None else None),
                        op_geoB=(op_geoB_b_diag[k_pair]
                                 if op_geoB_b_diag is not None else None),
                        uA_norm=uA_norm_b[k_pair],
                        uB_norm=uB_norm_b[k_pair],
                        gA_norm=gA_norm_b[k_pair],
                        gB_norm=gB_norm_b[k_pair],
                        picard_coeff_t=(picard_coeff_b[k_pair]
                                        if picard_coeff_b is not None else None),
                        rho=(rho[k_pair] if rho is not None else None),
                        s_AB=(s_AB_b[k_pair] if s_AB_b is not None else None),
                        lr=lr,
                        ssc_c_A=(ssc_c_A_b[k_pair] if ssc_c_A_b is not None else None),
                        ssc_c_B=(ssc_c_B_b[k_pair] if ssc_c_B_b is not None else None),
                    )
                    diag_records.append(rec)
                    # Cache so the next step's NaN-trigger event can
                    # reference the immediately-prior per-pair state.
                    self.pair_state[gi]['last_diag'] = rec

            # Apply via `torch._foreach_*`: one multi-tensor kernel each for
            # add and zero, instead of 4N per-pair launches. ~4 ms saved
            # per step at OLMo r=64.
            with maybe_time(timer, "apply"):
                # Cast the entire (N, ...) dA/dB stack to native dtype in one op.
                target_dtype = A_list[0].dtype
                dA_native = dA.to(target_dtype)
                dB_native = dB.to(target_dtype)
                torch._foreach_add_(A_list, list(dA_native.unbind(0)))
                torch._foreach_add_(B_list, list(dB_native.unbind(0)))
                torch._foreach_zero_([A.grad for A in A_list])
                torch._foreach_zero_([B.grad for B in B_list])
            if log_non_finite_now:
                bad_after = {}
                for local_idx, A_param in enumerate(A_list):
                    if bool(~torch.isfinite(A_param).all()):
                        bad_after.setdefault("A_after", []).append(local_idx)
                for local_idx, B_param in enumerate(B_list):
                    if bool(~torch.isfinite(B_param).all()):
                        bad_after.setdefault("B_after", []).append(local_idx)
                if bad_after and _is_main_process():
                    where = {
                        name: [
                            {
                                "local": int(li),
                                "global": int(indices[li]),
                                "pair_name": (
                                    self.pair_names[indices[li]]
                                    if indices[li] < len(self.pair_names)
                                    else f"pair_{indices[li]}"
                                ),
                            }
                            for li in local_idxs
                        ]
                        for name, local_idxs in bad_after.items()
                    }
                    print(json.dumps({
                        "event": "non_finite_after_apply",
                        "step": int(step_count),
                        "where": where,
                    }, sort_keys=True), flush=True)
                    bad_locals = sorted({
                        int(li)
                        for local_idxs in bad_after.values()
                        for li in local_idxs
                    })
                    A_after = torch.stack([p.detach().float() for p in A_list])
                    B_after = torch.stack([p.detach().float() for p in B_list])
                    for local_bad in bad_locals:
                        self._save_debug_snapshot(
                            reason="non_finite_after_apply",
                            step_count=step_count,
                            group_state=gs,
                            local_idx=local_bad,
                            global_idx=indices[local_bad],
                            tensors={
                                **chain_tensors,
                                "A_before": A_f,
                                "B_before": B_f,
                                "A_after": A_after,
                                "B_after": B_after,
                                "dA_native": dA_native,
                                "dB_native": dB_native,
                                "gA": gs['gA_stack'],
                                "gB": gs['gB_stack'],
                                "m_A": gs['m_A'],
                                "m_B": gs['m_B'],
                                "v_A": gs['v_A'],
                                "v_B": gs['v_B'],
                            },
                            scalars=stats or {},
                            where=where,
                        )
                    if self.debug_abort_on_non_finite:
                        raise RuntimeError(
                            f"Non-finite optimizer parameter after apply at "
                            f"step {step_count}; "
                            f"snapshot_dir={self.debug_snapshot_dir!r}"
                        )

        if self.log_basic_diagnostics and diag_records:
            step_count_any = self.pair_state[0]['step']
            if step_count_any % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count_any, diag_records)

    # magnitude_rules that the batched path implements but `_step_per_pair`
    # does NOT. Routing these through per-pair silently runs the
    # adam_frobenius rescale instead of the intended chord/spectral rule
    # — produces wrong-magnitude updates and divergence. Refuse loudly.
    _MAGNITUDE_RULES_BATCHED_ONLY = frozenset({
        "spectral_chord_tight_clean",
        "spectral_chord_tight_no_rho",
    })

    @torch.no_grad()
    def _step_per_pair(self, closure=None):
        if self.magnitude_rule in self._MAGNITUDE_RULES_BATCHED_ONLY:
            raise NotImplementedError(
                f"_step_per_pair does not implement magnitude_rule="
                f"{self.magnitude_rule!r} (the post-_polar_pipeline "
                f"chord-rescale at line ~6225 only covers the legacy chord "
                f"family). Use a config that satisfies "
                f"_batched_path_eligible() — see that method for the "
                f"requirements (polar_method ∈ {{ns, ssc, polar_express}}, "
                f"compatible picard_iters / anderson / end_rms_align flags)."
            )
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_basic_diagnostics else None
        timer = getattr(self, "_step_timer", None)

        for i, (A, B) in enumerate(self.pairs):
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for AdamPolarProductLoRA update.")
            state = self.pair_state[i]
            state['step'] += 1
            # Diagnostics-only sentinels; chord variants assign these but the
            # frob / default branch doesn't. _emit_basic_diagnostics takes
            # them as kwargs unconditionally.
            s_AB = None
            sigma_A_t = None
            sigma_B_t = None
            op_geoA = None
            op_geoB = None
            rho = None
            picard_coeff_t = None

            gA = A.grad.float()
            gB = B.grad.float()

            # Adam direction in the polar pipeline's input frame. The default
            # implementation runs per-coord Adam on raw (gA, gB); subclasses
            # (e.g. AdamSOAPPolarProductLoRA) may override this hook to run
            # Adam in a data-derived eigenbasis.
            with maybe_time(timer, "adam_direction"):
                u_A, u_B = self._adam_direction(state, gA, gB)

            # Spectral square-root preconditioners. Refresh every K steps; reuse
            # cached value otherwise. K=1 ⇒ refresh every step (original behavior).
            # precond_method='higham' uses Newton-Schulz iteration instead of eigh
            # — ~10× faster at r=256 by avoiding the eigh kernel-launch storm.
            if (state['step'] - 1) % self.precond_refresh_every == 0:
                with maybe_time(timer, "precond_refresh"):
                    if self.disable_whitening:
                        r_A = A.shape[0]
                        r_B = B.shape[1]
                        state['SA_half_inv'] = torch.eye(
                            r_A, dtype=A.dtype, device=A.device).float()
                        state['SB_half_inv'] = torch.eye(
                            r_B, dtype=A.dtype, device=A.device).float()
                    else:
                        state['SA_half_inv'] = _spd_inv_half(
                            A.float() @ A.float().T, eps=self.delta,
                            method=self.precond_method, higham_iters=self.higham_iters,
                            eps_relative=self.precond_delta_relative,
                        )
                        state['SB_half_inv'] = _spd_inv_half(
                            B.float().T @ B.float(), eps=self.delta,
                            method=self.precond_method, higham_iters=self.higham_iters,
                            eps_relative=self.precond_delta_relative,
                        )
            SA_half_inv = state['SA_half_inv']
            SB_half_inv = state['SB_half_inv']

            # Experimental core-coordinate remix: replace the projections of
            # (u_A, u_B) onto row(A) / col(B) with remixed versions. The
            # exclusive (orthogonal) parts of (u_A, u_B) are unchanged.
            # alpha=0: no-op (baseline). Nonzero alpha is an ablation knob.
            if self.core_remix_alpha > 0:
                A_f_remix = A.float()
                B_f_remix = B.float()
                alpha = self.core_remix_alpha
                H_A = u_A @ A_f_remix.T            # (r, r)
                H_B = B_f_remix.T @ u_B            # (r, r)
                tilde_H_A = (1.0 - alpha) * H_A - alpha * H_B
                tilde_H_B = (1.0 - alpha) * H_B - alpha * H_A
                GA_inv = SA_half_inv @ SA_half_inv  # (r, r) ≈ (A A^T + δI)^{-1}
                GB_inv = SB_half_inv @ SB_half_inv  # (r, r) ≈ (B^T B + δI)^{-1}
                u_A_core_old = (H_A @ GA_inv) @ A_f_remix
                u_B_core_old = B_f_remix @ (GB_inv @ H_B)
                u_A_core_new = (tilde_H_A @ GA_inv) @ A_f_remix
                u_B_core_new = B_f_remix @ (GB_inv @ tilde_H_B)
                u_A = u_A - u_A_core_old + u_A_core_new
                u_B = u_B - u_B_core_old + u_B_core_new

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
            # Default picard cross-coupling coefficient — overridden below for
            # the chord-tight family. Initialized here so the diagnostic block
            # can read it for any magnitude_rule.
            picard_coeff_t = None
            # σ_max(A), σ_max(B) for the spectral_chord magnitude rules.
            # Computed via exact eigh on the r×r Gram (smaller side). For
            # spectral_chord_direction the values are still useful as a
            # sanity reference (dir_a_over_s diagnostic) but ρ uses the
            # direction-aware a, b instead. See diagnostics doc for the
            # power-iter-under-estimate story that motivated eigh here.
            if self.magnitude_rule in ("spectral_chord",
                                       "spectral_chord_tight",
                                       "spectral_chord_direction"):
                # σ_max via 8-iter power-iter (matches batched-path
                # implementation since `eigvalsh → power-iter` switch for
                # r=256 latency).
                sigma_A_t, _ = _sigma_max_power_iter(A_f, n_iters=8)
                sigma_B_t, _ = _sigma_max_power_iter(B_f, n_iters=8)
                if self.magnitude_rule == "spectral_chord":
                    rho = lr / (sigma_A_t + sigma_B_t + 1.0)
                elif self.magnitude_rule == "spectral_chord_tight":
                    s_AB = sigma_A_t + sigma_B_t
                    rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * lr)) / 2.0
                else:
                    # spectral_chord_direction: dA/dB use the per-Picard-iter
                    # λ_dir (direction-aware) rather than this scalar ρ. The
                    # tight-chord ρ value is computed anyway so the diagnostic
                    # probe block can report lambda_dir_gain = λ_dir / ρ_tight
                    # as a meaningful comparison.
                    s_AB = sigma_A_t + sigma_B_t
                    rho = (-s_AB + torch.sqrt(s_AB * s_AB + 4.0 * lr)) / 2.0
                # Cache GBB, GAA for the direction-aware σ_max(BP), σ_max(QA)
                # computations inside the Picard loop (variant 1).
                GBB_t = B_f.transpose(-1, -2) @ B_f                   # (r, r)
                GAA_t = A_f @ A_f.transpose(-1, -2)                   # (r, r)
                # Picard cross-coupling coefficient `2/(ρ·s)` for chord-tight
                # family. See batched-path comment above for derivation.
                if self.magnitude_rule in ("spectral_chord_tight",
                                           "spectral_chord_direction"):
                    picard_coeff_t = 2.0 / (rho * (sigma_A_t + sigma_B_t) + 1e-30)
                    # Unit-polar normalization (per-pair mirror of batched path).
                    # σ_max via 8-iter power-iter (matches batched-path).
                    XA_pre_t = SB_half_inv @ u_A
                    XB_pre_t = u_B @ SA_half_inv
                    sA_pre, _ = _sigma_max_power_iter(XA_pre_t.float(), n_iters=8)
                    sB_pre, _ = _sigma_max_power_iter(XB_pre_t.float(), n_iters=8)
                    sA_pre = sA_pre.clamp_min(1e-30)
                    sB_pre = sB_pre.clamp_min(1e-30)
                    u_A = u_A / sA_pre
                    u_B = u_B / sB_pre
                else:
                    picard_coeff_t = None
            # Anderson history: list of (x_flat, g_flat) where x is the input
            # to G and g = G(x) is the output. Only used when anderson_m > 0.
            and_xs = [] if self.anderson_m > 0 else None
            and_gs = [] if self.anderson_m > 0 else None
            shapeA = A_f.shape
            shapeB = B_f.shape
            nA_el = A_f.numel()
            for k in range(self.picard_iters):
                with maybe_time(timer, "picard_cross_coupling"):
                    if k == 0:
                        u_A_eff = u_A
                        u_B_eff = u_B
                        SA_half_inv_k = SA_half_inv
                        SB_half_inv_k = SB_half_inv
                    else:
                        if self.exact_chord:
                            # Exact-chord cross-coupling: keep ΔB·ΔA term (§2 remark).
                            A_eff = A_f + dA_prev
                            B_eff = B_f + dB_prev
                            if picard_coeff_t is not None:
                                u_A_eff = u_A + picard_coeff_t * (B_eff.T @ dB_prev @ A_f)
                                u_B_eff = u_B + picard_coeff_t * (B_f @ dA_prev @ A_eff.T)
                            else:
                                u_A_eff = u_A + self.picard_alpha * (B_eff.T @ dB_prev @ A_f) / lr
                                u_B_eff = u_B + self.picard_alpha * (B_f @ dA_prev @ A_eff.T) / lr
                            # Recompute spectral preconditioners against the effective
                            # factors. Required because S_{B+dB} ≠ S_B in general.
                            SA_half_inv_k = _spd_inv_half(
                                A_eff @ A_eff.T, eps=self.delta,
                                method=self.precond_method, higham_iters=self.higham_iters,
                                eps_relative=self.precond_delta_relative,
                            )
                            SB_half_inv_k = _spd_inv_half(
                                B_eff.T @ B_eff, eps=self.delta,
                                method=self.precond_method, higham_iters=self.higham_iters,
                                eps_relative=self.precond_delta_relative,
                            )
                        else:
                            if picard_coeff_t is not None:
                                u_A_eff = u_A + picard_coeff_t * (B_f.T @ dB_prev @ A_f)
                                u_B_eff = u_B + picard_coeff_t * (B_f @ dA_prev @ A_f.T)
                            else:
                                u_A_eff = u_A + self.picard_alpha * (B_f.T @ dB_prev @ A_f) / lr
                                u_B_eff = u_B + self.picard_alpha * (B_f @ dA_prev @ A_f.T) / lr
                            SA_half_inv_k = SA_half_inv
                            SB_half_inv_k = SB_half_inv
                with maybe_time(timer, "picard_polar_pipeline"):
                    dA, dB, geo_A, geo_B, uA_norm, uB_norm, gA_norm, gB_norm, _, _ = \
                        self._polar_pipeline(u_A_eff, u_B_eff, SA_half_inv_k, SB_half_inv_k, lr)
                use_chord_rescale = self.magnitude_rule in (
                    "spectral_chord", "spectral_chord_tight",
                    "spectral_chord_direction",
                )
                if use_chord_rescale:
                    # σ_max(geo_A), σ_max(geo_B) via exact eigh on the r×r
                    # Gram (geo · geoᵀ on the smaller side). Required for
                    # both the chord-rescale normalization (dA = -·geo_A/op_geoA)
                    # and the direction-aware-bound coefficients.
                    geoA_f = geo_A.float()
                    geoB_f = geo_B.float()
                    # σ_max via 8-iter power-iter (matches batched-path).
                    op_geoA, _ = _sigma_max_power_iter(geoA_f, n_iters=8)
                    op_geoB, _ = _sigma_max_power_iter(geoB_f, n_iters=8)
                    op_geoA = op_geoA + 1e-30
                    op_geoB = op_geoB + 1e-30
                    if self.magnitude_rule == "spectral_chord_direction":
                        # Variant 1: solve a·λ + b·λ² = lr with the
                        # direction-aware coefficients
                        #   a = ‖B·P‖_2 + ‖Q·A‖_2,  b = ‖Q·P‖_2,
                        # P = geo_A / σ_max(geo_A), Q = geo_B / σ_max(geo_B).
                        # σ_max(B·P), σ_max(Q·A), σ_max(Q·P) via Cholesky+
                        # eigvalsh symmetric reduction on r×r — exact, ~one
                        # eigvalsh launch per quantity per pair per Picard
                        # iter. See algorithm_tight_chord.md variant 1 and
                        # tight_chord_diagnostics_stage0.md F3 finding.
                        P_dir = (geoA_f / op_geoA).detach()           # (r, n), op-norm 1
                        Q_dir = (geoB_f / op_geoB).detach()           # (m, r), op-norm 1
                        PPt = P_dir @ P_dir.transpose(-1, -2)         # (r, r) sym PSD
                        QtQ = Q_dir.transpose(-1, -2) @ Q_dir         # (r, r) sym PSD
                        # Per-pair chord-direction σ_max via power-iter on
                        # N = G_outer · G_inner. No Cholesky; see _step_batched
                        # callsite above and `docs/notes/sigma_max_estimation.md`.
                        # Inputs are 2-D (r, r) here; add leading batch dim of 1.
                        _g_outer_BP = GBB_t.unsqueeze(0) if GBB_t.dim() == 2 else GBB_t
                        _g_inner_PP = PPt.unsqueeze(0) if PPt.dim() == 2 else PPt
                        _g_outer_QA = GAA_t.unsqueeze(0) if GAA_t.dim() == 2 else GAA_t
                        _g_inner_QQ = QtQ.unsqueeze(0) if QtQ.dim() == 2 else QtQ
                        sigma_BP, _ = _sigma_max_power_iter_nonsym(_g_outer_BP, _g_inner_PP)
                        sigma_QA, _ = _sigma_max_power_iter_nonsym(_g_outer_QA, _g_inner_QQ)
                        sigma_QP, _ = _sigma_max_power_iter_nonsym(_g_inner_PP, _g_inner_QQ)
                        sigma_BP = sigma_BP.squeeze(0) if sigma_BP.dim() else sigma_BP
                        sigma_QA = sigma_QA.squeeze(0) if sigma_QA.dim() else sigma_QA
                        sigma_QP = sigma_QP.squeeze(0) if sigma_QP.dim() else sigma_QP
                        a_dir = sigma_BP + sigma_QA
                        b_dir = sigma_QP
                        if float(b_dir) > 1e-30:
                            lam = (-a_dir + torch.sqrt(
                                a_dir * a_dir + 4.0 * b_dir * lr)) / (2.0 * b_dir)
                        else:
                            lam = lr / a_dir.clamp_min(1e-30)
                        dA = -lam * P_dir
                        dB = -self.lora_plus_multiplier * lam * Q_dir
                        gA_norm = op_geoA
                        gB_norm = op_geoB
                        uA_norm = lam.detach() if lam.dim() > 0 else lam
                        uB_norm = lam.detach() if lam.dim() > 0 else lam
                    else:
                        # spectral_chord / spectral_chord_tight: dA = -ρ · P
                        dA = -rho * geo_A / op_geoA
                        dB = -self.lora_plus_multiplier * rho * geo_B / op_geoB
                        gA_norm = op_geoA
                        gB_norm = op_geoB
                        uA_norm = rho.detach() if rho.dim() > 0 else rho
                        uB_norm = rho.detach() if rho.dim() > 0 else rho
                if self.end_rms_align and self.magnitude_rule == "adam_frobenius":
                    # Override the pipeline's RMS-align: rescale to the
                    # ORIGINAL Adam-direction norm rather than ‖u_A_eff‖.
                    # Re-expose uA_norm / gA_norm / rms_scale_A consistently
                    # so the diagnostics block below still reflects what
                    # was actually applied.
                    # GATED on adam_frobenius: under chord-magnitude rules
                    # (spectral_chord / spectral_chord_tight / chord_direction)
                    # the dA/dB above already carry the chord rescale; running
                    # end_rms_align would silently OVERWRITE that with a
                    # frob-magnitude step. The batched path is structurally
                    # not eligible for the (k>1, end_rms_align=True) combo,
                    # so this guard only affects the per-pair path.
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

            if self.log_basic_diagnostics:
                rec = self._emit_basic_diagnostics(
                    state=state, A=A, B=B, A_f=A_f, B_f=B_f,
                    u_A=u_A, u_B=u_B, dA=dA, dB=dB, gA=gA, gB=gB,
                    SA_half_inv=SA_half_inv, SB_half_inv=SB_half_inv,
                    geo_A=geo_A, geo_B=geo_B,
                    sigma_A_t=sigma_A_t, sigma_B_t=sigma_B_t,
                    op_geoA=op_geoA, op_geoB=op_geoB,
                    uA_norm=uA_norm, uB_norm=uB_norm,
                    gA_norm=gA_norm, gB_norm=gB_norm,
                    picard_coeff_t=picard_coeff_t,
                    rho=rho, s_AB=s_AB, lr=lr,
                )
                diag_records.append(rec)

            with maybe_time(timer, "apply"):
                A.add_(dA.to(dtype=A.dtype, device=A.device))
                B.add_(dB.to(dtype=B.dtype, device=B.device))
                A.grad.zero_()
                B.grad.zero_()

        if self.log_basic_diagnostics and diag_records:
            step_count = self.pair_state[0]['step']
            if step_count % self.diagnostics_every == 0:
                _emit_optim_diagnostics(step_count, diag_records)

    @torch.no_grad()
    @torch.no_grad()
    def _emit_basic_diagnostics(
        self, *, state, A, B, A_f, B_f, u_A, u_B, dA, dB, gA, gB,
        SA_half_inv, SB_half_inv, geo_A, geo_B,
        sigma_A_t, sigma_B_t, op_geoA, op_geoB,
        uA_norm, uB_norm, gA_norm, gB_norm, picard_coeff_t,
        rho, s_AB, lr,
        ssc_c_A=None, ssc_c_B=None,
    ):
        """Basic-tier diagnostics (default ON, ~2% wall). Returns the
        per-pair ``rec`` dict the caller appends to ``diag_records``.

        Pure function over the supplied per-pair tensors (mutates only the
        local ``rec`` it constructs and returns) so both the per-pair and
        batched step paths can call it on the same inputs to produce
        bit-identical diagnostic records. Calls into the heavy-tier
        helpers (``_emit_heavy_factor_accuracy_diag``,
        ``_emit_heavy_chord_slack_diag``, ``_emit_heavy_picard_diagnostics``)
        when their gates fire.
        """
        step_count_local = state['step']
        is_probe_step = (step_count_local % self.diagnostics_every == 0)
        # cos(applied_step, plain-AdamW-direction). See AdamScaledLoRAPost
        # for sign-convention rationale.
        sa_min, sa_max = _gram_eig_extremes_from_factor(A)
        sb_min, sb_max = _gram_eig_extremes_from_factor(B)
        rec = {
            "cos_A": _frob_cos(dA, -u_A),
            "cos_B": _frob_cos(dB, -u_B),
            **({"ssc_c_A": float(ssc_c_A)} if ssc_c_A is not None else {}),
            **({"ssc_c_B": float(ssc_c_B)} if ssc_c_B is not None else {}),
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
        # σ_max optimizer-vs-exact probe. ``sigma_{A,B}_t`` is the optimizer's
        # internal warm-start power-iter estimate that feeds the chord-tight ρ
        # (s_AB = σ_A_opt + σ_B_opt). Compare against the exact value via the
        # already-computed Gram eigenvalue. A persistent ~0 in `_optim` while
        # `_exact` is nonzero is the sticky-zero warm-start failure that froze
        # A in chord-tight whiten + Init[A] runs prior to the
        # `_sigma_max_power_iter_batched` v_init degeneracy guard.
        if sigma_A_t is not None and sa_max == sa_max:
            import math as _m
            sigma_A_exact = float(_m.sqrt(max(sa_max, 0.0)))
            sigma_A_opt = float(sigma_A_t)
            rec["sigma_A_opt"] = sigma_A_opt
            rec["sigma_A_exact"] = sigma_A_exact
            rec["sigma_A_relerr"] = (
                abs(sigma_A_opt - sigma_A_exact) / max(sigma_A_exact, 1e-30)
            )
        if sigma_B_t is not None and sb_max == sb_max:
            import math as _m
            sigma_B_exact = float(_m.sqrt(max(sb_max, 0.0)))
            sigma_B_opt = float(sigma_B_t)
            rec["sigma_B_opt"] = sigma_B_opt
            rec["sigma_B_exact"] = sigma_B_exact
            rec["sigma_B_relerr"] = (
                abs(sigma_B_opt - sigma_B_exact) / max(sigma_B_exact, 1e-30)
            )

        # Wasted-update probe: fraction of dA / dB that flows through to the
        # LoRA forward product dW = B·A. ‖B·dA‖_F / ‖dA‖_F low ⇒ A's update
        # is concentrated in directions of low σ(B) where it doesn't move the
        # loss (B's near-null modes absorb the update). Mirror probe for dB.
        # Specifically diagnostic for chord-tight whiten at high r with
        # rank-deficient B — measures the SB^{-1/2}-induced bias toward
        # B's small-σ directions which carries little loss gradient.
        with torch.no_grad():
            dA_f = dA.detach().to(torch.float32)
            dB_f = dB.detach().to(torch.float32)
            B_f_d = B.detach().to(torch.float32)
            A_f_d = A.detach().to(torch.float32)
            B_dA = B_f_d @ dA_f                      # (d_out, d_in)
            dB_A = dB_f @ A_f_d                      # (d_out, d_in)
            dA_norm = dA_f.norm() + 1e-30
            dB_norm = dB_f.norm() + 1e-30
            rec["frac_dA_through_B"] = float(B_dA.norm() / dA_norm)
            rec["frac_dB_through_A"] = float(dB_A.norm() / dB_norm)

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
        # magnitude of the perturbation that iter-2 would inject,
        # using the COEFFICIENT THAT IS ACTUALLY APPLIED IN THE
        # PICARD LOOP and the post-unit-polar-normalization u_A so
        # the ratio reflects what the polar map sees:
        #     γ_A = ‖picard_coeff · Bᵀ dB A‖_F / ‖ū_A‖_F
        #     γ_B = ‖picard_coeff · B  dA Aᵀ‖_F / ‖ū_B‖_F
        # For chord-tight family, picard_coeff = 2/(ρ·s) (computed
        # above as picard_coeff_t). For other variants, picard_coeff
        # = picard_alpha/lr (the legacy scaling). Note u_A here has
        # already been replaced by ū_A = u_A / σ_XA above when
        # unit-polar normalization is active, so the denominator
        # correctly reflects polar-input magnitude.
        coeff_t = (picard_coeff_t
                   if picard_coeff_t is not None
                   else (self.picard_alpha / lr))
        cross_A = coeff_t * (B_f.T @ dB.float() @ A_f)
        cross_B = coeff_t * (B_f @ dA.float() @ A_f.T)
        rec["gamma_A"] = float(cross_A.norm() / (u_A.norm() + 1e-30))
        rec["gamma_B"] = float(cross_B.norm() / (u_B.norm() + 1e-30))

        # Factor diagnostics (balance / stable-rank / nrank / σ_max) — shared
        # across all LoRA optimizers; pure function of (A, B). See
        # lora_playground.optim_diagnostics.factor_diagnostics.
        rec.update(factor_diagnostics(A_f, B_f))

        try:
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
            rec.setdefault("balance_resid", float("nan"))

        # AWC — Adam-direction compatibility diagnostic. Raw LoRA gradients
        # from one dense gradient G obey (g_A A^T) = (B^T g_B). The matrices
        # below apply the same core-coordinate check to Adam directions:
        # S_A_aw = SB^{-1/2} (u_A A^T) SA^{-1/2},
        # S_B_aw = SB^{-1/2} (B^T u_B) SA^{-1/2}. Their mismatch says the
        # independently Adam-preconditioned A/B directions are incompatible
        # with any single dense-gradient core. It does NOT by itself prove
        # noise, usefulness, or which polar/SSC spectrum should be preferred;
        # treat these fields as exploratory diagnostics only.
        # Cost: 2 r×r matmuls + norms per pair per step.
        H_A = u_A @ A_f.T                                   # (r, r)
        H_B = B_f.T @ u_B                                   # (r, r)
        S_A_aw = SB_half_inv @ H_A @ SA_half_inv            # (r, r)
        S_B_aw = SB_half_inv @ H_B @ SA_half_inv            # (r, r)
        fA = float(S_A_aw.norm())
        fB = float(S_B_aw.norm())
        fdiff = float((S_A_aw - S_B_aw).norm())
        fsum = float((S_A_aw + S_B_aw).norm())
        fdot = float((S_A_aw * S_B_aw).sum())
        rec["awc_SA_frob"] = fA
        rec["awc_SB_frob"] = fB
        rec["awc_diff_frob"] = fdiff
        rec["awc_sum_frob"] = fsum
        rec["awc_cos_AB"] = fdot / (fA * fB + 1e-30)
        # q_agree = E+/(E++E-) with E+ = ||S+||²/4, E- = ||S-||²/4.
        e_plus = (fsum ** 2) / 4.0
        e_minus = (fdiff ** 2) / 4.0
        rec["awc_q_agree"] = e_plus / (e_plus + e_minus + 1e-30)
        # Exploratory normalized disagreement scalar:
        #   λ_core = ||S_A - S_B||² / (2 (||S_A||² + ||S_B||²) + ε)
        # In [0, 1]: 0 when S_A ≡ S_B; 1 when S_A ≡ -S_B. This is a
        # compatibility residual, not a validated coupling policy.
        rec["awc_lambda_core"] = (fdiff ** 2) / (2.0 * (fA ** 2 + fB ** 2) + 1e-30)

        # Higham accuracy + S conditioning probe. Tracks the
        # iterative S^{-1/2} solver's quality vs an eigh reference.
        # higham_iters=10 (default) is safe for well-to-moderate-
        # conditioned S but degrades sharply on ill-conditioned S
        # (synthetic bench at σ_min/σ_max ≈ 0.01 gave 52% rel error
        # at iters=10 — needs iters=20 for ~1e-5 error). This probe
        # tells us whether REAL A·A^T spectra during training ever
        # enter the "ill" regime, vs only synthetic stress tests.
        # ~6 r×r ops per pair per probe step (cheap).
        # cond(S_A), cond(S_B) — CHEAP (r×r eigvalsh): kept in basic tier.
        try:
            SA_eigs = torch.linalg.eigvalsh(A_f @ A_f.T).clamp_min(0.0)
            SB_eigs = torch.linalg.eigvalsh(B_f.T @ B_f).clamp_min(0.0)
            SA_max = float(SA_eigs.max()); SA_min = float(SA_eigs.min())
            SB_max = float(SB_eigs.max()); SB_min = float(SB_eigs.min())
            rec["cond_SA"] = SA_max / max(SA_min, 1e-30)
            rec["cond_SB"] = SB_max / max(SB_min, 1e-30)
        except Exception:
            for k in ("cond_SA", "cond_SB"):
                rec[k] = float("nan")
        # Heavy factor-accuracy diagnostics (higham + power-iter probes).
        if self.log_heavy_diagnostics:
            self._emit_heavy_factor_accuracy_diag(
                rec, A_f=A_f, B_f=B_f, geo_A=geo_A, geo_B=geo_B,
                sigma_A_t=sigma_A_t, sigma_B_t=sigma_B_t,
                op_geoA=op_geoA, op_geoB=op_geoB,
            )

        # Stage-0 chord-tight diagnostics (plan
        # there-are-a-few-indexed-hickey).
        #
        # Probe D — Adam-preconditioning gauge residual.
        # E := u_A A^T − B^T u_B (r × r). For raw factor gradients
        # g_A = B^T G, g_B = G A^T (G = ∂L/∂W), so g_A A^T = B^T G A^T
        # = B^T g_B identically — E_raw ≡ 0. Adam preconditioning
        # breaks the identity; ‖E‖_F measures the geometric distortion
        # induced by the per-coordinate v_t scaling. Generic
        # diagnostic, useful across the broader optimizer family.
        E_gauge = H_A - H_B                                  # reuse awc H_*
        fE_gauge = float(E_gauge.norm())
        rec["adam_gauge_residual_frob"] = fE_gauge
        rec["adam_gauge_residual_rel"] = fE_gauge / max(fA, fB, 1e-30)

        # Probes A, B, C-tight only meaningful under spectral_chord_tight
        # (rho is the tight-chord scalar; op_geoA/op_geoB are the
        # operator norms ‖D_A‖_2, ‖D_B‖_2 of the polar directions).
        if self.magnitude_rule in ("spectral_chord_tight",
                                   "spectral_chord_direction"):
            lr_f = float(lr)
            rho_f = float(rho.detach()) if torch.is_tensor(rho) else float(rho)

            # Probe A — chord_slack = ‖ΔW‖₂ / lr. Compute it with plain
            # power iteration on the low-rank chord factors, not by forming
            # the dense d_out × d_in matrix.
            try:
                sigma_chord = _chord_update_opnorm_power_iter(
                    A_f, B_f, dA, dB, n_iters=8,
                )
                rec["chord_slack"] = float(sigma_chord / max(lr_f, 1e-30))
            except Exception:
                rec["chord_slack"] = float("nan")

            # Optional direct-SVD cross-check. This is intentionally kept in
            # the heavy tier and logged under a separate key so the cheap
            # power-iteration diagnostic is always the canonical field.
            if self.log_heavy_diagnostics:
                self._emit_heavy_chord_slack_diag(
                    rec, A_f=A_f, B_f=B_f, dA=dA, dB=dB, lr_f=lr_f,
                )

            # Probe B — direction-aware radius gain.
            # P = D_A / ‖D_A‖_2, Q = D_B / ‖D_B‖_2 are unit-norm
            # factor directions. With dA = -λ P, dB = -λ Q the safe
            # direction-aware bound is
            #   ‖ΔW‖_2 ≤ λ (‖B P‖_2 + ‖Q A‖_2) + λ² ‖Q P‖_2 = a λ + b λ².
            # Setting that to lr and solving for λ gives the tighter
            # (still safe) radius λ_dir vs the worst-case rho.
            try:
                op_A_safe = op_geoA + 1e-30
                op_B_safe = op_geoB + 1e-30
                P_dir = (geo_A.float() / op_A_safe).detach()         # (r, n)
                Q_dir = (geo_B.float() / op_B_safe).detach()         # (m, r)
                GBB = B_f.T @ B_f                                    # (r, r) sym PSD
                GAA = A_f @ A_f.T                                    # (r, r) sym PSD
                PPt = P_dir @ P_dir.T                                # (r, r) sym PSD
                QtQ = Q_dir.T @ Q_dir                                # (r, r) sym PSD
                # σ²_max(BP) = λ_max(L_PPt^T · GBB · L_PPt), L_PPt =
                # chol(PPt). Eigvalsh on the SYMMETRIC reduced form
                # (formerly used `eigvals` on the non-symmetric
                # PPt @ GBB which over-estimated; same issue as the
                # chord-slack probe).
                def _sigma_max_via_chol_eigh(G_outer, G_inner):
                    diag_load = 1e-12 * G_inner.diagonal().abs().mean().clamp_min(1e-30)
                    damped = G_inner + diag_load * torch.eye(
                        G_inner.shape[-1], dtype=G_inner.dtype,
                        device=G_inner.device)
                    Lc = torch.linalg.cholesky(damped)
                    M = Lc.T @ G_outer @ Lc
                    return float(torch.linalg.eigvalsh(M).clamp_min(0.0).max().sqrt())
                sigma_BP = _sigma_max_via_chol_eigh(GBB, PPt)
                sigma_QA = _sigma_max_via_chol_eigh(GAA, QtQ)
                sigma_QP = _sigma_max_via_chol_eigh(PPt, QtQ)
            except torch._C._LinAlgError:
                sigma_BP = sigma_QA = sigma_QP = float("nan")
            a_dir = sigma_BP + sigma_QA
            b_dir = sigma_QP
            if a_dir != a_dir:  # NaN propagation
                lambda_dir = float("nan")
            elif b_dir <= 1e-30:
                lambda_dir = lr_f / max(a_dir, 1e-30)
            else:
                lambda_dir = (-a_dir + (a_dir * a_dir + 4.0 * b_dir * lr_f) ** 0.5) / (2.0 * b_dir)
            s_AB_f = float(s_AB) if torch.is_tensor(s_AB) else float(s_AB)
            rec["lambda_dir"] = lambda_dir
            rec["lambda_dir_gain"] = lambda_dir / max(rho_f, 1e-30)
            rec["dir_a"] = a_dir
            rec["dir_b"] = b_dir
            rec["dir_a_over_s"] = a_dir / max(s_AB_f, 1e-30)

            # Probe C-tight — saturation fraction and polar-vs-clip
            # cosine in WHITENED singular-value space at the
            # tight-chord threshold τ_A = rho / ‖D_A‖_2, the §8
            # saturating-regime threshold from
            # algorithm_tight_chord.md. Distinct from the existing
            # cos_polar_clip_A which uses the R-equal threshold
            # τ_R = ‖X_unc‖_F / √r. Polar maps every nonzero σ → τ,
            # clip maps σ → min(η σ, τ); they share singular vectors
            # so the cosine is a singular-value-only comparison.
            try:
                c_A_mat = (SB_half_inv @ u_A).float()
                c_B_mat = (u_B @ SA_half_inv).float()
                sv_cA_tight = torch.linalg.svdvals(c_A_mat)
                sv_cB_tight = torch.linalg.svdvals(c_B_mat)
                tau_A_tight = rho_f / max(float(op_geoA), 1e-30)
                tau_B_tight = rho_f / max(float(op_geoB), 1e-30)
                eta_sA = lr_f * sv_cA_tight
                eta_sB = lr_f * sv_cB_tight
                clip_A_sv = torch.clamp(eta_sA, max=tau_A_tight)
                clip_B_sv = torch.clamp(eta_sB, max=tau_B_tight)
                # Polar vector (in σ-space): all entries τ, length √r·τ.
                n_cA = clip_A_sv.numel()
                n_cB = clip_B_sv.numel()
                polar_norm_A = (n_cA ** 0.5) * tau_A_tight
                polar_norm_B = (n_cB ** 0.5) * tau_B_tight
                clip_norm_A = float(clip_A_sv.norm()) + 1e-30
                clip_norm_B = float(clip_B_sv.norm()) + 1e-30
                rec["cos_polar_clip_tight_A"] = (
                    tau_A_tight * float(clip_A_sv.sum())
                    / (max(polar_norm_A, 1e-30) * clip_norm_A)
                )
                rec["cos_polar_clip_tight_B"] = (
                    tau_B_tight * float(clip_B_sv.sum())
                    / (max(polar_norm_B, 1e-30) * clip_norm_B)
                )
                rec["sat_frac_tight_A"] = float((eta_sA >= tau_A_tight).float().mean())
                rec["sat_frac_tight_B"] = float((eta_sB >= tau_B_tight).float().mean())
                rec["tau_tight_A"] = tau_A_tight
                rec["tau_tight_B"] = tau_B_tight
            except torch._C._LinAlgError:
                for _k in ("cos_polar_clip_tight_A", "cos_polar_clip_tight_B",
                           "sat_frac_tight_A", "sat_frac_tight_B",
                           "tau_tight_A", "tau_tight_B"):
                    rec[_k] = float("nan")

        # H2/H3 — Picard contraction + polar sensitivity probes (heavy).
        if is_probe_step and self.log_heavy_diagnostics:
            self._emit_heavy_picard_diagnostics(
                rec, u_A=u_A, u_B=u_B,
                SA_half_inv=SA_half_inv, SB_half_inv=SB_half_inv, lr=lr,
                A_f=A_f, B_f=B_f, dA=dA, dB=dB, gA=gA, gB=gB,
            )
        return rec

    def _emit_heavy_picard_diagnostics(
        self, rec, *, u_A, u_B, SA_half_inv, SB_half_inv, lr,
        A_f, B_f, dA, dB, gA, gB,
    ):
        """H2/H3 — Picard contraction + polar sensitivity probes.

        ~3 extra _polar_pipeline calls per pair per probe step → ~30 s/probe
        at r=64. Always runs 3 iters from zero so uncoupled/coupled compare
        symmetrically; independent of self.picard_iters. Mutates ``rec``.
        """
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
        rec["local_score_k3"] = _local_score(dA3, dB3)
        rec["local_score_k2_minus_k1"] = (
            rec["local_score_k2"] - rec["local_score_k1"]
        )
        rec["local_score_k3_minus_k1"] = (
            rec["local_score_k3"] - rec["local_score_k1"]
        )

        # Compatibility diagnostic over Picard iterates. Decompose each
        # iterate's (dA, dB) into the r x r components visible through A and B:
        #   C_A^(k) = dA^(k) A^T G_A^{-1}     (r × r)
        #   C_B^(k) = G_B^{-1} B^T dB^(k)     (r × r)
        # Then sum/diff modes:
        #   X_C^(k) = C_A + C_B
        #   X_D^(k) = C_A - C_B
        # These fields track magnitude evolution and direction preservation
        # across k = 1, 2, 3. They do not by themselves prove de-duplication,
        # signal quality, or a preferred NS/SSC spectrum.
        GA_inv = SA_half_inv @ SA_half_inv     # (r, r)
        GB_inv = SB_half_inv @ SB_half_inv     # (r, r)
        def _core_drives(dA_var, dB_var):
            dA32 = dA_var.float(); dB32 = dB_var.float()
            C_A = (dA32 @ A_f.T) @ GA_inv      # (r, r)
            C_B = GB_inv @ (B_f.T @ dB32)      # (r, r)
            return C_A + C_B, C_A - C_B
        XC1, XD1 = _core_drives(dA1, dB1)
        XC2, XD2 = _core_drives(dA2, dB2)
        XC3, XD3 = _core_drives(dA3, dB3)
        # Joint tangent norm at iter 3 (extends existing iter1/iter2)
        J3_full = B_f @ dA3.float() + dB3.float() @ A_f
        rec["frob_J_iter3"] = float(J3_full.norm())
        # Magnitudes
        nC1 = float(XC1.norm()) + 1e-30
        nC2 = float(XC2.norm()) + 1e-30
        nC3 = float(XC3.norm()) + 1e-30
        nD1 = float(XD1.norm()) + 1e-30
        nD2 = float(XD2.norm()) + 1e-30
        nD3 = float(XD3.norm()) + 1e-30
        rec["XC_iter1_frob"] = nC1
        rec["XC_iter2_frob"] = nC2
        rec["XC_iter3_frob"] = nC3
        rec["XD_iter1_frob"] = nD1
        rec["XD_iter2_frob"] = nD2
        rec["XD_iter3_frob"] = nD3
        # Shrink ratios (1 - ratio): positive ⇒ Picard shrunk it
        rec["XC_shrink_2v1"] = 1.0 - nC2 / nC1
        rec["XC_shrink_3v1"] = 1.0 - nC3 / nC1
        rec["XD_shrink_2v1"] = 1.0 - nD2 / nD1
        rec["XD_shrink_3v1"] = 1.0 - nD3 / nD1
        # Direction preservation
        rec["XC_cos_2v1"] = float((XC2 * XC1).sum()) / (nC1 * nC2)
        rec["XC_cos_3v1"] = float((XC3 * XC1).sum()) / (nC1 * nC3)
        rec["XD_cos_2v1"] = float((XD2 * XD1).sum()) / (nD1 * nD2)
        rec["XD_cos_3v1"] = float((XD3 * XD1).sum()) / (nD1 * nD3)
        # Energy split: what fraction of the total core energy is
        # in the agreed (C) vs ownership (D) mode at each iterate?
        rec["XC_frac_iter1"] = (nC1 ** 2) / (nC1 ** 2 + nD1 ** 2)
        rec["XC_frac_iter2"] = (nC2 ** 2) / (nC2 ** 2 + nD2 ** 2)
        rec["XC_frac_iter3"] = (nC3 ** 2) / (nC3 ** 2 + nD3 ** 2)

        # k=1→k=3 dense update difference: what fraction lives in
        # the rank-r shared-core subspace (i.e. is attributable to
        # ΔX_C = X_C^(3) - X_C^(1))?
        # ΔJ_total = B (dA3 - dA1) + (dB3 - dB1) A    (dense)
        # ΔJ_core  = B ΔX_C A                          (rank ≤ r)
        # Use trace identity ||B X A||² = tr(G_B X G_A X^T) for the
        # core norm; compute ΔJ_total dense (cheap matmul).
        GA_full = A_f @ A_f.T               # (r, r)
        GB_full = B_f.T @ B_f               # (r, r)
        dXC = XC3 - XC1                     # (r, r)
        sq_core_dJ = float(((GB_full @ dXC @ GA_full) * dXC).sum())
        dJ_full = B_f @ (dA3.float() - dA1.float()) + (dB3.float() - dB1.float()) @ A_f
        sq_dJ = float(dJ_full.pow(2).sum())
        rec["dJ_3v1_total_frob"] = sq_dJ ** 0.5
        rec["dJ_3v1_core_frob"] = max(sq_core_dJ, 0.0) ** 0.5
        rec["dJ_3v1_core_frac"] = max(sq_core_dJ, 0.0) / (sq_dJ + 1e-30)

        # Base-rate diagnostic: what fraction of J^(k) itself
        # (not just the k-difference) lives in the rank-r shared-
        # core subspace P_U J P_V where P_U = projector onto
        # col(B), P_V = projector onto row(A)? Compares against
        # dJ_3v1_core_frac to determine whether Picard's action is
        # specifically targeting the core (R_I = dJ_core_frac /
        # J_core_frac >> 1) or just inheriting the base rate
        # (R_I ≈ 1 means everything is in the core anyway).
        # Trace identity: ||P_U J P_V||² ≈ tr(M G_A^{-1} M^T G_B^{-1})
        # where M = B^T J A^T = G_B dA A^T + B^T dB G_A (r×r).
        def _core_frac(dA_var, dB_var):
            dA32 = dA_var.float(); dB32 = dB_var.float()
            M = GB_full @ (dA32 @ A_f.T) + (B_f.T @ dB32) @ GA_full
            sq_core = float(((M @ GA_inv) * (GB_inv @ M)).sum())
            Jt = B_f @ dA32 + dB32 @ A_f
            sq_full = float(Jt.pow(2).sum())
            return max(sq_core, 0.0) / (sq_full + 1e-30)
        rec["J_core_frac_iter1"] = _core_frac(dA1, dB1)
        rec["J_core_frac_iter2"] = _core_frac(dA2, dB2)
        rec["J_core_frac_iter3"] = _core_frac(dA3, dB3)

        # Scalar gain fit. Tests whether Picard's effect on X_C
        # and X_D is literally a scalar attenuation / amplification:
        #   X_C^(3) ≈ a_C · X_C^(1),   X_D^(3) ≈ a_D · X_D^(1).
        # Residuals r_C, r_D measure how well the scalar fit holds.
        aC_3v1 = float((XC3 * XC1).sum()) / (nC1 ** 2)
        aD_3v1 = float((XD3 * XD1).sum()) / (nD1 ** 2)
        rec["aC_3v1"] = aC_3v1
        rec["aD_3v1"] = aD_3v1
        rec["rC_3v1"] = float((XC3 - aC_3v1 * XC1).norm()) / nC3
        rec["rD_3v1"] = float((XD3 - aD_3v1 * XD1).norm()) / nD3

        # Hidden-motion ratio: how much core energy lives in the
        # first-order-invisible ownership mode X_D vs the
        # identifiable mode X_C, at each iterate.
        rec["h_iter1"] = (nD1 ** 2) / (nC1 ** 2)
        rec["h_iter2"] = (nD2 ** 2) / (nC2 ** 2)
        rec["h_iter3"] = (nD3 ** 2) / (nC3 ** 2)

        # Second-order pollution: finite-update second-order term
        # ΔB · ΔA (which is invisible to first-order J but real in
        # the actual factor update) relative to J magnitude.
        # Predicts: q grows more under Picard at low rank if X_D
        # amplification feeds a larger second-order interaction.
        def _q(dA_var, dB_var, J_norm):
            so = dB_var.float() @ dA_var.float()
            return float(so.norm()) / (J_norm + 1e-30)
        rec["q_iter1"] = _q(dA1, dB1, rec["frob_J_iter1"])
        rec["q_iter2"] = _q(dA2, dB2, rec["frob_J_iter2"])
        rec["q_iter3"] = _q(dA3, dB3, rec["frob_J_iter3"])

        # Gram drift: relative change in (A+ΔA)(A+ΔA)^T vs AA^T,
        # per iterate. Tests whether X_D growth at low rank feeds
        # downstream preconditioner instability.
        nGA = float(GA_full.norm()) + 1e-30
        nGB = float(GB_full.norm()) + 1e-30
        def _gram_drift(dA_var, dB_var):
            dA32 = dA_var.float(); dB32 = dB_var.float()
            A_new = A_f + dA32
            B_new = B_f + dB32
            gA = float((A_new @ A_new.T - GA_full).norm()) / nGA
            gB = float((B_new.T @ B_new - GB_full).norm()) / nGB
            return gA, gB
        gA1, gB1 = _gram_drift(dA1, dB1)
        gA2, gB2 = _gram_drift(dA2, dB2)
        gA3, gB3 = _gram_drift(dA3, dB3)
        rec["gA_iter1"] = gA1
        rec["gA_iter2"] = gA2
        rec["gA_iter3"] = gA3
        rec["gB_iter1"] = gB1
        rec["gB_iter2"] = gB2
        rec["gB_iter3"] = gB3


    @torch.no_grad()
    def _emit_heavy_factor_accuracy_diag(
        self, rec, *, A_f, B_f, geo_A, geo_B,
        sigma_A_t, sigma_B_t, op_geoA, op_geoB,
    ):
        """Heavy probe-step diagnostics: higham SPD-inverse accuracy vs eigh,
        and cold-start power-iter σ_max accuracy vs eigh reference. Mutates
        ``rec``. Each block has its own ``if self.log_heavy_diagnostics``
        gate (the powiter block additionally checks ``self.magnitude_rule``).
        """
        # higham accuracy — HEAVY (double SPD inversion via eigh + matmuls):
        # gated on log_heavy_diagnostics. ~5-30s per probe step at r=64
        # (one eigh inversion per LoRA pair × hundreds of pairs).
        if self.log_heavy_diagnostics:
          try:
            SA_grm = A_f @ A_f.T
            SB_grm = B_f.T @ B_f
            eyeA = torch.eye(SA_grm.shape[0], dtype=SA_grm.dtype, device=SA_grm.device)
            eyeB = torch.eye(SB_grm.shape[0], dtype=SB_grm.dtype, device=SB_grm.device)
            SA_inv_eigh = _spd_inv_half(SA_grm, eps=self.delta, method="eigh")
            SB_inv_eigh = _spd_inv_half(SB_grm, eps=self.delta, method="eigh")
            SA_inv_h = _spd_inv_half(
                SA_grm, eps=self.delta, method="higham",
                higham_iters=self.higham_iters)
            SB_inv_h = _spd_inv_half(
                SB_grm, eps=self.delta, method="higham",
                higham_iters=self.higham_iters)
            rec["higham_SA_rel_err_F"] = float(
                (SA_inv_h - SA_inv_eigh).norm() / (SA_inv_eigh.norm() + 1e-30))
            rec["higham_SB_rel_err_F"] = float(
                (SB_inv_h - SB_inv_eigh).norm() / (SB_inv_eigh.norm() + 1e-30))
            SA_resid = SA_inv_h @ (SA_grm + self.delta * eyeA) @ SA_inv_h - eyeA
            SB_resid = SB_inv_h @ (SB_grm + self.delta * eyeB) @ SB_inv_h - eyeB
            rec["higham_SA_residual_F"] = float(SA_resid.norm())
            rec["higham_SB_residual_F"] = float(SB_resid.norm())
          except Exception:
            for k in ("higham_SA_rel_err_F", "higham_SB_rel_err_F",
                      "higham_SA_residual_F", "higham_SB_residual_F"):
                rec[k] = float("nan")

        # Power-iter accuracy probe — methodology lesson from
        # 2026-05-11 (~/.claude/CLAUDE.md "Suspect the probe before
        # the theorem"): any iterative-numerical-method default in
        # an optimizer hot path should ship with a direct-accuracy
        # probe option that compares against an exact reference.
        # Logs cold-start power-iter σ_max estimates at n_iters ∈
        # {3, 8} for (A, B, geo_A, geo_B) as ratios to the exact
        # eigh σ_max already computed by the chord-tight rule.
        # ratio < 1 ⇒ power-iter under-estimates (the failure mode
        # we cared about); fires every probe step. Cost: 8 small
        # _sigma_max_power_iter calls per pair per probe step.
        # HEAVY — gated on log_heavy_diagnostics. 8 power-iter calls
        # per LoRA pair per probe step → ~20s/probe step at r=64.
        if self.log_heavy_diagnostics and self.magnitude_rule in (
                "spectral_chord", "spectral_chord_tight",
                "spectral_chord_direction"):
            try:
                for name, mat, sigma_eigh in [
                    ("A", A_f, sigma_A_t),
                    ("B", B_f, sigma_B_t),
                    ("geoA", geo_A.float(), op_geoA - 1e-30),
                    ("geoB", geo_B.float(), op_geoB - 1e-30),
                ]:
                    ref = float(sigma_eigh) + 1e-30
                    sig3, _ = _sigma_max_power_iter(mat, n_iters=3)
                    sig8, _ = _sigma_max_power_iter(mat, n_iters=8)
                    rec[f"powiter_ratio_{name}_n3"] = float(sig3) / ref
                    rec[f"powiter_ratio_{name}_n8"] = float(sig8) / ref
            except Exception:
                for name in ("A", "B", "geoA", "geoB"):
                    rec[f"powiter_ratio_{name}_n3"] = float("nan")
                    rec[f"powiter_ratio_{name}_n8"] = float("nan")


    @torch.no_grad()
    def _emit_heavy_chord_slack_diag(self, rec, *, A_f, B_f, dA, dB, lr_f):
        """Direct-SVD cross-check for chord_slack.

        Gated on log_heavy_diagnostics. Only meaningful under
        spectral_chord_tight / spectral_chord_direction (caller enforces).
        Mutates ``rec`` by adding ``chord_slack_svd_direct`` and, when the
        cheap power-iteration estimate is present, ``chord_slack_svd_relerr``.
        """
        # Probe A — chord slack u_chord = ‖B dA + dB A + dB dA‖_2 / lr.
        # ΔW = (B+dB)(A+dA) − BA. Direct SVD on the materialized
        # chord matrix (m × n). For non-lm_head pairs m,n ≤ 4096 so
        # this is ~10ms.
        #
        # Earlier attempts at a 2r×2r shortcut were both wrong:
        # (1) `eigvals` on the non-symmetric (R R^T)(L^T L) leaked
        #     complex parts into the real-eigenvalue max.
        # (2) chol(L^T L) + eigvalsh(L_chol^T · R R^T · L_chol) is
        #     mathematically equivalent in exact arithmetic, but
        #     L = [B+dB, B] makes L^T L rank-deficient by
        #     construction (overlapping column spans as dB → 0); the
        #     damping that keeps Cholesky safe also drifts the
        #     eigenvalues upward as ‖B‖_F grows, systematically
        #     over-estimating σ_max. Confirmed against direct SVD
        #     at steps 20-100 of `v1_debug_r64_400_v3` (commit
        #     ec95622): chol+eigvalsh crossed 1 by step 80 while
        #     direct-SVD stayed at 0.94. See docs/notes/polar_product
        #     /chord_slack_probe_resolution_2026_05_11.md.
        #
        # lm_head pairs (B.shape[0] ~100k) skip the probe — chord
        # matrix materialization is too expensive there.
        # HEAVY — gated on log_heavy_diagnostics. SVD on m×n
        # materialized chord matrix per LoRA pair → ~60s/probe step.
        if self.log_heavy_diagnostics:
            try:
                dA_f = dA.float()
                dB_f = dB.float()
                if B_f.shape[0] <= 4096:
                    with torch.no_grad():
                        chord_direct = (B_f + dB_f) @ (A_f + dA_f) - B_f @ A_f
                        sigma_chord = float(
                            torch.linalg.svdvals(chord_direct).max())
                else:
                    sigma_chord = float("nan")
            except Exception:
                sigma_chord = float("nan")
            slack_direct = sigma_chord / max(lr_f, 1e-30)
            rec["chord_slack_svd_direct"] = slack_direct
            slack_power = rec.get("chord_slack")
            if slack_power is not None and slack_power == slack_power:
                rec["chord_slack_svd_relerr"] = (
                    abs(slack_direct - slack_power) / max(abs(slack_direct), 1e-30)
                )


class AdamSOAPPolarProductLoRA(AdamPolarProductLoRA):
    # Override `_adam_direction`; the batched path inlines per-coord Adam
    # and would silently bypass SOAP's eigenbasis rotation. Force per-pair.
    _BATCHED_PATH_SUPPORTED = False

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
        if self.log_basic_diagnostics and state['step'] % self.diagnostics_every == 0:
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
        if self.log_basic_diagnostics:
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
                    if _is_main_process():
                        print(json.dumps(payload, sort_keys=True), flush=True)
        return out


class AdaFactorPolarProductLoRA(AdamPolarProductLoRA):
    # Override `_adam_direction` (rank-1 Adafactor v); batched path inlines
    # standard Adam and would silently bypass it. Force per-pair.
    _BATCHED_PATH_SUPPORTED = False

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
        if self.log_basic_diagnostics and state['step'] % self.diagnostics_every == 0:
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
        if self.log_basic_diagnostics:
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
                    if _is_main_process():
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


# Opt-in: compile the chord-tight-clean polar pipeline as a single graph
# when LORA_COMPILE_KERNELS=1. The method is mostly numeric (whiten →
# pre-rescale → Picard loop → polar → unwhiten → σ_max → scale) and
# compiles fullgraph-clean (verified by
# `tests/test_chord_tight_clean.py::test_no_graph_breaks_under_compile`).
# fullgraph=False is kept here as a safety net for future edits that
# might introduce a Python-only construct; the test will fail-fast if
# that happens so the comment isn't load-bearing.
#
# LORA_COMPILE_KERNELS=2 selects `mode='reduce-overhead'` which captures
# the compiled function as a CUDA graph (1 launch per call vs N).
# Requires fullgraph=True; the body satisfies that, but external state
# mutations (warm-start dict writes, `gs[...]` updates) may force
# extra recompiles. Bench script: `scripts/bench/bench_compile_modes.sh`.
_compile_setting = os.environ.get("LORA_COMPILE_KERNELS", "0")
if _compile_setting == "1":
    AdamPolarProductLoRA._chord_tight_clean_polar_pipeline = torch.compile(
        AdamPolarProductLoRA._chord_tight_clean_polar_pipeline,
        dynamic=False, fullgraph=False,
    )
elif _compile_setting == "2":
    AdamPolarProductLoRA._chord_tight_clean_polar_pipeline = torch.compile(
        AdamPolarProductLoRA._chord_tight_clean_polar_pipeline,
        dynamic=False, fullgraph=True, mode="reduce-overhead",
    )


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
                 precond_method="eigh", higham_iters=10,
                 precond_delta_relative=False,
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.precond_method = precond_method
        self.higham_iters = int(higham_iters)
        self.precond_delta_relative = bool(precond_delta_relative)
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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

    def effective_config(self) -> dict:
        """Gauge variant uses Newton-Schulz unconditionally — no polar_method
        or polar_sigma_power knobs to short-circuit."""
        return {
            "effective_picard_iters": int(self.picard_iters),
            "effective_inner_polar": "ns",
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_basic_diagnostics else None

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
                SA_half_inv = _spd_inv_half(S_A, eps=self.delta,
                                            method=self.precond_method,
                                            higham_iters=self.higham_iters,
                                            eps_relative=self.precond_delta_relative)
                if b_norm < 1e-8:
                    SB_half_inv = (self.delta ** -0.5) * torch.eye(
                        r, dtype=torch.float32, device=A_f.device,
                    )
                else:
                    S_B = spdify(B_f.T @ B_f, self.delta)
                    SB_half_inv = _spd_inv_half(S_B, eps=self.delta,
                                                method=self.precond_method,
                                                higham_iters=self.higham_iters,
                                                eps_relative=self.precond_delta_relative)
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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 precond_method="eigh", higham_iters=10,
                 precond_delta_relative=False,
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.precond_method = precond_method
        self.higham_iters = int(higham_iters)
        self.precond_delta_relative = bool(precond_delta_relative)
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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

    def effective_config(self) -> dict:
        """ClipGauge variant uses Newton-Schulz polar unconditionally (the
        clip path is operator-level, not polar-method-level)."""
        return {
            "effective_picard_iters": int(self.picard_iters),
            "effective_inner_polar": "ns",
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_basic_diagnostics else None

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
                SA_half_inv = _spd_inv_half(S_A, eps=self.delta,
                                            method=self.precond_method,
                                            higham_iters=self.higham_iters,
                                            eps_relative=self.precond_delta_relative)
                if b_norm < 1e-8:
                    SB_half_inv = (self.delta ** -0.5) * torch.eye(
                        r, dtype=torch.float32, device=A_f.device,
                    )
                else:
                    S_B = spdify(B_f.T @ B_f, self.delta)
                    SB_half_inv = _spd_inv_half(S_B, eps=self.delta,
                                                method=self.precond_method,
                                                higham_iters=self.higham_iters,
                                                eps_relative=self.precond_delta_relative)
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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20,
                 precond_refresh_every=1,
                 precond_method="eigh", higham_iters=10,
                 precond_delta_relative=False):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
        self.diagnostics_every = diagnostics_every
        self.precond_refresh_every = precond_refresh_every
        self.precond_method = precond_method
        self.higham_iters = higham_iters
        self.precond_delta_relative = bool(precond_delta_relative)

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
        diag_records = [] if self.log_basic_diagnostics else None

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
                    eps_relative=self.precond_delta_relative,
                )
                state['SB_half_inv'] = _spd_inv_half(
                    B.float().T @ B.float(), eps=self.delta,
                    method=self.precond_method, higham_iters=self.higham_iters,
                    eps_relative=self.precond_delta_relative,
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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
        self.diagnostics_every = diagnostics_every
        self.pair_state = {i: {"step": 0} for i in range(len(pairs))}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                diag_records.append(rec)

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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

            if self.log_basic_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                rec["norm_gA"] = float(gA.norm().item())
                rec["norm_gB"] = float(gB.norm().item())
                rec["norm_uA"] = float(u_A.norm().item())
                rec["norm_uB"] = float(u_B.norm().item())
                diag_records.append(rec)

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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
                if self.log_basic_diagnostics:
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

            if self.log_basic_diagnostics:
                rec = {k: float(v) for k, v in certs.items() if isinstance(v, (int, float))}
                rec["norm_dA"] = float(dA.norm().item())
                rec["norm_dB"] = float(dB.norm().item())
                rec["transport_residual"] = transport_residual
                rec["align_mom"] = certs["LB"]
                diag_records.append(rec)

        if self.log_basic_diagnostics and diag_records:
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
                 log_basic_diagnostics=False, log_heavy_diagnostics=False, diagnostics_every=20):
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
        self.log_basic_diagnostics = bool(log_basic_diagnostics)
        self.log_heavy_diagnostics = bool(log_heavy_diagnostics)
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
        diag_records = [] if self.log_basic_diagnostics else None

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
                if self.log_basic_diagnostics:
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

            if self.log_basic_diagnostics:
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

        if self.log_basic_diagnostics and diag_records:
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
    precond_delta: float = 1e-6,
    curvature_whitening: bool = False,
    curvature_beta: float = 0.99,
    log_non_finite: bool = False,
    log_non_finite_start_step: int = 1,
    debug_optimizer_state: bool = False,
    debug_optimizer_state_every: int = 1,
    debug_optimizer_state_start_step: int = 1,
    debug_snapshot_dir: str | None = None,
    debug_snapshot_limit: int = 8,
    debug_abort_on_non_finite: bool = False,
    psi_inner_iters: int = 1,
    psi_momentum: float = 0.9,
    psi_rho: float = 0.01,
    psi_momentum_rank: int | None = None,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
    muon_ns_steps: int = 5,
    muon_alpha: int = 16,
    muon_rank: int = 16,
    log_basic_diagnostics: bool = False,
    log_heavy_diagnostics: bool = False,
    optim_diagnostics_every: int = 20,
    precond_refresh_every: int = 1,
    precond_method: str = "higham",
    higham_iters: int = 10,
    picard_alpha: float = 1.0,
    htmuon_p: float | None = None,
    picard_iters_override: int | None = None,
    cw_picard_iters: int = 1,
    cw_nesterov: bool = False,
    cw_no_radius: bool = False,
    cw_no_diag_curv: bool = False,
    cw_factor_a: float = 0.0,
    cw_factor_b: float = 0.0,
    anderson_m: int = 0,
    anderson_reg: float = 1e-10,
    soap_beta: float = 0.95,
    soap_refresh_every: int = 1,
    polar_norm_dir: str = "frob",
    polar_sigma_power: float | None = None,
    polar_method: str = "ns",
    polar_core_remix_alpha: float = 0.0,
    ssc_c: float | None = None,
    ssc_nsteps: int = 10,
    ssc_kappa: float | None = None,
    ssc_kappa_refresh_every: int = 1,
    ssc_kappa_warmup_steps: int = 5,
    ssc_kappa_solver: str = "eigvalsh",
    ssc_kappa_bisect_iters: int = 3,
    ssc_kappa_cache_share_picard: bool = False,
    ssc_kappa_cache_ema_beta: float | None = None,
    ssc_kappa_bisect_mode: str = "sequential",
    ssc_kappa_bisect_nsteps_eval: int | None = None,
    ssc_kappa_cross_group_eigvalsh: bool = True,
    ssc_kappa_diagnose_eigvalsh: bool = False,
    ssc_kappa_diagnose_start_step: int = 1,
    ssc_kappa_diag_ema_beta: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    precond_delta_relative: bool = False,
    ns_form: str = "gram",
    higham_compute_dtype: str = "fp32",
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
            betas=(beta1, beta2),
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
            betas=(beta1, beta2),
            eps=1e-8,
            weight_decay=weight_decay,
            svd_niter=svd_niter,
        )

    if optimizer_type == "adamw":
        return LoRAPlusAdamW(
            model,
            lr=lr,
            lora_plus_multiplier=lora_plus_multiplier,
            betas=(beta1, beta2),
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
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
        )
    if optimizer_type == "adam-lin-lora":
        return AdamLinLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
        )
    if optimizer_type in ("curvature-whiten-lora", "curvature-whiten-polar-lora"):
        # SOAP on momentum in the S⊗D curvature eigenbasis, followed by the
        # chord-tight outer curvature sandwich. The two names share one class
        # and differ only in the polar toggle, so keep/drop-polar is a clean
        # optimizer-name axis.
        return CurvatureWhitenLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            curvature_beta=curvature_beta,
            use_polar=(optimizer_type == "curvature-whiten-polar-lora"),
            ns_steps=muon_ns_steps,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics,
            log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            # NOTE: curvature-whiten is soap_v=True (SOAP path); cw_nesterov is only
            # defined for soap_v=False, so it is NOT wired here (would raise). The
            # config event logs the EFFECTIVE cw_nesterov, so such runs read False.
        )
    if optimizer_type in ("kl-shampoo-lora", "kl-shampoo-polar-lora"):
        # KL-Shampoo-LoRA (soap_curvature_whitening.md exp 5): same class as
        # curvature-whiten, but the curvature factors are the coupled KL
        # fixed-point estimate (kl_coupled) and the SOAP v̂ is dropped (soap_v
        # =False), leaving the closed-form Shampoo core S^{-1/2} m̂ D^{-1/2}.
        # Polar on/off is the same optimizer-name axis as the curvature-whiten
        # pair — tests whether properly-solved curvature makes the polar redundant.
        return CurvatureWhitenLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            curvature_beta=curvature_beta,
            use_polar=(optimizer_type == "kl-shampoo-polar-lora"),
            ns_steps=muon_ns_steps,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics,
            log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            kl_coupled=True,
            soap_v=False,
            cw_picard_iters=cw_picard_iters,
            cw_nesterov=cw_nesterov,
        )
    if optimizer_type in ("kl-diag-lora", "kl-diag-polar-lora"):
        # Option (b) of kl_shampoo_polar_derivation.md §"Cross-coupling": the
        # consistent single diagonal metric (P,Q)=(D_out,D_in). Same class as
        # kl-shampoo, but the dense small-side S_curv is replaced by the
        # conjugate-diagonal-weighted geometric Gram (diag_metric=True), so the
        # whole step is one self-consistent two-sided program. Polar on/off is the
        # same optimizer-name axis. Tests the fidelity-vs-consistency fork: does
        # dropping the dense latent curvature cost peak loss?
        return CurvatureWhitenLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            curvature_beta=curvature_beta,
            use_polar=(optimizer_type == "kl-diag-polar-lora"),
            ns_steps=muon_ns_steps,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics,
            log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            kl_coupled=True,
            soap_v=False,
            diag_metric=True,
            cw_picard_iters=cw_picard_iters,
            cw_nesterov=cw_nesterov,
            cw_no_radius=cw_no_radius,
            cw_no_diag_curv=cw_no_diag_curv,
            cw_factor_a=cw_factor_a,
            cw_factor_b=cw_factor_b,
        )
    if optimizer_type == "kl-diag-polar-flatout-lora":
        # Robustness probe: kl-diag-polar with the un-whiten REMOVED (flat_outer).
        # dX ∝ φ(z) is flat-spectrum (chord-tight basin) with a curvature-chosen
        # frame. Heuristic (not the curvature-metric LMO) — tests whether the
        # tuned-lr peak survives without the basin-sharpening un-whiten.
        return CurvatureWhitenLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            curvature_beta=curvature_beta,
            use_polar=True,
            ns_steps=muon_ns_steps,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics,
            log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            kl_coupled=True,
            soap_v=False,
            diag_metric=True,
            cw_picard_iters=cw_picard_iters,
            flat_outer=True,
            cw_nesterov=cw_nesterov,
        )
    if optimizer_type in ("diag-shampoo-lora", "diag-shampoo-polar-lora"):
        # Non-KL ablation of kl-diag (option b): SAME consistent diagonal metric
        # (small side M_A = Bᵀ diag(D_out) B, diag_metric=True) and closed-form
        # Shampoo whiten (soap_v=False), but the diagonals D_in/D_out are textbook
        # grad-energy EMAs instead of the KL coupled fixed point (kl_coupled=False).
        # Isolates what the KL coupling buys at fixed diagonal-metric geometry.
        return CurvatureWhitenLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            curvature_beta=curvature_beta,
            use_polar=(optimizer_type == "diag-shampoo-polar-lora"),
            ns_steps=muon_ns_steps,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics,
            log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            kl_coupled=False,
            soap_v=False,
            diag_metric=True,
            cw_picard_iters=cw_picard_iters,
            cw_nesterov=cw_nesterov,
            cw_no_radius=cw_no_radius,
            cw_no_diag_curv=cw_no_diag_curv,
            cw_factor_a=cw_factor_a,
            cw_factor_b=cw_factor_b,
        )
    if optimizer_type == "adam-lin-core-lora":
        # Cross-check: same Sylvester solver as adam-lin-lora, but Adam-EMA
        # on the core-space K matrix instead of factor preconditioned grads.
        return AdamLinCoreLoRA(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
        )
    if optimizer_type == "adam-scaled-lora-post":
        return AdamScaledLoRAPost(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-lin-lora-post":
        return AdamLinLoRAPost(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-scaled-lora-matrix":
        return AdamScaledLoRAMatrix(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
        )
    if optimizer_type == "adam-lin-lora-matrix":
        return AdamLinLoRAMatrix(
            model,
            lr=lr,
            betas=(beta1, beta2),
            delta=1e-6,
            eps=1e-8,
            scaled_metric=scaled_metric,
            lora_plus_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "polar-product-lora":
        return PolarProductLoRA(
            model, lr=lr, delta=precond_delta, ns_steps=muon_ns_steps,
        )
    if optimizer_type == "adam-polar-product-lora":
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            core_remix_alpha=polar_core_remix_alpha,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled":
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-soap-polar-product-lora":
        return AdamSOAPPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adafactor-polar-product-lora":
        return AdaFactorPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "sign-momentum-polar-product-lora":
        return SignMomentumPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-endrms":
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=2,
            end_rms_align=True,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord":
        # Substitution 1' (algorithm.md §6.1): replace Frobenius-Adam-magnitude
        # rescale with spectral-chord rule ρ = lr/(σ_A+σ_B+1). Per-block prox
        # structure (cross-coupling Picard, whitening, polar) unchanged.
        # Note: lr should be retuned (~10-30× larger than the standard 3e-4
        # since "lr" is now interpreted as a spectral-norm trust-region radius
        # on ΔW rather than a Frobenius-rate scale).
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            magnitude_rule="spectral_chord",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight":
        # Tight chord-spectral rule (algorithm.md §6.1, exact-root variant):
        # ρ = (-s + sqrt(s²+4lr))/2 where s = σ_A+σ_B. Drops Spectron's "+1"
        # slack; ρ ≈ √lr at s→0 (early training, B≈0) and ρ ≈ lr/s at s→∞.
        # Same ‖ΔW‖_op ≤ lr guarantee, with no conservative substitution.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_tight",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight-clean":
        # §10-clean Algorithm 2′ (algorithm_tight_chord.md §10). Pre-rescales
        # u_A by σ_max(X_A) like spectral_chord_tight, but uses the doc-faithful
        # cross-coupling coefficient 1/η (Lemma 1, not 2/(ρs)) and the linear
        # ρ = η/s (not the quadratic root). Differs from spectral_chord_tight
        # at k≥2 only; bit-identical at k=1 up to a <1.1% ρ-formula gap.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            htmuon_p=htmuon_p,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            ssc_c=ssc_c,
            ssc_nsteps=ssc_nsteps,
            ssc_kappa=ssc_kappa,
            ssc_kappa_refresh_every=ssc_kappa_refresh_every,
            ssc_kappa_warmup_steps=ssc_kappa_warmup_steps,
            ssc_kappa_solver=ssc_kappa_solver,
            ssc_kappa_bisect_iters=ssc_kappa_bisect_iters,
            ssc_kappa_bisect_mode=ssc_kappa_bisect_mode,
            ssc_kappa_bisect_nsteps_eval=ssc_kappa_bisect_nsteps_eval,
            ssc_kappa_cache_share_picard=ssc_kappa_cache_share_picard,
            ssc_kappa_cache_ema_beta=ssc_kappa_cache_ema_beta,
            ssc_kappa_cross_group_eigvalsh=ssc_kappa_cross_group_eigvalsh,
            ssc_kappa_diagnose_eigvalsh=ssc_kappa_diagnose_eigvalsh,
            ssc_kappa_diagnose_start_step=ssc_kappa_diagnose_start_step,
            ssc_kappa_diag_ema_beta=ssc_kappa_diag_ema_beta,
            magnitude_rule="spectral_chord_tight_clean",
            precond_delta_relative=precond_delta_relative,
            ns_form=ns_form,
            higham_compute_dtype=higham_compute_dtype,
            curvature_whitening=curvature_whitening,
            curvature_beta=curvature_beta,
            log_non_finite=log_non_finite,
            log_non_finite_start_step=log_non_finite_start_step,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_optimizer_state_start_step=debug_optimizer_state_start_step,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw":
        # §6 full-residual Frank-Wolfe variant of chord-tight-clean. Identical
        # to chord-tight-clean except the Picard inner loop retains the
        # self-terms S_B·dA and dB·S_A in each block's polar input (doc §6).
        # Bit-identical to chord-tight-clean at picard_iters=1 (self-terms
        # vanish at dA⁽⁰⁾=dB⁽⁰⁾=0); diverges at k≥2.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            htmuon_p=htmuon_p,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_tight_clean",
            precond_delta_relative=precond_delta_relative,
            ns_form=ns_form,
            higham_compute_dtype=higham_compute_dtype,
            fw_linearization="full",
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight-no-rho":
        # §8 no-ρ ablation (algorithm_tight_chord.md §8). Same direction as
        # chord-tight (whiten → polar → unwhiten) but drops the ρ-routed
        # magnitude rescale: dA = -lr·geo_A directly. Constructor enforces
        # picard_iters=1 — §8 doesn't derive a cross-coupling coefficient
        # without ρ.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_tight_no_rho",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight-exact":
        # Chord-tight magnitude rule + exact-chord direction iteration.
        # Aligns the variational direction target (currently J = B·dA + dB·A,
        # the tangent — see algorithm_tight_chord.md §3) with the magnitude
        # program's actual target (ΔW = J + dB·dA, the chord — §8). Picard
        # cross-coupling correction uses (B + dB_prev) and (A + dA_prev)
        # instead of just B and A; both magnitude budget AND direction
        # iteration consistently target ΔW. Investigates whether the
        # observed lr=5e-3 loss bump at chord-tight k=3 r=64 is a symptom
        # of the J-vs-ΔW asymmetry (bump shows where dB has grown enough
        # that J ≠ ΔW but the default iter still optimizes J).
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_tight",
            exact_chord=True,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening":
        # Whitening-importance ablation: chord-tight with S_A^{-1/2} = S_B^{-1/2} = I.
        # Equivalent to per-factor Muon (algorithm_tight_chord.md §2 program W)
        # on the Adam direction, plus the chord-tight ρ. Tests whether
        # whitening matters for training quality at all — if not, the higham
        # accuracy question is moot.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_tight",
            disable_whitening=True,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-spectral-chord-direction":
        # Variant 1 of algorithm_tight_chord.md: direction-aware ρ in place
        # of worst-case ρ. Solves a·λ + b·λ² = lr per pair per Picard iter
        # with a = ‖B·P‖_2 + ‖Q·A‖_2, b = ‖Q·P‖_2 — strictly tighter than
        # chord_tight's s·ρ + ρ² = lr when P,Q misaligned with B,A top
        # singular directions. Stage-0 diagnostics measured λ_dir_gain ≈
        # 1.3-1.4 at r=64, growing with training; expected ~5-15% loss-
        # per-step improvement vs chord_tight at 2k canonical horizon.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            picard_alpha=picard_alpha,
            anderson_m=anderson_m,
            anderson_reg=anderson_reg,
            polar_norm_dir=polar_norm_dir,
            polar_sigma_power=polar_sigma_power,
            polar_method=polar_method,
            magnitude_rule="spectral_chord_direction",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-coupled-exact-chord":
        # Variational target is the actual ΔW = (B+ΔB)(A+ΔA) - BA, not its
        # tangent J = B·ΔA + ΔB·A. Picard iterates 2..k recompute S_{B+dB},
        # S_{A+dA} per inner step. Default picard_iters=3 to match the
        # leaderboard -coupled config; chord effect appears at k≥1, so picard=1
        # would make the flag a no-op.
        return AdamPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            exact_chord=True,
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-clip-product-lora":
        # Clip operator + RMS-align (no gauge/lift). Mirrors baseline
        # adam-polar-product-lora but uses clip instead of polar — tests
        # whether spectrum-preservation helps when cross-coupling is NOT
        # absorbed by a gauge constraint.
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(beta1, beta2), delta=precond_delta, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            operator_type="clip",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-clip-product-lora-coupled":
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(beta1, beta2), delta=precond_delta, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            picard_alpha=picard_alpha,
            operator_type="clip",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-clip-product-lora-coupled-endrms":
        return AdamPolarProductLoRA(
            model, lr=lr, betas=(beta1, beta2), delta=precond_delta, eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method, higham_iters=higham_iters,
            picard_iters=2, end_rms_align=True,
            operator_type="clip",
            precond_delta_relative=precond_delta_relative,
            log_non_finite=log_non_finite,
            debug_optimizer_state=debug_optimizer_state,
            debug_optimizer_state_every=debug_optimizer_state_every,
            debug_snapshot_dir=debug_snapshot_dir,
            debug_snapshot_limit=debug_snapshot_limit,
            debug_abort_on_non_finite=debug_abort_on_non_finite,
        )
    if optimizer_type == "adam-polar-product-lora-gauge":
        return AdamPolarProductLoRAGauge(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            precond_method=precond_method,
            higham_iters=higham_iters,
            precond_delta_relative=precond_delta_relative,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-gauge-coupled":
        return AdamPolarProductLoRAGauge(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            precond_method=precond_method,
            higham_iters=higham_iters,
            precond_delta_relative=precond_delta_relative,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-clip-gauge":
        return AdamPolarProductLoRAClipGauge(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 1,
            precond_method=precond_method,
            higham_iters=higham_iters,
            precond_delta_relative=precond_delta_relative,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adam-polar-product-lora-clip-gauge-coupled":
        return AdamPolarProductLoRAClipGauge(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lora_plus_multiplier=lora_plus_multiplier,
            picard_iters=picard_iters_override if picard_iters_override is not None else 2,
            precond_method=precond_method,
            higham_iters=higham_iters,
            precond_delta_relative=precond_delta_relative,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-scalar-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-preserve-scalar",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-preserve",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-imbalance-restore-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="imbalance-restore",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-balanced-scalar-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="balanced-scalar",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-state-rebalanced-lora":
        return PolarCoupledCoreLoRA(
            model, lr=lr, delta=1e-6,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-factor-adam-lora":
        # Rung-6 ablation: Adam-EMA on factor gradients, then projected-quotient-polar.
        # Direct theoretical comparison to Picard's adam-polar-product-lora-coupled.
        return PolarCoupledCoreFactorAdamLoRA(
            model, lr=lr, delta=1e-6,
            betas=(beta1, beta2), eps=1e-8,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "polar-coupled-core-factor-adam-rebalanced-lora":
        return PolarCoupledCoreFactorAdamLoRA(
            model, lr=lr, delta=1e-6,
            betas=(beta1, beta2), eps=1e-8,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
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
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-imbalance-scalar-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="imbalance-preserve-scalar",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-imbalance-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="imbalance-preserve",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-balanced-scalar-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="balanced-scalar",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-state-rebalanced-lora":
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            state_rebalance=True, rebalance_every=1,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-sign-lora":
        # variant 2 + sign norm: momentum AND per-coord adaptivity in core.
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-coupled-core-sign-rebalanced-lora":
        # variant 2 + sign norm + state rebalance: full stack.
        return MuonCoupledCoreLoRA(
            model, lr=lr, delta=1e-6, beta1=0.95,
            core_scale="squared_penalty", gauge="min-frobenius",
            pre_polar_normalize="sign",
            state_rebalance=True, rebalance_every=1,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "adamuon-polar-product-lora":
        return AdamuonPolarProductLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            delta=precond_delta,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            sign_stabilize=True,
            lora_plus_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
            precond_refresh_every=precond_refresh_every,
            precond_method=precond_method,
            higham_iters=higham_iters,
            precond_delta_relative=precond_delta_relative,
        )
    if optimizer_type == "adamuon-lora":
        return AdaMuonLoRA(
            model, lr=lr,
            beta=0.95,
            eps=1e-8,
            ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
            log_basic_diagnostics=log_basic_diagnostics, log_heavy_diagnostics=log_heavy_diagnostics,
            diagnostics_every=optim_diagnostics_every,
        )
    if optimizer_type == "muon-lora":
        return MuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "imuon-lora":
        # iMuon baseline = the authors' VENDORED reference (arXiv:2605.09238), `variant='v5_warmup'`
        # (their built-in init-stable variant: runs the joint `full` form to grow B from zero,
        # then v5). Measured ~3.9 s/step at OLMo r256. NOTE: we tried the authors' BATCHED
        # MuonBatched for speed — it was marginally SLOWER here (4.05 s/step) because the
        # all-linear LoRA shapes are heterogeneous (tiny per-shape groups → grouping overhead >
        # batching benefit), and lacks the warmup (higher param_l2). So we stay on per-pair Muon
        # v5_warmup. The decoupled Cor 4.1 is non-viable at B=0 (δ^{-1/2} blowup). Deviations
        # from default: wd=0 (our protocol), adjust_lr=False (scalar lr). JOINT-momentum (≠ Cor 4.1).
        from .third_party.imuon_muon import Muon as _IMuonRef
        pairs = collect_lora_pairs(model)
        if not pairs:
            raise ValueError("No LoRA (A,B) tensors found on model for imuon-lora.")
        muon_params = [p for A, B in pairs for p in (A, B)]
        return _IMuonRef(
            lr=lr, wd=0.0,
            muon_params=muon_params,
            momentum=0.95, nesterov=True, ns_steps=5,
            lora_pairs=pairs,
            lora_riemannian_muon=True,
            lora_riemannian_variant="v5_warmup",
            lora_riemannian_adjust_lr=False,
        )
    if optimizer_type == "product-muon-lora":
        return ProductMuonLoRA(
            model, lr=lr, ns_steps=muon_ns_steps,
            alpha=muon_alpha, rank=muon_rank,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "adam-muon-lora":
        return AdamMuonLoRA(
            model, lr=lr, betas=(beta1, beta2), ns_steps=muon_ns_steps,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "adam-product-muon-lora":
        return AdamProductMuonLoRA(
            model, lr=lr, betas=(beta1, beta2), ns_steps=muon_ns_steps,
            alpha=muon_alpha, rank=muon_rank,
            lr_b_multiplier=lora_plus_multiplier,
        )
    if optimizer_type == "adam-ucv-core-lora":
        return AdamOrthogonalCoreLoRA(
            model, lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
    if optimizer_type == "muon-adam-lora":
        return MuonAdamLoRA(
            model, lr=lr, betas=(beta1, beta2), ns_steps=muon_ns_steps,
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
