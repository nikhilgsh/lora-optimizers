"""``OptimizerConfig`` — the single source of truth for optimizer hyperparameters.

See ``docs/notes/optimizer_factory_redesign.md``. This dataclass mirrors the
tunable parameters of ``build_optimizer`` exactly (same names, same defaults), so:

  * ``from_namespace(args)`` is a trivial field-by-field copy (the argparse names
    match), and
  * it is logged DIRECTLY as the config event (``asdict(config)``), so the log can
    never drift from what was passed — there is no second hand-maintained list.

Structural constants (``eps``, ``magnitude_rule``, ``operator_type``,
``kl_coupled``, …) are NOT here — they are the optimizer's identity and live in
its ``OptimizerSpec.fixed`` (see ``optim_specs.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import Optional


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float

    # Adam / momentum
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0

    # preconditioner / damping
    precond_delta: float = 1e-6
    precond_delta_relative: bool = False
    precond_gamma: float = 0.5
    precond_ema_beta: float = 0.99
    # Gram-preconditioner cache refresh cadence. Live for adam-scaled-lora /
    # adam-lin-lora / adam-lin-core-lora / adam-polar-product-lora /
    # adamuon-polar-product-lora. For the curvature-whiten family
    # (CurvatureWhitenLoRA: kl-diag-*, kl-shampoo-*, diag-shampoo-*,
    # curvature-whiten-*) it gates ONLY the QR-eigenbasis refresh and therefore
    # does nothing unless precond_method == "eigh"; the production "gram_ns"
    # path (and "higham") rebuilds S^{-1/2} from the current Gram every step.
    precond_refresh_every: int = 1
    precond_method: str | None = None  # None → optimizer class's family default (cw=eigh, pp=higham)
    higham_iters: int = 10
    higham_compute_dtype: str = "fp32"
    curvature_beta: float = 0.99
    curvature_whitening: bool = False
    scaled_metric: bool = False
    lora_plus_multiplier: float = 1.0

    # SVD / GaLore (targets-based)
    svd_rank: Optional[int] = None
    svd_niter: int = 4
    galore_update_proj_gap: int = 200
    galore_scale: float = 0.25

    # PSI
    psi_inner_iters: int = 1
    psi_momentum: float = 0.9
    psi_rho: float = 0.01
    psi_momentum_rank: Optional[int] = None

    # Muon / polar
    muon_ns_steps: int = 5
    muon_alpha: int = 16
    muon_rank: int = 16
    polar_norm_dir: str = "frob"
    polar_sigma_power: Optional[float] = None
    polar_method: str = "ns"
    polar_core_remix_alpha: float = 0.0
    ns_form: str = "gram"
    htmuon_p: Optional[float] = None

    # Picard / Anderson
    picard_alpha: float = 1.0
    picard_iters_override: Optional[int] = None
    anderson_m: int = 0
    anderson_reg: float = 1e-10

    # curvature-whiten ablation flags
    # rdinv_variant: reference scale for the relative-damping floor in _rdinv
    # (the large-side diagonal metric). "A" = own op-norm (shipped/paper:
    # (x/x_max+δ)^{-1/2}, floor δ·x_max); "B" = raw/unbiased KL gauge
    # (x+δ·x_max)^{-1/2}, same op-norm floor (= A up to a per-step gauge, but the
    # D_in/D_out EMA diverges because SAinv carries the partner factor's
    # time-varying max); "VN" = von Neumann / matrix Adafactor (x+δ·Tr(partner))^{-1/2},
    # the trace-scaled projection (Wu Lin et al. Table 2, S_a=E[GGᵀ]/Tr(S_b)). δ is
    # NOT comparable across variants (op-norm-relative for A/B, trace-relative for
    # VN). Only "A" reproduces the paper figures; B/VN are the investigation.
    rdinv_variant: str = "A"
    # Decouples the _rdinv (P,Q diagonal-metric) damping floor from precond_delta
    # (which also sets the small-side C_A/C_B inverse-sqrt floor). None -> use
    # precond_delta (coupled). Set it to sweep the diagonal floor alone (e.g. VN's
    # trace-relative δ) while holding the curvature-inverse floor fixed.
    rdinv_delta: float | None = None
    # Init of the diagonal metric EMAs D_in (=Q) / D_out (=P). "zero" (shipped/paper):
    # step 1 uses the _rdinv identity fallback (the step-one rule). "ones": step 1
    # normalizes to the same identity without the special case, but the EMA carries a
    # decaying β₂ᵗ identity prior through warmup. Identical step-1 update; differs only
    # in the warmup transient. Ablation only — "zero" reproduces the paper figures.
    cw_metric_init: str = "1e-12"
    cw_picard_iters: int = 1
    cw_nesterov: bool = False
    cw_no_radius: bool = False
    cw_no_diag_curv: bool = False
    cw_no_rr_precond: bool = False
    cw_unpinned: bool = False
    cw_solved_rho: bool = False
    cw_factor_a: float = 0.0
    cw_factor_b: float = 0.0
    # Pre-polar (H) dump — the matrices msign is applied to, saved for the
    # offline approximate-LMO scores in lora_playground.lmo_diagnostics.
    # OFF at 0. Diagnostic only: the update is unchanged.
    dump_pre_polar_dir: Optional[str] = None
    dump_pre_polar_every: int = 0
    dump_pre_polar_pairs: Optional[str] = None
    dump_pre_polar_max_pairs: int = 6

    # SOAP
    soap_beta: float = 0.95
    soap_refresh_every: int = 1

    # SSC (spectral soft clip)
    ssc_c: Optional[float] = None
    ssc_nsteps: int = 10
    ssc_kappa: Optional[float] = None
    ssc_kappa_refresh_every: int = 1
    ssc_kappa_warmup_steps: int = 5
    ssc_kappa_solver: str = "eigvalsh"
    ssc_kappa_bisect_iters: int = 3
    ssc_kappa_cache_share_picard: bool = False
    ssc_kappa_cache_ema_beta: Optional[float] = None
    ssc_kappa_bisect_mode: str = "sequential"
    ssc_kappa_bisect_nsteps_eval: Optional[int] = None
    ssc_kappa_cross_group_eigvalsh: bool = True
    ssc_kappa_diagnose_eigvalsh: bool = False
    ssc_kappa_diagnose_start_step: int = 1
    ssc_kappa_diag_ema_beta: Optional[float] = None

    # diagnostics / debug
    log_basic_diagnostics: bool = False
    log_heavy_diagnostics: bool = False
    optim_diagnostics_every: int = 20
    log_non_finite: bool = False
    log_non_finite_start_step: int = 1
    debug_optimizer_state: bool = False
    debug_optimizer_state_every: int = 1
    debug_optimizer_state_start_step: int = 1
    debug_snapshot_dir: Optional[str] = None
    debug_snapshot_limit: int = 8
    debug_abort_on_non_finite: bool = False

    @classmethod
    def from_namespace(cls, args) -> "OptimizerConfig":
        """Populate from an argparse Namespace (or any attr-bag). Field names match
        the CLI flag dests, so this is a direct copy of the present fields."""
        kw = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                kw[f.name] = getattr(args, f.name)
        if "lr" not in kw:
            kw["lr"] = getattr(args, "learning_rate", None)
        return cls(**kw)

    def as_event(self) -> dict:
        """The config-event payload — the config object IS the log (no re-listing)."""
        return asdict(self)


# Field-name set for the generic builder's auto-forward (excludes lr, which is
# passed positionally, and the targets-only svd_rank which the builder routes).
CONFIG_FIELDS = {f.name for f in fields(OptimizerConfig)}


# constructor-kwarg name  ->  OptimizerConfig field name (when they differ).
# `betas` and `picard_iters` are special-cased in the builder (see optim_specs).
ALIAS = {
    "ns_steps": "muon_ns_steps",
    "delta": "precond_delta",
    "lr_b_multiplier": "lora_plus_multiplier",
    "diagnostics_every": "optim_diagnostics_every",
    "alpha": "muon_alpha",
    "rank": "muon_rank",
    "gamma": "precond_gamma",
    "ema_beta": "precond_ema_beta",
    "update_proj_gap": "galore_update_proj_gap",
    "scale": "galore_scale",
}
