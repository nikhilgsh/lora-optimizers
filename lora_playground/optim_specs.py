"""Declarative optimizer specs + the generic builder.

See ``docs/notes/optimizer_factory_redesign.md``. Each optimizer is an
``OptimizerSpec(cls, fixed, defaults, alias, takes_targets)``. The builder
introspects ``cls.__init__`` and auto-forwards every parameter the class accepts
from the ``OptimizerConfig`` (by name or via ``ALIAS``), applying ``fixed``
(identity constants) and ``defaults`` (per-optimizer defaults for an otherwise
auto-forwarded kwarg). **Forwarding is automatic — a silent drop requires an
explicit, visible ``fixed=`` entry.**

This module is being populated family-by-family (cw first — the paper
protagonist), each gated by ``tests/test_optim_factory_equivalence.py`` against
the legacy ``build_optimizer``. Until every optimizer is migrated, ``build_v2``
covers the registered subset.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch as _torch

from .optim_config import OptimizerConfig, ALIAS, CONFIG_FIELDS
from . import optim as _optim
from .utils import collect_lora_pairs as _collect_lora_pairs


@dataclass
class OptimizerSpec:
    cls: Optional[type] = None
    fixed: dict = field(default_factory=dict)      # identity constants (eps, kl_coupled, …)
    defaults: dict = field(default_factory=dict)   # per-optimizer default for an auto-forwarded kwarg
    alias: dict = field(default_factory=dict)      # per-spec override of the global ALIAS
    takes_targets: bool = False                    # built from `targets`, not `model`
    # Custom builder for optimizers NOT expressible as cls(model_or_targets, lr,
    # **introspected_kwargs): a function-built optimizer (adafactor), torch.optim
    # with a filtered param list (sgd), or a vendored reference with derived args
    # (imuon). A permanent category — new optimizers may need one too — not a
    # migration leftover. When set, build_from_spec calls it instead of cls.
    build: Optional[Callable] = None
    # Config fields this variant must NOT receive even though its class accepts
    # them — left at the class default. The legacy branch dropped them on purpose
    # (a flag invalid for THIS variant, e.g. cw_nesterov on the soap_v=True
    # curvature-whiten path, which the constructor rejects). Distinct from `fixed`
    # (a constant we DO pass): skip means "use the class default."
    skip: set = field(default_factory=set)


REGISTRY: dict[str, OptimizerSpec] = {}


def spec(name, cls=None, *, fixed=None, defaults=None, alias=None,
         takes_targets=False, build=None, skip=None):
    REGISTRY[name] = OptimizerSpec(cls, fixed or {}, defaults or {}, alias or {},
                                   takes_targets, build, skip or set())


def _forwardable_params(cls):
    """Constructor params to consider for forwarding: everything after self and the
    first positional (model/targets), excluding lr and *args/**kwargs.

    Walks the MRO (bounded to this package) so a subclass with ``__init__(self,
    model, lr, **kwargs)`` that forwards to a base (the soap/adafactor polar-product
    subclasses) contributes the base's named params too. Stops at the first class
    whose ``__init__`` does NOT absorb ``**kwargs`` (the real consumer)."""
    seen: dict[str, inspect.Parameter] = {}
    for c in cls.__mro__:
        if not getattr(c, "__module__", "").startswith("lora_playground"):
            continue
        params = list(inspect.signature(c.__init__).parameters.values())[1:]  # drop self
        dropped_first, has_var_kw = False, False
        for p in params:
            if p.kind == p.VAR_KEYWORD:
                has_var_kw = True
                continue
            if p.kind == p.VAR_POSITIONAL:
                continue
            if not dropped_first:
                dropped_first = True  # model/targets
                continue
            if p.name == "lr":
                continue
            seen.setdefault(p.name, p)
        if not has_var_kw:
            break  # this class consumes its args directly — don't walk further up
    return list(seen.values())


def build_from_spec(model_or_targets, name: str, config: OptimizerConfig):
    """The single declarative optimizer builder. A spec with a custom ``build``
    callable is built by it; otherwise config fields are forwarded by
    introspection over ``spec.cls.__init__`` (the generic path)."""
    if name not in REGISTRY:
        raise KeyError(f"optimizer '{name}' has no declarative spec")
    s = REGISTRY[name]
    if s.build is not None:
        return s.build(model_or_targets, config)
    kwargs = {}
    for p in _forwardable_params(s.cls):
        n = p.name
        if n in s.fixed:
            kwargs[n] = s.fixed[n]
        elif n == "betas":
            kwargs[n] = (config.beta1, config.beta2)
        elif n == "picard_iters":
            ov = config.picard_iters_override
            kwargs[n] = ov if ov is not None else s.defaults.get(n, p.default)
        elif n in s.defaults:
            kwargs[n] = s.defaults[n]
        else:
            fld = s.alias.get(n) or ALIAS.get(n, n)
            if fld in s.skip:
                continue  # this variant must not receive this field → class default
            if fld in CONFIG_FIELDS:
                val = getattr(config, fld)
                # precond_method=None means "use the class's family default" (cw→eigh,
                # polar-product→higham): omit it so each class applies its own default.
                # An explicit value still forwards (and raises if invalid for the family).
                if n == "precond_method" and val is None:
                    continue
                kwargs[n] = val
            # else: not a config field and not fixed → class default
    # Apply any fixed keys the signature loop didn't cover — these are absorbed
    # by the class's **kwargs (e.g. eps on the soap/adafactor polar-product
    # subclasses, which forward to the AdamPolarProductLoRA base).
    for k, v in s.fixed.items():
        kwargs.setdefault(k, v)
    return s.cls(model_or_targets, lr=config.lr, **kwargs)


# ─── Curvature-whiten family (the paper protagonist) ─────────────────────────
# All nine share CurvatureWhitenLoRA; `fixed` is the (kl_coupled, soap_v,
# diag_metric, use_polar[, flat_outer]) identity. `eps`=1e-8 is the class default
# (the legacy branch passed it explicitly; same value). The cw_* ablation flags
# auto-forward ONLY on the kl-diag / diag-shampoo protagonist branches; `skip`
# reproduces the legacy drops on the others (soap_v=True rejects cw_nesterov /
# cw_picard_iters; the radius/diag/factor ablations are scoped to kl-diag).
_CW = _optim.CurvatureWhitenLoRA
# precond_method/higham_iters now forward from the config (default None → the cw
# class default eigh, via build_from_spec's None-omit), so an explicit
# precond_method="gram_ns"/"higham" reaches the cw protagonist instead of being
# silently dropped. `_CW_PRECOND_SKIP` is kept (empty) as the composition anchor for
# the soap/ablation skip sets below, in case cw-only skips are needed later.
_CW_PRECOND_SKIP: set = set()
_CW_SOAP_SKIP = {"cw_nesterov", "cw_picard_iters", "cw_no_radius", "cw_no_diag_curv",
                 "cw_factor_a", "cw_factor_b"} | _CW_PRECOND_SKIP     # soap_v=True: all cw_* invalid
_CW_ABL_SKIP = {"cw_no_radius", "cw_no_diag_curv", "cw_factor_a", "cw_factor_b"} | _CW_PRECOND_SKIP  # kl-shampoo/flatout

spec("curvature-whiten-lora", _CW, skip=_CW_SOAP_SKIP,
     fixed={"kl_coupled": False, "soap_v": True, "diag_metric": False, "use_polar": False})
spec("curvature-whiten-polar-lora", _CW, skip=_CW_SOAP_SKIP,
     fixed={"kl_coupled": False, "soap_v": True, "diag_metric": False, "use_polar": True})
spec("kl-shampoo-lora", _CW, skip=_CW_ABL_SKIP,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": False, "use_polar": False})
spec("kl-shampoo-polar-lora", _CW, skip=_CW_ABL_SKIP,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": False, "use_polar": True})
spec("kl-diag-lora", _CW, skip=_CW_PRECOND_SKIP,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": False})
spec("kl-diag-polar-lora", _CW, skip=_CW_PRECOND_SKIP,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": True})
spec("kl-diag-polar-flatout-lora", _CW, skip=_CW_ABL_SKIP,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": True,
            "flat_outer": True})
spec("diag-shampoo-lora", _CW, skip=_CW_PRECOND_SKIP,
     fixed={"kl_coupled": False, "soap_v": False, "diag_metric": True, "use_polar": False})
spec("diag-shampoo-polar-lora", _CW, skip=_CW_PRECOND_SKIP,
     fixed={"kl_coupled": False, "soap_v": False, "diag_metric": True, "use_polar": True})


# ─── Polar-product family ────────────────────────────────────────────────────
# All AdamPolarProductLoRA-based; the branch differs only in `magnitude_rule` /
# operator_type / end_rms_align / exact_chord / etc. (the identity literals) and
# the per-optimizer `picard_iters` default. Everything else auto-forwards.
_APP = _optim.AdamPolarProductLoRA
_E8 = {"eps": 1e-8}
# base + coupled share AdamPolarProductLoRA's default magnitude rule (not passed).
spec("adam-polar-product-lora", _APP, fixed=_E8,
     alias={"core_remix_alpha": "polar_core_remix_alpha"})
spec("adam-polar-product-lora-coupled", _APP, fixed=_E8, defaults={"picard_iters": 3})
spec("adam-polar-product-lora-coupled-endrms", _APP,
     fixed={**_E8, "end_rms_align": True, "picard_iters": 2})
spec("adam-polar-product-lora-coupled-spectral-chord", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord"}, defaults={"picard_iters": 3})
spec("adam-polar-product-lora-coupled-spectral-chord-tight", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight"})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-clean", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_clean"})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_clean", "fw_linearization": "full"})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-no-rho", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_no_rho"})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-exact", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight", "exact_chord": True})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight", "disable_whitening": True})
spec("adam-polar-product-lora-coupled-spectral-chord-direction", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_direction"})
spec("adam-polar-product-lora-coupled-exact-chord", _APP,
     fixed={**_E8, "exact_chord": True}, defaults={"picard_iters": 3})
spec("adam-clip-product-lora", _APP,
     fixed={**_E8, "operator_type": "clip"})
spec("adam-clip-product-lora-coupled", _APP,
     fixed={**_E8, "operator_type": "clip"}, defaults={"picard_iters": 2})
spec("adam-clip-product-lora-coupled-endrms", _APP,
     fixed={**_E8, "operator_type": "clip", "end_rms_align": True, "picard_iters": 2})

spec("adam-soap-polar-product-lora", _optim.AdamSOAPPolarProductLoRA,
     fixed=_E8)
spec("adafactor-polar-product-lora", _optim.AdaFactorPolarProductLoRA,
     fixed=_E8)
spec("sign-momentum-polar-product-lora", _optim.SignMomentumPolarProductLoRA,
     fixed=_E8)

# bare polar-product (no Adam EMA — PolarProductLoRA). `delta` is the SWEPT
# precond_delta (legacy: delta=precond_delta) → ALIAS forwards it.
spec("polar-product-lora", _optim.PolarProductLoRA)

# gauge / clip-gauge variants (AdamPolarProductLoRAGauge / …ClipGauge). Same
# AdamPolarProductLoRA contract: eps=1e-8 literal, delta=precond_delta (ALIAS),
# precond_method/higham_iters/precond_delta_relative auto-forward. The bare/
# coupled pair differs ONLY in the per-optimizer picard_iters default (1 vs 2).
spec("adam-polar-product-lora-gauge", _optim.AdamPolarProductLoRAGauge,
     fixed=_E8)
spec("adam-polar-product-lora-gauge-coupled", _optim.AdamPolarProductLoRAGauge,
     fixed=_E8, defaults={"picard_iters": 2})
spec("adam-polar-product-lora-clip-gauge", _optim.AdamPolarProductLoRAClipGauge,
     fixed=_E8)
spec("adam-polar-product-lora-clip-gauge-coupled", _optim.AdamPolarProductLoRAClipGauge,
     fixed=_E8, defaults={"picard_iters": 2})


# ─── Polar-coupled-core family ───────────────────────────────────────────────
# All PolarCoupledCoreLoRA (or the FactorAdam subclass), differing only in the
# `gauge`/`pre_polar_normalize`/`state_rebalance` identity literals. `delta`=1e-6
# is the class default but legacy HARDCODES it (does NOT forward precond_delta),
# so it MUST be `fixed` — otherwise the swept precond_delta over-forwards via ALIAS.
# `core_scale="squared_penalty"` is the class default and legacy passes it
# explicitly (same value) — left to the class default, no need to pin.
_PCC = _optim.PolarCoupledCoreLoRA
_PCC_FA = _optim.PolarCoupledCoreFactorAdamLoRA
_D6 = {"delta": 1e-6}

spec("polar-coupled-core-lora", _PCC, fixed={**_D6})
spec("polar-coupled-core-imbalance-scalar-lora", _PCC,
     fixed={**_D6, "gauge": "imbalance-preserve-scalar"})
spec("polar-coupled-core-imbalance-lora", _PCC,
     fixed={**_D6, "gauge": "imbalance-preserve"})
spec("polar-coupled-core-imbalance-restore-lora", _PCC,
     fixed={**_D6, "gauge": "imbalance-restore"})
spec("polar-coupled-core-balanced-scalar-lora", _PCC,
     fixed={**_D6, "gauge": "balanced-scalar"})
spec("polar-coupled-core-state-rebalanced-lora", _PCC,
     fixed={**_D6, "state_rebalance": True, "rebalance_every": 1})
spec("polar-coupled-core-sign-lora", _PCC,
     fixed={**_D6, "pre_polar_normalize": "sign"})
spec("polar-coupled-core-sign-rebalanced-lora", _PCC,
     fixed={**_D6, "pre_polar_normalize": "sign",
            "state_rebalance": True, "rebalance_every": 1})
# FactorAdam subclass: Adam-EMA on factors (takes betas/eps) — betas special-cased,
# eps=1e-8 is the class default (legacy passes it explicitly, same value).
spec("polar-coupled-core-factor-adam-lora", _PCC_FA, fixed={**_D6})
spec("polar-coupled-core-factor-adam-rebalanced-lora", _PCC_FA,
     fixed={**_D6, "state_rebalance": True, "rebalance_every": 1})


# ─── Muon-coupled-core family ────────────────────────────────────────────────
# MuonCoupledCoreLoRA. Identical structure to polar-coupled-core but with a
# Muon-canonical HARDCODED momentum: legacy passes literal beta1=0.95 (a
# different momentum convention from config.beta1) → must be `fixed`, NOT the
# auto-forwarded config.beta1. Same `delta=1e-6` HARDCODED → fixed.
_MCC = _optim.MuonCoupledCoreLoRA
_MCC_FIXED = {**_D6, "beta1": 0.95}

spec("muon-coupled-core-lora", _MCC, fixed={**_MCC_FIXED})
spec("muon-coupled-core-imbalance-scalar-lora", _MCC,
     fixed={**_MCC_FIXED, "gauge": "imbalance-preserve-scalar"})
spec("muon-coupled-core-imbalance-lora", _MCC,
     fixed={**_MCC_FIXED, "gauge": "imbalance-preserve"})
spec("muon-coupled-core-balanced-scalar-lora", _MCC,
     fixed={**_MCC_FIXED, "gauge": "balanced-scalar"})
spec("muon-coupled-core-state-rebalanced-lora", _MCC,
     fixed={**_MCC_FIXED, "state_rebalance": True, "rebalance_every": 1})
spec("muon-coupled-core-sign-lora", _MCC,
     fixed={**_MCC_FIXED, "pre_polar_normalize": "sign"})
spec("muon-coupled-core-sign-rebalanced-lora", _MCC,
     fixed={**_MCC_FIXED, "pre_polar_normalize": "sign",
            "state_rebalance": True, "rebalance_every": 1})


# ─── Muon / AdaMuon / ProductMuon family ─────────────────────────────────────
# `ns_steps`←muon_ns_steps and `lr_b_multiplier`←lora_plus_multiplier are global
# ALIAS entries; `alpha`←muon_alpha, `rank`←muon_rank likewise. `beta`/`eps` are
# class defaults (legacy passes the same literals). `delta` on the Product
# variants is HARDCODED at the class default 1e-6 (legacy passes NO delta) →
# must be `fixed` so the swept precond_delta doesn't over-forward via ALIAS.
spec("muon-lora", _optim.MuonLoRA)
spec("adamuon-lora", _optim.AdaMuonLoRA)          # beta=0.95 / eps=1e-8 class defaults
spec("muon-adam-lora", _optim.MuonAdamLoRA)
spec("adam-muon-lora", _optim.AdamMuonLoRA)
spec("product-muon-lora", _optim.ProductMuonLoRA, fixed={**_D6})
spec("adam-product-muon-lora", _optim.AdamProductMuonLoRA, fixed={**_D6})
spec("adam-ucv-core-lora", _optim.AdamOrthogonalCoreLoRA)  # weight_decay auto-forwards
# AdamuonPolarProductLoRA: sign_stabilize=True is the class default AND legacy
# passes it explicitly (same value); delta=precond_delta (ALIAS), eps=1e-8 literal.
spec("adamuon-polar-product-lora", _optim.AdamuonPolarProductLoRA, fixed=_E8)


# ─── Baselines ───────────────────────────────────────────────────────────────
# adamw: LoRAPlusAdamW. betas special-cased, eps=1e-8 / weight_decay class
# defaults (legacy passes same), lora_plus_multiplier auto-forwards. NOTE: stores
# betas in param_groups, not self.beta1 (see test's _effective_betas).
spec("adamw", _optim.LoRAPlusAdamW)

# lin/scaled family. `delta`=1e-6 is HARDCODED in legacy (NOT the swept
# precond_delta) → must be `fixed`. eps=1e-8 is the class default. scaled_metric
# / lora_plus_multiplier auto-forward where the class takes them (adam-scaled
# does NOT take scaled_metric — auto-forward skips it).
spec("lin-lora", _optim.LinLoRA, fixed={**_D6})
spec("scaled-lora", _optim.ScaledLoRA, fixed={**_D6})
spec("adam-scaled-lora", _optim.AdamScaledLoRA, fixed={**_D6})
spec("adam-lin-lora", _optim.AdamLinLoRA, fixed={**_D6})
spec("adam-lin-core-lora", _optim.AdamLinCoreLoRA, fixed={**_D6})
spec("adam-scaled-lora-post", _optim.AdamScaledLoRAPost, fixed={**_D6})
spec("adam-lin-lora-post", _optim.AdamLinLoRAPost, fixed={**_D6})
spec("adam-scaled-lora-matrix", _optim.AdamScaledLoRAMatrix, fixed={**_D6})
spec("adam-lin-lora-matrix", _optim.AdamLinLoRAMatrix, fixed={**_D6})

# diag-scaled / kron-grad / psi: gamma←precond_gamma, ema_beta←precond_ema_beta,
# delta←precond_delta are all SWEPT (legacy forwards them) → global ALIAS. psi's
# momentum/inner_iters/proximal_rho/momentum_rank map to psi_* config fields.
spec("diag-scaled-lora", _optim.DiagScaledLoRA)
spec("kron-grad-lora", _optim.KronGradLoRA)
spec("psi-lora", _optim.PSILoRA,
     alias={"momentum": "psi_momentum", "inner_iters": "psi_inner_iters",
            "proximal_rho": "psi_rho", "momentum_rank": "psi_momentum_rank"})


# ─── Targets-based (dense-weight) optimizers ─────────────────────────────────
# Built from `targets` (TargetWeight list), not a LoRA model → takes_targets=True.
# `rank`←svd_rank (per-spec alias, NOT muon_rank). eps literals differ (galore
# 1e-6, svd 1e-8) → fixed. galore update_proj_gap/scale are global ALIAS;
# svd_niter / weight_decay auto-forward by name.
spec("galore-adamw", _optim.GaLoreAdamW,
     fixed={"eps": 1e-6}, alias={"rank": "svd_rank"}, takes_targets=True)
spec("svd-step-adamw", _optim.SVDStepAdamW,
     fixed={"eps": 1e-8}, alias={"rank": "svd_rank"}, takes_targets=True)
spec("svd-cumulative-adamw", _optim.SVDCumulativeAdamW,
     fixed={"eps": 1e-8}, alias={"rank": "svd_rank"}, takes_targets=True)


# ─── Custom-build optimizers ─────────────────────────────────────────────────
# Not expressible as cls(model_or_targets, lr, **introspected_kwargs): a function
# (adafactor), torch.optim.SGD over a filtered param list with a fixed momentum
# (sgd), or the vendored iMuon reference with derived muon_params/lora_pairs and
# bespoke Riemannian kwargs. Each carries an explicit `build=` callable.
def _build_adafactor(model, config):
    return _optim._build_lora_adafactor(
        model, lr=config.lr, lora_plus_multiplier=config.lora_plus_multiplier,
        weight_decay=config.weight_decay)


def _sgd_builder(momentum):
    def _build(model, config):
        params = [p for p in model.parameters() if p.requires_grad]
        return _torch.optim.SGD(params, lr=config.lr, momentum=momentum)
    return _build


def _build_imuon(model, config):
    pairs = _collect_lora_pairs(model)
    if not pairs:
        raise ValueError("No LoRA (A,B) tensors found on model for imuon-lora.")
    from .third_party.imuon_muon import Muon as _IMuonRef
    muon_params = [p for A, B in pairs for p in (A, B)]
    return _IMuonRef(
        lr=config.lr, wd=0.0, muon_params=muon_params,
        momentum=0.95, nesterov=True, ns_steps=5, lora_pairs=pairs,
        lora_riemannian_muon=True, lora_riemannian_variant="v5_warmup",
        lora_riemannian_adjust_lr=False)


spec("adafactor", build=_build_adafactor)
spec("sgd", build=_sgd_builder(0.0))
spec("sgd-m", build=_sgd_builder(0.9))
spec("imuon-lora", build=_build_imuon)
