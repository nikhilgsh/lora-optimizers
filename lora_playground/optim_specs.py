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

from .optim_config import OptimizerConfig, ALIAS, CONFIG_FIELDS
from . import optim as _optim


@dataclass
class OptimizerSpec:
    cls: type
    fixed: dict = field(default_factory=dict)      # identity constants (eps, kl_coupled, …)
    defaults: dict = field(default_factory=dict)   # per-optimizer default for an auto-forwarded kwarg
    alias: dict = field(default_factory=dict)      # per-spec override of the global ALIAS
    takes_targets: bool = False                    # built from `targets`, not `model`


REGISTRY: dict[str, OptimizerSpec] = {}


def spec(name, cls, *, fixed=None, defaults=None, alias=None, takes_targets=False):
    REGISTRY[name] = OptimizerSpec(cls, fixed or {}, defaults or {}, alias or {}, takes_targets)


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


def build_v2(model_or_targets, name: str, config: OptimizerConfig):
    """Generic builder. Forwarding is by introspection over ``spec.cls.__init__``."""
    if name not in REGISTRY:
        raise KeyError(f"optimizer '{name}' not yet migrated to a declarative spec")
    s = REGISTRY[name]
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
            if fld in CONFIG_FIELDS:
                kwargs[n] = getattr(config, fld)
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
# (the legacy branch passed it explicitly; same value). Everything else
# (betas, delta, ns_steps, polar_method, curvature_beta, precond_*, lora_plus_*,
# diagnostics, cw_*) auto-forwards.
_CW = _optim.CurvatureWhitenLoRA

spec("curvature-whiten-lora", _CW,
     fixed={"kl_coupled": False, "soap_v": True, "diag_metric": False, "use_polar": False})
spec("curvature-whiten-polar-lora", _CW,
     fixed={"kl_coupled": False, "soap_v": True, "diag_metric": False, "use_polar": True})
spec("kl-shampoo-lora", _CW,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": False, "use_polar": False})
spec("kl-shampoo-polar-lora", _CW,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": False, "use_polar": True})
spec("kl-diag-lora", _CW,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": False})
spec("kl-diag-polar-lora", _CW,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": True})
spec("kl-diag-polar-flatout-lora", _CW,
     fixed={"kl_coupled": True, "soap_v": False, "diag_metric": True, "use_polar": True,
            "flat_outer": True})
spec("diag-shampoo-lora", _CW,
     fixed={"kl_coupled": False, "soap_v": False, "diag_metric": True, "use_polar": False})
spec("diag-shampoo-polar-lora", _CW,
     fixed={"kl_coupled": False, "soap_v": False, "diag_metric": True, "use_polar": True})


# ─── Polar-product family ────────────────────────────────────────────────────
# All AdamPolarProductLoRA-based; the branch differs only in `magnitude_rule` /
# operator_type / end_rms_align / exact_chord / etc. (the identity literals) and
# the per-optimizer `picard_iters` default. Everything else auto-forwards.
_APP = _optim.AdamPolarProductLoRA
_E8 = {"eps": 1e-8}
# base + coupled share AdamPolarProductLoRA's default magnitude rule (not passed).
spec("adam-polar-product-lora", _APP, fixed=_E8, defaults={"picard_iters": 1},
     alias={"core_remix_alpha": "polar_core_remix_alpha"})
spec("adam-polar-product-lora-coupled", _APP, fixed=_E8, defaults={"picard_iters": 3})
spec("adam-polar-product-lora-coupled-endrms", _APP,
     fixed={**_E8, "end_rms_align": True, "picard_iters": 2})
spec("adam-polar-product-lora-coupled-spectral-chord", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord"}, defaults={"picard_iters": 3})
spec("adam-polar-product-lora-coupled-spectral-chord-tight", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight"}, defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-clean", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_clean"}, defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_clean", "fw_linearization": "full"},
     defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-no-rho", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight_no_rho"}, defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-exact", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight", "exact_chord": True},
     defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_tight", "disable_whitening": True},
     defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-spectral-chord-direction", _APP,
     fixed={**_E8, "magnitude_rule": "spectral_chord_direction"}, defaults={"picard_iters": 1})
spec("adam-polar-product-lora-coupled-exact-chord", _APP,
     fixed={**_E8, "exact_chord": True}, defaults={"picard_iters": 3})
spec("adam-clip-product-lora", _APP,
     fixed={**_E8, "operator_type": "clip"}, defaults={"picard_iters": 1})
spec("adam-clip-product-lora-coupled", _APP,
     fixed={**_E8, "operator_type": "clip"}, defaults={"picard_iters": 2})
spec("adam-clip-product-lora-coupled-endrms", _APP,
     fixed={**_E8, "operator_type": "clip", "end_rms_align": True, "picard_iters": 2})

spec("adam-soap-polar-product-lora", _optim.AdamSOAPPolarProductLoRA,
     fixed=_E8, defaults={"picard_iters": 1})
spec("adafactor-polar-product-lora", _optim.AdaFactorPolarProductLoRA,
     fixed=_E8, defaults={"picard_iters": 1})
spec("sign-momentum-polar-product-lora", _optim.SignMomentumPolarProductLoRA,
     fixed=_E8, defaults={"picard_iters": 1})
