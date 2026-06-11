"""Behavioral-equivalence gate for the declarative-factory migration.

For every migrated optimizer, the NEW generic builder (build_v2 + a spec) must
produce an optimizer that is attribute-identical AND step-identical to the LEGACY
``build_optimizer`` branch, at the same hyperparameters. A migrated optimizer is
not "done" until it passes here. See docs/notes/optimizer_factory_redesign.md.
"""
import copy
import inspect
from dataclasses import asdict, replace

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import build_optimizer
from lora_playground.optim_config import OptimizerConfig
from lora_playground.optim_specs import build_v2, REGISTRY


class _Pair(nn.Module):
    def __init__(self, r, di, do, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(di, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, do, bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.randn(r, di, generator=g) * 0.1)
            self.lora_B["default"].weight.copy_(torch.randn(do, r, generator=g) * 0.1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([_Pair(8, 32, 32, 1), _Pair(8, 32, 64, 2),
                                       _Pair(8, 32, 32, 3)])


_BO_PARAMS = set(inspect.signature(build_optimizer).parameters)


def _old(model, name, cfg):
    kw = {k: v for k, v in asdict(cfg).items() if k in _BO_PARAMS}
    return build_optimizer(model, optimizer_type=name, **kw)


def _scalars(opt):
    return {k: v for k, v in vars(opt).items()
            if isinstance(v, (int, float, bool, str)) or v is None}


def _inject_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def _assert_step_identical(name, cfg, n_steps=3):
    m_old = _Model()
    m_new = copy.deepcopy(m_old)
    opt_old = _old(m_old, name, cfg)
    opt_new = build_v2(m_new, name, cfg)
    for step in range(n_steps):
        _inject_grads(m_old, 100 + step)
        _inject_grads(m_new, 100 + step)
        opt_old.step()
        opt_new.step()
    # Compare the resulting model params (class-agnostic — not every optimizer
    # exposes `.pairs`). Identical optimizers leave identical weights.
    for (n1, p_old), (n2, p_new) in zip(m_old.named_parameters(), m_new.named_parameters()):
        assert torch.allclose(p_old, p_new, atol=1e-6, rtol=1e-5), f"{name}: {n1} diverged"


# Config non-default ONLY on the UNIVERSALLY-FORWARDED core (every branch passes
# these). Flags that some branches omit (curvature_beta, polar_method, …) stay at
# their default, so the auto-forward equals the class-default the omitting branch
# relied on — no spurious attr divergence on an unused over-forwarded param. The
# step test is the behavioral gate; family-specific flags get their own configs.
_COMMON = OptimizerConfig(
    lr=7e-3, beta1=0.93, beta2=0.991, precond_delta=3.3e-4,
    muon_ns_steps=7, precond_refresh_every=13, optim_diagnostics_every=50,
    lora_plus_multiplier=1.5,
)

_CW_FAMILY = ["curvature-whiten-lora", "curvature-whiten-polar-lora",
              "kl-shampoo-lora", "kl-shampoo-polar-lora",
              "kl-diag-lora", "kl-diag-polar-lora", "kl-diag-polar-flatout-lora",
              "diag-shampoo-lora", "diag-shampoo-polar-lora"]


@pytest.mark.parametrize("name", _CW_FAMILY)
def test_cw_attr_identical(name):
    old = _scalars(_old(_Model(), name, _COMMON))
    new = _scalars(build_v2(_Model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"


@pytest.mark.parametrize("name", _CW_FAMILY)
def test_cw_step_identical(name):
    _assert_step_identical(name, _COMMON)


# cw ablation flags + curvature_beta — valid only on the soap_v=False +
# diag_metric=True branches (which forward all of these).
_CW_ABLATION = replace(_COMMON, cw_nesterov=True, cw_no_radius=True,
                       cw_factor_a=0.3, cw_picard_iters=2, curvature_beta=0.97)
_CW_DIAG = ["kl-diag-lora", "kl-diag-polar-lora",
            "diag-shampoo-lora", "diag-shampoo-polar-lora"]


@pytest.mark.parametrize("name", _CW_DIAG)
def test_cw_ablation_flags_equivalent(name):
    """The protagonist family forwards its ablation flags identically old/new."""
    old = _scalars(_old(_Model(), name, _CW_ABLATION))
    new = _scalars(build_v2(_Model(), name, _CW_ABLATION))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: ablation-flag divergence old/new -> {diff}"
    _assert_step_identical(name, _CW_ABLATION)


_POLAR_FAMILY = [
    "adam-polar-product-lora", "adam-polar-product-lora-coupled",
    "adam-polar-product-lora-coupled-endrms",
    "adam-polar-product-lora-coupled-spectral-chord",
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean",
    "adam-polar-product-lora-coupled-spectral-chord-tight-clean-full-fw",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-rho",
    "adam-polar-product-lora-coupled-spectral-chord-tight-exact",
    "adam-polar-product-lora-coupled-spectral-chord-tight-no-whitening",
    "adam-polar-product-lora-coupled-spectral-chord-direction",
    "adam-polar-product-lora-coupled-exact-chord",
    "adam-clip-product-lora", "adam-clip-product-lora-coupled",
    "adam-clip-product-lora-coupled-endrms",
    "adam-soap-polar-product-lora", "adafactor-polar-product-lora",
    "sign-momentum-polar-product-lora",
]


@pytest.mark.parametrize("name", _POLAR_FAMILY)
def test_polar_attr_identical(name):
    old = _scalars(_old(_Model(), name, _COMMON))
    new = _scalars(build_v2(_Model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"


@pytest.mark.parametrize("name", _POLAR_FAMILY)
def test_polar_step_identical(name):
    _assert_step_identical(name, _COMMON)


def test_registry_specs_have_valid_fixed_keys():
    """Every spec.fixed key is a real __init__ param of its class."""
    bad = {}
    for name, s in REGISTRY.items():
        sig = inspect.signature(s.cls.__init__)
        params = set(sig.parameters)
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        # a fixed key is valid if it's a named param OR the class absorbs **kwargs
        unknown = (set(s.fixed) - params) if not has_var_kw else set()
        if unknown:
            bad[name] = unknown
    assert not bad, f"specs with fixed keys not in __init__ (no **kwargs): {bad}"
