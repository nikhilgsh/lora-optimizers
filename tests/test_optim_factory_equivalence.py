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


# ── Coupled-core family (polar / muon) ──────────────────────────────────────
# All PolarCoupledCoreLoRA / MuonCoupledCoreLoRA, differing only in the gauge /
# pre_polar_normalize / state_rebalance identity literals (+ muon's beta1=0.95).
# `delta`=1e-6 is HARDCODED in legacy (not the swept precond_delta) → in fixed.
_POLAR_COUPLED_CORE_FAMILY = [
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
]
_MUON_COUPLED_CORE_FAMILY = [
    "muon-coupled-core-lora",
    "muon-coupled-core-imbalance-scalar-lora",
    "muon-coupled-core-imbalance-lora",
    "muon-coupled-core-balanced-scalar-lora",
    "muon-coupled-core-state-rebalanced-lora",
    "muon-coupled-core-sign-lora",
    "muon-coupled-core-sign-rebalanced-lora",
]
_COUPLED_CORE_FAMILY = _POLAR_COUPLED_CORE_FAMILY + _MUON_COUPLED_CORE_FAMILY


@pytest.mark.parametrize("name", _COUPLED_CORE_FAMILY)
def test_coupled_core_attr_identical(name):
    old = _scalars(_old(_Model(), name, _COMMON))
    new = _scalars(build_v2(_Model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"


@pytest.mark.parametrize("name", _COUPLED_CORE_FAMILY)
def test_coupled_core_step_identical(name):
    _assert_step_identical(name, _COMMON)


def test_muon_coupled_core_beta1_is_pinned_not_config():
    """muon-coupled-core HARDCODES beta1=0.95 (Muon momentum convention), NOT the
    auto-forwarded config.beta1 (0.93 in _COMMON). Pin must survive."""
    for name in _MUON_COUPLED_CORE_FAMILY:
        opt = build_v2(_Model(), name, _COMMON)
        assert opt.beta1 == 0.95, f"{name}: beta1 should be the pinned 0.95, got {opt.beta1}"


# ── Muon / AdaMuon / ProductMuon family ─────────────────────────────────────
# `polar-product-lora` (bare) lives here too — same PolarProduct geometry, no Adam.
_MUON_FAMILY = [
    "polar-product-lora",
    "muon-lora", "adamuon-lora", "muon-adam-lora", "adam-muon-lora",
    "product-muon-lora", "adam-product-muon-lora",
    "adamuon-polar-product-lora",
    "adam-polar-product-lora-gauge", "adam-polar-product-lora-gauge-coupled",
    "adam-polar-product-lora-clip-gauge", "adam-polar-product-lora-clip-gauge-coupled",
]


@pytest.mark.parametrize("name", _MUON_FAMILY)
def test_muon_attr_identical(name):
    old = _scalars(_old(_Model(), name, _COMMON))
    new = _scalars(build_v2(_Model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"


@pytest.mark.parametrize("name", _MUON_FAMILY)
def test_muon_step_identical(name):
    _assert_step_identical(name, _COMMON)


# ── Baselines ────────────────────────────────────────────────────────────────
# `adam-ucv-core-lora` needs a UCV-injected model (separate test below); `sgd`,
# `sgd-m`, `adafactor`, `imuon-lora` are NOT migratable generically (see report).
_BASELINE_FAMILY = [
    "adamw",
    "lin-lora", "scaled-lora", "adam-scaled-lora", "adam-lin-lora",
    "adam-lin-core-lora", "adam-scaled-lora-post", "adam-lin-lora-post",
    "adam-scaled-lora-matrix", "adam-lin-lora-matrix",
    "diag-scaled-lora", "kron-grad-lora", "psi-lora",
]


@pytest.mark.parametrize("name", _BASELINE_FAMILY)
def test_baseline_attr_identical(name):
    old = _scalars(_old(_Model(), name, _COMMON))
    new = _scalars(build_v2(_Model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"


@pytest.mark.parametrize("name", _BASELINE_FAMILY)
def test_baseline_step_identical(name):
    _assert_step_identical(name, _COMMON)


def test_adamw_betas_forwarded_via_param_groups():
    """adamw (LoRAPlusAdamW) stores betas in param_groups, not self.beta1 — the
    generic attr test can't see them. Pin the (beta1, beta2) forward explicitly."""
    opt = build_v2(_Model(), "adamw", _COMMON)
    assert tuple(opt.param_groups[0]["betas"]) == (_COMMON.beta1, _COMMON.beta2)


def test_lin_scaled_delta_is_pinned_not_swept():
    """The lin/scaled family HARDCODES delta=1e-6 (structural Tikhonov reg), NOT
    the swept precond_delta (3.3e-4 in _COMMON). The pin must survive."""
    for name in ["lin-lora", "scaled-lora", "adam-scaled-lora", "adam-lin-lora",
                 "adam-lin-core-lora", "adam-scaled-lora-post", "adam-lin-lora-post",
                 "adam-scaled-lora-matrix", "adam-lin-lora-matrix"]:
        opt = build_v2(_Model(), name, _COMMON)
        assert opt.delta == 1e-6, f"{name}: delta should be pinned 1e-6, got {opt.delta}"


# ── UCV core (needs a UCV-injected model, not LoRA pairs) ────────────────────
def _ucv_model(seed=4):
    # inject_ucv_adapters' orthonormal U/V init draws from the GLOBAL torch RNG
    # (no generator arg), so seed it identically for old/new to get the same init.
    from lora_playground.ucv_layer import inject_ucv_adapters
    torch.manual_seed(seed)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 48, bias=False)
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.randn(32, 32, generator=g) * 0.1)
                self.v_proj.weight.copy_(torch.randn(48, 32, generator=g) * 0.1)

    m = _Net()
    torch.manual_seed(seed)  # re-seed: the orthonormal init consumes global RNG
    inject_ucv_adapters(m, ["q_proj", "v_proj"], r=8, alpha=8)
    return m


def test_ucv_core_attr_and_step_identical():
    """adam-ucv-core-lora needs a UCV-injected model (legacy fails on plain LoRA
    pairs too). Build both old/new on a UCV fixture and compare attrs + steps."""
    name = "adam-ucv-core-lora"
    old = _scalars(_old(_ucv_model(), name, _COMMON))
    new = _scalars(build_v2(_ucv_model(), name, _COMMON))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"
    m_old = _ucv_model()
    m_new = _ucv_model()
    opt_old = _old(m_old, name, _COMMON)
    opt_new = build_v2(m_new, name, _COMMON)
    for step in range(3):
        _inject_grads(m_old, 100 + step)
        _inject_grads(m_new, 100 + step)
        opt_old.step()
        opt_new.step()
    for (n1, p_old), (n2, p_new) in zip(m_old.named_parameters(), m_new.named_parameters()):
        assert torch.allclose(p_old, p_new, atol=1e-6, rtol=1e-5), f"{name}: {n1} diverged"


# ── Targets-based (dense-weight) optimizers ──────────────────────────────────
# Built from `targets` (a TargetWeight list), not a LoRA model. The generic
# step path (random-grad inject + compare params) works on the dense weights.
_TARGETS_FAMILY = ["galore-adamw", "svd-step-adamw", "svd-cumulative-adamw"]
# wd MUST be 0 (svd-* reject wd!=0) and svd_rank set (rank<=0 rejected).
_TARGETS_CFG = replace(_COMMON, weight_decay=0.0, svd_rank=4, svd_niter=4,
                       precond_delta=1e-6)


def _targets(seed=7):
    from lora_playground.utils import TargetWeight
    g = torch.Generator().manual_seed(seed)
    ts = []
    for i, (do, di) in enumerate([(32, 32), (64, 32), (48, 40)]):
        m = nn.Linear(di, do, bias=False)
        with torch.no_grad():
            m.weight.copy_(torch.randn(do, di, generator=g) * 0.1)
        ts.append(TargetWeight(name=f"t{i}", module=m, weight=m.weight,
                               base_weight=m.weight.detach().clone()))
    return ts


def _old_targets(targets, name, cfg):
    kw = {k: v for k, v in asdict(cfg).items() if k in _BO_PARAMS and k != "lr"}
    return build_optimizer(None, optimizer_type=name, targets=targets, lr=cfg.lr, **kw)


@pytest.mark.parametrize("name", _TARGETS_FAMILY)
def test_targets_attr_identical(name):
    old = _scalars(_old_targets(_targets(), name, _TARGETS_CFG))
    new = _scalars(build_v2(_targets(), name, _TARGETS_CFG))
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    assert not diff, f"{name}: attr divergence old/new -> {diff}"
    # rank + eps are the load-bearing identity values.
    o = build_v2(_targets(), name, _TARGETS_CFG)
    assert o.rank == _TARGETS_CFG.svd_rank
    assert o.param_groups[0]["eps"] == (1e-6 if name == "galore-adamw" else 1e-8)


@pytest.mark.parametrize("name", _TARGETS_FAMILY)
def test_targets_step_identical(name):
    t_old = _targets()
    t_new = _targets()
    opt_old = _old_targets(t_old, name, _TARGETS_CFG)
    opt_new = build_v2(t_new, name, _TARGETS_CFG)
    for step in range(3):
        g = torch.Generator().manual_seed(200 + step)
        grads = [torch.randn(tw.weight.shape, generator=g) for tw in t_old]
        for tw_o, tw_n, grad in zip(t_old, t_new, grads):
            tw_o.weight.grad = grad.clone()
            tw_n.weight.grad = grad.clone()
        # svd-* project the step via randomized SVD (torch.svd_lowrank draws from
        # the GLOBAL RNG) — seed identically around each .step() so the two
        # independent optimizers consume the same random projection.
        torch.manual_seed(300 + step)
        opt_old.step()
        torch.manual_seed(300 + step)
        opt_new.step()
    for tw_o, tw_n in zip(t_old, t_new):
        assert torch.allclose(tw_o.weight, tw_n.weight, atol=1e-6, rtol=1e-5), \
            f"{name}: {tw_o.name} diverged"


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


# These 4 are NOT migratable to a generic spec (see optimizer_factory_redesign
# report): adafactor is a builder FUNCTION (no __init__ to introspect); sgd/sgd-m
# take a filtered param LIST + a hardcoded momentum, not (model, lr, **kwargs);
# imuon-lora is a vendored ref with a bespoke (muon_params/lora_pairs/...) sig.
_UNMIGRATABLE = {"adafactor", "sgd", "sgd-m", "imuon-lora"}


def test_every_choice_migrated_or_flagged():
    """Every OPTIMIZER_CHOICES name is either in the REGISTRY or in the explicit
    not-generically-migratable set — no silent gaps."""
    from lora_playground.optim import OPTIMIZER_CHOICES
    missing = set(OPTIMIZER_CHOICES) - set(REGISTRY) - _UNMIGRATABLE
    assert not missing, f"unmigrated and unflagged optimizers: {sorted(missing)}"
