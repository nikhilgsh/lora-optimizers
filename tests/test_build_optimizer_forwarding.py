"""Guardrail: ``build_optimizer`` must FORWARD tunable flags, never silently
drop them behind a hardcoded literal in a branch.

History (the bug this prevents): the curvature-whiten branches passed
``betas=(0.9, 0.999)`` literally instead of ``betas=(beta1, beta2)``, so every
``kl-diag-polar-lora`` / ``diag-shampoo-polar-lora`` run executed at beta1=0.9
*while its config event logged --beta1* (0.95 for the "locked" protagonist).
Same class as the earlier silently-dropped ``cw_nesterov`` / ``cw_no_*`` flags.
A config event that records a value the optimizer did not use is a
reproducibility hazard; this test makes the drop a CI failure.

Scope: this enforces the *unambiguous* forwarding contracts —

  * ``betas`` (beta1/beta2): every optimizer whose ``__init__`` takes a ``betas``
    tuple must forward it. This is the recurring-bug axis.
  * ``precond_method``: the polar-product family (eigh/higham switch) must forward it.
  * the curvature-whiten ablation flags on the protagonist family.

Flags whose attribute name collides with an optimizer-internal structural
constant (e.g. ``delta`` is a fixed Tikhonov reg in adam-lin/adam-scaled, NOT the
swept ``precond_delta``; ``curvature_beta`` only applies to the whitening
families) are *not* asserted universally here — whether a given branch should
forward them is a domain judgment handled by the optim.py audit, not a blanket
CI rule. Two known out-of-contract spots, deliberately left to the audit:
``muon-coupled-core-*`` hardcodes a Muon-canonical ``beta1=0.95`` (different
momentum convention, takes ``beta1`` not ``betas``), and several product-lora
variants hardcode ``polar_method='ns'``.
"""
import inspect

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import build_optimizer, OPTIMIZER_CHOICES


class _Pair(nn.Module):
    """One (A, B) LoRA pair in PEFT convention so collect_lora_pairs finds it."""
    def __init__(self, r, d_in, d_out):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A["default"].weight)
            self.lora_B["default"].weight.zero_()


class _FakeModel(nn.Module):
    def __init__(self, specs=((8, 32, 32), (8, 32, 64))):
        super().__init__()
        self.adapters = nn.ModuleList([_Pair(*s) for s in specs])


# Dense-target optimizers: built from `targets`, not LoRA pairs — different path.
_NEEDS_TARGETS = {"galore-adamw", "svd-step-adamw", "svd-cumulative-adamw"}


def _build(opt_type, **over):
    return build_optimizer(_FakeModel(), optimizer_type=opt_type, lr=1e-3, **over)


def _takes(opt, kwarg):
    try:
        return kwarg in inspect.signature(type(opt).__init__).parameters
    except (TypeError, ValueError):
        return False


def _effective_betas(opt):
    """(beta1, beta2) as the optimizer actually holds them — custom optimizers
    store self.beta1/self.beta2; torch.optim-derived ones (LoRAPlusAdamW) keep
    them in param_groups[0]['betas']."""
    if hasattr(opt, "beta1") and hasattr(opt, "beta2"):
        return (opt.beta1, opt.beta2)
    pg = opt.param_groups[0] if getattr(opt, "param_groups", None) else {}
    return tuple(pg["betas"]) if "betas" in pg else None


@pytest.mark.parametrize("opt_type", sorted(OPTIMIZER_CHOICES))
def test_betas_forwarded(opt_type):
    """Any optimizer whose __init__ takes a `betas` tuple must receive
    build_optimizer's (beta1, beta2), not a hardcoded literal."""
    if opt_type in _NEEDS_TARGETS:
        pytest.skip("dense-target optimizer")
    try:
        opt = _build(opt_type, beta1=0.93, beta2=0.991)
    except Exception as e:  # noqa: BLE001 — unbuildable on the fake model; out of scope
        pytest.skip(f"build unsupported on fake model: {type(e).__name__}")
    if not _takes(opt, "betas"):
        pytest.skip("optimizer does not take a `betas` tuple")
    assert _effective_betas(opt) == (0.93, 0.991), (
        f"{opt_type}: betas not forwarded (got {_effective_betas(opt)}); "
        f"the branch must pass betas=(beta1, beta2)."
    )


@pytest.mark.parametrize("opt_type", sorted(o for o in OPTIMIZER_CHOICES if "polar-product" in o))
def test_precond_method_forwarded(opt_type):
    """The polar-product family stores precond_method (eigh/higham); the branch
    must forward it. Sentinel 'eigh' differs from the build default 'higham'."""
    try:
        opt = _build(opt_type, precond_method="eigh")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"build unsupported on fake model: {type(e).__name__}")
    if not hasattr(opt, "precond_method"):
        pytest.skip("optimizer does not store precond_method")
    assert opt.precond_method == "eigh", (
        f"{opt_type}: precond_method not forwarded (got {opt.precond_method!r})."
    )


CW_PROTAGONIST_FAMILY = [
    "kl-diag-polar-lora", "kl-diag-lora",
    "diag-shampoo-polar-lora", "diag-shampoo-lora",
]
CW_CONTRACT = [
    ("cw_nesterov", True, "cw_nesterov"),
    ("cw_no_radius", True, "cw_no_radius"),
    ("cw_no_diag_curv", True, "cw_no_diag_curv"),
    ("cw_picard_iters", 2, "cw_picard_iters"),
    ("precond_delta_relative", True, "precond_delta_relative"),
    ("curvature_beta", 0.97, "curvature_beta"),
    ("precond_delta", 3.3e-5, "delta"),
    ("muon_ns_steps", 7, "ns_steps"),
    ("polar_method", "polar_express", "polar_method"),
    ("precond_refresh_every", 13, "precond_refresh_every"),
]


@pytest.mark.parametrize("opt_type", CW_PROTAGONIST_FAMILY)
def test_curvature_whiten_flags_forwarded(opt_type):
    """The protagonist family forwards its momentum/ablation/whitening flags.
    The historical silent-drop bugs (beta1, cw_nesterov, cw_no_*) lived here."""
    opt = _build(opt_type, beta1=0.95, **{f: v for f, v, _ in CW_CONTRACT})
    drops = [f"  {flag} -> self.{attr}: got {getattr(opt, attr)!r}, expected {val!r}"
             for flag, val, attr in [("beta1", 0.95, "beta1"), *CW_CONTRACT]
             if getattr(opt, attr) != val]
    assert not drops, f"{opt_type} drops protagonist flags:\n" + "\n".join(drops)


def test_beta1_regression_curvature_whiten():
    """Pinpoint regression for the exact shipped bug: --beta1 reached the config
    event but not the optimizer for the curvature-whiten family."""
    for opt_type in ("kl-diag-polar-lora", "diag-shampoo-polar-lora"):
        opt = _build(opt_type, beta1=0.95)
        assert opt.beta1 == 0.95, (
            f"{opt_type}: --beta1 silently dropped (got {opt.beta1}); the branch "
            f"must forward betas=(beta1, beta2), not hardcode (0.9, 0.999)."
        )
