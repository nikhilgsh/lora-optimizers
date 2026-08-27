"""Convention test for optimizer_config_dict.

Walks OPTIMIZER_CHOICES, instantiates each via build_optimizer on a tiny LoRA
model, and asserts every non-skipped __init__ param of the resulting optimizer
class is recorded in optimizer_config_dict() (i.e. no "<unrecorded>" entries).

This is the anti-staleness guarantee for fix #1: a new optimizer that doesn't
store its __init__ args as same-named attributes (or via _CONFIG_DICT_ALIASES)
fails CI before merge, instead of silently producing config events that omit
algorithm-distinguishing hyperparameters.
"""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import (
    _CONFIG_DICT_SKIP,
    OPTIMIZER_CHOICES,
    build_optimizer,
    optimizer_config_dict,
    optimizer_effective_config,
)
from lora_playground.constructor_introspection import (
    forwardable_constructor_parameters,
)
from lora_playground.publication_semantics import (
    build_optimizer_variant_semantics_payload,
)


# Optimizer types that need dense `targets` (full-finetune SVD oracle, GaLore)
# or a non-BA model layout. Tested separately or skipped here.
_NEEDS_TARGETS = {"galore-adamw", "svd-step-adamw", "svd-cumulative-adamw"}
_NEEDS_UCV = {"adam-ucv-core-lora"}

class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.zeros_(self.lora_B["default"].weight)


class TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=2):
        super().__init__()
        self.layer0 = _FakeLoRALinear(d_in, d_out, r)
        self.layer1 = _FakeLoRALinear(d_out, d_in, r)


class _TinyUCVModel(nn.Module):
    """Plain Linear stack; UCV adapters are injected by the test fixture."""

    def __init__(self, d_in=8, d_out=6, r=2):
        super().__init__()
        self.layer0 = nn.Linear(d_in, d_out, bias=False)
        self.layer1 = nn.Linear(d_out, d_in, bias=False)


def _build_model_for(optimizer_type: str):
    if optimizer_type in _NEEDS_UCV:
        from lora_playground.ucv_layer import inject_ucv_adapters
        m = _TinyUCVModel()
        inject_ucv_adapters(m, target_modules="all-linear", r=2, alpha=2)
        return m
    return TinyLoRAModel()


_TESTABLE = sorted(OPTIMIZER_CHOICES - _NEEDS_TARGETS)


@pytest.mark.parametrize("optimizer_type", _TESTABLE)
def test_config_dict_records_all_init_params(optimizer_type):
    """Every __init__ param (sans construction-input skips) must be recorded."""
    model = _build_model_for(optimizer_type)
    opt = build_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr=1e-3,
    )
    cfg = optimizer_config_dict(opt)
    expected = {
        parameter.name
        for parameter in forwardable_constructor_parameters(type(opt))
        if parameter.name not in _CONFIG_DICT_SKIP
    } | {"lr"}
    assert expected <= cfg.keys(), (
        f"{optimizer_type} ({type(opt).__name__}) omitted constructor fields: "
        f"{sorted(expected - cfg.keys())}"
    )
    assert cfg["_optim_class"] == type(opt).__name__
    payload = build_optimizer_variant_semantics_payload(
        optimizer=optimizer_type,
        optimizer_instance=opt,
        optimizer_config=cfg,
        optimizer_effective=optimizer_effective_config(opt),
        semantic_revision=1,
        implementation_revision="test-source",
    )
    assert payload["optimizer"] == optimizer_type
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_config_dict_rejects_group_specific_betas():
    opt = build_optimizer(
        TinyLoRAModel(), optimizer_type="adamw", lr=1e-3
    )
    opt.param_groups[1]["betas"] = (0.8, 0.9)

    with pytest.raises(ValueError, match="group-specific betas"):
        optimizer_config_dict(opt)


def test_adamw_records_executed_betas_and_unscaled_base_lr():
    opt = build_optimizer(
        TinyLoRAModel(),
        optimizer_type="adamw",
        lr=3e-4,
        beta1=0.81,
        beta2=0.9564,
        lora_plus_multiplier=8.0,
    )

    cfg = optimizer_config_dict(opt)

    assert cfg["betas"] == (0.81, 0.9564)
    assert cfg["lr"] == 3e-4
    assert {group["lr"] for group in opt.param_groups} == {3e-4, 2.4e-3}


@pytest.mark.parametrize(
    "optimizer_type, inherited_fields",
    [
        (
            "adam-soap-polar-product-lora",
            {"betas", "polar_method", "precond_method", "magnitude_rule"},
        ),
        (
            "adafactor-polar-product-lora",
            {"betas", "polar_method", "precond_method", "magnitude_rule"},
        ),
    ],
)
def test_forwarding_subclasses_record_inherited_semantic_fields(
    optimizer_type, inherited_fields
):
    opt = build_optimizer(
        TinyLoRAModel(), optimizer_type=optimizer_type, lr=1e-3
    )

    cfg = optimizer_config_dict(opt)

    assert inherited_fields <= cfg.keys()


@pytest.mark.parametrize(
    "optimizer_type, expected",
    [
        ("sgd", {"momentum": 0, "weight_decay": 0}),
        ("adafactor", {"weight_decay": 0.0, "relative_step": False}),
    ],
)
def test_external_optimizers_record_their_concrete_constructor_fields(
    optimizer_type, expected
):
    opt = build_optimizer(
        TinyLoRAModel(), optimizer_type=optimizer_type, lr=1e-3
    )

    cfg = optimizer_config_dict(opt)

    assert cfg["lr"] == 1e-3
    for key, value in expected.items():
        assert cfg[key] == value
