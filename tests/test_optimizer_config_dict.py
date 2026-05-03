"""Convention test for optimizer_config_dict.

Walks OPTIMIZER_CHOICES, instantiates each via build_optimizer on a tiny LoRA
model, and asserts every non-skipped __init__ param of the resulting optimizer
class is recorded in optimizer_config_dict() (i.e. no "<unrecorded>" entries).

This is the anti-staleness guarantee for fix #1: a new optimizer that doesn't
store its __init__ args as same-named attributes (or via _CONFIG_DICT_ALIASES)
fails CI before merge, instead of silently producing config events that omit
algorithm-distinguishing hyperparameters.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import (
    OPTIMIZER_CHOICES,
    build_optimizer,
    optimizer_config_dict,
)


# Optimizer types that need dense `targets` (full-finetune SVD oracle, GaLore).
# build_optimizer raises without them; tested separately or skipped here.
_NEEDS_TARGETS = {"galore-adamw", "svd-step-adamw", "svd-cumulative-adamw"}

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


_TESTABLE = sorted(OPTIMIZER_CHOICES - _NEEDS_TARGETS)


@pytest.mark.parametrize("optimizer_type", _TESTABLE)
def test_config_dict_records_all_init_params(optimizer_type):
    """Every __init__ param (sans construction-input skips) must be recorded."""
    model = TinyLoRAModel()
    opt = build_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr=1e-3,
    )
    cfg = optimizer_config_dict(opt)
    unrecorded = sorted(k for k, v in cfg.items() if v == "<unrecorded>")
    assert not unrecorded, (
        f"{optimizer_type} ({type(opt).__name__}): __init__ params not stored "
        f"as same-named attributes: {unrecorded}. Either store them as "
        f"`self.<param_name>` in __init__, or add an entry to "
        f"`_CONFIG_DICT_ALIASES` in optim.py."
    )
    assert cfg["_optim_class"] == type(opt).__name__
