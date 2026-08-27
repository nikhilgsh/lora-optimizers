"""Resume ablation that freezes only the factorwise small-matrix slots."""

import copy
import sys

import pytest
import torch
import torch.nn as nn

import lora_playground.train as train_module
from lora_playground.optim import CurvatureWhitenLoRA, build_optimizer
from lora_playground.train import make_parser


class _LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, rank):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(d_in, rank, bias=False)})
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, d_out, bias=False)})
        nn.init.normal_(self.lora_A["default"].weight, std=0.2)
        nn.init.normal_(self.lora_B["default"].weight, std=0.1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = _LoRALinear(8, 6, 3)
        self.l1 = _LoRALinear(6, 8, 3)


_KW = dict(
    lr=2e-2,
    betas=(0.9, 0.999),
    curvature_beta=0.97,
    delta=1e-4,
    use_polar=True,
    ns_steps=8,
    precond_method="gram_ns",
    higham_iters=8,
    kl_coupled=True,
    soap_v=False,
    cw_nesterov=True,
    precond="factorwise",
)


def _inject_grads(model, seed):
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_freeze_holds_only_checkpointed_factorwise_slots():
    """Emulate the resume boundary, then verify only P_A/Q_B stop learning."""
    torch.manual_seed(2)
    model = _Model()
    optimizer = CurvatureWhitenLoRA(model, **_KW)

    # Learn a nontrivial state before the checkpoint/resume boundary.
    for step in range(3):
        _inject_grads(model, 100 + step)
        optimizer.step()

    optimizer.freeze_factorwise_slots = True
    slots_before = [
        (state["P_A"].clone(), state["Q_B"].clone())
        for state in optimizer.pair_state.values()
    ]
    large_before = [
        (state["D_in"].clone(), state["D_out"].clone())
        for state in optimizer.pair_state.values()
    ]
    moments_before = [
        (state["m_A"].clone(), state["m_B"].clone())
        for state in optimizer.pair_state.values()
    ]
    factors_before = [parameter.detach().clone() for parameter in model.parameters()]

    _inject_grads(model, 200)
    optimizer.step()

    for index, state in optimizer.pair_state.items():
        assert torch.equal(state["P_A"], slots_before[index][0])
        assert torch.equal(state["Q_B"], slots_before[index][1])
        assert not torch.equal(state["D_in"], large_before[index][0])
        assert not torch.equal(state["D_out"], large_before[index][1])
        assert not torch.equal(state["m_A"], moments_before[index][0])
        assert not torch.equal(state["m_B"], moments_before[index][1])
    assert all(
        not torch.equal(before, after)
        for before, after in zip(factors_before, model.parameters())
    )


def test_default_off_is_bit_identical_to_explicit_false():
    torch.manual_seed(7)
    default_model = _Model()
    explicit_model = copy.deepcopy(default_model)
    default = CurvatureWhitenLoRA(default_model, **_KW)
    explicit = CurvatureWhitenLoRA(
        explicit_model, freeze_factorwise_slots=False, **_KW)

    for step in range(4):
        _inject_grads(default_model, 300 + step)
        _inject_grads(explicit_model, 300 + step)
        default.step()
        explicit.step()

    for left, right in zip(default_model.parameters(), explicit_model.parameters()):
        assert torch.equal(left, right)
    assert default.pair_state.keys() == explicit.pair_state.keys()
    for index in default.pair_state:
        _assert_state_equal(default.pair_state[index], explicit.pair_state[index])


def test_freeze_requires_factorwise_and_reaches_factory():
    with pytest.raises(ValueError, match="requires precond='factorwise'"):
        CurvatureWhitenLoRA(
            _Model(), freeze_factorwise_slots=True,
            **{**_KW, "precond": "product"})

    optimizer = build_optimizer(
        _Model(), "kl-shampoo-polar-lora", lr=2e-2,
        precond_method="gram_ns", higham_iters=8,
        freeze_factorwise_slots=True,
    )
    assert optimizer.precond == "factorwise"
    assert optimizer.freeze_factorwise_slots is True

    with pytest.raises(ValueError, match="requires a factorwise"):
        build_optimizer(
            _Model(), "adamw", lr=2e-2,
            freeze_factorwise_slots=True,
        )


def test_cli_flag_is_boolean_and_default_off():
    parser = make_parser()
    assert parser.parse_args([]).freeze_factorwise_slots is False
    args = parser.parse_args([
        "--freeze_factorwise_slots", "--resume_from", "checkpoint",
    ])
    assert args.freeze_factorwise_slots is True


def test_cli_freeze_refuses_a_fresh_start(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", "--freeze_factorwise_slots"])
    with pytest.raises(ValueError, match="requires --resume_from"):
        train_module.main()
