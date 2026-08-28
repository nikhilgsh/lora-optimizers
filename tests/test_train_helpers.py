import copy
from types import SimpleNamespace

import torch

from lora_playground.train import (
    _resume_replays_original_dataloader,
    _resume_restores_rng_state,
    attach_heldout_factor_grads,
    format_example,
    format_example_with_boundary,
    make_parser,
    measure_heldout_factor_directions,
    parse_target_modules,
)
from lora_playground.training_kernel import run_one_train_step


class _PairModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(2, 3))
        self.B = torch.nn.Parameter(torch.randn(4, 2))

    def forward(self, input_ids, labels):
        pred = input_ids @ (self.B @ self.A).T
        return SimpleNamespace(loss=(pred - labels.float()).square().mean())


def test_format_prompt_completion_example():
    text = format_example({"prompt": "Write Python.", "completion": "print('ok')"})
    assert text == "Write Python.\nprint('ok')"


def test_format_alpaca_example_with_input():
    text = format_example(
        {
            "instruction": "Add two numbers.",
            "input": "1 2",
            "output": "3",
        }
    )
    assert "Instruction:" in text
    assert "Input:" in text
    assert "Response:" in text
    assert text.endswith("3")


def test_format_instruction_response_example():
    text = format_example({"instruction": "Square x.", "response": "return x * x"})
    assert text == "Instruction:\nSquare x.\n\nResponse:\nreturn x * x"


def test_format_instruction_output_no_input():
    """OpenCoder opc-sft-stage2 schema: {instruction, output} with no input."""
    ex = {"instruction": "Reverse the list.", "output": "lst[::-1]"}
    text = format_example(ex)
    assert "Instruction:\nReverse the list." in text
    assert "Input:" not in text
    assert text.endswith("Response:\nlst[::-1]")
    prompt, response = format_example_with_boundary(ex)
    assert prompt == "Instruction:\nReverse the list.\n\nResponse:\n"
    assert response == "lst[::-1]"


def test_format_instruction_output_empty_input():
    """Defensive: input field present but empty should be skipped."""
    ex = {"instruction": "Reverse the list.", "input": "", "output": "lst[::-1]"}
    text = format_example(ex)
    assert "Input:" not in text
    prompt, response = format_example_with_boundary(ex)
    assert "Input:" not in prompt


def test_parse_target_modules():
    assert parse_target_modules("all-linear") == "all-linear"
    assert parse_target_modules("q_proj,k_proj, v_proj") == ["q_proj", "k_proj", "v_proj"]


def test_resume_debug_replay_implies_dataloader_replay_and_rng_restore():
    args = make_parser().parse_args(["--resume_debug_replay"])
    assert _resume_replays_original_dataloader(args) is True
    assert _resume_restores_rng_state(args) is True


def test_resume_replay_original_dataloader_does_not_restore_rng_by_default():
    args = make_parser().parse_args(["--resume_replay_original_dataloader"])
    assert _resume_replays_original_dataloader(args) is True
    assert _resume_restores_rng_state(args) is False


def test_heldout_probe_multi_batch_fast_exit_flags_parse():
    args = make_parser().parse_args([
        "--optim_heldout_probe",
        "--optim_heldout_probe_batches", "8",
        "--optim_heldout_probe_exit",
        "--optim_heldout_identity_scale", "0.3333333333333333",
        "--optim_small_slot_microbatch_probe",
    ])
    assert args.optim_heldout_probe_batches == 8
    assert args.optim_heldout_probe_exit is True
    assert args.optim_heldout_identity_scale == 1 / 3
    assert args.optim_small_slot_microbatch_probe is True


def test_training_microbatch_capture_mean_matches_aggregate_with_zero_slot():
    torch.manual_seed(61)
    model = _PairModel()
    control = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    control_optimizer = torch.optim.SGD(control.parameters(), lr=0.1)
    optimizer.pairs = [(model.A, model.B)]
    control_optimizer.pairs = [(control.A, control.B)]
    batches = [
        {
            "input_ids": torch.randn(1, 3),
            "labels": torch.randn(1, 4),
        },
        {
            "input_ids": torch.randn(2, 3),
            "labels": torch.full((2, 4), -100.0),
        },
        {
            "input_ids": torch.randn(3, 3),
            "labels": torch.randn(3, 4),
        },
    ]
    captured = {}

    def inspect_capture():
        payload = optimizer._factor_microbatch_grads
        captured.update(payload)
        captured["train_grads"] = (
            model.A.grad.detach().float().clone(),
            model.B.grad.detach().float().clone(),
        )

    run_one_train_step(
        model, optimizer, iter(batches), batches,
        grad_accum_steps=3, max_grad_norm=None, device=torch.device("cpu"),
        pre_step_callback=inspect_capture,
        capture_factor_microbatch_grads=True,
    )
    run_one_train_step(
        control, control_optimizer, iter(batches), batches,
        grad_accum_steps=3, max_grad_norm=None, device=torch.device("cpu"),
    )

    assert captured["microbatch_count"] == 3
    assert captured["valid_microbatch_count"] == 2
    zero_A, zero_B = captured["grads"][1][0]
    assert torch.count_nonzero(zero_A) == 0
    assert torch.count_nonzero(zero_B) == 0
    mean_A = torch.stack([record[0][0] for record in captured["grads"]]).mean(0)
    mean_B = torch.stack([record[0][1] for record in captured["grads"]]).mean(0)
    assert torch.allclose(mean_A, captured["aggregate_grads"][0][0])
    assert torch.allclose(mean_B, captured["aggregate_grads"][0][1])
    assert torch.equal(captured["train_grads"][0], captured["aggregate_grads"][0][0])
    assert torch.equal(captured["train_grads"][1], captured["aggregate_grads"][0][1])
    assert torch.equal(model.A, control.A)
    assert torch.equal(model.B, control.B)
    assert not hasattr(optimizer, "_factor_microbatch_grads")


def test_heldout_factor_grads_restore_train_state_and_rng():
    torch.manual_seed(7)
    model = _PairModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.pairs = [(model.A, model.B)]
    train_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }
    heldout_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }
    model(**train_batch).loss.backward()
    train_grads = (model.A.grad.clone(), model.B.grad.clone())
    rng = torch.get_rng_state().clone()

    attach_heldout_factor_grads(
        model, optimizer, heldout_batch, torch.device("cpu"))

    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(model.A.grad, train_grads[0])
    assert torch.equal(model.B.grad, train_grads[1])
    heldout_grads = optimizer._heldout_factor_grads[0]
    assert not torch.equal(heldout_grads[0], train_grads[0])
    assert not torch.equal(heldout_grads[1], train_grads[1])


def test_heldout_factor_grads_token_weight_multiple_batches():
    torch.manual_seed(8)
    model = _PairModel()
    expected_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.pairs = [(model.A, model.B)]
    train_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }
    heldout_batches = [
        {"input_ids": torch.randn(1, 3), "labels": torch.randn(1, 4)},
        {"input_ids": torch.randn(3, 3), "labels": torch.randn(3, 4)},
    ]
    model(**train_batch).loss.backward()
    train_grads = (model.A.grad.clone(), model.B.grad.clone())
    for parameter in expected_model.parameters():
        parameter.grad = None
    for batch, weight in zip(heldout_batches, (1 / 4, 3 / 4)):
        (expected_model(**batch).loss * weight).backward()

    attach_heldout_factor_grads(
        model, optimizer, heldout_batches, torch.device("cpu"))

    heldout_grads = optimizer._heldout_factor_grads[0]
    assert torch.allclose(heldout_grads[0], expected_model.A.grad)
    assert torch.allclose(heldout_grads[1], expected_model.B.grad)
    assert torch.equal(model.A.grad, train_grads[0])
    assert torch.equal(model.B.grad, train_grads[1])


def test_heldout_factor_probe_does_not_change_train_step():
    torch.manual_seed(11)
    control = _PairModel()
    probed = copy.deepcopy(control)
    control_optimizer = torch.optim.SGD(control.parameters(), lr=0.1)
    probed_optimizer = torch.optim.SGD(probed.parameters(), lr=0.1)
    control_optimizer.pairs = [(control.A, control.B)]
    probed_optimizer.pairs = [(probed.A, probed.B)]
    train_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }
    heldout_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }

    control_result = run_one_train_step(
        control,
        control_optimizer,
        iter([train_batch]),
        [train_batch],
        grad_accum_steps=1,
        max_grad_norm=None,
        device=torch.device("cpu"),
    )
    probed_result = run_one_train_step(
        probed,
        probed_optimizer,
        iter([train_batch]),
        [train_batch],
        grad_accum_steps=1,
        max_grad_norm=None,
        device=torch.device("cpu"),
        pre_step_callback=lambda: attach_heldout_factor_grads(
            probed,
            probed_optimizer,
            heldout_batch,
            torch.device("cpu"),
        ),
    )

    assert control_result[0] == probed_result[0]
    assert control_result[1] == probed_result[1]
    assert torch.equal(control.A, probed.A)
    assert torch.equal(control.B, probed.B)
    assert not hasattr(probed_optimizer, "_heldout_factor_grads")


def test_heldout_factor_direction_losses_restore_applied_step():
    torch.manual_seed(13)
    model = _PairModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.pairs = [(model.A, model.B)]
    heldout_batch = {
        "input_ids": torch.randn(2, 3),
        "labels": torch.randn(2, 4),
    }
    A_pre = model.A.detach().clone()
    B_pre = model.B.detach().clone()
    directions = {
        "lagged": (torch.randn_like(model.A) * 0.01,
                   torch.randn_like(model.B) * 0.01),
        "fresh": (torch.randn_like(model.A) * 0.01,
                  torch.randn_like(model.B) * 0.01),
        "identity": (torch.randn_like(model.A) * 0.01,
                     torch.randn_like(model.B) * 0.01),
    }
    optimizer._last_cw_heldout_directions = {
        0: {"A_pre": A_pre, "B_pre": B_pre, **directions},
    }
    with torch.no_grad():
        model.A.add_(directions["lagged"][0])
        model.B.add_(directions["lagged"][1])
    A_post = model.A.detach().clone()
    B_post = model.B.detach().clone()

    def expected_loss(label):
        if label == "pre":
            A, B = A_pre, B_pre
        else:
            dA, dB = directions[label]
            A, B = A_pre + dA, B_pre + dB
        pred = heldout_batch["input_ids"] @ (B @ A).T
        return float((pred - heldout_batch["labels"]).square().mean())

    result = measure_heldout_factor_directions(
        model, optimizer, heldout_batch, torch.device("cpu"), identity_scale=1 / 3)

    for label in ("pre", "lagged", "fresh", "identity"):
        assert result[f"heldout_loss_{label}"] == expected_loss(label)
    dA, dB = directions["identity"]
    pred = heldout_batch["input_ids"] @ (
        (B_pre + dB / 3) @ (A_pre + dA / 3)
    ).T
    expected_scaled = float((pred - heldout_batch["labels"]).square().mean())
    assert result["heldout_loss_identity_scaled"] == expected_scaled
    assert result["heldout_loss_pre_repeat_abs_diff"] == 0.0
    assert torch.equal(model.A, A_post)
    assert torch.equal(model.B, B_post)
    assert optimizer._last_cw_heldout_directions == {}


def test_heldout_factor_direction_losses_token_weight_multiple_batches():
    torch.manual_seed(17)
    model = _PairModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.pairs = [(model.A, model.B)]
    batches = [
        {"input_ids": torch.randn(1, 3), "labels": torch.randn(1, 4)},
        {"input_ids": torch.randn(3, 3), "labels": torch.randn(3, 4)},
    ]
    A_pre = model.A.detach().clone()
    B_pre = model.B.detach().clone()
    zero_A = torch.zeros_like(model.A)
    zero_B = torch.zeros_like(model.B)
    optimizer._last_cw_heldout_directions = {
        0: {
            "A_pre": A_pre,
            "B_pre": B_pre,
            "lagged": (zero_A, zero_B),
            "fresh": (zero_A, zero_B),
            "identity": (zero_A, zero_B),
        },
    }

    losses = [float(model(**batch).loss) for batch in batches]
    expected = (losses[0] + 3 * losses[1]) / 4
    result = measure_heldout_factor_directions(
        model, optimizer, batches, torch.device("cpu"))

    assert result["heldout_probe_batches"] == 2
    for label in ("pre", "lagged", "fresh", "identity"):
        assert result[f"heldout_loss_{label}"] == expected
