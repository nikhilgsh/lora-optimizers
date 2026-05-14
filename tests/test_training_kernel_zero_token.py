from types import SimpleNamespace

import pytest
import torch

from lora_playground.training_kernel import run_one_train_step


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids, labels):
        self.forward_calls += 1
        mask = labels != -100
        if not bool(mask.any()):
            raise AssertionError("zero-token microbatch should be skipped")
        target = input_ids[mask].float().mean()
        return SimpleNamespace(loss=(self.weight - target).pow(2))


def _batch(labels):
    input_ids = torch.arange(labels.numel(), dtype=torch.long).reshape_as(labels)
    return {"input_ids": input_ids, "labels": labels.clone()}


def test_run_one_train_step_skips_zero_token_microbatch():
    model = TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    zero = _batch(torch.full((1, 4), -100, dtype=torch.long))
    valid = _batch(torch.tensor([[-100, 1, 2, -100]], dtype=torch.long))

    step_loss, step_tokens, train_iter, norm_stats = run_one_train_step(
        model,
        optimizer,
        iter([zero, valid]),
        [zero, valid],
        grad_accum_steps=2,
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert model.forward_calls == 1
    assert step_tokens == 2
    assert torch.isfinite(torch.tensor(step_loss))
    assert norm_stats["skipped_zero_token_microbatches"] == 1
    assert norm_stats["n_non_finite_grads"] == 0
    assert model.weight.detach() != torch.tensor(1.0)
    assert train_iter is not None


def test_run_one_train_step_rejects_all_zero_token_macrostep():
    model = TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    zero = _batch(torch.full((1, 4), -100, dtype=torch.long))

    with pytest.raises(RuntimeError, match="zero supervised tokens"):
        run_one_train_step(
            model,
            optimizer,
            iter([zero, zero]),
            [zero, zero],
            grad_accum_steps=2,
            max_grad_norm=None,
            device=torch.device("cpu"),
        )

    assert model.forward_calls == 0
