import torch
from torch import nn

from lora_playground.optim import SVDCumulativeAdamW, SVDStepAdamW
from lora_playground.utils import (
    collect_dense_target_weights,
    effective_lora_delta,
    freeze_all_except_targets,
    svd_to_lora_factors,
    truncated_svd,
)


def matrix_rank(matrix, tol=1e-5):
    return int((torch.linalg.svdvals(matrix.float()) > tol).sum().item())


def set_single_entry_grad(weight, row, col):
    weight.grad = torch.zeros_like(weight)
    weight.grad[row, col] = 1.0


class DirectMatrix(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, cols))

    def forward(self, x):
        return x @ self.weight


class TinyTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(3, 3, bias=False)
        self.block.c_attn = DirectMatrix(3, 3)
        self.lm_head = nn.Linear(3, 3, bias=False)


def test_truncated_svd_reconstructs_low_rank_matrix():
    torch.manual_seed(0)
    left = torch.randn(5, 2)
    right = torch.randn(2, 4)
    matrix = left @ right
    projected = truncated_svd(matrix, rank=2)
    assert torch.allclose(projected, matrix, atol=1e-5, rtol=1e-5)


def test_svd_to_lora_factors_reconstruct_projected_delta_with_scale():
    torch.manual_seed(1)
    delta = torch.randn(4, 5)
    scale = 2.5
    A, B = svd_to_lora_factors(delta, rank=2, scale=scale)
    reconstructed = effective_lora_delta(A, B, scale)
    assert A.shape == (2, 5)
    assert B.shape == (4, 2)
    assert torch.allclose(reconstructed, truncated_svd(delta, 2), atol=1e-5, rtol=1e-5)


def test_collect_dense_target_weights_matches_explicit_suffix_and_freezes_others():
    model = TinyTargets()
    targets = collect_dense_target_weights(model, ["q_proj", "c_attn"])
    assert [target.name for target in targets] == ["block.q_proj", "block.c_attn"]

    freeze_all_except_targets(model, targets)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {"block.q_proj.weight", "block.c_attn.weight"}


def test_collect_dense_target_weights_all_linear_excludes_lm_head():
    model = TinyTargets()
    targets = collect_dense_target_weights(model, "all-linear")
    assert [target.name for target in targets] == ["block.q_proj"]


def test_collect_dense_target_weights_allows_explicit_lm_head():
    model = TinyTargets()
    targets = collect_dense_target_weights(model, ["lm_head"])
    assert [target.name for target in targets] == ["lm_head"]


def test_svd_step_adamw_applies_rank_limited_step_but_not_rank_limited_state():
    model = TinyTargets()
    targets = collect_dense_target_weights(model, ["q_proj"])
    target = targets[0]
    target.weight.data.zero_()
    target.base_weight = target.weight.detach().float().clone()
    optimizer = SVDStepAdamW(targets, rank=1, lr=1.0, betas=(0.0, 0.0), eps=1e-8)

    before = target.weight.detach().float().clone()
    set_single_entry_grad(target.weight, 0, 0)
    optimizer.step()
    first_step = target.weight.detach().float() - before
    assert matrix_rank(first_step) <= 1

    before = target.weight.detach().float().clone()
    set_single_entry_grad(target.weight, 1, 1)
    optimizer.step()
    second_step = target.weight.detach().float() - before
    assert matrix_rank(second_step) <= 1
    assert matrix_rank(target.weight.detach().float() - target.base_weight) == 2


def test_svd_cumulative_adamw_keeps_displacement_rank_limited_and_relative_to_base():
    model = TinyTargets()
    targets = collect_dense_target_weights(model, ["q_proj"])
    target = targets[0]
    target.weight.data.fill_(5.0)
    target.base_weight = target.weight.detach().float().clone()
    optimizer = SVDCumulativeAdamW(targets, rank=1, lr=1.0, betas=(0.0, 0.0), eps=1e-8)

    set_single_entry_grad(target.weight, 0, 0)
    optimizer.step()
    assert torch.allclose(target.weight.detach().float()[0, 1], target.base_weight[0, 1])

    set_single_entry_grad(target.weight, 1, 1)
    optimizer.step()
    displacement = target.weight.detach().float() - target.base_weight
    assert matrix_rank(displacement) <= 1
