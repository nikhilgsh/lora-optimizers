"""Shared test fixtures.

Currently one thing: the batching-correctness property for
`CurvatureWhitenLoRA._cw_apply_grouped`, used by four test files that each used
to check it by comparing against `_cw_apply_per_pair` — a second implementation
of the same step, now deleted.
"""
import torch
import torch.nn as nn
import pytest

D_IN, D_OUT, R = 32, 16, 4


class _LoraLin(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})


class Alone(nn.Module):
    """One pair, so `_cw_apply_grouped` sees a single group of size 1."""

    def __init__(self):
        super().__init__()
        self.l1 = _LoraLin(D_IN, D_OUT, R)


class WithCompanion(nn.Module):
    """`l1` plus a shape-identical `l2`, so both land in one group of size 2.

    `l3` and `l4` differ in d_in / d_out and form their own singleton groups, so
    a run also exercises the multiple-groups loop rather than a single bucket.
    """

    def __init__(self):
        super().__init__()
        self.l1 = _LoraLin(D_IN, D_OUT, R)          # groups with l2
        self.l2 = _LoraLin(D_IN, D_OUT, R)
        self.l3 = _LoraLin(64, D_OUT, R)            # singleton (different d_in)
        self.l4 = _LoraLin(D_IN, 48, R)             # singleton (different d_out)


def assert_companion_independent(steps=5, **opt_kwargs):
    """`l1`'s trajectory must be IDENTICAL alone and beside a companion.

    `_cw_apply_grouped` buckets pairs by `(d_in, d_out)` and runs each bucket
    through one stacked `torch.stack` / bmm / batched-Newton-Schulz sequence. The
    risk that creates is a reduction across the BATCH axis that should have
    stayed per-item — a `sum()` missing its `dim=`, a `sigma_max` estimated over
    the stack instead of per matrix. Such a bug makes a pair's update depend on
    which OTHER pairs share its shape, which is exactly what this checks.

    The comparison is BIT-EXACT. Measured across eight optimizer configurations:
    the clean delta is 0.000e+00 in every one, because each item's math is its
    own slice of the bmm. An earlier version of this helper used atol=1e-4 and
    was a NON-DETECTOR — injecting the very bug it claims to catch
    (`_smax_warm` returning `s.max().expand_as(s)`, one sigma for the whole
    group) moved `l1` by 5.1e-5 and passed. A tolerance sized for comparing two
    implementations is far too loose for comparing one against itself.
    """
    from lora_playground.optim import CurvatureWhitenLoRA

    torch.manual_seed(0)
    solo_model, group_model = Alone(), WithCompanion()
    with torch.no_grad():                       # l1 must START identical in both
        for name in ("lora_A", "lora_B"):
            src = getattr(group_model.l1, name)["default"].weight
            getattr(solo_model.l1, name)["default"].weight.copy_(src)

    o_solo = CurvatureWhitenLoRA(solo_model, **opt_kwargs)
    o_group = CurvatureWhitenLoRA(group_model, **opt_kwargs)
    solo_pair, group_pair = o_solo.pairs[0], o_group.pairs[0]
    assert solo_pair[0].shape == group_pair[0].shape == (R, D_IN)
    assert torch.equal(solo_pair[0], group_pair[0]), "l1 A differs before step 1"
    assert torch.equal(solo_pair[1], group_pair[1]), "l1 B differs before step 1"

    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        # l1 gets the SAME gradient in both runs; the companions get their own,
        # which is the point -- their values must not reach l1's update.
        gA = torch.randn(solo_pair[0].shape, generator=g)
        gB = torch.randn(solo_pair[1].shape, generator=g)
        for A, B in o_solo.pairs:
            A.grad, B.grad = gA.clone(), gB.clone()
        for k, (A, B) in enumerate(o_group.pairs):
            if k == 0:
                A.grad, B.grad = gA.clone(), gB.clone()
            else:
                A.grad = torch.randn(A.shape, generator=g)
                B.grad = torch.randn(B.shape, generator=g)
        o_solo.step()
        o_group.step()

    for which, s, gp in (("A", solo_pair[0], group_pair[0]),
                         ("B", solo_pair[1], group_pair[1])):
        assert torch.isfinite(gp).all(), f"l1 {which}: non-finite in the group run"
        assert torch.isfinite(s).all(), f"l1 {which}: non-finite in the solo run"
        assert torch.equal(s, gp), (
            f"l1 {which} depends on its shape-group companions: "
            f"max|delta|={(s - gp).abs().max().item():.2e}, expected exactly 0. "
            f"A reduction in _cw_apply_grouped is crossing the batch axis.")


@pytest.fixture
def companion_independent():
    """`assert_companion_independent` as a fixture, for readability at call sites."""
    return assert_companion_independent
