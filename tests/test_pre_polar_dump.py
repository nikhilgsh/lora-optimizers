"""Tests for the pre-polar (H) dump on CurvatureWhitenLoRA.

The dump saves the matrices ``msign`` is applied to — ``zA``/``zB`` at the
``_polar_ns_guarded`` call sites — so ``lora_playground.lmo_diagnostics`` can
score cheap substitutes for ``msign`` offline. Two properties matter:

  1. it produces the artifacts it claims to (count them, don't trust exit 0);
  2. it does not perturb the update — dumping on must be bit-identical to off.
"""
import glob
import os

import pytest
import torch
import torch.nn as nn

from lora_playground.lmo_diagnostics import lmo_scores
from lora_playground.optim import CurvatureWhitenLoRA


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        torch.nn.init.normal_(self.lora_B["default"].weight, std=0.05)

    def forward(self, x):
        return x @ self.lora_A["default"].weight.T @ self.lora_B["default"].weight.T


class _TinyLoRAModel(nn.Module):
    def __init__(self, d_in=8, d_out=6, r=4):
        super().__init__()
        self.l0 = _FakeLoRALinear(d_in, d_out, r)
        self.l1 = _FakeLoRALinear(d_out, d_in, r)

    def forward(self, x):
        return self.l1(self.l0(x))


def _make(seed=0):
    torch.manual_seed(seed)
    return _TinyLoRAModel(), torch.randn(3, 8), torch.randn(3, 8)


def _proto(model, batched, **kw):
    """The protagonist's identity flags (kl-diag-polar-lora), plus overrides."""
    opt = CurvatureWhitenLoRA(
        model, lr=1e-2, kl_coupled=True, soap_v=False, diag_metric=True,
        use_polar=True, cw_nesterov=True, polar_method="polar_express",
        precond_method="gram_ns", ns_steps=8, **kw)
    opt._batched_step = batched
    return opt


def _run(opt, model, x, target, steps):
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), target).backward()
        opt.step()


@pytest.mark.parametrize("batched", [True, False])
def test_dump_writes_expected_artifact_count(tmp_path, batched):
    """4 pairs, cadence 2, 4 steps -> one file per (dumped step, selected pair)."""
    m, x, target = _make()
    d = tmp_path / "H"
    opt = _proto(m, batched, dump_pre_polar_dir=str(d),
                 dump_pre_polar_every=2, dump_pre_polar_max_pairs=2)
    n_pairs = len(opt._dump_pair_idxs)
    assert n_pairs == 2
    _run(opt, m, x, target, steps=4)
    files = sorted(glob.glob(os.path.join(str(d), "*.pt")))
    # steps 2 and 4 fire (step counter is 1-based at the dump site).
    assert len(files) == 2 * n_pairs, files
    rec = torch.load(files[0], weights_only=False)
    assert rec["zA"].shape[0] == 4 and rec["zA"].dtype == torch.float32   # (r, d_in)
    assert rec["zB"].shape[1] == 4                                        # (d_out, r)
    # collect_lora_pairs_named labels each pair "<module path>[<adapter>]".
    assert rec["pair_name"] in {"l0[default]", "l1[default]"}, rec["pair_name"]
    assert rec["optimizer_hparams"]["precond_method"] == "gram_ns"


def test_dump_does_not_change_the_update(tmp_path):
    """Dumping is diagnostic: params must match bit-for-bit with it off."""
    m0, x, target = _make(seed=1)
    opt0 = _proto(m0, True)
    _run(opt0, m0, x, target, steps=3)

    m1, _, _ = _make(seed=1)
    opt1 = _proto(m1, True, dump_pre_polar_dir=str(tmp_path / "H"),
                 dump_pre_polar_every=1)
    _run(opt1, m1, x, target, steps=3)

    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(p0, p1)
    assert glob.glob(os.path.join(str(tmp_path / "H"), "*.pt"))


def test_dumped_H_feeds_lmo_scores(tmp_path):
    """The artifact is directly consumable by the scorer — the whole point."""
    m, x, target = _make(seed=2)
    d = tmp_path / "H"
    opt = _proto(m, True, dump_pre_polar_dir=str(d), dump_pre_polar_every=1,
                 dump_pre_polar_max_pairs=1)
    _run(opt, m, x, target, steps=1)
    rec = torch.load(sorted(glob.glob(os.path.join(str(d), "*.pt")))[0],
                     weights_only=False)
    for key in ("zA", "zB"):
        out = lmo_scores(rec[key], polar_ks=(1, 8), reg_alg1_iters=(1,))
        assert abs(out["rho_msign"] - 1.0) < 1e-9
        assert 0.0 <= out["rho_reg_oneside"] <= 1.0 + 1e-9


def test_selecting_pairs_by_name_substring(tmp_path):
    m, _, _ = _make()
    opt = _proto(m, True, dump_pre_polar_dir=str(tmp_path / "H"),
                 dump_pre_polar_every=1, dump_pre_polar_pairs="l1")
    names = [opt._dump_pair_names[i] for i in sorted(opt._dump_pair_idxs)]
    assert names and all("l1" in n for n in names)


def test_unmatched_pair_filter_raises(tmp_path):
    m, _, _ = _make()
    with pytest.raises(ValueError, match="matched no LoRA pair"):
        _proto(m, True, dump_pre_polar_dir=str(tmp_path / "H"),
               dump_pre_polar_every=1, dump_pre_polar_pairs="no_such_module")


def test_cadence_without_dir_raises():
    m, _, _ = _make()
    with pytest.raises(ValueError, match="requires dump_pre_polar_dir"):
        _proto(m, True, dump_pre_polar_every=10)


def test_off_by_default_writes_nothing(tmp_path):
    m, x, target = _make()
    opt = _proto(m, True)
    assert opt.dump_pre_polar_every == 0
    assert opt._dump_pair_idxs is None
    _run(opt, m, x, target, steps=2)
    assert not glob.glob(os.path.join(str(tmp_path), "**", "*.pt"), recursive=True)
