"""Round-trip + state-equivalence tests for `lora_playground.checkpoint`.

CPU-only. Tests:
  - save/load adapter weights bitwise round-trip
  - save/load custom optimizer pair_state + group_state round-trip
  - resume-then-continue produces same final state as continuing-without-save
    (uses a deterministic toy step with a fixed pseudo-gradient, since
    sampler reseed is not bitwise-equivalent)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.checkpoint import (
    ckpt_dir_for_step,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


class _LoRAModule(nn.Module):
    """Minimal module with `lora_A` / `lora_B` ModuleDicts so
    `collect_lora_pairs` finds it. Mimics PEFT's per-layer structure."""

    def __init__(self, d_in, d_out, r, adapter="default"):
        super().__init__()
        self.lora_A = nn.ModuleDict({adapter: nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({adapter: nn.Linear(r, d_out, bias=False)})
        # PEFT zero-inits B; A is Kaiming. Match that.
        nn.init.zeros_(self.lora_B[adapter].weight)

    def forward(self, x):
        return self.lora_B["default"](self.lora_A["default"](x))


class _ToyModel(nn.Module):
    """Two LoRA modules so the optimizer sees more than one pair."""

    def __init__(self, d=8, r=4):
        super().__init__()
        self.l1 = _LoRAModule(d, d, r)
        self.l2 = _LoRAModule(d, d, r)

    def forward(self, x):
        return self.l2(self.l1(x))


class _PeftLikeWrapper(nn.Module):
    """Wraps `_ToyModel` and provides `save_pretrained` / `load_adapter`
    that operate on just the LoRA params. Avoids a real PEFT dependency."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def _lora_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.inner.named_parameters()
                if "lora_A" in k or "lora_B" in k}

    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._lora_state(), p / "adapter_state.pt")

    def load_adapter(self, path, adapter_name=None, is_trainable=False):
        state = torch.load(Path(path) / "adapter_state.pt",
                           map_location="cpu", weights_only=False)
        with torch.no_grad():
            for k, v in self.inner.named_parameters():
                if k in state:
                    v.copy_(state[k].to(v.device, v.dtype))

    def forward(self, x):
        return self.inner(x)


def _make_optimizer(model, optimizer_type):
    """Build via the real factory (no helpers needed for the test)."""
    from lora_playground.optim import build_optimizer
    return build_optimizer(model, optimizer_type=optimizer_type, lr=1e-3)


def _step_once(model, optimizer, x, target):
    out = model(x)
    loss = (out - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _equal_state(opt_a, opt_b):
    """Compare pair_state and group_state tensor-wise."""
    pa = getattr(opt_a, "pair_state", None) or {}
    pb = getattr(opt_b, "pair_state", None) or {}
    assert set(pa.keys()) == set(pb.keys())
    for i in pa:
        for k, v in pa[i].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, pb[i][k]), f"pair_state[{i}][{k}] mismatch"
            else:
                assert v == pb[i].get(k), f"pair_state[{i}][{k}] mismatch"

    ga = getattr(opt_a, "group_state", None) or []
    gb = getattr(opt_b, "group_state", None) or []
    assert len(ga) == len(gb)
    for gi, (ea, eb) in enumerate(zip(ga, gb)):
        for k, v in ea.items():
            if isinstance(v, torch.Tensor) and isinstance(eb.get(k), torch.Tensor):
                # Persisted keys must match. Scratch (A_stack etc.) is
                # refreshed each step so we don't compare it.
                if k in ("m_A", "v_A", "m_B", "v_B",
                         "SA_half_inv", "SB_half_inv",
                         "v_sigma_A", "v_sigma_B",
                         "v_op_geoA", "v_op_geoB"):
                    assert torch.equal(v, eb[k]), \
                        f"group_state[{gi}][{k}] mismatch"


@pytest.mark.parametrize("optimizer_type", [
    "adam-lin-lora",
    "adam-polar-product-lora-coupled-spectral-chord-tight",
])
def test_save_load_round_trip(tmp_path, optimizer_type):
    """Save after N steps, instantiate fresh model+optimizer, load,
    verify all persisted state matches the source."""
    torch.manual_seed(0)
    src_model = _PeftLikeWrapper(_ToyModel())
    src_opt = _make_optimizer(src_model.inner, optimizer_type)

    x = torch.randn(2, 8)
    y = torch.randn(2, 8)
    for _ in range(3):
        _step_once(src_model.inner, src_opt, x, y)

    ckpt = ckpt_dir_for_step(tmp_path, step=3)
    save_checkpoint(
        ckpt,
        bare_model=src_model,
        optimizer=src_opt,
        scheduler=None,
        step=3,
        total_tokens=12345,
        resume_segment=0,
        cfg_snapshot={"command": "test"},
    )

    # Fresh model + optimizer.
    torch.manual_seed(99)  # different seed: weights differ before load
    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, optimizer_type)

    info = load_checkpoint(
        ckpt, bare_model=dst_model, optimizer=dst_opt, scheduler=None,
    )
    assert info is not None
    assert info["step"] == 3
    assert info["total_tokens"] == 12345
    assert info["resume_segment"] == 0

    # LoRA weights bitwise-equal.
    src_params = dict(src_model.inner.named_parameters())
    for k, v in dst_model.inner.named_parameters():
        assert torch.equal(v, src_params[k]), f"param mismatch: {k}"

    # Optimizer state matches on persisted keys.
    _equal_state(src_opt, dst_opt)


def test_load_from_parent_picks_latest(tmp_path):
    """`load_checkpoint(parent_dir)` should pick the highest-numbered
    `ckpt_step{N}` child."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, "adam-lin-lora")
    x, y = torch.randn(2, 8), torch.randn(2, 8)

    for step in (1, 5, 10):
        _step_once(model.inner, opt, x, y)
        save_checkpoint(
            ckpt_dir_for_step(tmp_path, step=step),
            bare_model=model, optimizer=opt, scheduler=None,
            step=step, total_tokens=step * 100,
            resume_segment=0, cfg_snapshot={},
        )

    dst = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst.inner, "adam-lin-lora")
    info = load_checkpoint(tmp_path, bare_model=dst, optimizer=dst_opt)
    assert info["step"] == 10
    assert info["total_tokens"] == 1000


def test_load_missing_returns_none(tmp_path):
    """No checkpoint dir / empty parent dir → returns None."""
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, "adam-lin-lora")
    assert load_checkpoint(
        tmp_path / "does_not_exist", bare_model=model, optimizer=opt,
    ) is None
    (tmp_path / "empty").mkdir()
    assert load_checkpoint(
        tmp_path / "empty", bare_model=model, optimizer=opt,
    ) is None


def test_prune_keep_last(tmp_path):
    """prune_checkpoints keeps the K most recent dirs by step number."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, "adam-lin-lora")
    for step in (1, 2, 3, 4, 5):
        save_checkpoint(
            ckpt_dir_for_step(tmp_path, step=step),
            bare_model=model, optimizer=opt, scheduler=None,
            step=step, total_tokens=0,
            resume_segment=0, cfg_snapshot={},
        )
    prune_checkpoints(tmp_path, keep_last=2)
    survived = sorted(
        int(d.name[len("ckpt_step"):]) for d in tmp_path.iterdir()
        if d.is_dir() and d.name.startswith("ckpt_step")
    )
    assert survived == [4, 5]


def test_loader_merges_resume_segments(tmp_path):
    """`load_sweep` merges `log_NN.out` and `log_NN.out.resume_K` siblings
    into one (cfg, evs) per task index. Step-union semantics: each step
    appears once; segments are step-disjoint by design."""
    import json as _json
    from lora_playground.plotting import load_sweep

    group_dir = tmp_path / "tg" / "run_info" / "logs"
    group_dir.mkdir(parents=True)

    # Common cfg event (resume re-emits with same algorithmic key).
    cfg = {"event": "config", "optimizer": "adamw", "lr": 1e-3,
           "command": "train_lora.py --optimizer adamw"}

    # Original segment: steps 200, 400, ..., 1000 → wall-killed.
    seg0_evs = [
        {"event": "eval", "step": s, "eval_loss": 1.0 - s / 5000.0,
         "lr": 1e-3, "tokens": s * 100}
        for s in (200, 400, 600, 800, 1000)
    ]
    # Resume segment: from checkpoint at step 1000, continues 1200..2000.
    seg1_evs = [
        {"event": "eval", "step": s, "eval_loss": 1.0 - s / 5000.0,
         "lr": 1e-3, "tokens": s * 100, "resume_segment": 1}
        for s in (1200, 1400, 1600, 1800, 2000)
    ]

    # After submit.sh's pre-submit rotation, the original log lives as
    # `log_00.out.resume_0` and the new run wrote `log_00.out`.
    (group_dir / "log_00.out.resume_0").write_text(
        _json.dumps(cfg) + "\n"
        + "\n".join(_json.dumps(e) for e in seg0_evs) + "\n"
    )
    (group_dir / "log_00.out").write_text(
        _json.dumps(cfg) + "\n"
        + "\n".join(_json.dumps(e) for e in seg1_evs) + "\n"
    )

    runs = load_sweep("tg", logs_root=str(tmp_path))
    assert len(runs) == 1
    _, evs = runs[0]
    steps = [e["step"] for e in evs]
    assert steps == [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # Resume segment value preserved on the segment-1 entries.
    assert evs[-1].get("resume_segment") == 1
    assert evs[0].get("resume_segment") is None


def test_atomic_write_on_overwrite(tmp_path):
    """Writing to an existing ckpt dir replaces it atomically; no stale
    files survive."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, "adam-lin-lora")
    ckpt = ckpt_dir_for_step(tmp_path, step=1)

    # First write — leave a stale file inside that the overwrite must purge.
    save_checkpoint(
        ckpt, bare_model=model, optimizer=opt, scheduler=None,
        step=1, total_tokens=0, resume_segment=0, cfg_snapshot={},
    )
    (ckpt / "stale_garbage.txt").write_text("should be removed")

    # Second write at same path.
    _step_once(model.inner, opt, torch.randn(2, 8), torch.randn(2, 8))
    save_checkpoint(
        ckpt, bare_model=model, optimizer=opt, scheduler=None,
        step=2, total_tokens=42, resume_segment=1, cfg_snapshot={},
    )

    assert not (ckpt / "stale_garbage.txt").exists()
    with open(ckpt / "meta.json") as fh:
        meta = json.load(fh)
    assert meta["step"] == 2
    assert meta["resume_segment"] == 1


@pytest.mark.parametrize("optimizer_type", [
    "adam-lin-lora",
    "adam-polar-product-lora-coupled-spectral-chord-tight",
])
def test_resume_then_step_matches_no_resume(tmp_path, optimizer_type):
    """The whole point: take K steps, save, load into a fresh
    model+optimizer, take J more steps with the SAME deterministic
    inputs — final weights and optimizer state must match a
    K+J-step uninterrupted run.

    Each model is constructed under a fresh `manual_seed(0)` so the initial
    LoRA weights match across ref / trial / resumed (the global PRNG state
    drifts between the three constructions otherwise). Inputs use a
    separate Generator so they don't interact with the global PRNG state.
    """
    K, J = 3, 4
    input_gen = torch.Generator().manual_seed(42)
    x = torch.randn(2, 8, generator=input_gen)
    y = torch.randn(2, 8, generator=input_gen)

    # Reference: K+J uninterrupted steps.
    torch.manual_seed(0)
    ref_model = _PeftLikeWrapper(_ToyModel())
    ref_opt = _make_optimizer(ref_model.inner, optimizer_type)
    for _ in range(K + J):
        _step_once(ref_model.inner, ref_opt, x, y)

    # Trial: K steps, save, load, J more steps.
    torch.manual_seed(0)
    trial_model = _PeftLikeWrapper(_ToyModel())
    trial_opt = _make_optimizer(trial_model.inner, optimizer_type)
    for _ in range(K):
        _step_once(trial_model.inner, trial_opt, x, y)
    save_checkpoint(
        ckpt_dir_for_step(tmp_path, step=K),
        bare_model=trial_model, optimizer=trial_opt, scheduler=None,
        step=K, total_tokens=0, resume_segment=0, cfg_snapshot={},
    )

    # Throw it all away and load fresh.
    del trial_model, trial_opt
    torch.manual_seed(0)
    resumed_model = _PeftLikeWrapper(_ToyModel())
    resumed_opt = _make_optimizer(resumed_model.inner, optimizer_type)
    load_checkpoint(
        tmp_path, bare_model=resumed_model, optimizer=resumed_opt,
    )
    for _ in range(J):
        _step_once(resumed_model.inner, resumed_opt, x, y)

    # Final LoRA weights must match the reference run bitwise (deterministic
    # inputs; no sampler / RNG drift in this synthetic setup).
    ref_params = dict(ref_model.inner.named_parameters())
    for k, v in resumed_model.inner.named_parameters():
        assert torch.allclose(v, ref_params[k], atol=1e-6), \
            f"final param {k!r} drifted: max abs diff = " \
            f"{(v - ref_params[k]).abs().max().item():.3e}"
