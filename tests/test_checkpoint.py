"""Round-trip + state-equivalence tests for `lora_playground.checkpoint`.

CPU-only. Tests:
  - save/load adapter weights bitwise round-trip
  - save/load custom optimizer pair_state + group_state round-trip
  - save/load process RNG state round-trip for debug replay
  - resume-then-continue produces same final state as continuing-without-save
    (uses a deterministic toy step with a fixed pseudo-gradient, since
    sampler reseed is not bitwise-equivalent)
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.checkpoint import (
    _pair_state_aliases,
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


def _assert_nested_equal(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{label} type mismatch"
        assert torch.equal(actual, expected), f"{label} tensor mismatch"
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{label} type mismatch"
        assert len(actual) == len(expected), f"{label} length mismatch"
        for i, (a_item, e_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(a_item, e_item, f"{label}[{i}]")
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple), f"{label} type mismatch"
        assert len(actual) == len(expected), f"{label} length mismatch"
        for i, (a_item, e_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(a_item, e_item, f"{label}[{i}]")
    else:
        assert actual == expected, f"{label} scalar mismatch"


def _next_rng_draws():
    draws = {
        "python": [random.random() for _ in range(3)],
        "numpy": np.random.random(3),
        "torch_cpu": torch.rand(3),
    }
    if torch.cuda.is_available():
        draws["torch_cuda"] = [
            torch.rand(3, device=f"cuda:{i}").cpu()
            for i in range(torch.cuda.device_count())
        ]
    return draws


def _assert_rng_draws_equal(actual, expected):
    assert actual["python"] == expected["python"]
    assert np.array_equal(actual["numpy"], expected["numpy"])
    assert torch.equal(actual["torch_cpu"], expected["torch_cpu"])
    if "torch_cuda" in expected:
        assert len(actual["torch_cuda"]) == len(expected["torch_cuda"])
        for a, e in zip(actual["torch_cuda"], expected["torch_cuda"]):
            assert torch.equal(a, e)


@pytest.mark.parametrize("optimizer_type", [
    "adam-lin-lora",
    "adam-polar-product-lora-coupled-spectral-chord-tight",
    # AdamSOAPPolarProductLoRA: keeps `L_A`/`R_B`/`Q_A`/`Q_B` with their
    # original SOAP meanings, i.e. the spellings CurvatureWhitenLoRA retired.
    # Its state must survive a real save/load with those keys untranslated —
    # the end-to-end guard on the rename shim never firing on the wrong class.
    "adam-soap-polar-product-lora",
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


def test_checkpoint_round_trips_explicit_attempt_lineage_metadata(tmp_path):
    model = _PeftLikeWrapper(_ToyModel())
    optimizer = _make_optimizer(model.inner, "adam-lin-lora")
    ckpt = ckpt_dir_for_step(tmp_path, step=1)

    save_checkpoint(
        ckpt,
        bare_model=model,
        optimizer=optimizer,
        scheduler=None,
        step=1,
        total_tokens=42,
        resume_segment=0,
        cfg_snapshot={"command": "test"},
        attempt_id="attempt-a",
        checkpoint_identity="sweep/task-0",
    )
    fresh_model = _PeftLikeWrapper(_ToyModel())
    fresh_optimizer = _make_optimizer(fresh_model.inner, "adam-lin-lora")

    info = load_checkpoint(
        ckpt,
        bare_model=fresh_model,
        optimizer=fresh_optimizer,
    )

    assert info["checkpoint_meta_schema_version"] == 2
    assert info["attempt_id"] == "attempt-a"
    assert info["checkpoint_identity"] == "sweep/task-0"


def test_save_load_round_trips_adaptive_kappa_transient_group_state(tmp_path):
    """Replay checkpoints must preserve adaptive-kappa warm starts.

    These are transient group-state entries, but they affect the next step:
    kpar refreshes warm-start from the previous cached c, and chord-tight-clean
    uses per-Picard sigma power-iter warm starts. Dropping them makes a resumed
    debug replay close but not identical to the uninterrupted run.
    """
    torch.manual_seed(0)
    src_model = _PeftLikeWrapper(_ToyModel())
    src_opt = _make_optimizer(
        src_model.inner,
        "adam-polar-product-lora-coupled-spectral-chord-tight",
    )
    gs = src_opt.group_state[0]
    n_pairs = gs["N"]
    r = gs["r"]
    expected = {
        "ssc_c_cached_A_n0": torch.linspace(0.1, 0.2, n_pairs),
        "ssc_c_cached_B_n1": torch.linspace(0.3, 0.4, n_pairs),
        "ssc_c_cached_A_step": 17,
        "ssc_c_cached_B_step": 17,
        "ssc_c_last_A": torch.linspace(0.5, 0.6, n_pairs),
        "ssc_c_last_B": torch.linspace(0.7, 0.8, n_pairs),
        "v_op_geoA_slots": [
            torch.full((n_pairs, r), 1.25),
            None,
            torch.full((n_pairs, r), 2.5),
        ],
        "v_op_geoB_slots": [
            torch.full((n_pairs, r), -1.25),
            None,
            torch.full((n_pairs, r), -2.5),
        ],
    }
    gs.update(expected)

    ckpt = ckpt_dir_for_step(tmp_path, step=17)
    save_checkpoint(
        ckpt,
        bare_model=src_model,
        optimizer=src_opt,
        scheduler=None,
        step=17,
        total_tokens=123,
        resume_segment=0,
        cfg_snapshot={"command": "test"},
    )

    torch.manual_seed(99)
    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(
        dst_model.inner,
        "adam-polar-product-lora-coupled-spectral-chord-tight",
    )
    load_checkpoint(ckpt, bare_model=dst_model, optimizer=dst_opt)

    dst_gs = dst_opt.group_state[0]
    for key, expected_value in expected.items():
        assert key in dst_gs, f"missing restored group_state key {key}"
        _assert_nested_equal(dst_gs[key], expected_value, f"group_state[{key}]")


def test_save_load_round_trips_rng_state_for_debug_replay(tmp_path):
    """Opt-in debug replay can restore Python/NumPy/torch RNG state."""
    torch.manual_seed(0)
    src_model = _PeftLikeWrapper(_ToyModel())
    src_opt = _make_optimizer(src_model.inner, "adam-lin-lora")

    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(987)

    ckpt = ckpt_dir_for_step(tmp_path, step=1)
    save_checkpoint(
        ckpt,
        bare_model=src_model,
        optimizer=src_opt,
        scheduler=None,
        step=1,
        total_tokens=0,
        resume_segment=0,
        cfg_snapshot={"command": "test"},
    )
    expected = _next_rng_draws()

    # Perturb every RNG before load; restore_rng=True should put all streams
    # back at the checkpoint boundary.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, "adam-lin-lora")
    info = load_checkpoint(
        ckpt,
        bare_model=dst_model,
        optimizer=dst_opt,
        scheduler=None,
        restore_rng=True,
    )
    assert info["rng_state_present"] is True
    assert info["rng_restore_status"]["python"] is True
    assert info["rng_restore_status"]["numpy"] is True
    assert info["rng_restore_status"]["torch_cpu"] is True

    payload = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False)
    assert "rng_state" in payload
    if torch.cuda.is_available():
        assert len(payload["rng_state"]["torch_cuda"]) == torch.cuda.device_count()
    else:
        assert payload["rng_state"]["torch_cuda"] is None

    actual = _next_rng_draws()
    _assert_rng_draws_equal(actual, expected)


def test_load_checkpoint_without_rng_state_still_loads(tmp_path):
    """Back-compat: old checkpoints predate RNG payloads."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, "adam-lin-lora")
    ckpt = ckpt_dir_for_step(tmp_path, step=1)
    save_checkpoint(
        ckpt,
        bare_model=model,
        optimizer=opt,
        scheduler=None,
        step=1,
        total_tokens=0,
        resume_segment=0,
        cfg_snapshot={},
    )

    optim_path = ckpt / "optimizer.pt"
    payload = torch.load(optim_path, map_location="cpu", weights_only=False)
    payload.pop("rng_state", None)
    torch.save(payload, optim_path)

    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, "adam-lin-lora")
    info = load_checkpoint(
        ckpt,
        bare_model=dst_model,
        optimizer=dst_opt,
        scheduler=None,
        restore_rng=True,
    )
    assert info is not None
    assert info["step"] == 1
    assert info["rng_state_present"] is False
    assert info["rng_restore_status"]["python"] is False
    assert info["rng_restore_status"]["numpy"] is False
    assert info["rng_restore_status"]["torch_cpu"] is False


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


def test_loader_does_not_merge_unversioned_resume_siblings(tmp_path):
    """Filename-related physical logs do not imply a resume lineage."""
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
    loaded_cfg, evs = runs[0]
    steps = [e["step"] for e in evs]
    assert steps == [1200, 1400, 1600, 1800, 2000]
    assert loaded_cfg["_log_filename"] == "log_00.out"
    assert evs[-1].get("resume_segment") == 1


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


# ── pair_state key rename: old checkpoints must still resume ────────────────
#
# `pair_state` keys are persisted verbatim ("persist EVERY entry"), so renaming
# one strands every checkpoint written before the rename — silently, since load
# simply inserts the unknown key and leaves the live buffer at its init value.
# `CurvatureWhitenLoRA`'s r x r slots `L_A`/`R_B` and eigenbasis `Q_A`/`Q_B`
# became `P_A`/`Q_B` and `U_A`/`U_B`; `checkpoint._pair_state_aliases` supplies
# the translation and `load_checkpoint` applies it.
#
# `tests/test_checkpoint_pair_state_renames.py` unit-tests the mapping function.
# These three go through the real save/load path, which is what pins the WIRING:
# that the aliases reach the load loop at all, and that a resume from a
# pre-rename checkpoint stays on the same trajectory.

_CW_OPT = "kl-diag-polar-lora"


def _downgrade_pair_state_keys(ckpt_dir, aliases):
    """Rewrite a saved checkpoint's pair_state keys to their pre-rename names.

    Faithful to a real old checkpoint because the rename changed key STRINGS
    only: the tensor written under 'L_A' by the old code is the tensor written
    under 'P_A' by the new one. Applied as one simultaneous permutation for the
    same reason load applies it that way — `Q_B` is both a retired name and a
    current one. Returns the number of keys downgraded.
    """
    path = Path(ckpt_dir) / "optimizer.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    back = {new: old for old, new in aliases.items()}
    assert len(back) == len(aliases), "alias map is not invertible"
    n = 0
    for i, entry in payload["pair_state"].items():
        renamed = {}
        for k, v in entry.items():
            if k in back:
                n += 1
            renamed[back.get(k, k)] = v
        payload["pair_state"][i] = renamed
    torch.save(payload, path)
    return n


def _expected_downgrade_count(optimizer, aliases):
    """How many keys `_downgrade_pair_state_keys` should rewrite, read off the
    LIVE optimizer rather than hand-typed, so a pair_state change cannot leave
    the assertion below checking a schema that no longer exists."""
    targets = set(aliases.values())
    return sum(len(targets & set(e)) for e in optimizer.pair_state.values())


def _save_cw_checkpoint(tmp_path, steps=3):
    """Train a CurvatureWhitenLoRA for `steps`, save, and downgrade the saved
    pair_state keys to the pre-rename spelling.

    Returns (model, optimizer, ckpt_dir, aliases)."""
    from lora_playground.optim import CurvatureWhitenLoRA

    torch.manual_seed(0)
    src_model = _PeftLikeWrapper(_ToyModel())
    src_opt = _make_optimizer(src_model.inner, _CW_OPT)
    assert isinstance(src_opt, CurvatureWhitenLoRA)

    x = torch.randn(2, 8)
    y = torch.randn(2, 8)
    for _ in range(steps):
        _step_once(src_model.inner, src_opt, x, y)

    ckpt = ckpt_dir_for_step(tmp_path, step=steps)
    save_checkpoint(
        ckpt, bare_model=src_model, optimizer=src_opt, scheduler=None,
        step=steps, total_tokens=0, resume_segment=0, cfg_snapshot={},
    )
    aliases = _pair_state_aliases(src_opt)
    expected = _expected_downgrade_count(src_opt, aliases)
    # Guards against a typo in the alias table making these tests vacuous: if
    # nothing was rewritten, the "old" checkpoint is just a current one.
    assert expected > 0, "live optimizer carries none of the renamed keys"
    n = _downgrade_pair_state_keys(ckpt, aliases)
    assert n == expected, f"downgraded {n} keys, expected {expected}"
    return src_model, src_opt, ckpt, aliases


def test_old_pair_state_names_load_into_the_renamed_optimizer(tmp_path):
    """A checkpoint written with 'L_A'/'R_B'/'Q_A'/'Q_B' must restore into
    'P_A'/'Q_B'/'U_A'/'U_B' with every tensor intact."""
    _src_model, src_opt, ckpt, aliases = _save_cw_checkpoint(tmp_path)

    torch.manual_seed(99)   # different weights before load
    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, _CW_OPT)
    live_before = {i: set(e) for i, e in dst_opt.pair_state.items()}

    info = load_checkpoint(ckpt, bare_model=dst_model, optimizer=dst_opt)
    assert info is not None

    # No retired key survived into the live state, and every current key matches.
    for i, entry in dst_opt.pair_state.items():
        retired = set(aliases) - live_before[i]
        leaked = retired & set(entry)
        assert not leaked, f"pair {i} kept a pre-rename key: {sorted(leaked)}"
    _equal_state(src_opt, dst_opt)


def test_the_alias_map_is_what_makes_the_old_checkpoint_load(tmp_path, monkeypatch):
    """Negative control: with the alias table emptied, the same load leaves the
    curvature state at its init value instead of erroring. That silence is
    exactly why the table has to exist."""
    from lora_playground.optim import CurvatureWhitenLoRA

    _src_model, src_opt, ckpt, aliases = _save_cw_checkpoint(tmp_path)

    # Pick a renamed key to watch, derived from the live optimizer:
    #   - `new` must be live, or there is no slot to check;
    #   - `new` must not itself be a retired name, or the un-aliased load would
    #     write some OTHER tensor into it and blur the two failure modes;
    #   - the trained value must differ from the fresh one, or a failed load is
    #     indistinguishable from a successful one.
    torch.manual_seed(99)
    probe_opt = _make_optimizer(_PeftLikeWrapper(_ToyModel()).inner, _CW_OPT)
    fresh, trained = probe_opt.pair_state[0], src_opt.pair_state[0]
    watched = [(o, n) for o, n in aliases.items()
               if n in fresh and n not in aliases
               and not torch.equal(trained[n], fresh[n])]
    assert watched, "no renamed key moved during training — test is vacuous"
    old_name, new_name = watched[0]

    # An explicitly EMPTY class attribute: a declared map wins over the
    # temporary module-level fallback, so this really does disable translation.
    monkeypatch.setattr(
        CurvatureWhitenLoRA, "PAIR_STATE_ALIASES", {}, raising=False)

    torch.manual_seed(99)
    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, _CW_OPT)
    load_checkpoint(ckpt, bare_model=dst_model, optimizer=dst_opt)

    assert not torch.equal(dst_opt.pair_state[0][new_name], trained[new_name]), \
        f"without the alias map {old_name!r} should NOT have reached {new_name!r}"
    assert torch.equal(dst_opt.pair_state[0][old_name], trained[new_name]), \
        f"the un-aliased {old_name!r} should have been kept verbatim instead"


def test_old_checkpoint_resume_then_step_matches_no_resume(tmp_path):
    """The operational claim: a sweep resuming from a pre-rename checkpoint
    continues the SAME trajectory it would have had without the interruption."""
    K, J = 3, 4
    input_gen = torch.Generator().manual_seed(42)
    x = torch.randn(2, 8, generator=input_gen)
    y = torch.randn(2, 8, generator=input_gen)

    torch.manual_seed(0)
    ref_model = _PeftLikeWrapper(_ToyModel())
    ref_opt = _make_optimizer(ref_model.inner, _CW_OPT)
    for _ in range(K + J):
        _step_once(ref_model.inner, ref_opt, x, y)

    torch.manual_seed(0)
    trial_model = _PeftLikeWrapper(_ToyModel())
    trial_opt = _make_optimizer(trial_model.inner, _CW_OPT)
    for _ in range(K):
        _step_once(trial_model.inner, trial_opt, x, y)
    ckpt = ckpt_dir_for_step(tmp_path, step=K)
    save_checkpoint(
        ckpt, bare_model=trial_model, optimizer=trial_opt, scheduler=None,
        step=K, total_tokens=0, resume_segment=0, cfg_snapshot={},
    )
    aliases = _pair_state_aliases(trial_opt)
    assert _downgrade_pair_state_keys(ckpt, aliases) > 0

    del trial_model, trial_opt
    torch.manual_seed(0)
    resumed_model = _PeftLikeWrapper(_ToyModel())
    resumed_opt = _make_optimizer(resumed_model.inner, _CW_OPT)
    load_checkpoint(tmp_path, bare_model=resumed_model, optimizer=resumed_opt)
    for _ in range(J):
        _step_once(resumed_model.inner, resumed_opt, x, y)

    ref_params = dict(ref_model.inner.named_parameters())
    for k, v in resumed_model.inner.named_parameters():
        assert torch.allclose(v, ref_params[k], atol=1e-6), \
            f"final param {k!r} drifted: max abs diff = " \
            f"{(v - ref_params[k]).abs().max().item():.3e}"
