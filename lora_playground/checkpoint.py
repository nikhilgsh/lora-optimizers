"""Step-continuous checkpoint save/load for the lora_playground train loop.

Saves PEFT adapter weights + optimizer state (including custom `pair_state` /
`group_state`) + scheduler + step / total_tokens metadata. Used by `train.py`
to resume from SLURM wall-timeouts.

Normal resume is not bitwise-resumable: by default the sampler is reseeded
with `(seed, step)` so the post-resume trajectory is stochastically continuous
but not identical to an uninterrupted run. Debug resumes can opt in to
restoring checkpointed RNG state and replaying the original dataloader stream.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# pair_state: persist EVERY entry. Each optimizer's pair_state schema is its
# own (Adam moments, per-pair step counters, side-channel raw-grad moments
# when diagnostics are on, etc.); listing keys here breaks any optimizer that
# adds new ones. Tensor values get detach/clone/CPU on save.
#
# group_state: persist only the explicit list below. Unlike pair_state, the
# batched optimizers stash scratch buffers (`A_stack`, `B_stack`, gradient
# stacks) and bookkeeping (`gid`, `indices`, shape tuples) here. Scratch is
# refreshed every step; bookkeeping is reconstructed from the live optimizer.
_GROUP_STATE_PERSIST = (
    "m_A", "v_A", "m_B", "v_B",
    "SA_half_inv", "SB_half_inv",
    "v_sigma_A", "v_sigma_B",
    "v_op_geoA", "v_op_geoB",
    "v_XA", "v_XB",
    "v_op_geoA_slots", "v_op_geoB_slots",
    "ssc_c_last_A", "ssc_c_last_B",
)
_GROUP_STATE_PERSIST_PREFIXES = (
    "ssc_c_cached_A",
    "ssc_c_cached_B",
)
# pair_state keys that hold tensor VIEWS into group_state buffers (and must
# not be re-pickled as independent storage). On save we clone them anyway;
# on load we copy_ into the live view, which writes through to the group
# buffer. No special exclusion required for the test-equality semantics.
_PAIR_STATE_SKIP_LOAD = {"_group", "_local_idx"}

# Renamed pair_state keys: old checkpoint key -> current key. Without a
# translation the load loop below takes its "key not in the live state" branch
# and INSERTS the stale key as a new entry, leaving the current buffer at its
# init value — a resume that silently starts its curvature EMAs from scratch
# rather than failing.
#
# The map is read off the OPTIMIZER (`PAIR_STATE_ALIASES` class attribute, see
# `_pair_state_aliases`), not kept as one module-level table, because the same
# key spelling means different things in different optimizers:
# `AdamSOAPPolarProductLoRA` uses `L_A`, `R_B`, `Q_A` and `Q_B` with their
# original SOAP meanings and must never be translated. Scoping the map to the
# class that did the renaming makes that collision unrepresentable instead of
# something a second table has to patch around.

# TEMPORARY — stands in for `CurvatureWhitenLoRA.PAIR_STATE_ALIASES`, which
# does not exist yet in lora_playground/optim.py. CurvatureWhitenLoRA's two
# r x r slots were `L_A`/`R_B` and its eigh eigenbasis `Q_A`/`Q_B`; the slots
# are now `P_A`/`Q_B` (the free Kronecker factors) and the eigenbasis
# `U_A`/`U_B`.
#
# REMOVAL CONDITION: the moment `CurvatureWhitenLoRA` declares
# `PAIR_STATE_ALIASES` with these same four pairs, delete this table, delete
# the fallback branch in `_pair_state_aliases`, make `aliases` a required
# argument of `_apply_pair_state_renames`, and delete
# `test_temporary_fallback_matches_the_class_attribute` in
# tests/test_checkpoint_pair_state_renames.py — that test is the tripwire: it
# skips while the attribute is absent and starts checking the two agree the
# moment it lands.
_PAIR_STATE_RENAMES_FALLBACK = {
    "L_A": "P_A", "R_B": "Q_B", "Q_A": "U_A", "Q_B": "U_B",
}


def _pair_state_aliases(optimizer) -> dict:
    """Old-name -> current-name map for THIS optimizer's `pair_state` keys.

    Because pair_state is persisted key by key, renaming a key would strand
    every checkpoint written before the rename — a running sweep would resume
    into a freshly-initialized buffer with no error. An optimizer that renames
    its state declares the translation as a `PAIR_STATE_ALIASES` class
    attribute and load applies it.

    Read off the optimizer instance rather than kept as a module-level table:
    different optimizers spell the same key name to mean different things, so
    the map has to be scoped to the class that renamed it. A declared attribute
    always wins, including an empty one.
    """
    aliases = getattr(optimizer, "PAIR_STATE_ALIASES", None)
    if aliases is not None:
        return dict(aliases)
    return dict(_PAIR_STATE_RENAMES_FALLBACK)   # temporary, see above


def _apply_pair_state_renames(entry, live_keys, aliases=None):
    """Map a loaded pair_state entry's keys onto the current schema.

    `aliases` is the optimizer's own old -> new map, from
    `_pair_state_aliases`; it defaults to the temporary fallback table above
    so the existing two-argument call sites keep their behaviour.

    The rename is applied as ONE SIMULTANEOUS PERMUTATION, never key by key,
    because a name can be retired and current at the same time: the old
    eigenbasis `Q_B` became `U_B` while `Q_B` is now the free Kronecker factor.
    A key-at-a-time "translate it only if the live state lacks it" rule would
    leave that tensor in the wrong slot.

    Whether the ENTRY is on the old schema is decided by the alias keys that
    are NOT live — derived from the map and the live optimizer rather than
    listed in a second table that can go stale. If no alias key is retired
    (`AdamSOAPPolarProductLoRA` under the fallback map has all four live)
    nothing distinguishes the two schemas, so the entry passes through
    untouched.
    """
    if aliases is None:
        aliases = _PAIR_STATE_RENAMES_FALLBACK
    if not aliases:
        return entry
    old_only = set(aliases) - set(live_keys)
    if not old_only or not (old_only & set(entry)):
        return entry
    return {aliases.get(k, k): v for k, v in entry.items()}


def _adapter_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "adapter"


def _optim_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "optimizer.pt"


def _meta_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "meta.json"


def _checkpoint_value(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone().cpu()
    if isinstance(v, list):
        return [_checkpoint_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_checkpoint_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _checkpoint_value(x) for k, x in v.items()}
    return v


def _restore_value(v, *, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, list):
        return [_restore_value(x, device=device) for x in v]
    if isinstance(v, tuple):
        return tuple(_restore_value(x, device=device) for x in v)
    if isinstance(v, dict):
        return {k: _restore_value(x, device=device) for k, x in v.items()}
    return v


def capture_rng_state() -> dict:
    """Capture process RNG state for opt-in bitwise-debug resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            s.detach().clone().cpu() for s in torch.cuda.get_rng_state_all()
        ]
    return _checkpoint_value(state)


def restore_rng_state(rng_state: Optional[dict]) -> dict:
    """Restore a state captured by `capture_rng_state`.

    Returns a small status dict for logging. Missing keys are tolerated so old
    checkpoints remain loadable.
    """
    status = {
        "python": False,
        "numpy": False,
        "torch_cpu": False,
        "torch_cuda": False,
        "torch_cuda_devices_restored": 0,
        "torch_cuda_devices_available": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "torch_cuda_devices_checkpointed": 0,
    }
    if not rng_state:
        return status

    if "python" in rng_state:
        random.setstate(rng_state["python"])
        status["python"] = True
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
        status["numpy"] = True
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        status["torch_cpu"] = True

    cuda_states = rng_state.get("torch_cuda")
    if cuda_states is not None:
        status["torch_cuda_devices_checkpointed"] = len(cuda_states)
    if cuda_states is not None and torch.cuda.is_available():
        n_restore = min(len(cuda_states), torch.cuda.device_count())
        for device_idx, cuda_state in enumerate(cuda_states[:n_restore]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_idx)
        status["torch_cuda_devices_restored"] = n_restore
        status["torch_cuda"] = n_restore == len(cuda_states) == torch.cuda.device_count()
    return status


def _keep_group_state_key(k, keep_keys):
    if keep_keys is None:
        return True
    return k in keep_keys or any(
        k.startswith(prefix) for prefix in _GROUP_STATE_PERSIST_PREFIXES
    )


def _filter_entries(entries, keep_keys=None):
    """Yield `(key, filtered_dict)` with tensor values detached/cloned/CPU.

    keep_keys=None means "keep all keys" (pair_state semantics — every key is
    meaningful per-optimizer state, no scratch). A non-None whitelist filters
    keys (group_state semantics — must drop scratch buffers).

    Cloning detaches views: without it, torch.save records the underlying
    storage backing every view in a shape-group at once and inflates the
    checkpoint by the size of the group's scratch buffers.
    """
    for key, entry in entries:
        kept = {}
        for k, v in entry.items():
            if not _keep_group_state_key(k, keep_keys):
                continue
            kept[k] = _checkpoint_value(v)
        yield key, kept


def save_checkpoint(
    ckpt_dir,
    *,
    bare_model,
    optimizer,
    scheduler,
    step: int,
    total_tokens: int,
    resume_segment: int,
    cfg_snapshot: dict,
) -> None:
    """Atomically save a step-continuous checkpoint.

    bare_model: the unwrapped PEFT model (uses save_pretrained for adapter).
    optimizer: standard PyTorch or one of the custom pair/group-state ones.
    """
    ckpt_dir = Path(ckpt_dir)
    tmp_dir = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. PEFT adapter weights only — base model frozen, not saved.
    bare_model.save_pretrained(str(_adapter_dir(tmp_dir)))

    # 2. Optimizer state — stock state_dict + custom pair/group state.
    state_payload: dict = {"pytorch_state": optimizer.state_dict()}
    if getattr(optimizer, "pair_state", None):
        # Save EVERY key in each pair entry (schema is optimizer-specific).
        state_payload["pair_state"] = dict(
            _filter_entries(optimizer.pair_state.items(), keep_keys=None)
        )
    if getattr(optimizer, "group_state", None):
        state_payload["group_state"] = [
            kept
            for _, kept in _filter_entries(
                enumerate(optimizer.group_state), _GROUP_STATE_PERSIST
            )
        ]
    if scheduler is not None:
        state_payload["scheduler"] = scheduler.state_dict()
    state_payload["rng_state"] = capture_rng_state()
    torch.save(state_payload, _optim_path(tmp_dir))

    # 3. Sidecar JSON with scalar metadata.
    meta = {
        "step": int(step),
        "total_tokens": int(total_tokens),
        "resume_segment": int(resume_segment),
        "cfg_snapshot": cfg_snapshot,
    }
    with open(_meta_path(tmp_dir), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # 4. Atomic rename: replace target dir with the freshly-written one.
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)


def _resolve_ckpt_dir(path) -> Optional[Path]:
    """Return the specific ckpt dir to load, or None if nothing usable."""
    p = Path(path)
    if not p.exists():
        return None
    if _meta_path(p).exists():
        return p
    if not p.is_dir():
        return None
    candidates = [
        d for d in p.iterdir()
        if d.is_dir()
        and re.fullmatch(r"ckpt_step\d+", d.name)
        and _meta_path(d).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: int(d.name[len("ckpt_step"):]))


def load_checkpoint(
    ckpt_dir_or_parent,
    *,
    bare_model,
    optimizer,
    scheduler=None,
    restore_rng: bool = False,
) -> Optional[dict]:
    """Resume in place. Returns `{step, total_tokens, resume_segment, ckpt_path}`
    or None if no usable checkpoint is present at the given path."""
    ckpt_dir = _resolve_ckpt_dir(ckpt_dir_or_parent)
    if ckpt_dir is None:
        return None

    with open(_meta_path(ckpt_dir)) as fh:
        meta = json.load(fh)

    # 1. Load adapter weights into the existing "default" slot. is_trainable
    #    True so grads keep flowing on resume.
    bare_model.load_adapter(
        str(_adapter_dir(ckpt_dir)),
        adapter_name="default",
        is_trainable=True,
    )

    # 2. Optimizer state. weights_only=False because pair_state contains
    #    Python scalars (`step`) alongside tensors; the legacy unpickling
    #    path is required.
    payload = torch.load(
        _optim_path(ckpt_dir), map_location="cpu", weights_only=False
    )

    if "pytorch_state" in payload:
        try:
            optimizer.load_state_dict(payload["pytorch_state"])
        except (ValueError, KeyError, TypeError):
            # Custom optimizers carry an empty stock state — mismatch is fine.
            pass

    if "pair_state" in payload and getattr(optimizer, "pair_state", None) is not None:
        aliases = _pair_state_aliases(optimizer)
        for i, entry in payload["pair_state"].items():
            dst = optimizer.pair_state.get(i)
            if dst is None:
                continue
            entry = _apply_pair_state_renames(entry, dst.keys(), aliases)
            for k, v in entry.items():
                if k in _PAIR_STATE_SKIP_LOAD:
                    continue
                cur = dst.get(k)
                if isinstance(v, torch.Tensor) and isinstance(cur, torch.Tensor):
                    cur.copy_(v.to(cur.device, cur.dtype))
                elif isinstance(v, torch.Tensor):
                    # Optimizer was just constructed but the diagnostic-state
                    # tensors (e.g. `m_A_raw`) weren't allocated because the
                    # builder didn't enable diagnostics this time. Insert the
                    # loaded tensor directly on the device of an existing
                    # pair-state tensor (or CPU if none).
                    ref = next(
                        (t for t in dst.values() if isinstance(t, torch.Tensor)),
                        None,
                    )
                    device = ref.device if ref is not None else "cpu"
                    dst[k] = v.to(device)
                else:
                    dst[k] = v

    if "group_state" in payload and getattr(optimizer, "group_state", None):
        for gid, entry in enumerate(payload["group_state"]):
            if gid >= len(optimizer.group_state):
                break
            dst = optimizer.group_state[gid]
            for k, v in entry.items():
                cur = dst.get(k)
                if isinstance(v, torch.Tensor) and isinstance(cur, torch.Tensor):
                    cur.copy_(v.to(cur.device, cur.dtype))
                elif isinstance(v, torch.Tensor):
                    # Power-iter warm starts (`v_sigma_A`, etc.) may not exist
                    # on the optimizer yet (allocated only after first step).
                    # Place on the same device as the group's m_A buffer.
                    ref = dst.get("m_A")
                    device = ref.device if isinstance(ref, torch.Tensor) else "cpu"
                    dst[k] = v.to(device)
                else:
                    ref = dst.get("m_A")
                    device = ref.device if isinstance(ref, torch.Tensor) else "cpu"
                    dst[k] = _restore_value(v, device=device)

    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    rng_state = payload.get("rng_state")
    rng_restore_status = None
    if restore_rng:
        rng_restore_status = restore_rng_state(rng_state)

    return {
        "step": int(meta["step"]),
        "total_tokens": int(meta["total_tokens"]),
        "resume_segment": int(meta["resume_segment"]),
        "ckpt_path": str(ckpt_dir),
        "rng_state": rng_state,
        "rng_state_present": rng_state is not None,
        "rng_restore_status": rng_restore_status,
    }


def prune_checkpoints(parent_dir, keep_last: int) -> None:
    """Delete all but the latest `keep_last` `ckpt_step{N}` dirs in
    `parent_dir`. Tolerant of missing dirs."""
    p = Path(parent_dir)
    if not p.exists():
        return
    candidates = [
        d for d in p.iterdir()
        if d.is_dir()
        and re.fullmatch(r"ckpt_step\d+", d.name)
        and _meta_path(d).exists()
    ]
    candidates.sort(key=lambda d: int(d.name[len("ckpt_step"):]), reverse=True)
    for d in candidates[keep_last:]:
        try:
            shutil.rmtree(d)
        except OSError:
            pass


def ckpt_dir_for_step(parent_dir, step: int) -> Path:
    return Path(parent_dir) / f"ckpt_step{step}"
