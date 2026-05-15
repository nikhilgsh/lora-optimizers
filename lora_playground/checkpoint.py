"""Step-continuous checkpoint save/load for the lora_playground train loop.

Saves PEFT adapter weights + optimizer state (including custom `pair_state` /
`group_state`) + scheduler + step / total_tokens metadata. Used by `train.py`
to resume from SLURM wall-timeouts.

NOT bitwise-resumable: RNG state and dataloader offset are not preserved.
After resume the sampler is reseeded with `(seed, step)` so the post-resume
trajectory is stochastically continuous but not identical to an uninterrupted
run. Within-sweep comparisons (same step index, same cfg) remain valid.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

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
)
# pair_state keys that hold tensor VIEWS into group_state buffers (and must
# not be re-pickled as independent storage). On save we clone them anyway;
# on load we copy_ into the live view, which writes through to the group
# buffer. No special exclusion required for the test-equality semantics.
_PAIR_STATE_SKIP_LOAD = {"_group", "_local_idx"}


def _adapter_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "adapter"


def _optim_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "optimizer.pt"


def _meta_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "meta.json"


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
            if keep_keys is not None and k not in keep_keys:
                continue
            if isinstance(v, torch.Tensor):
                kept[k] = v.detach().clone().cpu()
            else:
                kept[k] = v
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
        for i, entry in payload["pair_state"].items():
            dst = optimizer.pair_state.get(i)
            if dst is None:
                continue
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
                    dst[k] = v

    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    return {
        "step": int(meta["step"]),
        "total_tokens": int(meta["total_tokens"]),
        "resume_segment": int(meta["resume_segment"]),
        "ckpt_path": str(ckpt_dir),
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
