"""Single-node DDP utilities for `train.py`.

Single source of truth for the distributed-training contract:
- `init_distributed()` is idempotent and a no-op when `WORLD_SIZE` is unset
  (single-GPU/CPU paths require zero changes).
- `is_main()` / `get_rank()` / `get_world_size()` work in both modes.
- `all_reduce_mean()` aggregates a scalar across ranks weighted by token
  counts (used to make eval loss bit-for-bit identical to single-GPU).

Launch with `torchrun --nproc_per_node=N train_lora.py ...`. torchrun sets
`RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars; SLURM equivalents
(`SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`) are honored as fallback.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _env_int(name: str, fallbacks: tuple[str, ...] = ()) -> int | None:
    for key in (name, *fallbacks):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return None


def init_distributed() -> tuple[int, int, int]:
    """Initialize the default process group if launched under torchrun/SLURM.

    Returns (rank, world_size, local_rank). Single-process mode returns
    (0, 1, 0) and does nothing else. Idempotent — safe to call twice.
    """
    world_size = _env_int("WORLD_SIZE", ("SLURM_NTASKS",)) or 1
    rank = _env_int("RANK", ("SLURM_PROCID",)) or 0
    local_rank = _env_int("LOCAL_RANK", ("SLURM_LOCALID",)) or 0

    if world_size <= 1:
        return 0, 1, 0

    if not dist.is_available():
        raise RuntimeError("WORLD_SIZE > 1 but torch.distributed is unavailable.")

    if not dist.is_initialized():
        # nccl is the default for CUDA; gloo is the CPU fallback. Override via
        # LORA_DDP_BACKEND env var (used by CPU-only tests on a CUDA host).
        backend = os.environ.get("LORA_DDP_BACKEND")
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    else:
        backend = dist.get_backend()

    # Only pin a CUDA device when actually using nccl. Gloo runs CPU tensors,
    # and calling set_device on a CUDA host where rank > num_gpus would crash.
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main() -> bool:
    return get_rank() == 0


def all_reduce_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Weighted mean of `values` across ranks. `values` and `weights` should
    be 0-dim or 1-element tensors holding the per-rank sum-of-loss-tokens
    and token count respectively. Returns the global weighted mean as a
    Python float. Single-process: returns values/weights without reduction.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return float(values.item() / weights.item())
    # Pack into one 2-element tensor to combine into a single all-reduce.
    packed = torch.stack([values.detach(), weights.detach()]).to(torch.float64)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return float(packed[0].item() / packed[1].item())


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
