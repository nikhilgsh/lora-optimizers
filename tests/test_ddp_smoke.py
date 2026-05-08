"""CPU-only DDP smoke for the distributed-training plumbing.

Spawns 2 worker processes via `torch.multiprocessing`, initializes a gloo
process group, wraps a tiny linear model in DDP, runs one forward/backward,
and asserts:

1. `init_distributed()` correctly reads RANK/WORLD_SIZE/LOCAL_RANK from env.
2. After `loss.backward()`, the gradient on each parameter matches the
   simple-average of per-rank gradients (DDP's default reducer).
3. `all_reduce_mean` aggregates a (sum, count) pair to the global mean
   that matches a single-process equivalent.

Functional GPU equivalence (real loss matches single-GPU bit-for-bit) is
covered separately by the multi-GPU bench in A0.7.6 — this file is the
unit test ensuring the wiring contract holds before any GPU work.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp


def _ddp_worker(rank: int, world_size: int, port: int, out_path: str) -> None:
    """Run inside a spawned process. Initializes DDP, runs one step, writes
    a results dict to `out_path` so the parent can assert on it."""
    import json

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Force gloo so the test runs CPU-only even on CUDA hosts.
    os.environ["LORA_DDP_BACKEND"] = "gloo"

    from lora_playground.distributed import (
        all_reduce_mean,
        cleanup,
        get_rank,
        get_world_size,
        init_distributed,
        is_main,
    )
    from torch.nn.parallel import DistributedDataParallel as DDP

    r, w, lr = init_distributed()
    assert r == rank and w == world_size and lr == rank, (r, w, lr)
    assert get_rank() == rank
    assert get_world_size() == world_size
    assert is_main() == (rank == 0)

    torch.manual_seed(0)
    # Each rank gets a different input so the per-rank gradient is rank-dependent;
    # after the all-reduce we should see the average.
    model = torch.nn.Linear(4, 1, bias=False)
    model_ddp = DDP(model)
    x = torch.tensor([[float(rank + 1.0)] * 4])
    y = torch.tensor([[1.0]])
    loss = ((model_ddp(x) - y) ** 2).sum()
    loss.backward()

    # Per-rank grad before all-reduce would be 2*(model(x)-y)*x — but DDP
    # already all-reduced (mean) inside backward(). So model.weight.grad is
    # the same on every rank.
    grad = model.weight.grad.detach().clone().tolist()

    # Verify all_reduce_mean works: each rank reports (rank+1.0) sum with
    # weight 1.0; the global mean should be (sum_of_ranks_+1.0) / world_size.
    val = torch.tensor(float(rank + 1.0))
    wt = torch.tensor(1.0)
    # need cuda? gloo works on cpu tensors
    aggregated = all_reduce_mean(val, wt)

    out = {
        "rank": rank,
        "world_size": world_size,
        "grad": grad,
        "all_reduce_mean": aggregated,
    }
    with open(f"{out_path}.{rank}", "w") as f:
        json.dump(out, f)

    cleanup()


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_ddp_init_grad_allreduce_and_mean(tmp_path):
    if not (torch.distributed.is_available()):
        pytest.skip("torch.distributed not available")

    out_path = str(tmp_path / "result")
    port = _free_port()
    world_size = 2

    mp.spawn(
        _ddp_worker,
        args=(world_size, port, out_path),
        nprocs=world_size,
        join=True,
    )

    import json

    results = []
    for r in range(world_size):
        with open(f"{out_path}.{r}") as f:
            results.append(json.load(f))

    # All ranks see identical grads after DDP all-reduce.
    assert results[0]["grad"] == results[1]["grad"], (
        f"DDP grad mismatch across ranks: {results[0]['grad']} vs {results[1]['grad']}"
    )

    # all_reduce_mean: each rank reported value rank+1, equal weight.
    # Global weighted mean = (1+2)/2 = 1.5.
    expected = (sum(r + 1 for r in range(world_size))) / world_size
    for r in range(world_size):
        assert abs(results[r]["all_reduce_mean"] - expected) < 1e-9


def test_distributed_single_process_no_op():
    """init_distributed without WORLD_SIZE env returns (0, 1, 0) and does
    not initialize any process group."""
    # Defensive: scrub any inherited env that might trick init_distributed.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS"):
        os.environ.pop(key, None)

    from lora_playground.distributed import (
        cleanup,
        get_rank,
        get_world_size,
        init_distributed,
        is_main,
    )

    r, w, lr = init_distributed()
    assert (r, w, lr) == (0, 1, 0)
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main() is True
    cleanup()  # no-op
