"""Lightweight CUDA-event timer for component-level profiling inside an
optimizer step. Default-off: optimizer holds an attribute `_step_timer`
that may be `None`; when None, the `_maybe_time` helper is a no-op.

Records (start, end) event pairs lazily; `elapsed_time` is only called at
`summary()` time after a single synchronize, so per-scope overhead is just
the two event creations + records (~few μs each).

Names accumulate: every entry into `timer(name)` appends one pair to
`events[name]`. Re-using a name across the same step (e.g. polar pipeline
called once per Picard iter) is intentional — `summary()` reports n_calls
and totals so iter-scaling is recoverable.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


class CudaTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._stack: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def __call__(self, name: str):
        return _Scope(self, name)

    def start(self, name: str) -> None:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        self._stack.append((name, s, e))

    def stop(self) -> None:
        name, s, e = self._stack.pop()
        e.record()
        self.events.setdefault(name, []).append((s, e))

    def reset(self) -> None:
        self.events.clear()
        self._stack.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        torch.cuda.synchronize(self.device)
        out: dict[str, dict[str, float]] = {}
        for name, pairs in self.events.items():
            ms = [s.elapsed_time(e) for (s, e) in pairs]
            ms_sorted = sorted(ms)
            n = len(ms)
            out[name] = {
                "n": float(n),
                "mean_ms": sum(ms) / n,
                "median_ms": ms_sorted[n // 2],
                "min_ms": ms_sorted[0],
                "max_ms": ms_sorted[-1],
                "total_ms": sum(ms),
            }
        return out


class _Scope:
    __slots__ = ("t", "n")

    def __init__(self, t: CudaTimer, n: str):
        self.t = t
        self.n = n

    def __enter__(self):
        self.t.start(self.n)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t.stop()
        return False


@contextmanager
def maybe_time(timer: Optional[CudaTimer], name: str):
    """No-op when `timer` is None; otherwise a `timer(name)` scope.

    Use this at every instrumentation site so production paths
    (timer=None) carry zero overhead beyond an attribute lookup
    and a context-manager entry/exit.
    """
    if timer is None:
        yield
    else:
        with timer(name):
            yield
