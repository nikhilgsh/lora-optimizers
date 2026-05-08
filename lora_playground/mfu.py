"""Model FLOPs Utilization (MFU) diagnostic.

MFU = achieved-FLOPs/sec / peak-FLOPs/sec for the GPU. A standard
throughput summary that's roughly invariant to model size, sequence
length, and batch geometry — useful for spotting when something
(recompiles, padding waste, kernel choice) is leaving compute on the
floor.

Approximation: per-token compute is 6·N where N = total parameter count.
The 6 is "fwd cost (2N) + bwd cost (4N)"; bwd counts param grads (2N) +
activation grads (2N). For LoRA training we still count the full base
model in N — backward through activations passes through the whole
graph even though only LoRA params receive gradients, and that backward
is the dominant cost. This overestimates LoRA MFU slightly compared to
a "LoRA-only-params" denominator but matches the convention used in
nanoGPT and Schulman/Biderman writeups, so the numbers are comparable.

Peak-FLOPs values are dense bf16/fp16 throughput for the listed devices
from the vendor specs. For other devices, returns None and the caller
should drop the MFU field rather than misreport it.

Use only on isolated GPUs (per CLAUDE.md: timing measurements need
exclusive allocation). Reading MFU off a contended GPU gives a useless
number.
"""
from __future__ import annotations

import re

import torch


# Dense bf16/fp16 TFLOPS, per the published NVIDIA datasheets / architecture
# whitepapers. NOT sparsity-doubled — real training kernels run dense.
#
# Project-relevant entries (per CLAUDE.md): A100 is the canonical comparison
# baseline; A6000 (Ampere) is for functional smokes only; H100 + Blackwell
# RTX PRO 6000 cover the newer hosts. Other entries are conveniences for
# common dev hardware.
#
# Pattern matching is substring-uppercase against `torch.cuda.get_device_name`,
# so longer keys must come before shorter prefixes (e.g. "H200" before "H100"
# is fine since they don't overlap, but "RTX PRO 6000" must precede a "6000"
# substring rule). Keys are intentionally specific.
PEAK_TFLOPS_BF16: dict[str, float] = {
    # ── Ampere (sm_80 / sm_86) ────────────────────────────────────────────
    "A100":            312.0,    # H100 datasheet / Ampere whitepaper
    "RTX A6000":       154.8,    # GA102 dense BF16 (Ampere)
    "RTX 3090":        142.0,    # GA102 consumer
    # ── Hopper (sm_90) ────────────────────────────────────────────────────
    "H100 SXM":        989.0,    # H100 SXM5 datasheet
    "H100 PCIE":       756.0,    # H100 PCIe (binned lower)
    "H100":            989.0,    # generic fallback if no SXM/PCIe in name
    "H200":            989.0,    # same compute as H100 SXM, more HBM
    # ── Ada Lovelace (sm_89) ──────────────────────────────────────────────
    "RTX 6000 ADA":    364.2,    # AD102 workstation (different chip than
                                  # Ampere "RTX A6000")
    "RTX 4090":        165.2,    # AD102 consumer dense BF16
    "L40":             181.0,    # AD102 server
    # ── Blackwell (sm_100, sm_120) ────────────────────────────────────────
    "B200":          2250.0,    # datacenter Blackwell SXM, BF16 dense
    "RTX PRO 6000":   251.9,    # workstation Blackwell (GB202), dense BF16
                                  # — NOT the same chip as B200; user-facing
                                  # "Blackwell RTX" hosts run this number
    "RTX 5090":       210.0,    # GB202 consumer (approx, datasheet pending)
}


def device_peak_tflops(device_name: str | None = None) -> float | None:
    """Look up dense bf16 TFLOPS for the current CUDA device. None if
    unknown. Pass `device_name` to override the auto-detection (useful
    when the host has multiple GPU types or you want to force a baseline)."""
    if device_name is None:
        if not torch.cuda.is_available():
            return None
        device_name = torch.cuda.get_device_name(0)
    name = device_name.upper()
    for key, val in PEAK_TFLOPS_BF16.items():
        if key.upper() in name:
            return val
    return None


def count_total_params(model: torch.nn.Module) -> int:
    """Count *all* parameters (frozen + trainable). The 6N FLOPs
    approximation needs the full forward graph's param count, which
    for LoRA includes the frozen base model."""
    return sum(p.numel() for p in model.parameters())


def estimate_step_flops(
    n_params: int,
    tokens_per_step: int,
    *,
    flops_per_token_per_param: float = 6.0,
) -> float:
    """Rough fwd+bwd FLOPs for one optimizer step: 6 · N · T.

    Ignores the attention quadratic correction (s/(6h) term in the
    Megatron formula); for seq_length up to ~4k on hidden ~1k models
    this is < 10% of total compute, well under the noise of a peak
    estimate.
    """
    return float(flops_per_token_per_param) * float(n_params) * float(tokens_per_step)


def compute_mfu(
    n_params: int,
    tokens_per_step: int,
    step_time_sec: float,
    *,
    peak_tflops: float | None = None,
) -> float | None:
    """Returns MFU in [0, 1], or None if peak is unknown / step_time is
    non-positive."""
    if peak_tflops is None:
        peak_tflops = device_peak_tflops()
    if peak_tflops is None or step_time_sec <= 0:
        return None
    flops = estimate_step_flops(n_params, tokens_per_step)
    achieved_tflops = flops / step_time_sec / 1e12
    return achieved_tflops / peak_tflops
