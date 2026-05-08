"""Tests for lora_playground.mfu — MFU diagnostic helpers."""
import torch

from lora_playground.mfu import (
    PEAK_TFLOPS_BF16,
    compute_mfu,
    count_total_params,
    device_peak_tflops,
    estimate_step_flops,
    flops_per_token_for_mode,
)


def test_flops_per_token_for_mode_lora():
    assert flops_per_token_for_mode("lora") == 4.0
    assert flops_per_token_for_mode("lora", gradient_checkpointing=True) == 6.0


def test_flops_per_token_for_mode_ucv():
    assert flops_per_token_for_mode("ucv") == 4.0


def test_flops_per_token_for_mode_full_ft_modes():
    assert flops_per_token_for_mode("svd_step_oracle") == 6.0
    assert flops_per_token_for_mode("svd_cumulative_oracle") == 6.0
    assert flops_per_token_for_mode("galore") == 6.0
    assert flops_per_token_for_mode("galore", gradient_checkpointing=True) == 8.0


def test_compute_mfu_lora_factor():
    """4N/6N for LoRA gives 2/3 the MFU of the full-FT formula at the
    same step time + token count."""
    common = dict(
        n_params=1_000_000_000,
        tokens_per_step=32_768,
        step_time_sec=1.0,
        peak_tflops=312.0,
    )
    mfu_full_ft = compute_mfu(**common, flops_per_token_per_param=6.0)
    mfu_lora = compute_mfu(**common, flops_per_token_per_param=4.0)
    assert abs(mfu_lora / mfu_full_ft - 4 / 6) < 1e-9


def test_peak_table_lookup_known():
    assert device_peak_tflops("NVIDIA A100-SXM4-80GB") == PEAK_TFLOPS_BF16["A100"]
    assert device_peak_tflops("NVIDIA RTX A6000") == PEAK_TFLOPS_BF16["RTX A6000"]
    # H100 SXM5 in the name → SXM bin (989); generic "H100" without SXM/PCIE
    # falls through to the H100 fallback (also 989).
    assert device_peak_tflops("NVIDIA H100 SXM5 80GB") == PEAK_TFLOPS_BF16["H100 SXM"]
    assert device_peak_tflops("NVIDIA H100 80GB HBM3") == PEAK_TFLOPS_BF16["H100"]
    # Blackwell RTX PRO 6000 — distinct chip from datacenter B200.
    assert device_peak_tflops("NVIDIA RTX PRO 6000 Blackwell") == PEAK_TFLOPS_BF16["RTX PRO 6000"]
    # Datacenter Blackwell.
    assert device_peak_tflops("NVIDIA B200") == PEAK_TFLOPS_BF16["B200"]


def test_peak_table_lookup_unknown():
    assert device_peak_tflops("Some Random GPU That Doesn't Exist") is None


def test_count_total_params():
    m = torch.nn.Linear(10, 5, bias=True)
    # Linear(10→5): 50 weight + 5 bias = 55.
    assert count_total_params(m) == 55


def test_estimate_step_flops_scales():
    # 6·N·T should scale linearly in both N and T.
    f1 = estimate_step_flops(1_000, 100)
    f2 = estimate_step_flops(2_000, 100)
    f3 = estimate_step_flops(1_000, 200)
    assert f2 == 2 * f1
    assert f3 == 2 * f1
    assert f1 == 6 * 1_000 * 100


def test_compute_mfu_basic():
    # 1B params, 32k tokens/step, 1.0 s on a 312-TFLOPS device →
    # achieved = 6·1e9·32e3 / 1e12 = 192 TFLOPS → MFU = 192/312 ≈ 0.615.
    mfu = compute_mfu(
        n_params=1_000_000_000,
        tokens_per_step=32_000,
        step_time_sec=1.0,
        peak_tflops=312.0,
    )
    assert abs(mfu - (6 * 1e9 * 32e3 / 1e12 / 312.0)) < 1e-9


def test_compute_mfu_returns_none_unknown_device():
    mfu = compute_mfu(
        n_params=1_000_000,
        tokens_per_step=1_000,
        step_time_sec=1.0,
        peak_tflops=None,
    )
    # peak_tflops=None and no CUDA → device_peak_tflops returns None.
    if torch.cuda.is_available():
        # On CUDA, device_peak_tflops may resolve; just check it's either
        # None or a sensible value in [0, 1].
        assert mfu is None or (0.0 < mfu < 100.0)
    else:
        assert mfu is None


def test_compute_mfu_returns_none_zero_time():
    mfu = compute_mfu(
        n_params=1_000_000,
        tokens_per_step=1_000,
        step_time_sec=0.0,
        peak_tflops=312.0,
    )
    assert mfu is None
