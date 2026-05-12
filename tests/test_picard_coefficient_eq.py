"""Structural-equivalence check: the Picard cross-coupling block is gated
on `k_iter > 0` (batched path) and `k > 0` (per-pair path), so at
picard_iters=1 the body never executes. Changing the coefficient inside
that block from `picard_alpha/lr` to `2/(ρ·s)` cannot affect k=1 behavior.

This test asserts the gate is present in the source. It's a guard against
future refactors that might accidentally move the cross-coupling code
outside the `k_iter > 0` branch.
"""
import re
from pathlib import Path


def _read_optim_source() -> str:
    return (Path(__file__).resolve().parent.parent
            / "lora_playground" / "optim.py").read_text()


def test_picard_cross_coupling_gated_on_k_gt_0_batched():
    """Batched path: cross-coupling code must live inside `if k_iter > 0:`."""
    src = _read_optim_source()
    # Find batched Picard loop "for k_iter in range(self.picard_iters):"
    # then immediately under it expect "if k_iter > 0:" gate before any
    # `BT_dB_A` / `B_dA_AT` cross-coupling matmul.
    pattern = re.compile(
        r"for k_iter in range\(self\.picard_iters\):.*?if k_iter > 0:.*?BT_dB_A",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "batched Picard cross-coupling not gated on k_iter > 0 — "
        "k=1 might no longer be a no-op"
    )


def test_picard_cross_coupling_gated_on_k_gt_0_per_pair():
    """Per-pair path: same gate, different variable name."""
    src = _read_optim_source()
    pattern = re.compile(
        r"for k in range\(self\.picard_iters\):.*?if k == 0:.*?else:.*?@ dB_prev @ A_f",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "per-pair Picard cross-coupling not gated on k > 0 — "
        "k=1 might no longer be a no-op"
    )


def test_picard_coefficient_uses_2_over_rho_s_under_chord_tight():
    """The new coefficient `2 / (ρ · s)` is wired in for the chord-tight
    family. Soft check that the constant appears in the source.
    """
    src = _read_optim_source()
    assert "2.0 / (rho * s_AB" in src, "batched picard_coeff_s formula missing"
    assert "2.0 / (rho * (sigma_A_t + sigma_B_t)" in src, "per-pair picard_coeff_t formula missing"
