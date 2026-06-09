"""PolarExpress is a CLIFF, not a slope: the quintic-Remez polar polynomial
gives an exact polar (σ_out=1) when its input σ_max ≤ ~1.01 but goes fully
non-finite at a ≥3% σ_max UNDER-estimate (no finite-but-overscaled middle
ground; offline-verified on real snapshot momenta). The warm-σ_max floor
(max row/col L2) under-estimates σ_max by ~3.3× median at r=256, so an
estimate-based denominator would detonate most pairs every step.

`_polar_ns_guarded` therefore normalizes the polar_method='polar_express'
path by the Frobenius norm — a guaranteed UPPER bound (σ_max ≤ ‖·‖_F) — so the
input is always in-basin regardless of how stale/cold the warm σ_max estimate
is. These tests pin that the PE path is finite and fully polarizing (σ_out→1)
even with a worst-case (empty / cold) warm-start cache, and that cubic NS is
unaffected. CPU, tiny tensors, deterministic.
"""
import torch
import torch.nn as nn

from lora_playground.optim import CurvatureWhitenLoRA


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})

    def forward(self, x):
        return x @ self.lora_A["default"].weight.T @ self.lora_B["default"].weight.T


def _make_opt(polar_method):
    """A CurvatureWhitenLoRA on a tiny real LoRA module so __init__'s pair
    collection succeeds; only the polar dispatch + _smax_warm are exercised
    below, via synthetic batches and fresh per-pair state dicts."""
    m = _FakeLoRALinear(8, 6, 4)
    return CurvatureWhitenLoRA(m, lr=1e-3, use_polar=True, ns_steps=8,
                               polar_method=polar_method)


def _polarize(opt, Z):
    # Worst case for an estimate-based denominator: empty warm cache (cold start)
    # AND a key the states have never seen -> _smax_warm cold-starts every pair.
    states = [{} for _ in range(Z.shape[0])]
    return opt._polar_ns_guarded(Z, states, key="utest")


def _batch(seed, n=6):
    g = torch.Generator().manual_seed(seed)
    # Heterogeneous scales: multiply each matrix by a wildly different factor so
    # a single shared/stale σ_max estimate would be badly wrong for most pairs.
    base = torch.randn(n, 16, 64, generator=g)
    scales = torch.tensor([1e-3, 1.0, 7.3, 1e2, 5e-2, 9e1]).view(n, 1, 1)
    return (base * scales).float()


def test_polar_express_finite_and_unit_spectrum_with_cold_warmstart():
    opt = _make_opt("polar_express")
    for seed in range(4):
        Z = _batch(seed)
        Y = _polarize(opt, Z)
        assert torch.isfinite(Y).all(), f"seed={seed}: PolarExpress produced non-finite output"
        # Full polar => every singular value of every (non-degenerate) matrix → 1.
        for i in range(Z.shape[0]):
            s = torch.linalg.svdvals(Y[i].float())
            assert s.max() <= 1.02, f"seed={seed} pair={i}: σ_max overscaled to {s.max():.3f}"
            assert s.min() >= 0.98, f"seed={seed} pair={i}: σ_min={s.min():.3f} (not full polar)"


def test_polar_express_matches_true_smax_normalization():
    # The Frobenius-primary path must give the SAME polar as normalizing by the
    # exact σ_max — PolarExpress maps any σ∈(0,1] to 1, so both reach σ_out=1.
    opt = _make_opt("polar_express")
    Z = _batch(0)
    Y_guarded = _polarize(opt, Z)
    for i in range(Z.shape[0]):
        smax = torch.linalg.svdvals(Z[i].float()).max()
        ref = opt._polar_poly_batched((Z[i] / smax).unsqueeze(0).float())[0]
        assert torch.allclose(Y_guarded[i], ref, atol=2e-2), \
            f"pair={i}: Frob-primary polar disagrees with true-σ_max polar"


def test_cubic_ns_path_unaffected_and_finite():
    # The NS path keeps the tight σ_max denominator and must stay finite on the
    # same heterogeneous batch (it degrades, not detonates, on compression).
    opt = _make_opt("ns")
    for seed in range(4):
        Y = _polarize(opt, _batch(seed))
        assert torch.isfinite(Y).all(), f"seed={seed}: cubic NS produced non-finite output"


# ── Nesterov-momentum ablation ───────────────────────────────────────────────
# The `cw_nesterov` flag swaps the momentum fed to the whiten→polar→unwhiten
# core from plain bias-corrected EMA (m̂) to the Muon-style lookahead
# (β₁·m + (1−β₁)·g, equivalently g.lerp(m, β₁) in the official Muon repo). The
# operator-norm rescale makes this a direction-only change. These pin: (1) the
# flag is rejected on the SOAP-v̂ path, (2) it produces a finite, DIFFERENT
# trajectory from plain EMA, (3) the label surfaces it.
def _diag_shampoo_opt(cw_nesterov):
    from lora_playground.optim import build_optimizer
    torch.manual_seed(0)
    m = _FakeLoRALinear(8, 6, 4)
    return build_optimizer(m, "diag-shampoo-polar-lora", lr=3e-2,
                           cw_nesterov=cw_nesterov, muon_ns_steps=8,
                           polar_method="polar_express"), m


def test_cw_nesterov_rejected_on_soap_v_path():
    import pytest
    m = _FakeLoRALinear(8, 6, 4)
    with pytest.raises(ValueError, match="soap_v=False"):
        CurvatureWhitenLoRA(m, lr=1e-3, use_polar=True, soap_v=True,
                            cw_nesterov=True)


def test_cw_nesterov_finite_and_changes_trajectory():
    torch.manual_seed(1)
    x = torch.randn(4, 8)

    def run(flag):
        opt, m = _diag_shampoo_opt(flag)
        torch.manual_seed(2)
        for _ in range(5):
            opt.zero_grad()
            (m(x) ** 2).mean().backward()
            opt.step()
        return [p.detach().clone()
                for n, p in m.named_parameters() if "lora_" in n]

    plain, nest = run(False), run(True)
    assert all(torch.isfinite(p).all() for p in plain + nest)
    assert max((a - b).abs().max().item() for a, b in zip(plain, nest)) > 1e-6


def test_cw_nesterov_label():
    from lora_playground.plotting.labels import canonical_label
    base = {"optimizer": "diag-shampoo-polar-lora", "use_polar": True,
            "precond_refresh_every": 10, "curvature_beta": 0.99,
            "precond_delta": 1e-4, "muon_ns_steps": 8,
            "polar_method": "polar_express"}
    assert "+nesterov" in canonical_label({**base, "cw_nesterov": True})
    assert "+nesterov" not in canonical_label({**base, "cw_nesterov": False})
