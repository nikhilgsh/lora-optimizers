"""Unit coverage for the full-residual Frank-Wolfe variant of
`_chord_tight_clean_polar_pipeline` (`fw_linearization="full"`).

Per `docs/notes/polar_product/algorithm_tight_chord.md` §6, the full-FW
variant differs from the anchored production iteration only by retaining
the self-terms `S_B·dA` and `dB·S_A` inside each block's polar input.
Two properties this implies:

1. At `picard_iters=1`, the variant is bit-identical to anchored because
   `dA⁽⁰⁾ = dB⁽⁰⁾ = 0` zeroes out the new self-term.
2. At `picard_iters >= 2`, the updates differ — sanity check that the
   new term is actually active and not accidentally a no-op (e.g. δ
   collapsing the gram, or the gating flag never reaching the branch).

Additionally, the default `fw_linearization="anchored"` must produce
bit-identical output to the pre-existing chord-tight-clean path — no
silent perturbation of the production optimizer.
"""
import copy

import pytest
import torch

from lora_playground.optim import AdamPolarProductLoRA

from tests.test_chord_tight_clean import (
    FakeLoRAModel,
    _seed_grads,
    tiny_specs,
)


def _make_opt(model, *, fw_linearization, picard_iters):
    return AdamPolarProductLoRA(
        model,
        lr=3e-3,
        betas=(0.9, 0.999),
        delta=1e-6,
        eps=1e-8,
        ns_steps=5,
        lora_plus_multiplier=1.0,
        log_basic_diagnostics=False,
        picard_iters=picard_iters,
        precond_refresh_every=1,
        precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean",
        ns_form="rect",
        fw_linearization=fw_linearization,
    )


def _params_after_steps(model, fw_linearization, picard_iters, n_steps, seed=11):
    opt = _make_opt(model, fw_linearization=fw_linearization,
                    picard_iters=picard_iters)
    for s in range(n_steps):
        _seed_grads(model, seed + s)
        opt.step()
    out = []
    for adapter in model.adapters:
        out.append(adapter.lora_A["default"].weight.detach().clone())
        out.append(adapter.lora_B["default"].weight.detach().clone())
    return out


def test_k1_anchored_full_bit_identical(tiny_specs):
    """At picard_iters=1 the self-term multiplies dA⁽⁰⁾=0 ⇒ no effect."""
    m1 = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    m2 = copy.deepcopy(m1)
    p1 = _params_after_steps(m1, "anchored", picard_iters=1, n_steps=5)
    p2 = _params_after_steps(m2, "full",     picard_iters=1, n_steps=5)
    for a, b in zip(p1, p2):
        assert torch.equal(a, b), (
            "anchored and full FW must be bit-identical at picard_iters=1"
        )


def test_default_matches_anchored(tiny_specs):
    """No-flag construction = anchored (preserves production behavior)."""
    m1 = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    m2 = copy.deepcopy(m1)

    def _make_default(model):
        return AdamPolarProductLoRA(
            model, lr=3e-3, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
            ns_steps=5, lora_plus_multiplier=1.0,
            log_basic_diagnostics=False,
            picard_iters=2, precond_refresh_every=1,
            precond_method="higham",
            magnitude_rule="spectral_chord_tight_clean",
            ns_form="rect",
            # fw_linearization not passed -> default
        )

    opt1 = _make_default(m1)
    opt2 = _make_opt(m2, fw_linearization="anchored", picard_iters=2)
    for s in range(5):
        _seed_grads(m1, 7 + s); opt1.step()
        _seed_grads(m2, 7 + s); opt2.step()
    for adapter1, adapter2 in zip(m1.adapters, m2.adapters):
        for k in ("lora_A", "lora_B"):
            w1 = getattr(adapter1, k)["default"].weight
            w2 = getattr(adapter2, k)["default"].weight
            assert torch.equal(w1, w2), (
                f"default fw_linearization must match 'anchored' on {k}"
            )


def test_k2_anchored_full_diverge(tiny_specs):
    """At picard_iters=2 the self-term is non-zero ⇒ updates differ."""
    m1 = FakeLoRAModel(tiny_specs, init_B_nonzero=True)
    m2 = copy.deepcopy(m1)
    p1 = _params_after_steps(m1, "anchored", picard_iters=2, n_steps=3)
    p2 = _params_after_steps(m2, "full",     picard_iters=2, n_steps=3)
    max_rel = 0.0
    for a, b in zip(p1, p2):
        denom = float(a.float().norm()) + 1e-30
        rel = float((a - b).float().norm()) / denom
        max_rel = max(max_rel, rel)
    assert max_rel > 1e-4, (
        f"anchored and full FW should diverge at picard_iters=2; "
        f"max relative diff was {max_rel:.2e}"
    )


def test_invalid_fw_linearization_rejected():
    m = FakeLoRAModel([(4, 8, 8)])
    with pytest.raises(ValueError, match="fw_linearization"):
        AdamPolarProductLoRA(
            m, lr=3e-3, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
            ns_steps=5, lora_plus_multiplier=1.0,
            picard_iters=1, precond_refresh_every=1,
            precond_method="higham",
            magnitude_rule="spectral_chord_tight_clean",
            ns_form="rect",
            fw_linearization="bogus",
        )
