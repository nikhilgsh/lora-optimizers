"""Runtime-pin test: for each polar-method short-circuit corner, instantiate
AdamPolarProductLoRA on a tiny LoRA model, run one optimizer step with the
polar functions monkey-patched to record which one was actually called, then
assert (a) the right function was called AND (b) `effective_config()` reports
the matching label.

Most drift risk is already eliminated structurally — `_polar_op` and
`effective_config()` both call `resolve_effective_inner_polar`, so they
tautologically agree on the canonical method. This test guards the residual
risk: future edits to `_polar_op` that add a new dispatch branch without a
corresponding resolver case would slip through unit tests on the resolver.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground import optim as opt_mod
from lora_playground.optim import AdamPolarProductLoRA


class _FakeLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        torch.nn.init.normal_(self.lora_B["default"].weight, std=0.05)

    def forward(self, x):
        A = self.lora_A["default"].weight
        B = self.lora_B["default"].weight
        return x @ A.T @ B.T


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = _FakeLoRALinear(8, 6, 4)

    def forward(self, x):
        return self.l(x)


def _run_one_step(opt_kwargs):
    torch.manual_seed(0)
    m = _TinyModel()
    x = torch.randn(3, 8)
    target = torch.randn(3, 6)
    opt = AdamPolarProductLoRA(m, lr=1e-2, **opt_kwargs)
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    return opt


@pytest.fixture
def polar_probe(monkeypatch):
    """Monkey-patch every polar function in optim.py so each call is recorded.
    The recorder forwards to the original so the optimizer step doesn't blow up.

    Covers both the per-pair path (`_newton_schulz`, `_polar_express`,
    `_newton_schulz_hybrid_deepseek`, `_sigma_power_polar`) AND the batched
    path's `_newton_schulz_batched`. AdamPolarProductLoRA dynamically picks
    the batched path when eligible (default ns + no overrides).
    """
    calls: list[tuple[str, dict]] = []

    orig_newton_schulz = opt_mod._newton_schulz
    orig_newton_schulz_batched = opt_mod._newton_schulz_batched
    orig_hybrid = opt_mod._newton_schulz_hybrid_deepseek
    orig_express = opt_mod._polar_express
    orig_express_batched = opt_mod._polar_express_gram_batched
    orig_sigma_power = AdamPolarProductLoRA._sigma_power_polar

    def patched_newton_schulz(X, *args, **kwargs):
        calls.append(("ns", {}))
        return orig_newton_schulz(X, *args, **kwargs)

    def patched_newton_schulz_batched(X, *args, **kwargs):
        # The batched path is the production hot path for the default
        # polar_method="ns" config. From a "what method ran" perspective,
        # batched-NS is still "ns" — the math is identical, only the
        # tensor layout differs.
        calls.append(("ns", {"batched": True}))
        return orig_newton_schulz_batched(X, *args, **kwargs)

    def patched_hybrid(X, *args, **kwargs):
        calls.append(("ns_hybrid", {}))
        return orig_hybrid(X, *args, **kwargs)

    def patched_express(X, *args, **kwargs):
        calls.append(("polar_express", {}))
        return orig_express(X, *args, **kwargs)

    def patched_express_batched(X, *args, **kwargs):
        # polar_express is batched-eligible (de42c77), so the production
        # default config dispatches to the batched Gram form, not the
        # rectangular `_polar_express`. Same method, different tensor
        # layout — record it as "polar_express" just like batched-NS is
        # recorded as "ns" above.
        calls.append(("polar_express", {"batched": True}))
        return orig_express_batched(X, *args, **kwargs)

    # _sigma_power_polar is a @staticmethod — wrapper must NOT take `self`.
    @staticmethod
    def patched_sigma_power(X, p, *args, **kwargs):
        calls.append(("sigma_power", {"p": float(p)}))
        return orig_sigma_power(X, p, *args, **kwargs)

    monkeypatch.setattr(opt_mod, "_newton_schulz", patched_newton_schulz)
    monkeypatch.setattr(opt_mod, "_newton_schulz_batched", patched_newton_schulz_batched)
    monkeypatch.setattr(opt_mod, "_newton_schulz_hybrid_deepseek", patched_hybrid)
    monkeypatch.setattr(opt_mod, "_polar_express", patched_express)
    monkeypatch.setattr(opt_mod, "_polar_express_gram_batched", patched_express_batched)
    monkeypatch.setattr(
        AdamPolarProductLoRA, "_sigma_power_polar", patched_sigma_power,
    )
    return calls


@pytest.mark.parametrize("kwargs,expected_fn,expected_label,expected_p", [
    ({"polar_method": "ns"},            "ns",            "ns",            None),
    ({"polar_method": "ns_hybrid"},     "ns_hybrid",     "ns_hybrid",     None),
    ({"polar_method": "polar_express"}, "polar_express", "polar_express", None),
    ({"polar_sigma_power": 0.0},        "sigma_power",   "svd_exact",     0.0),
    ({"polar_sigma_power": 0.5},        "sigma_power",   "sigma_power(p=0.5)", 0.5),
])
def test_polar_dispatch_matches_effective_config(
    polar_probe, kwargs, expected_fn, expected_label, expected_p,
):
    opt = _run_one_step(kwargs)
    # (a) the right runtime function was called.
    fns_called = {c[0] for c in polar_probe}
    assert expected_fn in fns_called, (
        f"Expected runtime dispatch to {expected_fn} given {kwargs}; "
        f"got calls to {fns_called} instead."
    )
    if expected_p is not None:
        # For sigma_power, also verify the p argument.
        ps = [c[1]["p"] for c in polar_probe if c[0] == "sigma_power"]
        assert expected_p in ps, (
            f"Expected sigma_power called with p={expected_p}; got p∈{ps}."
        )
    # (b) effective_config()'s label matches what was actually run.
    eff = opt.effective_config()
    assert eff["effective_inner_polar"] == expected_label, (
        f"effective_config() reports {eff['effective_inner_polar']!r}; "
        f"runtime ran {expected_fn} (expected label {expected_label!r})."
    )
