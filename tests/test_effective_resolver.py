"""Unit tests for resolve_effective_inner_polar — the shared resolver used by
BOTH `_polar_pipeline._polar_op` (runtime dispatch) AND `effective_config()`
(per-run cfg event). Drift between the two cannot happen because both call
this function; the test pins the function's behavior at every short-circuit
corner.
"""
import pytest

from lora_playground.optim import (
    optimizer_effective_config,
    resolve_effective_inner_polar,
)


class TestResolveEffectiveInnerPolar:
    """Precedence (highest first): sigma_power == 0 → svd_exact,
    sigma_power != None → sigma_power(p=…), polar_method ∈ {ns, ns_hybrid,
    polar_express} → that string, optimizer-class fallback → ns, else None."""

    # ── sigma_power short-circuit ─────────────────────────────────────────
    def test_sigma_power_zero_is_svd_exact(self):
        r = resolve_effective_inner_polar(0.0, "ns")
        assert r == {"method": "svd_exact", "label": "svd_exact"}

    def test_sigma_power_zero_overrides_polar_method(self):
        # sigma_power=0.0 wins even when polar_method is set to a real value.
        for pm in ("ns", "ns_hybrid", "polar_express"):
            r = resolve_effective_inner_polar(0.0, pm)
            assert r["method"] == "svd_exact"

    def test_sigma_power_nonzero(self):
        r = resolve_effective_inner_polar(0.5, "ns")
        assert r["method"] == "sigma_power"
        assert r["sigma_power"] == 0.5
        assert r["label"] == "sigma_power(p=0.5)"

    def test_sigma_power_one(self):
        # σ → σ^1 = σ (identity). Valid per the [0,1] range. Still sigma_power
        # path, not "ns" — the math actually called is _sigma_power_polar.
        r = resolve_effective_inner_polar(1.0, "ns")
        assert r["method"] == "sigma_power"
        assert r["sigma_power"] == 1.0

    # ── polar_method dispatch when sigma_power is None ────────────────────
    def test_polar_method_ns(self):
        r = resolve_effective_inner_polar(None, "ns")
        assert r == {"method": "ns", "label": "ns"}

    def test_polar_method_ns_hybrid(self):
        r = resolve_effective_inner_polar(None, "ns_hybrid")
        assert r == {"method": "ns_hybrid", "label": "ns_hybrid"}

    def test_polar_method_polar_express(self):
        r = resolve_effective_inner_polar(None, "polar_express")
        assert r == {"method": "polar_express", "label": "polar_express"}

    # ── legacy fallback via optimizer_class_name ──────────────────────────
    def test_legacy_polar_product_class_falls_back_to_ns(self):
        r = resolve_effective_inner_polar(
            None, None, optimizer_class_name="AdamPolarProductLoRA",
        )
        assert r == {"method": "ns", "label": "ns"}

    def test_legacy_fallback_handles_hyphen_lower(self):
        # The legacy fallback normalizes case + separators so both the
        # cfg-side string ("adam-polar-product-lora") and the runtime-side
        # class name ("AdamPolarProductLoRA") map to the same result.
        r = resolve_effective_inner_polar(
            None, None, optimizer_class_name="adam-polar-product-lora",
        )
        assert r == {"method": "ns", "label": "ns"}

    def test_no_class_name_returns_none(self):
        assert resolve_effective_inner_polar(None, None) is None

    def test_non_polar_class_returns_none(self):
        assert resolve_effective_inner_polar(
            None, None, optimizer_class_name="AdamW",
        ) is None

    # ── string-typed inputs (from cfg-event JSON) ─────────────────────────
    def test_sigma_power_string_zero(self):
        # cfg events may carry numeric values as strings.
        r = resolve_effective_inner_polar("0.0", "ns")
        assert r["method"] == "svd_exact"

    def test_sigma_power_string_none_sentinel(self):
        # Older cfgs literally serialize None as "None"; treat as None.
        r = resolve_effective_inner_polar("None", "ns")
        assert r == {"method": "ns", "label": "ns"}

    def test_sigma_power_garbage_string_falls_through(self):
        # Malformed sigma_power doesn't crash — we fall through to polar_method.
        r = resolve_effective_inner_polar("not-a-float", "ns_hybrid")
        assert r == {"method": "ns_hybrid", "label": "ns_hybrid"}


class TestOptimizerEffectiveConfigHelper:
    """The module-level helper that calls opt.effective_config() if it exists.
    Designed for objects we don't own — never monkey-patch the base class."""

    def test_returns_empty_for_object_without_method(self):
        class FakeOpt:
            pass
        assert optimizer_effective_config(FakeOpt()) == {}

    def test_returns_method_result_when_present(self):
        class FakeOpt:
            def effective_config(self):
                return {"effective_inner_polar": "ns", "effective_picard_iters": 3}
        result = optimizer_effective_config(FakeOpt())
        assert result == {"effective_inner_polar": "ns", "effective_picard_iters": 3}

    def test_non_callable_effective_config_is_ignored(self):
        # Defensive: if some subclass stores a value under that name rather
        # than a method, the helper should not blow up.
        class FakeOpt:
            effective_config = "not a method"
        assert optimizer_effective_config(FakeOpt()) == {}
