"""Field-role registry contract tests.

`lora_playground.field_roles` answers ONE question — "does this recorded cfg
field distinguish runs?" — for the five consumers that used to answer it
separately (``arms.PINNED_FIELDS``, ``labels._shared_knobs``,
``dedup._series_items``, ``manifest.SERIES_AXIS_FIELDS``,
``run_records.RUNTIME_FIELDS``). These tests lock the properties the consumers
rely on:

  * the roles PARTITION the fields they cover, so no field has two answers;
  * each role still agrees with the set it is derived from;
  * a field in NO role is series-defining, never provenance;
  * every knob `labels` spells by name carries a label-bearing role, and a
    knob's ABSENCE from a cfg is not a distinguishing value.
"""
import pytest

from lora_playground import field_roles as fr
from lora_playground.manifest import SERIES_AXIS_FIELDS
from lora_playground.plotting.arms import PINNED_FIELDS
from lora_playground.plotting.dedup import series_id
from lora_playground.run_records import RUNTIME_FIELDS
from lora_playground.run_schema import REVISION_FIELDS


# ─── the partition ───────────────────────────────────────────────────────────

def test_roles_are_pairwise_disjoint():
    """The module's reason to exist: one field, one role.

    `field_roles` asserts this at import and raises `FieldRoleConflict`; this
    test states it independently so the assertion cannot be quietly weakened.
    """
    roles = fr.ROLE_NAMES
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            overlap = fr.FIELDS_BY_ROLE[role] & fr.FIELDS_BY_ROLE[other]
            assert not overlap, (
                f"{sorted(overlap)} is in both {role} and {other}; a field with "
                f"two roles has no answer to 'does it distinguish runs?'")


def test_every_role_is_populated_and_role_of_agrees():
    for role in fr.ROLE_NAMES:
        names = fr.FIELDS_BY_ROLE[role]
        assert names, f"role {role} derived to the empty set"
        for name in names:
            assert fr.role_of(name) == role


def test_conflict_is_reported_with_the_offending_field():
    """The assertion must NAME the field, not just fail."""
    with pytest.raises(fr.FieldRoleConflict) as exc:
        fr._assert_disjoint({
            fr.ALGORITHM: frozenset({"a", "shared"}),
            fr.PROVENANCE: frozenset({"b", "shared"}),
        })
    assert "shared" in str(exc.value)


# ─── each role still agrees with the set it is derived from ──────────────────

def test_every_pinned_field_is_algorithm():
    missing = sorted(PINNED_FIELDS() - fr.ALGORITHM_FIELDS)
    assert not missing, (
        f"{missing} are pinned by `arms.arm()` but carry no ALGORITHM role, so "
        f"`labels._residual_knobs` would stop spelling them")


def test_every_runtime_field_is_provenance():
    missing = sorted(RUNTIME_FIELDS - fr.PROVENANCE_FIELDS)
    assert not missing, f"{missing} are RUNTIME_FIELDS but not PROVENANCE"


def test_revision_fields_are_the_revision_role():
    assert fr.REVISION_FIELDS == frozenset(REVISION_FIELDS)


def test_the_two_existing_sources_agree_on_what_they_both_cover():
    """`SERIES_AXIS_FIELDS` and the roles derived from it must not contradict:
    everything `RUNTIME_FIELDS` or SWEEP claims is already a series axis."""
    assert RUNTIME_FIELDS <= SERIES_AXIS_FIELDS
    assert fr.SWEEP_FIELDS <= SERIES_AXIS_FIELDS


def test_only_explicitly_classified_fields_leave_series_identity():
    """`dedup._non_series_fields` unions the caller's ``axis_fields`` with the
    SWEEP and PROVENANCE roles, so a field in either role stops splitting
    `series_id`. Every such field beyond ``SERIES_AXIS_FIELDS`` must come from
    an EXPLICIT, justified classification — `_EXTRA_PROVENANCE`, or a
    constructor spelling that inherited a provenance role through
    ``optim_config.ALIAS`` (``diagnostics_every``). Nothing may slide out of
    series identity as a side effect of a derivation.
    """
    from lora_playground.optim_config import ALIAS
    deliberate = fr._EXTRA_PROVENANCE | frozenset(ALIAS)
    slipped = sorted(fr.fields_with_roles(fr.SWEEP, fr.PROVENANCE)
                     - SERIES_AXIS_FIELDS - deliberate)
    assert not slipped, (
        f"{slipped} stopped splitting series_id without being classified "
        f"explicitly; add each to field_roles._EXTRA_PROVENANCE with the "
        f"train.py line that reads it, or give it a series-defining role")


def test_workload_fields_are_not_algorithm_or_provenance():
    """A workload field splits series identity (two models are two results) but
    is never pinned by an arm predicate — the caller's cell selection carries
    it."""
    assert fr.WORKLOAD_FIELDS & PINNED_FIELDS() == frozenset()
    assert fr.WORKLOAD_FIELDS & RUNTIME_FIELDS == frozenset()
    for name in fr.WORKLOAD_FIELDS:
        assert fr.is_series_defining(name)


# ─── the constructor-only extras ────────────────────────────────────────────

def _registry_constructor_names() -> dict:
    """``{constructor kwarg: [optimizer names]}`` over the whole spec registry.

    The same walk `dedup._constructor_defaults` uses — MRO-aware
    ``forwardable_constructor_parameters``, not a bare ``inspect.signature``,
    which sees 3 parameters for a variant that delegates through ``**kwargs``.
    """
    from lora_playground.constructor_introspection import (
        forwardable_constructor_parameters,
    )
    from lora_playground.optim_specs import REGISTRY
    out: dict = {}
    for name, spec in REGISTRY.items():
        if spec.cls is None:
            continue
        try:
            params = forwardable_constructor_parameters(spec.cls)
        except (TypeError, ValueError):
            continue
        for p in params:
            out.setdefault(p.name, []).append(name)
    return out


def test_constructor_knobs_are_algorithm():
    """A name any registered optimizer's constructor accepts reaches its update,
    so it is ALGORITHM — derived from the registry, not listed."""
    ctor_names = _registry_constructor_names()
    assert ctor_names, "the registry walk found no constructor parameters"
    for name in ("flat_outer", "betas", "eps", "magnitude_rule", "diag_metric",
                 "picard_iters", "operator_type"):
        assert ctor_names.get(name), f"{name} is not a registry ctor parameter"
        assert fr.role_of(name) == fr.ALGORITHM, (
            f"{name} is a constructor knob but carries role "
            f"{fr.role_of(name)!r}")


def test_recorded_only_algorithm_entries_are_not_constructor_knobs():
    """`_RECORDED_ONLY_ALGORITHM` exists for knobs the registry CANNOT see: a
    name still accepted by some constructor belongs to the derived set instead,
    and listing it twice is the duplication this module removes."""
    ctor_names = _registry_constructor_names()
    still_live = sorted(n for n in fr._RECORDED_ONLY_ALGORITHM
                        if ctor_names.get(n))
    assert not still_live, (
        f"{still_live} are listed as recorded-only but ARE constructor "
        f"parameters; drop them and let `_constructor_knobs` classify them")
    for name in fr._RECORDED_ONLY_ALGORITHM:
        assert fr.role_of(name) == fr.ALGORITHM
    # `freeze_factorwise_slots` was the worked example here until the optimizer
    # side of the ablation merged. It is now a real `CurvatureWhitenLoRA`
    # constructor parameter, so the derived constructor walk classifies it --
    # and it must still come out ALGORITHM, because that is what lets
    # `labels._FEATURED_KNOBS` spell " frozen-slots".
    assert "freeze_factorwise_slots" not in fr._RECORDED_ONLY_ALGORITHM
    assert ctor_names.get("freeze_factorwise_slots")
    assert fr.role_of("freeze_factorwise_slots") == fr.ALGORITHM
    # The worked example is now a flag no registered constructor accepts, which
    # is the only case the hand-listed set is for.
    assert "cw_picard_mode" in fr._RECORDED_ONLY_ALGORITHM
    assert not ctor_names.get("cw_picard_mode")


def test_alias_ctor_spelling_inherits_its_config_fields_role():
    """A CONSTRUCTOR spelling takes the role of the config field
    ``optim_config.ALIAS`` maps it to, rather than being classified again.

    ``diagnostics_every`` is the case that matters: it is BOTH an alias of
    ``optim_diagnostics_every`` (a ``RUNTIME_FIELDS`` member) and a constructor
    parameter of 56 optimizers, and the provenance answer must win.
    """
    from lora_playground.optim_config import ALIAS
    for ctor_name, config_name in ALIAS.items():
        config_role = fr.role_of(config_name)
        if config_role is None:
            continue
        assert fr.role_of(ctor_name) == config_role, (
            f"ctor {ctor_name!r} is role {fr.role_of(ctor_name)!r} but the "
            f"config field it aliases ({config_name!r}) is {config_role!r}")
    assert fr.role_of("delta") == fr.ALGORITHM
    assert fr.role_of("diagnostics_every") == fr.PROVENANCE


def test_ns_steps_is_the_constructor_spelling_of_a_pinned_field():
    """`ns_steps` earns ALGORITHM by being ``muon_ns_steps`` under its ctor
    name, which is why `labels._LABELLED_ELSEWHERE` derives the alias rather
    than spelling the knob twice."""
    from lora_playground.optim_config import ALIAS
    from lora_playground.plotting.labels import _LABELLED_ELSEWHERE
    assert ALIAS["ns_steps"] == "muon_ns_steps"
    assert "muon_ns_steps" in fr.ALGORITHM_FIELDS
    assert {"ns_steps", "muon_ns_steps"} <= _LABELLED_ELSEWHERE


# ─── completeness against the runs on disk ──────────────────────────────────

def test_every_field_recorded_on_disk_carries_a_role():
    """The registry is only a single source of truth if it covers what is
    actually recorded. Unknown-means-series-defining is the right FAIL-LOUD
    default, but a field left unroled is a fact no set owns.

    Underscore-prefixed keys are loader/parser namespaces (`_derived`,
    `_cli_args`), not recorded config fields.
    """
    from lora_playground.loader import load_runs
    runs = load_runs(warn_cross_commit=False)
    assert runs, "no runs on disk to check coverage against"
    unroled = sorted({k for cfg, _ in runs for k in cfg
                      if not k.startswith("_") and fr.role_of(k) is None})
    assert not unroled, (
        f"{len(unroled)} recorded field(s) carry no role: {unroled}. Classify "
        f"each in field_roles (ALGORITHM / WORKLOAD / SWEEP / REVISION / "
        f"PROVENANCE) against the line in train.py that reads it.")


# ─── the label tail ─────────────────────────────────────────────────────────

def test_label_tail_is_restricted_to_declared_defaults():
    """`labels._residual_knobs` may only append fields whose default
    ``OptimizerConfig`` or `run_schema` DECLARES.

    ALGORITHM is far larger than that — it includes constructor spellings
    recorded only inside old runs' optimizer blocks and train.py harness flags —
    and appending those would put a `k=v` token on runs that chose nothing.
    """
    from lora_playground.plotting.labels import _tail_fields
    from lora_playground.run_schema import REVISION_FIELDS as _REV
    from lora_playground.plotting.arms import _config_defaults
    tail = _tail_fields()
    declared = frozenset(_config_defaults()) | frozenset(_REV)
    assert tail <= declared
    assert tail <= fr.fields_with_roles(*fr.LABELLED_ROLES)
    assert PINNED_FIELDS() <= tail          # the pre-refactor tail is preserved
    for name in ("magnitude_rule", "picard_iters", "eps", "training_mode",
                 "target_modules", "max_grad_norm", "batch_size"):
        assert name not in tail, (
            f"{name} has no OptimizerConfig default, so the generic tail cannot "
            f"say whether a recorded value was a choice")


# ─── unknown fields are series-defining, never provenance ───────────────────

def test_unknown_field_has_no_role_and_is_series_defining():
    assert fr.role_of("imagined_future_flag") is None
    assert fr.is_series_defining("imagined_future_flag")


def test_series_id_splits_on_an_unknown_field():
    """The consumer-side statement of the rule above: a field the registry has
    never seen must SPLIT the series, because silently pooling it is how a new
    ``OptimizerConfig`` flag merged two arms."""
    a = {"optimizer": "kl-diag-polar-lora", "lora_r": 16, "lr": 1e-3,
         "imagined_future_flag": None}
    b = dict(a, imagined_future_flag="enabled")
    assert series_id(a) != series_id(b)


def test_series_id_ignores_sweep_and_provenance_roles():
    a = {"optimizer": "kl-diag-polar-lora", "lora_r": 16, "lr": 1e-3,
         "seed": 0, "max_steps": 4000, "device": "cuda",
         "optim_diagnostics_every": 20}
    b = dict(a, lr=3e-4, seed=7, max_steps=9000, device="cpu",
             optim_diagnostics_every=1)
    assert series_id(a) == series_id(b)


def test_fields_with_roles_rejects_an_unknown_role():
    with pytest.raises(KeyError):
        fr.fields_with_roles("not-a-role")


# ─── the label consumer ─────────────────────────────────────────────────────

def test_featured_knobs_all_carry_a_label_bearing_role():
    """A knob `labels` spells by name must have a LABELLED_ROLES role, or
    `canonical_arm_label`'s role exclusion and `_residual_knobs`' duplicate
    suppression cannot see it."""
    from lora_playground.plotting.labels import _FEATURED_KNOBS
    labelled = fr.fields_with_roles(*fr.LABELLED_ROLES)
    for field, _render in _FEATURED_KNOBS:
        assert field in labelled, (
            f"{field!r} is spelled in the label but carries role "
            f"{fr.role_of(field)!r}, not one of {list(fr.LABELLED_ROLES)}")


def test_arm_label_drops_the_revision_role_not_a_regexed_token():
    from lora_playground.plotting.labels import (
        canonical_arm_label, canonical_label,
    )
    cfg = {"optimizer": "adamw", "lr": 1e-4}
    versioned = dict(cfg, optimizer_impl_revision=2,
                     measurement_semantics_revision=1)
    assert canonical_label(versioned) != canonical_label(cfg)
    assert canonical_arm_label(versioned) == canonical_arm_label(cfg)


def test_callable_pin_labels_the_same_as_omitting_the_field():
    """`canonical_label` is fed arm PREDICATE dicts as well as recorded configs
    (`paper_plots_lib._canonical_variant_key` derives an arm's expected label
    from its predicate). A predicate may pin a field with a CALLABLE, meaning
    "any value satisfying this" — which names no value to spell, and is TRUTHY,
    so a truthiness test made the slots-LIVE arm label itself as frozen and
    collide with the slots-frozen arm."""
    from lora_playground.plotting.labels import canonical_arm_label
    base = {"optimizer": "kl-diag-polar-lora", "precond": "factorwise"}
    assert (canonical_arm_label({**base, "freeze_factorwise_slots": lambda v: not v})
            == canonical_arm_label(base))
    # A membership pin is the same case: it names a SET, not a value.
    assert (canonical_arm_label({**base, "cw_metric_init": ("zero", "1e-12")})
            == canonical_arm_label(base))


def test_absent_extra_field_is_not_a_distinguishing_value():
    """`arms.field_matches` treats an absent key as no-match, so an
    ``_EXTRA_ALGORITHM`` flag that exists only on the branch implementing its
    ablation is ABSENT from most cfgs on disk. Absent must read as "at its
    default", or every legacy run would be suffixed."""
    from lora_playground.plotting.labels import canonical_label
    base = {"optimizer": "kl-diag-polar-lora", "precond": "factorwise"}
    assert canonical_label(base) == canonical_label(
        {**base, "freeze_factorwise_slots": None})
    assert "frozen-slots" not in canonical_label(base)
    assert "frozen-slots" not in canonical_label(
        {**base, "freeze_factorwise_slots": False})
    assert "frozen-slots" in canonical_label(
        {**base, "freeze_factorwise_slots": True})
