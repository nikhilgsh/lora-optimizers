"""One registry answering "what does this recorded config field mean?".

The fact this module owns
------------------------
Every consumer of a run's flattened cfg has to decide, per field, whether two
runs that disagree on it are two different things or the same thing recorded
twice. That one fact was re-declared in five independent places with five
different defaults:

  * ``optim_config.OptimizerConfig``'s field list — gates what `arms.arm()`
    can pin at all.
  * ``plotting.arms.PINNED_FIELDS()`` — what an arm predicate pins, plus
    hand-written per-arm exceptions.
  * ``plotting.labels._shared_knobs`` — per-field ``if cfg.get(...)`` lines
    deciding what reaches the display label.
  * ``plotting.dedup._series_items`` — what splits `series_id`.
  * ``manifest.SERIES_AXIS_FIELDS`` / ``run_records.RUNTIME_FIELDS`` — what a
    series is allowed to vary on.

A missed copy is not an error, it is a silently-wrong figure: the label suffix
added for ``optimizer_impl_revision`` made every revision-2 run match no arm;
``freeze_factorwise_slots`` was invisible to ``canonical_label``, so a
frozen-slot run shared a label with the run it forked from; the
KL-Shampoo-implies-factorwise rule was hand-typed in four places.

The five roles
--------------
``ALGORITHM``   the field names a choice about the update rule. Two runs that
                disagree on one are two algorithms: it splits series identity
                and it must reach the display label.
``WORKLOAD``    what was being adapted (model, corpus, rank, data pipeline).
                Splits series identity; carried by the caller's cell selection
                rather than by an arm predicate.
``SWEEP``       the axes a series is *allowed* to vary on and be averaged
                across: learning rate, seed, horizon, eval cadence.
``REVISION``    a semantics-revision counter (``run_schema.REVISION_FIELDS``).
                Splits series identity like ALGORITHM, but no CLI flag sets it,
                so an arm predicate can never pin it — which is why
                `labels.canonical_arm_label` drops exactly this role.
``PROVENANCE``  a record of how the run was executed, not of what it computed
                (``run_records.RUNTIME_FIELDS``): commits, paths, device,
                diagnostics cadence, checkpoint bookkeeping.

Roles are DERIVED wherever a set already owns the fact
------------------------------------------------------
Retyping every field name here would just be a sixth copy, so each role starts
as a projection of a set that already exists and is already tested:

    ALGORITHM  = every non-axis OptimizerConfig field (a superset of
                 arms.PINNED_FIELDS()), plus every constructor knob any
                 REGISTERED optimizer accepts (`_constructor_knobs`), plus the
                 explicit `_EXTRA_ALGORITHM` / `_RECORDED_ONLY_ALGORITHM` sets
    WORKLOAD   = `_WORKLOAD_FIELDS`
    SWEEP      = manifest.SERIES_AXIS_FIELDS - run_records.RUNTIME_FIELDS,
                 minus the two that are workload identity (lora_r/lora_alpha)
    REVISION   = run_schema.REVISION_FIELDS
    PROVENANCE = run_records.RUNTIME_FIELDS | `_EXTRA_PROVENANCE`

and a CONSTRUCTOR spelling inherits the role of the config field
``optim_config.ALIAS`` maps it to, rather than being classified a second time.
The explicit sets carry only what no existing set covers — train.py CLI flags
that are not ``OptimizerConfig`` fields, facts train.py records about the run,
the log-event scaffolding, and knobs that survive only inside old runs'
recorded optimizer blocks. Each of those is justified per field at its
definition, against the line in train.py that reads it.

The import-time disjointness assertion is the point of the module: the source
sets were measured disjoint (``RUNTIME_FIELDS`` a strict subset of
``SERIES_AXIS_FIELDS``, ``PINNED_FIELDS()`` disjoint from both), and if a future
edit puts one field in two roles the answer to "does this distinguish runs?"
becomes ambiguous again. That must fail loudly at import, naming the field.

Unknown fields
--------------
`role_of` returns ``None`` for a field in no set, and callers MUST treat that as
"unknown, therefore series-defining" — see `is_series_defining`. Never as
provenance: a field the registry has never seen is exactly the case that made
new ``OptimizerConfig`` flags silently merge two arms, and the only safe default
is to let it split. Loud beats quiet.
"""
from __future__ import annotations

# Role names. Strings rather than an enum so a role can be printed into an
# error message or a test name without unwrapping.
ALGORITHM = "algorithm"
WORKLOAD = "workload"
SWEEP = "sweep"
REVISION = "revision"
PROVENANCE = "provenance"

ROLE_NAMES: tuple[str, ...] = (ALGORITHM, WORKLOAD, SWEEP, REVISION, PROVENANCE)


class FieldRoleConflict(RuntimeError):
    """One field derived into two roles, so its meaning is ambiguous.

    Raised at import. The fix is at the SOURCE set, not here: decide whether
    the field names an algorithm choice, a workload, a sweep axis, a revision
    counter, or execution provenance, and remove it from the other set.
    """


# ─────────────────────────────────────────────────────────────────────────────
# The explicit sets. Everything else is derived (see `_derive`).
#
# These name the fields that reach a run cfg but that NO existing set covers:
# train.py CLI flags that are not ``OptimizerConfig`` fields, facts train.py
# computes and records itself (`train.py:1688-1709`), the log-event scaffolding,
# and constructor knobs recorded only inside a run's own ``optimizer_config`` /
# ``optimizer_effective`` block (`run_records.logged_effective_config` splices
# those onto the cfg as top-level scalars).
#
# The classifying question is always the same: **can this field change the
# sequence of parameter updates the run computes?**
#   * yes, through the optimizer or through the loss/gradient it sees → ALGORITHM
#   * no, but it defines what was adapted, on what data, at what batch
#     composition → WORKLOAD
#   * no at all — it gates aborting, logging, dumping, compilation, kernels,
#     dataloader workers, or names the record's own structure → PROVENANCE
# ─────────────────────────────────────────────────────────────────────────────

# ALGORITHM, from train.py's CLI and from what train.py records about the
# resolved update. Each is checked against its reader in train.py:
#   * ``optimizer`` / ``requested_optimizer`` — the update rule itself, before
#     and after alias resolution. The single most algorithm-defining field there
#     is, and it carried no role at all.
#   * ``training_mode`` — `lora` vs `svd_step_oracle` vs `svd_cumulative_oracle`
#     vs `galore` vs `ucv` are different updates, not different bookkeeping.
#   * ``rank_constraint`` / ``svd_projection`` — how the step is projected back
#     to rank r (`train.py:1688`, `:1693`).
#   * ``max_grad_norm`` — clipping, passed into the training step
#     (`train.py:2024`); it changes the gradient the optimizer receives.
#   * ``lora_init_b`` / ``lora_dropout`` — the adapter's initialization and its
#     stochasticity; `arms.py` pins ``lora_init_b`` explicitly on the NOMAG and
#     DOUBLE arms, i.e. it is already treated as an arm-defining choice.
#   * ``lr_scheduler_type`` / ``warmup_steps`` — the LR TRAJECTORY. Deliberately
#     not SWEEP: a series may be averaged across learning-rate VALUES, never
#     across two schedules.
#   * ``effective_inner_polar`` / ``effective_picard_iters`` /
#     ``effective_polar_pre_norm`` — the optimizer's own record of the update it
#     actually applied, which is why project notes prefer them to the raw knobs.
_EXTRA_ALGORITHM: frozenset[str] = frozenset({
    "optimizer", "requested_optimizer", "training_mode",
    "rank_constraint", "svd_projection", "max_grad_norm",
    "lora_init_b", "lora_dropout",
    "lr_scheduler_type", "warmup_steps",
    "effective_inner_polar", "effective_picard_iters",
    "effective_polar_pre_norm",
})

# ALGORITHM, recorded ONLY inside a run's optimizer blocks — no constructor of
# any REGISTERED optimizer accepts them today, so the constructor walk below
# cannot see them and there is no default to compare a recorded value against.
# They are still algorithm choices, and `dedup._series_items`' third tier is
# what keeps a value with no derivable default out of series identity; the role
# is what puts it in the display label.
#   * retired curvature-whiten ablation flags: ``cw_picard_mode``,
#     ``cw_no_rr_precond``, ``cw_eigh_seed``, ``freeze_factorwise_slots``
#     (the last is why `labels._FEATURED_KNOBS` can spell ` frozen-slots` at
#     all — nothing else in the codebase can see that flag).
#   * a side-branch Riemannian-Muon optimizer's own block (14 runs):
#     ``nesterov``, ``wd``, ``adamw_betas``, ``adamw_eps``, ``adamw_params``,
#     ``muon_params``, ``lora_precond``, ``lora_precond_eps``,
#     ``lora_riemannian_*``. ``*_params`` name which parameter groups went to
#     which sub-optimizer, which is part of the update, not a log detail.
_RECORDED_ONLY_ALGORITHM: frozenset[str] = frozenset({
    "cw_picard_mode", "cw_no_rr_precond", "cw_eigh_seed",
    "freeze_factorwise_slots",
    "nesterov", "wd", "adamw_betas", "adamw_eps", "adamw_params",
    "muon_params", "lora_precond", "lora_precond_eps",
    "lora_riemannian_adjust_lr", "lora_riemannian_muon",
    "lora_riemannian_ortho_method", "lora_riemannian_variant",
})

# WORKLOAD: what was adapted, with what adapter shape, on which data, at what
# batch composition, through which data pipeline.
# ``lora_r``/``lora_alpha`` are also `manifest.SERIES_AXIS_FIELDS` members (a
# rank-transfer figure varies them along one axis), so they are subtracted from
# SWEEP below to keep the roles disjoint; the axis membership they need is
# unchanged, because `dedup._series_items` still receives ``SERIES_AXIS_FIELDS``
# itself.
#   * adapter placement — ``target_modules`` (the selector),
#     ``target_module_names`` / ``target_module_count`` (what it resolved to),
#     ``exclude_lm_head_from_all_linear``.
#   * the corpus and how much of it — ``dataset_name``, ``dataset_config``,
#     ``train_split`` / ``eval_split``, ``max_train_samples`` /
#     ``max_eval_samples`` / ``eval_fraction``, the realized
#     ``train_samples`` / ``eval_samples`` / ``docs_per_slot_mean``,
#     ``max_seq_length``.
#   * tokens per step — ``global_batch_size`` and the composition that reaches
#     it (``batch_size``, ``per_rank_batch_size``, ``grad_accum_steps``,
#     ``world_size``). All five are WORKLOAD, not PROVENANCE, so two runs at
#     different batch composition stay two series. Which of them an ARM should
#     pin is a separate question that `arms.py` already answers: pin
#     ``global_batch_size``, because every 9000-step paper run is global 16
#     while the composition differs (Llama-3-8B is 2x8).
_WORKLOAD_FIELDS: frozenset[str] = frozenset({
    "model_name", "data_dir", "lora_r", "lora_alpha", "data_pipeline_version",
    "target_modules", "target_module_names", "target_module_count",
    "exclude_lm_head_from_all_linear",
    "dataset_name", "dataset_config", "train_split", "eval_split",
    "max_train_samples", "max_eval_samples", "eval_fraction",
    "train_samples", "eval_samples", "docs_per_slot_mean",
    "max_seq_length",
    "global_batch_size", "batch_size", "per_rank_batch_size",
    "grad_accum_steps", "world_size",
})

# PROVENANCE beyond ``run_records.RUNTIME_FIELDS``. None of these can reach the
# optimizer or the loss; each was checked at its reader in train.py:
#   * stop conditions — ``abort_on_nan_eval`` (`train.py:2144`),
#     ``abort_on_eval_loss_above`` (`:2153`), ``target_eval_loss`` (`:2165`),
#     ``allow_multi_epoch`` (`:1356`, a refusal guard). They can END a run
#     early; they change no update it computes, exactly like the
#     checkpoint-retention flags already in ``RUNTIME_FIELDS``.
#   * logging / dumping cadence — ``train_loss_every`` (`:2056`),
#     ``log_diagnostics`` (a legacy toggle), ``debug_higham_residual``
#     (`:1254`), ``snapshot_dir`` / ``snapshot_steps`` (`:1967`), and the
#     ``optim_heldout_*`` / ``optim_small_slot_microbatch_probe`` probes
#     (`:1591-1605`).
#   * execution / kernel / precision choices — ``bf16``, ``compile``,
#     ``compile_mode``, ``attn_implementation``, ``use_liger`` /
#     ``liger_family`` (`:1446`, `:1709`), ``gradient_checkpointing``
#     (`:1448`), ``num_workers``. Same kind as ``tf32`` / ``no_tf32`` /
#     ``device``, which ``RUNTIME_FIELDS`` already owns: they move throughput
#     and low-order numerics, not the algorithm.
#   * resume mechanics — ``resume_debug_replay`` /
#     ``resume_replay_original_dataloader`` (`:100-106`), siblings of the
#     ``resume_from`` / ``_resume`` fields ``RUNTIME_FIELDS`` already owns.
#   * the record's own structure — ``event`` and the composite blocks
#     ``optimizer_config`` / ``optimizer_effective`` /
#     ``optimizer_variant_semantics`` / ``semantic_revisions``, whose CONTENTS
#     are flattened onto the cfg and classified individually.
_EXTRA_PROVENANCE: frozenset[str] = frozenset({
    "abort_on_nan_eval", "abort_on_eval_loss_above", "target_eval_loss",
    "allow_multi_epoch",
    "train_loss_every", "log_diagnostics", "debug_higham_residual",
    "snapshot_dir", "snapshot_steps",
    "optim_heldout_probe", "optim_heldout_probe_batches",
    "optim_heldout_probe_exit", "optim_heldout_identity_scale",
    "optim_small_slot_microbatch_probe",
    "bf16", "compile", "compile_mode", "attn_implementation",
    "use_liger", "liger_family", "gradient_checkpointing", "num_workers",
    "resume_debug_replay", "resume_replay_original_dataloader",
    "event", "optimizer_config", "optimizer_effective",
    "optimizer_variant_semantics", "semantic_revisions",
})


def _constructor_knobs() -> frozenset[str]:
    """Every name a REGISTERED optimizer's constructor accepts, or that its spec
    pins for it — i.e. every knob that can reach an optimizer's update.

    Derived, not typed: the same MRO-aware walk `dedup._constructor_defaults`
    and `arms._inert_fields` use, plus each spec's ``defaults``/``fixed`` keys,
    over the whole registry (4 ms, ~110 names). This is what classifies the
    constructor spellings a run records inside its ``optimizer_config`` block —
    ``betas``, ``eps``, ``diag_metric``, ``magnitude_rule``, ``picard_iters``,
    ``operator_type``, ``flat_outer`` and the rest — without listing any of them.

    A bare ``inspect.signature`` would not do: most polar variants declare
    ``__init__(self, model, **kwargs)`` and delegate, so it sees 3 parameters
    for ``adam-soap-polar-product-lora`` against 57 real ones.
    """
    from .constructor_introspection import forwardable_constructor_parameters
    from .optim_specs import REGISTRY
    names: set[str] = set()
    for spec in REGISTRY.values():
        if spec.cls is not None:
            try:
                names |= {p.name
                          for p in forwardable_constructor_parameters(spec.cls)}
            except (TypeError, ValueError):
                pass
        names |= set(spec.defaults or {}) | set(spec.fixed or {})
    return frozenset(names)


def _derive() -> dict[str, frozenset[str]]:
    """``{role: fields}``, projected from the sets that already own each fact.

    Three tiers, in precedence order. A later tier only fills fields no earlier
    tier claimed, so the tiers are a FALLBACK CHAIN, not competing opinions:

      1. the existing sets plus the explicit sets above;
      2. ``optim_config.ALIAS`` — a CONSTRUCTOR spelling inherits the role of
         the config field it aliases, rather than being classified twice. This
         is what makes ctor ``delta`` ALGORITHM (it is ``precond_delta``) and
         ctor ``diagnostics_every`` PROVENANCE (it is
         ``optim_diagnostics_every``, a ``RUNTIME_FIELDS`` member). Listing both
         spellings independently is precisely how the two used to disagree;
      3. `_constructor_knobs` — anything else an optimizer constructor accepts
         is ALGORITHM.

    Tier 2 must precede tier 3: ``diagnostics_every`` is BOTH an alias of a
    provenance field and a constructor parameter of 56 optimizers, and the
    provenance answer is the right one.
    """
    from .manifest import SERIES_AXIS_FIELDS
    from .optim_config import ALIAS, CONFIG_FIELDS
    from .plotting.arms import PINNED_FIELDS
    from .run_records import RUNTIME_FIELDS
    from .run_schema import REVISION_FIELDS as _REVISION_FIELDS

    # `PINNED_FIELDS()` is `OptimizerConfig` minus the per-series axes minus the
    # fields with no CLI flag. That last exclusion is about what an ARM can
    # match on (no run cfg carries `muon_alpha`/`muon_rank`), not about what the
    # field MEANS, so the role set takes every non-axis config field. Derived
    # from `CONFIG_FIELDS`, so a new no-CLI-flag config field is classified
    # without an edit here.
    config_algorithm = (frozenset(CONFIG_FIELDS)
                        - frozenset(SERIES_AXIS_FIELDS)
                        - frozenset(_REVISION_FIELDS))
    roles = {
        ALGORITHM: (PINNED_FIELDS() | config_algorithm | _EXTRA_ALGORITHM
                    | _RECORDED_ONLY_ALGORITHM),
        WORKLOAD: _WORKLOAD_FIELDS,
        # The measured identity `SERIES_AXIS_FIELDS - RUNTIME_FIELDS` is exactly
        # {eval_every, lora_alpha, lora_r, lr, max_steps, seed}; the two adapter
        # -shape fields are workload identity, and the remaining four are the
        # axes a series may be averaged across.
        SWEEP: frozenset(SERIES_AXIS_FIELDS) - frozenset(RUNTIME_FIELDS)
        - _WORKLOAD_FIELDS,
        REVISION: frozenset(_REVISION_FIELDS),
        PROVENANCE: frozenset(RUNTIME_FIELDS) | _EXTRA_PROVENANCE,
    }
    # Tier 1 is where a genuine contradiction lives, so it is asserted BEFORE
    # the fallback tiers subtract anything.
    _assert_disjoint(roles)

    assigned = frozenset().union(*roles.values())
    aliased: dict[str, set[str]] = {role: set() for role in roles}
    for ctor_name, config_name in ALIAS.items():
        if ctor_name in assigned:
            continue
        for role, names in roles.items():
            if config_name in names:
                aliased[role].add(ctor_name)
                break
    roles = {role: names | frozenset(aliased[role])
             for role, names in roles.items()}

    assigned = frozenset().union(*roles.values())
    roles[ALGORITHM] |= (_constructor_knobs() - assigned)
    return roles


def _assert_disjoint(fields_by_role: dict[str, frozenset[str]]) -> None:
    conflicts: dict[str, list[str]] = {}
    for role, names in fields_by_role.items():
        for other, other_names in fields_by_role.items():
            if other <= role:                 # each unordered pair once
                continue
            for name in sorted(names & other_names):
                conflicts.setdefault(name, []).append(f"{role}+{other}")
    if conflicts:
        detail = "; ".join(
            f"{name!r} in {' and '.join(pairs)}"
            for name, pairs in sorted(conflicts.items())
        )
        raise FieldRoleConflict(
            f"{len(conflicts)} field(s) derived into more than one role: "
            f"{detail}. A field with two roles has no answer to 'does it "
            f"distinguish runs?'. Fix the source set (arms.PINNED_FIELDS / "
            f"manifest.SERIES_AXIS_FIELDS / run_records.RUNTIME_FIELDS / "
            f"run_schema.REVISION_FIELDS / field_roles._WORKLOAD_FIELDS), not "
            f"this assertion."
        )


FIELDS_BY_ROLE: dict[str, frozenset[str]] = _derive()
_assert_disjoint(FIELDS_BY_ROLE)

ALGORITHM_FIELDS: frozenset[str] = FIELDS_BY_ROLE[ALGORITHM]
WORKLOAD_FIELDS: frozenset[str] = FIELDS_BY_ROLE[WORKLOAD]
SWEEP_FIELDS: frozenset[str] = FIELDS_BY_ROLE[SWEEP]
REVISION_FIELDS: frozenset[str] = FIELDS_BY_ROLE[REVISION]
PROVENANCE_FIELDS: frozenset[str] = FIELDS_BY_ROLE[PROVENANCE]

_ROLE_OF: dict[str, str] = {
    name: role for role, names in FIELDS_BY_ROLE.items() for name in names
}

# Roles whose fields make two runs two different things. A field with NO role is
# series-defining too — see `is_series_defining`. SWEEP and PROVENANCE are the
# only roles a run may differ on and still be the same series.
SERIES_DEFINING_ROLES: frozenset[str] = frozenset({ALGORITHM, WORKLOAD, REVISION})

# Roles that must reach the display label, so a label can never collapse two
# distinct series. WORKLOAD is absent because the caller's cell selection
# carries it (one figure panel is one model/corpus/rank).
LABELLED_ROLES: tuple[str, ...] = (ALGORITHM, REVISION)


def role_of(field: str) -> str | None:
    """The field's role, or ``None`` when the registry has never seen it.

    ``None`` means "unknown, therefore series-defining" to every caller — never
    "provenance". See the module docstring.
    """
    return _ROLE_OF.get(field)


def fields_with_roles(*roles: str) -> frozenset[str]:
    """Union of the named roles' fields. Raises on an unknown role name."""
    unknown = sorted(set(roles) - set(FIELDS_BY_ROLE))
    if unknown:
        raise KeyError(
            f"unknown role(s) {unknown}; known roles are {list(ROLE_NAMES)}"
        )
    out: frozenset[str] = frozenset()
    for role in roles:
        out |= FIELDS_BY_ROLE[role]
    return out


def is_series_defining(field: str) -> bool:
    """Whether disagreement on ``field`` makes two runs two different series.

    True for ALGORITHM, WORKLOAD, REVISION and — deliberately — for a field in
    no role at all. False only for SWEEP and PROVENANCE, the two roles a series
    is allowed to vary on.
    """
    role = role_of(field)
    return role is None or role in SERIES_DEFINING_ROLES


__all__ = [
    "ALGORITHM",
    "ALGORITHM_FIELDS",
    "FIELDS_BY_ROLE",
    "FieldRoleConflict",
    "LABELLED_ROLES",
    "PROVENANCE",
    "PROVENANCE_FIELDS",
    "REVISION",
    "REVISION_FIELDS",
    "ROLE_NAMES",
    "SERIES_DEFINING_ROLES",
    "SWEEP",
    "SWEEP_FIELDS",
    "WORKLOAD",
    "WORKLOAD_FIELDS",
    "fields_with_roles",
    "is_series_defining",
    "role_of",
]
