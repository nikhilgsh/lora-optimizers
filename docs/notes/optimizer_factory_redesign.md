# Optimizer factory redesign — robust-by-construction `build_optimizer`

## Motivation

The same bug has shipped four times: a tunable flag reaches the `config` event but
not the optimizer (or reaches it as a hardcoded literal), so a run executes a
different hyperparameter than its log records.

| flag | symptom | fix commit |
|------|---------|-----------|
| `cw_nesterov` | dropped by the kl-diag branch | `7b7f807` |
| `cw_no_radius`/`cw_no_diag_curv`/`cw_factor_a`/`cw_factor_b` | dropped | `c838e5c` |
| `beta1` | hardcoded `betas=(0.9,0.999)` in 41 sites → **every** kl-diag/diag-shampoo run ran β₁=0.9 while logging 0.95 | (this work) |
| `curvature_whitening` | dropped by every `AdamPolarProductLoRA` branch except `…-chord-tight-clean` (latent) | (this work) |

These are not bad luck. The root cause is structural: a flag's journey has **four
hand-maintained touchpoints with no single source of truth** —

1. argparse (`--beta1`) in `train.py`
2. the `build_optimizer` signature param (`beta1`)
3. **the branch passing it to the class** — the *drop* point
4. **the `config` event re-listing it** (`"beta1": args.beta1`) — the *lie* point

`build_optimizer` is a ~1240-line if/elif where ~60 branches each **manually
re-list** their kwargs (the chord-tight-clean branch lists 45), and the config
event manually re-lists ~80 args. (3) and (4) drift independently from each other
and from the optimizer's actual `__init__`. Each prior fix patched one leak.

The post-audit state (forwarding fix + the `getattr(optimizer, …)` config-event
patch) stops the bleeding, but those are band-aids over the missing architecture.
This doc specifies the architecture.

## Design — declare once, derive the rest

### 1. `OptimizerConfig` — the single source of truth

One frozen dataclass holds every tunable optimizer field. `train.py` populates it
from argparse; `build_optimizer` consumes it; **it is logged directly as the config
event** (`asdict(config)`), so (4) can never drift from (1) — there is no second list.

```python
@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    precond_delta: float = 1e-6
    precond_delta_relative: bool = False
    curvature_beta: float = 0.99
    curvature_whitening: bool = False
    precond_refresh_every: int = 1
    precond_method: str = "higham"
    higham_iters: int = 10
    higham_compute_dtype: str = "fp32"
    muon_ns_steps: int = 5
    muon_alpha: int = 16
    muon_rank: int = 16
    polar_method: str = "ns"
    polar_norm_dir: str = "frob"
    polar_sigma_power: float | None = None
    lora_plus_multiplier: float = 1.0
    cw_picard_iters: int = 1
    cw_nesterov: bool = False
    cw_no_radius: bool = False
    cw_no_diag_curv: bool = False
    cw_factor_a: float = 0.0
    cw_factor_b: float = 0.0
    picard_iters_override: int | None = None
    picard_alpha: float = 1.0
    ns_form: str = "gram"
    optim_diagnostics_every: int = 20
    log_basic_diagnostics: bool = False
    log_heavy_diagnostics: bool = False
    # … ssc_*, anderson_*, debug_*, log_non_finite* (the long tail)
```

Structural constants (`eps`, `magnitude_rule`, `operator_type`, `kl_coupled`, …)
are **not** here — they belong to the optimizer's identity (the spec), not the
user's config.

### 2. The `ALIAS` table — constructor-kwarg ← config-field

Most constructor params name-match a config field (`curvature_beta` ↔
`curvature_beta`). The exceptions are a small, enumerable table:

```python
ALIAS = {
    "ns_steps":            "muon_ns_steps",
    "delta":               "precond_delta",
    "lr_b_multiplier":     "lora_plus_multiplier",   # Muon-family classes
    "diagnostics_every":   "optim_diagnostics_every",
    "alpha":               "muon_alpha",
    "rank":                "muon_rank",
    "gamma":               "precond_gamma",
    "ema_beta":            "precond_ema_beta",
    # `betas` is special-cased: (config.beta1, config.beta2)
    # `picard_iters` is special-cased: override-or-spec-default (see §4)
}
```

### 3. `OptimizerSpec` — the declarative branch

Each optimizer is `(class, fixed, defaults)`:

```python
@register("adam-polar-product-lora-coupled-spectral-chord-tight-clean")
SPEC = OptimizerSpec(
    cls=AdamPolarProductLoRA,
    fixed={"magnitude_rule": "spectral_chord_tight_clean", "eps": 1e-8},
    defaults={"picard_iters": 1},   # per-optimizer default for an auto-forwarded kwarg
)
```

`fixed` = this optimizer's identity (the values that make it *this* optimizer, not
a sibling). `defaults` = a per-optimizer default for a kwarg that is otherwise
config-forwarded (e.g. `picard_iters` defaults to 1 here, 3 for spectral-chord, 2
for end-rms). **This 3-line spec replaces the 45-kwarg hand-listed branch.**

The `-polar`/non-`-polar` pairs become two specs differing only in
`fixed={"use_polar": True/False}`. The targets-based optimizers (galore, svd)
carry `takes_targets=True` so the builder passes `targets` instead of `model`.

### 4. The generic builder — forwarding is automatic

```python
def build_optimizer(model_or_targets, name, config: OptimizerConfig):
    spec = REGISTRY[name]
    params = init_params(spec.cls)            # inspect.signature, minus self/model/targets/lr
    kwargs = {}
    for p in params:
        if p in spec.fixed:                    # identity constant
            kwargs[p] = spec.fixed[p]
        elif p == "betas":
            kwargs["betas"] = (config.beta1, config.beta2)
        elif p == "picard_iters":
            kwargs[p] = (config.picard_iters_override
                         if config.picard_iters_override is not None
                         else spec.defaults.get(p, _CLASS_DEFAULT))
        elif (field := ALIAS.get(p, p)) in CONFIG_FIELDS:
            kwargs[p] = getattr(config, field)  # auto-forward by name/alias
        # else: not a config field and not fixed → leave to the class default
    return spec.cls(targets if spec.takes_targets else model_or_targets,
                    lr=config.lr, **kwargs)
```

**A silent drop is now structurally impossible.** Forwarding happens by
introspection over the class signature; to *not* forward a param you must place it
in `spec.fixed` — a visible, intentional, single-line declaration. A param the
class accepts and the config supplies is *always* forwarded.

### 5. `effective_config()` — surface runtime-resolved behavior

Optimizers that compute/clamp values at runtime (the clip operator's
`effective_inner_polar`, the resolved Picard depth) expose `effective_config()`.
The config event is `asdict(config) | optimizer.effective_config()`. This closes
Codex's clip finding (clip branches inheriting the polar class's `effective_config`
and mis-reporting `effective_inner_polar='ns'`): the clip spec/class reports its
true operator.

## Why this kills each class

- **Silent drop (beta1, cw_*, curvature_whitening):** forwarding is automatic; a
  drop requires an explicit `fixed=` entry. The forwarding guardrail test becomes a
  *consequence* of the architecture, not a patch over it.
- **Config lie:** the config event *is* the config object — no re-listing to drift.
- **Effective-config mismatch (clip):** unified `effective_config()` contract.
- **Copy-paste branch divergence:** two specs can't accidentally be identical (a
  test asserts spec uniqueness); the `-polar` pairs differ only in one `fixed` key.

This is the "robust by construction" philosophy already enforced elsewhere in the
repo (the sweep-manifest contract, the timing-registry wall guard, the
checkpoint-wiring test with zero exemptions) — `build_optimizer` is the one
subsystem that never got it.

## Migration plan — behavioral-equivalence-gated, incremental

The factory builds the exact 60 optimizers the paper depends on, so the migration
must be **provably behavior-preserving**, verified per optimizer.

**Gate (the safety mechanism):** `tests/test_build_optimizer_equivalence.py` — for
every `OPTIMIZER_CHOICES` entry, build via the OLD `build_optimizer(**flags)` and
the NEW `build(config)` on a tiny multi-shape LoRA model with identical
non-default flags, and assert:
1. the two optimizers have identical stored attributes (`vars()` scalar diff), and
2. they step identically — N steps on seeded deterministic grads, compare `dA/dB`
   and moment buffers to fp32 tolerance (the pattern in
   `tests/test_polar_product_batched_equivalence.py`).

A migrated optimizer is not "done" until its equivalence test passes.

**Order (paper-critical first):**
1. Scaffolding: `OptimizerConfig`, `ALIAS`, `OptimizerSpec`, `register`, the generic
   builder — all *alongside* the existing `build_optimizer` (new entry point
   `build_optimizer_v2`, old one untouched).
2. **Curvature-whiten family** (5 specs: kl-diag(-polar), diag-shampoo(-polar),
   kl-diag-polar-flatout, kl-shampoo(-polar), curvature-whiten(-polar)) — the paper
   protagonist. Fold the Phase-1 `precond_method` plumbing into the cw spec here
   (it becomes one ALIAS line + the class change, no branch edit).
3. Polar-product family (~25 variants — the richest `fixed`/`magnitude_rule` set).
4. Muon / coupled-core family (name-mismatch `lr_b_multiplier`; the deliberate
   `beta1=0.95` becomes `fixed={"beta1": 0.95}` — intentional and visible).
5. Baselines + targets-based (adamw, lin/scaled, galore, svd, adafactor).
6. Switch `train.py`: populate `OptimizerConfig` from args, pass to the builder,
   log `asdict(config) | effective_config()`. Delete the manual config dict.
7. Retire the if/elif; `build_optimizer` becomes the generic dispatcher; keep the
   name for callers.

**Invariants enforced by tests throughout:**
- every `OPTIMIZER_CHOICES` name has exactly one spec (no orphan, no dup);
- every `spec.fixed` key is a real `__init__` param of `spec.cls`;
- every `OptimizerConfig` field is consumed by ≥1 spec (no dead config field);
- the equivalence gate passes for all 60.

## Status

- [x] forwarding fix + config-event `getattr` band-aid (keeps things correct during migration)
- [x] `tests/test_build_optimizer_forwarding.py` (forwarding guardrail; subsumed by the architecture later)
- [ ] scaffolding (OptimizerConfig/ALIAS/OptimizerSpec/builder)
- [ ] equivalence-gate test
- [ ] migrate cw family (+ fold Phase-1 precond_method)
- [ ] migrate polar-product / muon / baselines / targets
- [ ] switch train.py to OptimizerConfig; retire if/elif
