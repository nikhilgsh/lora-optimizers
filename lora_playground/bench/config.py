"""Production-config-first bench configuration.

The footgun this guards (CLAUDE.md "Benchmark the PRODUCTION config, not library
defaults"): ``build_optimizer`` defaults ``precond_method="eigh"`` and
``muon_ns_steps=5`` etc., while every production sweep passes its own flags
(``--polar_method polar_express --muon_ns_steps 8 ...``). Benchmarking the
argparse/build defaults silently measures a code path production never runs.

A ``BenchConfig`` carries BOTH the fixed protocol (model/rank/batch/seq/dtype/
compile/data pipeline — held constant across optimizer comparisons) AND the
optimizer-specific knobs, and knows how to emit ``build_optimizer`` kwargs and a
human-readable resolved-config banner. Named factories (``protagonist_config``)
encode the exact flags a sweep wrapper passes, with the wrapper cited as the
source of truth so the two cannot drift silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# Diagnostics cadence for TIMING benches. NEVER 1 (the per-pair B@dA matmul +
# float() syncs add seconds/step — CLAUDE.md "Timing benches: --optim_diagnostics_every 1
# is a trap"). 80 matches the production sbatch default; the harness forces
# log_basic_diagnostics off on top of this.
_TIMING_DIAG_EVERY = 80


@dataclass
class BenchConfig:
    """One benchmark cell: a (model, rank, optimizer) at a fixed protocol.

    Fields left at their defaults reproduce the project protocol (packed_v1,
    bf16, compile, seq 2048, global batch 16 via bs4×ga4). Optimizer knobs
    default to the *protagonist* production values, NOT build_optimizer defaults
    — override per family via the factories below or explicit kwargs.
    """

    model_name: str
    lora_r: int
    optimizer: str
    hardware: str = "blackwell"

    # ── fixed protocol (CLAUDE.md "Fix across all optimizer comparisons") ──
    lora_alpha: Optional[int] = None        # None → lora_r (project convention)
    target_modules: str = "all-linear"
    batch_size: int = 4                     # per-microbatch; 8B uses 2 (see factory)
    grad_accum_steps: int = 4               # 8B uses 8 → global batch 16 either way
    max_seq_length: int = 2048
    bf16: bool = True
    compile_mode: Optional[str] = "default"  # None → eager (smoke/debug only)
    data_pipeline_version: str = "packed_v1"

    # ── optimizer knobs (protagonist production values; cite the wrapper) ──
    muon_ns_steps: int = 8                   # full polar (PE8); build default is 5
    polar_method: str = "polar_express"      # build default is "ns"
    precond_method: str = "eigh"             # protagonist = QR today (no --precond_method
                                             #   in sweep_protagonist_generic.sh); Phase-1
                                             #   wires "higham" as the alternative
    higham_iters: int = 10
    precond_refresh_every: int = 10
    precond_delta: float = 1e-4
    precond_delta_relative: bool = False
    curvature_beta: float = 0.99
    cw_picard_iters: int = 1
    cw_nesterov: bool = True
    beta1: float = 0.95
    beta2: float = 0.999
    lora_plus_multiplier: float = 1.0
    ns_form: str = "gram"

    # ── timing hygiene ──
    optim_diagnostics_every: int = _TIMING_DIAG_EVERY

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r

    @property
    def dtype_str(self) -> str:
        return "bf16" if self.bf16 else "fp32"

    def build_optimizer_kwargs(self, *, precond_method: Optional[str] = None) -> dict:
        """Kwargs for ``lora_playground.optim.build_optimizer`` (minus the model
        and ``optimizer_type``/``lr``, which the caller supplies). ``precond_method``
        override lets the retiming bench flip QR↔Higham at the call site."""
        return dict(
            muon_ns_steps=self.muon_ns_steps,
            polar_method=self.polar_method,
            precond_method=(precond_method or self.precond_method),
            higham_iters=self.higham_iters,
            precond_refresh_every=self.precond_refresh_every,
            precond_delta=self.precond_delta,
            precond_delta_relative=self.precond_delta_relative,
            curvature_beta=self.curvature_beta,
            cw_picard_iters=self.cw_picard_iters,
            cw_nesterov=self.cw_nesterov,
            beta1=self.beta1,
            beta2=self.beta2,
            lora_plus_multiplier=self.lora_plus_multiplier,
            ns_form=self.ns_form,
            # timing hygiene: diagnostics OFF, not just cadence-throttled
            log_basic_diagnostics=False,
            log_heavy_diagnostics=False,
            optim_diagnostics_every=self.optim_diagnostics_every,
        )

    def banner(self) -> str:
        """One-line resolved config printed at the top of every bench so a run is
        self-documenting and a wrong-config bench is obvious in the log."""
        d = asdict(self)
        keys = ["hardware", "model_name", "lora_r", "optimizer", "precond_method",
                "polar_method", "muon_ns_steps", "cw_picard_iters", "precond_refresh_every",
                "precond_delta", "curvature_beta", "beta1", "batch_size", "grad_accum_steps",
                "max_seq_length", "dtype", "compile_mode", "data_pipeline_version"]
        d["dtype"] = self.dtype_str
        return "BenchConfig(" + ", ".join(f"{k}={d.get(k)}" for k in keys) + ")"


# ── Named production configs (cite the wrapper that defines the flags) ────────

def protagonist_config(model_name: str, lora_r: int, **overrides) -> BenchConfig:
    """The paper protagonist `kl-diag-polar-lora`, exactly as
    `scripts/sweep/sweep_protagonist_generic.sh` passes it (post-2026-06-11 pivot):
      --polar_method polar_express --muon_ns_steps 8 --cw_picard_iters 1
      --curvature_beta 0.99 --precond_refresh_every 10 --precond_delta 1e-4
      --cw_nesterov --beta1 0.9 --precond_method gram_ns --higham_iters 8.
    8B uses bs2/ga8 (sweep_protagonist_8b.sh); pass batch_size=2, grad_accum_steps=8.
    """
    cfg = dict(
        model_name=model_name, lora_r=lora_r, optimizer="kl-diag-polar-lora",
        muon_ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
        curvature_beta=0.99, precond_refresh_every=10, precond_delta=1e-4,
        cw_nesterov=True, beta1=0.9, precond_method="gram_ns", higham_iters=8,
    )
    cfg.update(overrides)
    return BenchConfig(**cfg)
