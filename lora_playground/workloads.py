"""Single source of truth for the leaderboard (model, dataset, rank) cells.

Both the doc generator (`scripts/analysis/build_leaderboard_doc.py`) and the
leaderboard notebooks (`notebooks/leaderboard/*.ipynb`) discover their runs from
this registry instead of hand-maintaining log-group lists (which drift and
silently miss whole campaigns).

Discovery is **predicate-based**: a cell is every completed long-horizon run at a
fixed `(model_name, lora_r)` whose resolved dataset matches, minus a deny-pattern
for probe/snapshot/intervention groups. New sweeps join a cell automatically.

Discovery deliberately keys on **cfg fields only**. Scope tags and
`data_pipeline_version` are NOT used — `phase_L` is applied inconsistently (the
eps_rel=1e-3, r64-epsrel and lrext campaigns lack it) and some manifests carry
string-splatted corrupt scopes. The horizon floor (`max_steps >= 8000`) cleanly
separates the 9000-step phase-L / robustness runs from the 4k/2k regimes.

This module is a leaf: it imports nothing from `lora_playground` at import time
(`load_runs` is imported lazily inside `workload_runs`) to avoid an import cycle
with the loader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = str(_REPO_ROOT / "logs")

# ── dataset resolution ───────────────────────────────────────────────────────
# A run's dataset is NOT cfg["dataset_name"] (that stays at the Magicoder
# argparse default for every packed-data-dir run). It is encoded in --data_dir
# inside cfg["command"].
_DATA_DIR_RE = re.compile(r"--data_dir\s+(\S+)")
_DATASET_SUBSTRINGS: tuple[tuple[str, str], ...] = (  # ordered; first match wins
    ("opc_sft_stage2", "opc"),
    ("aya_bengali", "bengali"),
    ("openmath", "openmath"),
    ("tulu3", "tulu3"),
    ("magicoder_seq512", "magicoder"),  # legacy; never a leaderboard cell
)


def resolve_dataset(cfg: dict) -> str | None:
    """Dataset id parsed from --data_dir in cfg['command'].

    Returns None when no --data_dir is present. Never consults
    cfg['dataset_name'] (stale Magicoder default).
    """
    m = _DATA_DIR_RE.search(cfg.get("command") or "")
    if not m:
        return None
    dd = m.group(1)
    for sub, name in _DATASET_SUBSTRINGS:
        if sub in dd:
            return name
    return None


# ── deny-list: probe / intervention / snapshot / smoke groups ────────────────
# These pass the max_steps>=8000 gate but are not leaderboard material; some
# share a (canonical_label, lr) with a real baseline and would trip the
# label-discrimination guard, so they must be removed before aggregation.
# NOTE: do NOT add a bare "_probe_" here — it would wrongly exclude the legit
# `chord_tight_phase_L_r256_epsrel_ns_probe_blackwell_v2` group (the ε_rel=1e-2
# arm). Diagnostic probes (`h1_pre_probe_*`) are sub-8000-step and already
# dropped by the horizon floor.
EXCLUDE_GROUP_PATTERNS: tuple[str, ...] = (
    r"_snapshot",     # *_snapshot_* one-off diagnostics
    r"_slack_",       # chord_tight_slack_*_lr2_* slack/headroom probes
    r"_smoke",
    r"_verify",
    r"_intervention",
)
_EXCLUDE_RE = re.compile("|".join(EXCLUDE_GROUP_PATTERNS))

# Allow-list overrides the deny patterns. `chord_tight_slack_phase_L_1b_r256_lr2_blackwell`
# matches `_slack_` but its only ≥8000-step opc-r256 runs are the legit
# `chord-tight ns=5 k=1 (abs=1e-6)` low-lr points (lr 3e-3/1e-2) absent from the
# phase_L sweep — without them that arm's lr curve looks pinned at its lowest swept
# lr (3e-2) when the optimum is interior. Scoping (opc, ≥8000 steps) already drops
# this group's cross-dataset/4k collisions, so it adds exactly those 2 cells with
# no label-discrimination trip (verified). The `_slack_` pattern still denies every
# OTHER slack-probe group.
ALLOW_GROUPS: frozenset[str] = frozenset({
    "chord_tight_slack_phase_L_1b_r256_lr2_blackwell",
})


def _denied(group: str | None) -> bool:
    if group in ALLOW_GROUPS:
        return False
    return bool(group) and bool(_EXCLUDE_RE.search(group))


# ── workload declaration ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Workload:
    model_name: str          # cfg["model_name"] — a load predicate
    dataset: str             # resolve_dataset() output: opc / openmath / tulu3
    rank: int                # cfg["lora_r"] — a load predicate
    model_display: str       # e.g. "OLMo-2-1B"
    dataset_display: str     # e.g. "opc-sft-stage2 (Magicoder)"
    horizon: int             # speed-to-target horizon (9000; slack absorbs tulu3's 8970)
    sigma_ref: float         # AdamW noise-floor proxy for Δ/σ reporting
    sigma_is_proxy: bool     # True unless a same-cell multiseed σ at this horizon exists
    min_completed_steps: int = 8000  # max_steps discovery floor

    @property
    def label(self) -> str:
        """Compact cross-setting key, e.g. 'OLMo/opc/r64'."""
        return f"{self.model_display.split('-')[0]}/{self.dataset}/r{self.rank}"

    @property
    def title(self) -> str:
        """Per-section doc heading, e.g. 'OLMo-2-1B × opc-sft-stage2 (Magicoder) × r=256'."""
        return f"{self.model_display} × {self.dataset_display} × r={self.rank}"


_OPC_DISPLAY = "opc-sft-stage2 (Magicoder)"
_OPENMATH_DISPLAY = "OpenMathInstruct-2"
_TULU3_DISPLAY = "Tulu-3 SFT mixture"
_BENGALI_DISPLAY = "Aya-Bengali"

# σ is a proxy everywhere: the only measured AdamW multiseed σ (OLMo-opc, ~0.0007
# at r64) is at the 2k horizon, not the 9k leaderboard horizon. 0.0017 mirrors the
# value the notebooks have always used so Δ/σ reporting is unchanged by this refactor.
_SIGMA = 0.0017

WORKLOADS: list[Workload] = [
    # ── OLMo-2-1B ────────────────────────────────────────────────────────────
    Workload("allenai/OLMo-2-0425-1B", "opc", 64, "OLMo-2-1B", _OPC_DISPLAY, 9000, _SIGMA, True),
    Workload("allenai/OLMo-2-0425-1B", "opc", 256, "OLMo-2-1B", _OPC_DISPLAY, 9000, _SIGMA, True),
    Workload("allenai/OLMo-2-0425-1B", "openmath", 64, "OLMo-2-1B", _OPENMATH_DISPLAY, 9000, _SIGMA, True),
    Workload("allenai/OLMo-2-0425-1B", "openmath", 256, "OLMo-2-1B", _OPENMATH_DISPLAY, 9000, _SIGMA, True),
    Workload("allenai/OLMo-2-0425-1B", "tulu3", 64, "OLMo-2-1B", _TULU3_DISPLAY, 9000, _SIGMA, True),
    Workload("allenai/OLMo-2-0425-1B", "tulu3", 256, "OLMo-2-1B", _TULU3_DISPLAY, 9000, _SIGMA, True),
    # ── Llama-3.2-1B ─────────────────────────────────────────────────────────
    Workload("meta-llama/Llama-3.2-1B", "opc", 64, "Llama-3.2-1B", _OPC_DISPLAY, 9000, _SIGMA, True),
    Workload("meta-llama/Llama-3.2-1B", "opc", 256, "Llama-3.2-1B", _OPC_DISPLAY, 9000, _SIGMA, True),
    Workload("meta-llama/Llama-3.2-1B", "openmath", 64, "Llama-3.2-1B", _OPENMATH_DISPLAY, 9000, _SIGMA, True),
    Workload("meta-llama/Llama-3.2-1B", "openmath", 256, "Llama-3.2-1B", _OPENMATH_DISPLAY, 9000, _SIGMA, True),
    # ── Qwen2.5-1.5B ─────────────────────────────────────────────────────────
    Workload("Qwen/Qwen2.5-1.5B", "opc", 256, "Qwen2.5-1.5B", _OPC_DISPLAY, 9000, _SIGMA, True),
    Workload("Qwen/Qwen2.5-1.5B", "bengali", 256, "Qwen2.5-1.5B", _BENGALI_DISPLAY, 9000, _SIGMA, True),
    # ── Meta-Llama-3-8B (scale-up; model_display gives label "Meta/..." so it
    #    does not collide with Llama-3.2-1B's "Llama/..." cross-setting key) ────
    Workload("meta-llama/Meta-Llama-3-8B", "opc", 256, "Meta-Llama-3-8B", _OPC_DISPLAY, 9000, _SIGMA, True),
]


def iter_workloads() -> list[Workload]:
    return list(WORKLOADS)


def find_workload(model_name: str, dataset: str, rank: int) -> Workload:
    """Look up a declared cell. Raises KeyError on miss.

    `model_name` may be the cfg model id (e.g. 'allenai/OLMo-2-0425-1B') or the
    display name (e.g. 'OLMo-2-1B').
    """
    for wl in WORKLOADS:
        if (model_name in (wl.model_name, wl.model_display)
                and wl.dataset == dataset and wl.rank == rank):
            return wl
    raise KeyError(f"no workload for (model={model_name!r}, dataset={dataset!r}, rank={rank})")


def workload_runs(wl: Workload, *, logs_root: str | None = None) -> list[tuple[dict, list]]:
    """Discover a cell's runs: predicate load → dataset filter → deny-pattern.

    Does NOT dedup — callers dedup by canonical (label, lr) via
    `labeled_completed_runs` / `compare_variants_figure`. `logs_root` defaults to
    the repo's logs/ dir (resolved from this file), so callers in any working
    directory — including notebooks in subdirs — get the right data.
    """
    from lora_playground.loader import load_runs  # lazy: avoid import cycle
    runs = load_runs(
        where={
            "model_name": wl.model_name,
            "lora_r": wl.rank,
            "max_steps": lambda s: isinstance(s, int) and s >= wl.min_completed_steps,
        },
        logs_root=logs_root or DEFAULT_LOGS_ROOT,
        warn_cross_commit=False,
        quiet=True,
    )
    return [(c, h) for c, h in runs
            if resolve_dataset(c) == wl.dataset and not _denied(c.get("log_group"))]
