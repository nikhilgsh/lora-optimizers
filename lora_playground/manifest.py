"""Optional sweep-manifest annotations and audit helpers.

Every sweep submitted via ``slurm_scripts/submit.sh`` writes a ``meta.json``
into ``logs/<group>/run_info/``. Physical populated log groups remain
discoverable when this annotation is missing or malformed; ``strict=True`` is
an explicit audit mode, not an admission policy.

Schema (written by submit.sh):

    {
      "group":         "adamuon_polar_product_2k",
      "submitted_at":  "2026-04-30T14:23:01-04:00",
      "slurm_job_id":  "6314009",
      "n_gpus":        6,
      "params_file":   "adamuon_polar_product_2k.json",
      "sweep_script":  "scripts/sweep/sweep_2k_r_diag.sh",
      "sbatch_script": "slurm_scripts/sbatch.sh",
      "git_commit":    "<sha>",
      "git_dirty":     false,
      "scope":         ["ext_compare", "polar_family"],
      "purpose":       "E2: AdaMuon-faithful + polar-product geometry",
      "data_pipeline_version": "packed_v1"
    }

The `data_pipeline_version` field marks which data path produced the runs:

  - "unpacked_v0": legacy DataCollatorForLanguageModeling, dynamic shapes,
                   no prompt-mask, full-text loss. All runs prior to
                   2026-05-08 are this version (backfilled automatically
                   by `scripts/data/backfill_pipeline_version.py`).
  - "packed_v1":   sequence-packed train side, pad-to-max eval side,
                   prompt-masked loss. New default from 2026-05-08.

Numbers from different versions are NOT directly comparable: prompt-mask
changes the loss objective; packing changes per-step token density. Use
`load_runs(where={"data_pipeline_version": "packed_v1"})` to filter.

Loading: ``lora_playground.loader.load_runs(where=...)`` — predicate-based,
no scope strings. To remove an old sweep from analysis, delete its log dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .run_records import RUNTIME_FIELDS


# Fields allowed to vary within a single analysis "series" (i.e. across
# seeds / lr-grid points / horizon extensions of the same algorithm at the
# same model config). Two runs whose cfgs disagree only on fields in this
# set are seeds of the same series and may be averaged. Two runs that
# disagree on ANY field outside this set are distinct series and MUST
# resolve to distinct display labels — `assert_label_discriminates` in
# `plot_utils` enforces that contract at every plotting entry point.
#
# Adding a new flag here means "this is a per-series axis, not a
# series-defining hyperparameter." Default for any new train.py CLI flag
# is series-defining — leave it OUT of this set unless you really mean
# "varying this within a series is fine, do not split."
SERIES_AXIS_FIELDS: frozenset[str] = frozenset({
    # Per-series axes — runs differing only on these are averaged together
    # as members of one series (seeds, lr-sweep points, horizon extensions).
    "seed", "lr", "lora_r", "lora_alpha",
    "max_steps", "eval_every",
    # CLI override flags whose canonical post-resolution value is
    # promoted by `_enrich_cfg` to a top-level scalar (e.g.
    # `effective_picard_iters`). The raw override is then redundant.
    # Series identity uses the effective value; the raw override is not
    # series-defining.
    "picard_iters_override",
}) | RUNTIME_FIELDS

# NOTE on what is intentionally NOT in SERIES_AXIS_FIELDS:
#   - `lora_plus_multiplier`: real algorithm parameter (LoRA+ B-multiplier);
#     different m = different setup, must split series.
#   - `precond_delta`, `precond_delta_relative`: damping scheme; different
#     ε_rel runs are distinct variants and must split series.
#   - `anderson_m`, `picard_alpha`: Picard inner-solver knobs — different
#     values change the algorithm.
#   - `polar_method`, `polar_sigma_power`, `polar_norm_dir`: polar/operator
#     choices — different values change the algorithm.


class UntaggedSweepError(RuntimeError):
    """An explicit manifest audit found a missing/malformed annotation.

    Fix:
      1. New sweep that bypassed submit.sh — re-run via
         ``slurm_scripts/submit.sh`` with ``SWEEP_SCOPE`` set, or hand-
         write ``logs/<group>/run_info/meta.json`` with the schema in
         ``manifest.py``.
      2. Old log dir restored without metadata — re-derive scope and
         drop a meta.json next to the run logs.
    Ordinary discovery uses ``load_manifests(strict=False)`` and keeps every
    populated group; strict mode exists for CI/audit callers only.
    """


# The manifest field set, in the module that READS manifests. It was written out
# twice -- once in `slurm_scripts/submit.sh`'s inline python and once in
# `train.py`'s stub writer -- with identical keys, so this module owned the
# schema for reading and neither writer for writing. A field added to one writer
# and not the other produces manifests that differ by provenance, which is
# exactly the drift `load_manifests` cannot see.
MANIFEST_FIELDS: tuple[str, ...] = (
    "group", "submitted_at", "slurm_job_id", "n_gpus", "params_file",
    "sweep_script", "sbatch_script", "git_commit", "git_dirty", "scope",
    "purpose", "data_pipeline_version",
)


def build_manifest(**fields) -> dict:
    """A manifest dict with every `MANIFEST_FIELDS` key present.

    Unsupplied fields land as None, so a writer that does not know a value
    records it as unknown rather than omitting the key -- `load_manifests` and
    `live_manifests_newest_first` read `submitted_at` and `scope` off every
    manifest, and an absent key and a null one are not the same thing to them.
    Extra keys (e.g. `_stub`) pass through: writers may annotate.

    Raises on a field name that is not in the schema, because a typo'd key is
    invisible -- it writes fine and reads as a missing field forever.
    """
    unknown = sorted(set(fields) - set(MANIFEST_FIELDS) - {"_stub"})
    if unknown:
        raise ValueError(
            f"unknown manifest field(s) {unknown}; known fields are "
            f"{list(MANIFEST_FIELDS)}. Add to MANIFEST_FIELDS if genuinely new.")
    out = {k: fields.get(k) for k in MANIFEST_FIELDS}
    if "_stub" in fields:
        out["_stub"] = fields["_stub"]
    return out


def write_manifest_atomic(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> Path:
    """Atomically replace one ``meta.json`` with a complete JSON object.

    The payload is serialized before the destination is touched, then written
    and fsynced in the destination directory before ``os.replace``.  A failed
    serialization or write therefore leaves any prior manifest intact.  This
    is the writer entry point submission code can adopt without duplicating an
    inline schema or exposing readers to a half-written JSON file.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    payload = json.dumps(dict(manifest), sort_keys=True, indent=2) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            mode = destination.stat().st_mode & 0o777
        except OSError:
            mode = 0o644
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    # A same-process reader should not need to wait for a signature comparison
    # to notice the write (particularly on coarse-mtime filesystems).
    _LOAD_MANIFESTS_CACHE.clear()
    return destination


def write_submission_manifest(
    path: str | os.PathLike[str],
    *,
    group: str,
    submitted_at: str,
    slurm_job_id: str,
    n_gpus: int,
    params_file: str,
    sweep_script: str,
    sbatch_script: str,
    git_commit: str,
    git_dirty: bool,
    scope: str | list[str] | tuple[str, ...],
    purpose: str,
    data_pipeline_version: str,
) -> Path:
    """Build and atomically write the canonical submission annotation."""
    if isinstance(scope, str):
        scope = [part.strip() for part in scope.split(",") if part.strip()]
    manifest = build_manifest(
        group=group,
        submitted_at=submitted_at,
        slurm_job_id=slurm_job_id,
        n_gpus=int(n_gpus),
        params_file=params_file,
        sweep_script=sweep_script,
        sbatch_script=sbatch_script,
        git_commit=git_commit,
        git_dirty=bool(git_dirty),
        scope=list(scope),
        purpose=purpose,
        data_pipeline_version=data_pipeline_version,
    )
    return write_manifest_atomic(path, manifest)


_LOAD_MANIFESTS_CACHE: dict[tuple[str, bool], tuple[tuple, list[dict]]] = {}
_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")


def _has_scope_tag(scope) -> bool:
    """Whether ``scope`` contains at least one non-blank tag.

    ``submit.sh`` writes a list, while a few historical hand-written
    manifests contain one string.  Accept both representations, but do not
    let whitespace-only values satisfy the manifest contract.
    """
    if isinstance(scope, str):
        return bool(scope.strip())
    if isinstance(scope, (list, tuple, set)):
        return any(isinstance(tag, str) and bool(tag.strip()) for tag in scope)
    return False


def _groups_with_run_info(logs_root: str) -> list[str]:
    """Sorted group names under ``logs_root`` that have a ``run_info/`` dir.

    Replaces ``root.glob("*/run_info")``: one ``scandir`` of ``logs/`` plus one
    ``stat`` per child, and it never follows into the group directories.
    """
    try:
        children = sorted(e.name for e in os.scandir(logs_root) if e.is_dir())
    except OSError:
        return []
    return [g for g in children
            if os.path.isdir(os.path.join(logs_root, g, "run_info"))]


def _has_runs(group: str, logs_root: str) -> bool:
    """Whether the physical group contains a populated task log.

    Kept local so manifest discovery does not import plotting.loading merely to
    answer a filesystem question.
    """
    log_dir = os.path.join(logs_root, group, "run_info", "logs")
    try:
        entries = os.scandir(log_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False
    with entries:
        for entry in entries:
            if not _TASK_FILE_RE.match(entry.name):
                continue
            try:
                if entry.is_file() and entry.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _parallel_map(fn, items: list[str]):
    """Small metadata-only parallel map with a serial failure fallback."""
    if len(items) < 2:
        return [fn(item) for item in items]
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(items)),
                                thread_name_prefix="manifest-stat") as pool:
            return list(pool.map(fn, items))
    except RuntimeError:
        return [fn(item) for item in items]


def _meta_stats(logs_root: str, groups: list[str]
                ) -> list[tuple[str, int, int]]:
    """``(group, meta.json mtime_ns, size)`` per group, missing file → (0, 0)."""
    def one(group: str) -> tuple[str, int, int]:
        try:
            st = os.stat(os.path.join(logs_root, group, "run_info", "meta.json"))
        except OSError:
            return (group, 0, 0)
        return (group, st.st_mtime_ns, st.st_size)

    return _parallel_map(one, groups)


def load_manifests(logs_root: str = "../logs", strict: bool = False) -> list[dict]:
    """Return one annotation record per populated physical log group.

    Groups without run output are skipped. Missing, corrupt, non-object, or
    empty-scope manifests are represented by flagged stub dictionaries and are
    never silently omitted.

    With explicit ``strict=True``, raises ``UntaggedSweepError`` if any
    populated log dir lacks a manifest, has corrupt JSON, or empty scope.
    The ordinary ``strict=False`` path returns those groups with stub manifests
    flagged (`_untagged`, `_corrupt`, `_empty_scope`) so the caller can
    decide.

    Cached by signature ``(logs_root_mtime, sorted (group, meta_mtime,
    meta_size))`` so analysis loops that call this once per ``load_runs``
    don't re-walk 178+ directories. New groups bump logs_root mtime;
    edited meta.json bumps its own mtime. Groups skipped as empty are the
    exception: their log directories are rechecked on a cache hit so the first
    arriving run becomes visible immediately.
    """
    root = Path(logs_root)
    if not root.exists():
        return []
    try:
        root_mtime = root.stat().st_mtime_ns
        groups = _groups_with_run_info(logs_root)
        # One `stat` per meta.json instead of the four the exists()+stat()+
        # exists()+stat() form cost, and issued concurrently: this runs on
        # every `load_runs` call (it is what decides the cache hit) over
        # 500+ groups on a shared parallel filesystem, where the syscall is
        # latency-bound and the GIL is released for its duration.
        meta_sig = tuple(_meta_stats(logs_root, groups))
        sig = (root_mtime, meta_sig)
        cache_key = (str(root.resolve()), strict)
        cached = _LOAD_MANIFESTS_CACHE.get(cache_key)
        if cached is not None and cached[0] == sig:
            # The signature above intentionally avoids walking every logs/
            # directory.  Consequently, the first log file added under an
            # already-existing run_info/ directory changes neither component
            # of ``sig``.  Recheck only groups that the cached result skipped
            # as empty; populated groups remain on the metadata-only fast
            # path.  A new top-level group still invalidates via root_mtime.
            populated = {m["group"] for m in cached[1]}
            previously_empty = [g for g in groups if g not in populated]
            if not any(_has_runs(g, logs_root) for g in previously_empty):
                return [dict(m) for m in cached[1]]
    except OSError:
        sig = None
        cache_key = None
        groups = _groups_with_run_info(logs_root)

    manifests: list[dict] = []
    bad: list[tuple[str, str]] = []  # (group, reason)
    for group in groups:
        run_info = root / group / "run_info"
        if not _has_runs(group, logs_root):
            continue
        meta_path = run_info / "meta.json"
        try:
            raw = meta_path.read_text()
        except OSError:
            raw = None
        if raw is not None:
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                m = {"group": group, "scope": [], "purpose": "", "_corrupt": True}
                bad.append((group, "corrupt meta.json"))
            else:
                if not isinstance(m, dict):
                    m = {
                        "group": group, "scope": [], "purpose": "",
                        "_corrupt": True, "_malformed_type": True,
                    }
                    bad.append((group, "meta.json is not an object"))
        else:
            m = {"group": group, "scope": [], "purpose": "", "_untagged": True}
            bad.append((group, "no meta.json"))
        recorded_group = m.get("group")
        if recorded_group not in (None, group):
            m["_manifest_group"] = recorded_group
            m["_group_mismatch"] = True
            bad.append((group, f"manifest group is {recorded_group!r}"))
        # Physical identity is authoritative; manifest content is annotation.
        m["group"] = group
        if (not _has_scope_tag(m.get("scope"))
                and not m.get("_untagged") and not m.get("_corrupt")):
            m["_empty_scope"] = True
            bad.append((group, "empty scope"))
        manifests.append(m)
    if strict and bad:
        details = "\n".join(f"  {g}: {why}" for g, why in bad)
        raise UntaggedSweepError(
            f"{len(bad)} log group(s) without valid manifest:\n{details}\n"
            f"See lora_playground/manifest.py for fix instructions."
        )
    if cache_key is not None and sig is not None:
        _LOAD_MANIFESTS_CACHE[cache_key] = (sig, manifests)
    return [dict(m) for m in manifests]


def warn_untagged(manifests: list[dict]) -> list[str]:
    """Return group names that lack a manifest or have an empty scope.

    Print is left to the caller — notebooks typically print once near the
    top of a section so the warning is visible without polluting every cell.
    """
    bad = []
    for m in manifests:
        if (m.get("_corrupt") or m.get("_untagged")
                or not _has_scope_tag(m.get("scope"))):
            bad.append(m["group"])
    return bad


def _submitted_at_sort_key(manifest: Mapping[str, Any]) -> tuple[int, float]:
    """Chronological key with missing/malformed timestamps ordered oldest."""
    raw = manifest.get("submitted_at")
    if not isinstance(raw, str) or not raw.strip():
        return (0, 0.0)
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return (1, value.timestamp())
    except (ValueError, OverflowError, OSError):
        return (0, 0.0)


def live_manifests_newest_first(manifests: list[dict]) -> list[dict]:
    """Return every populated-group annotation, newest known time first.

    The historical name is retained for API compatibility. Missing, corrupt,
    and empty-scope annotations remain in the result: manifest health is audit
    metadata, not admission. Nullable or malformed ``submitted_at`` values sort
    after valid timestamps without raising; stable input order breaks ties.
    """
    return sorted(manifests, key=_submitted_at_sort_key, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-submission")
    write.add_argument("--path", required=True)
    write.add_argument("--group", required=True)
    write.add_argument("--submitted-at", required=True)
    write.add_argument("--slurm-job-id", required=True)
    write.add_argument("--n-gpus", required=True, type=int)
    write.add_argument("--params-file", required=True)
    write.add_argument("--sweep-script", required=True)
    write.add_argument("--sbatch-script", required=True)
    write.add_argument("--git-commit", required=True)
    write.add_argument("--git-dirty", action="store_true")
    write.add_argument("--scope", default="")
    write.add_argument("--purpose", default="")
    write.add_argument("--data-pipeline-version", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-submission":
        path = write_submission_manifest(
            args.path,
            group=args.group,
            submitted_at=args.submitted_at,
            slurm_job_id=args.slurm_job_id,
            n_gpus=args.n_gpus,
            params_file=args.params_file,
            sweep_script=args.sweep_script,
            sbatch_script=args.sbatch_script,
            git_commit=args.git_commit,
            git_dirty=args.git_dirty,
            scope=args.scope,
            purpose=args.purpose,
            data_pipeline_version=args.data_pipeline_version,
        )
        print(f"Wrote manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
