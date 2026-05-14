"""Execution-scope provenance: closure walk, content hash, env capture.

Phase 4 of the data-loading robustness redesign (see
~/.claude/plans/the-data-loading-in-wobbly-pelican.md). The whole-tree
`git_diff_sha` recorded by Phase 1 was too coarse: edits to notebooks /
loader / docs / params all flagged dirty, hiding legitimate sweeps. This
module computes a content hash scoped to the AST-determined execution
closure starting from `train_lora.py`, so only edits to code that
actually runs flag the run.

Key properties:
- AST-determined (catches function-body lazy imports because `ast.walk`
  traverses the full tree). NOT augmented by `sys.modules` — to avoid
  conservative inclusion of unrelated modules.
- Source bytes come from a snapshot taken at process start (in
  `train_lora.py`), NOT from disk at hash time. Closes the
  mid-edit-during-import race.
- Content-based hash: `sha256(sorted (path, sha256(content)))`.
  Base-independent; survives rebases; identical content always hashes
  identically.
- External package versions captured separately (audit only), never in
  `source_tree_sha`.

The anti-decay test in `tests/test_execution_scope.py` enforces the
no-dynamic-project-imports property that makes AST-only safe.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterable


# ─── project root + snapshot bootstrap fallback ───────────────────────────────

# Same exclusion list as `train_lora.py`'s bootstrap. Single source of truth
# (train_lora.py imports this constant via execution_scope after its own
# stdlib-only first-pass bootstrap finishes). Keep these in sync if you edit.
SNAPSHOT_EXCLUDED_DIRS = frozenset({
    "tests", "scripts", "notebooks", "docs",
    "logs", "slurm_pending", "slurm_submitted",
    "slurm_cancel_pending", "slurm_cancel_submitted",
    "disbatch_logs", "slurm_logs", "backfill_records",
    "data", "params", "paper",
    "__pycache__", ".git",
})


def project_root() -> Path:
    """Project repo root, derived from this module's location."""
    return Path(__file__).resolve().parent.parent


def get_or_bootstrap_snapshot() -> tuple[dict[str, bytes], dict[str, str]]:
    """Return (snapshot, snapshot_sha) from `train_lora.py`'s bootstrap if it
    ran (i.e. process started via `python train_lora.py`); otherwise read
    fresh from disk and warn.

    Race-safe path: train_lora.py is the entry point → its module-level
    code ran before anything else → SOURCE_SNAPSHOT is populated from
    bytes-on-disk before any lora_playground import could trigger a cache.

    Fallback path: process started some other way (e.g. `python -m
    lora_playground.train`, pytest, jupyter). The fallback reads from disk
    NOW, which means edits between Python startup and this call go
    unnoticed. Mostly a concern for interactive dev sessions, not for
    sbatch sweeps where code is stable for the run's duration.
    """
    main_mod = sys.modules.get("__main__")
    snap = getattr(main_mod, "SOURCE_SNAPSHOT", None) if main_mod else None
    snap_sha = getattr(main_mod, "SOURCE_SNAPSHOT_SHA", None) if main_mod else None
    if snap and snap_sha:
        return snap, snap_sha
    warnings.warn(
        "execution_scope: no pre-imported snapshot found "
        "(entry was not train_lora.py?). Reading source bytes from disk "
        "now — not race-safe against mid-import edits.",
        stacklevel=2,
    )
    root = project_root()
    snap = {}
    snap_sha = {}
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in SNAPSHOT_EXCLUDED_DIRS for part in rel.parts):
            continue
        try:
            content = p.read_bytes()
        except OSError:
            continue
        key = str(rel)
        snap[key] = content
        snap_sha[key] = hashlib.sha256(content).hexdigest()
    return snap, snap_sha


# ─── closure walk ─────────────────────────────────────────────────────────────


def compute_closure(
    entry_path: Path,
    project_root: Path,
    snapshot: dict[str, bytes] | None = None,
) -> set[str]:
    """AST-walk from `entry_path`; follow project-relative imports to fixed
    point. Returns the set of project-internal relative paths in the closure.

    Parameters
    ----------
    entry_path : absolute path of the entry script (e.g. train_lora.py).
        Must be under `project_root`.
    project_root : project repo root. Only files under this root are
        considered project-internal.
    snapshot : optional {relative_path_str: bytes} dict. When provided,
        source bytes come from here (race-safe). When None, falls back to
        reading from disk (NOT race-safe; use only in tests / CI tools).

    Behavior notes:
      - `ast.walk` traverses the FULL tree, including function-body imports.
        So `from ._batched_polar import f` inside an optim.py function is
        caught.
      - Cycles handled via the `visited` set.
      - SyntaxError in any file → warn + skip (don't crash provenance).
      - Absolute imports of `lora_playground.X` ARE followed (not just `.X`).
      - `from . import X` resolved by enumerating `node.names` aliases as
        submodules.
    """
    if snapshot is None:
        snapshot = {}

    visited: set[str] = set()
    # Queue entries: (rel_path, current_package_dotted_or_None)
    entry_rel = str(entry_path.resolve().relative_to(project_root.resolve()))
    queue: list[tuple[str, str | None]] = [
        (entry_rel, _package_for_path(entry_rel))
    ]

    while queue:
        rel_path, current_package = queue.pop()
        if rel_path in visited:
            continue
        visited.add(rel_path)
        # Whenever Python loads a module inside a package, it ALSO loads
        # every parent package's __init__.py. So `from pkg.sub.X import Y`
        # loads pkg/__init__.py AND pkg/sub/__init__.py before pkg/sub/X.py.
        # Reflect this in the closure by enqueuing each ancestor __init__.py.
        if "/" in rel_path:
            parts = rel_path.split("/")
            for i in range(1, len(parts)):
                ancestor_init = "/".join(parts[:i]) + "/__init__.py"
                if ancestor_init in visited:
                    continue
                if ancestor_init in snapshot or (project_root / ancestor_init).exists():
                    queue.append((ancestor_init, _package_for_path(ancestor_init)))

        source_bytes = _get_source(rel_path, project_root, snapshot)
        if source_bytes is None:
            continue

        try:
            tree = ast.parse(source_bytes, filename=rel_path)
        except SyntaxError as e:
            warnings.warn(
                f"execution_scope: failed to parse {rel_path}: {e}; "
                f"skipping (file excluded from closure). Hash will be "
                f"computed without it; downstream auto-resolve may fail."
            )
            continue

        for node in ast.walk(tree):
            new_modules: list[list[str]] = []
            if isinstance(node, ast.ImportFrom):
                module_parts = _resolve_relative_import(
                    node.module, node.level, current_package,
                )
                if module_parts is None:
                    continue
                # The from-target itself.
                new_modules.append(module_parts)
                # Each alias might be a submodule (e.g. `from . import x`).
                for alias in node.names:
                    new_modules.append(module_parts + [alias.name])
            elif isinstance(node, ast.Import):
                # `import lora_playground.X.Y`
                for alias in node.names:
                    new_modules.append(alias.name.split("."))

            for parts in new_modules:
                for path in _module_to_paths(parts, project_root, snapshot):
                    if path not in visited:
                        queue.append((path, _package_for_path(path)))

    return visited


def _get_source(
    rel_path: str, project_root: Path, snapshot: dict[str, bytes],
) -> bytes | None:
    if rel_path in snapshot:
        return snapshot[rel_path]
    full = project_root / rel_path
    if not full.exists():
        return None
    return full.read_bytes()


def _resolve_relative_import(
    module: str | None, level: int, current_package: str | None,
) -> list[str] | None:
    """Resolve an ImportFrom node to a dotted module path as a list of parts.

    level=0 → absolute import. `module` is the full dotted name.
    level≥1 → relative. e.g. `from .X import Y` is level=1, `from ..X` is
    level=2. Without `current_package`, we can't resolve.
    """
    if level == 0:
        if module is None:
            return None
        return module.split(".")
    if current_package is None:
        return None
    parts = current_package.split(".")
    # `from .` is level=1 (current package); `from ..` is level=2 (parent).
    cut = level - 1
    base = parts[: len(parts) - cut] if cut > 0 else parts
    if module:
        return base + module.split(".")
    return base


def _module_to_paths(
    parts: list[str], project_root: Path, snapshot: dict[str, bytes],
) -> list[str]:
    """Given module parts like ["lora_playground", "optim"], return matching
    relative paths in the snapshot or on disk.

    A module can resolve to either `<parts>.py` (single-file module) or
    `<parts>/__init__.py` (package). We don't follow .pyi stubs.
    """
    candidate_module = "/".join(parts) + ".py"
    candidate_package = "/".join(parts) + "/__init__.py"
    out: list[str] = []
    for cand in (candidate_module, candidate_package):
        if cand in snapshot:
            out.append(cand)
        elif (project_root / cand).exists():
            out.append(cand)
    return out


def _package_for_path(rel_path: str) -> str | None:
    """train_lora.py → None (top-level script);
    lora_playground/optim.py → "lora_playground";
    lora_playground/sub/X.py → "lora_playground.sub";
    lora_playground/__init__.py → "lora_playground".
    """
    p = rel_path
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
        return p.replace("/", ".") if p else None
    if "/" not in p:
        return None
    return p.rsplit("/", 1)[0].replace("/", ".")


# ─── content hash ─────────────────────────────────────────────────────────────


def source_tree_sha_at_commit(
    commit: str, paths: Iterable[str], project_root: Path,
) -> str | None:
    """Compute the same `source_tree_sha` but with content read from
    `commit`'s git tree rather than the snapshot. Uses `git cat-file -p
    <commit>:<path>` for raw blob bytes — NOT `git show` (which can apply
    smudge/textconv filters that would diverge from snapshot bytes on
    some git configurations).

    Returns None if any path is missing from the commit's tree, or if git
    is unavailable. Callers treat None as "cannot compare; assume mismatch."
    """
    sha_map: dict[str, str] = {}
    paths_list = list(paths)
    for path in paths_list:
        try:
            result = subprocess.run(
                ["git", "cat-file", "-p", f"{commit}:{path}"],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5.0,
            )
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            return None
        sha_map[path] = hashlib.sha256(result.stdout).hexdigest()
    return source_tree_sha(paths_list, sha_map)


def source_tree_sha(
    closure: Iterable[str], snapshot_sha: dict[str, str],
) -> str:
    """sha256 over sorted (path, content_sha) pairs from the closure.

    Base-independent — depends only on which files and what their content
    is, not on any commit. Two runs with identical execution-relevant code
    state produce identical hashes regardless of `git_commit`.

    Paths NOT in `snapshot_sha` are silently omitted; the caller is
    expected to ensure closure ⊆ snapshot keys (the bootstrap snapshot
    covers all of lora_playground/ minus the exclusion list).
    """
    entries = sorted(
        (p, snapshot_sha[p]) for p in closure if p in snapshot_sha
    )
    payload = json.dumps(entries, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─── auto-resolution (loader-side) ────────────────────────────────────────────

# Cache: (base_commit, tuple(sorted(paths)), target_sha) -> resolved_commit | None.
# Module-level so repeated load_runs() calls in a notebook share results.
_AUTO_RESOLVE_CACHE: dict[tuple[str, tuple, str], str | None] = {}


def auto_resolve_by_content(
    base_commit: str,
    paths: list[str],
    target_source_sha: str,
    project_root: Path,
    max_commits: int = 100,
    max_wall_seconds: float = 5.0,
) -> str | None:
    """Walk descendants of `base_commit` looking for a commit `C` whose
    `source_tree_sha_at_commit(C, paths)` equals `target_source_sha`.

    Used by the loader to resolve dirty runs to a real commit that
    represents the same execution-relevant code state. If found, the run
    is treated as at that commit for invariant evaluation — no manual
    attestation needed.

    Returns the matching commit hash, or None if no match within bounds.

    Bounds:
      - At most `max_commits` candidates inspected (newest first by
        `git rev-list --reverse`).
      - At most `max_wall_seconds` of wall time before giving up.
    """
    import time
    paths_key = tuple(sorted(paths))
    cache_key = (base_commit, paths_key, target_source_sha)
    if cache_key in _AUTO_RESOLVE_CACHE:
        return _AUTO_RESOLVE_CACHE[cache_key]

    deadline = time.time() + max_wall_seconds
    try:
        result = subprocess.run(
            ["git", "rev-list", "--reverse", f"{base_commit}..HEAD",
             "-n", str(max_commits)],
            cwd=str(project_root),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True, timeout=3.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        _AUTO_RESOLVE_CACHE[cache_key] = None
        return None

    candidates = [c for c in result.stdout.decode().strip().split("\n") if c]
    resolved: str | None = None
    for c in candidates:
        if time.time() > deadline:
            break
        sha_at_c = source_tree_sha_at_commit(c, list(paths), project_root)
        if sha_at_c == target_source_sha:
            resolved = c
            break
    _AUTO_RESOLVE_CACHE[cache_key] = resolved
    return resolved


# ─── external env capture ─────────────────────────────────────────────────────


_AUDIT_PACKAGES: tuple[str, ...] = (
    "torch", "transformers", "peft", "datasets", "tokenizers",
    "accelerate", "numpy", "safetensors", "triton", "flash_attn",
)


def capture_env() -> tuple[dict[str, str], str]:
    """Snapshot Python + key package versions for audit. Returns
    ({pkg: version, ...}, env_sha).

    Excluded from `source_tree_sha` by design: a `pip install` should not
    invalidate prior runs' auto-resolution. Loader may surface env mismatch
    as a warning but never as an exclusion.
    """
    env: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in _AUDIT_PACKAGES:
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = "<missing>"
        except Exception:
            env[pkg] = "<error>"
    try:
        import torch  # type: ignore
        env["torch_cuda"] = getattr(torch.version, "cuda", None) or "<cpu>"
        try:
            env["torch_cudnn"] = str(torch.backends.cudnn.version() or "<missing>")
        except Exception:
            env["torch_cudnn"] = "<error>"
        try:
            if torch.cuda.is_available():
                nccl = torch.cuda.nccl.version()
                env["torch_nccl"] = ".".join(map(str, nccl)) if nccl else "<missing>"
            else:
                env["torch_nccl"] = "<cpu>"
        except Exception:
            env["torch_nccl"] = "<error>"
    except Exception:
        env.setdefault("torch_cuda", "<unknown>")
        env.setdefault("torch_cudnn", "<unknown>")
        env.setdefault("torch_nccl", "<unknown>")
    payload = json.dumps(sorted(env.items()), separators=(",", ":"))
    return env, hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─── public assembly helper ───────────────────────────────────────────────────


def compute_execution_provenance(
    entry_path: Path,
    project_root: Path,
    snapshot: dict[str, bytes],
    snapshot_sha: dict[str, str],
    git_commit: str | None = None,
) -> dict:
    """Top-level helper called by train.py at cfg-emit time. Returns the
    dict of new cfg-event fields:

      execution_source_sha       : str
      execution_source_paths     : sorted list[str]
      execution_source_dirty     : bool   (true iff snapshot hash differs
                                          from the same hash recomputed at
                                          `git_commit`'s tree; None if
                                          `git_commit` is None or unreachable)
      execution_env              : dict[str, str]
      execution_env_sha          : str
    """
    closure = compute_closure(entry_path, project_root, snapshot)
    # Guard: any closure path missing from the snapshot is a serious bug.
    missing = sorted(p for p in closure if p not in snapshot_sha)
    if missing:
        raise RuntimeError(
            f"execution_scope: {len(missing)} path(s) in execution closure "
            f"not found in source snapshot: {missing}. Check the snapshot "
            f"exclusion list (SNAPSHOT_EXCLUDED_DIRS) — a path imported by "
            f"training code lives in an excluded directory or under a typo."
        )
    paths_sorted = sorted(closure)
    src_sha = source_tree_sha(paths_sorted, snapshot_sha)

    # Compute dirty by comparing snapshot's hash to the same closure rehashed
    # at git_commit's tree state. None if git_commit unavailable or any path
    # is missing in that commit (e.g. an untracked-but-imported file).
    if git_commit:
        commit_sha = source_tree_sha_at_commit(
            git_commit, paths_sorted, project_root,
        )
        if commit_sha is None:
            # At least one closure file isn't in the commit tree (untracked
            # but imported). The run IS using uncommitted source.
            dirty = True
        else:
            dirty = (commit_sha != src_sha)
    else:
        dirty = None  # Can't compare without a commit.

    env, env_sha = capture_env()
    return {
        "execution_source_sha": src_sha,
        "execution_source_paths": paths_sorted,
        "execution_source_dirty": dirty,
        "execution_env": env,
        "execution_env_sha": env_sha,
    }
