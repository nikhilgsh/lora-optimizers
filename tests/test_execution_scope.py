"""Tests for execution-scope provenance (Phase 4).

Four kinds of test:
  1. Closure walk handles function-body lazy imports (synthetic fixture).
  2. Real-codebase closure has the expected shape (training-side only).
  3. Anti-decay: codebase contains no dynamic project imports.
  4. lora_playground/__init__.py has zero side effects (bootstrap-order).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.execution_scope import (  # noqa: E402
    capture_env,
    compute_closure,
    compute_execution_provenance,
    project_root,
    source_tree_sha,
)


# ─── 1. Synthetic fixture: function-body lazy import ──────────────────────────


def _make_fixture_package(tmp_path: Path) -> Path:
    """Build a tiny project structure mirroring lora_playground/ shape:
      tmp_path/
        entry.py                       (analogue of train_lora.py)
        fixture_pkg/
          __init__.py                  (empty)
          main.py                      (analogue of train.py; imports helpers)
          helpers.py                   (top-level import + function-body import)
          lazy_helper.py               (only referenced inside helpers.foo())
          unused.py                    (NOT imported by anything in the closure)
    """
    pkg = tmp_path / "fixture_pkg"
    pkg.mkdir()
    (tmp_path / "entry.py").write_text(textwrap.dedent("""
        from fixture_pkg.main import main
        if __name__ == "__main__":
            main()
    """).strip() + "\n")
    (pkg / "__init__.py").write_text('"""Fixture package."""\n')
    (pkg / "main.py").write_text(textwrap.dedent("""
        from .helpers import top_level_helper
        def main():
            return top_level_helper()
    """).strip() + "\n")
    (pkg / "helpers.py").write_text(textwrap.dedent("""
        def top_level_helper():
            # The lazy import is INSIDE a function body. ast.walk must
            # traverse it (this is the property the test pins).
            from .lazy_helper import lazy_fn
            return lazy_fn()
    """).strip() + "\n")
    (pkg / "lazy_helper.py").write_text(textwrap.dedent("""
        def lazy_fn():
            return 42
    """).strip() + "\n")
    (pkg / "unused.py").write_text("# Not imported anywhere.\n")
    return tmp_path


def test_closure_catches_function_body_lazy_imports(tmp_path):
    fixture_root = _make_fixture_package(tmp_path)
    closure = compute_closure(
        entry_path=fixture_root / "entry.py",
        project_root=fixture_root,
        snapshot=None,  # disk-read fallback OK for the test
    )
    expected = {
        "entry.py",
        "fixture_pkg/__init__.py",
        "fixture_pkg/main.py",
        "fixture_pkg/helpers.py",
        "fixture_pkg/lazy_helper.py",
    }
    assert closure == expected, (
        f"Closure mismatch.\nExpected: {sorted(expected)}\n"
        f"Got: {sorted(closure)}"
    )
    assert "fixture_pkg/unused.py" not in closure, (
        "unused.py should NOT be in the closure"
    )


def test_closure_skips_unparseable_files(tmp_path):
    """SyntaxError in a project file shouldn't crash provenance."""
    fixture_root = _make_fixture_package(tmp_path)
    # Make helpers.py unparseable.
    (fixture_root / "fixture_pkg" / "helpers.py").write_text(
        "def broken(:  # syntax error\n    pass\n"
    )
    with pytest.warns(UserWarning, match="failed to parse"):
        closure = compute_closure(
            entry_path=fixture_root / "entry.py",
            project_root=fixture_root,
        )
    # entry.py, fixture_pkg/__init__.py, main.py, helpers.py (still found
    # via main.py's import) — but lazy_helper not discovered because we
    # can't parse helpers.py. The closure stops gracefully, no crash.
    assert "entry.py" in closure
    assert "fixture_pkg/main.py" in closure


# ─── 2. Real-codebase closure shape ───────────────────────────────────────────


# Files we expect train.py's import chain to reach. If you add a new
# training-relevant module imported by train.py / optim.py / etc., add it here.
EXPECTED_IN_CLOSURE = frozenset({
    "train_lora.py",
    "lora_playground/__init__.py",
    "lora_playground/train.py",
    "lora_playground/optim.py",
    "lora_playground/data.py",
    "lora_playground/training_kernel.py",
    "lora_playground/ucv_layer.py",      # via training_kernel
    "lora_playground/_batched_polar.py",  # via function-body lazy import in optim.py
    "lora_playground/utils.py",
    "lora_playground/spectral.py",
    "lora_playground/_step_timer.py",
    "lora_playground/distributed.py",
    "lora_playground/mfu.py",
})

# Files that MUST NOT be in train's closure (analysis / loader only).
EXPECTED_OUT_OF_CLOSURE = frozenset({
    "lora_playground/loader.py",
    "lora_playground/plotting",   # analysis-side package — never in train.py closure
    "lora_playground/manifest.py",
    "lora_playground/synth_logs.py",
})


def test_real_codebase_closure_includes_expected():
    root = project_root()
    closure = compute_closure(
        entry_path=root / "train_lora.py",
        project_root=root,
    )
    missing_expected = EXPECTED_IN_CLOSURE - closure
    assert not missing_expected, (
        f"Closure missing expected training-side files: "
        f"{sorted(missing_expected)}"
    )


def test_real_codebase_closure_excludes_analysis_side():
    root = project_root()
    closure = compute_closure(
        entry_path=root / "train_lora.py",
        project_root=root,
    )
    leaked = EXPECTED_OUT_OF_CLOSURE & closure
    assert not leaked, (
        f"Analysis/loader-side files leaked into training closure: "
        f"{sorted(leaked)}. This means train.py (or something it imports) "
        f"now imports loader / plot_utils / etc., and execution_source_sha "
        f"will include changes to those — defeating the scoping. Either "
        f"remove the import, or move the imported helper out of analysis-side."
    )


# ─── 3. Anti-decay: no dynamic project imports ────────────────────────────────


_DYNAMIC_IMPORT_PATTERNS = [
    # importlib.import_module("lora_playground.X") etc.
    re.compile(r"importlib\.import_module\s*\(\s*['\"]lora_playground"),
    # __import__("lora_playground.X")
    re.compile(r"__import__\s*\(\s*['\"]lora_playground"),
    # getattr(some_mod, "lora_playground_member")
    re.compile(r"getattr\s*\([^)]*['\"]lora_playground"),
    # Bare string literals that LOOK like dotted module paths into the project.
    # The literal must CLOSE on the path: `"lora_playground.optim"` is an
    # import target, whereas `"lora_playground.lmo_diagnostics to score cheap
    # substitutes "` is an argparse help string that happens to name a module,
    # and flagging it reported prose as a dynamic import forever.
    re.compile(r"['\"]lora_playground\.[A-Za-z_][A-Za-z0-9_.]*['\"]"),
]


def test_no_dynamic_project_imports():
    """Load-bearing CI gate. The AST-only closure design is safe ONLY if
    nothing in the codebase imports project modules dynamically. If this
    test fails, either:
      (a) restructure the import to be static (e.g. `from .X import Y` at
          top of file or function), OR
      (b) extend `lora_playground/execution_scope.py` with an explicit
          handler before introducing the dynamic import.
    """
    root = project_root()
    offenders: list[tuple[str, int, str]] = []
    # Scan training-relevant code paths only — analysis/loader-side code
    # can use dynamic imports of project modules safely (it's not training).
    scan_paths = [
        root / "train_lora.py",
        *(root / "lora_playground").glob("*.py"),
    ]
    # Exclude the analysis/loader side; they're not in the training closure.
    excluded_files = {
        root / "lora_playground" / name for name in (
            "loader.py", "plot_utils.py", "manifest.py", "synth_logs.py",
            "execution_scope.py",
        )
    }
    for p in scan_paths:
        if p in excluded_files:
            continue
        if not p.is_file():
            continue
        text = p.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comments.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in _DYNAMIC_IMPORT_PATTERNS:
                if pat.search(line):
                    offenders.append((
                        str(p.relative_to(root)), lineno, line.strip(),
                    ))
    assert not offenders, (
        "Dynamic project imports detected in training-side code:\n  "
        + "\n  ".join(f"{f}:{ln}: {src}" for f, ln, src in offenders)
        + "\nAST-only closure cannot see dynamic imports. Either restructure "
        "to static imports or update execution_scope.py to handle the pattern."
    )


# ─── 4. __init__.py side-effect freeness (bootstrap-order contract) ──────────


def test_package_init_has_no_side_effects():
    """The Phase-4 bootstrap in train_lora.py runs BEFORE `from
    lora_playground.train import main`. When Python resolves that import,
    it first loads `lora_playground/__init__.py`. If __init__.py imports
    anything from the package, those modules get loaded BEFORE our
    snapshot runs — silently defeating the race protection.

    Lock the contract: __init__.py must have ZERO module-level side
    effects beyond an optional docstring. No imports, no assignments,
    no function/class definitions.
    """
    init_path = project_root() / "lora_playground" / "__init__.py"
    tree = ast.parse(init_path.read_text())
    allowed_node_types = {ast.Expr}  # docstring counts as Expr(Constant)
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring is fine
        if isinstance(node, ast.Expr):
            offenders.append(f"non-docstring expression at line {node.lineno}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            offenders.append(f"import at line {node.lineno}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            offenders.append(f"assignment at line {node.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            offenders.append(f"function def at line {node.lineno}")
        elif isinstance(node, ast.ClassDef):
            offenders.append(f"class def at line {node.lineno}")
        else:
            offenders.append(f"{type(node).__name__} at line {node.lineno}")
    assert not offenders, (
        f"lora_playground/__init__.py has side-effecting top-level statements: "
        f"{offenders}. The Phase-4 source snapshot in train_lora.py runs "
        f"BEFORE the package is imported, but only if __init__.py is "
        f"effect-free. Side effects here break the bootstrap-order contract."
    )


# ─── 5. Provenance assembly + env capture (smoke) ─────────────────────────────


def test_capture_env_returns_consistent_sha():
    env1, sha1 = capture_env()
    env2, sha2 = capture_env()
    assert sha1 == sha2, "capture_env must be deterministic across calls"
    assert "python" in env1
    assert "torch" in env1


def test_compute_execution_provenance_smoke():
    """End-to-end smoke: run compute_execution_provenance against the real
    codebase. Should produce all expected fields without raising."""
    root = project_root()
    snapshot = {}
    snapshot_sha = {}
    # Build a snapshot from disk (test-time only). RECURSIVE, matching
    # `_source_snapshot`'s own disk fallback: the import closure reaches
    # `lora_playground/third_party/imuon_muon.py` (the vendored iMuon optimizer
    # that `imuon-lora` runs verbatim), and a non-recursive glob left those two
    # files out of the snapshot while the closure still demanded them.
    for p in (root / "lora_playground").rglob("*.py"):
        rel = str(p.relative_to(root))
        content = p.read_bytes()
        snapshot[rel] = content
        import hashlib
        snapshot_sha[rel] = hashlib.sha256(content).hexdigest()
    # Also train_lora.py
    p = root / "train_lora.py"
    snapshot[p.name] = p.read_bytes()
    import hashlib as _hh
    snapshot_sha[p.name] = _hh.sha256(snapshot[p.name]).hexdigest()

    prov = compute_execution_provenance(
        entry_path=root / "train_lora.py",
        project_root=root,
        snapshot=snapshot,
        snapshot_sha=snapshot_sha,
        git_commit=None,  # skip dirty computation
    )
    for key in ("execution_source_sha", "execution_source_paths",
                "execution_source_dirty", "execution_env",
                "execution_env_sha"):
        assert key in prov, f"missing field: {key}"
    assert prov["execution_source_dirty"] is None  # since git_commit=None
    # Closure paths must be a non-empty sorted list of strings.
    paths = prov["execution_source_paths"]
    assert isinstance(paths, list) and paths == sorted(paths)
    assert len(paths) >= 10
