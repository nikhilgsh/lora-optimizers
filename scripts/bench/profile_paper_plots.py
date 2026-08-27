#!/usr/bin/env python3
"""Time the plotting calls that are actual entry points in paper_plots.ipynb."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lora-paper-plots-mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def discover_entrypoints(notebook: Path) -> list[str]:
    """Find code cells whose sole statement is a call rooted at ``P``."""
    operations = []
    for cell in json.loads(notebook.read_text()).get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        if cell.get("cell_type") != "code" or not source:
            continue
        try:
            body = ast.parse(source).body
        except SyntaxError:  # IPython setup cells are not plotting entry points.
            continue
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            continue
        call = body[0].value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "P"
        ):
            operations.append(source)
    return operations


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def compare_results(
    current: dict,
    baseline: dict,
    *,
    max_regression_fraction: float,
) -> dict:
    """Compare total time only when both runs used the same notebook calls."""
    if current["entrypoints"] != baseline.get("entrypoints"):
        raise ValueError("baseline notebook entrypoints do not match")
    baseline_sec = float(baseline["elapsed_sec"])
    current_sec = float(current["elapsed_sec"])
    limit_sec = baseline_sec * (1.0 + max_regression_fraction)
    return {
        "baseline_elapsed_sec": baseline_sec,
        "current_elapsed_sec": current_sec,
        "ratio": current_sec / baseline_sec if baseline_sec else float("inf"),
        "limit_sec": limit_sec,
        "passed": current_sec <= limit_sec,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook", type=Path, default=ROOT / "paper" / "paper_plots.ipynb"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/paper_plots_profile.json")
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-pct", type=float, default=25.0)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import lora_playground.plotting.paper_figs as paper_figs
    import lora_playground.plotting.paper_plots_lib as P

    paper_figs.FIGS = args.out.parent / f".{args.out.stem}-figures"
    paper_figs.FIGS.mkdir(parents=True, exist_ok=True)

    entrypoints = discover_entrypoints(args.notebook)
    call_elapsed_sec = []
    error = None
    started = time.perf_counter()
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink):
            for index, expression in enumerate(entrypoints):
                call_started = time.perf_counter()
                try:
                    eval(expression, {"P": P})
                except Exception as exc:
                    error = f"entrypoint {index}: {type(exc).__name__}: {exc}"
                finally:
                    call_elapsed_sec.append(time.perf_counter() - call_started)
                    plt.close("all")
                if error is not None:
                    break

    result = {
        "notebook": str(args.notebook.resolve()),
        "entrypoints": entrypoints,
        "call_elapsed_sec": call_elapsed_sec,
        "elapsed_sec": time.perf_counter() - started,
        "error": error,
    }
    # Preserve the raw measurement before any summary is printed or a baseline
    # is parsed. A bad baseline must not discard the completed timing run.
    _write_result(args.out, result)

    comparison_error = None
    if args.baseline is not None and error is None:
        try:
            result["comparison"] = compare_results(
                result,
                json.loads(args.baseline.read_text()),
                max_regression_fraction=args.max_regression_pct / 100.0,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            comparison_error = f"{type(exc).__name__}: {exc}"
            result["comparison_error"] = comparison_error
        _write_result(args.out, result)

    print(
        f"profile written to {args.out}: {len(call_elapsed_sec)}/"
        f"{len(entrypoints)} calls in {result['elapsed_sec']:.3f}s"
    )
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if comparison_error is not None:
        print(f"BASELINE ERROR: {comparison_error}", file=sys.stderr)
        return 2
    comparison = result.get("comparison")
    if comparison is not None:
        verdict = "PASS" if comparison["passed"] else "REGRESSION"
        print(f"baseline {verdict}: {comparison['ratio']:.3f}x")
        return 0 if comparison["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
