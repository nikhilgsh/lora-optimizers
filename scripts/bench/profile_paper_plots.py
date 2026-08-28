#!/usr/bin/env python3
"""Time the plotting calls that are actual entry points in paper_plots.ipynb.

The checked baseline is used by default.  Use ``--no-baseline`` only when
capturing a replacement measurement, and run the opt-in test gate with
``RUN_PAPER_PLOT_PROFILE=1 python -m pytest
tests/test_paper_plots_profile_harness.py -q``.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import io
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lora-paper-plots-mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "scripts" / "bench" / "baselines" / "paper_plots.json"
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
            # Formatting and comments do not change the timed operation.
            operations.append(ast.unparse(call))
    return operations


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=False, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def render_open_figures(plt) -> tuple[float, int, int]:
    """Encode every open figure as PNG, matching notebook inline rendering."""
    started = time.perf_counter()
    total_bytes = 0
    figure_count = 0
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).canvas.print_png(buffer)
        total_bytes += buffer.tell()
        figure_count += 1
    return time.perf_counter() - started, total_bytes, figure_count


def compare_results(
    current: dict,
    baseline: dict,
    *,
    max_regression_fraction: float,
) -> dict:
    """Compare total time only when both runs used the same notebook calls."""
    if current["entrypoints"] != baseline.get("entrypoints"):
        raise ValueError("baseline notebook entrypoints do not match")
    if current.get("render_mode") != baseline.get("render_mode"):
        raise ValueError("baseline render mode does not match")
    baseline_sec = float(
        baseline.get("benchmark_elapsed_sec", baseline["elapsed_sec"])
    )
    current_sec = float(
        current.get("benchmark_elapsed_sec", current["elapsed_sec"])
    )
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
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="record a measurement without comparing it to the checked baseline",
    )
    parser.add_argument("--max-regression-pct", type=float, default=25.0)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="repeat all entrypoints in one process to expose the warm floor",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    process_started = time.perf_counter()
    import matplotlib.pyplot as plt
    import lora_playground.plotting.paper_figs as paper_figs
    import lora_playground.plotting.paper_plots_lib as P
    P.begin_notebook_snapshot(refresh=True)
    startup_elapsed_sec = time.perf_counter() - process_started

    paper_figs.FIGS = args.out.parent / f".{args.out.stem}-figures"
    paper_figs.FIGS.mkdir(parents=True, exist_ok=True)

    entrypoints = discover_entrypoints(args.notebook)
    passes = []
    error = None
    entrypoints_started = time.perf_counter()
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink):
            for repeat in range(args.repeats):
                call_elapsed_sec = []
                render_elapsed_sec = []
                rendered_png_bytes = []
                rendered_figure_count = []
                pass_started = time.perf_counter()
                for index, expression in enumerate(entrypoints):
                    call_started = time.perf_counter()
                    try:
                        eval(expression, {"P": P})
                        call_elapsed_sec.append(
                            time.perf_counter() - call_started
                        )
                        render_sec, png_bytes, figure_count = (
                            render_open_figures(plt)
                        )
                        render_elapsed_sec.append(render_sec)
                        rendered_png_bytes.append(png_bytes)
                        rendered_figure_count.append(figure_count)
                    except Exception as exc:
                        error = (
                            f"repeat {repeat}, entrypoint {index}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    finally:
                        plt.close("all")
                    if error is not None:
                        break
                passes.append({
                    "call_elapsed_sec": call_elapsed_sec,
                    "render_elapsed_sec": render_elapsed_sec,
                    "rendered_png_bytes": rendered_png_bytes,
                    "rendered_figure_count": rendered_figure_count,
                    "elapsed_sec": time.perf_counter() - pass_started,
                })
                if error is not None:
                    break

    entrypoints_elapsed_sec = time.perf_counter() - entrypoints_started
    result = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd().resolve()),
        "repo_root": _git_output("rev-parse", "--show-toplevel"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "notebook": str(args.notebook.resolve()),
        "entrypoints": entrypoints,
        "render_mode": "png",
        "repeats": args.repeats,
        "passes": passes,
        "call_elapsed_sec": passes[0]["call_elapsed_sec"],
        "render_elapsed_sec": passes[0]["render_elapsed_sec"],
        "rendered_png_bytes": passes[0]["rendered_png_bytes"],
        "rendered_figure_count": passes[0]["rendered_figure_count"],
        "startup_elapsed_sec": startup_elapsed_sec,
        "entrypoints_elapsed_sec": entrypoints_elapsed_sec,
        "benchmark_elapsed_sec": startup_elapsed_sec + passes[0]["elapsed_sec"],
        "elapsed_sec": time.perf_counter() - process_started,
        "error": error,
    }
    # Preserve the raw measurement before any summary is printed or a baseline
    # is parsed. A bad baseline must not discard the completed timing run.
    _write_result(args.out, result)

    comparison_error = None
    baseline = None if args.no_baseline else args.baseline
    if baseline is not None and error is None:
        try:
            result["comparison"] = compare_results(
                result,
                json.loads(baseline.read_text()),
                max_regression_fraction=args.max_regression_pct / 100.0,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            comparison_error = f"{type(exc).__name__}: {exc}"
            result["comparison_error"] = comparison_error
        _write_result(args.out, result)

    print(
        f"profile written to {args.out}: {len(passes[0]['call_elapsed_sec'])}/"
        f"{len(entrypoints)} calls x {len(passes)} in "
        f"{entrypoints_elapsed_sec:.3f}s "
        f"({result['benchmark_elapsed_sec']:.3f}s first-pass benchmark)"
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
