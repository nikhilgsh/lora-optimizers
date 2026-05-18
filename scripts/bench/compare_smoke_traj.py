"""Compare two training trajectories from `smoke_higham_compare.sh`."""
import json
import sys
from pathlib import Path

FP32 = Path("/tmp/smoke_higham_runs/fp32.jsonl")
FP16 = Path("/tmp/smoke_higham_runs/fp16.jsonl")


def load_evals(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("event") == "eval":
            out.append(obj)
    return out


def main():
    fp32 = load_evals(FP32)
    fp16 = load_evals(FP16)
    if not fp32 or not fp16:
        print(f"empty traj: fp32 n={len(fp32)}, fp16 n={len(fp16)}")
        sys.exit(1)

    print(f"{'step':>6s} | {'fp32 loss':>12s} | {'fp16 loss':>12s} | {'Δ':>10s} | "
          f"{'rel Δ':>10s}")
    print("-" * 65)
    for a, b in zip(fp32, fp16):
        if a["step"] != b["step"]:
            continue
        la = a["eval_loss"]
        lb = b["eval_loss"]
        d = lb - la
        rel = d / max(abs(la), 1e-12)
        print(f"{a['step']:>6d} | {la:12.6f} | {lb:12.6f} | "
              f"{d:+10.6f} | {rel:+10.4%}")


if __name__ == "__main__":
    main()
