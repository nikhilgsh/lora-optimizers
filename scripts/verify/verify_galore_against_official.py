"""Behavioral equivalence test: our GaLoreAdamW vs the official ~/GaLore/galore_torch.AdamW.

Same weights, same gradients, same hyperparameters, run a few optimizer steps,
compare the weight trajectories. Tolerance ~1e-5 (float32 numerics).

Usage: python scripts/verify/verify_galore_against_official.py
Requires ~/GaLore on PYTHONPATH.
"""
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
GALORE_REPO = Path("/mnt/home/nghosh/GaLore")
sys.path.insert(0, str(REPO_ROOT))

# Copy official galore_projector.py + adamw.py into a fresh package name to
# avoid transformers' import_utils detecting the `galore_torch` namespace and
# breaking. We import the verbatim source files; no monkey-patching.
import importlib.util, types

OFFICIAL_PKG = "_galore_official"
pkg = types.ModuleType(OFFICIAL_PKG); pkg.__path__ = []
sys.modules[OFFICIAL_PKG] = pkg

def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"{OFFICIAL_PKG}.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{OFFICIAL_PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

_load("galore_projector", GALORE_REPO / "galore_torch" / "galore_projector.py")
# Stub the tensor projector (uses tensorly; we don't exercise it).
_tensor = types.ModuleType(f"{OFFICIAL_PKG}.galore_projector_tensor")
_tensor.GaLoreProjectorTensor = None
sys.modules[f"{OFFICIAL_PKG}.galore_projector_tensor"] = _tensor

# adamw.py uses relative imports; rewrite source to absolute and exec.
_adamw_src = (GALORE_REPO / "galore_torch" / "adamw.py").read_text()
_adamw_src = _adamw_src.replace(
    "from .galore_projector import GaLoreProjector",
    f"from {OFFICIAL_PKG}.galore_projector import GaLoreProjector",
).replace(
    "from .galore_projector_tensor import GaLoreProjectorTensor",
    f"from {OFFICIAL_PKG}.galore_projector_tensor import GaLoreProjectorTensor",
)
_adamw_ns = {"__name__": f"{OFFICIAL_PKG}.adamw"}
exec(compile(_adamw_src, str(GALORE_REPO / "galore_torch" / "adamw.py"), "exec"), _adamw_ns)
OfficialAdamW = _adamw_ns["AdamW"]

from lora_playground.optim import GaLoreAdamW as OursGaLore
from lora_playground.utils import TargetWeight


def make_model(seed=0):
    """Tiny model with both tall (d_out > d_in) and wide (d_out < d_in) linears,
    so we exercise both projection axes."""
    torch.manual_seed(seed)
    return nn.ModuleDict({
        "tall": nn.Linear(8, 16, bias=False),    # weight: (16, 8) tall
        "wide": nn.Linear(16, 8, bias=False),    # weight: (8, 16) wide
    })


def set_grads(model, seed=42):
    """Deterministic gradients on each weight."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.grad = 0.1 * torch.randn_like(p)


def main():
    rank = 4
    update_proj_gap = 5
    scale = 1.0
    lr = 3e-4
    eps = 1e-6
    betas = (0.9, 0.999)
    n_steps = 12  # ≥ 2 × update_proj_gap so we exercise the proj refresh path

    # === Ours ===
    ours_model = make_model()
    targets = [
        TargetWeight(name=name, module=mod, weight=mod.weight,
                     base_weight=mod.weight.detach().clone())
        for name, mod in ours_model.items()
    ]
    ours_opt = OursGaLore(
        targets, rank=rank, lr=lr, betas=betas, eps=eps, weight_decay=0.0,
        update_proj_gap=update_proj_gap, scale=scale, proj_type="std",
    )

    # === Official ===
    official_model = make_model()  # same seed → same init
    galore_param_group = {
        "params": list(official_model.parameters()),
        "rank": rank,
        "update_proj_gap": update_proj_gap,
        "scale": scale,
        "proj_type": "std",
    }
    official_opt = OfficialAdamW(
        [galore_param_group], lr=lr, betas=betas, eps=eps,
        weight_decay=0.0, no_deprecation_warning=True,
    )

    # Sanity check: starting weights identical
    for (n, p_ours), (_, p_off) in zip(ours_model.named_parameters(),
                                        official_model.named_parameters()):
        assert torch.equal(p_ours, p_off), f"init mismatch on {n}"

    # === Run n_steps with synced gradients ===
    max_diff_per_step = []
    for step in range(1, n_steps + 1):
        # Generate identical gradients for both models from same seed
        set_grads(ours_model, seed=step)
        set_grads(official_model, seed=step)

        # Verify grads identical pre-step
        for (n, p_ours), (_, p_off) in zip(ours_model.named_parameters(),
                                            official_model.named_parameters()):
            assert torch.equal(p_ours.grad, p_off.grad), \
                f"step {step}: grad mismatch on {n}"

        ours_opt.step()
        official_opt.step()

        # Compare resulting weights
        max_diff = 0.0
        worst = None
        for (n, p_ours), (_, p_off) in zip(ours_model.named_parameters(),
                                            official_model.named_parameters()):
            d = (p_ours - p_off).abs().max().item()
            if d > max_diff:
                max_diff = d
                worst = n
        max_diff_per_step.append((step, max_diff, worst))
        print(f"  step {step:2d}: max |Δw| = {max_diff:.3e} (worst layer: {worst})")

    final_max = max_diff_per_step[-1][1]
    tol = 1e-5
    print(f"\nFinal max |Δw| across {n_steps} steps: {final_max:.3e}")
    if final_max < tol:
        print(f"✅ PASS (tol={tol})")
        sys.exit(0)
    else:
        print(f"❌ FAIL — divergence > {tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
