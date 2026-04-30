"""Behavioral equivalence test: our PSILoRA vs official ScaledOPLoraOptimizer
(diagonal K-FAC mode) from ~/PSI-LoRA/src/oplora/optimizer.py.

Same convention as scripts/verify_galore_against_official.py: build a tiny
LoRA model, run a forward+backward to populate the hook caches, run one
optimizer step on each, compare resulting weights.

We start with momentum=0 (α₁=0) — both formulations should agree there.
A passing test under that config validates the F-LoRSUM ALS math + diagonal
K-FAC stats. Momentum-on requires resolving a paper-vs-reference-impl
coefficient discrepancy first (see test output for divergence breakdown).

Usage: python scripts/verify_psilora_against_official.py
"""
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
PSI_REPO = Path("/mnt/home/nghosh/PSI-LoRA")
sys.path.insert(0, str(REPO_ROOT))

# Import the reference's oplora module without polluting global namespace.
# adamw.py is a sibling; only need oplora.{utils, optimizer}.
import importlib.util, types

OFFICIAL_PKG = "_psilora_official"
pkg = types.ModuleType(OFFICIAL_PKG); pkg.__path__ = []
sys.modules[OFFICIAL_PKG] = pkg


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"{OFFICIAL_PKG}.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{OFFICIAL_PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_utils = _load("utils", PSI_REPO / "src" / "oplora" / "utils.py")

# optimizer.py uses relative imports — rewrite source to absolute.
_opt_src = (PSI_REPO / "src" / "oplora" / "optimizer.py").read_text()
_opt_src = _opt_src.replace(
    "from .utils import",
    f"from {OFFICIAL_PKG}.utils import",
)
_opt_ns = {"__name__": f"{OFFICIAL_PKG}.optimizer"}
exec(compile(_opt_src, str(PSI_REPO / "src" / "oplora" / "optimizer.py"), "exec"), _opt_ns)
ScaledOPLoraOptimizer = _opt_ns["ScaledOPLoraOptimizer"]

from lora_playground.optim import PSILoRA as OursPSI


class FakeLoRA(nn.Module):
    """Mimics PEFT's lora.Linear: a module with lora_A / lora_B ModuleDicts.
    Both reference and ours hook on `hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B')`.
    """
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        # PEFT default init: A kaiming, B zero.
        torch.nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        # Standard LoRA: y = (alpha/r) · B(A(x)). We absorb alpha/r into the gradient
        # for simplicity — what matters is the hook caches see X (input) and S
        # (output grad) consistently between both optimizers.
        h = self.lora_A["default"](x)
        return self.lora_B["default"](h)


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(FakeLoRA(8, 16, 4), FakeLoRA(16, 8, 4))


def main():
    # Hyperparameter map (ours → reference):
    #   gamma=0.5         ↔ metric_power=0.5
    #   ema_beta=0.99     ↔ betas[1]=0.99
    #   delta=1e-5        ↔ damping=1e-5
    #   momentum=α₁       ↔ betas[0]
    #   proximal_rho      ↔ lmbd  (LR_LMBD=True scales both by η internally)
    #   inner_iters=1     ↔ lookahead_iters=1
    lr = 3e-4
    rho = 0.01
    delta = 1e-5
    gamma = 0.5
    ema_beta = 0.99
    alpha1 = 0.0   # IMPORTANT: momentum off — eliminates a paper-vs-ref-impl
                   # coefficient discrepancy on the gradient term. Momentum-on
                   # is a separate investigation.
    K = 1

    ours_model = make_model(seed=0)
    ref_model = make_model(seed=0)
    # Sanity: identical init
    for (n, p_ours), (_, p_ref) in zip(ours_model.named_parameters(),
                                        ref_model.named_parameters()):
        assert torch.equal(p_ours, p_ref), f"init mismatch on {n}"

    ours_opt = OursPSI(
        ours_model, lr=lr,
        gamma=gamma, ema_beta=ema_beta, delta=delta,
        momentum=alpha1, inner_iters=K, proximal_rho=rho,
    )
    ref_opt = ScaledOPLoraOptimizer(
        ref_model, lr=lr,
        lmbd=rho, lookahead_iters=K,
        betas=(alpha1, ema_beta),
        damping=delta,
        kfac_metric=True,
        metric_type="diagonal",
        metric_power=gamma,
        init_metric_scale=1.0,
        loss_reduction="mean",
        rank_multiplier=1.0,
    )

    # Multi-step run with deterministic-but-changing inputs.
    # We need both lora_B's to leave 0 by feeding non-zero gradient through, so
    # we set B to a small init *after* optimizer construction (otherwise loss
    # would stay 0 and gradients would all be 0 forever).
    with torch.no_grad():
        for mod in [ours_model, ref_model]:
            for layer in mod:
                layer.lora_B["default"].weight.fill_(0.01)

    n_steps = 5
    max_diff_per_step = []
    for step in range(1, n_steps + 1):
        torch.manual_seed(100 + step)
        x = torch.randn(4, 8)

        # Note: float32 noise from earlier steps drifts the weights apart
        # slightly, so gradients diverge proportionally. We don't assert grad
        # equality here — the test is on cumulative drift staying below tol.
        ours_model(x).pow(2).mean().backward()
        ref_model(x).pow(2).mean().backward()

        ours_opt.step()
        ref_opt.step()
        for p in ours_model.parameters(): p.grad = None
        for p in ref_model.parameters(): p.grad = None

        step_max = 0.0; worst = None
        for (n, p_ours), (_, p_ref) in zip(ours_model.named_parameters(),
                                            ref_model.named_parameters()):
            d = (p_ours - p_ref).abs().max().item()
            if d > step_max:
                step_max = d; worst = n
        max_diff_per_step.append((step, step_max, worst))
        print(f"  step {step}: max|Δw|={step_max:.3e} (worst: {worst})")

    final_max = max(d for _, d, _ in max_diff_per_step)
    # Tolerance: PSI-LoRA's nested ALS solves are more numerically delicate than
    # plain Adam, so float32 drift across multi-step is ~1e-5 even when single-step
    # is bit-exact. 1e-4 ≈ 0.2% rel on 0.01-scale weights — still tight enough to
    # catch any algorithmic divergence.
    tol = 1e-4
    print(f"\nFinal max |Δw| across {n_steps} steps: {final_max:.3e}")
    if final_max < tol:
        print(f"✅ PASS (tol={tol}) — F-LoRSUM math matches reference with α₁=0, "
              f"diagonal K-FAC, lookahead_iters=1")
        sys.exit(0)
    else:
        print(f"❌ FAIL — max |Δw| {final_max:.3e} > tol {tol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
