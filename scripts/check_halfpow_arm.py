"""kl-diag-flatout-lora must give D_A = C_B^-1/2 Mhat_A Q^-1/2 (metric applied ONCE),
distinct from kl-diag-lora's C_B^-1 Mhat_A Q^-1 (metric applied TWICE).

The point of the arm: dropping msign from the protagonist also doubles the metric's
power, because the inner whiten and the outer un-whiten compose. So "no msign" and
"over-preconditioned" are confounded in kl-diag-lora, and this is the control.
"""
import copy
import torch
import torch.nn as nn

from lora_playground.optim import CurvatureWhitenLoRA, gram_ns_inv_sqrt
from lora_playground.optim_specs import REGISTRY

torch.manual_seed(0)
R, DIN, DOUT = 8, 32, 48
A0 = torch.randn(R, DIN) * 0.1
B0 = torch.randn(DOUT, R) * 0.1
GA = [torch.randn(R, DIN) * 1e-2 for _ in range(6)]
GB = [torch.randn(DOUT, R) * 1e-2 for _ in range(6)]


def mk():
    m = nn.Module()
    m.lora_A = nn.ModuleDict({"default": nn.Linear(DIN, R, bias=False)})
    m.lora_B = nn.ModuleDict({"default": nn.Linear(R, DOUT, bias=False)})
    with torch.no_grad():
        m.lora_A["default"].weight.copy_(A0)
        m.lora_B["default"].weight.copy_(B0)
    t = nn.Module()
    t.mods = nn.ModuleList([m])
    return t


def build(name):
    return CurvatureWhitenLoRA(
        mk(), lr=1e-2, betas=(0.9, 0.999), delta=1e-4, curvature_beta=0.99,
        ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
        cw_nesterov=True, precond_refresh_every=10, precond_method="gram_ns",
        higham_iters=8, cw_metric_init="1e-12", **dict(REGISTRY[name].fixed))


def run(o, n=5):
    for s in range(n):
        for A, B in o.pairs:
            A.grad = GA[s].clone()
            B.grad = GB[s].clone()
        o.step()
    st = copy.deepcopy(o.pair_state[0])
    A, B = o.pairs[0]
    Ab = A.detach().clone()
    A.grad = GA[n].clone()
    B.grad = GB[n].clone()
    o.step()
    return st, (A.detach() - Ab)


def unit(X):
    return X / X.norm()


def cos(X, Y):
    return float((unit(X).flatten() @ unit(Y).flatten()).clamp(-1, 1))


for name in ["kl-diag-flatout-lora", "kl-diag-lora", "kl-diag-polar-lora"]:
    o = build(name)
    st, dA = run(o)
    b1 = o.beta1
    gA = GA[5]
    mA = st['m_A'] * b1 + gA * (1 - b1)
    Mhat = mA * b1 + gA * (1 - b1)
    Q_isqrt = o._rdinv(st['Q'].unsqueeze(0)).squeeze(0)
    P_isqrt = o._rdinv(st['P'].unsqueeze(0)).squeeze(0)
    Bw = o.pairs[0][1].detach().float()
    CB = Bw.T @ (P_isqrt.square().reciprocal().unsqueeze(-1) * Bw)
    CB = 0.5 * (CB + CB.T)
    CBh = gram_ns_inv_sqrt(CB.unsqueeze(0), nsteps=o.higham_iters,
                           eps=o.delta, eps_relative=True).squeeze(0)
    HALF = CBh @ Mhat * Q_isqrt.unsqueeze(0)                            # C_B^-1/2 Mhat Q^-1/2
    FULL = CBh @ (CBh @ Mhat * Q_isqrt.unsqueeze(0)) * Q_isqrt.unsqueeze(0)  # C_B^-1 Mhat Q^-1
    sv = torch.linalg.svdvals(dA.float())
    print(f"{name:26s} flat_outer={str(o.flat_outer):5s} use_polar={str(o.use_polar):5s}")
    print(f"{'':26s} cos(dA, HALF) = {cos(-dA, HALF):.6f}   "
          f"cos(dA, FULL) = {cos(-dA, FULL):.6f}   "
          f"stable_rank(dA) = {float((sv**2).sum()/sv[0]**2):.2f} (r={R})")
