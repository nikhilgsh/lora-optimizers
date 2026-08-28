"""Which metric slot actually steers the PoLoRA update?

The A-update in Alg 1 (paper/manuscript/main.tex) is

    D_A = C_B^{-1/2} msign( C_B^{-1/2} Mhat_A Q^{-1/2} ) Q^{-1/2},

with C_B = B^T P B an r x r matrix and Q a d_in-wide diagonal. If the LEFT slot
held any scalar multiple c*I, it would vanish from the applied update exactly:
msign ignores scale, and the final rescale to ||dA||_2 = rho divides out the
leftover c^{-1/2}. So each slot can only act through its EIGENVALUE SPREAD.

This measures how much each slot's spread is worth, on REAL gradients at the
paper's r=256 setting, by recomputing D_A with that slot flattened to a scalar
multiple of the identity (its mean eigenvalue) and taking the cosine against the
true D_A. cos ~ 1 means the slot's content is nearly inert; cos << 1 means it is
doing real work.

Reported per LoRA pair and aggregated. Also reports each slot's effective
spectrum AFTER the relative damping the optimizer actually applies, since that
damping caps the conditioning at ~1/delta and is what the update really sees.
"""
import argparse, json, os, torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from lora_playground.data import PackingCollator
from lora_playground.training_kernel import build_peft_model
from lora_playground.optim import CurvatureWhitenLoRA, gram_ns_inv_sqrt
from lora_playground.optim_specs import REGISTRY


def unit(X):
    n = X.norm()
    return X / n if n > 1e-30 else X


def cos(X, Y):
    return float((unit(X).flatten() @ unit(Y).flatten()).clamp(-1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data_dir", default="data/openmath_instruct_2_2m_packed_seq2048_llama32")
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warm_steps", type=int, default=30,
                    help="steps to warm the P/Q metrics and momentum before measuring")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--out", default="bench_abl_out/metric_slots.json")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)

    # `PeftModel` here is this repo's container (training_kernel.py), not peft's
    # class: `.bare_model` is what the optimizer collects LoRA pairs from and
    # `.train_model` is what you call forward on. Same split bench/harness.py
    # uses in measure_step_wall.
    peft = build_peft_model(
        model_name=a.model_name, training_mode="lora", target_modules="all-linear",
        lora_r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
        dtype=torch.bfloat16, attn_implementation="sdpa", use_liger=False,
        liger_flce=False, gradient_checkpointing=False, compile_mode=None,
        device=dev, world_size=1, local_rank=0)
    bare, model = peft.bare_model, peft.train_model
    model.train()

    ds = load_from_disk(os.path.join(a.data_dir, "train"))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    collate_fn=PackingCollator(seq_length=a.max_seq_length,
                                               mask_dtype=torch.bfloat16))

    # EXACTLY the protagonist identity flags, pulled from the registry so they
    # cannot drift from what the sweeps run.
    identity = dict(REGISTRY["kl-diag-polar-lora"].fixed)
    opt = CurvatureWhitenLoRA(
        bare, lr=a.lr, betas=(0.9, 0.999), delta=1e-4, curvature_beta=0.99,
        ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
        cw_nesterov=True, precond_refresh_every=10, precond_method="gram_ns",
        higham_iters=8, cw_metric_init="1e-12", **identity)

    it = iter(dl)
    for step in range(a.warm_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum_steps):
            batch = {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}
            out = model(**batch)
            (out.loss / a.grad_accum_steps).backward()
        opt.step()
        if step % 10 == 0:
            print(f"warm step {step}: loss {out.loss.item():.4f}", flush=True)

    # One more set of gradients to measure against.
    opt.zero_grad(set_to_none=True)
    for _ in range(a.grad_accum_steps):
        batch = {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}
        out = model(**batch)
        (out.loss / a.grad_accum_steps).backward()

    rows = []
    b1 = opt.beta1
    with torch.no_grad():
        for i, (A, B) in enumerate(opt.pairs):
            st = opt.pair_state[i]
            gA = A.grad.float()
            mA = st['m_A'] * b1 + gA * (1 - b1)
            Mhat = mA * b1 + gA * (1 - b1)              # Nesterov look-ahead
            Bw = B.detach().float()

            Q_isqrt = opt._rdinv(st['Q'].unsqueeze(0)).squeeze(0)
            P_isqrt = opt._rdinv(st['P'].unsqueeze(0)).squeeze(0)
            P = P_isqrt.square().reciprocal()
            CB = Bw.T @ (P.unsqueeze(-1) * Bw)                        # C_B = B^T P B
            CB = 0.5 * (CB + CB.T)

            def polar(z):
                return opt._polar_ns_guarded(z.unsqueeze(0), [dict(st)], 'v_m').squeeze(0)

            CBh = gram_ns_inv_sqrt(CB.unsqueeze(0), nsteps=opt.higham_iters,
                                   eps=opt.delta, eps_relative=True).squeeze(0)
            # true update direction
            D_true = CBh @ polar(CBh @ Mhat * Q_isqrt.unsqueeze(0)) * Q_isqrt.unsqueeze(0)
            # LEFT slot flattened to a scalar multiple of I (its mean eigenvalue):
            # the scalar cancels, so the direction is msign(Mhat Q^-1/2) Q^-1/2.
            D_isoCB = polar(Mhat * Q_isqrt.unsqueeze(0)) * Q_isqrt.unsqueeze(0)
            # RIGHT slot flattened likewise: direction is C_B^-1/2 msign(C_B^-1/2 Mhat).
            D_isoQ = CBh @ polar(CBh @ Mhat)

            ev = torch.linalg.eigvalsh(CB.double())
            ev = ev.clamp_min(0)
            # effective spectrum after the optimizer's relative damping
            ev_eff = ev + opt.delta * ev.max()
            q = st['Q'].double()
            q_eff = q / q.max().clamp_min(1e-30) + opt.delta
            rows.append(dict(
                pair=i, r=A.shape[0], d_in=A.shape[1], d_out=B.shape[0],
                cos_isoCB=cos(D_true, D_isoCB), cos_isoQ=cos(D_true, D_isoQ),
                CB_cond_eff=float(ev_eff.max() / ev_eff.min()),
                CB_disp=float(ev_eff.max() / ev_eff.mean()),
                Q_cond_eff=float(q_eff.max() / q_eff.min()),
                Q_disp=float(q_eff.max() / q_eff.mean()),
            ))

    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n
           for k in ("cos_isoCB", "cos_isoQ", "CB_cond_eff", "CB_disp",
                     "Q_cond_eff", "Q_disp")}
    print("\n=== metric-slot ablation, real gradients, "
          f"{a.model_name} r={a.lora_r}, {n} LoRA pairs, after {a.warm_steps} steps ===")
    print("cos(D_A true, D_A with that slot flattened to a scalar multiple of I):")
    print(f"  LEFT  slot C_B (r x r)      flattened -> cos = {agg['cos_isoCB']:.4f}")
    print(f"  RIGHT slot Q   (d_in diag)  flattened -> cos = {agg['cos_isoQ']:.4f}")
    print("effective spectrum actually seen (after the optimizer's relative damping):")
    print(f"  C_B: lam_max/lam_min = {agg['CB_cond_eff']:.1f}   lam_max/mean = {agg['CB_disp']:.1f}")
    print(f"  Q  : q_max/q_min     = {agg['Q_cond_eff']:.1f}   q_max/mean   = {agg['Q_disp']:.1f}")
    lo = min(rows, key=lambda r: r['cos_isoCB'])
    print(f"per-pair spread of cos_isoCB: min {lo['cos_isoCB']:.4f} (pair {lo['pair']}), "
          f"max {max(r['cos_isoCB'] for r in rows):.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(config=vars(a), aggregate=agg, per_pair=rows), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
