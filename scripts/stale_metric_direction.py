"""Does the one-step-stale metric corrupt the update DIRECTION, or only its scale?

PoLoRA's diagonal metric state Q/P is read at the top of
CurvatureWhitenLoRA._cw_apply_grouped (lora_playground/optim.py:1849-1850), used to
build the entire update, and only THEN accumulated with the current step's gradient
(optim.py:2079-2080 coupled, :2087-2088 else). So the metric that shapes step t has
never seen g_t.

Two reasons that might not matter, and one reason it might:

  * msign is scale-invariant -- msign(cX) = msign(X) for scalar c > 0 -- so a stale
    metric that is globally mis-scaled washes out entirely.
  * the magnitude rule sets the merged update to spectral norm eta regardless, so a
    stale metric cannot make the step explode or vanish.
  * NEITHER of those touches an ANISOTROPIC error. A stale S~ = S + E with E not a
    multiple of S rotates S^{-1/2} m_hat, and msign can amplify that rotation, since
    near-degenerate singular values let a small perturbation move singular vectors a
    lot. That is a direction channel, and nothing in the design closes it.

scripts/grad_shape_splithalf.py does NOT answer this: it measures how much the
metric's diagonal energy shape moves (cos ~0.96 between disjoint halves of one
batch), which is a statement in the metric's space, not in the update's. A 4%
perturbation of the metric does not imply a 4% perturbation of msign(S^{-1/2} m D^{-1/2}).

The contrast needs no new optimizer code, because one ordinary step already produces
the fresh metric as a side effect:

    stale:  from state (D^{t-1}, m, W) step with gradient g  ->  dW_stale
            (this leaves D^t = cb*D^{t-1} + (1-cb)*f(g) in pair_state)
    fresh:  restore W and m, KEEP D^t, step again with the SAME g  ->  dW_fresh

Same gradient, same momentum, same weights; the only difference is whether the
metric saw g. Reports, per curvature_beta:

    cos(dW)      cosine between the merged updates B_post A_post - B_pre A_pre
    ang(dW)      the same as an angle in degrees -- the number that matters
    |dW| ratio   fresh/stale spectral-norm ratio (should be ~1: the rho rule pins it)
    cos(dA)      cosine on the A-factor step alone
    cos(dB)      cosine on the B-factor step alone

Prediction if staleness is only a scale effect: cos(dW) ~ 1 at every beta2. If there
is a direction channel, cos(dW) should fall as beta2 falls, because the excluded
gradient carries weight (1-beta2) in the metric -- 1% at beta2=0.99 against 19% at
beta2 = beta1^2 = 0.81.

Usage:
  python scripts/stale_metric_direction.py --data_dir data/<...> --tag <label>
"""
import argparse
import copy
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from lora_playground.data import PackingCollator
from lora_playground.optim import CurvatureWhitenLoRA
from lora_playground.optim_specs import REGISTRY
from lora_playground.training_kernel import build_peft_model


def snapshot(opt, npairs):
    """Full restore point: factor weights, every pair_state entry, and the grads
    (step() zeroes A.grad/B.grad on its way out, so they must be saved too)."""
    st = []
    for i in range(npairs):
        A, B = opt.pairs[i]
        st.append(dict(
            A=A.detach().clone(), B=B.detach().clone(),
            gA=A.grad.detach().clone(), gB=B.grad.detach().clone(),
            state={k: (v.clone() if torch.is_tensor(v) else v)
                   for k, v in opt.pair_state[i].items()},
        ))
    return st


def restore(opt, snap, npairs, keep_metric=None):
    """Put the optimizer back at the snapshot. `keep_metric`, when given, is a list of
    (Q, P) that override the snapshot's -- that is how the fresh arm keeps the
    post-accumulation metric while rewinding everything else."""
    for i in range(npairs):
        A, B = opt.pairs[i]
        with torch.no_grad():
            A.copy_(snap[i]['A']); B.copy_(snap[i]['B'])
        A.grad = snap[i]['gA'].clone()
        B.grad = snap[i]['gB'].clone()
        opt.pair_state[i] = {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in snap[i]['state'].items()}
        if keep_metric is not None:
            opt.pair_state[i]['Q'] = keep_metric[i][0].clone()
            opt.pair_state[i]['P'] = keep_metric[i][1].clone()


def merged_delta(pre_A, pre_B, post_A, post_B):
    """The update the loss actually sees: B_post A_post - B_pre A_pre (d_out x d_in)."""
    return (post_B.float() @ post_A.float()) - (pre_B.float() @ pre_A.float())


def cos_flat(x, y):
    nx, ny = x.norm(), y.norm()
    if nx == 0 or ny == 0:
        return float("nan")
    return float((x.flatten() @ y.flatten()) / (nx * ny))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora_r", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warm_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--max_pairs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3, help="probe steps averaged per beta2")
    ap.add_argument("--betas2", default="0.81,0.9090,0.9564,0.9791,0.99",
                    help="curvature_beta grid (the sweep's grid by default)")
    ap.add_argument("--out", default="bench_abl_out/stale_metric_direction.jsonl")
    a = ap.parse_args()

    grid = [float(x) for x in a.betas2.split(",")]
    dev = torch.device("cuda")
    tokens_per_step = a.batch_size * a.grad_accum_steps * a.max_seq_length
    print(f"{a.model_name} r={a.lora_r} | {tokens_per_step} tokens/step "
          f"({a.batch_size}x{a.grad_accum_steps}x{a.max_seq_length}) | lr={a.lr}", flush=True)

    ds = load_from_disk(os.path.join(a.data_dir, "train"))
    results = {}
    for cb in grid:
        torch.manual_seed(0)
        peft = build_peft_model(
            model_name=a.model_name, training_mode="lora", target_modules="all-linear",
            lora_r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
            dtype=torch.bfloat16, attn_implementation="sdpa", use_liger=False,
            liger_flce=False, gradient_checkpointing=False, compile_mode=None,
            device=dev, world_size=1, local_rank=0)
        bare, model = peft.bare_model, peft.train_model
        model.train()
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=PackingCollator(seq_length=a.max_seq_length,
                                                   mask_dtype=torch.bfloat16))
        it = iter(dl)

        def nextb():
            return {k: v.to(dev) for k, v in next(it).items() if torch.is_tensor(v)}

        # Production flags, not build_optimizer defaults: gram_ns (NOT the eigh
        # fallback), polar_express, k=1, nesterov -- what scripts/sweep/ passes.
        opt = CurvatureWhitenLoRA(
            bare, lr=a.lr, betas=(0.9, 0.999), delta=1e-4, curvature_beta=cb,
            ns_steps=8, polar_method="polar_express", cw_picard_iters=1,
            cw_nesterov=True, precond_refresh_every=10, precond_method="gram_ns",
            higham_iters=8, cw_metric_init="1e-12",
            **dict(REGISTRY["kl-diag-polar-lora"].fixed))
        npairs = min(a.max_pairs, len(opt.pairs))

        def grads():
            opt.zero_grad(set_to_none=False)
            for _ in range(a.grad_accum_steps):
                out = model(**nextb())
                (out.loss / a.grad_accum_steps).backward()
            return out.loss.item()

        for _ in range(a.warm_steps):        # warm Q/P to a realistic metric
            grads(); opt.step()

        cW, cA_, cB_, ratio = [], [], [], []
        for _rep in range(a.reps):
            loss = grads()
            snap = snapshot(opt, npairs)

            # --- stale arm: the production step. Its accumulation produces D^t.
            opt.step()
            stale = [(opt.pairs[i][0].detach().clone(), opt.pairs[i][1].detach().clone())
                     for i in range(npairs)]
            fresh_metric = [(opt.pair_state[i]['Q'].clone(),
                             opt.pair_state[i]['P'].clone())
                            for i in range(npairs)]

            # --- CONTROL: restore with the snapshot's OWN metric and re-step. This
            # must reproduce the stale update bit-for-bit. Without it, a restore bug
            # that silently left the metric (or the momentum) untouched would make
            # both arms identical and print cos=1.0, which reads exactly like
            # "staleness is harmless" -- the wrong answer, arrived at confidently.
            if _rep == 0:
                restore(opt, snap, npairs)
                opt.step()
                ctl = [(opt.pairs[i][0].detach().clone(), opt.pairs[i][1].detach().clone())
                       for i in range(npairs)]
                worst = max(float((ctl[i][0] - stale[i][0]).abs().max())
                            for i in range(npairs))
                if worst != 0.0:
                    raise SystemExit(
                        f"CONTROL FAILED: restore+re-step did not reproduce the stale "
                        f"update (max |dA| difference {worst:.3e}, expected exactly 0). "
                        f"The snapshot is missing state, so the fresh-vs-stale contrast "
                        f"below would be measuring that omission instead of the metric.")
                print(f"  control ok (restore+re-step reproduces the stale step exactly)",
                      flush=True)

            # --- fresh arm: same g, same m, same W, metric that HAS seen g.
            restore(opt, snap, npairs, keep_metric=fresh_metric)
            opt.step()
            fresh = [(opt.pairs[i][0].detach().clone(), opt.pairs[i][1].detach().clone())
                     for i in range(npairs)]

            for i in range(npairs):
                pA, pB = snap[i]['A'], snap[i]['B']
                dW_s = merged_delta(pA, pB, *stale[i])
                dW_f = merged_delta(pA, pB, *fresh[i])
                cW.append(cos_flat(dW_s, dW_f))
                cA_.append(cos_flat(stale[i][0].float() - pA.float(),
                                    fresh[i][0].float() - pA.float()))
                cB_.append(cos_flat(stale[i][1].float() - pB.float(),
                                    fresh[i][1].float() - pB.float()))
                ns, nf = dW_s.norm(), dW_f.norm()
                if ns > 0:
                    ratio.append(float(nf / ns))

            # leave the optimizer on the STALE trajectory (the production one)
            restore(opt, snap, npairs)
            opt.step()

        def mean(v):
            return sum(v) / len(v) if v else float("nan")
        m = dict(cos_dW=mean(cW), cos_dA=mean(cA_), cos_dB=mean(cB_), norm_ratio=mean(ratio))
        m["ang_dW_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, m["cos_dW"]))))
        results[cb] = m
        print(f"  beta2={cb:<7g} excluded-mass={1-cb:5.1%}  cos(dW)={m['cos_dW']:.4f}  "
              f"ang={m['ang_dW_deg']:5.2f} deg  |dW|f/s={m['norm_ratio']:.4f}  "
              f"cos(dA)={m['cos_dA']:.4f}  cos(dB)={m['cos_dB']:.4f}", flush=True)
        del opt, model, bare, peft
        torch.cuda.empty_cache()

    print(f"\n=== stale vs fresh metric, update direction: {a.tag} ===")
    print(f"{a.model_name}  r={a.lora_r}  {tokens_per_step} tokens/step  "
          f"{a.max_pairs} pairs  {a.reps} probe steps per point")
    print(f"{'beta2':>8} {'excluded':>9} {'cos(dW)':>9} {'angle_deg':>10} {'|dW| f/s':>9}")
    for cb in grid:
        m = results[cb]
        print(f"{cb:8g} {1-cb:8.1%} {m['cos_dW']:9.4f} {m['ang_dW_deg']:10.2f} "
              f"{m['norm_ratio']:9.4f}")
    print()
    print("cos(dW) ~ 1 at every beta2  => staleness is a scale effect only; the rho rule")
    print("   and msign's scale-invariance already absorb it.")
    print("cos(dW) falling as beta2 falls => an anisotropic DIRECTION channel that neither")
    print("   the magnitude rule nor msign closes; the fresh-metric variant is warranted.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(tag=a.tag, model=a.model_name, data_dir=a.data_dir,
                                lora_r=a.lora_r, lr=a.lr, pairs=a.max_pairs,
                                reps=a.reps, tokens_per_step=tokens_per_step,
                                results={str(k): v for k, v in results.items()})) + "\n")
    print("appended to", a.out)


if __name__ == "__main__":
    main()
