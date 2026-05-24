"""Canned ablation figures used directly from notebook cells.

Each helper returns `(fig, *dataframes)` so the calling cell stays at ~5
lines: invoke the helper, `display(df)` the summaries, `plt.show()`. No
inline `load_runs(where=...)` predicate dicts, no inline aggregation loops,
no inline color/linestyle dicts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .colors import NS_AXIS_COLORS, PICARD_LINESTYLES, ssc_overlay_palette


# Canonical chord-tight-clean optimizer name. Promoted out of every cell that
# previously defined CLEAN inline.
CHORD_TIGHT_CLEAN = "adam-polar-product-lora-coupled-spectral-chord-tight-clean"


def _default_eps_predicate(v) -> bool:
    """Match runs whose precond_delta is the default 1e-6 (absolute, not relative)."""
    return abs(float(v or 1e-6) - 1e-6) < 1e-9


def ns_iters_ssc_overlay_figure(
    lora_r: int,
    *,
    logs_root: str = "../logs",
    optimizer: str = CHORD_TIGHT_CLEAN,
    picards: tuple[int, ...] = (1, 3),
    ns_steps: tuple[int, ...] = (2, 5, 10),
    ssc_picard: int = 3,
    ssc_nsteps: int = 10,
    data_pipeline_version: str = "packed_v1",
    max_steps: int = 4000,
    seed: int = 0,
    figsize: tuple[float, float] = (13, 5),
):
    """NS-iters × picard-iters ablation with optional SSC overlay.

    Loads two run-sets at the given rank: (a) NS variants spanning
    `(picard ∈ picards, muon_ns_steps ∈ ns_steps)`, and (b) SSC variants
    at `picard=ssc_picard, ssc_nsteps=ssc_nsteps` over every `ssc_c` that
    landed under `logs_root`. SSC overlay is drawn when present; the figure
    degrades gracefully to NS-only when no SSC runs exist for the rank.

    Color = ns_steps (`NS_AXIS_COLORS`), linestyle = picard
    (`PICARD_LINESTYLES`). SSC overlays use `ssc_overlay_palette(n)` —
    library-validated to be distinct from NS_AXIS_COLORS.

    Returns
    -------
    (fig, coverage_df, best_df) — `coverage_df` has columns
        ('variant', 'picard', 'ns', 'ssc_c', 'lr_count', 'lrs', 'pinned');
        `best_df` has ('variant', 'picard', 'ns', 'ssc_c', 'best_lr',
        'final', 'delta_vs_ns_pic3_ns10').
    """
    import pandas as pd
    from lora_playground.loader import load_runs

    common = dict(
        optimizer=optimizer,
        data_pipeline_version=data_pipeline_version,
        max_steps=max_steps,
        lora_r=lora_r,
        seed=seed,
        precond_delta_relative=False,
        precond_delta=_default_eps_predicate,
    )

    ns_runs = load_runs(
        where={
            **common,
            "effective_picard_iters": lambda p: p in picards,
            "muon_ns_steps": lambda n: n in ns_steps,
            "ns_form": lambda v: v in ("gram", None),
            "htmuon_p": None,
            "polar_method": lambda v: v in (None, "ns"),
        },
        unique_on=("effective_picard_iters", "muon_ns_steps", "lr"),
        allow_axes=("snapshot_dir", "snapshot_steps"),
        logs_root=logs_root,
        warn_cross_commit=False,
    )

    ssc_runs = load_runs(
        where={
            **common,
            "polar_method": "ssc",
            "ssc_nsteps": ssc_nsteps,
            "picard_iters_override": ssc_picard,
        },
        unique_on=("ssc_c", "lr"),
        logs_root=logs_root,
        warn_cross_commit=False,
    )

    # NS aggregation: keep only completed runs; track best per (picard, ns).
    coverage_ns: dict[int, dict[int, list[float]]] = {p: {n: [] for n in ns_steps} for p in picards}
    final_by_cell: dict[tuple, float] = {}
    best_ns: dict[tuple, tuple] = {}
    for cfg, evs in ns_runs:
        if not evs or evs[-1].get("step") != max_steps:
            continue
        pic = cfg["effective_picard_iters"]
        ns = cfg.get("muon_ns_steps") or cfg.get("_derived", {}).get("effective_muon_ns_steps")
        lr = cfg["lr"]
        if pic not in picards or ns not in ns_steps:
            continue
        coverage_ns[pic][ns].append(lr)
        final = evs[-1]["eval_loss"]
        final_by_cell[(pic, ns, lr)] = final
        key = (pic, ns)
        if key not in best_ns or final < best_ns[key][0]:
            best_ns[key] = (final, lr, cfg, evs)

    # SSC aggregation: grouped by c, then by lr; keep best per cell.
    ssc_by_c: dict[float, dict[float, tuple]] = {}
    for cfg, evs in ssc_runs:
        if not evs or evs[-1].get("step") != max_steps:
            continue
        c = cfg["ssc_c"]
        lr = cfg["lr"]
        final = evs[-1]["eval_loss"]
        d = ssc_by_c.setdefault(c, {})
        if lr not in d or final < d[lr][0]:
            d[lr] = (final, cfg, evs)
    ssc_cs = sorted(ssc_by_c)
    ssc_best: dict[float, tuple] = {
        c: min(ssc_by_c[c].items(), key=lambda kv: kv[1][0])
        for c in ssc_cs
    }  # c -> (lr, (final, cfg, evs))

    # ── Summary DataFrames ──
    cov_rows = []
    for pic in picards:
        for ns in ns_steps:
            lrs = sorted(coverage_ns[pic][ns])
            cov_rows.append({
                "variant": "NS", "picard": pic, "ns": ns, "ssc_c": None,
                "lr_count": len(lrs), "lrs": lrs, "pinned": len(lrs) < 3,
            })
    for c in ssc_cs:
        lrs = sorted(ssc_by_c[c])
        cov_rows.append({
            "variant": "SSC", "picard": ssc_picard, "ns": ssc_nsteps, "ssc_c": c,
            "lr_count": len(lrs), "lrs": lrs, "pinned": len(lrs) < 3,
        })
    coverage_df = pd.DataFrame(cov_rows)

    ref_final = best_ns.get((3, 10), (None,))[0]
    best_rows = []
    for pic in picards:
        for ns in ns_steps:
            if (pic, ns) in best_ns:
                final, lr, _cfg, _evs = best_ns[(pic, ns)]
                best_rows.append({
                    "variant": "NS", "picard": pic, "ns": ns, "ssc_c": None,
                    "best_lr": lr, "final": final,
                    "delta_vs_ns_pic3_ns10": (final - ref_final) if ref_final is not None else None,
                })
    for c in ssc_cs:
        lr, (final, _cfg, _evs) = ssc_best[c]
        best_rows.append({
            "variant": "SSC", "picard": ssc_picard, "ns": ssc_nsteps, "ssc_c": c,
            "best_lr": lr, "final": final,
            "delta_vs_ns_pic3_ns10": (final - ref_final) if ref_final is not None else None,
        })
    best_df = pd.DataFrame(best_rows)

    # ── Figure ──
    fig, (ax_lr, ax_traj) = plt.subplots(1, 2, figsize=figsize)
    for pic in picards:
        for ns in ns_steps:
            pts = [(lr, final_by_cell[(pic, ns, lr)])
                   for lr in sorted(coverage_ns[pic][ns])
                   if (pic, ns, lr) in final_by_cell]
            if pts:
                xs, ys = zip(*pts)
                ax_lr.plot(xs, ys, marker="o", ms=6, lw=1.6,
                           color=NS_AXIS_COLORS[ns], linestyle=PICARD_LINESTYLES[pic],
                           label=f"picard={pic}, NS={ns}")
            if (pic, ns) in best_ns:
                final, lr, _cfg, evs = best_ns[(pic, ns)]
                ax_traj.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                             marker="o", ms=3, lw=1.6,
                             color=NS_AXIS_COLORS[ns], linestyle=PICARD_LINESTYLES[pic],
                             label=f"picard={pic}, NS={ns}  (lr={lr:g}, final={final:.4f})")

    if ssc_cs:
        shades = ssc_overlay_palette(len(ssc_cs))
        for c, color in zip(ssc_cs, shades):
            d = ssc_by_c[c]
            xs = sorted(d)
            ys = [d[lr][0] for lr in xs]
            ax_lr.plot(xs, ys, marker="s", ms=6, lw=1.6,
                       color=color, linestyle="--",
                       label=f"SSC c={c} ns={ssc_nsteps} (pic={ssc_picard})")
            lr_b, (final_b, _cfg, evs_b) = ssc_best[c]
            ax_traj.plot([e["step"] for e in evs_b], [e["eval_loss"] for e in evs_b],
                         marker="s", ms=3, lw=1.6, color=color, linestyle="--",
                         label=f"SSC c={c} ns={ssc_nsteps} (pic={ssc_picard})"
                               f"  (lr={lr_b:g}, final={final_b:.4f})")

    ax_lr.set_xscale("log")
    ax_lr.set_xlabel("learning rate")
    ax_lr.set_ylabel(f"final eval_loss @ {max_steps // 1000}k")
    ax_lr.set_title("final loss vs lr")
    ax_lr.grid(True, alpha=0.3)
    ax_lr.legend(loc="best", fontsize=8, handlelength=3.5)

    ax_traj.set_xlabel("step")
    ax_traj.set_ylabel("eval_loss")
    ax_traj.set_title("best-lr trajectory")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper right", fontsize=8, handlelength=3.5)

    _ls_name = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}
    picard_legend = ", ".join(
        f"picard={p} ({_ls_name.get(PICARD_LINESTYLES[p], PICARD_LINESTYLES[p])})"
        for p in picards if p in PICARD_LINESTYLES
    )
    ssc_tag = " + SSC overlay" if ssc_cs else ""
    fig.suptitle(
        f"NS-iters ablation{ssc_tag}: {picard_legend} "
        f"at r={lora_r} ({data_pipeline_version} {max_steps // 1000}k, default-delta, seed={seed})"
    )
    plt.tight_layout()
    return fig, coverage_df, best_df


def compare_variants_figure(
    variants: dict,
    *,
    common_where: dict,
    ref_label: str,
    logs_root: str = "../logs",
    sigma_ref: float = 0.0007,
    suptitle: str | None = None,
    colors: dict | None = None,
    markers: dict | None = None,
    figsize: tuple[float, float] = (13, 5),
    max_steps: int = 4000,
):
    """Compare named optimizer variants at a fixed config; 2-panel + tables.

    `variants` maps `label -> extra_where_dict`. Each variant is loaded as
    `load_runs(where={**common_where, **extra}, ...)`. The figure shows
    final-loss vs lr (left) and best-lr trajectory (right). Δ vs `ref_label`
    is reported in σ-units (`sigma_ref`).

    Returns
    -------
    (fig, table_df, summary_df) — `table_df` has lr as index and one column
    per variant; `summary_df` has ('variant', 'best_lr', 'final', 'delta',
    'delta_sigma').
    """
    import pandas as pd
    from lora_playground.loader import load_runs

    per_variant = {}
    for label, extra in variants.items():
        runs = load_runs(
            where={**common_where, **extra},
            logs_root=logs_root,
            warn_cross_commit=False,
        )
        d = {}
        for c, h in runs:
            if not h or h[-1].get("step") != max_steps:
                continue
            lr = float(c["lr"])
            f = h[-1]["eval_loss"]
            if lr not in d or f < d[lr][0]:
                d[lr] = (f, c, h)
        per_variant[label] = d

    # Final-loss table (rows = lr, columns = variant).
    all_lr = sorted({lr for v in per_variant.values() for lr in v})
    table_df = pd.DataFrame(
        {label: [per_variant[label].get(lr, (None,))[0] for lr in all_lr]
         for label in variants},
        index=pd.Index(all_lr, name="lr"),
    )

    # Best per variant + Δ vs ref.
    if ref_label not in per_variant or not per_variant[ref_label]:
        ref_best = None
    else:
        ref_best = min(per_variant[ref_label].values(), key=lambda v: v[0])[0]
    summary_rows = []
    for label, d in per_variant.items():
        if not d:
            continue
        best_lr = min(d, key=lambda lr: d[lr][0])
        final = d[best_lr][0]
        delta = (final - ref_best) if ref_best is not None and label != ref_label else None
        summary_rows.append({
            "variant": label, "best_lr": best_lr, "final": final,
            "delta": delta,
            "delta_sigma": (delta / sigma_ref) if delta is not None else None,
        })
    summary_df = pd.DataFrame(summary_rows)

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {label: cmap(i % 10) for i, label in enumerate(variants)}
    if markers is None:
        marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
        markers = {label: marker_cycle[i % len(marker_cycle)] for i, label in enumerate(variants)}

    fig, (ax_lr, ax_traj) = plt.subplots(1, 2, figsize=figsize)
    for label, d in per_variant.items():
        if not d:
            continue
        lrs = sorted(d)
        finals = [d[lr][0] for lr in lrs]
        ax_lr.plot(lrs, finals, marker=markers[label], ms=6, lw=1.4,
                   color=colors[label], label=label)
    ax_lr.set_xscale("log")
    ax_lr.set_xlabel("lr")
    ax_lr.set_ylabel(f"final eval_loss @ {max_steps // 1000}k")
    ax_lr.set_title("final loss vs lr")
    ax_lr.grid(True, alpha=0.3)
    ax_lr.legend(fontsize=9)

    for label, d in per_variant.items():
        if not d:
            continue
        best_lr = min(d, key=lambda lr: d[lr][0])
        final, _, h = d[best_lr]
        ax_traj.plot([e["step"] for e in h], [e["eval_loss"] for e in h],
                     marker=markers[label], ms=3, lw=1.4, color=colors[label],
                     label=f"{label}  (lr={best_lr:g}, final={final:.4f})")
    ax_traj.set_xlabel("step")
    ax_traj.set_ylabel("eval_loss")
    ax_traj.set_title("best-lr trajectory")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=9)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig, table_df, summary_df
