# Paper name → source mapping

Reader-facing prose in `main.tex` uses domain names only (paper-writing convention).
This file maps each to its implementation. Released alongside source; not cited in the paper.

| Paper name | Source identifier | Where |
|---|---|---|
| `Polar-LoRA` (protagonist) | `diag-shampoo-polar-lora`, **full polar PE=8** (`--polar_method polar_express --muon_ns_steps 8`), `k=1` (`--cw_picard_iters 1`), Nesterov momentum `β1=0.95` (`--cw_nesterov --beta1 0.95`), `β_c=0.99`, `δ=1e-4`, `--precond_refresh_every 10` | `lora_playground/optim.py` (`OPTIMIZER_CHOICES`); leaderboard label `diag-Shampoo +polar (f=10, β_c=0.99, δ=1e-4)` |
| polar map φ | spectral-cap via Newton–Schulz | `lora_playground/spectral.py` |
| spectral trust-region rescale | ρ = η/(σ_max(A)+σ_max(B)), guarded σ_max | `lora_playground/spectral.py` (`_smax_warm`) |
| full KL-Shampoo+polar (ablation) | KL-Kronecker coupled curvature | label `KL-Shampoo +polar` |
| KL-diag / SOAP-curv (ablation) | curvature-flavor variants | labels `KL-diag +polar`, `SOAP-curv +polar` |
| Picard cross-coupling `k≥2` (ablation) | inner alternating loop | `k=2` labels |
| chord-tight (baseline arm) | exact-root ρ tight-chord | `adam-polar-product-lora-coupled-spectral-chord-tight`; `docs/notes/polar_product/algorithm_tight_chord.md` |
| speedup-vs-AdamW metric | horizon / steps-to-match | `lora_playground/leaderboard.py`, `lora_playground/workloads.py` |
| performance profile (Fig 1) | AlgoPerf-style profile | `lora_playground.leaderboard.performance_profile` |
| AdamW noise floor σ | multi-seed AdamW std | `logs/adamw_multiseed*/` |

Method derivation of record: `docs/notes/polar_product/kl_shampoo_polar_derivation.md`.
