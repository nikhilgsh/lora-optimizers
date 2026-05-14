# Phase L re-baseline + LR-extension — submission checklist

Status as of 2026-05-14:

- ✅ Loader fix committed (`bf2a5b8`): Phase L runs now load via `load_runs`.
- ✅ Pack-time zero-supervision filter committed (`69a7646`): `packed_v1.1`.
- 🔄 **Refactor in progress**: packing moved from train-time to prepare_data-time so the Arrow cache holds pre-packed slots. train.py detects pre-packed caches and skips the runtime pack. Existing `data/opc_sft_stage2_all_packed_seq2048/` archived to `_v1.0/`; canonical path being regenerated as packed-slot Arrow.
- ⚠️ **Earlier sbatch submission was cancelled** (jobs 6407007–6407010). The sbatches that ran at submission time used the train-time-packing path; cancelled to avoid wasted GPU + non-deterministic slot ordering. Sbatches now in `slurm_pending/_holding/` pending re-smoke against the new cache.
- ✅ Audit confirmed the previous training-time zero-token-microbatch skip was NOT live at Phase L commit `e05b80e`.
- ✅ Params files written: `params/{adamw,chord_tight}_phase_L_lrsweep_1b_r64_{repack,extension}.json`.

### Remaining prep (Claude is doing this autonomously):
1. Finish cache regeneration (`prepare_data.py --data_pipeline_version packed_v1.1`).
2. Re-run 4 pre-flight smokes against the new cache; confirm `train.py` reads pre-packed slots (no "docs → slots" line in the smoke output).
3. Move sbatches from `slurm_pending/_holding/` back to `slurm_pending/`.

## What to submit, in order

### Step 1 — re-baseline sbatches (6 runs, ~7.5h wall on 6 GPUs total)

```bash
cd /mnt/home/nghosh/lora
submit-pending slurm_pending/adamw_phase_L_lrsweep_r64_repack_blackwell.sbatch \
               slurm_pending/chord_tight_phase_L_lrsweep_r64_repack_blackwell.sbatch
```

This re-runs the original 6 Phase L cells (same η grid) under `packed_v1.1` (the pack-time filter is now active). Watches: completion of both groups, then check leaderboard.

### Step 2 — verify re-baseline against pre-repack Phase L

Once both groups complete (~6–10 hours after submit):

```bash
cd /mnt/home/nghosh/lora
conda run -n ffcv-pl jupyter nbconvert --to notebook --execute notebooks/opc_1b_leaderboard.ipynb --output notebooks/opc_1b_leaderboard.ipynb
```

Then check the per-cell Δ (repack vs original Phase L) for both arms. Target: |Δ| < 1σ_AdamW(4k r=64) = 0.0017 at every cell. If bounded, the original Phase L Δ stood; the extension can target the repack baseline. If any cell exceeds 1σ, retract the original Phase L Δ in `docs/notes/polar_product/tight_chord_paper_plan.md` and re-anchor on the repack numbers.

### Step 3 — extension sbatches (4 runs, ~6h wall on 4 GPUs total)

```bash
cd /mnt/home/nghosh/lora
submit-pending slurm_pending/adamw_phase_L_lrsweep_r64_extension_blackwell.sbatch \
               slurm_pending/chord_tight_phase_L_lrsweep_r64_extension_blackwell.sbatch
```

### Step 4 — check pinning post-extension

```bash
conda run -n ffcv-pl jupyter nbconvert --to notebook --execute notebooks/opc_1b_leaderboard.ipynb --output notebooks/opc_1b_leaderboard.ipynb
```

**Stop rule** per arm: if the new high-η point is *worse* than the previous best, optimum is interior — accept the previous best as the headline. If still pinned high (new η is the best), queue ONE more decade (`5e-3`/`1e-2` for AdamW; `1e0` for chord-tight) as a final 2-cell sbatch. Plan does not authorize unbounded extension.

## What changed in the codebase

- `lora_playground/exclusions/commit_exclusions.json` — dropped `e05b80e` and `91122ce` blanket entries (moved to `_removed` audit log).
- `lora_playground/loader.py` — `_wrapped_filter` records per-reason example pairs; `RunInventory.groups_all_excluded` audit; `load_runs` validates `where`-keys against the candidate cfg pool.
- `lora_playground/data.py::pack_documents` — `drop_zero_supervision_slots: bool = True` filter; per-call print of drop count.
- `lora_playground/train.py` — `data_pipeline_version` default `packed_v1` → `packed_v1.1`; `packed_v1` and `unpacked_v0` retained for backward-compat loading.
- `scripts/sweep/sweep_phase_L_1b_r64.sh` — `--data_pipeline_version` now reads `${DATA_PIPELINE_VERSION:-packed_v1.1}` from env.
- `slurm_scripts/submit.sh` — `repack_baseline`, `lr_extension` added to known-scopes docstring.
- `tests/test_loader.py` — regression tests for blanket-excluded group surfacing + where-key typo warning.
- `tests/test_data_pipeline.py` — regression tests for the zero-supervision filter.

## Out of scope follow-ups (not part of this submission)

- Phase L multi-seed σ anchor — 3 seeds × AdamW at the post-extension best η, ~13 GPU-h. Defer until the extension picks the winning η.
- HumanEval / BigCode pass@1 hook (P2 in `tight_chord_paper_plan.md`).
- Consolidate `plot_utils.load_sweep` and `loader.load_runs` into a single parse path.
