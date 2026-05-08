#!/usr/bin/env bash
# A0.1 — profile per-step wall-time and peak GPU memory at one base
# tier in the campaign ladder. Run three of these in parallel (one per
# tier) to get full coverage in ~one-tier wall-time.
#
# Used (a) as the BASELINE before flash-attn-2 + compile interventions
# and (b) AFTER, to measure the speedup. Both runs go to the same JSONL
# (the output records `attn_implementation` and `compile_mode` so we
# can split rows in analysis).
#
# Usage:
#   BASE_TIER=1B COND=baseline bash scripts/bench/profile_walltime_sweep.sh
#   BASE_TIER=3B COND=baseline bash scripts/bench/profile_walltime_sweep.sh
#   BASE_TIER=8B COND=baseline bash scripts/bench/profile_walltime_sweep.sh
#
# After-condition example:
#   BASE_TIER=8B COND=flash_compile ATTN=flash_attention_2 COMPILE=default \
#       bash scripts/bench/profile_walltime_sweep.sh
#
# Run on an isolated A100-80G (not A6000 — see CLAUDE.md timing rule).

set -euo pipefail

REPO=/mnt/home/nghosh/lora
cd "$REPO"

BASE_TIER="${BASE_TIER:?must set BASE_TIER to one of: 1B 3B 8B}"
COND="${COND:-baseline}"
ATTN="${ATTN:-}"          # one of "" (HF default), sdpa, flash_attention_2
COMPILE="${COMPILE:-}"    # one of "" (eager), default, reduce-overhead

OUT="logs/bench_profile_walltime/sweep_${COND}_${BASE_TIER}.jsonl"
mkdir -p "$(dirname "$OUT")"

# Common bench knobs.
N_WARMUP=3
N_CYCLES=4
# Tight-chord = the chord-radius + Picard cross-coupling variant of the
# spectral-chord rule, derived in docs/notes/polar_product/algorithm_tight_chord.md.
# The bare `adam-polar-product-lora` is a simpler per-factor warmup variant
# (Muon-style, no cross-coupling, no chord trust region) — NOT what the
# campaign profiles.
OPTIMIZERS=(adamw adam-polar-product-lora-coupled-spectral-chord-tight)
PRECOND_METHODS=(eigh higham)   # measure both: eigh = current production default,
                                # higham = batched fast path validated on the loose
                                # variant (profile_a100_canonical_2026_05_04.md).
K=1

run_cell () {
  local model="$1" r="$2" alpha="$3" seq="$4" bs="$5" accum="$6"
  echo "=== ($model, r=$r, alpha=$alpha, seq=$seq, batch=$bs×accum=$accum) cond=$COND ==="

  local extra_args=()
  [[ -n "$ATTN" ]] && extra_args+=( --attn_implementation "$ATTN" )
  [[ -n "$COMPILE" ]] && extra_args+=( --compile "$COMPILE" )

  python scripts/bench/bench_optimizer_step.py \
    --model_name "$model" \
    --lora_r "$r" --lora_alpha "$alpha" \
    --target_modules all-linear \
    --bf16 \
    --batch_size "$bs" --seq_len "$seq" --grad_accum_steps "$accum" \
    --n_warmup "$N_WARMUP" --n_cycles "$N_CYCLES" \
    --optimizers "${OPTIMIZERS[@]}" \
    --precond_refresh_every "$K" \
    --precond_method "${PRECOND_METHODS[@]}" \
    --out "$OUT" \
    "${extra_args[@]}"
}

case "$BASE_TIER" in
  1B)
    # OLMo-2-0425-1B at seq=512, batch_eff=16
    run_cell allenai/OLMo-2-0425-1B 128 128 512 2 8
    run_cell allenai/OLMo-2-0425-1B 512 512 512 2 8
    ;;
  3B)
    # Llama-3.2-3B at seq=1024, batch_eff=16
    run_cell meta-llama/Llama-3.2-3B 128 128 1024 2 8
    run_cell meta-llama/Llama-3.2-3B 256 256 1024 2 8
    ;;
  8B)
    # Meta-Llama-3-8B at seq=2048, batch_eff=16 (micro=1 × accum=16 to fit).
    # Note: campaign Phase C plan calls for Llama-3.1-8B; the cached Llama-3-8B
    # is architecturally identical for timing purposes (same d_model, n_layers,
    # head config). Switch to 3.1 variant for the actual training runs in Phase C.
    run_cell meta-llama/Meta-Llama-3-8B 64 64 2048 1 16
    run_cell meta-llama/Meta-Llama-3-8B 256 256 2048 1 16
    ;;
  *)
    echo "Unknown BASE_TIER: $BASE_TIER (expected: 1B, 3B, 8B)" >&2
    exit 1
    ;;
esac

echo
echo "Done [$BASE_TIER, $COND]. Output: $OUT"
echo "Inspect with:  jq -c '.' $OUT"
