#!/bin/bash
# Packing-curve microbench: measure per-step cost as a function of how many
# independent single-GPU training jobs run concurrently on ONE node.
#
# Motivation: a disBatch sweep packed 8 tasks on one Blackwell node and 2 on
# another; the 8-packed tasks ran ~2.5x slower per step than the 2-packed ones,
# even though every task had its own GPU. That is host-side contention (shared
# power/cooling envelope + RAM/PCIe bandwidth), not GPU oversubscription. This
# script quantifies penalty(N) = s/step at N co-located jobs / s/step at N=1,
# and samples GPU clocks/power/temp so the mechanism (throttling vs bandwidth)
# can be read off rather than guessed.
#
# Each co-located job is one `bench_optimizer_step.py` process pinned to a
# distinct GPU via CUDA_VISIBLE_DEVICES. bench reports mean_sec_per_step over
# full fwd+bwd+optimizer.step cycles (no eval/data confound), so the node load
# is a realistic training step and the per-step number is directly comparable.
#
# MUST run on an EXCLUSIVE node (otherwise other jobs contaminate the curve).
# The conda env (ffcv-pl) must already be active. Outputs under $OUT_DIR.
#
# Usage:
#   scripts/bench/packing_curve.sh            # full curve
#   SMOKE=1 scripts/bench/packing_curve.sh    # 2-rep / N=2 smoke gate
set -eo pipefail

REPO="${REPO:-/mnt/home/nghosh/lora}"
OUT_DIR="${OUT_DIR:-$REPO/logs/bench/packing}"
LORA_R="${LORA_R:-256}"
NWARMUP="${NWARMUP:-3}"
NCYCLES="${NCYCLES:-8}"          # bench reps per cell (GRAM opts: x K). >=4 for stable median.
TELE_EVERY="${TELE_EVERY:-2}"    # seconds between nvidia-smi samples
BENCH="$REPO/scripts/bench/bench_optimizer_step.py"
mkdir -p "$OUT_DIR"

# Polar (expensive) gets the full curve; adamw + no-polar get endpoints only to
# confirm the penalty is optimizer-independent (i.e. host-side, not algorithmic).
POLAR_OPT="curvature-whiten-polar-lora"
FULL_NS="${FULL_NS:-1 2 4 8}"
ENDPOINT_NS="${ENDPOINT_NS:-1 8}"
ENDPOINT_OPTS="${ENDPOINT_OPTS:-adamw curvature-whiten-lora}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    NCYCLES=2; FULL_NS="2"; ENDPOINT_NS=""; ENDPOINT_OPTS=""
    echo "# SMOKE mode: NCYCLES=2, N=2 only, polar only"
fi

# Sample all GPUs' clocks/power/temp to a CSV until killed.
_start_telemetry() {
    local csv="$1"
    echo "timestamp,index,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c,util_pct" > "$csv"
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
                --format=csv,noheader,nounits >> "$csv" 2>/dev/null || true
            sleep "$TELE_EVERY"
        done
    ) &
    echo $!
}

# packing_round N opt  — launch N pinned bench processes concurrently + telemetry.
packing_round() {
    local N="$1" opt="$2"
    local tag="N${N}_${opt}"
    echo "### packing_round N=$N opt=$opt  ($(date '+%T'))"
    local tele_csv="$OUT_DIR/telemetry_${tag}.csv"
    local tele_pid; tele_pid=$(_start_telemetry "$tele_csv")
    local pids=() g
    for ((g=0; g<N; g++)); do
        CUDA_VISIBLE_DEVICES="$g" python "$BENCH" \
            --lora_r "$LORA_R" --bf16 --compile --data_pipeline_version packed_v1 \
            --batch_size 4 --seq_len 2048 --grad_accum_steps 4 \
            --precond_refresh_every 10 \
            --optimizers "$opt" \
            --n_warmup "$NWARMUP" --n_cycles "$NCYCLES" \
            --out "$OUT_DIR/pack_${tag}_gpu${g}.jsonl" \
            > "$OUT_DIR/pack_${tag}_gpu${g}.log" 2>&1 &
        pids+=($!)
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    kill "$tele_pid" 2>/dev/null || true
    if [[ $rc -ne 0 ]]; then
        echo "!!! packing_round N=$N opt=$opt had a FAILED process; check $OUT_DIR/pack_${tag}_gpu*.log" >&2
    fi
    return $rc
}

# ── full curve for the expensive polar optimizer ──
for N in $FULL_NS; do
    packing_round "$N" "$POLAR_OPT"
done

# ── endpoint confirmation rows (optimizer-independence) ──
for opt in $ENDPOINT_OPTS; do
    for N in $ENDPOINT_NS; do
        packing_round "$N" "$opt"
    done
done

# ── optional NUMA probe: N=2 same-socket (0,1) vs cross-socket (0,4) ──
if [[ "${NUMA_PROBE:-0}" == "1" && "${SMOKE:-0}" != "1" ]]; then
    echo "### NUMA probe: 2 jobs cross-socket (GPUs 0,4)"
    tele_csv="$OUT_DIR/telemetry_N2cross_${POLAR_OPT}.csv"
    tele_pid=$(_start_telemetry "$tele_csv")
    pids=()
    for g in 0 4; do
        CUDA_VISIBLE_DEVICES="$g" python "$BENCH" \
            --lora_r "$LORA_R" --bf16 --compile --data_pipeline_version packed_v1 \
            --batch_size 4 --seq_len 2048 --grad_accum_steps 4 \
            --precond_refresh_every 10 --optimizers "$POLAR_OPT" \
            --n_warmup "$NWARMUP" --n_cycles "$NCYCLES" \
            --out "$OUT_DIR/pack_N2cross_gpu${g}.jsonl" \
            > "$OUT_DIR/pack_N2cross_gpu${g}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    kill "$tele_pid" 2>/dev/null || true
fi

echo "# packing_curve done ($(date '+%T')). Outputs in $OUT_DIR"
