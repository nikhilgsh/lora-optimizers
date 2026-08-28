#!/bin/bash
# Matched post-fix r=256 state needed by the factorwise investigation.
# Positional arg 1 selects one of two already-tuned arms; no Cartesian LR grid.
set -eo pipefail

arm=${1:-factorwise_eta1e-2}
case "$arm" in
    factorwise_eta1e-2)
        lr=1e-2
        precond=factorwise
        ;;
    one-sided_eta3e-3)
        lr=3e-3
        precond=one-sided
        ;;
    *)
        echo "sweep_precond_r256_postfix_state: bad arm '$arm'" >&2
        exit 2
        ;;
esac

# Retain the exact late-training state used by the r=16 mechanism probe and
# the final state. submit.sh gives every disBatch task a unique CHECKPOINT_DIR.
export CHECKPOINT_EVERY=7000
export CHECKPOINT_KEEP_LAST=0
export KEEP_CHECKPOINTS=1

exec scripts/sweep/sweep_protagonist_precond.sh \
    "$lr" \
    kl-diag-polar-lora \
    0 \
    1e-4 \
    0.9 \
    meta-llama/Llama-3.2-1B \
    data/openmath_instruct_2_2m_packed_seq2048_llama32 \
    256 \
    "$precond" \
    full
