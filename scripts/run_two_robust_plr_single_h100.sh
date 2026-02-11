#!/bin/bash
# Run two robust-PLR training scripts concurrently on one GPU with per-process JAX memory caps.
#
# Usage:
#   MEM_FRAC1=0.45 MEM_FRAC2=0.45 SEED1=0 SEED2=1 N_WALLS1=60 N_WALLS2=60 \
#   ./scripts/run_two_robust_plr_single_h100.sh [script1] [script2]
#
# Defaults:
#   script1: scripts/train_maxmc_robust_plr.sh
#   script2: scripts/train_pvl_robust_plr.sh
#   GPU_ID: 0
#   MEM_FRAC1: 0.45
#   MEM_FRAC2: 0.45
#   SEED1: 0
#   SEED2: 1
#   N_WALLS1: 60
#   N_WALLS2: 60
#   XLA_PYTHON_CLIENT_PREALLOCATE: true

set -uo pipefail

SCRIPT1="${1:-scripts/train_maxmc_robust_plr.sh}"
SCRIPT2="${2:-scripts/train_pvl_robust_plr.sh}"
GPU_ID="${GPU_ID:-0}"
MEM_FRAC1="${MEM_FRAC1:-0.45}"
MEM_FRAC2="${MEM_FRAC2:-0.45}"
SEED1="${SEED1:-0}"
SEED2="${SEED2:-1}"
N_WALLS1="${N_WALLS1:-60}"
N_WALLS2="${N_WALLS2:-60}"
PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"

is_nonneg_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [ ! -f "$SCRIPT1" ]; then
    echo "Error: script not found: $SCRIPT1" >&2
    exit 1
fi
if [ ! -f "$SCRIPT2" ]; then
    echo "Error: script not found: $SCRIPT2" >&2
    exit 1
fi
if ! is_nonneg_int "$GPU_ID"; then
    echo "Error: GPU_ID must be a non-negative integer, got '$GPU_ID'." >&2
    exit 1
fi
if ! is_nonneg_int "$SEED1"; then
    echo "Error: SEED1 must be a non-negative integer, got '$SEED1'." >&2
    exit 1
fi
if ! is_nonneg_int "$SEED2"; then
    echo "Error: SEED2 must be a non-negative integer, got '$SEED2'." >&2
    exit 1
fi
if ! is_nonneg_int "$N_WALLS1"; then
    echo "Error: N_WALLS1 must be a non-negative integer, got '$N_WALLS1'." >&2
    exit 1
fi
if ! is_nonneg_int "$N_WALLS2"; then
    echo "Error: N_WALLS2 must be a non-negative integer, got '$N_WALLS2'." >&2
    exit 1
fi
if ! awk -v a="$MEM_FRAC1" -v b="$MEM_FRAC2" 'BEGIN { exit !((a + 0) > 0 && (a + 0) <= 1 && (b + 0) > 0 && (b + 0) <= 1 && (a + b) <= 1) }'; then
    echo "Error: MEM_FRAC1 and MEM_FRAC2 must be in (0, 1], and MEM_FRAC1+MEM_FRAC2 must be <= 1." >&2
    exit 1
fi
if [[ "$PREALLOCATE" != "true" && "$PREALLOCATE" != "false" ]]; then
    echo "Error: XLA_PYTHON_CLIENT_PREALLOCATE must be 'true' or 'false', got '$PREALLOCATE'." >&2
    exit 1
fi

mkdir -p logs
NAME1="$(basename "$SCRIPT1" .sh)"
NAME2="$(basename "$SCRIPT2" .sh)"
LOG1="logs/${NAME1}_seed${SEED1}_walls${N_WALLS1}.log"
LOG2="logs/${NAME2}_seed${SEED2}_walls${N_WALLS2}.log"

echo "Launching on GPU $GPU_ID"
echo "job1: $SCRIPT1 seed=$SEED1 n_walls=$N_WALLS1 mem_frac=$MEM_FRAC1 preallocate=$PREALLOCATE"
echo "job2: $SCRIPT2 seed=$SEED2 n_walls=$N_WALLS2 mem_frac=$MEM_FRAC2 preallocate=$PREALLOCATE"
echo "logs: $LOG1"
echo "logs: $LOG2"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC1" \
XLA_PYTHON_CLIENT_PREALLOCATE="$PREALLOCATE" \
SEED="$SEED1" \
N_WALLS="$N_WALLS1" \
bash "$SCRIPT1" >"$LOG1" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC2" \
XLA_PYTHON_CLIENT_PREALLOCATE="$PREALLOCATE" \
SEED="$SEED2" \
N_WALLS="$N_WALLS2" \
bash "$SCRIPT2" >"$LOG2" 2>&1 &
PID2=$!

echo "pids: job1=$PID1 job2=$PID2"

wait "$PID1"
STATUS1=$?
wait "$PID2"
STATUS2=$?

if [ "$STATUS1" -ne 0 ] || [ "$STATUS2" -ne 0 ]; then
    echo "One or both jobs failed (job1=$STATUS1, job2=$STATUS2)." >&2
    exit 1
fi

echo "Both jobs completed successfully."
