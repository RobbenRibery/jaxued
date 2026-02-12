#!/bin/bash
# Train maze PLR with Positive Value Loss (pvl) scoring function
#
# Usage:
#   SEED=3 N_WALLS=60 ./scripts/train_robust_plr_pvl.sh [seed] [n_walls] [extra args...]
# Positional integers (if provided) take precedence over env vars.
SEED="${SEED:-0}"
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    SEED="$1"
    shift
fi
N_WALLS="${N_WALLS:-60}"
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N_WALLS="$1"
    shift
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a non-negative integer, got '${SEED}'." >&2
    exit 1
fi
if ! [[ "${N_WALLS}" =~ ^[0-9]+$ ]]; then
    echo "Error: n_walls must be a non-negative integer, got '${N_WALLS}'." >&2
    exit 1
fi
PROJECT_NAME="ued"
SCORE_FUNCTION="pvl"
METHOD_NAME="robust_plr"
WANDB_EXPERIMENT_NAME="${METHOD_NAME}-pvl-seed${SEED}-walls${N_WALLS}"

uv run -m examples.maze_plr \
    --score_function pvl \
    --no-exploratory_grad_updates \
    --level_buffer_capacity 4000 \
    --replay_prob 0.8 \
    --staleness_coeff 0.3 \
    --temperature 0.3 \
    --topk_k 4 \
    --minimum_fill_ratio 0.5 \
    --prioritization rank \
    --buffer_duplicate_check \
    --no-use_accel \
    --num_edits 5 \
    --project "${PROJECT_NAME}" \
    --wandb_experiment_name "${WANDB_EXPERIMENT_NAME}" \
    --seed "$SEED" \
    --mode train \
    --checkpoint_save_interval 2 \
    --max_number_of_checkpoints 60 \
    --eval_freq 250 \
    --eval_num_attempts 10 \
    --eval_levels "SixteenRooms" "SixteenRooms2" "Labyrinth" "LabyrinthFlipped" "Labyrinth2" "StandardMaze" "StandardMaze2" "StandardMaze3" \
    --lr 1e-4 \
    --max_grad_norm 0.5 \
    --num_updates 30000 \
    --num_steps 256 \
    --num_train_envs 32 \
    --num_minibatches 1 \
    --gamma 0.995 \
    --epoch_ppo 5 \
    --clip_eps 0.2 \
    --gae_lambda 0.98 \
    --entropy_coeff 1e-3 \
    --critic_coeff 0.5 \
    --agent_view_size 5 \
    --n_walls "$N_WALLS" \
    "$@"
