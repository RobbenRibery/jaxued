#!/bin/bash
# Train maze PLR with absolute policy gradient scoring function

uv run -m examples.maze_plr \
    --score_function abs_pg \
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
    --project maze_plr_robust_abs_pg \
    --seed 0 \
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
    --n_walls 25
