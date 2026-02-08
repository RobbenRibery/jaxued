#!/bin/bash
# Train maze PLR with absolute policy gradient scoring function

uv run -m examples.maze_plr \
    --score_function abs_pg
