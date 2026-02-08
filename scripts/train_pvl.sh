#!/bin/bash
# Train maze PLR with Positive Value Loss (pvl) scoring function

uv run -m examples.maze_plr \
    --score_function pvl
