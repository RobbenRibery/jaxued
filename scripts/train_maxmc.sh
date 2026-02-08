#!/bin/bash
# Train maze PLR with MaxMC scoring function (default)

uv run -m examples.maze_plr \
    --score_function MaxMC
