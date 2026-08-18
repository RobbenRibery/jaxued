"""Launch Maze PLR with exploratory gradient updates on Modal.

Run with: ``modal run modal-run/maze/plr_plus.py``
"""

from __future__ import annotations

import modal

from _common import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_VOLUME_NAME,
    DEFAULT_GPU,
    MazeRun,
    build_plr_plus_command,
    checkpoint_volume,
    image,
    run_training,
    wandb_secret,
)


app = modal.App("jaxued-maze-plr-plus", image=image)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[wandb_secret],
    volumes={str(CHECKPOINT_DIRECTORY): checkpoint_volume},
)
def run_plr_plus(
    run_name: str = "maze_plr_plus",
    project: str = "JAXUED_TEST",
    seed: int = 0,
    num_updates: int = 30_000,
    checkpoint_save_interval: int = 10,
) -> str:
    run = MazeRun(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        checkpoint_save_interval=checkpoint_save_interval,
    )
    return run_training(build_plr_plus_command(run), run)


@app.local_entrypoint()
def main(
    run_name: str = "maze_plr_plus",
    project: str = "JAXUED_TEST",
    seed: int = 0,
    num_updates: int = 30_000,
    checkpoint_save_interval: int = 10,
) -> None:
    checkpoint_path = run_plr_plus.remote(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        checkpoint_save_interval=checkpoint_save_interval,
    )
    print(f"Checkpoints persisted at {CHECKPOINT_VOLUME_NAME}:{checkpoint_path}")
