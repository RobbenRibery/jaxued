"""Launch mean-positive-delight Maze PLR for three seeds on Modal.

Run with::

    modal run modal-run/maze/mean_positive_delight_three_seeds.py
"""

from __future__ import annotations

import modal

from _common import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_VOLUME_NAME,
    DEFAULT_GPU,
    MazeRun,
    build_mean_positive_delight_plr_command,
    checkpoint_volume,
    image,
    parse_three_seeds,
    run_training,
    wandb_secret,
)


DEFAULT_SEEDS = (0, 1, 2)
SeedRequest = tuple[str, str, int, int, int]

app = modal.App("jaxued-maze-mean-positive-delight-three-seeds", image=image)


def parse_seeds(raw_seeds: str) -> tuple[int, int, int]:
    """Parse exactly three distinct, non-negative comma-separated seeds."""
    return parse_three_seeds(raw_seeds)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[wandb_secret],
    volumes={str(CHECKPOINT_DIRECTORY): checkpoint_volume},
)
def run_mean_positive_delight_seed(request: SeedRequest) -> tuple[int, str]:
    """Run one seed through Robust PLR with mean-positive-delight scoring."""
    run_name, project, seed, num_updates, checkpoint_save_interval = request
    run = MazeRun(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        checkpoint_save_interval=checkpoint_save_interval,
    )
    checkpoint_path = run_training(
        build_mean_positive_delight_plr_command(run),
        run,
    )
    return seed, checkpoint_path


@app.local_entrypoint()
def main(
    run_name: str = "maze_mean_positive_delight_three_seeds",
    project: str = "JAXUED_TEST",
    seeds: str = "0,1,2",
    num_updates: int = 30_000,
    checkpoint_save_interval: int = 10,
) -> None:
    """Run three seeds concurrently and wait for every result."""
    seed_values = parse_seeds(seeds)
    requests = [
        (run_name, project, seed, num_updates, checkpoint_save_interval)
        for seed in seed_values
    ]
    results = list(
        run_mean_positive_delight_seed.map(requests, return_exceptions=True)
    )

    failures = []
    for seed, result in zip(seed_values, results):
        if isinstance(result, BaseException):
            failures.append((seed, result))
            print(f"Seed {seed} failed: {type(result).__name__}: {result}")
        else:
            completed_seed, checkpoint_path = result
            print(
                f"Seed {completed_seed} checkpoints persisted at "
                f"{CHECKPOINT_VOLUME_NAME}:{checkpoint_path}"
            )

    if failures:
        failed_seeds = ", ".join(str(seed) for seed, _ in failures)
        raise RuntimeError(
            f"Mean-positive-delight PLR failed for seed(s): {failed_seeds}"
        )
