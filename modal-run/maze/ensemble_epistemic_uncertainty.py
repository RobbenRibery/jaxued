"""Launch eight-agent epistemic-uncertainty Maze PLR on Modal.

Run with: ``modal run modal-run/maze/ensemble_epistemic_uncertainty.py``
"""

from __future__ import annotations

import modal

from _common import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_VOLUME_NAME,
    DEFAULT_GPU,
    DEFAULT_VIRTUAL_LEVEL_BATCH_SIZE,
    EnsembleEpistemicRun,
    build_ensemble_epistemic_command,
    checkpoint_volume,
    image,
    run_training,
    wandb_secret,
)


app = modal.App("jaxued-maze-ensemble-epistemic", image=image)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[wandb_secret],
    volumes={str(CHECKPOINT_DIRECTORY): checkpoint_volume},
)
def run_ensemble_epistemic(
    run_name: str = "maze_ensemble_epistemic_8_agents",
    project: str = "JAXUED_TEST",
    seed: int = 0,
    num_updates: int = 30_000,
    num_agents: int = 8,
    virtual_rollout_phases: int = 3,
    virtual_epoch_ppo: int = 5,
    virtual_level_batch_size: int = DEFAULT_VIRTUAL_LEVEL_BATCH_SIZE,
    checkpoint_save_interval: int = 2_500,
) -> str:
    run = EnsembleEpistemicRun(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        num_agents=num_agents,
        virtual_rollout_phases=virtual_rollout_phases,
        virtual_epoch_ppo=virtual_epoch_ppo,
        virtual_level_batch_size=virtual_level_batch_size,
        checkpoint_save_interval=checkpoint_save_interval,
    )
    return run_training(build_ensemble_epistemic_command(run), run)


@app.local_entrypoint()
def main(
    run_name: str = "maze_ensemble_epistemic_8_agents",
    project: str = "JAXUED_TEST",
    seed: int = 0,
    num_updates: int = 30_000,
    num_agents: int = 8,
    virtual_rollout_phases: int = 3,
    virtual_epoch_ppo: int = 5,
    virtual_level_batch_size: int = DEFAULT_VIRTUAL_LEVEL_BATCH_SIZE,
    checkpoint_save_interval: int = 2_500,
) -> None:
    checkpoint_path = run_ensemble_epistemic.remote(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        num_agents=num_agents,
        virtual_rollout_phases=virtual_rollout_phases,
        virtual_epoch_ppo=virtual_epoch_ppo,
        virtual_level_batch_size=virtual_level_batch_size,
        checkpoint_save_interval=checkpoint_save_interval,
    )
    print(f"Checkpoints persisted at {CHECKPOINT_VOLUME_NAME}:{checkpoint_path}")
