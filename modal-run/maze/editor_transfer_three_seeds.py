"""Launch editor-transfer Robust PLR for three seeds on Modal.

The default interface matches the other Robust PLR metric sweeps: ACCEL and
exploratory gradient updates are disabled, and the GPU defaults to an NVIDIA
RTX PRO 6000. Run it with::

    modal run modal-run/maze/editor_transfer_three_seeds.py

Pass ``--exploratory-grad-updates`` only to select the PLR/PLR+ variant, which
persists the source-level PPO update instead of discarding it after scoring.
"""

from __future__ import annotations

import modal

from _common import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_VOLUME_NAME,
    DEFAULT_GPU,
    DEFAULT_TRANSFER_NUM_EDITS,
    DEFAULT_TRANSFER_TARGET_COUNT,
    EditorTransferRun,
    build_editor_transfer_command,
    checkpoint_volume,
    image,
    parse_three_seeds,
    run_training,
    wandb_secret,
)


DEFAULT_SEEDS = (0, 1, 2)
SeedRequest = tuple[str, str, int, int, int, int, int, bool]

app = modal.App("jaxued-maze-editor-transfer-three-seeds", image=image)


def parse_seeds(raw_seeds: str) -> tuple[int, int, int]:
    """Parse exactly three distinct, non-negative comma-separated seeds."""
    return parse_three_seeds(raw_seeds)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[wandb_secret],
    volumes={str(CHECKPOINT_DIRECTORY): checkpoint_volume},
)
def run_editor_transfer_seed(request: SeedRequest) -> tuple[int, str]:
    """Run one editor-transfer seed and return its checkpoint location."""
    (
        run_name,
        project,
        seed,
        num_updates,
        checkpoint_save_interval,
        transfer_target_count,
        transfer_num_edits,
        exploratory_grad_updates,
    ) = request
    run = EditorTransferRun(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        checkpoint_save_interval=checkpoint_save_interval,
        transfer_target_count=transfer_target_count,
        transfer_num_edits=transfer_num_edits,
        exploratory_grad_updates=exploratory_grad_updates,
    )
    checkpoint_path = run_training(build_editor_transfer_command(run), run)
    return seed, checkpoint_path


@app.local_entrypoint()
def main(
    run_name: str = "maze_editor_transfer_three_seeds",
    project: str = "JAXUED_TEST",
    seeds: str = "0,1,2",
    num_updates: int = 30_000,
    checkpoint_save_interval: int = 10,
    transfer_target_count: int = DEFAULT_TRANSFER_TARGET_COUNT,
    transfer_num_edits: int = DEFAULT_TRANSFER_NUM_EDITS,
    exploratory_grad_updates: bool = False,
) -> None:
    """Run three seeds concurrently and wait for every result."""
    seed_values = parse_seeds(seeds)
    requests = [
        (
            run_name,
            project,
            seed,
            num_updates,
            checkpoint_save_interval,
            transfer_target_count,
            transfer_num_edits,
            exploratory_grad_updates,
        )
        for seed in seed_values
    ]
    results = list(run_editor_transfer_seed.map(requests, return_exceptions=True))

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
            f"Editor-transfer PLR failed for seed(s): {failed_seeds}"
        )
