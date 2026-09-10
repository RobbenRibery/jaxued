"""Launch solved-informed log-relative Robust PLR for three seeds on Modal.

This launcher is matched to the existing editor-transfer controls: ACCEL and
exploratory updates are disabled, each source owns 128 fixed targets produced
by 16 edits, and the GPU defaults to an NVIDIA L40S.

Run with::

    modal run modal-run/maze/editor_solved_informed_log_relative_transfer_three_seeds.py
"""

from __future__ import annotations

import modal

from _common import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_VOLUME_NAME,
    DEFAULT_GPU,
    DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    DEFAULT_TRANSFER_NUM_EDITS,
    DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA,
    DEFAULT_TRANSFER_SOLVED_PRIOR_BETA,
    DEFAULT_TRANSFER_TARGET_COUNT,
    EditorSolvedInformedLogRelativeTransferRun,
    build_editor_solved_informed_log_relative_transfer_command,
    checkpoint_volume,
    image,
    parse_three_seeds,
    run_training,
    wandb_secret,
)


DEFAULT_SEEDS = (0, 1, 2)
SeedRequest = tuple[
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    bool,
]

app = modal.App(
    "jaxued-maze-editor-solved-informed-log-relative-transfer-three-seeds",
    image=image,
)


def parse_seeds(raw_seeds: str) -> tuple[int, int, int]:
    """Parse exactly three distinct, non-negative comma-separated seeds."""
    return parse_three_seeds(raw_seeds)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[wandb_secret],
    volumes={str(CHECKPOINT_DIRECTORY): checkpoint_volume},
)
def run_editor_solved_informed_log_relative_transfer_seed(
    request: SeedRequest,
) -> tuple[int, str]:
    """Run one solved-informed transfer seed and return its checkpoint path."""
    (
        run_name,
        project,
        seed,
        num_updates,
        checkpoint_save_interval,
        transfer_target_count,
        transfer_num_edits,
        transfer_log_relative_tau,
        transfer_solved_prior_alpha,
        transfer_solved_prior_beta,
        transfer_solved_confidence,
        exploratory_grad_updates,
    ) = request
    run = EditorSolvedInformedLogRelativeTransferRun(
        run_name=run_name,
        project=project,
        seed=seed,
        num_updates=num_updates,
        checkpoint_save_interval=checkpoint_save_interval,
        transfer_target_count=transfer_target_count,
        transfer_num_edits=transfer_num_edits,
        transfer_log_relative_tau=transfer_log_relative_tau,
        transfer_solved_prior_alpha=transfer_solved_prior_alpha,
        transfer_solved_prior_beta=transfer_solved_prior_beta,
        transfer_solved_confidence=transfer_solved_confidence,
        exploratory_grad_updates=exploratory_grad_updates,
    )
    checkpoint_path = run_training(
        build_editor_solved_informed_log_relative_transfer_command(run),
        run,
    )
    return seed, checkpoint_path


@app.local_entrypoint()
def main(
    run_name: str = "maze_editor_solved_informed_log_relative_transfer_three_seeds",
    project: str = "JAXUED_TEST",
    seeds: str = "0,1,2",
    num_updates: int = 30_000,
    checkpoint_save_interval: int = 10,
    transfer_target_count: int = DEFAULT_TRANSFER_TARGET_COUNT,
    transfer_num_edits: int = DEFAULT_TRANSFER_NUM_EDITS,
    transfer_log_relative_tau: float = DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    transfer_solved_prior_alpha: float = DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA,
    transfer_solved_prior_beta: float = DEFAULT_TRANSFER_SOLVED_PRIOR_BETA,
    transfer_solved_confidence: float = DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    exploratory_grad_updates: bool = False,
) -> None:
    """Run three matched solved-informed transfer seeds concurrently."""
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
            transfer_log_relative_tau,
            transfer_solved_prior_alpha,
            transfer_solved_prior_beta,
            transfer_solved_confidence,
            exploratory_grad_updates,
        )
        for seed in seed_values
    ]
    results = list(
        run_editor_solved_informed_log_relative_transfer_seed.map(
            requests,
            return_exceptions=True,
        )
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
            "Solved-informed log-relative editor-transfer PLR failed for seed(s): "
            f"{failed_seeds}"
        )
