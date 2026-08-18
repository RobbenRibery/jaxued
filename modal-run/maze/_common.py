"""Shared, typed configuration and runtime plumbing for Maze Modal runs."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import modal


REMOTE_REPOSITORY = Path("/root/jaxued")
CHECKPOINT_DIRECTORY = REMOTE_REPOSITORY / "checkpoints"
LOCAL_REPOSITORY = Path(__file__).resolve().parents[2]

DEFAULT_GPU = os.environ.get("JAXUED_MODAL_GPU", "L40S")
VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU = {
    # Keep this mapping GPU-specific so measured winners can be applied without
    # changing launcher signatures or experiment configuration. The L40S uses
    # the complete default candidate set; other GPUs remain conservative until
    # their benchmark sweeps are recorded.
    "L40S": 32,
    "RTX-PRO-6000": 4,
}
DEFAULT_VIRTUAL_LEVEL_BATCH_SIZE = int(
    os.environ.get(
        "JAXUED_VIRTUAL_LEVEL_BATCH_SIZE",
        VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU.get(DEFAULT_GPU, 4),
    )
)
WANDB_SECRET_NAME = os.environ.get("JAXUED_WANDB_SECRET", "wandb-secret")
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "JAXUED_CHECKPOINT_VOLUME",
    "jaxued-checkpoints",
)


@dataclass(frozen=True)
class MazeRun:
    """Configuration shared by every Maze training launcher."""

    run_name: str
    project: str = "JAXUED_TEST"
    seed: int = 0
    num_updates: int = 30_000
    checkpoint_save_interval: int = 10

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must not be empty")
        if not self.project.strip():
            raise ValueError("project must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.num_updates <= 0:
            raise ValueError("num_updates must be positive")
        if self.checkpoint_save_interval <= 0:
            raise ValueError("checkpoint_save_interval must be positive")


@dataclass(frozen=True)
class EnsembleEpistemicRun(MazeRun):
    """Configuration for ensemble epistemic-uncertainty PLR."""

    num_agents: int = 8
    virtual_rollout_phases: int = 3
    virtual_epoch_ppo: int = 5
    virtual_level_batch_size: int = DEFAULT_VIRTUAL_LEVEL_BATCH_SIZE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.virtual_rollout_phases <= 0:
            raise ValueError("virtual_rollout_phases must be positive")
        if self.virtual_epoch_ppo <= 0:
            raise ValueError("virtual_epoch_ppo must be positive")
        if self.virtual_level_batch_size <= 0:
            raise ValueError("virtual_level_batch_size must be positive")


def parse_three_seeds(raw_seeds: str) -> tuple[int, int, int]:
    """Parse exactly three distinct, non-negative comma-separated seeds."""
    parts = [part.strip() for part in raw_seeds.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("seeds must contain exactly three comma-separated integers")

    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "seeds must contain exactly three comma-separated integers"
        ) from error

    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(seeds)) != 3:
        raise ValueError("seeds must be distinct")
    return seeds


def _base_command(
    script: str,
    run: MazeRun,
    *,
    python_executable: str,
) -> tuple[str, ...]:
    return (
        python_executable,
        script,
        "--project",
        run.project,
        "--run_name",
        run.run_name,
        "--seed",
        str(run.seed),
        "--num_updates",
        str(run.num_updates),
    )


def build_robust_plr_command(
    run: MazeRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the Robust PLR command without performing any side effects."""
    return _base_command(
        "examples/maze_plr.py",
        run,
        python_executable=python_executable,
    ) + (
        "--score_function",
        "MaxMC",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(run.checkpoint_save_interval),
    )


def build_mean_absolute_advantage_plr_command(
    run: MazeRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build Robust PLR scored by mean absolute advantage."""
    return _base_command(
        "examples/maze_plr.py",
        run,
        python_executable=python_executable,
    ) + (
        "--score_function",
        "mean_absolute_advantage",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(run.checkpoint_save_interval),
    )


def build_mean_positive_delight_plr_command(
    run: MazeRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build Robust PLR scored by mean positive delight."""
    return _base_command(
        "examples/maze_plr.py",
        run,
        python_executable=python_executable,
    ) + (
        "--score_function",
        "mean_positive_delight",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(run.checkpoint_save_interval),
    )


def build_plr_plus_command(
    run: MazeRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build standard PLR with exploratory gradient updates (PLR+)."""
    return _base_command(
        "examples/maze_plr.py",
        run,
        python_executable=python_executable,
    ) + (
        "--score_function",
        "MaxMC",
        "--exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(run.checkpoint_save_interval),
    )


def build_ensemble_epistemic_command(
    run: EnsembleEpistemicRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the ensemble epistemic-uncertainty PLR command."""
    return _base_command(
        "examples/maze_ensemble_plr.py",
        run,
        python_executable=python_executable,
    ) + (
        "--num_agents",
        str(run.num_agents),
        "--virtual_rollout_phases",
        str(run.virtual_rollout_phases),
        "--virtual_epoch_ppo",
        str(run.virtual_epoch_ppo),
        "--virtual_level_batch_size",
        str(run.virtual_level_batch_size),
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(run.checkpoint_save_interval),
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("uv")
    .add_local_dir(
        LOCAL_REPOSITORY,
        remote_path=str(REMOTE_REPOSITORY),
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            ".env",
            "**/__pycache__/**",
            "checkpoints/**",
            "results/**",
            "wandb/**",
        ],
    )
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "-e '/root/jaxued[examples]' 'jax[cuda12]'"
    )
    .env(
        {
            "PYTHONPATH": "/root/jaxued/modal-run/maze:/root/jaxued",
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "online",
        }
    )
)

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)


def run_training(command: tuple[str, ...], run: MazeRun) -> str:
    """Run training remotely and persist its checkpoint directory."""
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            f"Modal secret {WANDB_SECRET_NAME!r} must contain WANDB_API_KEY"
        )

    try:
        subprocess.run(
            command,
            cwd=REMOTE_REPOSITORY,
            env={**os.environ, "WANDB_MODE": "online"},
            check=True,
        )
    finally:
        checkpoint_volume.commit()

    return str(CHECKPOINT_DIRECTORY / run.run_name / str(run.seed))
