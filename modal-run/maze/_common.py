"""Shared configuration and runtime plumbing for Maze Modal runs."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import modal


REMOTE_REPOSITORY = Path("/root/jaxued")
CHECKPOINT_DIRECTORY = REMOTE_REPOSITORY / "checkpoints"
LOCAL_REPOSITORY = Path(__file__).resolve().parents[2]

DEFAULT_EDITOR_TRANSFER_GPU = "L40S"
DEFAULT_TRANSFER_TARGET_COUNT = 128
DEFAULT_TRANSFER_NUM_EDITS = 16
DEFAULT_TRANSFER_LOG_RELATIVE_TAU = 0.1
DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA = 1.0
DEFAULT_TRANSFER_SOLVED_PRIOR_BETA = 1.0
DEFAULT_TRANSFER_SOLVED_CONFIDENCE = 0.8
WANDB_SECRET_NAME = os.environ.get("JAXUED_WANDB_SECRET", "wandb-secret")
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "JAXUED_CHECKPOINT_VOLUME",
    "jaxued-checkpoints",
)


def resolve_editor_transfer_gpu(environ: Mapping[str, str]) -> str:
    """Resolve the transfer sweep GPU while preserving the shared override."""
    return environ.get("JAXUED_MODAL_GPU", DEFAULT_EDITOR_TRANSFER_GPU)


DEFAULT_GPU = resolve_editor_transfer_gpu(os.environ)


@dataclass(frozen=True)
class MazeRun:
    """Configuration shared by Maze training launchers."""

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
class EditorTransferRun(MazeRun):
    """Configuration for editor-transfer PLR scoring."""

    transfer_target_count: int = DEFAULT_TRANSFER_TARGET_COUNT
    transfer_num_edits: int = DEFAULT_TRANSFER_NUM_EDITS
    exploratory_grad_updates: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transfer_target_count <= 0:
            raise ValueError("transfer_target_count must be positive")
        if self.transfer_num_edits <= 0:
            raise ValueError("transfer_num_edits must be positive")


@dataclass(frozen=True)
class EditorLogRelativeTransferRun(EditorTransferRun):
    """Configuration for smoothed log-relative editor-transfer scoring."""

    transfer_log_relative_tau: float = DEFAULT_TRANSFER_LOG_RELATIVE_TAU

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isfinite(self.transfer_log_relative_tau)
            or self.transfer_log_relative_tau <= 0
        ):
            raise ValueError("transfer_log_relative_tau must be positive")


@dataclass(frozen=True)
class EditorSolvedInformedLogRelativeTransferRun(EditorLogRelativeTransferRun):
    """Configuration for solved-informed log-relative transfer scoring."""

    transfer_solved_prior_alpha: float = DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA
    transfer_solved_prior_beta: float = DEFAULT_TRANSFER_SOLVED_PRIOR_BETA
    transfer_solved_confidence: float = DEFAULT_TRANSFER_SOLVED_CONFIDENCE

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isfinite(self.transfer_solved_prior_alpha)
            or self.transfer_solved_prior_alpha != 1.0
        ):
            raise ValueError("transfer_solved_prior_alpha must equal 1")
        if (
            not isfinite(self.transfer_solved_prior_beta)
            or self.transfer_solved_prior_beta <= 0
        ):
            raise ValueError("transfer_solved_prior_beta must be positive")
        if not isfinite(self.transfer_solved_confidence) or not (
            0 < self.transfer_solved_confidence < 1
        ):
            raise ValueError("transfer_solved_confidence must be between 0 and 1")


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
    """Build the shared Maze launcher command prefix."""
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
    """Build the standardized Robust PLR baseline command."""
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


def _build_editor_transfer_command(
    run: EditorTransferRun,
    score_function: str,
    score_options: tuple[str, ...] = (),
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the options shared by both editor-transfer score variants."""
    exploratory_flag = (
        "--exploratory_grad_updates"
        if run.exploratory_grad_updates
        else "--no-exploratory_grad_updates"
    )
    return (
        _base_command(
            "examples/maze_plr.py",
            run,
            python_executable=python_executable,
        )
        + (
            "--score_function",
            score_function,
            "--transfer_target_count",
            str(run.transfer_target_count),
            "--transfer_num_edits",
            str(run.transfer_num_edits),
        )
        + score_options
        + (
            exploratory_flag,
            "--no-use_accel",
            "--checkpoint_save_interval",
            str(run.checkpoint_save_interval),
        )
    )


def build_editor_transfer_command(
    run: EditorTransferRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the historical absolute-gain editor-transfer command."""
    return _build_editor_transfer_command(
        run,
        "editor_transfer",
        python_executable=python_executable,
    )


def build_editor_log_relative_transfer_command(
    run: EditorLogRelativeTransferRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the separately named log-relative editor-transfer command."""
    return _build_editor_transfer_command(
        run,
        "editor_log_relative_transfer",
        (
            "--transfer_log_relative_tau",
            str(run.transfer_log_relative_tau),
        ),
        python_executable=python_executable,
    )


def build_editor_solved_informed_log_relative_transfer_command(
    run: EditorSolvedInformedLogRelativeTransferRun,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the separately named solved-informed log-relative command."""
    return _build_editor_transfer_command(
        run,
        "editor_solved_informed_log_relative_transfer",
        (
            "--transfer_log_relative_tau",
            str(run.transfer_log_relative_tau),
            "--transfer_solved_prior_alpha",
            str(run.transfer_solved_prior_alpha),
            "--transfer_solved_prior_beta",
            str(run.transfer_solved_prior_beta),
            "--transfer_solved_confidence",
            str(run.transfer_solved_confidence),
        ),
        python_executable=python_executable,
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
        "uv pip install --system --compile-bytecode -e /root/jaxued 'jax[cuda12]'"
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
