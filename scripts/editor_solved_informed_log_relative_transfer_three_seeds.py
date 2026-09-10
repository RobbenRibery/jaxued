"""Run a sequential solved-informed log-relative editor-transfer seed sweep.

Run from the repository root, for example::

    python scripts/editor_solved_informed_log_relative_transfer_three_seeds.py \
        --seeds 3,4,5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_TRANSFER_TARGET_COUNT = 128
DEFAULT_TRANSFER_NUM_EDITS = 16
DEFAULT_TRANSFER_LOG_RELATIVE_TAU = 0.1
DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA = 1.0
DEFAULT_TRANSFER_SOLVED_PRIOR_BETA = 1.0
DEFAULT_TRANSFER_SOLVED_CONFIDENCE = 0.8


@dataclass(frozen=True)
class LocalSweep:
    """Validated configuration for a sequential local seed sweep."""

    run_name: str = "maze_editor_solved_informed_log_relative_transfer_three_seeds"
    project: str = "JAXUED_TEST"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    num_updates: int = 30_000
    checkpoint_save_interval: int = 10
    transfer_target_count: int = DEFAULT_TRANSFER_TARGET_COUNT
    transfer_num_edits: int = DEFAULT_TRANSFER_NUM_EDITS
    transfer_log_relative_tau: float = DEFAULT_TRANSFER_LOG_RELATIVE_TAU
    transfer_solved_prior_alpha: float = DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA
    transfer_solved_prior_beta: float = DEFAULT_TRANSFER_SOLVED_PRIOR_BETA
    transfer_solved_confidence: float = DEFAULT_TRANSFER_SOLVED_CONFIDENCE
    exploratory_grad_updates: bool = False

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must not be empty")
        if not self.project.strip():
            raise ValueError("project must not be empty")
        if not self.seeds:
            raise ValueError("seeds must contain at least one integer")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        positive_fields = {
            "num_updates": self.num_updates,
            "checkpoint_save_interval": self.checkpoint_save_interval,
            "transfer_target_count": self.transfer_target_count,
            "transfer_num_edits": self.transfer_num_edits,
            "transfer_log_relative_tau": self.transfer_log_relative_tau,
            "transfer_solved_prior_beta": self.transfer_solved_prior_beta,
        }
        for field_name, value in positive_fields.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if (
            not isfinite(self.transfer_solved_prior_alpha)
            or self.transfer_solved_prior_alpha != 1.0
        ):
            raise ValueError("transfer_solved_prior_alpha must equal 1")
        if not isfinite(self.transfer_solved_confidence) or not (
            0 < self.transfer_solved_confidence < 1
        ):
            raise ValueError("transfer_solved_confidence must be between 0 and 1")


@dataclass(frozen=True)
class SeedRunResult:
    """Outcome and checkpoint location for one attempted seed."""

    seed: int
    returncode: int
    checkpoint_path: Path


def parse_seeds(raw_seeds: str) -> tuple[int, ...]:
    """Parse one or more distinct, non-negative comma-separated seeds."""
    parts = [part.strip() for part in raw_seeds.split(",")]
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "seeds must contain at least one comma-separated integer"
        )
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must contain comma-separated integers"
        ) from error
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return cast(tuple[int, ...], seeds)


def build_seed_command(
    sweep: LocalSweep,
    seed: int,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build one matched solved-informed Robust PLR training command."""
    exploratory_flag = (
        "--exploratory_grad_updates"
        if sweep.exploratory_grad_updates
        else "--no-exploratory_grad_updates"
    )
    return (
        python_executable,
        "examples/maze_plr.py",
        "--project",
        sweep.project,
        "--run_name",
        sweep.run_name,
        "--seed",
        str(seed),
        "--num_updates",
        str(sweep.num_updates),
        "--score_function",
        "editor_solved_informed_log_relative_transfer",
        "--transfer_target_count",
        str(sweep.transfer_target_count),
        "--transfer_num_edits",
        str(sweep.transfer_num_edits),
        "--transfer_log_relative_tau",
        str(sweep.transfer_log_relative_tau),
        "--transfer_solved_prior_alpha",
        str(sweep.transfer_solved_prior_alpha),
        "--transfer_solved_prior_beta",
        str(sweep.transfer_solved_prior_beta),
        "--transfer_solved_confidence",
        str(sweep.transfer_solved_confidence),
        exploratory_flag,
        "--no-use_accel",
        "--checkpoint_save_interval",
        str(sweep.checkpoint_save_interval),
    )


def run_seed(sweep: LocalSweep, seed: int) -> SeedRunResult:
    """Run one seed in the current environment and return its outcome."""
    command = build_seed_command(sweep, seed)
    print(f"Starting seed {seed}: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
        },
        check=False,
    )
    checkpoint_path = REPOSITORY / "checkpoints" / sweep.run_name / str(seed)
    return SeedRunResult(
        seed=seed,
        returncode=completed.returncode,
        checkpoint_path=checkpoint_path,
    )


def run_sweep(sweep: LocalSweep) -> tuple[SeedRunResult, ...]:
    """Run every requested seed sequentially on the current GPU."""
    return tuple(run_seed(sweep, seed) for seed in sweep.seeds)


def parse_args(argv: list[str] | None = None) -> LocalSweep:
    """Parse the local sweep CLI into a validated configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default="maze_editor_solved_informed_log_relative_transfer_three_seeds",
    )
    parser.add_argument("--project", default="JAXUED_TEST")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--num-updates", "--num_updates", type=int, default=30_000)
    parser.add_argument(
        "--checkpoint-save-interval",
        "--checkpoint_save_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--transfer-target-count",
        "--transfer_target_count",
        type=int,
        default=DEFAULT_TRANSFER_TARGET_COUNT,
    )
    parser.add_argument(
        "--transfer-num-edits",
        "--transfer_num_edits",
        type=int,
        default=DEFAULT_TRANSFER_NUM_EDITS,
    )
    parser.add_argument(
        "--transfer-log-relative-tau",
        "--transfer_log_relative_tau",
        type=float,
        default=DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    )
    parser.add_argument(
        "--transfer-solved-prior-alpha",
        "--transfer_solved_prior_alpha",
        type=float,
        default=DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA,
    )
    parser.add_argument(
        "--transfer-solved-prior-beta",
        "--transfer_solved_prior_beta",
        type=float,
        default=DEFAULT_TRANSFER_SOLVED_PRIOR_BETA,
    )
    parser.add_argument(
        "--transfer-solved-confidence",
        "--transfer_solved_confidence",
        type=float,
        default=DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    )
    parser.add_argument(
        "--exploratory-grad-updates",
        action="store_true",
        help="Persist source PPO updates (PLR/PLR+ instead of Robust PLR).",
    )
    args = parser.parse_args(argv)
    try:
        return LocalSweep(**vars(args))
    except ValueError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> int:
    """Run the local sweep and report all failed seeds."""
    sweep = parse_args(argv)
    results = run_sweep(sweep)
    failures = [result for result in results if result.returncode != 0]
    for result in results:
        if result.returncode == 0:
            print(
                f"Seed {result.seed} checkpoints saved at {result.checkpoint_path}",
                flush=True,
            )
        else:
            print(
                f"Seed {result.seed} failed with exit code {result.returncode}",
                file=sys.stderr,
                flush=True,
            )
    if failures:
        failed_seeds = ", ".join(str(result.seed) for result in failures)
        print(f"Local sweep failed for seed(s): {failed_seeds}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
