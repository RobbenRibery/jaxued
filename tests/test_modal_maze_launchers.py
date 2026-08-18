"""Tests for the shared Maze Modal launcher configurations."""

import sys
from pathlib import Path

import pytest


pytest.importorskip("modal")

LAUNCHER_DIRECTORY = Path(__file__).resolve().parents[1] / "modal-run" / "maze"
sys.path.insert(0, str(LAUNCHER_DIRECTORY))

from _common import (  # noqa: E402
    EnsembleEpistemicRun,
    MazeRun,
    VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU,
    build_ensemble_epistemic_command,
    build_mean_absolute_advantage_plr_command,
    build_mean_positive_delight_plr_command,
    build_plr_plus_command,
    build_robust_plr_command,
)
from mean_absolute_advantage_three_seeds import (  # noqa: E402
    DEFAULT_SEEDS as MEAN_ABSOLUTE_ADVANTAGE_DEFAULT_SEEDS,
    parse_seeds as parse_mean_absolute_advantage_seeds,
)
from mean_positive_delight_three_seeds import (  # noqa: E402
    DEFAULT_SEEDS as MEAN_POSITIVE_DELIGHT_DEFAULT_SEEDS,
    parse_seeds as parse_mean_positive_delight_seeds,
)
from robust_plr_three_seeds import DEFAULT_SEEDS, parse_seeds  # noqa: E402


def test_build_robust_plr_command() -> None:
    command = build_robust_plr_command(
        MazeRun(run_name="maze_robust_plr"),
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "maze_robust_plr",
        "--seed",
        "0",
        "--num_updates",
        "30000",
        "--score_function",
        "MaxMC",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


def test_build_mean_absolute_advantage_plr_command() -> None:
    command = build_mean_absolute_advantage_plr_command(
        MazeRun(run_name="maze_mean_absolute_advantage"),
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "maze_mean_absolute_advantage",
        "--seed",
        "0",
        "--num_updates",
        "30000",
        "--score_function",
        "mean_absolute_advantage",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


def test_build_mean_positive_delight_plr_command() -> None:
    command = build_mean_positive_delight_plr_command(
        MazeRun(run_name="maze_mean_positive_delight"),
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "maze_mean_positive_delight",
        "--seed",
        "0",
        "--num_updates",
        "30000",
        "--score_function",
        "mean_positive_delight",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


def test_three_seed_sweep_defaults_to_distinct_seeds() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert parse_seeds(",".join(map(str, DEFAULT_SEEDS))) == DEFAULT_SEEDS


def test_mean_absolute_advantage_sweep_defaults_to_distinct_seeds() -> None:
    assert MEAN_ABSOLUTE_ADVANTAGE_DEFAULT_SEEDS == (0, 1, 2)
    assert (
        parse_mean_absolute_advantage_seeds(
            ",".join(map(str, MEAN_ABSOLUTE_ADVANTAGE_DEFAULT_SEEDS))
        )
        == MEAN_ABSOLUTE_ADVANTAGE_DEFAULT_SEEDS
    )


def test_mean_positive_delight_sweep_defaults_to_distinct_seeds() -> None:
    assert MEAN_POSITIVE_DELIGHT_DEFAULT_SEEDS == (0, 1, 2)
    assert (
        parse_mean_positive_delight_seeds(
            ",".join(map(str, MEAN_POSITIVE_DELIGHT_DEFAULT_SEEDS))
        )
        == MEAN_POSITIVE_DELIGHT_DEFAULT_SEEDS
    )


@pytest.mark.parametrize(
    ("raw_seeds", "message"),
    (
        ("0,1", "exactly three"),
        ("0,1,1", "distinct"),
        ("0,-1,2", "non-negative"),
        ("0,one,2", "integers"),
    ),
)
def test_three_seed_sweep_rejects_invalid_seeds(
    raw_seeds: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_seeds(raw_seeds)


def test_build_plr_plus_command_enables_exploratory_updates() -> None:
    command = build_plr_plus_command(
        MazeRun(run_name="maze_plr_plus"),
        python_executable="python",
    )

    assert "--exploratory_grad_updates" in command
    assert "--no-exploratory_grad_updates" not in command
    assert command[-2:] == ("--checkpoint_save_interval", "10")


def test_build_ensemble_epistemic_command_uses_eight_agents() -> None:
    command = build_ensemble_epistemic_command(
        EnsembleEpistemicRun(
            run_name="maze_ensemble_epistemic_8_agents",
            checkpoint_save_interval=2_500,
        ),
        python_executable="python",
    )

    assert command[1] == "examples/maze_ensemble_plr.py"
    assert command[command.index("--num_agents") + 1] == "8"
    assert command[command.index("--virtual_rollout_phases") + 1] == "3"
    assert command[command.index("--virtual_epoch_ppo") + 1] == "5"
    assert command[command.index("--virtual_level_batch_size") + 1] == "32"
    assert command[-2:] == ("--checkpoint_save_interval", "2500")


def test_virtual_level_batch_defaults_are_gpu_specific() -> None:
    assert set(VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU) == {"L40S", "RTX-PRO-6000"}
    assert VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU["L40S"] == 32
    assert VIRTUAL_LEVEL_BATCH_SIZE_BY_GPU["RTX-PRO-6000"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_name", ""),
        ("project", ""),
        ("seed", -1),
        ("num_updates", 0),
        ("checkpoint_save_interval", 0),
    ),
)
def test_maze_run_rejects_invalid_inputs(field: str, value: object) -> None:
    inputs = {
        "run_name": "valid",
        "project": "JAXUED_TEST",
        "seed": 0,
        "num_updates": 30_000,
        "checkpoint_save_interval": 10,
    }
    inputs[field] = value

    with pytest.raises(ValueError):
        MazeRun(**inputs)


def test_ensemble_requires_at_least_two_agents() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleEpistemicRun(run_name="ensemble", num_agents=1)


@pytest.mark.parametrize(
    "field",
    (
        "virtual_rollout_phases",
        "virtual_epoch_ppo",
        "virtual_level_batch_size",
    ),
)
def test_ensemble_requires_positive_virtual_controls(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EnsembleEpistemicRun(run_name="ensemble", **{field: 0})
