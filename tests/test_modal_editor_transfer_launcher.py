"""Tests for the editor-transfer Modal sweep configuration."""

import sys
from pathlib import Path

import pytest


pytest.importorskip("modal")

LAUNCHER_DIRECTORY = Path(__file__).resolve().parents[1] / "modal-run" / "maze"
sys.path.insert(0, str(LAUNCHER_DIRECTORY))

from _common import (  # noqa: E402
    EditorTransferRun,
    MazeRun,
    build_editor_transfer_command,
    build_robust_plr_command,
)
from editor_transfer_three_seeds import (  # noqa: E402
    DEFAULT_SEEDS,
    parse_seeds,
)


def test_editor_transfer_command_defaults_to_robust_plr() -> None:
    command = build_editor_transfer_command(
        EditorTransferRun(run_name="maze_editor_transfer"),
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "maze_editor_transfer",
        "--seed",
        "0",
        "--num_updates",
        "30000",
        "--score_function",
        "editor_transfer",
        "--transfer_target_count",
        "128",
        "--transfer_num_edits",
        "16",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


def test_editor_transfer_matches_robust_plr_shared_modal_config() -> None:
    robust_command = build_robust_plr_command(MazeRun(run_name="matched"))
    transfer_command = build_editor_transfer_command(
        EditorTransferRun(run_name="matched")
    )

    value_options = (
        "--project",
        "--run_name",
        "--seed",
        "--num_updates",
        "--checkpoint_save_interval",
    )
    for option in value_options:
        assert transfer_command[transfer_command.index(option) + 1] == (
            robust_command[robust_command.index(option) + 1]
        )

    assert "--no-exploratory_grad_updates" in robust_command
    assert "--no-exploratory_grad_updates" in transfer_command
    assert "--no-use_accel" in robust_command
    assert "--no-use_accel" in transfer_command


def test_editor_transfer_command_can_select_plr_plus() -> None:
    command = build_editor_transfer_command(
        EditorTransferRun(
            run_name="maze_editor_transfer_exploratory",
            exploratory_grad_updates=True,
        ),
        python_executable="python",
    )

    assert "--exploratory_grad_updates" in command
    assert "--no-exploratory_grad_updates" not in command


def test_editor_transfer_command_exposes_target_bank_shape() -> None:
    command = build_editor_transfer_command(
        EditorTransferRun(
            run_name="custom_bank",
            transfer_target_count=32,
            transfer_num_edits=5,
        )
    )

    assert command[command.index("--transfer_target_count") + 1] == "32"
    assert command[command.index("--transfer_num_edits") + 1] == "5"


def test_editor_transfer_sweep_defaults_to_three_distinct_seeds() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert parse_seeds(",".join(map(str, DEFAULT_SEEDS))) == DEFAULT_SEEDS


@pytest.mark.parametrize(
    ("raw_seeds", "message"),
    (
        ("0,1", "exactly three"),
        ("0,1,1", "distinct"),
        ("0,-1,2", "non-negative"),
        ("0,one,2", "integers"),
    ),
)
def test_editor_transfer_sweep_rejects_invalid_seeds(
    raw_seeds: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_seeds(raw_seeds)


@pytest.mark.parametrize(
    "field",
    ("transfer_target_count", "transfer_num_edits"),
)
def test_editor_transfer_run_requires_positive_bank_controls(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EditorTransferRun(run_name="invalid", **{field: 0})
