"""Tests for the editor-transfer Modal sweep configuration."""

import sys
from pathlib import Path

import pytest


pytest.importorskip("modal")

LAUNCHER_DIRECTORY = Path(__file__).resolve().parents[1] / "modal-run" / "maze"
sys.path.insert(0, str(LAUNCHER_DIRECTORY))

from _common import (  # noqa: E402
    DEFAULT_EDITOR_TRANSFER_GPU,
    DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    EditorLogRelativeTransferRun,
    EditorTransferRun,
    MazeRun,
    build_editor_log_relative_transfer_command,
    build_editor_transfer_command,
    build_robust_plr_command,
    resolve_editor_transfer_gpu,
)
from editor_transfer_three_seeds import (  # noqa: E402
    DEFAULT_SEEDS,
    parse_seeds,
)
from editor_log_relative_transfer_three_seeds import (  # noqa: E402
    DEFAULT_SEEDS as LOG_RELATIVE_DEFAULT_SEEDS,
    parse_seeds as parse_log_relative_seeds,
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


def test_editor_transfer_sweeps_default_to_l40s() -> None:
    assert DEFAULT_EDITOR_TRANSFER_GPU == "L40S"
    assert resolve_editor_transfer_gpu({}) == "L40S"
    assert resolve_editor_transfer_gpu({"JAXUED_MODAL_GPU": "H100"}) == "H100"


def test_log_relative_transfer_defaults_to_tau_point_one() -> None:
    assert DEFAULT_TRANSFER_LOG_RELATIVE_TAU == pytest.approx(0.1)
    assert EditorLogRelativeTransferRun(
        run_name="log_relative"
    ).transfer_log_relative_tau == pytest.approx(0.1)


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


def test_log_relative_transfer_has_distinct_score_name_and_tau() -> None:
    command = build_editor_log_relative_transfer_command(
        EditorLogRelativeTransferRun(
            run_name="log_relative",
            transfer_log_relative_tau=0.25,
        )
    )

    assert command[command.index("--score_function") + 1] == (
        "editor_log_relative_transfer"
    )
    assert command[command.index("--transfer_log_relative_tau") + 1] == "0.25"


def _command_without_option(command: tuple[str, ...], option: str) -> tuple[str, ...]:
    option_index = command.index(option)
    return command[:option_index] + command[option_index + 2 :]


def test_log_relative_sweep_controls_match_absolute_transfer() -> None:
    absolute = build_editor_transfer_command(
        EditorTransferRun(run_name="matched", seed=2),
        python_executable="python",
    )
    relative = build_editor_log_relative_transfer_command(
        EditorLogRelativeTransferRun(run_name="matched", seed=2),
        python_executable="python",
    )

    relative_without_tau = _command_without_option(
        relative,
        "--transfer_log_relative_tau",
    )
    absolute_score_index = absolute.index("--score_function") + 1
    relative_score_index = relative_without_tau.index("--score_function") + 1

    assert absolute_score_index == relative_score_index
    assert absolute[:absolute_score_index] == (
        relative_without_tau[:relative_score_index]
    )
    assert absolute[absolute_score_index + 1 :] == (
        relative_without_tau[relative_score_index + 1 :]
    )


def test_editor_transfer_sweep_defaults_to_three_distinct_seeds() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert parse_seeds(",".join(map(str, DEFAULT_SEEDS))) == DEFAULT_SEEDS
    assert LOG_RELATIVE_DEFAULT_SEEDS == DEFAULT_SEEDS
    assert parse_log_relative_seeds("0,1,2") == DEFAULT_SEEDS


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
    (
        "transfer_target_count",
        "transfer_num_edits",
    ),
)
def test_editor_transfer_run_requires_positive_bank_controls(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EditorTransferRun(run_name="invalid", **{field: 0})


def test_log_relative_transfer_run_requires_positive_tau() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EditorLogRelativeTransferRun(
            run_name="invalid",
            transfer_log_relative_tau=0,
        )
