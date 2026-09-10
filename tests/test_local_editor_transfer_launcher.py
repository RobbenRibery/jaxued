"""Tests for the local/Runpod editor-transfer sweep launcher."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "editor_log_relative_transfer_three_seeds.py"
)
SPEC = importlib.util.spec_from_file_location(
    "local_editor_transfer_launcher", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

SOLVED_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "editor_solved_informed_log_relative_transfer_three_seeds.py"
)
SOLVED_SPEC = importlib.util.spec_from_file_location(
    "local_solved_informed_editor_transfer_launcher",
    SOLVED_SCRIPT_PATH,
)
assert SOLVED_SPEC is not None and SOLVED_SPEC.loader is not None
solved_launcher = importlib.util.module_from_spec(SOLVED_SPEC)
sys.modules[SOLVED_SPEC.name] = solved_launcher
SOLVED_SPEC.loader.exec_module(solved_launcher)


def test_parses_requested_seed_sweeps() -> None:
    assert launcher.parse_seeds("3") == (3,)
    assert launcher.parse_seeds("3,4,5") == (3, 4, 5)
    single_seed_sweep = launcher.parse_args(["--seeds", "3"])
    sweep = launcher.parse_args(["--seeds", "3,4,5"])
    assert single_seed_sweep.seeds == (3,)
    assert sweep.seeds == (3, 4, 5)


@pytest.mark.parametrize(
    ("raw_seeds", "message"),
    (
        ("", "at least one"),
        ("0,1,1", "distinct"),
        ("0,-1,2", "non-negative"),
        ("0,one,2", "integers"),
    ),
)
def test_rejects_invalid_seed_sweeps(raw_seeds: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        launcher.parse_seeds(raw_seeds)


def test_builds_matched_log_relative_robust_plr_command() -> None:
    sweep = launcher.LocalSweep(
        run_name="local_log_relative",
        seeds=(3, 4, 5),
    )

    command = launcher.build_seed_command(
        sweep,
        seed=3,
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "local_log_relative",
        "--seed",
        "3",
        "--num_updates",
        "30000",
        "--score_function",
        "editor_log_relative_transfer",
        "--transfer_target_count",
        "128",
        "--transfer_num_edits",
        "16",
        "--transfer_log_relative_tau",
        "0.1",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


def test_runs_seeds_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted_seeds = []

    def fake_run_seed(sweep, seed):
        attempted_seeds.append(seed)
        return launcher.SeedRunResult(
            seed=seed,
            returncode=0,
            checkpoint_path=Path("checkpoints") / sweep.run_name / str(seed),
        )

    monkeypatch.setattr(launcher, "run_seed", fake_run_seed)

    results = launcher.run_sweep(launcher.LocalSweep(seeds=(3, 4, 5)))

    assert attempted_seeds == [3, 4, 5]
    assert tuple(result.seed for result in results) == (3, 4, 5)


def test_builds_matched_solved_informed_command_with_exact_defaults() -> None:
    sweep = solved_launcher.LocalSweep(
        run_name="local_solved_informed",
        seeds=(3, 4, 5),
    )

    command = solved_launcher.build_seed_command(
        sweep,
        seed=3,
        python_executable="/usr/bin/python3",
    )

    assert command == (
        "/usr/bin/python3",
        "examples/maze_plr.py",
        "--project",
        "JAXUED_TEST",
        "--run_name",
        "local_solved_informed",
        "--seed",
        "3",
        "--num_updates",
        "30000",
        "--score_function",
        "editor_solved_informed_log_relative_transfer",
        "--transfer_target_count",
        "128",
        "--transfer_num_edits",
        "16",
        "--transfer_log_relative_tau",
        "0.1",
        "--transfer_solved_prior_alpha",
        "1.0",
        "--transfer_solved_prior_beta",
        "1.0",
        "--transfer_solved_confidence",
        "0.8",
        "--no-exploratory_grad_updates",
        "--no-use_accel",
        "--checkpoint_save_interval",
        "10",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("transfer_solved_prior_alpha", 0, "must equal 1"),
        ("transfer_solved_prior_alpha", 2, "must equal 1"),
        ("transfer_solved_prior_alpha", float("nan"), "must equal 1"),
        ("transfer_solved_prior_beta", 0, "must be positive"),
        ("transfer_solved_prior_beta", float("inf"), "must be positive"),
        ("transfer_solved_confidence", 0, "between 0 and 1"),
        ("transfer_solved_confidence", 1, "between 0 and 1"),
        ("transfer_solved_confidence", float("nan"), "between 0 and 1"),
    ),
)
def test_local_solved_informed_launcher_rejects_invalid_controls(
    field: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solved_launcher.LocalSweep(**{field: value})


def test_solved_informed_launcher_runs_seeds_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_seeds = []

    def fake_run_seed(sweep, seed):
        attempted_seeds.append(seed)
        return solved_launcher.SeedRunResult(
            seed=seed,
            returncode=0,
            checkpoint_path=Path("checkpoints") / sweep.run_name / str(seed),
        )

    monkeypatch.setattr(solved_launcher, "run_seed", fake_run_seed)

    results = solved_launcher.run_sweep(solved_launcher.LocalSweep(seeds=(3, 4, 5)))

    assert attempted_seeds == [3, 4, 5]
    assert tuple(result.seed for result in results) == (3, 4, 5)
