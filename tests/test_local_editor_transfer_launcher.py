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
SPEC = importlib.util.spec_from_file_location("local_editor_transfer_launcher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


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
