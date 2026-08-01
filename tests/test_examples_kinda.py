import subprocess
import pytest
import os
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples")
EXAMPLE_SCRIPTS = [
    "maze_dr.py",
    "maze_plr.py",
    "maze_ensemble_plr.py",
    "maze_paired.py",
]

EXAMPLE_ARGUMENTS = {
    "maze_ensemble_plr.py": [
        "--num_agents",
        "2",
        "--num_updates",
        "1",
        "--eval_freq",
        "1",
        "--eval_num_attempts",
        "1",
        "--eval_levels",
        "StandardMaze",
        "--num_steps",
        "2",
        "--num_train_envs",
        "2",
        "--epoch_ppo",
        "1",
        "--level_buffer_capacity",
        "4",
        "--n_walls",
        "5",
        "--no-buffer_duplicate_check",
    ],
}

@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_run_example(script):
    script_path = os.path.join(EXAMPLES_DIR, script)
    assert os.path.exists(script_path), f"Script {script} not found."

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    try:
        process = subprocess.run(
            [sys.executable, script_path, *EXAMPLE_ARGUMENTS.get(script, [])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=env,
        )
        assert process.returncode in [None, 0], f"Script {script} failed:\n{process.stderr.decode()}"
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pytest.fail(f"Error running {script}: {e}")
