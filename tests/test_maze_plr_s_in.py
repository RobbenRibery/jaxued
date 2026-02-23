"""Static integration tests for s_in wiring in examples/maze_plr.py."""

import os
import re


def _maze_plr_source() -> str:
    script = os.path.join(os.path.dirname(__file__), "../examples/maze_plr.py")
    with open(script, "r", encoding="utf-8") as f:
        return f.read()


def test_maze_plr_registers_s_in_cli_flags() -> None:
    """Parser should expose s_in score mode and related args."""
    source = _maze_plr_source()
    assert '"--score_function"' in source
    assert '"s_in"' in source
    assert '"--sin_n_virtual_updates"' in source
    assert '"--sin_eps"' in source


def test_maze_plr_s_in_is_wired_into_training_branches() -> None:
    """s_in branch logic should exist for new/replay/mutate update paths."""
    source = _maze_plr_source()

    # on_new_levels, on_replay_levels, on_mutate_levels
    assert source.count('config["score_function"] == "s_in"') >= 3
    assert "compute_s_in_scores(" in source
    assert "measure_s_in(" in source


def test_maze_plr_logs_s_in_metrics() -> None:
    """S_in logging keys should be present in log_eval."""
    source = _maze_plr_source()
    assert re.search(r'"lp/s_in_mean"', source)
    assert re.search(r'"lp/loss_before_mean"', source)
    assert re.search(r'"lp/loss_after_mean"', source)


def test_maze_plr_s_in_uses_independent_a_and_b_rollouts() -> None:
    """s_in branches should construct dedicated A and B rollouts."""
    source = _maze_plr_source()

    # Each branch should split rng into both A and B reset keys.
    assert source.count("rng_reset_a_sin") >= 3
    assert source.count("rng_reset_b") >= 3
    assert source.count("obs_a_sin") >= 3
    assert source.count("obs_b") >= 3


def test_maze_plr_s_in_uses_ppo_eval_loss() -> None:
    """s_in scoring should evaluate held-out data with PPO (not value-only) loss."""
    source = _maze_plr_source()

    assert "def build_ppo_eval_batch_from_rollout(" in source
    assert "def ppo_loss_fn_for_s_in(" in source
    assert "loss_fn=lambda state, batch: ppo_loss_fn_for_s_in(" in source
