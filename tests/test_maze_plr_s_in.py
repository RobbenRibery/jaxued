"""Static integration tests for s_in wiring in examples/maze_plr.py."""

import os
import re


def _read_repo_file(*parts: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", *parts)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _maze_plr_source() -> str:
    return _read_repo_file("examples", "maze_plr.py")


def _robust_plr_s_in_script_source() -> str:
    return _read_repo_file("scripts", "train_robust_plr_s_in.sh")


def _function_source(source: str, name: str) -> str:
    match = re.search(rf"^def {name}\(.*?^def ", source, flags=re.M | re.S)
    if match:
        return match.group(0).rsplit("\ndef ", maxsplit=1)[0]
    match = re.search(rf"^def {name}\(.*", source, flags=re.M | re.S)
    assert match is not None, f"Could not find function {name}"
    return match.group(0)


def test_maze_plr_registers_required_s_in_cli_flags() -> None:
    """Parser should expose s_in score mode and related args."""
    source = _maze_plr_source()
    assert '"--score_function"' in source
    assert '"s_in"' in source
    assert '"--sin_n_virtual_updates"' in source
    assert '"--sin_eps"' in source
    assert '"--sin_num_rollouts_per_level"' in source
    assert '"--sin_score_batch_size"' in source
    assert "default=None" in source
    assert "default=1" in source


def test_maze_plr_validates_s_in_rollout_count() -> None:
    """s_in should require an explicit positive rollout count per level."""
    source = _maze_plr_source()
    assert 'config["sin_num_rollouts_per_level"] < 1' in source
    assert (
        'parser.error("--sin_num_rollouts_per_level must be >= 1 when provided.")'
        in source
    )
    assert 'config["sin_score_batch_size"] < 1' in source
    assert 'parser.error("--sin_score_batch_size must be >= 1.")' in source
    assert 'config["score_function"] == "s_in"' in source
    assert 'config["sin_num_rollouts_per_level"] is None' in source
    assert (
        'parser.error("--sin_num_rollouts_per_level is required for --score_function s_in.")'
        in source
    )


def test_train_script_passes_explicit_s_in_rollout_count() -> None:
    """The s_in launch script should expose rollout count via shell env vars."""
    source = _robust_plr_s_in_script_source()
    assert "--score_function s_in" in source
    assert 'SIN_NUM_ROLLOUTS_PER_LEVEL="${SIN_NUM_ROLLOUTS_PER_LEVEL:-8}"' in source
    assert 'SIN_SCORE_BATCH_SIZE="${SIN_SCORE_BATCH_SIZE:-2}"' in source
    assert '--sin_num_rollouts_per_level "$SIN_NUM_ROLLOUTS_PER_LEVEL"' in source
    assert '--sin_score_batch_size "$SIN_SCORE_BATCH_SIZE"' in source
    assert 'SIN_N_VIRTUAL_UPDATES="${SIN_N_VIRTUAL_UPDATES:-4}"' in source
    assert '--sin_n_virtual_updates "$SIN_N_VIRTUAL_UPDATES"' in source
    assert (
        "-v${SIN_N_VIRTUAL_UPDATES}-r${SIN_NUM_ROLLOUTS_PER_LEVEL}"
        "-b${SIN_SCORE_BATCH_SIZE}-"
    ) in source


def test_maze_plr_s_in_collects_g_rollouts_per_set() -> None:
    """Dedicated s_in rollout collection should expand N slots to N * G envs."""
    source = _maze_plr_source()
    helper = _function_source(source, "collect_s_in_rollout_set")

    assert 'num_levels = config["num_train_envs"]' in helper
    assert 'num_rollouts_per_level = config["sin_num_rollouts_per_level"]' in helper
    assert "num_rollout_envs = num_levels * num_rollouts_per_level" in helper
    assert "jnp.repeat(x, num_rollouts_per_level, axis=0)" in helper
    assert "num_envs=num_rollout_envs" in helper
    assert re.search(
        r"x\.shape\[0\],\s*num_levels,\s*num_rollouts_per_level",
        helper,
        flags=re.S,
    )
    assert re.search(
        r"x\.reshape\(num_levels,\s*num_rollouts_per_level",
        helper,
        flags=re.S,
    )


def test_maze_plr_s_in_branches_use_independent_a_and_b_rollout_sets() -> None:
    """new/replay/mutate branches should collect Set A and Set B via the G helper."""
    source = _maze_plr_source()

    assert source.count("rollout_a_sin = collect_s_in_rollout_set(") >= 3
    assert source.count("rollout_b_sin = collect_s_in_rollout_set(") >= 3
    assert "levels=new_levels" in source
    assert "levels=levels" in source
    assert "levels=child_levels" in source
    assert source.count("rollout_a=rollout_a_sin") >= 3
    assert source.count("rollout_b=rollout_b_sin") >= 3


def test_maze_plr_s_in_returns_slotwise_plr_scores() -> None:
    """S_in should produce one score per sampled training environment slot."""
    source = _maze_plr_source()
    s_in_fn = _function_source(source, "compute_s_in_scores")

    assert "obs_a leaves: (T, N, G, ...)" in s_in_fn
    assert "scores: S_in per level. Shape: (N,)" in s_in_fn
    assert "num_envs = actions_a.shape[1]" in s_in_fn
    assert "jnp.arange(num_envs, dtype=jnp.int32)" in s_in_fn
    assert '"lp_s_in_mean": scores.mean()' in s_in_fn


def test_maze_plr_s_in_scores_levels_in_bounded_batches_for_tpu_compile_safety() -> None:
    """S_in should avoid vmapping virtual TrainState updates across all levels."""
    source = _maze_plr_source()
    s_in_fn = _function_source(source, "compute_s_in_scores")

    assert 'score_batch_size = min(config["sin_score_batch_size"], num_envs)' in s_in_fn
    assert "if score_batch_size == 1:" in s_in_fn
    assert "def _score_one(" in s_in_fn
    assert "def _score_batch(" in s_in_fn
    assert "jax.lax.scan(" in s_in_fn
    assert "num_score_batches = (num_envs + score_batch_size - 1) // score_batch_size" in s_in_fn
    assert "score_batch_size, *rng_levels.shape[1:]" in s_in_fn
    assert "jax.vmap(_per_level_score, in_axes=(0, 0))" in s_in_fn


def test_maze_plr_s_in_scores_drive_plr_insert_and_update_paths() -> None:
    """S_in scores should be consumed by PLR slotwise insert/update calls."""
    source = _maze_plr_source()

    assert source.count("scores=scores") >= 3
    assert "level_sampler.insert_batch(" in source
    assert "levels=new_levels" in source
    assert "levels=child_levels" in source
    assert "level_sampler.update_batch(" in source
    assert "level_inds=level_inds" in source


def test_maze_plr_s_in_uses_ppo_eval_loss_aggregated_over_t_and_g() -> None:
    """Held-out PPO loss should reduce one environment slot's (T, G) data to a scalar."""
    source = _maze_plr_source()
    loss_fn = _function_source(source, "ppo_loss_fn_for_s_in")

    assert "shape (T, G, ...)" in loss_fn
    assert "Shape: ()." in loss_fn
    assert "adv_mean = advantages.mean()" in loss_fn
    assert "adv_std = advantages.std()" in loss_fn
    assert ").mean()" in loss_fn
    assert ".mean(axis=0)" not in loss_fn
    assert "loss_fn=lambda state, batch: ppo_loss_fn_for_s_in(" in source


def test_maze_plr_s_in_virtual_update_uses_g_rollouts_as_env_axis() -> None:
    """Virtual PPO update should train on the selected level's G rollout batch."""
    source = _maze_plr_source()
    update_fn = _function_source(source, "virtual_update_fn_for_s_in")

    assert "(T, G, ...)" in update_fn
    assert "num_envs=actions.shape[1]" in update_fn
    assert "n_minibatch=1" in update_fn
