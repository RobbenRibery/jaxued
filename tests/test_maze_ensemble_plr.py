"""Integration tests for the Maze ensemble rollout and virtual update."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from examples.maze_ensemble_plr import (
    ActorCritic,
    CHECKPOINT_FORMAT,
    PPOParameters,
    VirtualLearningParameters,
    _load_ensemble_eval_config,
    build_parser,
    collect_ensemble_trajectories,
    encode_maze_state,
    main as run_ensemble,
    replay_action_probabilities,
    run_virtual_phases_for_level,
    score_ensemble_rollout,
    score_multi_phase_ensemble_rollout,
    update_ensemble_members,
)
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.wrappers import AutoReplayWrapper


def _make_rollout_fixture():
    base_env = Maze(
        max_height=7,
        max_width=7,
        agent_view_size=5,
        normalize_obs=True,
    )
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params.replace(max_steps_in_episode=2)
    level_generator = make_level_generator(7, 7, 5)
    levels = jax.vmap(level_generator)(jax.random.split(jax.random.PRNGKey(1), 2))
    init_obs, init_env_state = jax.vmap(
        env.reset_to_level,
        in_axes=(0, 0, None),
    )(
        jax.random.split(jax.random.PRNGKey(2), 2),
        levels,
        env_params,
    )

    network = ActorCritic(env.action_space(env_params).n)
    init_inputs = (
        jax.tree_util.tree_map(lambda value: value[None, ...], init_obs),
        jnp.zeros((1, 2), dtype=bool),
    )
    optimizer = optax.adam(1e-3)

    def _create_agent(rng):
        params = network.init(
            rng,
            init_inputs,
            ActorCritic.initialize_carry((2,)),
        )
        return TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optimizer,
        )

    agents = jax.vmap(_create_agent)(jax.random.split(jax.random.PRNGKey(3), 2))
    rollout = collect_ensemble_trajectories(
        jax.random.split(jax.random.PRNGKey(4), 2),
        env,
        env_params,
        agents,
        init_obs,
        init_env_state,
        num_levels=2,
        num_steps=3,
        max_width=7,
    )
    return base_env, agents, rollout


def _make_virtual_learning_fixture():
    base_env, agents, rollout = _make_rollout_fixture()
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params.replace(max_steps_in_episode=2)
    levels = jax.vmap(make_level_generator(7, 7, 5))(
        jax.random.split(jax.random.PRNGKey(1), 2)
    )
    init_obs, init_env_state = jax.vmap(
        env.reset_to_level,
        in_axes=(0, 0, None),
    )(
        jax.random.split(jax.random.PRNGKey(2), 2),
        levels,
        env_params,
    )
    ppo = PPOParameters(
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=1,
        num_epochs=1,
        clip_eps=0.2,
        entropy_coeff=1e-3,
        critic_coeff=0.5,
    )
    return base_env, env, env_params, agents, rollout, init_obs, init_env_state, ppo


def test_pose_encoding_excludes_elapsed_time() -> None:
    """The probe identity should contain only x, y, and direction."""
    base_env = Maze(max_height=7, max_width=7)
    env = AutoReplayWrapper(base_env)
    level = make_level_generator(7, 7, 5)(jax.random.PRNGKey(0))
    _, state = env.reset_to_level(jax.random.PRNGKey(1), level, env.default_params)
    first = state.replace(
        env_state=state.env_state.replace(
            agent_pos=jnp.array([3, 2]),
            agent_dir=jnp.asarray(1),
            time=jnp.asarray(0),
        )
    )
    later = first.replace(env_state=first.env_state.replace(time=jnp.asarray(99)))

    assert encode_maze_state(first, max_width=7) == 4 * (2 * 7 + 3) + 1
    assert encode_maze_state(first, max_width=7) == encode_maze_state(
        later,
        max_width=7,
    )


def test_members_share_start_but_have_independent_parameters_and_rollout_keys() -> None:
    """Initial physical states are shared while policy parameters differ."""
    _, agents, rollout = _make_rollout_fixture()
    parameter_leaves = jax.tree_util.tree_leaves(agents.params)

    assert any(
        not jnp.array_equal(leaf[0], leaf[1])
        for leaf in parameter_leaves
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    assert not jnp.array_equal(rollout.rngs[0], rollout.rngs[1])
    assert jnp.array_equal(
        rollout.trajectory.state_ids[0, 0],
        rollout.trajectory.state_ids[1, 0],
    )
    for observation_leaf in jax.tree_util.tree_leaves(rollout.trajectory.observations):
        assert jnp.array_equal(observation_leaf[0, 0], observation_leaf[1, 0])


def test_recurrent_replay_exactly_reconstructs_pre_update_policy() -> None:
    """Stored observation/reset sequences are sufficient for this actor."""
    _, agents, rollout = _make_rollout_fixture()

    def _replay_member(agent, trajectory):
        return replay_action_probabilities(
            agent,
            trajectory.observations,
            trajectory.resets,
            ActorCritic.initialize_carry((2,)),
        )

    reconstructed = jax.vmap(_replay_member)(agents, rollout.trajectory)

    assert not rollout.trajectory.resets[:, 0].any()
    assert rollout.trajectory.resets[:, 1:].any()
    assert jnp.allclose(
        reconstructed,
        rollout.trajectory.action_probabilities,
        atol=1e-6,
    )


def test_virtual_update_scores_frozen_visited_states_without_mutating_agents() -> None:
    """Virtual PPO should change replayed probabilities but not persistent state."""
    base_env, agents, rollout = _make_rollout_fixture()
    params_before = jax.tree_util.tree_map(jnp.copy, agents.params)
    ppo = PPOParameters(
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=1,
        num_epochs=1,
        clip_eps=0.2,
        entropy_coeff=1e-3,
        critic_coeff=0.5,
    )
    virtual_rngs = jax.random.split(jax.random.PRNGKey(5), 4).reshape(2, 2, 2)

    scored = score_ensemble_rollout(
        virtual_rngs,
        agents,
        rollout.trajectory,
        rollout.last_values,
        ppo,
        num_states=base_env.max_height * base_env.max_width * 4,
    )

    assert scored.post_update_action_probabilities.shape == (2, 3, 2, 7)
    assert jnp.allclose(
        scored.post_update_action_probabilities.sum(axis=-1),
        1.0,
    )
    assert not jnp.allclose(
        scored.post_update_action_probabilities,
        rollout.trajectory.action_probabilities,
    )
    assert scored.disagreement.scores.shape == (2,)
    assert scored.disagreement.eligible_state_count.shape == (2,)
    for before, after in zip(
        jax.tree_util.tree_leaves(params_before),
        jax.tree_util.tree_leaves(agents.params),
    ):
        assert jnp.array_equal(before, after)


def test_one_phase_multi_phase_scorer_matches_existing_scorer() -> None:
    """One phase should retain the existing score under the same PPO keys."""
    (
        base_env,
        env,
        env_params,
        agents,
        rollout,
        init_obs,
        init_env_state,
        ppo,
    ) = _make_virtual_learning_fixture()
    virtual_rng = jax.random.PRNGKey(5)
    legacy = score_ensemble_rollout(
        jax.random.split(virtual_rng, 4).reshape(2, 2, 2),
        agents,
        rollout.trajectory,
        rollout.last_values,
        ppo,
        num_states=base_env.max_height * base_env.max_width * 4,
    )

    scored = score_multi_phase_ensemble_rollout(
        virtual_rng,
        env,
        env_params,
        agents,
        rollout.trajectory,
        rollout.last_values,
        init_obs,
        init_env_state,
        num_steps=3,
        max_width=7,
        num_states=base_env.max_height * base_env.max_width * 4,
        parameters=VirtualLearningParameters(ppo, 1, 1),
    )

    assert jnp.allclose(
        scored.disagreement.scores,
        legacy.disagreement.scores,
        atol=2e-6,
    )
    assert jnp.allclose(
        scored.disagreement.mean_uncertainty_before,
        legacy.disagreement.mean_uncertainty_before,
        atol=2e-6,
    )
    assert jnp.array_equal(
        scored.disagreement.eligible_state_count,
        legacy.disagreement.eligible_state_count,
    )


def test_virtual_phases_chain_clones_and_restart_from_same_state() -> None:
    """Each phase should advance its clone but restart the level and LSTM."""
    (
        _,
        env,
        env_params,
        agents,
        rollout,
        init_obs,
        init_env_state,
        ppo,
    ) = _make_virtual_learning_fixture()
    trajectory = jax.tree_util.tree_map(
        lambda value: value[:, :, 0],
        rollout.trajectory,
    )
    level_init_obs = jax.tree_util.tree_map(lambda value: value[0], init_obs)
    level_init_state = jax.tree_util.tree_map(
        lambda value: value[0],
        init_env_state,
    )
    phase_zero_rngs = jax.random.split(jax.random.PRNGKey(5), 4).reshape(2, 2, 2)[:, 0]
    params_before = jax.tree_util.tree_map(jnp.copy, agents.params)

    final_agents, probes = run_virtual_phases_for_level(
        jax.random.PRNGKey(9),
        phase_zero_rngs,
        agents,
        trajectory,
        rollout.last_values[:, 0],
        level_init_obs,
        level_init_state,
        env,
        env_params,
        num_steps=3,
        max_width=7,
        parameters=VirtualLearningParameters(ppo, 3, 1),
    )

    assert probes.state_ids.shape == (3, 2, 3)
    assert jnp.all(probes.state_ids[:, :, 0] == probes.state_ids[0, 0, 0])
    assert not jnp.array_equal(probes.state_ids[1], probes.state_ids[2])
    assert jnp.array_equal(final_agents.step, agents.step + 3)
    for before, after in zip(
        jax.tree_util.tree_leaves(params_before),
        jax.tree_util.tree_leaves(agents.params),
    ):
        assert jnp.array_equal(before, after)
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(agents.params),
            jax.tree_util.tree_leaves(final_agents.params),
        )
    )


def test_multi_phase_level_batching_preserves_scores_and_support() -> None:
    """Level batching should change throughput rather than score semantics."""
    (
        base_env,
        env,
        env_params,
        agents,
        rollout,
        init_obs,
        init_env_state,
        ppo,
    ) = _make_virtual_learning_fixture()

    def _score(batch_size):
        return score_multi_phase_ensemble_rollout(
            jax.random.PRNGKey(5),
            env,
            env_params,
            agents,
            rollout.trajectory,
            rollout.last_values,
            init_obs,
            init_env_state,
            num_steps=3,
            max_width=7,
            num_states=base_env.max_height * base_env.max_width * 4,
            parameters=VirtualLearningParameters(ppo, 3, batch_size),
        )

    serial = _score(1)
    batched = _score(2)

    for expected, actual in zip(
        jax.tree_util.tree_leaves(serial),
        jax.tree_util.tree_leaves(batched),
    ):
        if jnp.issubdtype(expected.dtype, jnp.inexact):
            assert jnp.allclose(actual, expected, atol=2e-6)
        else:
            assert jnp.array_equal(actual, expected)
    assert jnp.any(serial.diagnostics.new_state_count_by_phase[:, 1:] > 0)
    assert jnp.all(jnp.isfinite(serial.diagnostics.policy_kl))


def test_old_checkpoint_config_receives_virtual_learning_defaults(tmp_path) -> None:
    """Evaluation should remain compatible with one-phase ensemble checkpoints."""
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "config.json").write_text(
        '{"checkpoint_format": "' + CHECKPOINT_FORMAT + '", "epoch_ppo": 4}',
        encoding="utf-8",
    )
    requested = {
        "mode": "eval",
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_to_eval": -1,
        "eval_num_attempts": 2,
    }

    restored = _load_ensemble_eval_config(requested)

    assert restored["virtual_rollout_phases"] == 1
    assert restored["virtual_epoch_ppo"] == 4
    assert restored["virtual_level_batch_size"] == 1


def test_parser_defaults_to_three_virtual_phases() -> None:
    """The public CLI should expose independent virtual-learning controls."""
    config = vars(build_parser().parse_args([]))

    assert config["virtual_rollout_phases"] == 3
    assert config["virtual_epoch_ppo"] == 5
    assert config["virtual_level_batch_size"] == 32


@pytest.mark.parametrize(
    "field",
    (
        "virtual_rollout_phases",
        "virtual_epoch_ppo",
        "virtual_level_batch_size",
    ),
)
def test_invalid_virtual_controls_fail_before_training(field: str) -> None:
    """Invalid static controls should be rejected before W&B or JIT setup."""
    config = vars(build_parser().parse_args([]))
    config[field] = 0

    with pytest.raises(ValueError):
        run_ensemble(config)


def test_persistent_members_train_only_on_their_own_trajectories() -> None:
    """Changing member 1 data must not alter member 0's PPO result."""
    _, agents, rollout = _make_rollout_fixture()
    ppo = PPOParameters(
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=1,
        num_epochs=1,
        clip_eps=0.2,
        entropy_coeff=1e-3,
        critic_coeff=0.5,
    )
    update_rngs = jax.random.split(jax.random.PRNGKey(6), 2)
    updated, _ = update_ensemble_members(
        update_rngs,
        agents,
        rollout.trajectory,
        rollout.last_values,
        ppo,
        update_grad=True,
    )

    changed_trajectory = rollout.trajectory._replace(
        rewards=rollout.trajectory.rewards.at[1].add(
            jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        )
    )
    changed, _ = update_ensemble_members(
        update_rngs,
        agents,
        changed_trajectory,
        rollout.last_values,
        ppo,
        update_grad=True,
    )

    for baseline, perturbed in zip(
        jax.tree_util.tree_leaves(updated.params),
        jax.tree_util.tree_leaves(changed.params),
    ):
        assert jnp.array_equal(baseline[0], perturbed[0])
    assert any(
        not jnp.array_equal(baseline[1], perturbed[1])
        for baseline, perturbed in zip(
            jax.tree_util.tree_leaves(updated.params),
            jax.tree_util.tree_leaves(changed.params),
        )
    )
