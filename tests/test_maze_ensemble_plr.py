"""Integration tests for the Maze ensemble rollout and virtual update."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from examples.maze_ensemble_plr import (
    ActorCritic,
    PPOParameters,
    collect_ensemble_trajectories,
    encode_maze_state,
    replay_action_probabilities,
    score_ensemble_rollout,
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
    levels = jax.vmap(level_generator)(
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

    agents = jax.vmap(_create_agent)(
        jax.random.split(jax.random.PRNGKey(3), 2)
    )
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
    for observation_leaf in jax.tree_util.tree_leaves(
        rollout.trajectory.observations
    ):
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
