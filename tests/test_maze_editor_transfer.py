"""Focused tests for fixed editor-target transfer scoring in Maze PLR."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from examples.maze_plr import (
    ActorCritic,
    DEFAULT_EVAL_FREQ,
    DEFAULT_EXPLORATORY_GRAD_UPDATES,
    DEFAULT_USE_ACCEL,
    TrainState,
    _agent_log_metrics,
    _level_match_mask,
    aggregate_replay_transfer_updates,
    compute_gae,
    compute_editor_transfer_scores,
    generate_transfer_target_bank,
    resolve_transfer_target_banks,
    sample_trajectories_rnn,
    select_editor_transfer_states,
    update_actor_critic_rnn,
    validate_editor_transfer_config,
)
from jaxued.environments.maze import (
    Maze,
    make_level_generator,
    make_level_mutator_minimax,
)
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper
import optax


def test_editor_transfer_uses_modular_metrics_evaluation_cadence() -> None:
    assert DEFAULT_EVAL_FREQ == 200


def test_editor_transfer_defaults_to_robust_plr_protocol() -> None:
    assert DEFAULT_EXPLORATORY_GRAD_UPDATES is False
    assert DEFAULT_USE_ACCEL is False


def test_editor_transfer_logs_all_agent_metrics() -> None:
    losses = (
        jnp.array([1.0, 2.0]),
        (
            jnp.array([0.1, 0.2]),
            jnp.array([0.3, 0.4]),
            jnp.array([0.2, 0.4]),
        ),
    )

    metrics = _agent_log_metrics(losses)

    assert set(metrics) == {
        "agent/loss",
        "agent/value_loss",
        "agent/policy_loss",
        "agent/entropy",
    }
    assert float(metrics["agent/loss"]) == pytest.approx(1.5)
    assert float(metrics["agent/value_loss"]) == pytest.approx(0.15)
    assert float(metrics["agent/policy_loss"]) == pytest.approx(0.35)
    assert float(metrics["agent/entropy"]) == pytest.approx(0.3)


def _stack_levels(*levels):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *levels)


def _tree_equal(left, right) -> bool:
    comparisons = jax.tree_util.tree_map(
        lambda x, y: jnp.array_equal(x, y), left, right
    )
    return all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))


def test_transfer_target_bank_uses_configured_chain_length() -> None:
    """Every independently sampled chain should receive the configured length."""
    source = make_level_generator(5, 5, 3)(jax.random.PRNGKey(0))

    def counting_mutator(rng, level, num_edits):
        del rng
        return level.replace(agent_dir=level.agent_dir + num_edits)

    targets = generate_transfer_target_bank(
        rng=jax.random.PRNGKey(1),
        source_level=source,
        mutate_level=counting_mutator,
        target_count=64,
        num_edits=8,
    )

    assert targets.wall_map.shape[0] == 64
    assert jnp.all(targets.agent_dir == source.agent_dir + 8)


def test_transfer_minimax_mutator_disables_single_step_no_op() -> None:
    """A one-editor transfer chain must always change the source level."""
    source = make_level_generator(5, 5, 3)(jax.random.PRNGKey(2))
    transfer_mutator = make_level_mutator_minimax(1, allow_no_op=False)
    targets = jax.vmap(transfer_mutator, in_axes=(0, None, None))(
        jax.random.split(jax.random.PRNGKey(3), 64),
        source,
        1,
    )

    assert not bool(_level_match_mask(targets, source).any())


def test_minimax_mutator_keeps_allow_no_op_backward_compatible_default() -> None:
    parameter = inspect.signature(make_level_mutator_minimax).parameters["allow_no_op"]
    assert parameter.default is True


def test_target_bank_generation_is_deterministic_and_keeps_all_draws() -> None:
    """A fixed key should reproduce all raw chain results without deduplication."""
    source = make_level_generator(5, 5, 3)(jax.random.PRNGKey(4))
    transfer_mutator = make_level_mutator_minimax(3, allow_no_op=False)

    first = generate_transfer_target_bank(
        jax.random.PRNGKey(5), source, transfer_mutator, target_count=64, num_edits=3
    )
    second = generate_transfer_target_bank(
        jax.random.PRNGKey(5), source, transfer_mutator, target_count=64, num_edits=3
    )

    assert first.wall_map.shape[0] == 64
    assert _tree_equal(first, second)


def test_resolver_reuses_stored_and_first_in_batch_target_banks() -> None:
    """Stored and within-batch source duplicates should not receive new banks."""
    generator = make_level_generator(5, 5, 3)
    stored_source = generator(jax.random.PRNGKey(6))
    new_source = generator(jax.random.PRNGKey(7))
    transfer_mutator = make_level_mutator_minimax(2, allow_no_op=False)
    placeholder_bank = jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(jnp.asarray(leaf)[None, ...], 4, axis=0),
        stored_source,
    )
    sampler_api = LevelSampler(capacity=4, duplicate_check=True)
    sampler = sampler_api.initialize(
        stored_source,
        {
            "max_return": -jnp.inf,
            "transfer_targets": placeholder_bank,
            "has_transfer_targets": jnp.array(False),
        },
    )
    stored_bank = generate_transfer_target_bank(
        jax.random.PRNGKey(8),
        stored_source,
        transfer_mutator,
        target_count=4,
        num_edits=2,
    )
    sampler, stored_index = sampler_api.insert(
        sampler,
        stored_source,
        score=jnp.array(1.0),
        level_extra={
            "max_return": jnp.array(0.0),
            "transfer_targets": stored_bank,
            "has_transfer_targets": jnp.array(True),
        },
    )
    assert int(stored_index) == 0

    candidates = _stack_levels(stored_source, new_source, new_source)
    _, resolved = resolve_transfer_target_banks(
        rng=jax.random.PRNGKey(9),
        sampler=sampler,
        source_levels=candidates,
        mutate_level=transfer_mutator,
        target_count=4,
        num_edits=2,
    )

    assert _tree_equal(
        jax.tree_util.tree_map(lambda leaf: leaf[0], resolved), stored_bank
    )
    assert _tree_equal(
        jax.tree_util.tree_map(lambda leaf: leaf[1], resolved),
        jax.tree_util.tree_map(lambda leaf: leaf[2], resolved),
    )


def test_transfer_score_is_plain_mean_gain_with_diagnostic_se() -> None:
    """PLR scores must not include a variance or standard-error penalty."""
    returns_before = jnp.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    returns_after = jnp.array([[1.0, 3.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0]])

    scores, diagnostics = compute_editor_transfer_scores(returns_before, returns_after)
    gains = returns_after - returns_before
    expected_se = gains.std(axis=1, ddof=1) / jnp.sqrt(4)

    assert jnp.allclose(scores, gains.mean(axis=1))
    assert jnp.allclose(diagnostics["transfer_standard_error_mean"], expected_se.mean())


def test_repeated_replay_indices_receive_order_independent_aggregates() -> None:
    """Repeated PLR indices should share mean scores and maximum returns."""
    scores, max_returns = aggregate_replay_transfer_updates(
        level_indices=jnp.array([2, 5, 2, 9]),
        scores=jnp.array([1.0, 4.0, 3.0, -1.0]),
        max_returns=jnp.array([1.0, 7.0, 2.0, 3.0]),
    )

    assert jnp.allclose(scores, jnp.array([2.0, 4.0, 2.0, -1.0]))
    assert jnp.allclose(max_returns, jnp.array([2.0, 7.0, 2.0, 3.0]))


def test_virtual_and_persistent_modes_differ_only_in_continuing_state() -> None:
    """Both modes score with the updated state, but robust mode discards it."""
    original = {"params": jnp.array(1.0), "optimizer_step": jnp.array(0)}
    updated = {"params": jnp.array(2.0), "optimizer_step": jnp.array(1)}

    persistent, scoring = select_editor_transfer_states(original, updated, True)
    assert persistent is updated
    assert scoring is updated

    persistent, scoring = select_editor_transfer_states(original, updated, False)
    assert persistent is original
    assert scoring is updated


def test_full_batch_ppo_state_is_scored_then_persisted_or_discarded() -> None:
    """One real PPO computation should back both exploratory mode semantics."""
    num_envs = 2
    num_steps = 2
    raw_env = Maze(
        max_height=5,
        max_width=5,
        agent_view_size=3,
        normalize_obs=True,
    )
    env = AutoReplayWrapper(raw_env)
    env_params = env.default_params
    generator = make_level_generator(5, 5, 3)
    levels = jax.vmap(generator)(jax.random.split(jax.random.PRNGKey(10), num_envs))
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(jax.random.PRNGKey(11), num_envs),
        levels,
        env_params,
    )
    init_hstate = ActorCritic.initialize_carry((num_envs,))
    init_network_obs = jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(leaf[None, ...], num_steps, axis=0), init_obs
    )
    network = ActorCritic(env.action_space(env_params).n)
    params = network.init(
        jax.random.PRNGKey(12),
        (init_network_obs, jnp.zeros((num_steps, num_envs), dtype=jnp.bool_)),
        init_hstate,
    )
    original_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.adam(1e-3),
        sampler={},
        update_state=0,
        num_dr_updates=0,
        num_replay_updates=0,
        num_mutation_updates=0,
        dr_last_level_batch=levels,
        replay_last_level_batch=levels,
        mutation_last_level_batch=levels,
    )
    (
        (_, rollout_state, _, _, _, last_value),
        (obs, actions, rewards, dones, log_probs, values, _),
    ) = sample_trajectories_rnn(
        rng=jax.random.PRNGKey(13),
        env=env,
        env_params=env_params,
        train_state=original_state,
        init_hstate=init_hstate,
        init_obs=init_obs,
        init_env_state=init_env_state,
        num_envs=num_envs,
        max_episode_length=num_steps,
    )
    advantages, targets = compute_gae(
        gamma=0.99,
        lambd=0.95,
        last_value=last_value,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    (_, updated_state), _ = update_actor_critic_rnn(
        rng=jax.random.PRNGKey(14),
        train_state=rollout_state,
        init_hstate=init_hstate,
        batch=(obs, actions, dones, log_probs, values, targets, advantages),
        num_envs=num_envs,
        n_steps=num_steps,
        n_minibatch=1,
        n_epochs=1,
        clip_eps=0.2,
        entropy_coeff=1e-3,
        critic_coeff=0.5,
        update_grad=True,
    )

    robust_persistent, robust_scoring = select_editor_transfer_states(
        rollout_state, updated_state, persist_update=False
    )
    exploratory_persistent, exploratory_scoring = select_editor_transfer_states(
        rollout_state, updated_state, persist_update=True
    )

    assert int(updated_state.step) == int(rollout_state.step) + 1
    assert robust_persistent is rollout_state
    assert robust_scoring is updated_state
    assert exploratory_persistent is updated_state
    assert exploratory_scoring is updated_state


@pytest.mark.parametrize("field", ["transfer_target_count", "transfer_num_edits"])
def test_editor_transfer_config_requires_positive_static_sizes(field: str) -> None:
    config = {
        "score_function": "editor_transfer",
        "transfer_target_count": 64,
        "transfer_num_edits": 8,
        "use_accel": False,
    }
    config[field] = 0

    with pytest.raises(ValueError):
        validate_editor_transfer_config(config)


def test_editor_transfer_rejects_accel() -> None:
    with pytest.raises(ValueError, match="does not support --use_accel"):
        validate_editor_transfer_config(
            {
                "score_function": "editor_transfer",
                "transfer_target_count": 64,
                "transfer_num_edits": 8,
                "use_accel": True,
            }
        )
