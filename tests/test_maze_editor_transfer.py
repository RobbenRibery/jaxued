"""Focused tests for fixed editor-target transfer scoring in Maze PLR."""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from examples.maze_plr import (
    ActorCritic,
    DEFAULT_EVAL_FREQ,
    DEFAULT_EXPLORATORY_GRAD_UPDATES,
    DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    DEFAULT_USE_ACCEL,
    EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    EDITOR_TRANSFER_SCORE_FUNCTION,
    EDITOR_TRANSFER_STAT_METRIC_KEYS,
    TrainState,
    _agent_log_metrics,
    _level_match_mask,
    aggregate_editor_transfer_interval_metrics,
    aggregate_replay_transfer_updates,
    count_transfer_target_duplicates,
    compute_gae,
    compute_configured_editor_transfer_scores,
    compute_editor_log_relative_transfer_scores,
    compute_editor_transfer_scores,
    compute_log_relative_transfer_gains,
    compute_plr_bank_score_diagnostics,
    editor_transfer_interval_log_dict,
    generate_transfer_target_bank,
    resolve_transfer_target_banks,
    sample_trajectories_rnn,
    select_editor_transfer_states,
    train_state_to_log_dict,
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
    assert DEFAULT_TRANSFER_LOG_RELATIVE_TAU == pytest.approx(0.1)


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


def test_target_bank_duplicate_count_counts_repeated_targets_after_first() -> None:
    generator = make_level_generator(5, 5, 3)
    first = generator(jax.random.PRNGKey(51))
    second = generator(jax.random.PRNGKey(52))
    bank = _stack_levels(first, second, first, first, second)

    duplicate_count = count_transfer_target_duplicates(bank)

    assert int(duplicate_count) == 3


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
            "transfer_target_duplicate_count": jnp.array(0, dtype=jnp.int32),
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
            "transfer_target_duplicate_count": jnp.array(2, dtype=jnp.int32),
        },
    )
    assert int(stored_index) == 0

    candidates = _stack_levels(stored_source, new_source, new_source)
    _, resolved, duplicate_counts = resolve_transfer_target_banks(
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
    assert int(duplicate_counts[0]) == 2
    assert int(duplicate_counts[1]) == int(duplicate_counts[2])


def test_editor_transfer_score_remains_mean_raw_gain_with_raw_se() -> None:
    """The historical score name remains the absolute-delta control."""
    returns_before = jnp.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    returns_after = jnp.array([[1.0, 3.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0]])

    scores, diagnostics = compute_editor_transfer_scores(returns_before, returns_after)
    raw_gains = returns_after - returns_before
    expected_se = raw_gains.std(axis=1, ddof=1) / jnp.sqrt(4)

    assert jnp.allclose(scores, raw_gains.mean(axis=1))
    assert jnp.allclose(diagnostics["transfer_gain_mean"], raw_gains.mean())
    assert jnp.allclose(diagnostics["transfer_standard_error_mean"], expected_se.mean())


def test_log_relative_score_uses_transformed_gain_and_se() -> None:
    returns_before = jnp.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    returns_after = jnp.array([[1.0, 3.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0]])

    scores, diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
    )
    log_relative_gains = compute_log_relative_transfer_gains(
        returns_before, returns_after
    )
    expected_se = log_relative_gains.std(axis=1, ddof=1) / jnp.sqrt(4)

    assert jnp.allclose(scores, log_relative_gains.mean(axis=1))
    assert jnp.allclose(
        diagnostics["transfer_log_relative_gain_mean"], log_relative_gains.mean()
    )
    assert jnp.allclose(diagnostics["transfer_standard_error_mean"], expected_se.mean())


def test_log_relative_transfer_prioritizes_unlocking_from_zero() -> None:
    returns_before = jnp.array([[0.0], [0.8]], dtype=jnp.float32)
    returns_after = jnp.array([[0.1], [0.9]], dtype=jnp.float32)

    scores, _ = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
        log_relative_tau=0.1,
    )

    assert float(scores[0]) == pytest.approx(float(jnp.log(2.0)))
    assert float(scores[1]) == pytest.approx(float(jnp.log(1.0 / 0.9)))
    assert float(scores[0]) > float(scores[1])


def _returns_from_log_relative_gains(
    log_relative_gains: jnp.ndarray,
    *,
    baseline: float = 0.5,
    tau: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    returns_before = jnp.full_like(log_relative_gains, baseline)
    returns_after = (returns_before + tau) * jnp.exp(log_relative_gains) - tau
    return returns_before, returns_after


def test_transfer_score_distribution_metrics_match_linear_quantiles() -> None:
    scores = jnp.arange(32, dtype=jnp.float32) * 0.01
    log_relative_gains = jnp.stack((scores - 0.01, scores + 0.01), axis=1)
    returns_before, returns_after = _returns_from_log_relative_gains(
        log_relative_gains
    )
    _, diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
    )

    assert float(diagnostics["transfer_score_q10"]) == pytest.approx(0.031)
    assert float(diagnostics["transfer_score_q50"]) == pytest.approx(0.155)
    assert float(diagnostics["transfer_score_q90"]) == pytest.approx(0.279)
    assert float(diagnostics["transfer_top_score_gap"]) == pytest.approx(
        0.01, abs=1e-6
    )


def test_transfer_score_se_metrics_preserve_sign_and_strict_thresholds() -> None:
    score_se_ratios = jnp.array([-3.0, -1.0, 0.0, 1.5])
    scores = score_se_ratios * 0.01
    log_relative_gains = jnp.stack((scores - 0.01, scores + 0.01), axis=1)
    returns_before, returns_after = _returns_from_log_relative_gains(
        log_relative_gains
    )
    _, diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
    )

    assert float(diagnostics["transfer_score_se_ratio_mean"]) == pytest.approx(
        -0.625, abs=1e-5
    )
    assert float(diagnostics["transfer_score_se_abs_ratio_q50"]) == pytest.approx(
        1.25, abs=1e-5
    )
    assert float(diagnostics["transfer_score_se_abs_ratio_q90"]) == pytest.approx(
        2.55, abs=1e-5
    )
    assert float(
        diagnostics["transfer_score_se_positive_gt_1_fraction"]
    ) == pytest.approx(0.25)
    assert float(
        diagnostics["transfer_score_se_positive_gt_2_fraction"]
    ) == pytest.approx(0.0)


def test_zero_score_and_zero_se_produce_finite_zero_ratio() -> None:
    returns = jnp.zeros((1, 1), dtype=jnp.float32)
    _, diagnostics = compute_editor_log_relative_transfer_scores(returns, returns)

    assert float(diagnostics["transfer_score_se_ratio_mean"]) == 0.0
    assert float(diagnostics["transfer_score_se_abs_ratio_q50"]) == 0.0
    assert jnp.isfinite(diagnostics["transfer_score_se_ratio_mean"])


def test_configured_transfer_dispatch_keeps_score_names_distinct() -> None:
    returns_before = jnp.array([[0.0], [0.8]], dtype=jnp.float32)
    returns_after = jnp.array([[0.1], [0.9]], dtype=jnp.float32)

    raw_scores, _ = compute_configured_editor_transfer_scores(
        EDITOR_TRANSFER_SCORE_FUNCTION,
        returns_before,
        returns_after,
    )
    relative_scores, _ = compute_configured_editor_transfer_scores(
        EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
        returns_before,
        returns_after,
    )

    assert jnp.allclose(raw_scores, jnp.array([0.1, 0.1]))
    assert float(relative_scores[0]) > float(relative_scores[1])


def _transfer_interval_metrics(is_new: jnp.ndarray) -> dict[str, jnp.ndarray]:
    metrics = {
        metric_key: jnp.array([1.0, 3.0], dtype=jnp.float32)
        for metric_key in EDITOR_TRANSFER_STAT_METRIC_KEYS
    }
    metrics.update(
        {
            "transfer_attached_bank_count": jnp.array([32, 32]),
            "transfer_eval_env_steps": jnp.array([5, 7]),
            "transfer_virtual_optimizer_steps": jnp.array([2, 0]),
            "transfer_update_is_virtual": jnp.array([1.0, 0.0]),
            "transfer_update_is_new": is_new,
            "unrelated_metric": jnp.array([10.0, 20.0]),
        }
    )
    return metrics


def test_transfer_interval_metrics_split_new_and_replay_on_device() -> None:
    aggregated = jax.jit(aggregate_editor_transfer_interval_metrics)(
        _transfer_interval_metrics(jnp.array([1.0, 0.0]))
    )

    assert float(aggregated["transfer_score_q50"]) == pytest.approx(2.0)
    assert float(aggregated["transfer_new_score_q50"]) == pytest.approx(1.0)
    assert float(aggregated["transfer_replay_score_q50"]) == pytest.approx(3.0)
    assert int(aggregated["transfer_new_update_count_interval"]) == 1
    assert int(aggregated["transfer_replay_update_count_interval"]) == 1
    assert float(aggregated["transfer_new_update_fraction"]) == pytest.approx(0.5)
    assert float(aggregated["transfer_update_is_virtual"]) == pytest.approx(0.5)
    assert int(aggregated["transfer_eval_env_steps"]) == 12
    assert int(aggregated["transfer_virtual_optimizer_steps"]) == 2
    assert jnp.array_equal(aggregated["unrelated_metric"], jnp.array([10.0, 20.0]))
    assert "transfer_update_is_new" not in aggregated


def test_transfer_interval_empty_branch_logs_zero_count_and_nan_statistics() -> None:
    aggregated = aggregate_editor_transfer_interval_metrics(
        _transfer_interval_metrics(jnp.ones(2))
    )

    assert int(aggregated["transfer_replay_update_count_interval"]) == 0
    assert float(aggregated["transfer_replay_update_fraction"]) == 0.0
    assert jnp.isnan(aggregated["transfer_replay_score_q50"])


def test_plr_bank_score_metrics_ignore_unpopulated_slots() -> None:
    diagnostics = compute_plr_bank_score_diagnostics(
        scores=jnp.array([1.0, 3.0, -999.0, 100.0]),
        size=jnp.array(2),
    )

    assert float(diagnostics["score_q10"]) == pytest.approx(1.2)
    assert float(diagnostics["score_q50"]) == pytest.approx(2.0)
    assert float(diagnostics["score_q90"]) == pytest.approx(2.8)
    assert float(diagnostics["top_score_gap"]) == pytest.approx(2.0)


def test_plr_bank_score_metrics_handle_empty_and_single_entry_banks() -> None:
    scores = jnp.array([4.0, -jnp.inf, -jnp.inf])
    empty = compute_plr_bank_score_diagnostics(scores, jnp.array(0))
    single = compute_plr_bank_score_diagnostics(scores, jnp.array(1))

    assert all(jnp.isnan(value) for value in empty.values())
    assert float(single["score_q10"]) == 4.0
    assert float(single["score_q50"]) == 4.0
    assert float(single["score_q90"]) == 4.0
    assert jnp.isnan(single["top_score_gap"])


def test_level_sampler_log_dict_exposes_bank_score_diagnostics() -> None:
    source = make_level_generator(5, 5, 3)(jax.random.PRNGKey(15))
    sampler_api = LevelSampler(capacity=4)
    sampler = sampler_api.initialize(source, {"max_return": -jnp.inf})
    sampler, _ = sampler_api.insert(
        sampler,
        source,
        score=jnp.array(4.0),
        level_extra={"max_return": jnp.array(0.0)},
    )
    state = SimpleNamespace(
        sampler=sampler,
        num_dr_updates=1,
        num_replay_updates=0,
        num_mutation_updates=0,
    )

    log = train_state_to_log_dict(state, sampler_api)["log"]

    assert float(log["level_sampler/score_q10"]) == 4.0
    assert float(log["level_sampler/score_q50"]) == 4.0
    assert float(log["level_sampler/score_q90"]) == 4.0
    assert jnp.isnan(log["level_sampler/top_score_gap"])


def test_level_sampler_log_dict_reduces_stored_target_duplicate_scalars() -> None:
    source = make_level_generator(5, 5, 3)(jax.random.PRNGKey(53))
    target_bank = jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(jnp.asarray(leaf)[None, ...], 4, axis=0), source
    )
    sampler_api = LevelSampler(capacity=4)
    sampler = sampler_api.initialize(
        source,
        {
            "max_return": -jnp.inf,
            "transfer_targets": target_bank,
            "has_transfer_targets": jnp.array(False),
            "transfer_target_duplicate_count": jnp.array(0, dtype=jnp.int32),
        },
    )
    for index, duplicate_count in enumerate((1, 3)):
        sampler, _ = sampler_api.insert(
            sampler,
            source.replace(agent_dir=source.agent_dir + index),
            score=jnp.array(4.0),
            level_extra={
                "max_return": jnp.array(0.0),
                "transfer_targets": target_bank,
                "has_transfer_targets": jnp.array(True),
                "transfer_target_duplicate_count": jnp.array(
                    duplicate_count, dtype=jnp.int32
                ),
            },
        )
    state = SimpleNamespace(
        sampler=sampler,
        num_dr_updates=1,
        num_replay_updates=0,
        num_mutation_updates=0,
    )

    log = train_state_to_log_dict(state, sampler_api)["log"]

    assert int(log["level_sampler/transfer_target_bank_count"]) == 2
    assert int(log["level_sampler/transfer_target_duplicate_count"]) == 4


def test_editor_transfer_interval_log_dict_emits_all_diagnostic_keys() -> None:
    stats = aggregate_editor_transfer_interval_metrics(
        _transfer_interval_metrics(jnp.array([1.0, 0.0]))
    )
    log = editor_transfer_interval_log_dict(stats)

    expected_suffixes = {
        metric_key[len("transfer_") :]
        for metric_key in EDITOR_TRANSFER_STAT_METRIC_KEYS
    }
    for metric_suffix in expected_suffixes:
        assert f"transfer/{metric_suffix}" in log
        assert f"transfer/new/{metric_suffix}" in log
        assert f"transfer/replay/{metric_suffix}" in log
    assert "transfer/new/update_count_interval" in log
    assert "transfer/new/update_fraction" in log
    assert "transfer/replay/update_count_interval" in log
    assert "transfer/replay/update_fraction" in log


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


@pytest.mark.parametrize(
    "score_function",
    [
        EDITOR_TRANSFER_SCORE_FUNCTION,
        EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "transfer_target_count",
        "transfer_num_edits",
        "transfer_log_relative_tau",
    ],
)
def test_editor_transfer_config_requires_positive_controls(
    score_function: str,
    field: str,
) -> None:
    config = {
        "score_function": score_function,
        "transfer_target_count": 64,
        "transfer_num_edits": 8,
        "transfer_log_relative_tau": 0.1,
        "use_accel": False,
    }
    config[field] = 0

    with pytest.raises(ValueError):
        validate_editor_transfer_config(config)


@pytest.mark.parametrize(
    "score_function",
    [
        EDITOR_TRANSFER_SCORE_FUNCTION,
        EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    ],
)
def test_editor_transfer_rejects_accel(score_function: str) -> None:
    with pytest.raises(ValueError, match="does not support --use_accel"):
        validate_editor_transfer_config(
            {
                "score_function": score_function,
                "transfer_target_count": 64,
                "transfer_num_edits": 8,
                "use_accel": True,
            }
        )
