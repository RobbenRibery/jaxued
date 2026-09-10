"""Focused tests for fixed editor-target transfer scoring in Maze PLR."""

import inspect
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from examples.maze_plr import (
    ActorCritic,
    DEFAULT_EVAL_FREQ,
    DEFAULT_EXPLORATORY_GRAD_UPDATES,
    DEFAULT_TRANSFER_LOG_RELATIVE_TAU,
    DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA,
    DEFAULT_TRANSFER_SOLVED_PRIOR_BETA,
    DEFAULT_USE_ACCEL,
    EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    EDITOR_TRANSFER_SCORE_FUNCTION,
    EDITOR_TRANSFER_STAT_METRIC_KEYS,
    TrainState,
    _agent_log_metrics,
    _level_match_mask,
    aggregate_editor_transfer_interval_metrics,
    aggregate_replay_transfer_updates,
    beta_one_upper_confidence_quantile,
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
    resolve_new_solved_informed_history,
    sample_trajectories_rnn,
    select_editor_transfer_states,
    compute_solved_informed_transfer_update,
    solved_informed_transfer_diagnostics,
    train_state_to_log_dict,
    update_actor_critic_rnn,
    validate_editor_transfer_config,
    validate_log_relative_tau,
    validate_solved_informed_numerics,
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


def test_new_solved_history_inherits_resident_entry_and_groups_duplicates() -> None:
    generator = make_level_generator(5, 5, 3)
    resident = generator(jax.random.PRNGKey(61))
    fresh = generator(jax.random.PRNGKey(62))
    sampler_api = LevelSampler(capacity=4, duplicate_check=True)
    sampler = sampler_api.initialize(
        resident,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
        },
    )
    sampler, resident_idx = sampler_api.insert(
        sampler,
        resident,
        score=jnp.array(1.0),
        level_extra={
            "max_return": jnp.array(0.0),
            "transfer_unsolved_failure_count": jnp.array(7, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
        },
    )
    candidates = _stack_levels(resident, fresh, fresh)

    group_ids, failures, solved, max_returns = resolve_new_solved_informed_history(
        sampler,
        candidates,
        duplicate_check=True,
    )

    assert int(resident_idx) == 0
    assert int(group_ids[0]) == 0
    assert int(group_ids[1]) == int(group_ids[2])
    assert int(group_ids[1]) >= sampler_api.capacity
    assert jnp.array_equal(failures, jnp.array([7, 0, 0]))
    assert not bool(solved.any())
    assert float(max_returns[0]) == 0.0
    assert jnp.isneginf(max_returns[1:]).all()


def test_new_solved_history_is_independent_when_duplicate_check_is_disabled() -> None:
    generator = make_level_generator(5, 5, 3)
    level = generator(jax.random.PRNGKey(63))
    sampler_api = LevelSampler(capacity=4, duplicate_check=False)
    sampler = sampler_api.initialize(
        level,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
        },
    )
    candidates = _stack_levels(level, level)

    group_ids, failures, solved, max_returns = resolve_new_solved_informed_history(
        sampler,
        candidates,
        duplicate_check=False,
    )

    assert int(group_ids[0]) != int(group_ids[1])
    assert jnp.array_equal(failures, jnp.zeros(2, dtype=jnp.int32))
    assert not bool(solved.any())
    assert jnp.isneginf(max_returns).all()


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


def test_beta_one_upper_confidence_matches_analytic_quantile() -> None:
    failure_counts = jnp.array([0.0, 1.0, 10.0, 100.0])
    posterior_beta = DEFAULT_TRANSFER_SOLVED_PRIOR_BETA + failure_counts
    quantiles = beta_one_upper_confidence_quantile(
        beta=posterior_beta,
        confidence=DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    )
    expected = 1 - (1 - DEFAULT_TRANSFER_SOLVED_CONFIDENCE) ** (1 / posterior_beta)

    assert jnp.allclose(quantiles, expected, atol=2e-6)
    assert float(quantiles[0]) == pytest.approx(0.8, abs=2e-6)
    assert jnp.all(jnp.diff(quantiles) < 0)


def test_beta_one_upper_confidence_is_jittable_and_finite_for_large_counts() -> None:
    quantile_fn = jax.jit(beta_one_upper_confidence_quantile)
    quantiles = quantile_fn(
        jnp.array([2.0, 11.0, 1_001.0, 1_000_001.0]),
        DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    )

    assert jnp.all(jnp.isfinite(quantiles))
    assert jnp.all((quantiles >= 0) & (quantiles <= 1))


def test_beta_one_upper_confidence_is_monotone_through_960000_failures() -> None:
    failure_counts = jnp.arange(960_001, dtype=jnp.float32)
    posterior_beta = DEFAULT_TRANSFER_SOLVED_PRIOR_BETA + failure_counts
    quantiles = jax.jit(beta_one_upper_confidence_quantile)(
        posterior_beta,
        DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
    )
    penalties = jnp.log(quantiles)

    assert jnp.all(jnp.isfinite(quantiles))
    assert jnp.all(quantiles > 0)
    assert jnp.all(jnp.diff(quantiles) <= 0)
    assert jnp.all(jnp.diff(penalties) <= 0)


def _solved_informed_update(
    *,
    group_ids: jnp.ndarray,
    base_scores: jnp.ndarray,
    current_solved: jnp.ndarray,
    prior_failure_counts: jnp.ndarray | None = None,
    prior_ever_solved: jnp.ndarray | None = None,
):
    count = group_ids.shape[0]
    if prior_failure_counts is None:
        prior_failure_counts = jnp.zeros(count, dtype=jnp.int32)
    if prior_ever_solved is None:
        prior_ever_solved = jnp.zeros(count, dtype=jnp.bool_)
    return compute_solved_informed_transfer_update(
        group_ids=group_ids,
        base_scores=base_scores,
        current_solved=current_solved,
        prior_failure_counts=prior_failure_counts,
        prior_ever_solved=prior_ever_solved,
        current_max_returns=jnp.where(current_solved, 1.0, 0.0),
        prior_max_returns=jnp.full(count, -jnp.inf),
    )


def test_first_failure_applies_documented_finite_penalty() -> None:
    update = _solved_informed_update(
        group_ids=jnp.array([0]),
        base_scores=jnp.array([0.02]),
        current_solved=jnp.array([False]),
    )
    expected_confidence = 1 - (
        1 - DEFAULT_TRANSFER_SOLVED_CONFIDENCE
    ) ** 0.5
    expected_penalty = jnp.log(expected_confidence)

    assert int(update.failure_counts[0]) == 1
    assert float(update.confidences[0]) == pytest.approx(expected_confidence, abs=2e-6)
    assert float(update.penalties[0]) == pytest.approx(
        float(expected_penalty), abs=2e-6
    )
    assert float(update.scores[0]) == pytest.approx(
        0.02 + float(expected_penalty), abs=2e-6
    )


def test_log_q_enforces_solved_order_at_default_confidence_boundary() -> None:
    best_unsolved_base, _ = compute_editor_log_relative_transfer_scores(
        returns_before=jnp.array([[0.0]]),
        returns_after=jnp.array([[1.0]]),
    )
    worst_solved_base, _ = compute_editor_log_relative_transfer_scores(
        returns_before=jnp.array([[1.0]]),
        returns_after=jnp.array([[0.0]]),
    )
    base_scores = jnp.concatenate((best_unsolved_base, worst_solved_base))
    current_solved = jnp.array([False, True])

    before_boundary = _solved_informed_update(
        group_ids=jnp.array([0, 1]),
        base_scores=base_scores,
        current_solved=current_solved,
        prior_failure_counts=jnp.array([191, 0]),
    )
    after_boundary = _solved_informed_update(
        group_ids=jnp.array([0, 1]),
        base_scores=base_scores,
        current_solved=current_solved,
        prior_failure_counts=jnp.array([192, 0]),
    )

    assert int(before_boundary.failure_counts[0]) == 192
    assert float(before_boundary.confidences[0]) > 1 / 121
    assert float(before_boundary.scores[0]) > float(before_boundary.scores[1])
    assert int(after_boundary.failure_counts[0]) == 193
    assert float(after_boundary.confidences[0]) < 1 / 121
    assert float(after_boundary.scores[0]) < float(after_boundary.scores[1])


def test_first_success_removes_penalty_and_freezes_future_evidence() -> None:
    solved = _solved_informed_update(
        group_ids=jnp.array([3]),
        base_scores=jnp.array([0.02]),
        current_solved=jnp.array([True]),
        prior_failure_counts=jnp.array([7]),
    )
    failed_later = _solved_informed_update(
        group_ids=jnp.array([3]),
        base_scores=jnp.array([-0.01]),
        current_solved=jnp.array([False]),
        prior_failure_counts=solved.failure_counts,
        prior_ever_solved=solved.ever_solved,
    )

    assert bool(solved.ever_solved[0])
    assert bool(solved.newly_solved[0])
    assert int(solved.failure_counts[0]) == 7
    assert float(solved.confidences[0]) == 1.0
    assert float(solved.penalties[0]) == 0.0
    assert float(solved.scores[0]) == pytest.approx(0.02)
    assert int(failed_later.failure_counts[0]) == 7
    assert bool(failed_later.ever_solved[0])
    assert float(failed_later.penalties[0]) == 0.0
    assert float(failed_later.scores[0]) == pytest.approx(-0.01)


def test_repeated_entry_failures_count_every_rollout_and_share_one_score() -> None:
    update = _solved_informed_update(
        group_ids=jnp.array([2, 5, 2, 2]),
        base_scores=jnp.array([0.01, 0.04, 0.03, 0.02]),
        current_solved=jnp.array([False, False, False, False]),
        prior_failure_counts=jnp.array([4, 1, 4, 4]),
    )

    assert jnp.array_equal(update.failure_counts, jnp.array([7, 2, 7, 7]))
    assert jnp.allclose(update.base_scores, jnp.array([0.02, 0.04, 0.02, 0.02]))
    assert jnp.allclose(update.scores[jnp.array([0, 2, 3])], update.scores[0])
    assert int(update.repeated_source_count) == 1
    assert int(update.max_source_multiplicity) == 3


def test_any_duplicate_success_latches_shared_entry_without_order_dependence() -> None:
    arguments = {
        "group_ids": jnp.array([8, 8, 8, 8]),
        "base_scores": jnp.array([0.01, 0.02, 0.03, 0.04]),
        "prior_failure_counts": jnp.array([5, 5, 5, 5]),
    }
    first = _solved_informed_update(
        **arguments,
        current_solved=jnp.array([False, True, False, False]),
    )
    second = _solved_informed_update(
        **arguments,
        current_solved=jnp.array([False, False, False, True]),
    )

    assert jnp.all(first.ever_solved)
    assert jnp.all(first.failure_counts == 5)
    assert jnp.all(first.confidences == 1)
    assert jnp.allclose(first.scores, 0.025)
    assert jnp.allclose(first.scores, second.scores)
    assert jnp.array_equal(first.failure_counts, second.failure_counts)


def test_duplicate_diagnostics_separate_rollout_and_unique_group_fractions() -> None:
    update = _solved_informed_update(
        group_ids=jnp.array([0, 0, 0, 1]),
        base_scores=jnp.zeros(4),
        current_solved=jnp.array([False, True, False, False]),
    )
    _, base_diagnostics = compute_editor_log_relative_transfer_scores(
        jnp.zeros((2, 4)),
        jnp.zeros((2, 4)),
    )
    diagnostics = solved_informed_transfer_diagnostics(base_diagnostics, update)

    assert float(diagnostics["transfer_current_solved_fraction"]) == pytest.approx(0.25)
    assert float(
        diagnostics["transfer_current_solved_group_fraction"]
    ) == pytest.approx(0.5)
    assert float(diagnostics["transfer_newly_solved_fraction"]) == pytest.approx(0.5)
    assert float(diagnostics["transfer_ever_solved_fraction"]) == pytest.approx(0.5)
    assert float(diagnostics["transfer_failure_increment_mean"]) == pytest.approx(0.5)


def test_solved_informed_diagnostics_separate_final_score_from_base_se() -> None:
    returns_before = jnp.array([[0.2, 0.4], [0.5, 0.5]])
    returns_after = jnp.array([[0.3, 0.5], [0.6, 0.4]])
    base_scores, base_diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before, returns_after
    )
    update = _solved_informed_update(
        group_ids=jnp.array([0, 1]),
        base_scores=base_scores,
        current_solved=jnp.array([False, True]),
    )
    diagnostics = solved_informed_transfer_diagnostics(base_diagnostics, update)

    assert jnp.isnan(diagnostics["transfer_score_se_ratio_mean"])
    assert jnp.isfinite(diagnostics["transfer_base_score_se_ratio_mean"])
    assert float(diagnostics["transfer_solved_confidence_mean"]) < 1
    assert float(diagnostics["transfer_ever_solved_fraction"]) == pytest.approx(0.5)


def test_solved_informed_diagnostics_use_duplicate_aggregated_base_scores() -> None:
    returns_before = jnp.zeros((3, 2))
    returns_after = jnp.array([[0.01, 0.01], [0.03, 0.03], [0.1, 0.1]])
    base_scores, base_diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
    )
    update = _solved_informed_update(
        group_ids=jnp.array([0, 0, 1]),
        base_scores=base_scores,
        current_solved=jnp.array([False, False, False]),
    )

    diagnostics = solved_informed_transfer_diagnostics(base_diagnostics, update)
    expected_base_q50 = jnp.sort(update.base_scores)[1]

    assert float(diagnostics["transfer_base_score_q50"]) == pytest.approx(
        float(expected_base_q50)
    )


def _solved_informed_level_extra(update, index: int) -> dict[str, jax.Array]:
    return {
        "max_return": update.max_returns[index],
        "transfer_unsolved_failure_count": update.failure_counts[index],
        "transfer_ever_solved": update.ever_solved[index],
        "transfer_solved_confidence": update.confidences[index],
        "transfer_base_log_relative_score": update.base_scores[index],
    }


def test_failed_level_has_finite_score_and_can_enter_plr_buffer() -> None:
    level = make_level_generator(5, 5, 3)(jax.random.PRNGKey(64))
    update = _solved_informed_update(
        group_ids=jnp.array([0]),
        base_scores=jnp.array([0.02]),
        current_solved=jnp.array([False]),
    )
    sampler_api = LevelSampler(capacity=2, duplicate_check=True)
    sampler = sampler_api.initialize(
        level,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
            "transfer_solved_confidence": jnp.array(
                DEFAULT_TRANSFER_SOLVED_CONFIDENCE
            ),
            "transfer_base_log_relative_score": jnp.array(0.0),
        },
    )

    sampler, inserted_index = sampler_api.insert(
        sampler,
        level,
        update.scores[0],
        _solved_informed_level_extra(update, 0),
    )

    assert jnp.isfinite(update.scores[0])
    assert int(inserted_index) == 0
    assert int(sampler["size"]) == 1
    assert int(sampler["levels_extra"]["transfer_unsolved_failure_count"][0]) == 1


def test_solved_informed_score_lowers_but_does_not_zero_unsolved_sampling() -> None:
    generator = make_level_generator(5, 5, 3)
    solved_level = generator(jax.random.PRNGKey(65))
    unsolved_level = generator(jax.random.PRNGKey(66))
    update = _solved_informed_update(
        group_ids=jnp.array([0, 1]),
        base_scores=jnp.array([0.02, 0.02]),
        current_solved=jnp.array([True, False]),
    )
    sampler_api = LevelSampler(
        capacity=2,
        staleness_coeff=0.0,
        duplicate_check=True,
    )
    sampler = sampler_api.initialize(
        solved_level,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
            "transfer_solved_confidence": jnp.array(
                DEFAULT_TRANSFER_SOLVED_CONFIDENCE
            ),
            "transfer_base_log_relative_score": jnp.array(0.0),
        },
    )
    sampler, _ = sampler_api.insert_batch(
        sampler,
        _stack_levels(solved_level, unsolved_level),
        update.scores,
        jax.tree_util.tree_map(
            lambda *values: jnp.stack(values),
            _solved_informed_level_extra(update, 0),
            _solved_informed_level_extra(update, 1),
        ),
    )

    weights = sampler_api.level_weights(sampler)

    assert float(update.scores[1]) < float(update.scores[0])
    assert 0 < float(weights[1]) < float(weights[0])


def test_eviction_discards_buffer_local_failure_metadata() -> None:
    """Lifetime successes live in the separate, non-evicting archive."""
    generator = make_level_generator(5, 5, 3)
    evicted_level = generator(jax.random.PRNGKey(67))
    replacement_level = generator(jax.random.PRNGKey(68))
    sampler_api = LevelSampler(capacity=1, duplicate_check=True)
    sampler = sampler_api.initialize(
        evicted_level,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
            "transfer_solved_confidence": jnp.array(
                DEFAULT_TRANSFER_SOLVED_CONFIDENCE
            ),
            "transfer_base_log_relative_score": jnp.array(0.0),
        },
    )
    failed_update = _solved_informed_update(
        group_ids=jnp.array([0]),
        base_scores=jnp.array([0.0]),
        current_solved=jnp.array([False]),
        prior_failure_counts=jnp.array([8]),
    )
    sampler, _ = sampler_api.insert(
        sampler,
        evicted_level,
        jnp.array(-1.0),
        _solved_informed_level_extra(failed_update, 0),
    )
    sampler, replacement_index = sampler_api.insert(
        sampler,
        replacement_level,
        jnp.array(1.0),
        {
            "max_return": jnp.array(1.0),
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(True),
            "transfer_solved_confidence": jnp.array(1.0),
            "transfer_base_log_relative_score": jnp.array(1.0),
        },
    )

    _, failures, ever_solved, prior_max_returns = resolve_new_solved_informed_history(
        sampler,
        _stack_levels(evicted_level),
        duplicate_check=True,
    )

    assert int(replacement_index) == 0
    assert int(failures[0]) == 0
    assert not bool(ever_solved[0])
    assert jnp.isneginf(prior_max_returns[0])


def test_solved_history_survives_standard_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "solved-informed-state"
    checkpointer = ocp.StandardCheckpointer()
    state = {
        "sampler": {
            "levels_extra": {
                "transfer_unsolved_failure_count": jnp.array([9, 0], dtype=jnp.int32),
                "transfer_ever_solved": jnp.array([False, True]),
                "transfer_solved_confidence": jnp.array([0.25, 1.0]),
                "transfer_base_log_relative_score": jnp.array([-0.1, 0.2]),
            }
        }
    }

    checkpointer.save(checkpoint_path, state)
    restored = checkpointer.restore(checkpoint_path)
    checkpointer.close()

    restored_extras = restored["sampler"]["levels_extra"]
    assert jnp.array_equal(
        restored_extras["transfer_unsolved_failure_count"],
        state["sampler"]["levels_extra"]["transfer_unsolved_failure_count"],
    )
    assert jnp.array_equal(
        restored_extras["transfer_ever_solved"],
        state["sampler"]["levels_extra"]["transfer_ever_solved"],
    )
    assert jnp.allclose(
        restored_extras["transfer_solved_confidence"],
        state["sampler"]["levels_extra"]["transfer_solved_confidence"],
    )
    assert jnp.allclose(
        restored_extras["transfer_base_log_relative_score"],
        state["sampler"]["levels_extra"]["transfer_base_log_relative_score"],
    )


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
    returns_before, returns_after = _returns_from_log_relative_gains(log_relative_gains)
    _, diagnostics = compute_editor_log_relative_transfer_scores(
        returns_before,
        returns_after,
    )

    assert float(diagnostics["transfer_score_q10"]) == pytest.approx(0.031)
    assert float(diagnostics["transfer_score_q50"]) == pytest.approx(0.155)
    assert float(diagnostics["transfer_score_q90"]) == pytest.approx(0.279)
    assert float(diagnostics["transfer_top_score_gap"]) == pytest.approx(0.01, abs=1e-6)


def test_transfer_score_se_metrics_preserve_sign_and_strict_thresholds() -> None:
    score_se_ratios = jnp.array([-3.0, -1.0, 0.0, 1.5])
    scores = score_se_ratios * 0.01
    log_relative_gains = jnp.stack((scores - 0.01, scores + 0.01), axis=1)
    returns_before, returns_after = _returns_from_log_relative_gains(log_relative_gains)
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


def test_level_sampler_log_dict_summarizes_resident_solved_history() -> None:
    generator = make_level_generator(5, 5, 3)
    first = generator(jax.random.PRNGKey(69))
    second = generator(jax.random.PRNGKey(70))
    sampler_api = LevelSampler(capacity=4, duplicate_check=True)
    sampler = sampler_api.initialize(
        first,
        {
            "max_return": -jnp.inf,
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
            "transfer_solved_confidence": jnp.array(
                DEFAULT_TRANSFER_SOLVED_CONFIDENCE
            ),
            "transfer_base_log_relative_score": jnp.array(0.0),
        },
    )
    sampler, _ = sampler_api.insert_batch(
        sampler,
        _stack_levels(first, second),
        jnp.array([0.1, 0.2]),
        {
            "max_return": jnp.array([0.0, 1.0]),
            "transfer_unsolved_failure_count": jnp.array([5, 2], dtype=jnp.int32),
            "transfer_ever_solved": jnp.array([False, True]),
            "transfer_solved_confidence": jnp.array([0.4, 1.0]),
            "transfer_base_log_relative_score": jnp.array([0.1, 0.2]),
        },
    )
    state = SimpleNamespace(
        sampler=sampler,
        num_dr_updates=1,
        num_replay_updates=0,
        num_mutation_updates=0,
    )

    log = train_state_to_log_dict(state, sampler_api)["log"]

    assert int(log["level_sampler/solved_informed/ever_solved_count"]) == 1
    assert float(
        log["level_sampler/solved_informed/ever_solved_fraction"]
    ) == pytest.approx(0.5)
    assert int(log["level_sampler/solved_informed/never_solved_confidence_count"]) == 1
    assert float(
        log["level_sampler/solved_informed/never_solved_confidence_mean"]
    ) == pytest.approx(0.4)
    assert float(
        log["level_sampler/solved_informed/never_solved_failure_count_mean"]
    ) == pytest.approx(5.0)
    assert float(log["level_sampler/solved_informed/base_score_mean"]) == pytest.approx(
        0.15
    )


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
        EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
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
        EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transfer_solved_prior_alpha", 0.0),
        ("transfer_solved_prior_alpha", 2.0),
        ("transfer_solved_prior_alpha", jnp.nan),
        ("transfer_solved_prior_beta", 0.0),
        ("transfer_solved_prior_beta", jnp.inf),
        ("transfer_solved_confidence", 0.0),
        ("transfer_solved_confidence", 1.0),
        ("transfer_solved_confidence", jnp.nan),
    ],
)
def test_solved_informed_transfer_rejects_invalid_controls(
    field: str,
    value: float,
) -> None:
    config = {
        "score_function": EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
        "transfer_target_count": 64,
        "transfer_num_edits": 8,
        "transfer_log_relative_tau": 0.1,
        "transfer_solved_prior_alpha": DEFAULT_TRANSFER_SOLVED_PRIOR_ALPHA,
        "transfer_solved_prior_beta": DEFAULT_TRANSFER_SOLVED_PRIOR_BETA,
        "transfer_solved_confidence": DEFAULT_TRANSFER_SOLVED_CONFIDENCE,
        "use_accel": False,
    }
    config[field] = value

    with pytest.raises(ValueError):
        validate_editor_transfer_config(config)


@pytest.mark.parametrize(
    ("prior_beta", "confidence", "message"),
    [
        (1.0, 0.99999999, "confidence.*after float32 conversion"),
        (1.0, 1e-50, "confidence.*after float32 conversion"),
        (1e40, 0.8, "prior_beta.*after float32 conversion"),
        (1e-50, 0.8, "prior_beta.*after float32 conversion"),
        (0.01, 0.8, "at 0 failures"),
        (3e38, 0.8, "at 0 failures"),
        (1.0, 1e-34, "at 960000 failures"),
    ],
)
def test_solved_informed_numerics_reject_float32_breakdown(
    prior_beta: float, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_solved_informed_numerics(prior_beta, confidence, 960_000)


@pytest.mark.parametrize(
    ("prior_beta", "confidence", "max_failures"),
    [
        (1.0, 0.8, 0),
        (1.0, 0.8, 960_000),
        (1.0, 0.8, 2**31 - 1),
        (2.0, 0.2, 960_000),
        (10.0, 0.99, 960_000),
        (1.0, 1e-34, 1),
    ],
)
def test_solved_informed_numerics_accept_safe_parameters(
    prior_beta: float, confidence: float, max_failures: int
) -> None:
    validate_solved_informed_numerics(prior_beta, confidence, max_failures)


@pytest.mark.parametrize("max_failures", [-1, 2**31])
def test_solved_informed_numerics_reject_counter_overflow(max_failures: int) -> None:
    with pytest.raises(ValueError, match="int32"):
        validate_solved_informed_numerics(1.0, 0.8, max_failures)


def test_solved_informed_config_checks_full_duplicate_failure_budget() -> None:
    config = {
        "score_function": EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
        "transfer_target_count": 64,
        "transfer_num_edits": 8,
        "use_accel": False,
        "num_updates": 1,
        "num_train_envs": 1,
        "transfer_solved_confidence": 1e-34,
    }
    validate_editor_transfer_config(config)

    config.update(num_updates=30_000, num_train_envs=32)
    with pytest.raises(ValueError, match="at 960000 failures"):
        validate_editor_transfer_config(config)


@pytest.mark.parametrize(
    "tau",
    [0.0, -0.1, float("nan"), float("inf"), -float("inf"), 1e-100, 1e100, 1e-40],
)
def test_log_relative_tau_guard_rejects_unsafe_values(tau: float) -> None:
    with pytest.raises(ValueError, match="transfer_log_relative_tau"):
        validate_log_relative_tau(tau)


@pytest.mark.parametrize(
    "tau", [float(jnp.finfo(jnp.float32).tiny), 0.01, 0.1, 1.0, 10.0]
)
def test_log_relative_tau_guard_accepts_safe_values(tau: float) -> None:
    validate_log_relative_tau(tau)
    returns = jnp.array([0.0, 0.25, 0.5, 1.0], dtype=jnp.float32)
    gains = jax.jit(compute_log_relative_transfer_gains)(
        returns[:, None], returns[None, :], tau
    )
    assert jnp.all(jnp.isfinite(gains))


def test_default_tau_point_one_is_safe_for_zero_returns_and_full_return_range() -> None:
    assert DEFAULT_TRANSFER_LOG_RELATIVE_TAU == 0.1
    validate_log_relative_tau(DEFAULT_TRANSFER_LOG_RELATIVE_TAU)
    returns = jnp.linspace(0.0, 1.0, 129, dtype=jnp.float32)
    gains = jax.jit(compute_log_relative_transfer_gains)(
        returns[:, None], returns[None, :]
    )
    assert jnp.all(jnp.isfinite(gains))
    assert jnp.all(jnp.diag(gains) == 0)
    assert float(gains[0, -1]) == pytest.approx(float(jnp.log(11.0)), abs=1e-6)
    assert float(gains[-1, 0]) == pytest.approx(-float(jnp.log(11.0)), abs=1e-6)


@pytest.mark.parametrize(
    "score_function",
    [
        EDITOR_TRANSFER_SCORE_FUNCTION,
        EDITOR_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
        EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
    ],
)
def test_all_editor_transfer_configs_validate_tau_and_accept_default(
    score_function: str,
) -> None:
    config = {
        "score_function": score_function,
        "transfer_target_count": 1,
        "transfer_num_edits": 1,
        "use_accel": False,
        "num_updates": 30_000,
        "num_train_envs": 32,
    }
    validate_editor_transfer_config(config)  # Omitted tau uses the default 0.1.
    for unsafe_tau in (1e-100, 1e100, 1e-40):
        with pytest.raises(ValueError, match="float32"):
            validate_editor_transfer_config(
                {**config, "transfer_log_relative_tau": unsafe_tau}
            )


@pytest.mark.parametrize(
    "extra_arguments, message",
    [
        (["--transfer_log_relative_tau", "1e-100"], "normal float32"),
        (["--transfer_log_relative_tau", "1e100"], "normal float32"),
        (["--transfer_log_relative_tau", "1e-40"], "normal float32"),
        (["--transfer_solved_confidence", "0.99999999"], "after float32 conversion"),
        (["--transfer_solved_prior_beta", "1e40"], "after float32 conversion"),
        (["--transfer_solved_confidence", "1e-34"], "at 960000 failures"),
        (
            ["--num_env_steps", str(2**31), "--num_train_envs", "1", "--num_steps", "1"],
            "int32",
        ),
    ],
)
def test_solved_informed_cli_rejects_unsafe_configuration_before_training(
    extra_arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            inspect.getfile(validate_editor_transfer_config),
            "--score_function",
            EDITOR_SOLVED_INFORMED_LOG_RELATIVE_TRANSFER_SCORE_FUNCTION,
            *extra_arguments,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
