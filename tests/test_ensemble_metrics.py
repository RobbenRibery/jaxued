"""Tests for visited-state ensemble disagreement metrics."""

import jax
import jax.numpy as jnp

from jaxued.metrics import (
    EnsembleDisagreementInputs,
    aggregate_state_action_probabilities,
    compute_ensemble_disagreement_reduction,
)


def test_repeated_visits_are_averaged_within_each_policy() -> None:
    """A policy should contribute one mean distribution per visited state."""
    state_ids = jnp.array(
        [
            [[0], [0], [1], [2]],
            [[0], [1], [1], [3]],
        ]
    )
    action_probabilities = jnp.array(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.8, 0.2]],
                [[0.6, 0.4]],
            ],
            [
                [[0.25, 0.75]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.4, 0.6]],
            ],
        ]
    )

    summary = aggregate_state_action_probabilities(
        state_ids,
        action_probabilities,
        num_states=4,
    )

    assert summary.action_probabilities.shape == (1, 2, 4, 2)
    assert jnp.allclose(summary.action_probabilities[0, 0, 0], jnp.array([0.5, 0.5]))
    assert jnp.allclose(summary.action_probabilities[0, 1, 1], jnp.array([0.5, 0.5]))
    assert jnp.array_equal(
        summary.visit_count[0],
        jnp.array([[2, 1, 1, 0], [1, 2, 0, 1]]),
    )
    assert jnp.array_equal(summary.visited, summary.visit_count > 0)


def test_repeated_visits_do_not_create_an_extra_ensemble_visitor() -> None:
    """Eligibility counts distinct policies rather than raw occurrences."""
    state_ids = jnp.array(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
        ]
    )
    probabilities = jnp.array(
        [
            [[[0.8, 0.2]], [[0.8, 0.2]], [[0.8, 0.2]]],
            [[[0.2, 0.8]], [[0.2, 0.8]], [[0.2, 0.8]]],
        ]
    )
    summary = aggregate_state_action_probabilities(
        state_ids,
        probabilities,
        num_states=2,
    )

    result = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(
            summary.action_probabilities,
            summary.action_probabilities,
            summary.visited,
        )
    )

    assert jnp.array_equal(summary.visit_count[0, :, 0], jnp.array([3, 0]))
    assert jnp.array_equal(result.eligible_state_count, jnp.array([0]))


def test_duplicate_identical_occurrence_does_not_change_member_vote() -> None:
    """A member's repeated identical visit should leave its vote unchanged."""
    once_ids = jnp.array([[[0]], [[0]]])
    twice_ids = jnp.array([[[0], [0]], [[0], [0]]])
    once_before = jnp.array([[[[0.9, 0.1]]], [[[0.1, 0.9]]]])
    once_after = jnp.array([[[[0.6, 0.4]]], [[[0.4, 0.6]]]])
    twice_before = jnp.repeat(once_before, 2, axis=1)
    twice_after = jnp.repeat(once_after, 2, axis=1)

    def _score(ids, before, after):
        before_summary = aggregate_state_action_probabilities(ids, before, 1)
        after_summary = aggregate_state_action_probabilities(ids, after, 1)
        return compute_ensemble_disagreement_reduction(
            EnsembleDisagreementInputs(
                before_summary.action_probabilities,
                after_summary.action_probabilities,
                before_summary.visited,
            )
        )

    once = _score(once_ids, once_before, once_after)
    twice = _score(twice_ids, twice_before, twice_after)

    assert jnp.allclose(twice.scores, once.scores)


def test_only_states_with_two_actual_visitors_contribute() -> None:
    """Non-visitors and single-visitor states should not affect the score."""
    before = jnp.array(
        [
            [
                [[1.0, 0.0], [0.3, 0.7]],
                [[0.0, 1.0], [0.9, 0.1]],
                [[0.2, 0.8], [1.0, 0.0]],
            ]
        ]
    )
    after = jnp.array(
        [
            [
                [[0.5, 0.5], [0.1, 0.9]],
                [[0.5, 0.5], [0.8, 0.2]],
                [[0.9, 0.1], [0.0, 1.0]],
            ]
        ]
    )
    visited = jnp.array([[[True, False], [True, False], [False, True]]])

    result = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(before, after, visited)
    )

    assert jnp.allclose(result.mean_uncertainty_before, jnp.log(2.0))
    assert jnp.allclose(result.mean_uncertainty_after, 0.0)
    assert jnp.allclose(result.scores, jnp.log(2.0))
    assert jnp.array_equal(result.eligible_state_count, jnp.array([1]))


def test_disagreement_increase_produces_negative_score() -> None:
    """The scorer should retain the sign when virtual learning diverges."""
    before = jnp.array([[[[0.5, 0.5]], [[0.5, 0.5]]]])
    after = jnp.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    visited = jnp.ones((1, 2, 1), dtype=bool)

    result = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(before, after, visited)
    )

    assert jnp.allclose(result.scores, -jnp.log(2.0))


def test_identical_endpoint_policies_have_zero_learning_progress() -> None:
    """Fixed-support before and after equality should imply a zero score."""
    probabilities = jnp.array(
        [
            [
                [[0.8, 0.2]],
                [[0.2, 0.8]],
                [[0.5, 0.5]],
            ]
        ]
    )
    visited = jnp.ones((1, 3, 1), dtype=bool)

    result = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(probabilities, probabilities, visited)
    )

    upper_bound = min(jnp.log(2.0), jnp.log(3.0))
    assert result.mean_uncertainty_before >= 0.0
    assert result.mean_uncertainty_before <= upper_bound + 1e-6
    assert jnp.allclose(result.scores, 0.0)


def test_no_eligible_states_returns_finite_zeros() -> None:
    """A level without shared visits should receive the defensive zero score."""
    probabilities = jnp.array([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    visited = jnp.array([[[True], [False]]])

    result = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(probabilities, probabilities, visited)
    )

    assert jnp.array_equal(result.scores, jnp.zeros(1))
    assert jnp.array_equal(result.mean_uncertainty_before, jnp.zeros(1))
    assert jnp.array_equal(result.mean_uncertainty_after, jnp.zeros(1))
    assert jnp.array_equal(result.eligible_state_count, jnp.zeros(1, dtype=int))


def test_ensemble_metric_pipeline_is_jittable() -> None:
    """Aggregation and scoring should compose under one JAX transformation."""
    state_ids = jnp.array([[[0], [1]], [[0], [1]]])
    before = jnp.array(
        [
            [[[1.0, 0.0]], [[0.6, 0.4]]],
            [[[0.0, 1.0]], [[0.4, 0.6]]],
        ]
    )
    after = jnp.full_like(before, 0.5)

    def score(ids, before_probabilities, after_probabilities):
        before_summary = aggregate_state_action_probabilities(
            ids,
            before_probabilities,
            num_states=2,
        )
        after_summary = aggregate_state_action_probabilities(
            ids,
            after_probabilities,
            num_states=2,
        )
        return compute_ensemble_disagreement_reduction(
            EnsembleDisagreementInputs(
                before_summary.action_probabilities,
                after_summary.action_probabilities,
                before_summary.visited,
            )
        )

    eager = score(state_ids, before, after)
    compiled = jax.jit(score)(state_ids, before, after)

    assert jnp.allclose(compiled.scores, eager.scores)
    assert jnp.array_equal(
        compiled.eligible_state_count,
        eager.eligible_state_count,
    )
