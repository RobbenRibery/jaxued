"""Visited-state ensemble disagreement metrics.

This module contains pure JAX transformations for aggregating action
distributions over repeated state visits and measuring how ensemble
disagreement changes on a fixed visitor set. Environment interaction and
virtual policy updates remain the responsibility of the calling pipeline.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


class StateVisitSummary(NamedTuple):
    """Action distributions aggregated over repeated state visits.

    Attributes:
        action_probabilities: Mean action probabilities with shape
            ``(levels, policies, states, actions)``.
        visited: Boolean visitor mask with shape
            ``(levels, policies, states)``.
        visit_count: Number of visits with shape
            ``(levels, policies, states)``.
    """

    action_probabilities: chex.Array
    visited: chex.Array
    visit_count: chex.Array


class EnsembleDisagreementInputs(NamedTuple):
    """Fixed-support inputs for ensemble disagreement reduction.

    Attributes:
        before_action_probabilities: Visit-averaged pre-update distributions
            with shape ``(levels, policies, states, actions)``.
        after_action_probabilities: Visit-averaged post-update distributions
            with the same shape and state alignment as the pre-update values.
        visited: Frozen visitor mask with shape
            ``(levels, policies, states)``. It may represent one rollout or a
            union of several probe phases. A state is eligible only when at
            least two policies visited it.
    """

    before_action_probabilities: chex.Array
    after_action_probabilities: chex.Array
    visited: chex.Array


class EnsembleDisagreementResult(NamedTuple):
    """Environment-level ensemble disagreement statistics.

    Attributes:
        scores: Signed disagreement reduction with shape ``(levels,)``.
        mean_uncertainty_before: Mean pre-update disagreement across eligible
            states, with shape ``(levels,)``.
        mean_uncertainty_after: Mean post-update disagreement across eligible
            states, with shape ``(levels,)``.
        eligible_state_count: Number of states visited by at least two
            policies, with shape ``(levels,)``.
    """

    scores: chex.Array
    mean_uncertainty_before: chex.Array
    mean_uncertainty_after: chex.Array
    eligible_state_count: chex.Array


def categorical_entropy(action_probabilities: chex.Array) -> chex.Array:
    """Compute categorical entropy while treating ``0 log 0`` as zero.

    Args:
        action_probabilities: Probability vectors whose final axis enumerates
            actions.

    Returns:
        Entropy in nats with the final action axis removed.
    """
    return -jnp.sum(xlogy(action_probabilities, action_probabilities), axis=-1)


def aggregate_state_action_probabilities(
    state_ids: chex.Array,
    action_probabilities: chex.Array,
    num_states: int,
) -> StateVisitSummary:
    """Average each policy's distributions over repeated visits to a state.

    The state identifiers and action probabilities must come from the same
    pre-action decision points. State membership is therefore collected once
    and can be reused for both pre-update and post-update distributions.

    Args:
        state_ids: Integer state identifiers with shape
            ``(policies, time, levels)``.
        action_probabilities: Action probabilities with shape
            ``(policies, time, levels, actions)``.
        num_states: Static size of the encoded state space.

    Returns:
        Visit-averaged action distributions, visitor mask, and visit counts.
    """
    policy_count, _, level_count = state_ids.shape
    action_count = action_probabilities.shape[-1]

    state_ids_by_trajectory = state_ids.transpose(0, 2, 1).reshape(
        policy_count * level_count,
        -1,
    )
    probabilities_by_trajectory = action_probabilities.transpose(0, 2, 1, 3).reshape(
        policy_count * level_count,
        -1,
        action_count,
    )

    def _aggregate_trajectory(ids, probabilities):
        probability_sum = (
            jnp.zeros(
                (num_states, action_count),
                dtype=probabilities.dtype,
            )
            .at[ids]
            .add(probabilities)
        )
        visit_count = jnp.zeros((num_states,), dtype=jnp.int32).at[ids].add(1)
        mean_probabilities = probability_sum / jnp.maximum(visit_count[:, None], 1)
        return mean_probabilities, visit_count

    mean_probabilities, visit_count = jax.vmap(_aggregate_trajectory)(
        state_ids_by_trajectory,
        probabilities_by_trajectory,
    )
    mean_probabilities = mean_probabilities.reshape(
        policy_count,
        level_count,
        num_states,
        action_count,
    ).transpose(1, 0, 2, 3)
    visit_count = visit_count.reshape(
        policy_count,
        level_count,
        num_states,
    ).transpose(1, 0, 2)
    return StateVisitSummary(
        action_probabilities=mean_probabilities,
        visited=visit_count > 0,
        visit_count=visit_count,
    )


def compute_ensemble_disagreement_reduction(
    inputs: EnsembleDisagreementInputs,
) -> EnsembleDisagreementResult:
    """Compute signed disagreement reduction on a frozen visited-state set.

    For each state, only policies marked by ``inputs.visited`` contribute.
    States with fewer than two visitors are excluded. The same visitor mask is
    applied before and after the update, so the result measures policy change
    rather than a change in state visitation.

    Args:
        inputs: Visit-averaged before/after distributions and their shared
            pre-update visitor mask.

    Returns:
        Signed level scores and their before/after uncertainty components.
    """
    visitor_mask = inputs.visited.astype(inputs.before_action_probabilities.dtype)
    visitor_count = visitor_mask.sum(axis=1)
    safe_visitor_count = jnp.maximum(visitor_count, 1)
    eligible = visitor_count >= 2
    eligible_state_count = eligible.sum(axis=-1)
    safe_eligible_state_count = jnp.maximum(eligible_state_count, 1)

    def _state_uncertainty(action_probabilities):
        mean_distribution = (action_probabilities * visitor_mask[..., None]).sum(
            axis=1
        ) / safe_visitor_count[..., None]
        mean_member_entropy = (
            categorical_entropy(action_probabilities) * visitor_mask
        ).sum(axis=1) / safe_visitor_count
        return categorical_entropy(mean_distribution) - mean_member_entropy

    uncertainty_before = _state_uncertainty(inputs.before_action_probabilities)
    uncertainty_after = _state_uncertainty(inputs.after_action_probabilities)
    mean_uncertainty_before = (uncertainty_before * eligible).sum(
        axis=-1
    ) / safe_eligible_state_count
    mean_uncertainty_after = (uncertainty_after * eligible).sum(
        axis=-1
    ) / safe_eligible_state_count
    scores = mean_uncertainty_before - mean_uncertainty_after

    return EnsembleDisagreementResult(
        scores=scores,
        mean_uncertainty_before=mean_uncertainty_before,
        mean_uncertainty_after=mean_uncertainty_after,
        eligible_state_count=eligible_state_count,
    )
