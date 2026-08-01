"""Rollout-derived environment utility metrics.

The functions in this module preserve JaxUED's original MaxMC and positive
value-loss calculations. The typed input record and registry adapters separate
those calculations from configuration and training-loop orchestration.

This file is based on the UED score implementation in Minimax:
https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py
"""

from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxued.metrics.base import EnvironmentMetric, MetricRegistry


class RolloutMetricInputs(NamedTuple):
    """Signals shared by the built-in rollout utility metrics.

    Attributes:
        dones: Episode-completion mask with shape ``(time, environments)``.
        values: Critic predictions with shape ``(time, environments)``.
        max_returns: Best completed-episode return for each environment, with
            shape ``(environments,)``.
        advantages: Generalized advantage estimates with shape
            ``(time, environments)``.
    """

    dones: chex.Array
    values: chex.Array
    max_returns: chex.Array
    advantages: chex.Array


def accumulate_rollout_stats(
    dones: chex.Array,
    metrics: chex.ArrayTree,
    *,
    time_average: bool,
) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.Array]:
    """Aggregate per-step values over completed episodes.

    Values are accumulated independently for each vectorized environment. An
    incomplete final episode is excluded from the returned mean and maximum.

    Args:
        dones: Boolean or binary episode-completion mask with shape
            ``(time, environments)``.
        metrics: Array or PyTree whose leaves have leading dimensions
            ``(time, environments)``.
        time_average: Whether to divide each completed episode's accumulated
            value by its number of steps before aggregating episodes.

    Returns:
        A tuple ``(mean_values, max_values, episode_count)``. The first two
        entries preserve the structure of ``metrics`` and contain one value per
        environment. ``episode_count`` contains the number of completed
        episodes for each environment.
    """

    def _accumulate_step(carry, step):
        sum_value, max_value, accumulated_value, step_count, episode_count = carry
        done, step_value = step

        accumulated_value = jax.tree_util.tree_map(
            lambda current, value: current + value,
            accumulated_value,
            step_value,
        )
        step_count += 1

        if time_average:
            completed_value = jax.tree_util.tree_map(
                lambda value: value / step_count,
                accumulated_value,
            )
        else:
            completed_value = accumulated_value

        sum_value = jax.tree_util.tree_map(
            lambda total, value: total + done * value,
            sum_value,
            completed_value,
        )
        max_value = jax.tree_util.tree_map(
            lambda maximum, value: (1 - done) * maximum
            + done * jnp.maximum(maximum, value),
            max_value,
            completed_value,
        )

        episode_count += done
        accumulated_value = jax.tree_util.tree_map(
            lambda value: (1 - done) * value,
            accumulated_value,
        )
        step_count = (1 - done) * step_count

        return (
            sum_value,
            max_value,
            accumulated_value,
            step_count,
            episode_count,
        ), None

    batch_size = dones.shape[1]
    zeros = jax.tree_util.tree_map(lambda value: jnp.zeros_like(value[0]), metrics)
    (sum_value, max_value, _, _, episode_count), _ = jax.lax.scan(
        _accumulate_step,
        (
            zeros,
            zeros,
            zeros,
            jnp.zeros(batch_size, dtype=jnp.uint32),
            jnp.zeros(batch_size, dtype=jnp.uint32),
        ),
        (dones, metrics),
    )

    mean_value = jax.tree_util.tree_map(
        lambda value: value / jnp.maximum(episode_count, 1),
        sum_value,
    )
    return mean_value, max_value, episode_count


def compute_max_returns(dones: chex.Array, rewards: chex.Array) -> chex.Array:
    """Compute the maximum completed-episode return per environment.

    Args:
        dones: Episode-completion mask with shape ``(time, environments)``.
        rewards: Per-step rewards with shape ``(time, environments)``.

    Returns:
        Maximum completed-episode returns with shape ``(environments,)``.
    """
    _, max_returns, _ = accumulate_rollout_stats(
        dones,
        rewards,
        time_average=False,
    )
    return max_returns


def compute_max_mean_returns_epcount(
    dones: chex.Array,
    rewards: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Compute mean return, maximum return, and completion count.

    Args:
        dones: Episode-completion mask with shape ``(time, environments)``.
        rewards: Per-step rewards with shape ``(time, environments)``.

    Returns:
        A tuple ``(mean_returns, max_returns, episode_count)`` with one value
        per vectorized environment.
    """
    return accumulate_rollout_stats(dones, rewards, time_average=False)


def max_mc(
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    incomplete_value: float = -jnp.inf,
) -> chex.Array:
    """Compute the MaxMC utility score for each environment.

    The score is the mean, across completed episodes, of the gap between the
    environment's best observed return and the critic's predicted value.

    Args:
        dones: Episode-completion mask with shape ``(time, environments)``.
        values: Critic predictions with shape ``(time, environments)``.
        max_returns: Best return for each environment, with shape
            ``(environments,)``.
        incomplete_value: Score assigned to environments without a completed
            episode in the rollout.

    Returns:
        MaxMC utility scores with shape ``(environments,)``.
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones,
        max_returns[None, :] - values,
        time_average=True,
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)


def positive_value_loss(
    dones: chex.Array,
    advantages: chex.Array,
    incomplete_value: float = -jnp.inf,
) -> chex.Array:
    """Compute positive value-loss utility for each environment.

    Negative advantages are clipped to zero before values are averaged over
    completed episodes.

    Args:
        dones: Episode-completion mask with shape ``(time, environments)``.
        advantages: Advantage estimates with shape ``(time, environments)``.
        incomplete_value: Score assigned to environments without a completed
            episode in the rollout.

    Returns:
        Positive value-loss scores with shape ``(environments,)``.
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones,
        jnp.maximum(advantages, 0),
        time_average=True,
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)


def max_mc_utility(inputs: RolloutMetricInputs) -> chex.Array:
    """Adapt :func:`max_mc` to the common rollout metric contract.

    Args:
        inputs: Rollout signals produced by a PLR training step.

    Returns:
        MaxMC utility scores with one score per environment.
    """
    return max_mc(inputs.dones, inputs.values, inputs.max_returns)


def positive_value_loss_utility(inputs: RolloutMetricInputs) -> chex.Array:
    """Adapt :func:`positive_value_loss` to the rollout metric contract.

    Args:
        inputs: Rollout signals produced by a PLR training step.

    Returns:
        Positive value-loss scores with one score per environment.
    """
    return positive_value_loss(inputs.dones, inputs.advantages)


def create_rollout_metric_registry() -> MetricRegistry[RolloutMetricInputs]:
    """Create a registry containing JaxUED's built-in rollout metrics.

    A fresh registry is returned so callers may add project-specific metrics
    without mutating process-wide state.

    Returns:
        A registry containing the original ``"MaxMC"`` and ``"pvl"`` metrics.
    """
    registry = MetricRegistry[RolloutMetricInputs]()
    registry.register("MaxMC", max_mc_utility)
    registry.register("pvl", positive_value_loss_utility)
    return registry


def compute_rollout_utility(
    metric: EnvironmentMetric[RolloutMetricInputs],
    *,
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    advantages: chex.Array,
) -> chex.Array:
    """Evaluate a selected utility metric from rollout tensors.

    Args:
        metric: Metric previously selected from a compatible registry.
        dones: Episode-completion mask with shape ``(time, environments)``.
        values: Critic predictions with shape ``(time, environments)``.
        max_returns: Best completed-episode return for each environment.
        advantages: Advantage estimates with shape ``(time, environments)``.

    Returns:
        Utility scores with one score per environment.
    """
    return metric(
        RolloutMetricInputs(
            dones=dones,
            values=values,
            max_returns=max_returns,
            advantages=advantages,
        )
    )
