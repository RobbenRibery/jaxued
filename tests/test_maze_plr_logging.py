"""Tests for Maze PLR training metrics."""

import jax
import jax.numpy as jnp
import pytest

from examples.maze_plr import (
    DEFAULT_EVAL_FREQ,
    _agent_log_metrics,
    _rollout_return_stats,
    _training_return_log_metrics,
)


def test_default_eval_frequency_is_200_updates() -> None:
    assert DEFAULT_EVAL_FREQ == 200


def test_agent_metrics_are_averaged_into_wandb_metrics() -> None:
    """PLR should expose the same PPO agent metrics as the ensemble run."""
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


def test_rollout_return_stats_include_only_completed_training_episodes() -> None:
    dones = jnp.array(
        [[1, 0], [0, 1], [1, 0], [0, 0]],
        dtype=bool,
    )
    rewards = jnp.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 100.0], [999.0, 1000.0]]
    )

    max_returns, return_sum, episode_count = jax.jit(_rollout_return_stats)(
        dones,
        rewards,
    )

    assert jnp.allclose(max_returns, jnp.array([5.0, 30.0]))
    assert float(return_sum) == pytest.approx(36.0)
    assert int(episode_count) == 3


def test_training_return_is_weighted_by_completed_episode_count() -> None:
    metrics = _training_return_log_metrics(
        return_sums=jnp.array([36.0, 4.0]),
        episode_counts=jnp.array([3, 1]),
    )

    assert set(metrics) == {"return/train", "return/train_episode_count"}
    assert float(metrics["return/train"]) == pytest.approx(10.0)
    assert int(metrics["return/train_episode_count"]) == 4


def test_training_return_is_nan_when_no_episode_completed() -> None:
    metrics = _training_return_log_metrics(
        return_sums=jnp.array([0.0]),
        episode_counts=jnp.array([0]),
    )

    assert bool(jnp.isnan(metrics["return/train"]))
    assert int(metrics["return/train_episode_count"]) == 0
