"""Tests for Maze PLR training metrics."""

import jax.numpy as jnp
import pytest

from examples.maze_plr import _agent_log_metrics


def test_agent_entropy_is_averaged_into_wandb_metric() -> None:
    """PLR should expose mean PPO action entropy under the ensemble key."""
    losses = (
        jnp.array([1.0, 2.0]),
        (
            jnp.array([0.1, 0.2]),
            jnp.array([0.3, 0.4]),
            jnp.array([0.2, 0.4]),
        ),
    )

    metrics = _agent_log_metrics(losses)

    assert set(metrics) == {"agent/entropy"}
    assert float(metrics["agent/entropy"]) == pytest.approx(0.3)
