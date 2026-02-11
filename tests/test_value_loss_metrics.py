"""Tests for PPO value loss scoring function.

Tests for ppo_value_loss in value_loss_utils.py.
"""

import jax.numpy as jnp


def test_ppo_value_loss_shape() -> None:
    """Test ppo_value_loss returns correct (N,) shape."""
    from examples.value_loss_utils import ppo_value_loss

    T, N = 10, 4
    values = jnp.ones((T, N))
    targets = jnp.zeros((T, N))

    scores = ppo_value_loss(values, targets)

    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"


def test_ppo_value_loss_nonnegative() -> None:
    """Test that scores are always non-negative (squared error)."""
    from examples.value_loss_utils import ppo_value_loss

    T, N = 8, 3
    values = jnp.array([[1.0, -2.0, 3.0]] * T)   # (T, N)
    targets = jnp.array([[0.0, 0.0, 0.0]] * T)    # (T, N)

    scores = ppo_value_loss(values, targets)

    assert jnp.all(scores >= 0), f"Scores should be non-negative, got {scores}"


def test_ppo_value_loss_zero_when_perfect() -> None:
    """When values == targets, loss should be exactly zero."""
    from examples.value_loss_utils import ppo_value_loss

    T, N = 5, 4
    values = jnp.ones((T, N)) * 3.0
    targets = jnp.ones((T, N)) * 3.0

    scores = ppo_value_loss(values, targets)

    assert jnp.allclose(scores, 0.0), f"Expected all zeros, got {scores}"


def test_ppo_value_loss_known_value() -> None:
    """Test against a hand-computed expected value.

    values = [[1, 2],      targets = [[0, 0],
              [3, 4]]                  [0, 0]]

    per_step_loss = 0.5 * (values - targets)^2
                  = [[0.5,  2.0],
                     [4.5,  8.0]]

    mean over time (axis 0):
        env 0: (0.5 + 4.5) / 2 = 2.5
        env 1: (2.0 + 8.0) / 2 = 5.0
    """
    from examples.value_loss_utils import ppo_value_loss

    values = jnp.array([[1.0, 2.0],
                         [3.0, 4.0]])
    targets = jnp.zeros((2, 2))

    scores = ppo_value_loss(values, targets)

    assert jnp.allclose(scores[0], 2.5), f"Expected 2.5 for env 0, got {scores[0]}"
    assert jnp.allclose(scores[1], 5.0), f"Expected 5.0 for env 1, got {scores[1]}"


def test_ppo_value_loss_per_env_independence() -> None:
    """Scores for each env depend only on that env's data.

    Env 0 has zero error, env 1 has non-zero error.
    Changing env 1 should not affect env 0's score.
    """
    from examples.value_loss_utils import ppo_value_loss

    T, N = 6, 2
    values = jnp.zeros((T, N))
    targets = jnp.zeros((T, N))
    # env 1 has large error
    targets = targets.at[:, 1].set(10.0)

    scores = ppo_value_loss(values, targets)

    assert jnp.allclose(scores[0], 0.0), (
        f"Env 0 should be 0.0, got {scores[0]}"
    )
    assert scores[1] > 0.0, f"Env 1 should be > 0, got {scores[1]}"


def test_ppo_value_loss_higher_error_higher_score() -> None:
    """Env with larger prediction error should get a higher score."""
    from examples.value_loss_utils import ppo_value_loss

    T = 10
    # env 0: small error (off by 1), env 1: large error (off by 5)
    values = jnp.zeros((T, 2))
    targets = jnp.array([[1.0, 5.0]] * T)

    scores = ppo_value_loss(values, targets)

    assert scores[1] > scores[0], (
        f"Env 1 (error=5) should score higher than env 0 (error=1), "
        f"got {scores[0]:.4f} vs {scores[1]:.4f}"
    )


def test_ppo_value_loss_single_step() -> None:
    """With T=1, mean over time is just the single step loss."""
    from examples.value_loss_utils import ppo_value_loss

    values = jnp.array([[2.0, -3.0]])   # (1, 2)
    targets = jnp.array([[0.0, 0.0]])   # (1, 2)

    scores = ppo_value_loss(values, targets)

    # 0.5 * (2 - 0)^2 = 2.0, 0.5 * (-3 - 0)^2 = 4.5
    assert jnp.allclose(scores[0], 2.0), f"Expected 2.0, got {scores[0]}"
    assert jnp.allclose(scores[1], 4.5), f"Expected 4.5, got {scores[1]}"
