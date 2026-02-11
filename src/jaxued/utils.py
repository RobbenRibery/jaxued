import jax
import jax.flatten_util
import jax.numpy as jnp
import chex
from typing import Tuple

# This file is a modified version of
# https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py.
# Credit: minimax

# Type aliases for accumulate_rollout_stats scan function
_Carry = Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
_Input = Tuple[chex.Array, chex.Array]


def accumulate_rollout_stats(
    dones: chex.Array, metrics: chex.Array, *, time_average: bool
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Accumulate episode-level statistics from rollout data.

    Iterates through timesteps, accumulating metrics and computing per-episode
    statistics when episodes terminate (done=True).

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        metrics: Per-timestep metric values. Shape: (num_steps, num_envs).
        time_average: If True, compute time-averaged metrics per episode.
            If False, compute cumulative sum per episode.

    Returns:
        Tuple of (mean_val, max_val, episode_count):
            - mean_val: Mean metric value across completed episodes. Shape: (num_envs,).
            - max_val: Max metric value across completed episodes. Shape: (num_envs,).
            - episode_count: Number of completed episodes per env. Shape: (num_envs,).
    """

    def iter(carry: _Carry, input: _Input) -> Tuple[_Carry, None]:
        sum_val, max_val, accum_val, step_count, episode_count = carry
        done, step_val = input

        accum_val = jax.tree_util.tree_map(lambda x, y: x + y, accum_val, step_val)
        step_count += 1

        if time_average:
            # val = jax.tree_util.tree_map(lambda x, b: jax.lax.select(b, x / step_count, x), accum_val, time_average)
            val = jax.tree_util.tree_map(lambda x: x / step_count, accum_val)
        else:
            val = accum_val

        sum_val = jax.tree_util.tree_map(lambda x, y: x + done * y, sum_val, val)
        max_val = jax.tree_util.tree_map(
            lambda x, y: (1 - done) * x + done * jnp.maximum(x, y), max_val, val
        )

        episode_count += done

        accum_val = jax.tree_util.tree_map(lambda x: (1 - done) * x, accum_val)
        step_count = (1 - done) * step_count

        return (sum_val, max_val, accum_val, step_count, episode_count), None

    batch_size = dones.shape[1]
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[0]), metrics)
    (sum_val, max_val, _, _, episode_count), _ = jax.lax.scan(
        f = iter,
        init = (
            zeros,
            zeros,
            zeros,
            jnp.zeros(batch_size, dtype=jnp.uint32),
            jnp.zeros(batch_size, dtype=jnp.uint32),
        ),
        xs = (dones, metrics),
    )

    mean_val = jax.tree_util.tree_map(
        lambda x: x / jnp.maximum(episode_count, 1), sum_val
    )

    return mean_val, max_val, episode_count


def compute_max_returns(dones: chex.Array, rewards: chex.Array) -> chex.Array:
    """Compute maximum episode return per environment from rollout data.

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        rewards: Per-timestep rewards. Shape: (num_steps, num_envs).

    Returns:
        Maximum cumulative return across completed episodes per env. Shape: (num_envs,).
    """
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
    return max_returns


def compute_max_mean_returns_epcount(
    dones: chex.Array, rewards: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Compute max return, mean return, and episode count from rollout data.

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        rewards: Per-timestep rewards. Shape: (num_steps, num_envs).

    Returns:
        Tuple of (mean_returns, max_returns, episode_count), each Shape: (num_envs,).
    """
    return accumulate_rollout_stats(dones, rewards, time_average=False)


def max_mc(
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    incomplete_value: float = -jnp.inf,
) -> chex.Array:
    """Compute MaxMC (Maximum Monte Carlo) regret score.

    Regret is computed as the time-averaged difference between max returns
    and value estimates over completed episodes.

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        values: Value function estimates per timestep. Shape: (num_steps, num_envs).
        max_returns: Maximum return observed per environment/level. Shape: (num_envs,).
        incomplete_value: Value to return for environments with no completed episodes.

    Returns:
        Mean regret score per environment. Shape: (num_envs,).
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones, max_returns[None, :] - values, time_average=True
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)


def positive_value_loss(
    dones: chex.Array,
    advantages: chex.Array,
    incomplete_value: float = -jnp.inf,
) -> chex.Array:
    """Compute PVL (Positive Value Loss) regret score.

    Regret is computed as the time-averaged positive advantages over completed episodes.

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        advantages: Advantage estimates per timestep. Shape: (num_steps, num_envs).
        incomplete_value: Value to return for environments with no completed episodes.

    Returns:
        Mean positive advantage per environment. Shape: (num_envs,).
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones, jnp.maximum(advantages, 0), time_average=True
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)


def compute_grad_norm(grads: chex.ArrayTree) -> chex.Array:
    """Compute L2 norm of a gradient pytree.

    Flattens every leaf into a single 1-D vector and returns its Euclidean norm.

    Args:
        grads: Gradient pytree (same structure as params).
            Each leaf is an array of arbitrary shape.

    Returns:
        Scalar L2 norm of the concatenated gradient vector.  Shape: ().
    """
    flat, _ = jax.flatten_util.ravel_pytree(grads)
    return jnp.linalg.norm(flat)


def abs_policy_grad(
    dones: chex.Array,
    grad_norms: chex.Array,
    incomplete_value: float = -jnp.inf,
) -> chex.Array:
    """Compute absolute policy gradient score (Method 1).

    Time-averaged mean of gradient norms across all steps per completed episode,
    then averaged across episodes per environment.

    Args:
        dones: Episode termination flags. Shape: (num_steps, num_envs).
        grad_norms: Per-step gradient norms. Shape: (num_steps, num_envs).
        incomplete_value: Value for envs with no completed episodes.

    Returns:
        Mean gradient norm per environment. Shape: (num_envs,).
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones=dones,
        metrics=grad_norms,
        time_average=True,
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)
