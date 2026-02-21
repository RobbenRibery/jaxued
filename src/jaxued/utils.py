import jax
import jax.flatten_util
import jax.numpy as jnp
import chex
from typing import Any, Callable, Dict, Tuple

# This file is a modified version of
# https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py.
# Credit: minimax

# Type aliases for accumulate_rollout_stats scan function
_Carry = Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
_Input = Tuple[chex.Array, chex.Array]
LossFn = Callable[[chex.ArrayTree, Any], chex.Array]
VirtualUpdateFn = Callable[
    [chex.PRNGKey, chex.ArrayTree, Any], Tuple[chex.PRNGKey, chex.ArrayTree]
]


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


def rnn_value_mse_loss(
    apply_fn: Callable[..., Tuple[chex.ArrayTree, Any, chex.Array]],
    params: chex.ArrayTree,
    eval_batch: Tuple[chex.ArrayTree, chex.Array, chex.Array, chex.ArrayTree],
) -> chex.Array:
    """Compute per-environment value MSE for RNN actor-critic models.

    Args:
        apply_fn: Model apply function returning (_, policy_dist, values).
        params: Model parameters.
        eval_batch: (obs, last_dones, targets, init_hstate), where:
            obs leaves have shape (T, B, ...),
            last_dones has shape (T, B),
            targets has shape (T, B),
            init_hstate leaves have shape (B, ...).

    Returns:
        Per-environment value MSE. Shape: (B,).
    """
    obs, last_dones, targets, init_hstate = eval_batch
    _, _, values_pred = apply_fn(params, (obs, last_dones), init_hstate)
    per_step_loss = 0.5 * (values_pred - targets) ** 2
    return per_step_loss.mean(axis=0)


def s_in_from_losses(
    loss_before: chex.Array,
    loss_after: chex.Array,
    eps: float = 1e-6,
) -> chex.Array:
    """Compute normalized holdout learning progress score.

    S_in = (L_before - L_after) / (L_before + eps)

    Args:
        loss_before: Loss before virtual updates. Shape: (B,) or ().
        loss_after: Loss after virtual updates. Shape: (B,) or ().
        eps: Positive stabilizer for denominator.

    Returns:
        Normalized learning progress score. Shape matches inputs.
    """
    return (loss_before - loss_after) / (loss_before + eps)


def run_k_virtual_updates(
    rng: chex.PRNGKey,
    params: chex.ArrayTree,
    update_batch: Any,
    virtual_update_fn: VirtualUpdateFn,
    n_virtual_updates: int,
) -> Tuple[chex.PRNGKey, chex.ArrayTree]:
    """Apply a virtual update rule K times.

    Args:
        rng: PRNG key.
        params: Starting parameters.
        update_batch: Batch used for each virtual update.
        virtual_update_fn: One-step virtual update function.
        n_virtual_updates: Number of virtual updates K.

    Returns:
        Tuple of (rng, updated_params) after K steps.
    """

    def _body(_, carry):
        rng_curr, params_curr = carry
        return virtual_update_fn(rng_curr, params_curr, update_batch)

    return jax.lax.fori_loop(0, n_virtual_updates, _body, (rng, params))


def measure_s_in(
    rng: chex.PRNGKey,
    params: chex.ArrayTree,
    update_batch: Any,
    eval_batch: Any,
    loss_fn: LossFn,
    virtual_update_fn: VirtualUpdateFn,
    n_virtual_updates: int,
    eps: float = 1e-6,
) -> Tuple[chex.PRNGKey, chex.Array, Dict[str, chex.Array]]:
    """Measure attainable holdout learning progress on eval_batch.

    Args:
        rng: PRNG key.
        params: Base parameters theta.
        update_batch: Virtual update data D_A.
        eval_batch: Holdout evaluation data D_B.
        loss_fn: Callable L(params, eval_batch) -> per-env loss.
        virtual_update_fn: Callable for one virtual update.
        n_virtual_updates: Number of virtual updates K.
        eps: Stabilizer for normalized reduction.

    Returns:
        (rng, s_in, diagnostics) where diagnostics has:
            "loss_before": loss before K virtual updates
            "loss_after": loss after K virtual updates
    """
    loss_before = loss_fn(params, eval_batch)
    rng, params_k = run_k_virtual_updates(
        rng=rng,
        params=params,
        update_batch=update_batch,
        virtual_update_fn=virtual_update_fn,
        n_virtual_updates=n_virtual_updates,
    )
    loss_after = loss_fn(params_k, eval_batch)
    return rng, s_in_from_losses(loss_before, loss_after, eps=eps), {
        "loss_before": loss_before,
        "loss_after": loss_after,
    }
