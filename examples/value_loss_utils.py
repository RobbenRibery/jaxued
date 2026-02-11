"""Value loss utilities for computing PPO value loss scores.

Produces a (N,) vector of per-environment value loss magnitudes using
the unclipped PPO value loss. Used as a level prioritization signal
in Prioritized Level Replay (PLR).

All public functions are top-level for clarity and testability.
"""
import jax.numpy as jnp
import chex


def ppo_value_loss(
    values: chex.Array,
    targets: chex.Array,
) -> chex.Array:
    """Compute unclipped PPO value loss score for level prioritization.

    Uses the magnitude of the unclipped value loss as a prioritization
    signal. A high value loss indicates that the critic is poorly
    calibrated for a given level, suggesting the level is informative
    for training.

    The per-step, per-env unclipped value loss is:

        l_vf(t, n) = 0.5 * (values(t, n) - targets(t, n))^2

    This is the standard MSE between the value predictions produced
    during the rollout and the GAE targets (advantages + values).

    Reduction (two-stage averaging):
        1. Compute per-step, per-env value loss.       Shape: (T, N)
        2. Average over all rollout steps (axis 0).    Shape: (N,)

    Step 2 averages across the T rollout steps for each of the N
    parallel environments, collapsing the time dimension. The result
    is one scalar score per environment in the vectorized batch.

    Args:
        values: Value function predictions from the rollout.
            Shape: (T, N), where T = num_steps (rollout length),
            N = num_envs (number of parallel environments).
        targets: GAE targets, computed as advantages + values.
            Shape: (T, N).

    Returns:
        Mean value loss per environment. Shape: (N,).
    """
    # --- Step 1: Per-step, per-env squared error ---
    # values: (T, N), targets: (T, N)
    # Squared prediction error at every (timestep, env) pair
    # per_step_loss: (T, N)
    per_step_loss = 0.5 * (values - targets) ** 2

    # --- Step 2: Average over rollout steps (time axis) ---
    # For each environment n, average the T per-step losses:
    #   score(n) = (1/T) * sum_{t=0}^{T-1} l_vf(t, n)
    # per_step_loss: (T, N) -> mean over axis 0 -> (N,)
    mean_loss_per_env = per_step_loss.mean(axis=0)

    return mean_loss_per_env  # (N,)
