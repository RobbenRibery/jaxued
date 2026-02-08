"""Policy gradient utilities for computing per-step raw policy gradient norms.

Produces a (T, N) matrix of gradient L2 norms using REINFORCE-style loss
(no PPO clipping). Supports minibatch processing for memory control.

All public functions are top-level for clarity and testability.
"""
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import chex
from jaxued.utils import compute_grad_norm


# ---------------------------------------------------------------------------
# Core: single (timestep, env) primitives
# ---------------------------------------------------------------------------

def raw_pg_loss_single(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_tn: chex.ArrayTree,
    done_tn: chex.Array,
    action_tn: chex.Array,
    adv_tn: chex.Array,
    hstate_n: chex.ArrayTree,
) -> chex.Array:
    """Raw policy gradient loss for one timestep, one environment.

    L = -log pi_theta(a | s) * A

    The network expects (seq_len, batch, ...) inputs and (batch, ...) hstate.
    This function adds seq_len=1 and batch=1 dims internally.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_tn: Single observation, no batch/time dims.
            e.g. image: (H, W, C), agent_dir: ()
        done_tn: Scalar done flag. Shape: ().
        action_tn: Scalar action. Shape: ().
        adv_tn: Scalar advantage. Shape: ().
        hstate_n: RNN hidden state for one env, no batch dim.
            e.g. LSTM carry: tuple of two arrays each (features,).

    Returns:
        Scalar loss.
    """
    # obs: (...) -> (seq=1, batch=1, ...)   e.g. image: (1, 1, H, W, C)
    obs_b = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs_tn)
    # done: () -> (seq=1, batch=1)
    done_b = done_tn[None, None]
    # hstate: (...) -> (batch=1, ...)   e.g. LSTM: tuple of (1, features)
    hstate_b = jax.tree_util.tree_map(lambda x: x[None, ...], hstate_n)

    # Forward pass: network returns (hstate, pi, value)
    # pi has logits shape (seq=1, batch=1, action_dim)
    _, pi, _ = apply_fn(params, (obs_b, done_b), hstate_b)
    # action: () -> (1, 1), log_prob: (1, 1) -> squeeze to scalar
    log_prob = pi.log_prob(action_tn[None, None]).squeeze()
    return -(log_prob * adv_tn)


def raw_pg_grad_norm_single(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_tn: chex.ArrayTree,
    done_tn: chex.Array,
    action_tn: chex.Array,
    adv_tn: chex.Array,
    hstate_n: chex.ArrayTree,
) -> chex.Array:
    """Gradient L2 norm of raw PG loss for one timestep, one env.

    Computes ||grad_theta L||_2 where L = -log pi(a|s) * A.
    hstate_n is treated as a constant (stop-gradient on hidden state chain).

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_tn: Single observation, no batch/time dims.
        done_tn: Scalar done flag. Shape: ().
        action_tn: Scalar action. Shape: ().
        adv_tn: Scalar advantage. Shape: ().
        hstate_n: RNN hidden state for one env, no batch dim.

    Returns:
        Scalar gradient L2 norm.
    """
    # Close over everything except params so jax.grad differentiates w.r.t. params only.
    # hstate_n is a constant here -- gradients do NOT flow through the hidden state chain.
    def loss_fn(p):
        return raw_pg_loss_single(
            apply_fn, p, obs_tn, done_tn, action_tn, adv_tn, hstate_n
        )

    grads = jax.grad(loss_fn)(params)  # pytree same structure as params
    return compute_grad_norm(grads)    # scalar: sqrt(sum of squared leaves)


# ---------------------------------------------------------------------------
# Hidden state propagation
# ---------------------------------------------------------------------------

def propagate_hstate(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_t: chex.ArrayTree,
    done_t: chex.Array,
    hstate: chex.ArrayTree,
) -> chex.ArrayTree:
    """Propagate RNN hidden state through one timestep (batched over envs).

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_t: Observations at timestep t. Shape: (num_envs, ...).
        done_t: Done flags at timestep t. Shape: (num_envs,).
        hstate: Current RNN hidden state. Shape: (num_envs, ...).

    Returns:
        New RNN hidden state. Shape: (num_envs, ...).
    """
    obs_batched = jax.tree_util.tree_map(lambda x: x[None, ...], obs_t)
    done_batched = done_t[None]
    new_hstate, _, _ = apply_fn(params, (obs_batched, done_batched), hstate)
    return new_hstate


def propagate_hstate_single(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_tn: chex.ArrayTree,
    done_tn: chex.Array,
    hstate_n: chex.ArrayTree,
) -> chex.ArrayTree:
    """Propagate RNN hidden state for a single env through one timestep.

    Adds batch=1 dim internally; returns hstate with batch dim removed.
    This mirrors the ResetRNN scan step: if done_tn is True, the hidden
    state is reset to zeros before processing the current observation.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_tn: Single observation, no batch/time dims.
        done_tn: Scalar done flag. Shape: ().
        hstate_n: RNN hidden state for one env, no batch dim.
            e.g. LSTM carry: tuple of two arrays each (features,).

    Returns:
        New hidden state, no batch dim.
            e.g. LSTM carry: tuple of two arrays each (features,).
    """
    # Add dims: obs (...) -> (1,1,...), done () -> (1,1), hstate (...) -> (1,...)
    obs_b = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs_tn)
    done_b = done_tn[None, None]
    hstate_b = jax.tree_util.tree_map(lambda x: x[None, ...], hstate_n)
    # Forward pass returns new_hstate with batch dim (1, ...)
    new_hstate_b, _, _ = apply_fn(params, (obs_b, done_b), hstate_b)
    # Remove batch dim: (1, ...) -> (...)
    return jax.tree_util.tree_map(lambda x: x[0], new_hstate_b)


# ---------------------------------------------------------------------------
# Per-timestep and per-environment grad norm computation
# ---------------------------------------------------------------------------

def grad_per_timestep(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_t: chex.ArrayTree,
    done_t: chex.Array,
    action_t: chex.Array,
    adv_t: chex.Array,
    hstate: chex.ArrayTree,
) -> chex.Array:
    """Compute raw PG grad norms for all envs at a single timestep.

    vmaps raw_pg_grad_norm_single over the env axis (axis 0).
    Each env is processed independently with its own hidden state.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_t: Observations. Shape: (N, ...).
        done_t: Done flags. Shape: (N,).
        action_t: Actions. Shape: (N,).
        adv_t: Advantages. Shape: (N,).
        hstate: Hidden states. Shape: (N, ...).

    Returns:
        Grad norms per env. Shape: (N,).
    """
    # Closure captures apply_fn and params; vmap maps over axis 0 of all args.
    # Each call receives: obs_tn (...), done_tn (), action_tn (), adv_tn (), hstate_n (...)
    def _single(obs_tn, done_tn, action_tn, adv_tn, hstate_n):
        return raw_pg_grad_norm_single(
            apply_fn, params, obs_tn, done_tn, action_tn, adv_tn, hstate_n
        )

    return jax.vmap(_single)(obs_t, done_t, action_t, adv_t, hstate)


def grad_per_environment(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs_n: chex.ArrayTree,
    dones_n: chex.Array,
    actions_n: chex.Array,
    advs_n: chex.Array,
    init_hstate_n: chex.ArrayTree,
) -> chex.Array:
    """Compute raw PG grad norms for all timesteps of one env trajectory.

    Scans sequentially over T to propagate the RNN hidden state, matching the
    training loop forward pass (uses last_dones for ResetRNN resets, starts
    from zero-initialized hidden state).

    Sequential iteration over T is required because h_t depends on h_{t-1}
    through the RNN. Environments (N) can be parallelized via vmap at the
    call site.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs_n: Observations for one env. Shape: (T, ...).
        dones_n: Done flags (last_dones) for one env. Shape: (T,).
        actions_n: Actions for one env. Shape: (T,).
        advs_n: Advantages for one env. Shape: (T,).
        init_hstate_n: Initial hidden state, no batch dim.
            e.g. LSTM carry: tuple of two arrays each (features,).

    Returns:
        Grad norms per timestep. Shape: (T,).
    """
    def scan_fn(hstate_n, inputs):
        # Each iteration receives one timestep slice:
        #   obs_tn: (...), done_tn: (), action_tn: (), adv_tn: ()
        obs_tn, done_tn, action_tn, adv_tn = inputs

        # 1) Compute grad norm at this step (hstate_n is constant w.r.t. grad)
        grad_norm = raw_pg_grad_norm_single(
            apply_fn, params, obs_tn, done_tn, action_tn, adv_tn, hstate_n
        )
        # 2) Propagate hidden state: h_t -> h_{t+1}
        #    (separate forward pass, no gradient computation)
        new_hstate_n = propagate_hstate_single(
            apply_fn, params, obs_tn, done_tn, hstate_n
        )
        return new_hstate_n, grad_norm  # carry, output

    # scan over axis 0 of (obs_n, dones_n, actions_n, advs_n) = time axis T
    # carry: hstate_n (...), outputs: grad_norms (T,)
    _, grad_norms = jax.lax.scan(
        scan_fn, init_hstate_n, (obs_n, dones_n, actions_n, advs_n)
    )
    return grad_norms


# ---------------------------------------------------------------------------
# Top-level: full (T, N) matrix computation with minibatch support
# ---------------------------------------------------------------------------

def _process_env_chunk(
    apply_fn: Callable,
    params: chex.ArrayTree,
    chunk_data: Tuple[
        chex.ArrayTree, chex.Array, chex.Array, chex.Array, chex.ArrayTree
    ],
) -> chex.Array:
    """Process a chunk of B_pg envs, returning (T, B_pg) grad norms.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        chunk_data: Tuple of (obs, dones, actions, advs, hstate) where
            obs: (T, B_pg, ...), dones/actions/advs: (T, B_pg),
            hstate: (B_pg, ...).

    Returns:
        Grad norms. Shape: (T, B_pg).
    """
    obs_c, dones_c, actions_c, advs_c, hstate_c = chunk_data
    # Input shapes: obs_c (T, B_pg, ...), dones_c/actions_c/advs_c (T, B_pg),
    #               hstate_c (B_pg, ...)

    # Transpose time-first to env-first for vmap: (T, B_pg, ...) -> (B_pg, T, ...)
    obs_t = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), obs_c)
    dones_t = jnp.swapaxes(dones_c, 0, 1)    # (B_pg, T)
    actions_t = jnp.swapaxes(actions_c, 0, 1)  # (B_pg, T)
    advs_t = jnp.swapaxes(advs_c, 0, 1)      # (B_pg, T)
    jax.debug.print(
        "_process_env_chunk: dones_t={dt} actions_t={at} advs_t={avt}",
        dt=dones_t.shape, at=actions_t.shape, avt=advs_t.shape,
    )

    # vmap grad_per_environment over B_pg envs (axis 0).
    # Each call receives: obs_n (T, ...), dones_n (T,), ..., hstate_n (...).
    # Each call returns: (T,) grad norms via internal scan over T.
    def _single_env(obs_n, dones_n, actions_n, advs_n, hstate_n):
        return grad_per_environment(
            apply_fn, params, obs_n, dones_n, actions_n, advs_n, hstate_n
        )

    grad_norms = jax.vmap(_single_env)(
        obs_t, dones_t, actions_t, advs_t, hstate_c
    )
    # grad_norms: (B_pg, T)

    return jnp.swapaxes(grad_norms, 0, 1)  # (T, B_pg)


def compute_raw_pg_grad_norms(
    apply_fn: Callable,
    params: chex.ArrayTree,
    obs: chex.ArrayTree,
    last_dones: chex.Array,
    actions: chex.Array,
    advantages: chex.Array,
    init_hstate: chex.ArrayTree,
    pg_n_minibatch: int = 1,
) -> chex.Array:
    """Compute raw policy gradient norms per timestep per environment.

    Produces a (T, N) matrix of gradient L2 norms. Uses REINFORCE-style loss
    (no PPO clipping). Environments are processed in chunks of
    N / pg_n_minibatch for memory control.

    Forward pass alignment: uses last_dones and zero-initialized init_hstate,
    matching the PPO update in update_minibatch. Hidden state propagation is
    step-by-step, producing identical h_t values to the full-sequence forward
    pass through ResetRNN.

    Args:
        apply_fn: Network forward function.
        params: Network parameters.
        obs: Observations. Shape: (T, N, ...).
        last_dones: Done flags shifted by 1. Shape: (T, N).
        actions: Actions taken. Shape: (T, N).
        advantages: Advantages (will be normalized). Shape: (T, N).
        init_hstate: Initial RNN hidden state. Shape: (N, ...).
        pg_n_minibatch: Number of env chunks for sequential processing.

    Returns:
        Gradient L2 norms. Shape: (T, N).
    """
    # Normalize advantages globally (same as PPO with n_minibatch=1)
    # adv_norm shape: (T, N) -- same as input advantages
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    num_envs = actions.shape[1]  # N
    if num_envs % pg_n_minibatch != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by "
            f"pg_n_minibatch ({pg_n_minibatch})"
        )
    b_pg = num_envs // pg_n_minibatch  # envs per chunk

    # Split N envs into pg_n_minibatch chunks of B_pg envs each.
    # Reshape: (T, N, ...) -> (T, pg_n_minibatch, B_pg, ...) -> (pg_n_minibatch, T, B_pg, ...)
    def add_mb_dim(x):
        return x.reshape(
            x.shape[0], pg_n_minibatch, b_pg, *x.shape[2:]
        ).swapaxes(0, 1)

    obs_mb = jax.tree_util.tree_map(add_mb_dim, obs)  # leaves: (M, T, B_pg, ...)
    dones_mb = add_mb_dim(last_dones)                  # (M, T, B_pg)
    actions_mb = add_mb_dim(actions)                    # (M, T, B_pg)
    advs_mb = add_mb_dim(adv_norm)                     # (M, T, B_pg)

    # Reshape hstate: (N, ...) -> (pg_n_minibatch, B_pg, ...)
    hstate_mb = jax.tree_util.tree_map(
        lambda x: x.reshape(pg_n_minibatch, b_pg, *x.shape[1:]),
        init_hstate,
    )

    # Process chunks sequentially via jax.lax.map (like vmap but sequential,
    # reducing peak memory). Each call to _process_env_chunk handles B_pg envs
    # in parallel via internal vmap.
    chunks: chex.Array = jax.lax.map(
        lambda data: _process_env_chunk(apply_fn, params, data),
        (obs_mb, dones_mb, actions_mb, advs_mb, hstate_mb),
    )
    # chunks: (pg_n_minibatch, T, B_pg)

    # Reassemble: (M, T, B_pg) -> transpose -> (T, M, B_pg) -> reshape -> (T, N)
    return chunks.transpose(1, 0, 2).reshape(chunks.shape[1], num_envs)


# ---------------------------------------------------------------------------
# Reduction functions
# ---------------------------------------------------------------------------

def reduced_mean_over_envs(grad_norms: chex.Array) -> chex.Array:
    """Reduce (T, N) grad norm matrix to (T,) by averaging over envs."""
    return grad_norms.mean(axis=1)


def reduced_mean_over_time(grad_norms: chex.Array) -> chex.Array:
    """Reduce (T, N) grad norm matrix to (N,) by averaging over time."""
    return grad_norms.mean(axis=0)


def reduced_mean_grad(grad_norms: chex.Array) -> chex.Array:
    """Reduce (T, N) grad norm matrix to scalar by averaging over both."""
    return grad_norms.mean()


# ---------------------------------------------------------------------------
# Public API (backward-compatible entry point)
