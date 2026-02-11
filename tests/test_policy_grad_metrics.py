"""Tests for policy gradient metrics.

Tests for compute_grad_norm, abs_policy_grad, and raw policy gradient norm computation.
"""

import jax
import jax.numpy as jnp
import distrax


def test_compute_grad_norm() -> None:
    """Test compute_grad_norm returns correct L2 norm."""
    from jaxued.utils import compute_grad_norm

    # Simple case: gradient pytree with known norm
    grads = {"a": jnp.array([3.0, 4.0])}  # L2 norm = 5.0
    norm = compute_grad_norm(grads)
    assert jnp.allclose(norm, 5.0), f"Expected 5.0, got {norm}"

    # Nested pytree
    grads = {"layer1": jnp.array([1.0, 0.0]), "layer2": jnp.array([0.0, 1.0])}
    norm = compute_grad_norm(grads)
    assert jnp.allclose(norm, jnp.sqrt(2.0)), f"Expected sqrt(2), got {norm}"


def test_abs_policy_grad_shape() -> None:
    """Test abs_policy_grad returns correct shape."""
    from jaxued.utils import abs_policy_grad

    num_steps, num_envs = 10, 4
    dones = jnp.zeros((num_steps, num_envs), dtype=jnp.bool_)
    dones = dones.at[-1, :].set(True)  # Episode ends at last step
    grad_norms = jnp.ones((num_steps, num_envs))

    scores = abs_policy_grad(dones, grad_norms)

    assert scores.shape == (num_envs,)
    assert jnp.all(scores >= 0), "Scores should be non-negative"


def test_abs_policy_grad_incomplete_episode() -> None:
    """Test that incomplete episodes return incomplete_value."""
    from jaxued.utils import abs_policy_grad

    num_steps, num_envs = 10, 2
    dones = jnp.zeros((num_steps, num_envs), dtype=jnp.bool_)
    dones = dones.at[-1, 0].set(True)  # Only env 0 completes
    grad_norms = jnp.ones((num_steps, num_envs))

    scores = abs_policy_grad(dones, grad_norms, incomplete_value=-999.0)

    assert scores[0] != -999.0, "Completed env should have valid score"
    assert scores[1] == -999.0, "Incomplete env should have incomplete_value"


# ---------------------------------------------------------------------------
# Helpers for raw policy gradient tests
# ---------------------------------------------------------------------------

def _make_mock_apply_fn_and_params(obs_dim: int = 3, action_dim: int = 2):
    """Create a simple linear mock network for testing.

    The network is a plain function (no RNN) that maps obs -> logits and value.
    The hstate is passed through unchanged.

    Returns:
        (apply_fn, params, obs_dim, action_dim)
    """
    rng = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    params = {
        "actor": {
            "w": jax.random.normal(k1, (obs_dim, action_dim)) * 0.1,
            "b": jax.random.normal(k2, (action_dim,)) * 0.1,
        },
        "critic": {
            "w": jax.random.normal(k3, (obs_dim, 1)) * 0.1,
            "b": jax.random.normal(k4, (1,)) * 0.1,
        },
    }

    def apply_fn(params, inputs, hstate):
        obs, dones = inputs
        logits = obs @ params["actor"]["w"] + params["actor"]["b"]
        value = (obs @ params["critic"]["w"] + params["critic"]["b"]).squeeze(-1)
        pi = distrax.Categorical(logits=logits)
        return hstate, pi, value

    return apply_fn, params, obs_dim, action_dim


# ---------------------------------------------------------------------------
# Tests for raw policy gradient primitives
# ---------------------------------------------------------------------------

def test_raw_pg_loss_single() -> None:
    """Verify raw PG loss equals -log_prob * advantage."""
    from examples.policy_grad_utils import raw_pg_loss_single

    apply_fn, params, obs_dim, action_dim = _make_mock_apply_fn_and_params()

    obs_tn = jnp.array([1.0, 0.0, 0.0])
    done_tn = jnp.array(False)
    action_tn = jnp.array(0)
    adv_tn = jnp.array(2.0)
    hstate_n = jnp.zeros(4)

    loss = raw_pg_loss_single(
        apply_fn, params, obs_tn, done_tn, action_tn, adv_tn, hstate_n
    )

    # Compute expected log_prob manually
    obs_b = obs_tn[None, None, :]
    logits = obs_b @ params["actor"]["w"] + params["actor"]["b"]
    log_probs = jax.nn.log_softmax(logits)
    expected_log_prob = log_probs[0, 0, 0]
    expected_loss = -(expected_log_prob * 2.0)

    assert jnp.allclose(loss, expected_loss, atol=1e-5), (
        f"Expected {expected_loss}, got {loss}"
    )


def test_raw_pg_grad_norm_single_nonzero() -> None:
    """Verify grad norm is positive for non-zero advantage."""
    from examples.policy_grad_utils import raw_pg_grad_norm_single

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()

    obs_tn = jnp.ones(obs_dim)
    done_tn = jnp.array(False)
    action_tn = jnp.array(1)
    adv_tn = jnp.array(3.0)
    hstate_n = jnp.zeros(4)

    norm = raw_pg_grad_norm_single(
        apply_fn, params, obs_tn, done_tn, action_tn, adv_tn, hstate_n
    )

    assert norm > 0, f"Expected positive grad norm, got {norm}"


def test_zero_advantage_zero_grad() -> None:
    """When advantage is zero, gradient norm should be zero."""
    from examples.policy_grad_utils import raw_pg_grad_norm_single

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()

    obs_tn = jnp.ones(obs_dim)
    done_tn = jnp.array(False)
    action_tn = jnp.array(0)
    adv_tn = jnp.array(0.0)
    hstate_n = jnp.zeros(4)

    norm = raw_pg_grad_norm_single(
        apply_fn, params, obs_tn, done_tn, action_tn, adv_tn, hstate_n
    )

    assert jnp.allclose(norm, 0.0, atol=1e-7), f"Expected 0, got {norm}"


def test_grad_per_timestep_shape() -> None:
    """Verify grad_per_timestep returns (N,) shape."""
    from examples.policy_grad_utils import grad_per_timestep

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()
    N = 4

    obs_t = jnp.ones((N, obs_dim))
    done_t = jnp.zeros(N, dtype=jnp.bool_)
    action_t = jnp.zeros(N, dtype=jnp.int32)
    adv_t = jnp.ones(N)
    hstate = jnp.zeros((N, 4))

    norms = grad_per_timestep(
        apply_fn, params, obs_t, done_t, action_t, adv_t, hstate
    )

    assert norms.shape == (N,), f"Expected shape ({N},), got {norms.shape}"
    assert jnp.all(norms >= 0), "All norms should be non-negative"


def test_grad_per_environment_shape() -> None:
    """Verify grad_per_environment returns (T,) shape."""
    from examples.policy_grad_utils import grad_per_environment

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()
    T = 5

    obs_n = jnp.ones((T, obs_dim))
    dones_n = jnp.zeros(T, dtype=jnp.bool_)
    actions_n = jnp.zeros(T, dtype=jnp.int32)
    advs_n = jnp.ones(T)
    hstate_n = jnp.zeros(4)

    norms = grad_per_environment(
        apply_fn, params, obs_n, dones_n, actions_n, advs_n, hstate_n
    )

    assert norms.shape == (T,), f"Expected shape ({T},), got {norms.shape}"
    assert jnp.all(norms >= 0), "All norms should be non-negative"


def test_compute_raw_pg_grad_norms_shape() -> None:
    """Verify compute_raw_pg_grad_norms returns (T, N) shape."""
    from examples.policy_grad_utils import compute_raw_pg_grad_norms

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()
    T, N = 5, 4

    obs = jax.random.normal(jax.random.PRNGKey(0), (T, N, obs_dim))
    last_dones = jnp.zeros((T, N), dtype=jnp.bool_)
    actions = jnp.zeros((T, N), dtype=jnp.int32)
    advantages = jax.random.normal(jax.random.PRNGKey(1), (T, N))
    init_hstate = jnp.zeros((N, 4))

    norms = compute_raw_pg_grad_norms(
        apply_fn, params, obs, last_dones, actions, advantages, init_hstate
    )

    assert norms.shape == (T, N), f"Expected shape ({T}, {N}), got {norms.shape}"
    assert jnp.all(norms >= 0), "All norms should be non-negative"


def test_compute_raw_pg_grad_norms_minibatch_equivalence() -> None:
    """Verify pg_n_minibatch=1 and pg_n_minibatch=N produce the same result."""
    from examples.policy_grad_utils import compute_raw_pg_grad_norms

    apply_fn, params, obs_dim, _ = _make_mock_apply_fn_and_params()
    T, N = 5, 4

    obs = jax.random.normal(jax.random.PRNGKey(0), (T, N, obs_dim))
    last_dones = jnp.zeros((T, N), dtype=jnp.bool_)
    actions = jnp.zeros((T, N), dtype=jnp.int32)
    advantages = jax.random.normal(jax.random.PRNGKey(1), (T, N))
    init_hstate = jnp.zeros((N, 4))

    norms_1 = compute_raw_pg_grad_norms(
        apply_fn, params, obs, last_dones, actions, advantages,
        init_hstate, pg_n_minibatch=1,
    )
    norms_n = compute_raw_pg_grad_norms(
        apply_fn, params, obs, last_dones, actions, advantages,
        init_hstate, pg_n_minibatch=N,
    )
    norms_2 = compute_raw_pg_grad_norms(
        apply_fn, params, obs, last_dones, actions, advantages,
        init_hstate, pg_n_minibatch=2,
    )

    assert jnp.allclose(norms_1, norms_n, atol=1e-5), (
        f"Mismatch between pg_n_minibatch=1 and pg_n_minibatch={N}"
    )
    assert jnp.allclose(norms_1, norms_2, atol=1e-5), (
        "Mismatch between pg_n_minibatch=1 and pg_n_minibatch=2"
    )


def test_reduced_mean_functions() -> None:
    """Verify reduction functions produce correct shapes and values."""
    from examples.policy_grad_utils import (
        reduced_mean_over_envs,
        reduced_mean_over_time,
        reduced_mean_grad,
    )

    T, N = 3, 4
    grad_norms = jnp.arange(T * N, dtype=jnp.float32).reshape(T, N)

    over_envs = reduced_mean_over_envs(grad_norms)
    assert over_envs.shape == (T,), f"Expected ({T},), got {over_envs.shape}"
    # Row 0: [0,1,2,3] -> mean 1.5
    assert jnp.allclose(over_envs[0], 1.5), f"Expected 1.5, got {over_envs[0]}"

    over_time = reduced_mean_over_time(grad_norms)
    assert over_time.shape == (N,), f"Expected ({N},), got {over_time.shape}"
    # Col 0: [0,4,8] -> mean 4.0
    assert jnp.allclose(over_time[0], 4.0), f"Expected 4.0, got {over_time[0]}"

    scalar = reduced_mean_grad(grad_norms)
    assert scalar.shape == (), f"Expected scalar, got shape {scalar.shape}"
    assert jnp.allclose(scalar, jnp.arange(12.0).mean()), (
        f"Expected {jnp.arange(12.0).mean()}, got {scalar}"
    )
