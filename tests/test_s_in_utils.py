"""Tests for S_in helper utilities in jaxued.utils."""

import jax.numpy as jnp


def test_s_in_from_losses_known_value() -> None:
    """Normalized reduction should match hand-computed values."""
    from jaxued.utils import s_in_from_losses

    loss_before = jnp.array([2.0, 4.0])
    loss_after = jnp.array([1.0, 2.0])

    scores = s_in_from_losses(loss_before, loss_after, eps=1e-6)
    expected = jnp.array([0.5, 0.5])

    assert jnp.allclose(scores, expected, atol=1e-6), (
        f"Expected {expected}, got {scores}"
    )


def test_run_k_virtual_updates_applies_k_steps() -> None:
    """K-step helper should apply the virtual update exactly K times."""
    from jaxued.utils import run_k_virtual_updates
    import jax

    def virtual_update_fn(rng, params, update_batch):
        del rng
        return jax.random.PRNGKey(0), params + update_batch

    rng = jax.random.PRNGKey(0)
    params = jnp.array(0.0)
    update_batch = jnp.array(1.0)

    _, updated_params = run_k_virtual_updates(
        rng=rng,
        params=params,
        update_batch=update_batch,
        virtual_update_fn=virtual_update_fn,
        n_virtual_updates=5,
    )

    assert jnp.allclose(updated_params, 5.0), f"Expected 5.0, got {updated_params}"


def test_measure_s_in_with_toy_functions() -> None:
    """measure_s_in should return expected score and diagnostics."""
    from jaxued.utils import measure_s_in
    import jax

    def loss_fn(params, eval_batch):
        target = eval_batch
        return (params - target) ** 2

    def virtual_update_fn(rng, params, update_batch):
        del update_batch
        return rng, params - 0.5

    rng = jax.random.PRNGKey(123)
    params = jnp.array(2.0)
    update_batch = jnp.array(0.0)
    eval_batch = jnp.array(0.0)

    _, score, diagnostics = measure_s_in(
        rng=rng,
        params=params,
        update_batch=update_batch,
        eval_batch=eval_batch,
        loss_fn=loss_fn,
        virtual_update_fn=virtual_update_fn,
        n_virtual_updates=2,
        eps=1e-6,
    )

    # before: 4, after: 1 -> (4 - 1) / (4 + eps) ~ 0.75
    assert jnp.allclose(diagnostics["loss_before"], 4.0), diagnostics
    assert jnp.allclose(diagnostics["loss_after"], 1.0), diagnostics
    assert jnp.allclose(score, 0.75, atol=1e-6), f"Expected 0.75, got {score}"


def test_rnn_value_mse_loss_shape_and_value() -> None:
    """Default RNN value-loss helper should return per-env scores."""
    from jaxued.utils import rnn_value_mse_loss

    def apply_fn(params, inputs, hidden):
        obs, _last_dones = inputs
        values = obs[..., 0] * params["w"]
        return hidden, None, values

    params = {"w": jnp.array(2.0)}
    obs = jnp.ones((3, 2, 1))
    last_dones = jnp.zeros((3, 2), dtype=jnp.bool_)
    targets = jnp.zeros((3, 2))
    init_hstate = jnp.zeros((2, 4))

    losses = rnn_value_mse_loss(
        apply_fn=apply_fn,
        params=params,
        eval_batch=(obs, last_dones, targets, init_hstate),
    )

    # predictions are 2 everywhere -> per-step loss = 0.5 * (2^2) = 2.0
    assert losses.shape == (2,), f"Expected shape (2,), got {losses.shape}"
    assert jnp.allclose(losses, 2.0), f"Expected [2,2], got {losses}"
