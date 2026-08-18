"""Tests for modular environment utility metrics."""

import jax
import jax.numpy as jnp
import pytest

from jaxued import utils
from jaxued.metrics import (
    MetricRegistry,
    RolloutMetricInputs,
    compute_max_mean_returns_epcount,
    compute_rollout_utility,
    create_rollout_metric_registry,
    max_mc,
    mean_absolute_advantage,
    mean_positive_delight,
    positive_value_loss,
)


@pytest.fixture
def rollout_inputs() -> RolloutMetricInputs:
    """Create a batch containing completed and incomplete episodes."""
    return RolloutMetricInputs(
        dones=jnp.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
            ],
            dtype=bool,
        ),
        values=jnp.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [10.0, 2.0],
                [10.0, 10.0],
            ]
        ),
        max_returns=jnp.array([4.0, 6.0]),
        advantages=jnp.array(
            [
                [-1.0, 1.0],
                [4.0, -2.0],
                [10.0, 5.0],
                [10.0, 10.0],
            ]
        ),
        log_probs=jnp.array(
            [
                [-2.0, -1.0],
                [-0.5, -3.0],
                [-4.0, -0.2],
                [-1.0, -1.0],
            ]
        ),
    )


def test_rollout_aggregates_ignore_incomplete_episodes() -> None:
    """Rollout return aggregation should preserve the original semantics."""
    dones = jnp.array(
        [[0, 0], [1, 0], [0, 1], [0, 0]],
        dtype=bool,
    )
    rewards = jnp.array([[1.0, 2.0], [3.0, 2.0], [100.0, 2.0], [100.0, 9.0]])

    mean_returns, max_returns, episode_count = compute_max_mean_returns_epcount(
        dones, rewards
    )

    assert jnp.allclose(mean_returns, jnp.array([4.0, 6.0]))
    assert jnp.allclose(max_returns, jnp.array([4.0, 6.0]))
    assert jnp.array_equal(episode_count, jnp.array([1, 1]))


def test_mean_positive_delight_is_episode_aware() -> None:
    """Positive delight should average episode means and reject incomplete data."""
    scores = mean_positive_delight(
        dones=jnp.array(
            [
                [1, 0],
                [0, 0],
                [1, 0],
            ],
            dtype=bool,
        ),
        advantages=jnp.array(
            [
                [-1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        log_probs=jnp.array(
            [
                [-2.0, -1.0],
                [-4.0, -1.0],
                [-4.0, -1.0],
            ]
        ),
    )

    assert jnp.allclose(scores[0], 2.0)
    assert jnp.isneginf(scores[1])


def test_builtin_metric_adapters_match_primitive_functions(
    rollout_inputs: RolloutMetricInputs,
) -> None:
    """Registry-selected metrics should match their primitive calculations."""
    registry = create_rollout_metric_registry()

    max_mc_scores = registry.resolve("MaxMC")(rollout_inputs)
    pvl_scores = registry.resolve("pvl")(rollout_inputs)
    mean_positive_delight_scores = registry.resolve("mean_positive_delight")(
        rollout_inputs
    )
    mean_absolute_advantage_scores = registry.resolve("mean_absolute_advantage")(
        rollout_inputs
    )

    assert jnp.allclose(
        max_mc_scores,
        max_mc(
            rollout_inputs.dones,
            rollout_inputs.values,
            rollout_inputs.max_returns,
        ),
    )
    assert jnp.allclose(
        pvl_scores,
        positive_value_loss(
            rollout_inputs.dones,
            rollout_inputs.advantages,
        ),
    )
    assert jnp.allclose(max_mc_scores, jnp.array([2.5, 5.0]))
    assert jnp.allclose(pvl_scores, jnp.array([2.0, 2.0]))
    assert jnp.allclose(
        mean_positive_delight_scores,
        mean_positive_delight(
            rollout_inputs.dones,
            rollout_inputs.advantages,
            rollout_inputs.log_probs,
        ),
    )
    assert jnp.allclose(
        mean_positive_delight_scores,
        jnp.array([1.0, 2.0 / 3.0]),
    )
    assert jnp.allclose(
        mean_absolute_advantage_scores,
        mean_absolute_advantage(rollout_inputs.advantages),
    )
    assert jnp.allclose(
        mean_absolute_advantage_scores,
        jnp.array([6.25, 4.5]),
    )


def test_legacy_rollout_metrics_do_not_require_log_probs(
    rollout_inputs: RolloutMetricInputs,
) -> None:
    """Existing MaxMC/PVL callers should retain the four-signal contract."""
    legacy_inputs = RolloutMetricInputs(
        rollout_inputs.dones,
        rollout_inputs.values,
        rollout_inputs.max_returns,
        rollout_inputs.advantages,
    )
    pvl_metric = create_rollout_metric_registry().resolve("pvl")

    actual = compute_rollout_utility(
        pvl_metric,
        dones=legacy_inputs.dones,
        values=legacy_inputs.values,
        max_returns=legacy_inputs.max_returns,
        advantages=legacy_inputs.advantages,
    )

    assert jnp.allclose(actual, positive_value_loss(legacy_inputs.dones, legacy_inputs.advantages))


def test_mean_positive_delight_requires_log_probs(
    rollout_inputs: RolloutMetricInputs,
) -> None:
    """The delight metric should identify its additional evidence requirement."""
    inputs_without_log_probs = RolloutMetricInputs(
        rollout_inputs.dones,
        rollout_inputs.values,
        rollout_inputs.max_returns,
        rollout_inputs.advantages,
    )
    metric = create_rollout_metric_registry().resolve("mean_positive_delight")

    with pytest.raises(ValueError, match="requires collection-time"):
        metric(inputs_without_log_probs)


@pytest.mark.parametrize(
    "metric_name",
    ["MaxMC", "pvl", "mean_positive_delight", "mean_absolute_advantage"],
)
def test_selected_metrics_remain_jittable(
    metric_name: str,
    rollout_inputs: RolloutMetricInputs,
) -> None:
    """Configuration-time metric selection should work inside JAX tracing."""
    metric = create_rollout_metric_registry().resolve(metric_name)
    jitted_metric = jax.jit(
        lambda inputs: compute_rollout_utility(
            metric,
            dones=inputs.dones,
            values=inputs.values,
            max_returns=inputs.max_returns,
            advantages=inputs.advantages,
            log_probs=inputs.log_probs,
        )
    )

    expected = metric(rollout_inputs)
    actual = jitted_metric(rollout_inputs)

    assert jnp.allclose(actual, expected)


def test_builtin_metric_names_are_cli_selectable() -> None:
    """Built-in registry names should populate PLR CLI score choices."""
    assert create_rollout_metric_registry().names == (
        "MaxMC",
        "pvl",
        "mean_positive_delight",
        "mean_absolute_advantage",
    )


def test_registry_accepts_a_custom_metric_without_dispatch_changes(
    rollout_inputs: RolloutMetricInputs,
) -> None:
    """A compatible metric should be selectable after one registration call."""

    def constant_utility(inputs: RolloutMetricInputs) -> jax.Array:
        return jnp.full_like(inputs.max_returns, 7.0)

    registry = MetricRegistry[RolloutMetricInputs]()
    registry.register("constant", constant_utility, aliases=("constant_alias",))

    assert registry.names == ("constant",)
    assert jnp.array_equal(
        registry.resolve("constant_alias")(rollout_inputs),
        jnp.array([7.0, 7.0]),
    )


def test_registry_rejects_invalid_or_duplicate_names() -> None:
    """Registry errors should identify configuration mistakes immediately."""
    registry = MetricRegistry[RolloutMetricInputs]()
    registry.register("MaxMC", lambda inputs: inputs.max_returns)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("MaxMC", lambda inputs: inputs.max_returns)
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", lambda inputs: inputs.max_returns)
    with pytest.raises(ValueError, match="Unknown environment metric"):
        registry.resolve("missing")


def test_legacy_utils_module_reexports_metric_functions() -> None:
    """Existing imports from ``jaxued.utils`` should remain compatible."""
    assert utils.max_mc is max_mc
    assert utils.mean_absolute_advantage is mean_absolute_advantage
    assert utils.mean_positive_delight is mean_positive_delight
    assert utils.positive_value_loss is positive_value_loss
