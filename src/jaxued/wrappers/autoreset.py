"""Wrappers that reset environments to newly sampled levels."""

from typing import Any, Callable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import (
    EnvState,
    Observation,
    Level,
    EnvParams,
)


@struct.dataclass
class AutoResetState:
    """State for automatic resets from a procedural level generator.

    Attributes:
        env_state: State of the wrapped environment.
        rng: Key that generated the current level.
    """

    env_state: EnvState
    rng: chex.PRNGKey


class AutoResetWrapper(UnderspecifiedEnv):
    """Reset terminal episodes to levels drawn from a generator.

    Args:
        env: Environment to wrap.
        sample_level: Pure callable mapping a JAX random key to a level.

    Warning:
        Callers may still reset explicitly to levels outside the generator's
        support. In that case the stored level key is
        ``jax.random.PRNGKey(0)`` because no generator key exists.
    """

    def __init__(
        self, env: UnderspecifiedEnv, sample_level: Callable[[chex.PRNGKey], Level]
    ):
        """Initialize the procedural auto-reset wrapper.

        Args:
            env: Environment to wrap.
            sample_level: Level generator called after terminal transitions.
        """
        self._env = env
        self.sample_level = sample_level

    @property
    def default_params(self) -> EnvParams:
        """Return the wrapped environment's default parameters.

        Returns:
            Parameters exposed by the wrapped environment.
        """
        return self._env.default_params

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Step and replace terminal states with freshly sampled levels.

        Args:
            rng: JAX random key split between sampling, reset, and transition.
            state: Current wrapped environment state.
            action: Action passed to the wrapped environment.
            params: Shared environment parameters.

        Returns:
            The selected observation, updated wrapper state, reward, terminal
            flag, and transition information. ``info["rng"]`` identifies the
            generator key associated with the active level.
        """
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)

        new_level = self.sample_level(rng_sample)

        obs_re, env_state_re = self._env.reset_to_level(rng_reset, new_level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng, state.env_state, action, params
        )

        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st
        )
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        level_rng = jax.lax.select(done, rng_sample, state.rng)

        info["rng"] = level_rng

        return (
            obs,
            AutoResetState(env_state=env_state, rng=level_rng),
            reward,
            done,
            info,
        )

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Reset explicitly to a caller-provided level.

        Args:
            rng: JAX random key used by the wrapped reset.
            level: Level to instantiate.
            params: Shared environment parameters.

        Returns:
            Initial observation and wrapper state. The stored generator key is
            a fixed sentinel because the level was not sampled by the wrapper.
        """
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoResetState(env_state=env_state, rng=jax.random.PRNGKey(0))

    def action_space(self, params: EnvParams) -> Any:
        """Return the wrapped environment's action space.

        Args:
            params: Shared environment parameters.

        Returns:
            Action space exposed by the wrapped environment.
        """
        return self._env.action_space(params)


@struct.dataclass
class AutoResetFiniteState:
    """State for automatic resets from a finite level collection.

    Attributes:
        env_state: State of the wrapped environment.
        level_idx: Index of the active level, or ``-1`` when it is not in the
            configured collection.
    """

    env_state: EnvState
    level_idx: int


class AutoResetFiniteWrapper(UnderspecifiedEnv):
    """Reset terminal episodes from a finite categorical level distribution.

    Args:
        env: Environment to wrap.
        levels: Batched PyTree containing the finite level support.
        p: Sampling probabilities for ``levels``. Defaults to uniform.
        check_reset_to_level: Whether explicit resets search ``levels`` for the
            supplied level and record its index.

    Warning:
        Explicit resets may use levels outside the finite support. When checking
        is enabled, those levels receive index ``-1``.
    """

    def __init__(
        self, env: UnderspecifiedEnv, levels, p=None, check_reset_to_level=True
    ):
        """Initialize the finite-distribution auto-reset wrapper.

        Args:
            env: Environment to wrap.
            levels: Batched PyTree of supported levels.
            p: Optional categorical probabilities over the first dimension.
            check_reset_to_level: Whether to recover a support index during
                explicit resets.
        """
        self._env = env
        self._num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
        self._levels = levels
        self._p = (
            p if p is not None else (1 / self._num_levels) * jnp.ones(self._num_levels)
        )
        self._check_reset_to_level = True

    @property
    def default_params(self) -> EnvParams:
        """Return the wrapped environment's default parameters.

        Returns:
            Parameters exposed by the wrapped environment.
        """
        return self._env.default_params

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Step and reset terminal states from the finite distribution.

        Args:
            rng: JAX random key split between sampling, reset, and transition.
            state: Current wrapper state.
            action: Action passed to the wrapped environment.
            params: Shared environment parameters.

        Returns:
            The selected observation, updated wrapper state, reward, terminal
            flag, and transition information. ``info["level_idx"]`` identifies
            the active support element.
        """
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)

        new_level_idx = jax.random.choice(rng_sample, self._num_levels, p=self._p)
        new_level = jax.tree_util.tree_map(lambda x: x[level_idx], self._levels)

        obs_re, env_state_re = self._env.reset_to_level(rng_reset, new_level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng, state.env_state, action, params
        )

        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st
        )
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        level_idx = jax.lax.select(done, new_level_idx, state.level_idx)

        info["level_idx"] = level_idx

        return (
            obs,
            AutoResetFiniteState(env_state=env_state, level_idx=level_idx),
            reward,
            done,
            info,
        )

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Reset explicitly and optionally locate the level in the support.

        Args:
            rng: JAX random key used by the wrapped reset.
            level: Level to instantiate.
            params: Shared environment parameters.

        Returns:
            Initial observation and wrapper state containing the matching
            support index, or ``-1`` when no match is recorded.
        """
        obs, env_state = self._env.reset_to_level(rng, level, params)

        if self._check_reset_to_level:
            eq_tree = jax.tree_util.tree_map(
                lambda X, y: (X == y).reshape(self._num_levels, -1).all(axis=-1),
                self._levels,
                level,
            )
            eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
            eq_mask = jnp.array(eq_tree_flat).all(
                axis=0
            )  # & (self._p > 0) # ignores levels with no support
            level_idx = jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
        else:
            level_idx = -1

        return obs, AutoResetFiniteState(env_state=env_state, level_idx=level_idx)

    def action_space(self, params: EnvParams) -> Any:
        """Return the wrapped environment's action space.

        Args:
            params: Shared environment parameters.

        Returns:
            Action space exposed by the wrapped environment.
        """
        return self._env.action_space(params)
