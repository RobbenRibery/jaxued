"""Wrapper that automatically restarts the current level."""

from typing import Any, Tuple, Union

import chex
import jax
from flax import struct

from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import (
    EnvState,
    Observation,
    Level,
    EnvParams,
)


@struct.dataclass
class AutoReplayState:
    """State carried by :class:`AutoReplayWrapper`.

    Attributes:
        env_state: State of the wrapped environment.
        level: Level replayed whenever the current episode terminates.
    """

    env_state: EnvState
    level: Level


class AutoReplayWrapper(UnderspecifiedEnv):
    """Replay the same level after every episode termination.

    Args:
        env: Environment whose transition and reset operations are wrapped.
    """

    def __init__(self, env: UnderspecifiedEnv):
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
        """
        self._env = env

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
        """Step the environment and reset terminal elements to the same level.

        Args:
            rng: JAX random key for the transition and possible reset.
            state: Wrapped state containing the current level.
            action: Action passed to the wrapped environment.
            params: Shared environment parameters.

        Returns:
            The selected next observation, updated wrapper state, reward,
            terminal flag, and transition information.
        """
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, state.level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st
        )
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        return obs, state.replace(env_state=env_state), reward, done, info

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Reset to a level and retain it for future automatic replays.

        Args:
            rng: JAX random key used by the wrapped reset.
            level: Level to instantiate and subsequently replay.
            params: Shared environment parameters.

        Returns:
            Initial observation and wrapper state.
        """
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)

    def action_space(self, params: EnvParams) -> Any:
        """Return the wrapped environment's action space.

        Args:
            params: Shared environment parameters.

        Returns:
            Action space exposed by the wrapped environment.
        """
        return self._env.action_space(params)
