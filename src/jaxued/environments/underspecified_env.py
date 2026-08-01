"""Base types and interface for underspecified environments."""

from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
from flax import struct


@struct.dataclass
class EnvState:
    """Base PyTree type for environment state records."""

    pass


@struct.dataclass
class Observation:
    """Base PyTree type for environment observation records."""

    pass


@struct.dataclass
class Level:
    """Base PyTree type for parameters that distinguish environment levels."""

    pass


@struct.dataclass
class EnvParams:
    """Base PyTree type for parameters shared across environment levels."""

    pass


class UnderspecifiedEnv(object):
    """Interface for a level-parameterized partially observable environment.

    The interface resembles Gymnax while making the level an explicit input to
    reset. Subclasses implement :meth:`step_env`, :meth:`reset_env_to_level`,
    and :meth:`action_space`; the public :meth:`step` and
    :meth:`reset_to_level` methods supply default parameters and are JIT
    compiled with the environment instance treated as static.

    Unlike an auto-resetting Gymnax environment, this base class does not choose
    or replay a level when an episode terminates. Use a wrapper from
    ``jaxued.wrappers`` when automatic reset behavior is required.

    Example:
        >>> params = env.default_params
        >>> observation, state = env.reset_to_level(rng, level, params)
        >>> observation, state, reward, done, info = env.step(
        ...     rng, state, action, params
        ... )
    """

    @property
    def default_params(self) -> EnvParams:
        """Return default parameters shared by all levels.

        Returns:
            An empty :class:`EnvParams` record. Subclasses may override this
            property with an environment-specific parameter record.
        """
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Advance the environment by one transition.

        Args:
            rng: JAX random key used by stochastic transition logic.
            state: Current environment state.
            action: Discrete or continuous action accepted by the environment.
            params: Shared environment parameters. Defaults to
                :attr:`default_params`.

        Returns:
            A tuple ``(observation, state, reward, done, info)`` containing the
            next observation and state, scalar reward, terminal flag, and
            auxiliary transition information.
        """
        if params is None:
            params = self.default_params
        return self.step_env(rng, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: Optional[EnvParams] = None
    ) -> Tuple[Observation, EnvState]:
        """Reset the environment to a caller-specified level.

        Args:
            rng: JAX random key used by stochastic reset logic.
            level: Level parameters to instantiate.
            params: Shared environment parameters. Defaults to
                :attr:`default_params`.

        Returns:
            The initial observation and environment state for ``level``.
        """
        if params is None:
            params = self.default_params
        return self.reset_env_to_level(rng, level, params)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Implement one environment-specific transition.

        Args:
            rng: JAX random key used by the transition.
            state: Current environment-specific state.
            action: Action to apply.
            params: Shared environment parameters.

        Returns:
            The next observation, next state, reward, terminal flag, and info
            mapping.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Implement an environment-specific reset to ``level``.

        Args:
            rng: JAX random key used by the reset.
            level: Level parameters to instantiate.
            params: Shared environment parameters.

        Returns:
            The initial observation and state.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def action_space(self, params: EnvParams) -> Any:
        """Return the action space under the supplied parameters.

        Args:
            params: Shared environment parameters.

        Returns:
            A Gymnax-compatible action-space object.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError
