"""Level-parameterized CartPole environment."""

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from chex._src.pytypes import PRNGKey
from flax import struct
from gymnax.environments import spaces
from jax import lax

from jaxued.environments.underspecified_env import Observation, UnderspecifiedEnv


@struct.dataclass
class Level:
    """Physical parameters that distinguish CartPole levels.

    Attributes:
        gravity: Gravitational acceleration.
        masscart: Cart mass.
        masspole: Pole mass.
        total_mass: Sum of cart and pole mass.
        polemass_length: Product of pole mass and half-length.
        length: Pole half-length.
        force_mag: Magnitude of the left or right force.
        tau: Euler integration timestep.
    """

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    polemass_length: float = 0.05  # (masspole * length)
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02


@struct.dataclass
class EnvState:
    """Dynamic CartPole state.

    Attributes:
        x: Cart position.
        x_dot: Cart velocity.
        theta: Pole angle.
        theta_dot: Pole angular velocity.
        time: Number of elapsed transitions.
        level_params: Physical parameters of the active level.
    """

    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int
    level_params: Level


@struct.dataclass
class EnvParams:
    """Runtime limits shared across CartPole levels.

    Attributes:
        max_steps_in_episode: Episode time limit.
        theta_threshold_radians: Terminal pole-angle magnitude.
        x_threshold: Terminal cart-position magnitude.
    """

    max_steps_in_episode: int = 500
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4


class CartPole(UnderspecifiedEnv):
    """JAX CartPole whose physical constants are supplied by each level."""

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int | float, params: EnvParams
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Integrate one CartPole transition using Euler dynamics.

        Args:
            rng: JAX random key retained for environment interface consistency.
            state: Current CartPole state.
            action: Binary action selecting left or right force.
            params: Shared runtime limits.

        Returns:
            Next observation and state, survival reward, terminal flag, and an
            empty information mapping.
        """
        prev_terminal = self.is_terminal(state, params)
        force = state.level_params.force_mag * action - state.level_params.force_mag * (
            1 - action
        )
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + state.level_params.polemass_length * state.theta_dot**2 * sintheta
        ) / state.level_params.total_mass
        thetaacc = (state.level_params.gravity * sintheta - costheta * temp) / (
            state.level_params.length
            * (
                4.0 / 3.0
                - state.level_params.masspole
                * costheta**2
                / state.level_params.total_mass
            )
        )
        xacc = (
            temp
            - state.level_params.polemass_length
            * thetaacc
            * costheta
            / state.level_params.total_mass
        )

        # Only default Euler integration option available here!
        x = state.x + state.level_params.tau * state.x_dot
        x_dot = state.x_dot + state.level_params.tau * xacc
        theta = state.theta + state.level_params.tau * state.theta_dot
        theta_dot = state.theta_dot + state.level_params.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal

        # Update state dict and evaluate termination conditions
        state = EnvState(
            x, x_dot, theta, theta_dot, state.time + 1, level_params=state.level_params
        )
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    def reset_env_to_level(
        self, rng: PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation | EnvState]:
        """Reset to a level with a small random initial state.

        Args:
            rng: JAX random key used to sample the initial coordinates.
            level: Physical parameters for the episode.
            params: Shared runtime limits.

        Returns:
            Initial observation and state.
        """
        init_state = jax.random.uniform(rng, minval=-0.05, maxval=0.05, shape=(4,))
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            time=0,
            level_params=level,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Convert state coordinates to the policy observation.

        Args:
            state: Current CartPole state.

        Returns:
            Array ``[x, x_dot, theta, theta_dot]``.
        """
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    @property
    def default_params(self) -> EnvParams:
        """Return default CartPole runtime limits.

        Returns:
            Default :class:`EnvParams`.
        """
        return EnvParams()

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Return the binary force action space.

        Args:
            params: Optional runtime parameters retained for interface
                consistency.

        Returns:
            Discrete action space of size two.
        """
        return spaces.Discrete(2)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check position, angle, and time-limit termination.

        Args:
            state: Current CartPole state.
            params: Shared runtime limits.

        Returns:
            True when any terminal threshold is crossed.
        """
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done


def make_eval_levels_and_names():
    """Construct the fixed CartPole evaluation grid.

    Returns:
        A tuple containing 100 combinations of logarithmically spaced pole
        lengths and masses plus the default level, and aligned display names.
    """
    length = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    masspole = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])

    def make_level(v):
        length, mass = v
        return Level(
            length=length,
            masspole=mass,
            total_mass=1.0 + mass,
            polemass_length=length * mass,
        )

    arrs = jax.vmap(jax.vmap(get_arr, (0, None)), (None, 0))(length, masspole).reshape(
        -1, 2
    )

    levels = jax.vmap(make_level)(arrs)
    default = Level()
    levels = jax.tree_util.tree_map(
        lambda x, new: jnp.concatenate([x, jnp.array(new)[None]], axis=0),
        levels,
        default,
    )
    return levels, [f"length_{i:<2}_mass_{j:<2}" for i, j in arrs] + ["default"]


def make_level_generator() -> Callable[[chex.PRNGKey], Level]:
    """Create a generator over CartPole length and pole mass.

    Returns:
        Callable mapping a JAX random key to a randomized :class:`Level`.
    """

    def sample(rng: chex.PRNGKey) -> Level:
        """Sample one randomized CartPole level.

        Args:
            rng: JAX random key used to sample length and mass.

        Returns:
            Level with internally consistent derived mass terms.
        """
        rng1, rng2 = jax.random.split(rng)
        length = jax.random.uniform(rng1) * 10 + 0.01
        mass = jax.random.uniform(rng2) * 10 + 0.01
        return Level(
            length=length,
            masspole=mass,
            total_mass=1.0 + mass,
            polemass_length=length * mass,
        )  # default

    return sample
