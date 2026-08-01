"""Level-parameterized Pendulum environment."""

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from jax import lax

from jaxued.environments.underspecified_env import UnderspecifiedEnv


@struct.dataclass
class Level:
    """Physical parameters that distinguish Pendulum levels.

    Attributes:
        dt: Integration timestep.
        g: Gravitational acceleration.
        m: Pendulum mass.
        l: Pendulum length.
    """

    dt: float = 0.05
    g: float = 10.0  # gravity
    m: float = 1.0  # mass
    l: float = 1.0  # noqa: E741 - public field retained for compatibility


@struct.dataclass
class EnvState:
    """Dynamic Pendulum state.

    Attributes:
        theta: Pendulum angle in radians.
        theta_dot: Angular velocity.
        last_u: Most recently applied torque, retained for rendering.
        time: Number of elapsed transitions.
        level_params: Physical parameters of the active level.
    """

    theta: float
    theta_dot: float
    last_u: float  # Only needed for rendering
    time: int
    level_params: Level


@struct.dataclass
class EnvParams:
    """Runtime limits shared across Pendulum levels.

    Attributes:
        max_speed: Angular-velocity clipping magnitude.
        max_torque: Action clipping magnitude.
        max_steps_in_episode: Episode time limit.
    """

    max_speed: float = 8.0
    max_torque: float = 2.0
    max_steps_in_episode: int = 200


class Pendulum(UnderspecifiedEnv):
    """JAX-compatible Pendulum with level-specific physical parameters.

    The dynamics follow OpenAI Gym's Pendulum environment while moving gravity,
    mass, length, and timestep into :class:`Level`.
    """

    def __init__(self):
        """Initialize the fixed observation shape."""
        super().__init__()
        self.obs_shape = (3,)

    @property
    def default_params(self) -> EnvParams:
        """Return default Pendulum runtime limits.

        Returns:
            Default :class:`EnvParams`.
        """
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Integrate one pendulum transition.

        Args:
            key: JAX random key retained for environment interface consistency.
            state: Current Pendulum state.
            action: Torque action.
            params: Shared runtime limits.

        Returns:
            Next observation and state, negative control cost, terminal flag,
            and an empty information mapping.
        """
        u = jnp.clip(action, -params.max_torque, params.max_torque)
        reward = -(
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot**2
            + 0.001 * (u**2)
        )
        reward = reward.squeeze()

        newthdot = state.theta_dot + (
            (
                3
                * state.level_params.g
                / (2 * state.level_params.l)
                * jnp.sin(state.theta)
                + 3.0 / (state.level_params.m * state.level_params.l**2) * u
            )
            * state.level_params.dt
        )

        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * state.level_params.dt

        # Update state dict and evaluate termination conditions
        state = EnvState(
            newth.squeeze(),
            newthdot.squeeze(),
            u.reshape(),
            state.time + 1,
            level_params=state.level_params,
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
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> tuple[chex.Array, EnvState]:
        """Reset to a level with random angle and angular velocity.

        Args:
            rng: JAX random key used to sample the initial state.
            level: Physical parameters for the episode.
            params: Shared runtime limits.

        Returns:
            Initial observation and state.
        """
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(rng, shape=(2,), minval=-high, maxval=high)
        state = EnvState(
            theta=state[0], theta_dot=state[1], last_u=0.0, time=0, level_params=level
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Encode angle periodically and append angular velocity.

        Args:
            state: Current Pendulum state.

        Returns:
            Array ``[cos(theta), sin(theta), theta_dot]``.
        """
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check the episode time limit.

        Args:
            state: Current Pendulum state.
            params: Shared runtime limits.

        Returns:
            True once ``max_steps_in_episode`` transitions have elapsed.
        """
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Return the Gym-compatible environment name.

        Returns:
            ``"Pendulum-v1"``.
        """
        return "Pendulum-v1"

    @property
    def num_actions(self) -> int:
        """Return the continuous action-vector width.

        Returns:
            One.
        """
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Return the bounded torque action space.

        Args:
            params: Runtime limits. Defaults to :attr:`default_params`.

        Returns:
            One-dimensional continuous space bounded by ``max_torque``.
        """
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return bounds for the encoded observation.

        Args:
            params: Shared runtime limits.

        Returns:
            Three-dimensional continuous observation space.
        """
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Return a Gymnax space description for dynamic state fields.

        Args:
            params: Shared runtime limits.

        Returns:
            Dictionary space for angle, velocity, last torque, and time.
        """
        return spaces.Dict(
            {
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def angle_normalize(x: float) -> float:
    """Wrap an angle to ``[-pi, pi)``.

    Args:
        x: Angle in radians.

    Returns:
        Equivalent wrapped angle.
    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def make_eval_levels_and_names():
    """Construct the fixed Pendulum evaluation grid.

    Returns:
        A tuple containing 100 logarithmically spaced length/mass combinations
        plus the default level, and aligned display names.
    """
    length = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    mass = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])

    def make_level(v):
        length, mass = v
        return Level(l=length, m=mass)

    arrs = jax.vmap(jax.vmap(get_arr, (0, None)), (None, 0))(length, mass).reshape(
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
    """Create a generator over Pendulum length and mass.

    Returns:
        Callable mapping a JAX random key to a randomized :class:`Level`.
    """

    def sample(rng: chex.PRNGKey) -> Level:
        """Sample one randomized Pendulum level.

        Args:
            rng: JAX random key used to sample length and mass.

        Returns:
            Randomized Pendulum level.
        """
        rng1, rng2 = jax.random.split(rng)
        length = jax.random.uniform(rng1) * 10 + 0.01
        mass = jax.random.uniform(rng2) * 10 + 0.01
        return Level(
            l=length,
            m=mass,
            g=10.0,
            dt=0.05,
        )  # default

    return sample
