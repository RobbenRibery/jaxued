"""Level-parameterized Acrobot environment."""

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
    """Physical parameters that distinguish Acrobot levels.

    Attributes:
        dt: Integration timestep.
        link_length_1: Length of the first link.
        link_length_2: Length of the second link.
        link_mass_1: Mass of the first link.
        link_mass_2: Mass of the second link.
        force_multiplier: Scale applied to available torque actions.
    """

    dt: float = 0.2
    link_length_1: float = 1.0
    link_length_2: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    force_multiplier: float = 1.0


@struct.dataclass
class EnvState:
    """Dynamic Acrobot state.

    Attributes:
        joint_angle1: Angle of the first joint.
        joint_angle2: Angle of the second joint.
        velocity_1: Angular velocity of the first joint.
        velocity_2: Angular velocity of the second joint.
        time: Number of elapsed transitions.
        level_params: Physical parameters of the active level.
    """

    joint_angle1: float
    joint_angle2: float
    velocity_1: float
    velocity_2: float
    time: int
    level_params: Level


@struct.dataclass
class EnvParams:
    """Runtime parameters shared across Acrobot levels.

    Attributes:
        available_torque: Torque values indexed by the discrete action.
        link_com_pos_1: First link center-of-mass position.
        link_com_pos_2: Second link center-of-mass position.
        link_moi: Link moment of inertia used by both links.
        max_vel_1: First-joint velocity clipping magnitude.
        max_vel_2: Second-joint velocity clipping magnitude.
        torque_noise_max: Uniform action-noise magnitude.
        max_steps_in_episode: Episode time limit.
    """

    available_torque: chex.Array
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * jnp.pi
    max_vel_2: float = 9 * jnp.pi
    torque_noise_max: float = 0.0
    max_steps_in_episode: int = 500


class Acrobot(UnderspecifiedEnv):
    """JAX-compatible Acrobot with level-specific physical parameters.

    The dynamics follow the default ``"book"`` formulation from OpenAI Gym's
    Acrobot environment.
    """

    def __init__(self):
        """Initialize the fixed observation shape."""
        super().__init__()
        self.obs_shape = (6,)

    @property
    def default_params(self) -> EnvParams:
        """Return default torque choices and runtime limits.

        Returns:
            Default :class:`EnvParams`.
        """
        return EnvParams(available_torque=jnp.array([-1.0, 0.0, +1.0]))

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Integrate one Acrobot transition.

        Args:
            rng: JAX random key used for torque noise.
            state: Current Acrobot state.
            action: Index into ``params.available_torque``.
            params: Shared runtime parameters.

        Returns:
            Next observation and state, swing-up reward, terminal flag, and an
            empty information mapping.
        """
        torque = params.available_torque[action] * state.level_params.force_multiplier
        # Add noise to force action - always sample - conditionals in JAX
        torque = torque + jax.random.uniform(
            rng,
            shape=(),
            minval=-params.torque_noise_max,
            maxval=params.torque_noise_max,
        )

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
                torque,
            ]
        )
        ns = rk4(s_augmented, params, state.level_params)
        joint_angle1 = wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle2 = wrap(ns[1], -jnp.pi, jnp.pi)
        velocity_1 = jnp.clip(ns[2], -params.max_vel_1, params.max_vel_1)
        velocity_2 = jnp.clip(ns[3], -params.max_vel_2, params.max_vel_2)

        done_angle = -jnp.cos(joint_angle1) - jnp.cos(joint_angle2 + joint_angle1) > 1.0
        reward = -1.0 * (1 - done_angle)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            joint_angle1,
            joint_angle2,
            velocity_1,
            velocity_2,
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
            # {"discount": self.discount(state, params)},
        )

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> tuple[chex.Array, EnvState]:
        """Reset to a level with a small random initial state.

        Args:
            rng: JAX random key used to sample initial angles and velocities.
            level: Physical parameters for the episode.
            params: Shared runtime parameters.

        Returns:
            Initial observation and state.
        """
        init_state = jax.random.uniform(rng, shape=(4,), minval=-0.1, maxval=0.1)
        state = EnvState(
            joint_angle1=init_state[0],
            joint_angle2=init_state[1],
            velocity_1=init_state[2],
            velocity_2=init_state[3],
            time=0,
            level_params=level,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Encode joint angles periodically and append velocities.

        Args:
            state: Current Acrobot state.

        Returns:
            Six-dimensional policy observation.
        """
        return jnp.array(
            [
                jnp.cos(state.joint_angle1),
                jnp.sin(state.joint_angle1),
                jnp.cos(state.joint_angle2),
                jnp.sin(state.joint_angle2),
                state.velocity_1,
                state.velocity_2,
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check swing-up completion or the episode time limit.

        Args:
            state: Current Acrobot state.
            params: Shared runtime parameters.

        Returns:
            True when the tip reaches target height or the time limit elapses.
        """
        # Check termination and construct updated state
        done_angle = (
            -jnp.cos(state.joint_angle1)
            - jnp.cos(state.joint_angle2 + state.joint_angle1)
            > 1.0
        )
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_angle, done_steps)
        return done

    @property
    def name(self) -> str:
        """Return the Gym-compatible environment name.

        Returns:
            ``"Acrobot-v1"``.
        """
        return "Acrobot-v1"

    @property
    def num_actions(self) -> int:
        """Return the number of discrete torque choices.

        Returns:
            Three.
        """
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Return the discrete torque action space.

        Args:
            params: Optional runtime parameters retained for interface
                consistency.

        Returns:
            Discrete action space of size three.
        """
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Return bounds for the encoded observation.

        Args:
            params: Shared runtime parameters.

        Returns:
            Six-dimensional continuous observation space.
        """
        high = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                params.max_vel_1,
                params.max_vel_2,
            ],
            dtype=jnp.float32,
        )
        return spaces.Box(-high, high, (6,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Return a Gymnax space description for dynamic state fields.

        Args:
            params: Shared runtime parameters.

        Returns:
            Dictionary space for joint angles, velocities, and time.
        """
        high = jnp.array(
            [
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                params.max_vel_1,
                params.max_vel_2,
            ],
            dtype=jnp.float32,
        )
        return spaces.Dict(
            {
                "joint_angle1": spaces.Box(-high[0], high[0], (), jnp.float32),
                "joint_angle2": spaces.Box(-high[1], high[1], (), jnp.float32),
                "velocity_1": spaces.Box(-high[2], high[2], (), jnp.float32),
                "velocity_2": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def dsdt(
    s_augmented: chex.Array, t: float, params: EnvParams, level: Level
) -> chex.Array:
    """Compute the Acrobot state derivative.

    Args:
        s_augmented: State vector with applied torque appended.
        t: Integration time retained for ODE-function compatibility.
        params: Shared dynamics parameters.
        level: Level-specific link lengths and masses.

    Returns:
        Derivatives of both angles and velocities, with a zero derivative for
        the appended torque.
    """
    m1, m2 = level.link_mass_1, level.link_mass_2
    l1 = level.link_length_1
    lc1, lc2 = params.link_com_pos_1, params.link_com_pos_2
    I1, I2 = params.link_moi, params.link_moi
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def wrap(x: float, m: float, M: float) -> float:
    """Wrap a scalar into the half-open interval ``[m, M)``.

    Args:
        x: Value to wrap.
        m: Inclusive lower bound.
        M: Exclusive upper bound.

    Returns:
        Value equivalent to ``x`` modulo the interval width.
    """
    diff = M - m
    go_up = x < m  # Wrap if x is outside the left bound
    go_down = x >= M  # Wrap if x is outside OR on the right bound

    how_often = (
        go_up * jnp.ceil((m - x) / diff)  # if m - x is an integer, keep it
        + go_down * jnp.floor((x - M) / diff + 1)  # if x - M is an integer, round up
    )
    x_out = x - how_often * diff * go_down + how_often * diff * go_up
    return x_out


def rk4(y0: chex.Array, params: EnvParams, level: Level):
    """Integrate one timestep with fourth-order Runge-Kutta.

    Args:
        y0: Augmented state at the start of the timestep.
        params: Shared dynamics parameters.
        level: Level-specific physical parameters and timestep.

    Returns:
        Integrated augmented state after one level timestep.
    """
    dt2 = level.dt / 2.0
    k1 = dsdt(y0, 0, params, level)
    k2 = dsdt(y0 + dt2 * k1, dt2, params, level)
    k3 = dsdt(y0 + dt2 * k2, dt2, params, level)
    k4 = dsdt(y0 + level.dt * k3, level.dt, params, level)
    yout = y0 + level.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def make_eval_levels_and_names():
    """Construct the fixed Acrobot evaluation grid.

    Returns:
        A tuple containing 100 logarithmically spaced link-length/mass
        combinations plus the default level, and aligned display names.
    """
    length = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    masspole = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])

    def make_level(v):
        length, mass = v
        return Level(
            dt=0.2,
            link_length_1=length,
            link_length_2=length,
            link_mass_1=mass,
            link_mass_2=mass,
            force_multiplier=1.0,
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
    """Create a generator over Acrobot link length and mass.

    Returns:
        Callable mapping a JAX random key to a randomized :class:`Level`.
    """

    def sample(rng: chex.PRNGKey) -> Level:
        """Sample one randomized Acrobot level.

        Args:
            rng: JAX random key used to sample length and mass.

        Returns:
            Level with equal lengths and masses for both links.
        """
        rng1, rng2 = jax.random.split(rng)
        length = jax.random.uniform(rng1) * 10 + 0.01
        mass = jax.random.uniform(rng2) * 10 + 0.01
        return Level(
            dt=0.2,
            link_length_1=length,
            link_length_2=length,
            link_mass_1=mass,
            link_mass_2=mass,
            force_multiplier=1.0,
        )  # default

    return sample
