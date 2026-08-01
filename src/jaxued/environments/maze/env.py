"""Minigrid-style maze environment with explicit level parameters."""

from enum import IntEnum
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from jaxued.environments import UnderspecifiedEnv
from .level import Level


class Actions(IntEnum):
    """Discrete actions accepted by :class:`Maze`."""

    left = 0  # Turn left
    right = 1  # Turn right
    forward = 2  # Move forward
    pickup = 3  # Pick up an object
    drop = 4  # Drop an object
    toggle = 5  # Toggle/activate an object
    done = 6  # Done completing task


OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

COLORS = {
    "red": jnp.array([255, 0, 0]),
    "green": jnp.array([0, 255, 0]),
    "blue": jnp.array([0, 0, 255]),
    "purple": jnp.array([112, 39, 195]),
    "yellow": jnp.array([255, 255, 0]),
    "grey": jnp.array([100, 100, 100]),
}

COLOR_TO_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array(
    [
        (1, 0),  # right
        (0, 1),  # down
        (-1, 0),  # left
        (0, -1),  # up
    ],
    dtype=jnp.int8,
)


@struct.dataclass
class EnvState:
    """Dynamic state of a maze episode.

    Attributes:
        agent_pos: Integer ``(x, y)`` position of the agent.
        agent_dir: Direction index into :data:`DIR_TO_VEC`.
        goal_pos: Integer ``(x, y)`` position of the goal.
        wall_map: Boolean grid whose true entries are walls.
        maze_map: Encoded and padded observation grid.
        time: Number of transitions taken in the episode.
        terminal: Whether the goal has been reached.
    """

    agent_pos: chex.Array
    agent_dir: int
    goal_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


@struct.dataclass
class Observation:
    """Agent observation returned by :class:`Maze`.

    Attributes:
        image: Encoded full or egocentric grid observation.
        agent_dir: Current direction index.
    """

    image: chex.Array
    agent_dir: int


@struct.dataclass
class EnvParams:
    """Runtime parameters shared by maze levels.

    Attributes:
        max_steps_in_episode: Time limit applied to each episode.
    """

    max_steps_in_episode: int = 250


class Maze(UnderspecifiedEnv):
    """Minigrid-style navigation environment.

    Args:
        max_height: Maximum grid height represented by the environment.
        max_width: Maximum grid width represented by the environment.
        agent_view_size: Side length of the egocentric observation window.
        see_agent: Whether partial observations encode the agent tile.
        normalize_obs: Whether encoded image values are divided by ten.
        fully_obs: Whether observations expose the complete grid.
        penalize_time: Whether goal reward decreases with episode time.
    """

    def __init__(
        self,
        max_height=13,
        max_width=13,
        agent_view_size=5,
        see_agent=False,
        normalize_obs=False,
        fully_obs=False,
        penalize_time=True,
    ):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.agent_view_size = agent_view_size
        self.see_agent = see_agent
        self.normalize_obs = normalize_obs
        self.fully_obs = fully_obs
        self.penalize_time = penalize_time

    @property
    def default_params(self) -> EnvParams:
        """Return default maze runtime parameters.

        Returns:
            Parameters with the default episode time limit.
        """
        return EnvParams()

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Apply one maze action.

        Args:
            rng: JAX random key reserved for transition stochasticity.
            state: Current maze state.
            action: Integer member of :class:`Actions`.
            params: Runtime maze parameters.

        Returns:
            Next observation and state, reward, terminal flag, and an empty
            information mapping.
        """
        state, reward = self._step_agent(rng, state, action, params)
        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return (
            self.get_obs(state),
            state,
            reward.astype(jnp.float32),
            done,
            {},
        )

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Reset the maze to an explicit level.

        Args:
            rng: JAX random key reserved for reset stochasticity.
            level: Maze layout and initial agent configuration.
            params: Runtime maze parameters.

        Returns:
            Initial observation and state.
        """
        state = self.init_state_from_level(level)
        return self.get_obs(state), state

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return the discrete maze action space.

        Args:
            params: Runtime maze parameters.

        Returns:
            A discrete space containing every :class:`Actions` member.
        """
        return spaces.Discrete(len(Actions))

    # ===

    def init_state_from_level(self, level: Level) -> EnvState:
        """Construct episode state from a level definition.

        Args:
            level: Maze layout and starting configuration.

        Returns:
            State at time zero with an encoded observation map.
        """
        maze_map = make_maze_map(level, self.agent_view_size - 1)
        return EnvState(
            agent_pos=jnp.array(level.agent_pos, dtype=jnp.uint32),
            agent_dir=jnp.array(level.agent_dir, dtype=jnp.uint8),
            goal_pos=jnp.array(level.goal_pos, dtype=jnp.uint32),
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

    def get_obs(self, state: EnvState) -> Observation:
        """Encode the configured full or partial observation.

        Args:
            state: Current maze state.

        Returns:
            Observation selected according to ``fully_obs``.
        """
        if self.fully_obs:
            return self._get_full_obs(state)
        return self._get_partial_obs(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check goal completion or the episode time limit.

        Args:
            state: Current maze state.
            params: Runtime maze parameters.

        Returns:
            True when the goal was reached or the time limit elapsed.
        """
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_steps, state.terminal)

    def _get_full_obs(self, state: EnvState) -> Observation:
        """Return coomplete grid view"""
        padding = self.agent_view_size - 1
        obs = jax.lax.dynamic_slice(
            state.maze_map, (padding, padding, 0), (self.max_height, self.max_width, 3)
        )
        obs = obs.at[state.agent_pos[1], state.agent_pos[0]].set(
            jnp.array(
                [OBJECT_TO_INDEX["agent"], COLOR_TO_INDEX["red"], state.agent_dir],
                dtype=jnp.uint8,
            )
        )
        image = obs.astype(jnp.uint8)
        if self.normalize_obs:
            image = image / 10.0
        return Observation(image=image, agent_dir=state.agent_dir)

    def _get_partial_obs(self, state: EnvState) -> Observation:
        """Return limited grid view ahead of agent."""
        dir_vec = DIR_TO_VEC[state.agent_dir]

        obs_fwd_bound1 = state.agent_pos
        obs_fwd_bound2 = state.agent_pos + dir_vec * (self.agent_view_size - 1)

        side_offset = self.agent_view_size // 2
        obs_side_bound1 = state.agent_pos + (dir_vec == 0) * side_offset
        obs_side_bound2 = state.agent_pos - (dir_vec == 0) * side_offset

        all_bounds = jnp.stack(
            [obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2]
        )

        # Clip obs to grid bounds appropriately
        padding = self.agent_view_size - 1
        xmin, ymin = jnp.min(all_bounds, 0) + padding
        obs = jax.lax.dynamic_slice(
            state.maze_map,
            (ymin, xmin, 0),
            (self.agent_view_size, self.agent_view_size, 3),
        )

        obs = (
            (state.agent_dir == 0) * jnp.rot90(obs, 1)
            + (state.agent_dir == 1) * jnp.rot90(obs, 2)
            + (state.agent_dir == 2) * jnp.rot90(obs, 3)
            + (state.agent_dir == 3) * jnp.rot90(obs, 4)
        )

        if self.see_agent:
            obs = obs.at[-1, side_offset].set(
                jnp.array(
                    [OBJECT_TO_INDEX["agent"], COLOR_TO_INDEX["red"], state.agent_dir],
                    dtype=jnp.uint8,
                )
            )

        image = obs.astype(jnp.uint8)
        if self.normalize_obs:
            image = image / 10.0

        return Observation(image=image.transpose(1, 0, 2), agent_dir=state.agent_dir)

    def _step_agent(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[EnvState, float]:
        # Update agent position (forward action)
        fwd_pos = jnp.minimum(
            jnp.maximum(
                state.agent_pos
                + (action == Actions.forward) * DIR_TO_VEC[state.agent_dir],
                0,
            ),
            jnp.array([self.max_width - 1, self.max_height - 1], dtype=jnp.uint32),
        )

        # Can't go past wall or goal
        fwd_pos_has_wall = state.wall_map[fwd_pos[1], fwd_pos[0]]
        fwd_pos_has_goal = jnp.logical_and(
            fwd_pos[0] == state.goal_pos[0], fwd_pos[1] == state.goal_pos[1]
        )
        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal)
        agent_pos = (
            fwd_pos_blocked * state.agent_pos + (~fwd_pos_blocked) * fwd_pos
        ).astype(jnp.uint32)

        # Update agent direction (left_turn or right_turn action)
        agent_dir_offset = 0 + (action == Actions.right) - (action == Actions.left)
        agent_dir = (state.agent_dir + agent_dir_offset) % 4

        if self.penalize_time:
            reward = (
                1.0 - 0.9 * ((state.time + 1) / params.max_steps_in_episode)
            ) * fwd_pos_has_goal
        else:
            reward = jax.lax.select(fwd_pos_has_goal, 1.0, 0.0)

        return (
            state.replace(
                agent_pos=agent_pos, agent_dir=agent_dir, terminal=fwd_pos_has_goal
            ),
            reward,
        )


def make_maze_map(
    level: Level,
    padding=0,
    ignore_goal=False,
) -> chex.Array:
    """Encode a level as a Minigrid-style object/color/state tensor.

    Walls and, unless ignored, the goal are encoded. The agent is omitted so
    callers can place it according to their observation requirements.

    Args:
        level: Maze level to encode.
        padding: Number of wall-filled cells added on every side.
        ignore_goal: Whether to leave the goal cell encoded as empty.

    Returns:
        Unsigned integer tensor with shape
        ``(height + 2 * padding, width + 2 * padding, 3)``.
    """
    # Expand maze map to H x W x C
    empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=jnp.uint8)
    wall = jnp.array(
        [OBJECT_TO_INDEX["wall"], COLOR_TO_INDEX["grey"], 0], dtype=jnp.uint8
    )
    maze_map = jnp.array(jnp.expand_dims(level.wall_map, -1), dtype=jnp.uint8)
    maze_map = jnp.where(maze_map > 0, wall, empty)

    goal = jnp.array(
        [OBJECT_TO_INDEX["goal"], COLOR_TO_INDEX["green"], 0], dtype=jnp.uint8
    )
    goal_x, goal_y = level.goal_pos
    maze_map = jax.lax.select(
        ignore_goal, maze_map, maze_map.at[goal_y, goal_x, :].set(goal)
    )

    if padding > 0:
        maze_map_padded = jnp.tile(
            wall.reshape((1, 1, *empty.shape)),
            (maze_map.shape[0] + 2 * padding, maze_map.shape[1] + 2 * padding, 1),
        )
        maze_map_padded = maze_map_padded.at[padding:-padding, padding:-padding, :].set(
            maze_map
        )

        # Add surrounding walls
        wall_start = padding - 1  # start index for walls
        wall_end_y = maze_map_padded.shape[0] - wall_start - 1
        wall_end_x = maze_map_padded.shape[1] - wall_start - 1
        maze_map_padded = maze_map_padded.at[
            wall_start, wall_start : wall_end_x + 1, :
        ].set(wall)  # top
        maze_map_padded = maze_map_padded.at[
            wall_end_y, wall_start : wall_end_x + 1, :
        ].set(wall)  # bottom
        maze_map_padded = maze_map_padded.at[
            wall_start : wall_end_y + 1, wall_start, :
        ].set(wall)  # left
        maze_map_padded = maze_map_padded.at[
            wall_start : wall_end_y + 1, wall_end_x, :
        ].set(wall)  # right

        return maze_map_padded
    else:
        return maze_map
