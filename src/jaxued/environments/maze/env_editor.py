"""Adversarial environment for editing maze levels."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from .level import Level
from jaxued.environments import UnderspecifiedEnv
from .env import COLOR_TO_INDEX, OBJECT_TO_INDEX, Maze, make_maze_map


@struct.dataclass
class EnvState:
    """State of an adversarial maze-editing episode.

    Attributes:
        level: Maze level currently being constructed.
        time: Edit stage index.
        terminal: Whether editing has terminated.
    """

    level: Level
    time: int
    terminal: bool


@struct.dataclass
class Observation:
    """Observation exposed to the maze-editing policy.

    Attributes:
        image: Encoded maze under construction.
        action_mask: Boolean mask of currently valid edit locations.
        time: Edit stage index.
        random_z: Optional random conditioning vector.
    """

    image: chex.Array
    action_mask: chex.Array
    time: int
    random_z: chex.Array


@struct.dataclass
class EnvParams:
    """Runtime parameters for :class:`MazeEditor`."""

    pass


class MazeEditor(UnderspecifiedEnv):
    """Allow an adversarial policy to construct a maze level.

    The action identifies a grid cell. The meaning of that cell depends on the
    edit stage: move the goal, rotate the agent, move the agent, then toggle
    walls on subsequent steps.

    Args:
        env: Maze whose dimensions and encoding are edited.
        random_z_dimensions: Length of the random conditioning vector.
        zero_out_random_z: Existing multiplier controlling whether random
            conditioning values are retained.
    """

    def __init__(
        self, env: Maze, random_z_dimensions: int = 16, zero_out_random_z: bool = False
    ):
        """Initialize the level editor.

        Args:
            env: Maze environment that will consume constructed levels.
            random_z_dimensions: Number of random conditioning features.
            zero_out_random_z: Existing flag used as the random-vector
                multiplier.
        """
        super().__init__()
        self._env = env
        self.random_z_dimensions = random_z_dimensions
        self.zero_out_random_z = zero_out_random_z

    @property
    def default_params(self) -> EnvParams:
        """Return default editor parameters.

        Returns:
            Empty editor parameter record.
        """
        return EnvParams()

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Apply one edit to the level under construction.

        Args:
            rng: JAX random key used by stochastic edits and observations.
            state: Current editor state.
            action: Grid-cell index interpreted according to the edit stage.
            params: Editor runtime parameters.

        Returns:
            Updated observation and state, zero reward, terminal flag, and an
            empty information mapping.
        """
        # Do not edit level if in terminal state
        rng, rng_obs = jax.random.split(rng)
        new_level = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(state.terminal, x, y),
            state.level,
            self._edit_level(rng, state, action, params),
        )
        # Check game condition & no. steps for termination condition
        state = state.replace(level=new_level, time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        return self.get_obs(rng_obs, state), state, 0, done, {}

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Start an editing episode from an existing level.

        Args:
            rng: JAX random key used to construct the observation.
            level: Initial level to edit.
            params: Editor runtime parameters.

        Returns:
            Initial editor observation and state.
        """
        state = self.init_state_from_level(level)
        return self.get_obs(rng, state), state

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Return one discrete action per maze cell.

        Args:
            params: Editor runtime parameters.

        Returns:
            Discrete space of size :attr:`num_actions`.
        """
        return spaces.Discrete(self.num_actions)

    # ===

    @property
    def num_actions(self) -> int:
        """Return the number of addressable maze cells.

        Returns:
            Product of the wrapped maze's maximum width and height.
        """
        return self._env.max_width * self._env.max_height

    def get_obs(self, rng: chex.Array, state: EnvState):
        """Encode the current level and stage-dependent action mask.

        Args:
            rng: JAX random key used for the conditioning vector.
            state: Current editor state.

        Returns:
            Editor observation containing the maze image, valid-action mask,
            stage, and random conditioning vector.
        """
        goal_idx = (
            state.level.goal_pos[1] * self._env.max_width + state.level.goal_pos[0]
        )
        action_mask = jax.lax.switch(
            state.time.clip(None, 3),
            [
                lambda: jnp.ones(self.num_actions, dtype=jnp.bool_),
                lambda: jnp.arange(self.num_actions) < 4,
                lambda: jnp.ones(self.num_actions, dtype=jnp.bool_)
                .at[goal_idx]
                .set(False),
                lambda: jnp.ones(self.num_actions, dtype=jnp.bool_),
            ],
        )

        maze_map = make_maze_map(state.level, ignore_goal=state.time == 0)
        maze_map_with_agent = maze_map.at[
            state.level.agent_pos[1], state.level.agent_pos[0]
        ].set(
            jnp.array(
                [
                    OBJECT_TO_INDEX["agent"],
                    COLOR_TO_INDEX["red"],
                    state.level.agent_dir,
                ],
                dtype=jnp.uint8,
            )
        )
        maze_map = jax.lax.select(state.time > 2, maze_map_with_agent, maze_map)

        return Observation(
            image=maze_map,
            action_mask=action_mask,
            time=state.time,
            random_z=(
                jax.random.uniform(rng, (self.random_z_dimensions,))
                * self.zero_out_random_z
            ).astype(jnp.float32),
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Report whether the editor episode has terminated.

        Args:
            state: Current editor state.
            params: Editor runtime parameters.

        Returns:
            False in the current editor implementation.
        """
        return False

    def init_state_from_level(self, level):
        """Construct editor state from an existing level.

        Args:
            level: Level used as the initial construction state.

        Returns:
            Editor state at stage zero.
        """
        return EnvState(
            level=level,
            time=jnp.array(0, dtype=jnp.uint32),
            terminal=False,
        )

    def _edit_level(
        self, rng: chex.PRNGKey, state: EnvState, edit_idx: int, params: EnvParams
    ) -> Tuple[EnvState, float]:
        max_w, max_h = self._env.max_width, self._env.max_height

        def move_goal():
            level = state.level
            x, y = edit_idx % max_w, edit_idx // max_w
            return level.replace(
                wall_map=level.wall_map.at[y, x].set(False),
                goal_pos=jnp.array([x, y], dtype=jnp.uint32),
            )

        def rotate_agent():
            level = state.level
            return level.replace(agent_dir=jnp.array(edit_idx % 4, dtype=jnp.uint8))

        def move_agent():
            level = state.level

            # if attempting to place agent on top of goal, move agent to a random valid position
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            p = jnp.ones(max_w * max_h, dtype=jnp.bool_).at[goal_idx].set(False)
            new_edit_idx = jax.lax.select(
                edit_idx == goal_idx,
                jax.random.choice(rng, max_w * max_h, p=p),
                edit_idx,
            )
            x, y = new_edit_idx % max_w, new_edit_idx // max_w

            return level.replace(
                wall_map=level.wall_map.at[y, x].set(False),
                agent_pos=jnp.array([x, y], dtype=jnp.uint32),
            )

        def toggle_wall():
            level = state.level

            # if attempting to toggle wall on top of agent or goal, do nothing
            goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
            agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
            x, y = edit_idx % max_w, edit_idx // max_w
            wall_val = jax.lax.select(
                (edit_idx == goal_idx) | (edit_idx == agent_idx),
                False,
                ~level.wall_map[y, x],
            )

            return level.replace(wall_map=level.wall_map.at[y, x].set(wall_val))

        edit_action = state.time.clip(None, 3)
        level = jax.lax.switch(
            edit_action, [move_goal, rotate_agent, move_agent, toggle_wall]
        )

        return level
