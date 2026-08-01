"""Maze variant with exact shortest-path information."""

import chex
import jax
import jax.numpy as jnp
from flax import struct

from .env import EnvParams, Maze


@struct.dataclass
class EnvState:
    """Maze state augmented with shortest-path distances.

    Attributes:
        agent_pos: Integer ``(x, y)`` position of the agent.
        agent_dir: Current direction index.
        goal_pos: Integer ``(x, y)`` goal position.
        wall_map: Boolean maze occupancy grid.
        maze_map: Encoded observation grid.
        time: Number of transitions taken.
        terminal: Whether the episode has terminated.
        min_steps: Shortest action distance indexed by direction, row, and
            column.
    """

    agent_pos: chex.Array
    agent_dir: int
    goal_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool
    min_steps: chex.Array


class MazeSolved(Maze):
    """Maze environment with precomputed shortest paths to the goal.

    The augmented state supports exact reachability and optimal-value queries,
    which can be used when evaluating regret-based UED methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize using the base :class:`Maze` configuration.

        Args:
            *args: Positional arguments forwarded to :class:`Maze`.
            **kwargs: Keyword arguments forwarded to :class:`Maze`.
        """
        super().__init__(*args, **kwargs)

    def init_state_from_level(self, level):
        """Construct maze state and precompute all shortest-path distances.

        Args:
            level: Maze level to instantiate.

        Returns:
            Augmented state containing shortest distances for every orientation
            and grid location.
        """
        state = super().init_state_from_level(level)
        return EnvState(
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            maze_map=state.maze_map,
            time=state.time,
            terminal=state.terminal,
            min_steps=self._precompute_min_steps_to_goal(level),
        )

    def min_steps_to_goal(self, state: EnvState):
        """Read the shortest action distance from the current state.

        Args:
            state: Augmented maze state.

        Returns:
            Minimum number of actions required to reach the goal, or positive
            infinity when it is unreachable.
        """
        return state.min_steps[state.agent_dir, state.agent_pos[1], state.agent_pos[0]]

    def optimal_value(self, state: EnvState, gamma: float, params: EnvParams):
        """Compute discounted optimal return from the current state.

        Args:
            state: Augmented maze state.
            gamma: Per-step discount factor.
            params: Runtime maze parameters controlling the time penalty.

        Returns:
            Discounted reward for a shortest-path policy, or zero when the goal
            is unreachable.
        """
        n = self.min_steps_to_goal(state)
        if self.penalize_time:
            N = state.time + n
            value = (1.0 - 0.9 * ((N + 1) / params.max_steps_in_episode)) * gamma**n
        else:
            value = 1.0 * gamma**n
        return jnp.where(n != jnp.inf, value, 0)

    def is_solveable(self, state: EnvState, params: EnvParams):
        """Check whether the goal is reachable from the current state.

        Args:
            state: Augmented maze state.
            params: Runtime maze parameters retained for interface consistency.

        Returns:
            True when the shortest-path distance is finite.
        """
        return self.min_steps_to_goal(state) != jnp.inf

    def _precompute_min_steps_to_goal(self, level):
        """Solve shortest distances for all positions and orientations.

        Args:
            level: Maze level whose transition graph is solved.

        Returns:
            Array with shape ``(4, max_height, max_width)`` containing shortest
            action counts.
        """
        wall_values = jnp.repeat(
            jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0
        )

        def compute_next(values):
            fwd_values = jnp.array(
                [
                    jnp.roll(values[0], -1, axis=1)
                    .astype(float)
                    .at[:, -1]
                    .set(jnp.inf),
                    jnp.roll(values[1], -1, axis=0)
                    .astype(float)
                    .at[-1, :]
                    .set(jnp.inf),
                    jnp.roll(values[2], 1, axis=1).astype(float).at[:, 0].set(jnp.inf),
                    jnp.roll(values[3], 1, axis=0).astype(float).at[0, :].set(jnp.inf),
                ]
            )
            new_values = jnp.empty_like(values)
            for i in range(4):
                new_values = new_values.at[i].set(
                    jnp.min(
                        jnp.array(
                            [
                                values[i],
                                values[i - 1] + 1,
                                values[(i + 1) % 4] + 1,
                                fwd_values[i] + 1,
                            ]
                        ),
                        axis=0,
                    )
                )
            return jnp.maximum(new_values, wall_values)

        def cond_fn(carry):
            values, next_values = carry
            return jnp.any(values != next_values)

        def body_fn(carry):
            _, values = carry
            return values, compute_next(values)

        values = (
            jnp.full((4, self.max_height, self.max_width), jnp.inf)
            .at[:, level.goal_pos[1], level.goal_pos[0]]
            .set(0)
        )
        return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]
