"""Exact, append-only source-level success history, independent of PLR residency."""

import chex
from flax import struct
import jax
import jax.numpy as jnp

from .level import Level


@struct.dataclass
class SolvedLevelHistory:
    """A collision-checked set saved as part of the training checkpoint.

    Capacity covers every possible source observation in the configured run at
    at most 50% load. Entries are never removed. Exhaustion is an explicit error
    condition, not eviction or a probabilistic claim of success.
    """

    keys: chex.Array
    occupied: chex.Array
    size: chex.Array
    overflow: chex.Array


@struct.dataclass
class SolvedLevelObservation:
    history: SolvedLevelHistory
    previously_solved: chex.Array
    ever_solved: chex.Array


def encode_level_keys(levels: Level) -> chex.Array:
    """Losslessly encode a batch of valid Maze definitions, not just wall maps."""
    count, rows, columns = levels.wall_map.shape
    walls = levels.wall_map.reshape(count, -1).astype(jnp.uint32)
    padding = (-walls.shape[1]) % 32
    walls = jnp.pad(walls, ((0, 0), (0, padding))).reshape(count, -1, 32)
    packed_walls = (walls << jnp.arange(32, dtype=jnp.uint32)).sum(
        axis=-1, dtype=jnp.uint32
    )
    return jnp.concatenate(
        [
            jnp.broadcast_to(jnp.array([rows, columns], jnp.uint32), (count, 2)),
            *[
                jnp.asarray(field, dtype=jnp.uint32).reshape(count, -1)
                for field in (
                    levels.width,
                    levels.height,
                    levels.agent_pos,
                    levels.agent_dir,
                    levels.goal_pos,
                )
            ],
            packed_walls,
        ],
        axis=1,
    )


def create_solved_level_history(
    placeholder: Level, max_observations: int
) -> SolvedLevelHistory:
    """Reserve exact history for the entire finite run, not the PLR capacity."""
    if max_observations < 0:
        raise ValueError("max_observations must be non-negative")
    if max_observations > (2**31 - 1) // 2:
        raise ValueError("Training horizon exceeds the exact history index range")
    capacity = max(2, 2 * max_observations)
    batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], placeholder)
    key_width = encode_level_keys(batch).shape[1]
    return SolvedLevelHistory(
        keys=jnp.zeros((capacity, key_width), dtype=jnp.uint32),
        occupied=jnp.zeros(capacity, dtype=jnp.bool_),
        size=jnp.array(0, dtype=jnp.int32),
        overflow=jnp.array(False),
    )


def _key_hash(key: chex.Array) -> chex.Array:
    """FNV-1a word hash; equality always checks the full lossless key."""
    return jax.lax.fori_loop(
        0,
        key.shape[0],
        lambda i, value: (value ^ key[i]) * jnp.uint32(16777619),
        jnp.uint32(2166136261),
    )


def _find_slot(
    history: SolvedLevelHistory, key: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Find the key or first empty slot with bounded linear probing."""
    capacity = history.keys.shape[0]
    start = (_key_hash(key) % jnp.uint32(capacity)).astype(jnp.int32)

    def needs_probe(state):
        slot, probes = state
        return (
            (probes < capacity)
            & history.occupied[slot]
            & ~jnp.all(history.keys[slot] == key)
        )

    slot, _ = jax.lax.while_loop(
        needs_probe,
        lambda state: ((state[0] + 1) % capacity, state[1] + 1),
        (start, jnp.array(0, dtype=jnp.int32)),
    )
    found = history.occupied[slot] & jnp.all(history.keys[slot] == key)
    return slot, found


def _lookup_keys(history: SolvedLevelHistory, keys: chex.Array) -> chex.Array:
    return jax.vmap(lambda key: _find_slot(history, key)[1])(keys)


def lookup_solved_levels(history: SolvedLevelHistory, levels: Level) -> chex.Array:
    return _lookup_keys(history, encode_level_keys(levels))


def observe_level_successes(
    history: SolvedLevelHistory, levels: Level, current_solved: chex.Array
) -> SolvedLevelObservation:
    """Record positive source-rollout outcomes, including rejected candidates.

    Read before and after the batch so all identical definitions share a success
    immediately, even when PLR duplicate checking is disabled.
    """
    keys = encode_level_keys(levels)
    previously_solved = _lookup_keys(history, keys)

    def insert_one(history, item):
        key, solved = item

        def insert(history):
            slot, found = _find_slot(history, key)
            can_insert = ~history.occupied[slot]

            def write(history):
                return history.replace(
                    keys=history.keys.at[slot].set(key),
                    occupied=history.occupied.at[slot].set(True),
                    size=history.size + 1,
                )

            history = jax.lax.cond(can_insert, write, lambda value: value, history)
            return history.replace(overflow=history.overflow | (~found & ~can_insert))

        history = jax.lax.cond(solved, insert, lambda value: value, history)
        return history, None

    history, _ = jax.lax.scan(insert_one, history, (keys, current_solved))
    return SolvedLevelObservation(
        history=history,
        previously_solved=previously_solved,
        ever_solved=_lookup_keys(history, keys),
    )


def check_solved_history_capacity(history: SolvedLevelHistory) -> None:
    """Host-side guard: never continue or checkpoint after losing a success."""
    if bool(history.overflow):
        raise RuntimeError(
            "Solved-level history capacity exhausted; refusing to forget a success. "
            "Allocate history for the full training horizon before continuing."
        )
