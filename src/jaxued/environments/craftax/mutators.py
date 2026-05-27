from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

from jaxued.environments.underspecified_env import EnvParams, Level


def _load_block_type() -> Any:
    try:
        from craftax.craftax.constants import BlockType
    except ImportError:
        try:
            from craftax.constants import BlockType
        except ImportError as exc:
            raise ImportError(
                "Craftax support is optional. Install it with "
                "`uv sync --extra craftax`."
            ) from exc
    return BlockType


def _normalize_num_edits(n: int) -> int:
    return int(n)


def make_mutator_craftax_swap(static_env_params: Any, only_middle: bool = True):
    BlockType = _load_block_type()
    size = static_env_params.map_size[0] * static_env_params.map_size[1]

    def add_blocktype(rng, level: Level, blocktype: int) -> Level:
        vals = jnp.arange(size)
        player_idx = (
            level.player_position[0] * static_env_params.map_size[1]
            + level.player_position[1]
        )
        probs = jnp.ones_like(vals).at[player_idx].set(0.0)

        if only_middle:
            temp = jnp.zeros_like(level.map[0], dtype=jnp.float32)
            mid_r, mid_c = [
                static_env_params.map_size[0] // 2,
                static_env_params.map_size[1] // 2,
            ]
            extent = 16 // 2
            temp = temp.at[
                mid_r - extent : mid_r + extent,
                mid_c - extent : mid_c + extent,
            ].set(1.0)
            probs = probs * temp.flatten()

        rng, rng_position = jax.random.split(rng)
        position = jax.random.choice(rng_position, vals, p=probs)
        row = position // static_env_params.map_size[1]
        col = position % static_env_params.map_size[1]

        tile_to_delete = level.map[0, row, col]
        block_check = (level.map[0] == blocktype).flatten()
        rng, rng_old = jax.random.split(rng)

        def do_swap(level):
            flat_idx = jax.random.choice(rng_old, jnp.arange(size), (), p=block_check)
            row_old = flat_idx // static_env_params.map_size[1]
            col_old = flat_idx % static_env_params.map_size[1]
            level = level.replace(
                map=level.map.at[0, row_old, col_old].set(tile_to_delete)
            )
            return level.replace(map=level.map.at[0, row, col].set(blocktype))

        def dont_swap(level):
            return level.replace(map=level.map.at[0, row, col].set(blocktype))

        return jax.lax.cond(block_check.any(), do_swap, dont_swap, level)

    good_blocks = [
        BlockType.GRASS,
        BlockType.WATER,
        BlockType.STONE,
        BlockType.TREE,
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
        BlockType.LAVA,
        BlockType.RIPE_PLANT,
    ]
    good_blocks_to_choose_from = jnp.array([b.value for b in good_blocks])

    def mutate_level(rng: chex.PRNGKey, level: Level, n: int = 1) -> Level:
        def _single_mutate(carry, _):
            rng, level = carry
            rng, rng_block, rng_add = jax.random.split(rng, 3)
            block_to_add = good_blocks_to_choose_from[
                jax.random.randint(rng_block, (), 0, len(good_blocks))
            ]
            level = add_blocktype(rng_add, level, block_to_add)
            return (rng, level), None

        (_, level), _ = jax.lax.scan(
            _single_mutate, (rng, level), None, length=_normalize_num_edits(n)
        )
        return level

    return mutate_level


def make_mutator_craftax_swap_restricted(
    static_env_params: Any,
    one_should_be_middle: bool = False,
):
    BlockType = _load_block_type()
    size = static_env_params.map_size[0] * static_env_params.map_size[1]

    def get_random_index_of_block(rng, level, blocktype_value, is_middle=False):
        block_check = (level.map[0] == blocktype_value).flatten()

        def answer1():
            if one_should_be_middle and is_middle:
                temp = jnp.zeros_like(level.map[0], dtype=jnp.float32)
                mid_r, mid_c = [
                    static_env_params.map_size[0] // 2,
                    static_env_params.map_size[1] // 2,
                ]
                extent = 16 // 2
                temp = temp.at[
                    mid_r - extent : mid_r + extent,
                    mid_c - extent : mid_c + extent,
                ].set(1.0)
                middle_probs = block_check * temp.flatten()
                flat_idx = jax.lax.select(
                    middle_probs.any(),
                    jax.random.choice(rng, jnp.arange(size), (), p=middle_probs),
                    jax.random.choice(rng, jnp.arange(size), (), p=block_check),
                )
            else:
                flat_idx = jax.random.choice(rng, jnp.arange(size), (), p=block_check)

            row = flat_idx // static_env_params.map_size[1]
            col = flat_idx % static_env_params.map_size[1]
            return row, col

        def answer2():
            return 0, 0

        return jax.lax.cond(block_check.any(), answer1, answer2)

    def single_step(rng, level: Level):
        rng, rng_block, rng_stone, rng_coal, rng_diamond, rng_iron, rng1, rng2 = (
            jax.random.split(rng, 8)
        )
        block_to_add = good_blocks_to_choose_from[
            jax.random.randint(rng_block, (), 0, len(good_blocks))
        ]

        new_block = BlockType.GRASS.value
        new_block = jax.lax.select(
            block_to_add == BlockType.GRASS.value, BlockType.TREE.value, new_block
        )
        new_block = jax.lax.select(
            block_to_add == BlockType.TREE.value, BlockType.GRASS.value, new_block
        )
        new_block = jax.lax.select(
            block_to_add == BlockType.STONE.value,
            jax.random.choice(rng_stone, for_stone),
            new_block,
        )
        new_block = jax.lax.select(
            block_to_add == BlockType.COAL.value,
            jax.random.choice(rng_coal, for_coal),
            new_block,
        )
        new_block = jax.lax.select(
            block_to_add == BlockType.DIAMOND.value,
            jax.random.choice(rng_diamond, for_diamond),
            new_block,
        )
        new_block = jax.lax.select(
            block_to_add == BlockType.IRON.value,
            jax.random.choice(rng_iron, for_iron),
            new_block,
        )

        r1, c1 = get_random_index_of_block(rng1, level, block_to_add, is_middle=True)
        r2, c2 = get_random_index_of_block(rng2, level, new_block, is_middle=False)

        level = level.replace(map=level.map.at[0, r1, c1].set(new_block))
        return level.replace(map=level.map.at[0, r2, c2].set(block_to_add))

    for_stone = jnp.array(
        [b.value for b in [BlockType.COAL, BlockType.IRON, BlockType.DIAMOND]]
    )
    for_coal = jnp.array(
        [b.value for b in [BlockType.STONE, BlockType.IRON, BlockType.DIAMOND]]
    )
    for_diamond = jnp.array(
        [b.value for b in [BlockType.STONE, BlockType.IRON, BlockType.COAL]]
    )
    for_iron = jnp.array(
        [b.value for b in [BlockType.STONE, BlockType.DIAMOND, BlockType.COAL]]
    )
    good_blocks = [
        BlockType.GRASS,
        BlockType.STONE,
        BlockType.TREE,
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
    ]
    good_blocks_to_choose_from = jnp.array([b.value for b in good_blocks])

    def mutate_level(rng: chex.PRNGKey, level: Level, n: int = 1) -> Level:
        def _single_mutate(carry, _):
            rng, level = carry
            rng, rng_step = jax.random.split(rng)
            level = single_step(rng_step, level)
            return (rng, level), None

        (_, level), _ = jax.lax.scan(
            _single_mutate, (rng, level), None, length=_normalize_num_edits(n)
        )
        return level

    return mutate_level


def make_mutator_craftax_mutate_angles(
    generate_world: Callable[..., Level],
    static_env_params: Any,
    env_params: EnvParams,
):
    def mutate_level(rng: chex.PRNGKey, level: Level, n: int = 1) -> Level:
        del n
        rng, *rngs = jax.random.split(rng, 5)
        new_angles = jax.tree_util.tree_map(
            lambda x, key: jnp.clip(
                x + jax.random.uniform(key, x.shape, minval=-0.2, maxval=0.2),
                0,
                1,
            ),
            level.fractal_noise_angles,
            tuple(rngs),
        )
        return generate_world(
            rng,
            env_params.replace(fractal_noise_angles=new_angles),
            static_env_params,
        ).replace(fractal_noise_angles=new_angles)

    return mutate_level


def make_craftax_symbolic_mutator(
    mutation: str,
    *,
    generate_world: Callable[..., Level],
    static_env_params: Any,
    env_params: EnvParams,
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    if mutation == "noise":
        return make_mutator_craftax_mutate_angles(
            generate_world, static_env_params, env_params
        )
    if mutation == "swap_restricted":
        return make_mutator_craftax_swap_restricted(
            static_env_params, one_should_be_middle=True
        )
    if mutation == "swap":
        return make_mutator_craftax_swap(static_env_params, only_middle=True)
    raise ValueError(f"Unknown mutation type: {mutation}")
