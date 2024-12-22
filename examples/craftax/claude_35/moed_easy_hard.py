# imports
import jax
import jax.numpy as jnp
import chex
from functools import partial
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import *

# utility functions
def modify_mob_stats(mobs, health_mult, cooldown_mult):
    """Helper to modify mob statistics"""
    return mobs.replace(
        health= mobs.health * health_mult,
        attack_cooldown= (mobs.attack_cooldown * cooldown_mult).astype(jnp.int32)
    )

# Mixture of Editors (Easy to Hard)

@jax.jit
def editor_abundant_resources(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game easier by increasing resource availability"""
    # Double the occurrence of resource blocks
    new_map = jnp.where(
        state.map == BlockType.TREE.value,
        BlockType.TREE.value,
        state.map
    )
    new_map = jnp.where(
        state.map == BlockType.IRON.value,
        BlockType.IRON.value,
        new_map
    )
    return state.replace(map=new_map)

@jax.jit
def editor_peaceful_mobs(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game easier by reducing mob difficulty"""
    health_mult = 0.75
    cooldown_mult = 1.5
    return state.replace(
        melee_mobs=modify_mob_stats(state.melee_mobs, health_mult, cooldown_mult),
        ranged_mobs=modify_mob_stats(state.ranged_mobs, health_mult, cooldown_mult)
    )

@jax.jit
def editor_lenient_survival(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game easier by reducing survival pressures"""
    return state.replace(
        player_hunger=state.player_hunger * 0.75,
        player_thirst=state.player_thirst * 0.75,
        player_fatigue=state.player_fatigue * 0.75
    )

@jax.jit
def editor_scarce_resources(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game harder by reducing resource availability"""
    # Reduce resource blocks by 25%
    rng, _rng = jax.random.split(rng)
    resource_mask = jax.random.uniform(_rng, state.map.shape) > 0.25
    new_map = jnp.where(
        jnp.logical_and(
            state.map == BlockType.TREE.value,
            resource_mask
        ),
        BlockType.GRASS.value,
        state.map
    )
    return state.replace(map=new_map)

@jax.jit
def editor_aggressive_mobs(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game harder by increasing mob difficulty"""
    health_mult = 1.5
    cooldown_mult = 0.75
    return state.replace(
        melee_mobs=modify_mob_stats(state.melee_mobs, health_mult, cooldown_mult),
        ranged_mobs=modify_mob_stats(state.ranged_mobs, health_mult, cooldown_mult)
    )

@jax.jit
def editor_harsh_survival(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game harder by increasing survival pressures"""
    return state.replace(
        player_hunger=state.player_hunger * 1.25,
        player_thirst=state.player_thirst * 1.25,
        player_fatigue=state.player_fatigue * 1.25
    )

@jax.jit
def editor_extreme_environment(rng: chex.PRNGKey, state: EnvState) -> EnvState:
    """Makes the game harder through environmental challenges"""
    # Add more hazards and reduce light
    new_light_level = state.light_level * 0.75
    rng, _rng = jax.random.split(rng)
    hazard_mask = jax.random.uniform(_rng, state.map.shape) > 0.9
    new_map = jnp.where(
        jnp.logical_and(
            state.map == BlockType.PATH.value,
            hazard_mask
        ),
        BlockType.LAVA.value,
        state.map
    )
    return state.replace(
        map=new_map,
        light_level=new_light_level
    )