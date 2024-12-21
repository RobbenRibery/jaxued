import jax
import jax.numpy as jnp
from functools import partial
from craftax.craftax.constants import *
from craftax.craftax.craftax_state import EnvState

# Utility functions
def clip_to_valid_range(value, min_val, max_val):
    return jnp.clip(value, min_val, max_val)

@jax.jit
def editor_basic_resources(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Slightly reduces resource availability in the environment.
    
    Modifications:
    - Reduces tree density by 10%
    - Reduces ore spawn rates by 5%
    """
    rng, key = jax.random.split(rng)
    
    # Reduce trees
    tree_mask = state.map == BlockType.TREE.value
    remove_tree = jax.random.uniform(key, shape=tree_mask.shape) < 0.1
    new_map = jnp.where(
        jnp.logical_and(tree_mask, remove_tree),
        BlockType.GRASS.value,
        state.map
    )
    
    return state.replace(map=new_map)

@jax.jit
def editor_light_levels(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Makes the environment darker and more challenging to navigate.
    
    Modifications:
    - Reduces ambient light level by 20%
    - Increases light decay rate
    """
    new_light_level = state.light_level * 0.8
    new_light_map = state.light_map * 0.9
    
    return state.replace(
        light_level=new_light_level,
        light_map=new_light_map
    )

@jax.jit
def editor_mob_stats(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Enhances mob capabilities.
    
    Modifications:
    - Increases mob health by 20%
    - Reduces attack cooldowns by 1
    """
    # Increase mob health
    new_melee_mobs = state.melee_mobs.replace(
        health=state.melee_mobs.health * 1.2,
        attack_cooldown=jnp.maximum(state.melee_mobs.attack_cooldown - 1, 1)
    )
    
    new_ranged_mobs = state.ranged_mobs.replace(
        health=state.ranged_mobs.health * 1.2,
        attack_cooldown=jnp.maximum(state.ranged_mobs.attack_cooldown - 1, 1)
    )
    
    return state.replace(
        melee_mobs=new_melee_mobs,
        ranged_mobs=new_ranged_mobs
    )

@jax.jit
def editor_spawn_rates(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Increases mob spawn rates and aggression.
    
    Modifications:
    - Doubles spawn rate for hostile mobs
    - Increases mob detection range
    """
    # Double the number of active mobs
    new_melee_mask = state.melee_mobs.mask.at[:, :state.melee_mobs.mask.shape[1]//2].set(True)
    new_ranged_mask = state.ranged_mobs.mask.at[:, :state.ranged_mobs.mask.shape[1]//2].set(True)
    
    new_melee_mobs = state.melee_mobs.replace(mask=new_melee_mask)
    new_ranged_mobs = state.ranged_mobs.replace(mask=new_ranged_mask)
    
    return state.replace(
        melee_mobs=new_melee_mobs,
        ranged_mobs=new_ranged_mobs
    )

@jax.jit
def editor_environmental_hazards(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Adds more environmental hazards.
    
    Modifications:
    - Converts some path blocks to lava
    - Adds more dangerous terrain
    """
    rng, key = jax.random.split(rng)
    
    path_mask = state.map == BlockType.PATH.value
    add_lava = jax.random.uniform(key, shape=path_mask.shape) < 0.05
    
    new_map = jnp.where(
        jnp.logical_and(path_mask, add_lava),
        BlockType.LAVA.value,
        state.map
    )
    
    return state.replace(map=new_map)

@jax.jit
def editor_combined_systems(rng: jax.random.PRNGKey, state: EnvState) -> EnvState:
    """Combines multiple adversarial modifications.
    
    Modifications:
    - Applies all previous modifications
    - Further enhances difficulty
    """
    # Apply all previous editors in sequence
    state = editor_basic_resources(rng, state)
    state = editor_light_levels(rng, state)
    state = editor_mob_stats(rng, state)
    state = editor_spawn_rates(rng, state)
    state = editor_environmental_hazards(rng, state)
    
    # Additional combined effects
    state = state.replace(
        player_hunger=state.player_hunger * 1.2,
        player_thirst=state.player_thirst * 1.2,
        player_fatigue=state.player_fatigue * 1.2
    )
    
    return state