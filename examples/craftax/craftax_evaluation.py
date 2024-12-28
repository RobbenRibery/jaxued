import jax
import jax.numpy as jnp
import chex
from functools import partial

from craftax.craftax.constants import BlockType, ItemType
from craftax.craftax.craftax_state import EnvState, StaticEnvParams

# pick 
@jax.jit
def compute_natural_resource_density_all_level(env_state:EnvState) -> float:
    """
    Evaluate resource density of the environment.
    
    Considers:
    - Tree density
    - Ore spawn rates
    
    Returns a float between 0 and 1, 
    where 0 means no resources are available and 1 means all resources are available.
    """
    resource_blocks = jnp.array([
        BlockType.WOOD.value,
        BlockType.TREE.value,
        BlockType.STONE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
        BlockType.SAPPHIRE.value,
        BlockType.RUBY.value
    ])

    @jax.jit 
    def _level_func(map_layer:chex.Array) -> float:
        presented_resources = jax.vmap(
            lambda layer, block: (layer == block).sum(),
            in_axes= (None, 0)
        )(map_layer, resource_blocks)
        return presented_resources.sum()
    
    return (
        (
            jax.vmap(_level_func)(env_state.map)
        ).sum()
    )/env_state.map.size

# pick 
@jax.jit
def compute_path_density_all_level(env_state:EnvState) -> float:
    """Measure the density of the walkable path
    In put restrcit to a single env state 
    """
    walkable_blocks = jnp.array([
        BlockType.PATH.value,
        BlockType.GRASS.value,
        BlockType.FIRE_GRASS.value,
        BlockType.ICE_GRASS.value,
        BlockType.SAND.value,
    ])

    @jax.jit
    def _level_func(map_layer:chex.Array) -> float:
        presented_paths = jax.vmap(
            lambda layer, block: (layer == block).sum(),
            in_axes= (None, 0)
        )(map_layer, walkable_blocks)
        return presented_paths.sum()
    
    return (
        (
            jax.vmap(_level_func)(env_state.map)
        ).sum()
    )/env_state.map.size

# pick 
@jax.jit
def compute_survival_resource_density_all_level(env_state:EnvState) -> float:
    """Evaluate balance of survival resources."""

    map_layer, item_layer = env_state.map, env_state.item_map
    water_sources = (map_layer == BlockType.WATER.value).sum()
    food_sources = ((map_layer == BlockType.PLANT.value) | (map_layer == BlockType.RIPE_PLANT.value)).sum()
    torches = (item_layer == ItemType.TORCH.value).sum()
    
    total_sources = water_sources + food_sources + torches
    max_possible = map_layer.size * 0.2  # Assuming 30% is the maximum density you could every have density
    return jnp.minimum(total_sources / max_possible, 1.0)

# pick 
@jax.jit
def compute_crafting_potential_all_level(env_state:EnvState) -> float:
    """Evaluate crafting opportunities."""
    crafting_blocks = jnp.array([
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.ENCHANTMENT_TABLE_FIRE.value,
        BlockType.ENCHANTMENT_TABLE_ICE.value
    ])

    @jax.jit 
    def _level_func(map_layer:chex.Array) -> float:
        presented_resources = jax.vmap(
            lambda layer, block: (layer == block).sum(),
            in_axes= (None, 0)
        )(map_layer, crafting_blocks)
        return presented_resources.sum()
    
    crafting_density = (
        (
            jax.vmap(_level_func)(env_state.map)
        ).sum()
    )/(env_state.map.size * 0.05)  # Assuming 5% is the maximum density you could every have density
    return jnp.minimum(crafting_density, 1.0)

# pick
@jax.jit
def compute_progression_potential(state: EnvState) -> float:
    """
    Evaluate potential for character progression.
    Considers:
    - Crafting station availability
    - Resource tier progression
    - XP sources
    """
    map_layer = state.map[state.player_level]
    
    # Crafting progression
    crafting_blocks = jnp.array([
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.ENCHANTMENT_TABLE_FIRE.value,
        BlockType.ENCHANTMENT_TABLE_ICE.value
    ])
    crafting_score = jax.vmap(
        lambda layer, block: (layer == block).sum(),
        in_axes=(None, 0)
    )(map_layer, crafting_blocks).sum()
    
    # Resource tiers
    resource_tiers = {
        1: [BlockType.WATER.value, BlockType.PLANT.value],
        2: [BlockType.TREE.value, BlockType.STONE.value],
        3: [BlockType.COAL.value, BlockType.IRON.value],
        4: [BlockType.DIAMOND.value, BlockType.SAPPHIRE.value, BlockType.RUBY.value],
    }
    
    tier_scores = jnp.zeros(4)
    for tier_idx, blocks in resource_tiers.items():
        blocks = jnp.array(blocks)
        tier_scores = tier_scores.at[tier_idx-1].set(
            (
                jax.vmap(
                    lambda layer, block: (layer == block).sum(),
                    in_axes=(None, 0),
                )(map_layer, blocks)
            ).sum()
        )
    
    # Weight higher tiers more
    tier_weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    resource_score = jnp.sum(tier_scores * tier_weights)
    
    return (
        0.5 * crafting_score/map_layer.size + \
        0.5 * jnp.minimum(resource_score/map_layer.size * 5, 1.0)
    )

# Don't use for now
@jax.jit
def compute_risk_reward_balance(state: EnvState) -> float:
    """
    # TODO: Make this function more realistic such 
    that it does not only consider the strongest mobs
    Comprehensive evaluation of risk-reward balance.
    
    Considers:
    1. Resource value vs danger proximity
    2. Safe zones vs hazard zones
    3. Progressive difficulty scaling
    4. Resource accessibility
    5. Survival resource distribution
    """
    map_layer = state.map[state.player_level]
    mob_layer = state.mob_map[state.player_level]
    
    # 1. Resource Value Mapping
    resource_value_map = jnp.zeros_like(map_layer, dtype=jnp.float32)
    resource_values = {
        BlockType.WOOD.value: 1.0,
        BlockType.COAL.value: 2.0,
        BlockType.IRON.value: 3.0,
        BlockType.DIAMOND.value: 4.0,
        BlockType.SAPPHIRE.value: 4.5,
        BlockType.RUBY.value: 4.5,
        BlockType.CHEST.value: 5.0,
    }
    
    for block_type, value in resource_values.items():
        resource_value_map += (map_layer == block_type) * value
    
    # 2. Danger Mapping
    danger_value_map = jnp.zeros_like(map_layer, dtype=jnp.float32)
    danger_values = {
        BlockType.LAVA.value: 4.0,
        BlockType.NECROMANCER.value: 5.0,
        BlockType.GRAVE.value: 3.0,
        BlockType.GRAVE2.value: 3.5,
        BlockType.GRAVE3.value: 4.0,
    }
    #state.melee_mobs.
    
    for block_type, value in danger_values.items():
        danger_value_map += (map_layer == block_type) * value
    
    # Add mob danger
    danger_value_map += mob_layer * 3.0

    # 3. Safe Zone Analysis
    safe_blocks = jnp.array([
        BlockType.PATH.value,
        BlockType.GRASS.value,
        BlockType.FIRE_GRASS.value,
        BlockType.ICE_GRASS.value,
    ])
    safe_zones = sum(map_layer == block for block in safe_blocks)
    
    # 4. Distance-based Risk Scaling
    kernel_small = jnp.ones((3, 3))
    kernel_medium = jnp.ones((5, 5))
    kernel_large = jnp.ones((7, 7))
    
    # Compute danger proximity at different scales
    danger_proximity_small = jax.scipy.signal.convolve2d(
        danger_value_map, kernel_small, mode='same'
    )
    danger_proximity_medium = jax.scipy.signal.convolve2d(
        danger_value_map, kernel_medium, mode='same'
    )
    danger_proximity_large = jax.scipy.signal.convolve2d(
        danger_value_map, kernel_large, mode='same'
    )
    
    # Weighted danger proximity
    danger_proximity = (
        0.5 * danger_proximity_small +
        0.3 * danger_proximity_medium +
        0.2 * danger_proximity_large
    )
    
    # 5. Resource Accessibility Score
    resource_proximity = jax.scipy.signal.convolve2d(
        resource_value_map, kernel_medium, mode='same'
    )

    # 6. Risk-Reward Correlation
    risk_reward_correlation = jnp.corrcoef(
        danger_proximity.flatten(), 
        resource_proximity.flatten()
    )[0, 1]
    
    # 7. Survival Resource Distribution
    survival_blocks = jnp.array([
        BlockType.WATER.value,
        BlockType.FOUNTAIN.value,
        BlockType.RIPE_PLANT.value
    ])
    survival_resources = sum(map_layer == block for block in survival_blocks)
    survival_distribution = jax.scipy.signal.convolve2d(
        survival_resources, kernel_medium, mode='same'
    )
    
    # Compute final scores
    risk_reward_score = jnp.abs(risk_reward_correlation)  # Want moderate correlation
    safe_zone_score = safe_zones.sum() / map_layer.size
    survival_score = jnp.minimum(survival_resources.sum() / map_layer.size * 10, 1.0)
    
    # Weight and combine scores
    weights = jnp.array([0.4, 0.3, 0.3])
    scores = jnp.array([
        risk_reward_score,
        safe_zone_score,
        survival_score
    ])
    
    return jnp.sum(weights * scores)

# Don't use for now
@jax.jit
def compute_exploration_incentives(state: EnvState) -> float:
    """
    Evaluate how well the environment encourages exploration.
    Considers:
    - Points of interest distribution
    - Path complexity
    - Hidden resources
    """
    map_layer = state.map[state.player_level]
    #print(map_layer.shape)
    
    # Points of interest
    poi_blocks = jnp.array([
        BlockType.CHEST.value,
        BlockType.FOUNTAIN.value,
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value
    ])
    poi_map = sum(map_layer == block for block in poi_blocks)
    #print(poi_map.shape)
    
    # Compute POI distribution evenness
    kernel = jnp.ones((5, 5))
    poi_density = jax.scipy.signal.convolve2d(poi_map, kernel, mode='same')
    distribution_evenness = 1.0 - (jnp.std(poi_density) / jnp.mean(poi_density + 1e-6))
    
    # Path complexity
    path_blocks = jnp.array([
        BlockType.PATH.value,
        BlockType.GRASS.value,
        BlockType.FIRE_GRASS.value,
        BlockType.ICE_GRASS.value,
        BlockType.SAND.value,
    ])
    path_map = sum(map_layer == block for block in path_blocks)
    
    # Measure path branching
    kernel_cross = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    path_connections = jax.scipy.signal.convolve2d(path_map, kernel_cross, mode='same')
    branching_score = jnp.sum(path_connections > 2) / path_map.sum()
    
    return (
        0.4 * distribution_evenness + \
        0.3 * branching_score + \
        0.3 * jnp.minimum(poi_map.sum() / map_layer.size * 10, 1.0)
    )

# pick 
@jax.jit 
def compute_mob_challengeness(state: EnvState) -> float:

    # Melee mobs
    melee_challenge = jnp.sum(
        state.melee_mobs.health[state.player_level] * \
        (state.melee_mobs.type_id[state.player_level] + 1)  # Higher ID = harder
    )
    
    # Ranged mobs
    ranged_challenge = jnp.sum(
        state.ranged_mobs.health[state.player_level] * \
        (state.ranged_mobs.type_id[state.player_level] + 1.5)  # Ranged bonus
    )
    
    return (melee_challenge + ranged_challenge)/2


@partial(jax.jit, static_argnames=['static_params'])
def compute_combat_challenge(
    state: EnvState,
    static_params: StaticEnvParams
) -> float:
    """
    Evaluate the combat challenge level of the environment.
    
    Considers:
    1. Mob type distribution and difficulty
    2. Combat space constraints
    3. Mob positioning and density
    4. Projectile combat potential
    5. Combat resource availability
    6. Escape route availability
    """
    map_layer = state.map[state.player_level]
    
    # 1. Mob Challenge Scoring
    def compute_mob_challenge():
        # Melee mobs
        melee_challenge = jnp.sum(
            state.melee_mobs.mask[state.player_level] *
            state.melee_mobs.health[state.player_level] *
            (state.melee_mobs.type_id[state.player_level] + 1)  # Higher ID = harder
        )
        
        # Ranged mobs
        ranged_challenge = jnp.sum(
            state.ranged_mobs.mask[state.player_level] *
            state.ranged_mobs.health[state.player_level] *
            (state.ranged_mobs.type_id[state.player_level] + 1.5)  # Ranged bonus
        )
        
        # Projectiles
        projectile_challenge = jnp.sum(
            state.mob_projectiles.mask[state.player_level] *
            (state.mob_projectiles.type_id[state.player_level] + 1)
        )
        
        return melee_challenge, ranged_challenge, projectile_challenge
    
    melee_challenge, ranged_challenge, projectile_challenge = compute_mob_challenge()
    
    # 2. Combat Space Analysis
    combat_blocks = jnp.array([
        BlockType.PATH.value,
        BlockType.GRASS.value,
        BlockType.FIRE_GRASS.value,
        BlockType.ICE_GRASS.value
    ])
    
    combat_space = sum(map_layer == block for block in combat_blocks)
    
    # 3. Combat Constraints
    kernel = jnp.ones((3, 3))
    wall_blocks = (map_layer == BlockType.WALL.value)
    combat_constraints = jax.scipy.signal.convolve2d(wall_blocks, kernel, mode='same')
    
    # 4. Strategic Position Analysis
    strategic_positions = jnp.logical_and(
        combat_space,
        combat_constraints > 0
    )
    
    # 5. Combat Resource Availability
    combat_resources = jnp.array([
        BlockType.CHEST.value,
        BlockType.FOUNTAIN.value
    ])
    resource_availability = sum(map_layer == block for block in combat_resources)
    
    # 6. Escape Route Analysis
    escape_routes = jax.scipy.signal.convolve2d(combat_space, kernel, mode='same')
    escape_score = jnp.sum(escape_routes >= 3) / combat_space.sum()
    
    # Normalize scores
    total_mob_challenge = (
        melee_challenge + 
        ranged_challenge * 1.5 +  # Ranged mobs are harder
        projectile_challenge
    )

    normalized_mob_challenge = jnp.minimum(
        total_mob_challenge / (static_params.max_melee_mobs * 5), 
        1.0
    )
    
    combat_space_score = combat_space.sum() / map_layer.size
    strategic_position_score = strategic_positions.sum() / combat_space.sum()
    resource_score = jnp.minimum(
        resource_availability / (map_layer.size * 0.05),  # Expect 5% resources
        1.0
    )
    
    # Weight and combine scores
    weights = jnp.array([0.3, 0.2, 0.2, 0.15, 0.15])
    scores = jnp.array([
        normalized_mob_challenge,
        combat_space_score,
        strategic_position_score,
        resource_score,
        escape_score
    ])
    
    return jnp.sum(weights * scores)