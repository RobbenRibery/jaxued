"""Persistent-ensemble PLR for Maze using virtual learning progress.

Each ensemble member has independently initialized parameters and optimizer
state.  For every candidate level all members start from the same instantiated
environment state, collect their own stochastic on-policy trajectory, and
train only on that trajectory.

The level score is the signed reduction in ensemble action-distribution
disagreement after an isolated virtual PPO update.  Its support is frozen to
physical states visited during the original rollouts.  Post-update recurrent
states are reconstructed from zero by replaying the stored observation/reset
sequence; the environment is never stepped during this replay.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from enum import IntEnum
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import serialization, struct
from flax.training.train_state import TrainState as BaseTrainState

from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import (
    Level,
    make_level_generator,
    make_level_mutator_minimax,
)
from jaxued.environments.underspecified_env import (
    EnvParams,
    EnvState,
    Observation,
    UnderspecifiedEnv,
)
from jaxued.metrics import (
    EnsembleDisagreementInputs,
    EnsembleDisagreementResult,
    aggregate_state_action_probabilities,
    compute_ensemble_disagreement_reduction,
    compute_max_returns,
)
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

try:  # Supports both ``python examples/...`` and importing from the repo root.
    from maze_plr import (
        ActorCritic,
        compute_gae,
        evaluate_rnn,
        update_actor_critic_rnn,
    )
except ImportError:  # pragma: no cover - exercised by package-style imports.
    from examples.maze_plr import (
        ActorCritic,
        compute_gae,
        evaluate_rnn,
        update_actor_critic_rnn,
    )


CHECKPOINT_FORMAT = "jaxued-maze-ensemble-v1"


class UpdateState(IntEnum):
    """Most recent PLR update branch."""

    DR = 0
    REPLAY = 1


@struct.dataclass
class EnsembleTrainState:
    """Persistent policies and shared PLR state.

    ``agents`` is a stacked :class:`flax.training.train_state.TrainState`:
    every array leaf has a leading ensemble-member axis.  Consequently each
    member retains independent parameters, optimizer moments, and optimizer
    step while all members share the level sampler.
    """

    agents: BaseTrainState
    sampler: chex.ArrayTree
    update_state: chex.Array
    num_dr_updates: chex.Array
    num_replay_updates: chex.Array
    num_mutation_updates: chex.Array
    dr_last_level_batch: chex.ArrayTree
    replay_last_level_batch: chex.ArrayTree
    mutation_last_level_batch: chex.ArrayTree


class PPOParameters(NamedTuple):
    """PPO constants shared by persistent and virtual member updates."""

    gamma: float
    gae_lambda: float
    num_minibatches: int
    num_epochs: int
    clip_eps: float
    entropy_coeff: float
    critic_coeff: float


class EnsembleTrajectory(NamedTuple):
    """Frozen decision-point data for all ensemble trajectories.

    Every field has leading dimensions ``(policies, time, levels)``.  The
    observation, reset flag, action probabilities, and physical ``state_ids``
    at one index all describe the same pre-action decision point.
    """

    observations: Observation
    actions: chex.Array
    rewards: chex.Array
    dones: chex.Array
    resets: chex.Array
    log_probabilities: chex.Array
    values: chex.Array
    action_probabilities: chex.Array
    state_ids: chex.Array


class EnsembleRollout(NamedTuple):
    """An ensemble trajectory plus bootstrapping values and advanced keys."""

    rngs: chex.Array
    trajectory: EnsembleTrajectory
    last_values: chex.Array


class ScoredEnsembleRollout(NamedTuple):
    """Virtual-update output aligned to one frozen ensemble rollout."""

    disagreement: EnsembleDisagreementResult
    post_update_action_probabilities: chex.Array


def encode_maze_state(env_state: Any, max_width: int) -> chex.Array:
    """Encode a wrapped Maze pose ``z = (x, y, direction)`` as an integer.

    Elapsed time and all other dynamic fields are intentionally excluded.  The
    encoding is ``4 * (y * max_width + x) + direction``.

    Args:
        env_state: Batched ``AutoReplayState`` whose ``env_state`` field is a
            Maze environment state.
        max_width: Static width of the Maze coordinate system.

    Returns:
        Integer identifiers with the same leading batch shape as ``agent_dir``.
    """
    maze_state = env_state.env_state
    x = maze_state.agent_pos[..., 0].astype(jnp.int32)
    y = maze_state.agent_pos[..., 1].astype(jnp.int32)
    direction = jnp.asarray(maze_state.agent_dir, dtype=jnp.int32)
    return 4 * (y * max_width + x) + direction


def collect_policy_trajectory(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    agent: BaseTrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_levels: int,
    num_steps: int,
    max_width: int,
) -> tuple[chex.PRNGKey, EnsembleTrajectory, chex.Array]:
    """Collect one member's on-policy rollout on a shared batch of levels.

    State identifiers and action distributions are captured before each
    action.  Independent action and environment keys are derived from this
    member's rollout key at every step.
    """

    def _sample_step(carry, _):
        step_rng, hstate, obs, env_state, previous_done = carry
        step_rng, action_rng, env_rng = jax.random.split(step_rng, 3)

        network_inputs = jax.tree_util.tree_map(
            lambda value: value[None, ...],
            (obs, previous_done),
        )
        next_hstate, policy, value = agent.apply_fn(
            agent.params,
            network_inputs,
            hstate,
        )
        action = policy.sample(seed=action_rng).squeeze(0)
        log_probability = policy.log_prob(action[None, ...]).squeeze(0)
        action_probabilities = policy.probs.squeeze(0)
        value = value.squeeze(0)
        state_id = encode_maze_state(env_state, max_width)

        next_obs, next_env_state, reward, done, _ = jax.vmap(
            env.step,
            in_axes=(0, 0, 0, None),
        )(
            jax.random.split(env_rng, num_levels),
            env_state,
            action,
            env_params,
        )

        next_carry = (
            step_rng,
            next_hstate,
            next_obs,
            next_env_state,
            done,
        )
        transition = EnsembleTrajectory(
            observations=obs,
            actions=action,
            rewards=reward,
            dones=done,
            resets=previous_done,
            log_probabilities=log_probability,
            values=value,
            action_probabilities=action_probabilities,
            state_ids=state_id,
        )
        return next_carry, transition

    initial_done = jnp.zeros((num_levels,), dtype=bool)
    (rng, hstate, last_obs, _, last_done), trajectory = jax.lax.scan(
        _sample_step,
        (rng, init_hstate, init_obs, init_env_state, initial_done),
        None,
        length=num_steps,
    )

    final_inputs = jax.tree_util.tree_map(
        lambda value: value[None, ...],
        (last_obs, last_done),
    )
    _, _, last_value = agent.apply_fn(agent.params, final_inputs, hstate)
    return rng, trajectory, last_value.squeeze(0)


def collect_ensemble_trajectories(
    rollout_rngs: chex.Array,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    agents: BaseTrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    num_levels: int,
    num_steps: int,
    max_width: int,
) -> EnsembleRollout:
    """Collect independent member rollouts from the same initial states."""

    def _collect_member(member_rng, agent):
        return collect_policy_trajectory(
            member_rng,
            env,
            env_params,
            agent,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            num_levels,
            num_steps,
            max_width,
        )

    rngs, trajectories, last_values = jax.vmap(_collect_member)(
        rollout_rngs,
        agents,
    )
    return EnsembleRollout(rngs, trajectories, last_values)


def replay_action_probabilities(
    agent: BaseTrainState,
    observations: Observation,
    resets: chex.Array,
    init_hstate: chex.ArrayTree,
) -> chex.Array:
    """Reconstruct recurrent state and action probabilities on stored inputs.

    The carry is recomputed by scanning the supplied observation/reset sequence
    from ``init_hstate``.  This is sufficient for the current Maze actor because
    its recurrent input contains only the current observation; previous actions
    and rewards are not network inputs.
    """
    _, policy, _ = agent.apply_fn(
        agent.params,
        (observations, resets),
        init_hstate,
    )
    return policy.probs


def virtual_update_action_probabilities(
    virtual_update_rngs: chex.Array,
    agents: BaseTrainState,
    trajectory: EnsembleTrajectory,
    last_values: chex.Array,
    ppo: PPOParameters,
) -> chex.Array:
    """Run isolated one-level PPO updates and replay their frozen probes.

    Each virtual update begins from the corresponding persistent member's
    complete parameter *and optimizer* state.  A member-level clone trains only
    on that member's trajectory for that level.  Updated clones are discarded;
    only post-update action probabilities are returned.

    Returns:
        Probabilities with shape ``(policies, time, levels, actions)``.
    """

    def _update_member(agent, member_trajectory, member_last_values, member_rngs):
        # ``lax.map`` bounds peak memory: every level starts from the same
        # closed-over persistent agent rather than carrying virtual state on to
        # the next level.
        trajectories_by_level = jax.tree_util.tree_map(
            lambda value: value.swapaxes(0, 1),
            member_trajectory,
        )

        def _update_level(inputs):
            level_rng, level_trajectory, last_value = inputs
            singleton_trajectory = jax.tree_util.tree_map(
                lambda value: value[:, None, ...],
                level_trajectory,
            )
            advantages, targets = compute_gae(
                ppo.gamma,
                ppo.gae_lambda,
                last_value[None],
                singleton_trajectory.values,
                singleton_trajectory.rewards,
                singleton_trajectory.dones,
            )
            batch = (
                singleton_trajectory.observations,
                singleton_trajectory.actions,
                singleton_trajectory.dones,
                singleton_trajectory.log_probabilities,
                singleton_trajectory.values,
                targets,
                advantages,
            )
            (_, updated_agent), _ = update_actor_critic_rnn(
                level_rng,
                agent,
                ActorCritic.initialize_carry((1,)),
                batch,
                num_envs=1,
                n_steps=singleton_trajectory.actions.shape[0],
                n_minibatch=1,
                n_epochs=ppo.num_epochs,
                clip_eps=ppo.clip_eps,
                entropy_coeff=ppo.entropy_coeff,
                critic_coeff=ppo.critic_coeff,
                update_grad=True,
            )
            post_update_probabilities = replay_action_probabilities(
                updated_agent,
                singleton_trajectory.observations,
                singleton_trajectory.resets,
                ActorCritic.initialize_carry((1,)),
            )
            return post_update_probabilities[:, 0, :]

        return jax.lax.map(
            _update_level,
            (member_rngs, trajectories_by_level, member_last_values),
        )

    probabilities_by_member_level = jax.vmap(_update_member)(
        agents,
        trajectory,
        last_values,
        virtual_update_rngs,
    )
    return probabilities_by_member_level.swapaxes(1, 2)


def score_ensemble_rollout(
    virtual_update_rngs: chex.Array,
    agents: BaseTrainState,
    trajectory: EnsembleTrajectory,
    last_values: chex.Array,
    ppo: PPOParameters,
    num_states: int,
) -> ScoredEnsembleRollout:
    """Score levels by virtual disagreement reduction on frozen state visits."""
    post_update_probabilities = virtual_update_action_probabilities(
        virtual_update_rngs,
        agents,
        trajectory,
        last_values,
        ppo,
    )
    before = aggregate_state_action_probabilities(
        trajectory.state_ids,
        trajectory.action_probabilities,
        num_states,
    )
    after = aggregate_state_action_probabilities(
        trajectory.state_ids,
        post_update_probabilities,
        num_states,
    )
    disagreement = compute_ensemble_disagreement_reduction(
        EnsembleDisagreementInputs(
            before.action_probabilities,
            after.action_probabilities,
            before.visited,
        )
    )
    return ScoredEnsembleRollout(disagreement, post_update_probabilities)


def update_ensemble_members(
    update_rngs: chex.Array,
    agents: BaseTrainState,
    trajectory: EnsembleTrajectory,
    last_values: chex.Array,
    ppo: PPOParameters,
    *,
    update_grad: bool,
) -> tuple[BaseTrainState, chex.ArrayTree]:
    """Apply PPO to every member using only that member's full rollout batch."""

    def _update_member(member_rng, agent, member_trajectory, last_value):
        advantages, targets = compute_gae(
            ppo.gamma,
            ppo.gae_lambda,
            last_value,
            member_trajectory.values,
            member_trajectory.rewards,
            member_trajectory.dones,
        )
        batch = (
            member_trajectory.observations,
            member_trajectory.actions,
            member_trajectory.dones,
            member_trajectory.log_probabilities,
            member_trajectory.values,
            targets,
            advantages,
        )
        (_, updated_agent), losses = update_actor_critic_rnn(
            member_rng,
            agent,
            ActorCritic.initialize_carry((member_trajectory.actions.shape[1],)),
            batch,
            num_envs=member_trajectory.actions.shape[1],
            n_steps=member_trajectory.actions.shape[0],
            n_minibatch=ppo.num_minibatches,
            n_epochs=ppo.num_epochs,
            clip_eps=ppo.clip_eps,
            entropy_coeff=ppo.entropy_coeff,
            critic_coeff=ppo.critic_coeff,
            update_grad=update_grad,
        )
        return updated_agent, losses

    return jax.vmap(_update_member)(
        update_rngs,
        agents,
        trajectory,
        last_values,
    )


def _as_plain_config(config: Any) -> dict[str, Any]:
    """Convert a W&B config or mapping to a serializable dictionary."""
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    return dict(config)


def setup_ensemble_checkpointing(
    config: Any,
) -> ocp.CheckpointManager:
    """Create the dedicated ensemble checkpoint directory and write metadata."""
    config_dict = _as_plain_config(config)
    config_dict["checkpoint_format"] = CHECKPOINT_FORMAT
    save_directory = os.path.join(
        os.getcwd(),
        "checkpoints",
        str(config_dict["run_name"]),
        str(config_dict["seed"]),
    )
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, "config.json"), "w") as config_file:
        json.dump(config_dict, config_file, indent=2)

    return ocp.CheckpointManager(
        os.path.join(save_directory, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config_dict["checkpoint_save_interval"],
            max_to_keep=config_dict["max_number_of_checkpoints"],
        ),
    )


def ensemble_train_state_to_log_dict(
    train_state: EnsembleTrainState,
    level_sampler: LevelSampler,
) -> dict[str, dict[str, Any]]:
    """Extract compact sampler and counter values without copying policies."""
    sampler = train_state.sampler
    populated = jnp.arange(level_sampler.capacity) < sampler["size"]
    safe_size = jnp.maximum(populated.sum(), 1)
    return {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (
                sampler["scores"] * level_sampler.level_weights(sampler)
            ).sum(),
            "level_sampler/mean_score": (
                sampler["scores"] * populated
            ).sum()
            / safe_size,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def _load_ensemble_eval_config(requested_config: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the configuration stored beside an ensemble checkpoint."""
    checkpoint_directory = requested_config.get("checkpoint_directory")
    if checkpoint_directory is None:
        raise ValueError("--checkpoint_directory is required in eval mode")
    with open(os.path.join(checkpoint_directory, "config.json")) as config_file:
        stored_config = json.load(config_file)
    if stored_config.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "The checkpoint is not a maze ensemble checkpoint with format "
            f"{CHECKPOINT_FORMAT!r}."
        )

    # Architecture and training dimensions come from the checkpoint.  Only
    # evaluation controls supplied by the current invocation are overridden.
    for name in (
        "mode",
        "checkpoint_directory",
        "checkpoint_to_eval",
        "eval_num_attempts",
    ):
        stored_config[name] = requested_config[name]
    return stored_config


def _conform_checkpoint_tree(template: Any, restored: Any) -> Any:
    """Conform Orbax's raw containers to a Flax state-dictionary schema.

    Orbax 0.5.3 restores tuples as lists and zero-leaf Optax ``EmptyState``
    nodes as ``None``.  Flax state dictionaries represent both as dictionaries
    (numeric string keys for tuples and an empty mapping for ``EmptyState``).
    """
    if isinstance(template, dict):
        if restored is None:
            restored = {}
        elif isinstance(restored, (list, tuple)):
            restored = {str(index): value for index, value in enumerate(restored)}
        if not isinstance(restored, dict):
            raise TypeError(
                "Checkpoint structure does not match the ensemble template: "
                f"expected a mapping, received {type(restored).__name__}."
            )
        return {
            key: _conform_checkpoint_tree(value, restored.get(key))
            for key, value in template.items()
        }
    if restored is None:
        raise TypeError("Checkpoint is missing a non-empty ensemble state leaf.")
    return restored


def restore_ensemble_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    template: EnsembleTrainState,
) -> EnsembleTrainState:
    """Restore all policies, optimizer states, and shared sampler state."""
    restored = checkpoint_manager.restore(step)
    template_state_dict = serialization.to_state_dict(template)
    conformed = _conform_checkpoint_tree(template_state_dict, restored)
    return serialization.from_state_dict(template, conformed)


def main(config: Any, project: str = "JAXUED_TEST") -> Any:
    """Train or evaluate persistent Maze policies with ensemble PLR scoring.

    Args:
        config: Mapping of command-line-compatible configuration values.
        project: Weights & Biases project name.

    Returns:
        The final :class:`EnsembleTrainState` in training mode.  Evaluation
        mode returns per-member states, returns, and episode lengths.
    """
    requested_config = _as_plain_config(config)
    if requested_config["mode"] == "eval":
        requested_config = _load_ensemble_eval_config(requested_config)
        os.environ.setdefault("WANDB_MODE", "disabled")

    if requested_config["num_agents"] < 2:
        raise ValueError("Ensemble disagreement requires at least two agents.")
    if (
        requested_config["num_train_envs"]
        % requested_config["num_minibatches"]
        != 0
    ):
        raise ValueError("num_train_envs must be divisible by num_minibatches")
    if requested_config["eval_freq"] <= 0:
        raise ValueError("eval_freq must be positive")

    tags = ["ensemble-disagreement"]
    if not requested_config["exploratory_grad_updates"]:
        tags.append("robust")
    tags.append("ACCEL" if requested_config["use_accel"] else "PLR")
    wandb.init(
        config=requested_config,
        project=project,
        group=requested_config["run_name"],
        tags=tags,
    )
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    for namespace in (
        "solve_rate/*",
        "return/*",
        "eval_ep_lengths/*",
        "level_sampler/*",
        "agent/*",
        "ensemble/*",
    ):
        wandb.define_metric(namespace, step_metric="num_updates")

    base_env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    eval_env = base_env
    sample_random_level = make_level_generator(
        base_env.max_height,
        base_env.max_width,
        config["n_walls"],
    )
    env_renderer = MazeRenderer(base_env, tile_size=8)
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)
    num_states = base_env.max_height * base_env.max_width * 4

    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={
            "temperature": config["temperature"],
            "k": config["topk_k"],
        },
        duplicate_check=config["buffer_duplicate_check"],
    )
    ppo = PPOParameters(
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        num_minibatches=config["num_minibatches"],
        num_epochs=config["epoch_ppo"],
        clip_eps=config["clip_eps"],
        entropy_coeff=config["entropy_coeff"],
        critic_coeff=config["critic_coeff"],
    )

    network = ActorCritic(env.action_space(env_params).n)

    def _linear_schedule(count):
        updates_completed = count // (
            config["num_minibatches"] * config["epoch_ppo"]
        )
        fraction_remaining = 1.0 - updates_completed / jnp.maximum(
            config["num_updates"],
            1,
        )
        return config["lr"] * fraction_remaining

    optimizer = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=_linear_schedule, eps=1e-5),
    )

    placeholder_level = sample_random_level(jax.random.PRNGKey(0))
    dummy_obs, _ = env.reset_to_level(
        jax.random.PRNGKey(1),
        placeholder_level,
        env_params,
    )
    dummy_obs = jax.tree_util.tree_map(
        lambda value: jnp.repeat(
            value[None, ...],
            config["num_train_envs"],
            axis=0,
        )[None, ...],
        dummy_obs,
    )
    dummy_inputs = (
        dummy_obs,
        jnp.zeros((1, config["num_train_envs"]), dtype=bool),
    )

    def create_ensemble_train_state(rng: chex.PRNGKey) -> EnsembleTrainState:
        """Create independently seeded members and one shared level buffer."""

        def _create_agent(member_rng):
            params = network.init(
                member_rng,
                dummy_inputs,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
            )
            return BaseTrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=optimizer,
            )

        member_init_rngs = jax.random.split(rng, config["num_agents"])
        agents = jax.vmap(_create_agent)(member_init_rngs)
        sampler = level_sampler.initialize(
            placeholder_level,
            {"max_return": -jnp.inf},
        )
        placeholder_batch = jax.tree_util.tree_map(
            lambda value: jnp.repeat(
                jnp.asarray(value)[None, ...],
                config["num_train_envs"],
                axis=0,
            ),
            placeholder_level,
        )
        return EnsembleTrainState(
            agents=agents,
            sampler=sampler,
            update_state=jnp.asarray(UpdateState.DR, dtype=jnp.int32),
            num_dr_updates=jnp.asarray(0, dtype=jnp.int32),
            num_replay_updates=jnp.asarray(0, dtype=jnp.int32),
            num_mutation_updates=jnp.asarray(0, dtype=jnp.int32),
            dr_last_level_batch=placeholder_batch,
            replay_last_level_batch=placeholder_batch,
            mutation_last_level_batch=placeholder_batch,
        )

    def _process_level_batch(
        rng: chex.PRNGKey,
        agents: BaseTrainState,
        levels: Level,
        *,
        update_grad: bool,
    ):
        """Roll out, score virtually, then optionally update persistent members."""
        (
            next_rng,
            reset_rng,
            rollout_rng,
            virtual_update_rng,
            persistent_update_rng,
        ) = jax.random.split(rng, 5)

        # Instantiate each candidate once.  These observations and states are
        # shared by all members; only their subsequent rollout streams differ.
        init_obs, init_env_state = jax.vmap(
            env.reset_to_level,
            in_axes=(0, 0, None),
        )(
            jax.random.split(reset_rng, config["num_train_envs"]),
            levels,
            env_params,
        )
        rollout = collect_ensemble_trajectories(
            jax.random.split(rollout_rng, config["num_agents"]),
            env,
            env_params,
            agents,
            init_obs,
            init_env_state,
            config["num_train_envs"],
            config["num_steps"],
            base_env.max_width,
        )

        virtual_update_rngs = jax.random.split(
            virtual_update_rng,
            config["num_agents"] * config["num_train_envs"],
        ).reshape(config["num_agents"], config["num_train_envs"], 2)
        scored_rollout = score_ensemble_rollout(
            virtual_update_rngs,
            agents,
            rollout.trajectory,
            rollout.last_values,
            ppo,
            num_states,
        )

        # This is a separate PPO pass from the virtual copies.  Every
        # persistent member consumes its own full on-policy level batch.
        updated_agents, losses = update_ensemble_members(
            jax.random.split(persistent_update_rng, config["num_agents"]),
            agents,
            rollout.trajectory,
            rollout.last_values,
            ppo,
            update_grad=update_grad,
        )
        member_max_returns = jax.vmap(compute_max_returns)(
            rollout.trajectory.dones,
            rollout.trajectory.rewards,
        )
        max_returns = member_max_returns.max(axis=0)
        disagreement = scored_rollout.disagreement
        metrics = {
            "losses": jax.tree_util.tree_map(jnp.mean, losses),
            "mean_num_blocks": levels.wall_map.sum()
            / config["num_train_envs"],
            "uncertainty_before": disagreement.mean_uncertainty_before.mean(),
            "uncertainty_after": disagreement.mean_uncertainty_after.mean(),
            "disagreement_reduction": disagreement.scores.mean(),
            "eligible_state_count": disagreement.eligible_state_count.mean(),
            "no_eligible_state_fraction": (
                disagreement.eligible_state_count == 0
            ).mean(),
        }
        return (
            next_rng,
            updated_agents,
            disagreement.scores,
            max_returns,
            metrics,
        )

    def train_step(carry: tuple[chex.PRNGKey, EnsembleTrainState], _):
        """Run one new-level, replay, or mutation update."""

        def _on_new_levels(rng, train_state):
            rng, level_rng = jax.random.split(rng)
            levels = jax.vmap(sample_random_level)(
                jax.random.split(level_rng, config["num_train_envs"])
            )
            rng, agents, scores, max_returns, metrics = _process_level_batch(
                rng,
                train_state.agents,
                levels,
                update_grad=config["exploratory_grad_updates"],
            )
            sampler, _ = level_sampler.insert_batch(
                train_state.sampler,
                levels,
                scores,
                {"max_return": max_returns},
            )
            train_state = train_state.replace(
                agents=agents,
                sampler=sampler,
                update_state=jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        def _on_replay_levels(rng, train_state):
            rng, level_rng = jax.random.split(rng)
            sampler, (level_indices, levels) = level_sampler.sample_replay_levels(
                train_state.sampler,
                level_rng,
                config["num_train_envs"],
            )
            rng, agents, scores, observed_max_returns, metrics = (
                _process_level_batch(
                    rng,
                    train_state.agents,
                    levels,
                    update_grad=True,
                )
            )
            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_indices)[
                    "max_return"
                ],
                observed_max_returns,
            )
            sampler = level_sampler.update_batch(
                sampler,
                level_indices,
                scores,
                {"max_return": max_returns},
            )
            train_state = train_state.replace(
                agents=agents,
                sampler=sampler,
                update_state=jnp.asarray(UpdateState.REPLAY, dtype=jnp.int32),
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        def _on_mutated_levels(rng, train_state):
            rng, mutation_rng = jax.random.split(rng)
            levels = jax.vmap(mutate_level, in_axes=(0, 0, None))(
                jax.random.split(mutation_rng, config["num_train_envs"]),
                train_state.replay_last_level_batch,
                config["num_edits"],
            )
            rng, agents, scores, max_returns, metrics = _process_level_batch(
                rng,
                train_state.agents,
                levels,
                update_grad=config["exploratory_grad_updates"],
            )
            sampler, _ = level_sampler.insert_batch(
                train_state.sampler,
                levels,
                scores,
                {"max_return": max_returns},
            )
            train_state = train_state.replace(
                agents=agents,
                sampler=sampler,
                update_state=jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, replay_decision_rng = jax.random.split(rng)
        replay = level_sampler.sample_replay_decision(
            train_state.sampler,
            replay_decision_rng,
        ).astype(jnp.int32)
        if config["use_accel"]:
            branch = jnp.where(
                train_state.update_state == UpdateState.REPLAY,
                2,
                replay,
            )
        else:
            branch = replay
        return jax.lax.switch(
            branch,
            (_on_new_levels, _on_replay_levels, _on_mutated_levels),
            rng,
            train_state,
        )

    eval_levels = Level.load_prefabs(config["eval_levels"])
    num_eval_levels = len(config["eval_levels"])

    def _evaluate_attempt(rng: chex.PRNGKey, agent: BaseTrainState):
        rng, reset_rng = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(
            eval_env.reset_to_level,
            in_axes=(0, 0, None),
        )(
            jax.random.split(reset_rng, num_eval_levels),
            eval_levels,
            env_params,
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            agent,
            ActorCritic.initialize_carry((num_eval_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        active = (
            jnp.arange(env_params.max_steps_in_episode)[:, None]
            < episode_lengths
        )
        returns = (rewards * active).sum(axis=0)
        return states, returns, episode_lengths

    def _evaluate_ensemble(
        rng: chex.PRNGKey,
        train_state: EnsembleTrainState,
    ):
        def _evaluate_member(member_rng, agent):
            return jax.vmap(_evaluate_attempt, in_axes=(0, None))(
                jax.random.split(member_rng, config["eval_num_attempts"]),
                agent,
            )

        return jax.vmap(_evaluate_member)(
            jax.random.split(rng, config["num_agents"]),
            train_state.agents,
        )

    def _make_train_and_eval_step(segment_length: int):
        @jax.jit
        def _train_and_eval_step(runner_state):
            (rng, train_state), training_metrics = jax.lax.scan(
                train_step,
                runner_state,
                None,
                length=segment_length,
            )
            training_metrics = jax.tree_util.tree_map(
                jnp.mean,
                training_metrics,
            )

            rng, eval_rng = jax.random.split(rng)
            states, returns, episode_lengths = _evaluate_ensemble(
                eval_rng,
                train_state,
            )
            solved = (returns > 0).astype(jnp.float32)
            aggregate_axes = (0, 1)  # members and stochastic attempts
            training_metrics["eval_returns_mean"] = returns.mean(
                axis=aggregate_axes
            )
            training_metrics["eval_returns_std"] = returns.std(
                axis=aggregate_axes
            )
            training_metrics["eval_solve_rates_mean"] = solved.mean(
                axis=aggregate_axes
            )
            training_metrics["eval_solve_rates_std"] = solved.std(
                axis=aggregate_axes
            )
            training_metrics["eval_ep_lengths_mean"] = episode_lengths.mean(
                axis=aggregate_axes
            )
            training_metrics["eval_ep_lengths_std"] = episode_lengths.std(
                axis=aggregate_axes
            )

            # Animation is deliberately policy 0, attempt 0 only.  Scalar
            # metrics above always aggregate every member and attempt.
            animation_states = jax.tree_util.tree_map(
                lambda value: value[0, 0],
                states,
            )
            animation_lengths = episode_lengths[0, 0]
            images = jax.vmap(
                jax.vmap(env_renderer.render_state, in_axes=(0, None)),
                in_axes=(0, None),
            )(animation_states, env_params)
            frames = images.transpose(0, 1, 4, 2, 3)
            training_metrics["eval_animation"] = (frames, animation_lengths)

            training_metrics["update_count"] = (
                train_state.num_dr_updates
                + train_state.num_replay_updates
                + train_state.num_mutation_updates
            )
            training_metrics["dr_levels"] = jax.vmap(
                env_renderer.render_level,
                in_axes=(0, None),
            )(train_state.dr_last_level_batch, env_params)
            training_metrics["replay_levels"] = jax.vmap(
                env_renderer.render_level,
                in_axes=(0, None),
            )(train_state.replay_last_level_batch, env_params)
            training_metrics["mutation_levels"] = jax.vmap(
                env_renderer.render_level,
                in_axes=(0, None),
            )(train_state.mutation_last_level_batch, env_params)

            highest_scoring_level = level_sampler.get_levels(
                train_state.sampler,
                train_state.sampler["scores"].argmax(),
            )
            highest_weighted_level = level_sampler.get_levels(
                train_state.sampler,
                level_sampler.level_weights(train_state.sampler).argmax(),
            )
            training_metrics["highest_scoring_level"] = (
                env_renderer.render_level(highest_scoring_level, env_params)
            )
            training_metrics["highest_weighted_level"] = (
                env_renderer.render_level(highest_weighted_level, env_params)
            )
            return (rng, train_state), training_metrics

        return _train_and_eval_step

    def _log_eval(stats, train_state_info, segment_length, elapsed):
        update_count = stats["update_count"]
        print(f"Logging update: {int(update_count)}")
        total_env_steps = (
            update_count
            * config["num_agents"]
            * config["num_train_envs"]
            * config["num_steps"]
        )
        segment_env_steps = (
            segment_length
            * config["num_agents"]
            * config["num_train_envs"]
            * config["num_steps"]
        )
        loss, (value_loss, policy_loss, entropy) = stats["losses"]
        log_dict = {
            "num_updates": update_count,
            "num_env_steps": total_env_steps,
            "sps": segment_env_steps / elapsed,
            "agent/loss": loss,
            "agent/value_loss": value_loss,
            "agent/policy_loss": policy_loss,
            "agent/entropy": entropy,
            "ensemble/uncertainty_before": stats["uncertainty_before"],
            "ensemble/uncertainty_after": stats["uncertainty_after"],
            "ensemble/disagreement_reduction": stats[
                "disagreement_reduction"
            ],
            "ensemble/eligible_state_count": stats["eligible_state_count"],
            "ensemble/no_eligible_state_fraction": stats[
                "no_eligible_state_fraction"
            ],
            "levels/mean_num_blocks": stats["mean_num_blocks"],
        }

        for index, level_name in enumerate(config["eval_levels"]):
            log_dict[f"solve_rate/{level_name}/mean"] = stats[
                "eval_solve_rates_mean"
            ][index]
            log_dict[f"solve_rate/{level_name}/std"] = stats[
                "eval_solve_rates_std"
            ][index]
            log_dict[f"return/{level_name}/mean"] = stats[
                "eval_returns_mean"
            ][index]
            log_dict[f"return/{level_name}/std"] = stats[
                "eval_returns_std"
            ][index]
        log_dict["solve_rate/mean"] = stats["eval_solve_rates_mean"].mean()
        log_dict["return/mean"] = stats["eval_returns_mean"].mean()
        log_dict["return/member_attempt_std_mean"] = stats[
            "eval_returns_std"
        ].mean()
        log_dict["eval_ep_lengths/mean"] = stats[
            "eval_ep_lengths_mean"
        ].mean()
        log_dict["eval_ep_lengths/std"] = stats["eval_ep_lengths_std"].mean()
        log_dict.update(train_state_info["log"])

        log_dict["images/highest_scoring_level"] = wandb.Image(
            np.asarray(stats["highest_scoring_level"]),
            caption="Highest scoring level",
        )
        log_dict["images/highest_weighted_level"] = wandb.Image(
            np.asarray(stats["highest_weighted_level"]),
            caption="Highest weighted level",
        )
        for branch_name in ("dr", "replay", "mutation"):
            if train_state_info["info"][f"num_{branch_name}_updates"] > 0:
                log_dict[f"images/{branch_name}_levels"] = [
                    wandb.Image(np.asarray(image))
                    for image in stats[f"{branch_name}_levels"]
                ]

        frames, episode_lengths = stats["eval_animation"]
        for index, level_name in enumerate(config["eval_levels"]):
            length = int(episode_lengths[index])
            log_dict[f"animations/{level_name}"] = wandb.Video(
                np.asarray(frames[:length, index]),
                fps=4,
                format="gif",
            )
        wandb.log(log_dict)

    rng = jax.random.PRNGKey(config["seed"])
    init_rng, runner_rng = jax.random.split(rng)
    train_state = create_ensemble_train_state(init_rng)

    if config["mode"] == "eval":
        checkpoint_manager = ocp.CheckpointManager(
            os.path.join(config["checkpoint_directory"], "models"),
            item_handlers=ocp.StandardCheckpointHandler(),
        )
        checkpoint_step = (
            checkpoint_manager.latest_step()
            if config["checkpoint_to_eval"] == -1
            else config["checkpoint_to_eval"]
        )
        if checkpoint_step is None:
            raise ValueError("No checkpoint was found in checkpoint_directory")
        train_state = restore_ensemble_checkpoint(
            checkpoint_manager,
            checkpoint_step,
            train_state,
        )
        runner_rng, eval_rng = jax.random.split(runner_rng)
        states, returns, episode_lengths = jax.jit(_evaluate_ensemble)(
            eval_rng,
            train_state,
        )
        results_directory = config["checkpoint_directory"].replace(
            "checkpoints",
            "results",
            1,
        )
        os.makedirs(results_directory, exist_ok=True)
        np.savez_compressed(
            os.path.join(results_directory, "results.npz"),
            agent_positions=np.asarray(states.agent_pos),
            agent_directions=np.asarray(states.agent_dir),
            returns=np.asarray(returns),
            episode_lengths=np.asarray(episode_lengths),
            levels=np.asarray(config["eval_levels"]),
        )
        return states, returns, episode_lengths

    runner_state = (runner_rng, train_state)
    checkpoint_manager = None
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_ensemble_checkpointing(config)

    full_segments, remainder = divmod(config["num_updates"], config["eval_freq"])
    segment_lengths = [config["eval_freq"]] * full_segments
    if remainder:
        segment_lengths.append(remainder)

    compiled_segments: dict[int, Any] = {}
    for segment_length in segment_lengths:
        if segment_length not in compiled_segments:
            compiled_segments[segment_length] = _make_train_and_eval_step(
                segment_length
            )
        start_time = time.time()
        runner_state, metrics = compiled_segments[segment_length](runner_state)
        elapsed = time.time() - start_time
        _log_eval(
            metrics,
            ensemble_train_state_to_log_dict(runner_state[1], level_sampler),
            segment_length,
            elapsed,
        )
        if checkpoint_manager is not None:
            checkpoint_step = int(metrics["update_count"])
            checkpoint_manager.save(
                checkpoint_step,
                args=ocp.args.StandardSave(runner_state[1]),
            )
            checkpoint_manager.wait_until_finished()
    return runner_state[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for ensemble Maze PLR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default="maze_ensemble_plr")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    parser.add_argument("--checkpoint_save_interval", type=int, default=0)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument(
        "--eval_levels",
        nargs="+",
        default=[
            "SixteenRooms",
            "SixteenRooms2",
            "Labyrinth",
            "LabyrinthFlipped",
            "Labyrinth2",
            "StandardMaze",
            "StandardMaze2",
            "StandardMaze3",
        ],
    )

    training = parser.add_argument_group("Training params")
    training.add_argument("--num_agents", type=int, default=8)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--max_grad_norm", type=float, default=0.5)
    update_limit = training.add_mutually_exclusive_group()
    update_limit.add_argument("--num_updates", type=int, default=30000)
    update_limit.add_argument("--num_env_steps", type=int, default=None)
    training.add_argument("--num_steps", type=int, default=256)
    training.add_argument("--num_train_envs", type=int, default=32)
    training.add_argument("--num_minibatches", type=int, default=1)
    training.add_argument("--gamma", type=float, default=0.995)
    training.add_argument("--epoch_ppo", type=int, default=5)
    training.add_argument("--clip_eps", type=float, default=0.2)
    training.add_argument("--gae_lambda", type=float, default=0.98)
    training.add_argument("--entropy_coeff", type=float, default=1e-3)
    training.add_argument("--critic_coeff", type=float, default=0.5)

    training.add_argument(
        "--exploratory_grad_updates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    training.add_argument("--level_buffer_capacity", type=int, default=4000)
    training.add_argument("--replay_prob", type=float, default=0.8)
    training.add_argument("--staleness_coeff", type=float, default=0.3)
    training.add_argument("--temperature", type=float, default=0.3)
    training.add_argument("--topk_k", type=int, default=4)
    training.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    training.add_argument(
        "--prioritization",
        choices=("rank", "topk"),
        default="rank",
    )
    training.add_argument(
        "--buffer_duplicate_check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument(
        "--use_accel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    training.add_argument("--num_edits", type=int, default=5)
    training.add_argument("--agent_view_size", type=int, default=5)
    training.add_argument("--n_walls", type=int, default=25)
    return parser


if __name__ == "__main__":
    argument_parser = build_parser()
    cli_config = vars(argument_parser.parse_args())
    if cli_config["num_env_steps"] is not None:
        transitions_per_update = (
            cli_config["num_agents"]
            * cli_config["num_train_envs"]
            * cli_config["num_steps"]
        )
        cli_config["num_updates"] = (
            cli_config["num_env_steps"] // transitions_per_update
        )
    main(cli_config, project=cli_config["project"])
