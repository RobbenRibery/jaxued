import json
import time
import re
from typing import Callable, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import (
    EnvParams,
    EnvState,
    Observation,
    UnderspecifiedEnv,
)
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import (
    Level,
    make_level_generator,
    make_level_mutator_minimax,
)
from jaxued.level_sampler import LevelSampler
from jaxued.utils import (
    measure_s_in,
    compute_max_returns,
    max_mc,
    positive_value_loss,
    abs_policy_grad,
)
from jaxued.wrappers import AutoReplayWrapper
import chex
from enum import IntEnum
from typing import Optional, Dict, Any

try:
    from examples.policy_grad_utils import compute_raw_pg_grad_norms
    from examples.value_loss_utils import ppo_value_loss
except ModuleNotFoundError:
    from policy_grad_utils import compute_raw_pg_grad_norms
    from value_loss_utils import ppo_value_loss


DEFAULT_EVAL_FREQ = 200
DEFAULT_EXPLORATORY_GRAD_UPDATES = False
DEFAULT_USE_ACCEL = False


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


@struct.dataclass
class TransferEvaluationBatch:
    """Fixed initial conditions and random stream for paired target evaluation."""

    init_hstate: chex.ArrayTree = struct.field(pytree_node=True)
    init_obs: Observation = struct.field(pytree_node=True)
    init_env_state: EnvState = struct.field(pytree_node=True)
    rollout_rng: chex.PRNGKey = struct.field(pytree_node=True)
    source_count: int = struct.field(pytree_node=False)
    target_count: int = struct.field(pytree_node=False)


# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float):
        lambd (float):
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array],
    Tuple[
        Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict
    ],
]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Shape legend:
        T = max_episode_length
        N = num_envs

    Args:

        rng (chex.PRNGKey): Singleton
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.

    Loop tensor trace (`sample_step`, scanned T times):
        carry:
            hstate leaves: (N, ...)
            obs leaves: (N, ...)
            env_state leaves: (N, ...)
            last_done: (N,) bool
        model input `x`:
            obs leaves: (1, N, ...)
            done: (1, N)
        model outputs:
            action/log_prob/value before squeeze: (1, N)
            action/log_prob/value after squeeze: (N,)
        `env.step` vmapped over N envs:
            next_obs leaves: (N, ...)
            reward/done: (N,)
            info leaves: (N, ...)
        scan outputs `traj`:
            obs leaves: (T, N, ...)
            actions/rewards/dones/log_probs/values: (T, N)
            info leaves: (T, N, ...)
        post-scan bootstrap value:
            last_value: (N,)
    """

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = (
        jax.lax.scan(
            sample_step,
            (
                rng,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                jnp.zeros(num_envs, dtype=bool),
            ),
            None,
            length=max_episode_length,
        )
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (
        rng,
        train_state,
        hstate,
        last_obs,
        last_env_state,
        last_value.squeeze(0),
    ), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Shape legend:
        T = max_episode_length
        L = num_levels

    Args:
        rng (chex.PRNGKey):
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState):
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int):

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)

    Loop tensor trace (`step`, scanned T times):
        carry:
            hstate leaves: (L, ...)
            obs/state leaves: (L, ...)
            done/mask: (L,) bool
            episode_length: (L,) int32
        model input `x`:
            obs leaves: (1, L, ...)
            done: (1, L)
        sampled action after squeeze: (L,)
        `env.step` vmapped over L levels:
            next obs/state leaves: (L, ...)
            reward/done: (L,)
        scan outputs:
            states leaves: (T, L, ...)
            rewards: (T, L)
            episode_lengths: (L,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_levels), state, action, env_params
        )

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (
            state,
            reward,
        )

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def _level_match_mask(levels: Level, level: Level) -> chex.Array:
    """Return exact pytree equality for one level against a level batch."""
    equality_tree = jax.tree_util.tree_map(
        lambda batch_leaf, leaf: (batch_leaf == leaf)
        .reshape(batch_leaf.shape[0], -1)
        .all(axis=-1),
        levels,
        level,
    )
    equality_leaves = jax.tree_util.tree_leaves(equality_tree)
    return jnp.stack(equality_leaves, axis=0).all(axis=0)


def generate_transfer_target_bank(
    rng: chex.PRNGKey,
    source_level: Level,
    mutate_level: Callable[[chex.PRNGKey, Level, int], Level],
    target_count: int,
    num_edits: int,
) -> Level:
    """Generate fixed editor-chain targets for one source level.

    Each target is one independent Minimax editor chain. The mutator supplied by
    the transfer scorer is configured with ``allow_no_op=False`` and
    ``max_num_edits=num_edits``, so every chain applies exactly ``num_edits``
    sampled editor operations. Targets are intentionally not deduplicated.

    Args:
        rng: Key used to sample independent editor chains.
        source_level: Level whose local transfer neighborhood is generated.
        mutate_level: Configured Minimax mutation function.
        target_count: Number of editor chains in the target bank.
        num_edits: Number of editor applications in each chain.

    Returns:
        Batched target levels with leading shape ``(target_count, ...)``.
    """
    return jax.vmap(mutate_level, in_axes=(0, None, None))(
        jax.random.split(rng, target_count),
        source_level,
        num_edits,
    )


def resolve_transfer_target_banks(
    rng: chex.PRNGKey,
    sampler: core.FrozenDict[str, chex.ArrayTree],
    source_levels: Level,
    mutate_level: Callable[[chex.PRNGKey, Level, int], Level],
    target_count: int,
    num_edits: int,
) -> Tuple[chex.PRNGKey, Level]:
    """Resolve one fixed target bank for every new source candidate.

    Stored PLR duplicates reuse the stored target bank. Later duplicates within
    the current candidate batch reuse the first resolved bank. Only a source
    with neither kind of match runs the editor-chain generator.

    Args:
        rng: Target-generation key.
        sampler: Current PLR sampler containing target-bank level extras.
        source_levels: New candidate source batch with leading shape ``(N, ...)``.
        mutate_level: Transfer-specific Minimax mutation function.
        target_count: Number of targets attached to each source.
        num_edits: Editor applications in every chain.

    Returns:
        Updated key and target banks with leading shape ``(N, target_count, ...)``.
    """
    source_count = jax.tree_util.tree_leaves(source_levels)[0].shape[0]
    stored_target_banks = sampler["levels_extra"]["transfer_targets"]
    placeholder_bank = jax.tree_util.tree_map(lambda leaf: leaf[0], stored_target_banks)
    resolved_banks = jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(leaf[None, ...], source_count, axis=0),
        placeholder_bank,
    )
    source_indices = jnp.arange(source_count)
    stored_indices = jnp.arange(sampler["scores"].shape[0])

    def resolve_one(carry, source_index):
        rng_carry, banks = carry
        rng_carry, rng_generate = jax.random.split(rng_carry)
        source = jax.tree_util.tree_map(lambda leaf: leaf[source_index], source_levels)

        stored_matches = (
            _level_match_mask(sampler["levels"], source)
            & (stored_indices < sampler["size"])
            & sampler["levels_extra"]["has_transfer_targets"]
        )
        has_stored_match = stored_matches.any()
        stored_index = stored_matches.argmax()

        earlier_matches = _level_match_mask(source_levels, source) & (
            source_indices < source_index
        )
        has_earlier_match = earlier_matches.any()
        earlier_index = earlier_matches.argmax()

        def use_stored_bank(_):
            return jax.tree_util.tree_map(
                lambda leaf: leaf[stored_index], stored_target_banks
            )

        def use_earlier_or_generate(_):
            return jax.lax.cond(
                has_earlier_match,
                lambda __: jax.tree_util.tree_map(
                    lambda leaf: leaf[earlier_index], banks
                ),
                lambda __: generate_transfer_target_bank(
                    rng=rng_generate,
                    source_level=source,
                    mutate_level=mutate_level,
                    target_count=target_count,
                    num_edits=num_edits,
                ),
                operand=None,
            )

        target_bank = jax.lax.cond(
            has_stored_match,
            use_stored_bank,
            use_earlier_or_generate,
            operand=None,
        )
        banks = jax.tree_util.tree_map(
            lambda all_banks, bank: all_banks.at[source_index].set(bank),
            banks,
            target_bank,
        )
        return (rng_carry, banks), None

    (rng, resolved_banks), _ = jax.lax.scan(
        resolve_one,
        (rng, resolved_banks),
        source_indices,
    )
    return rng, resolved_banks


def evaluate_returns_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> chex.Array:
    """Evaluate one masked episode return per level without storing trajectories."""
    num_levels = jax.tree_util.tree_leaves(init_obs)[0].shape[0]

    def step(carry, _):
        rng_carry, hstate, obs, state, done, active, returns = carry
        rng_carry, rng_action, rng_step = jax.random.split(rng_carry, 3)

        network_input = jax.tree_util.tree_map(
            lambda leaf: leaf[None, ...], (obs, done)
        )
        hstate, policy, _ = train_state.apply_fn(
            train_state.params, network_input, hstate
        )
        action = policy.sample(seed=rng_action).squeeze(0)
        obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_levels),
            state,
            action,
            env_params,
        )
        returns = returns + reward * active
        active = active & ~done
        return (rng_carry, hstate, obs, state, done, active, returns), None

    initial_carry = (
        rng,
        init_hstate,
        init_obs,
        init_env_state,
        jnp.zeros(num_levels, dtype=jnp.bool_),
        jnp.ones(num_levels, dtype=jnp.bool_),
        jnp.zeros(num_levels, dtype=jnp.float32),
    )
    (_, _, _, _, _, _, returns), _ = jax.lax.scan(
        step,
        initial_carry,
        None,
        length=max_episode_length,
    )
    return returns


def prepare_editor_transfer_evaluation(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    target_banks: Level,
) -> Tuple[chex.PRNGKey, TransferEvaluationBatch]:
    """Create shared reset states and randomness for paired target evaluation."""
    bank_leaf = jax.tree_util.tree_leaves(target_banks)[0]
    source_count, target_count = bank_leaf.shape[:2]
    flat_target_count = source_count * target_count
    flat_targets = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape(flat_target_count, *leaf.shape[2:]), target_banks
    )

    rng, rng_reset, rng_evaluate = jax.random.split(rng, 3)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, flat_target_count),
        flat_targets,
        env_params,
    )
    return rng, TransferEvaluationBatch(
        init_hstate=ActorCritic.initialize_carry((flat_target_count,)),
        init_obs=init_obs,
        init_env_state=init_env_state,
        rollout_rng=rng_evaluate,
        source_count=source_count,
        target_count=target_count,
    )


def evaluate_editor_transfer_returns(
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    policy: TrainState,
    evaluation_batch: TransferEvaluationBatch,
    max_episode_length: int,
) -> chex.Array:
    """Evaluate paired target returns for one policy state."""
    flat_returns = evaluate_returns_rnn(
        rng=evaluation_batch.rollout_rng,
        env=env,
        env_params=env_params,
        train_state=policy,
        init_hstate=evaluation_batch.init_hstate,
        init_obs=evaluation_batch.init_obs,
        init_env_state=evaluation_batch.init_env_state,
        max_episode_length=max_episode_length,
    )
    return flat_returns.reshape(
        evaluation_batch.source_count, evaluation_batch.target_count
    )


def compute_editor_transfer_scores(
    returns_before: chex.Array,
    returns_after: chex.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Compute per-source mean gains and aggregate diagnostics."""
    target_count = returns_before.shape[1]

    gains = returns_after - returns_before
    scores = gains.mean(axis=1)
    std_ddof = 1 if target_count > 1 else 0
    standard_errors = gains.std(axis=1, ddof=std_ddof) / jnp.sqrt(target_count)
    diagnostics = {
        "transfer_pre_return_mean": returns_before.mean(),
        "transfer_post_return_mean": returns_after.mean(),
        "transfer_gain_mean": gains.mean(),
        "transfer_gain_std": gains.std(),
        "transfer_standard_error_mean": standard_errors.mean(),
        "transfer_positive_gain_fraction": (gains > 0).mean(),
    }
    return scores, diagnostics


def aggregate_replay_transfer_updates(
    level_indices: chex.Array,
    scores: chex.Array,
    max_returns: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Give repeated replay indices identical mean-score/max-return updates."""
    same_index = level_indices[:, None] == level_indices[None, :]
    counts = same_index.sum(axis=1)
    aggregate_scores = (same_index * scores[None, :]).sum(axis=1) / counts
    aggregate_max_returns = jnp.where(same_index, max_returns[None, :], -jnp.inf).max(
        axis=1
    )
    return aggregate_scores, aggregate_max_returns


def select_editor_transfer_states(
    original_state: TrainState,
    updated_state: TrainState,
    persist_update: bool,
) -> Tuple[TrainState, TrainState]:
    """Select continuing and scoring states after one full-batch PPO update."""
    if persist_update:
        return updated_state, updated_state
    return original_state, updated_state


def empty_editor_transfer_metrics() -> Dict[str, chex.Array]:
    """Return a stable metric pytree for branches without transfer scoring."""
    return {
        "transfer_pre_return_mean": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_post_return_mean": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_gain_mean": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_gain_std": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_standard_error_mean": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_positive_gain_fraction": jnp.array(jnp.nan, dtype=jnp.float32),
        "transfer_attached_bank_count": jnp.array(0, dtype=jnp.int32),
        "transfer_eval_env_steps": jnp.array(0, dtype=jnp.int32),
        "transfer_virtual_optimizer_steps": jnp.array(0, dtype=jnp.int32),
        "transfer_update_is_virtual": jnp.array(0.0, dtype=jnp.float32),
    }


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
    compute_per_step_grads: bool = False,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState], Tuple[chex.ArrayTree, Optional[chex.Array]]
]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Shape legend:
        T = n_steps
        N = num_envs
        M = n_minibatch
        B = N // M (minibatch env count)

    Args:
        rng (chex.PRNGKey):
        train_state (TrainState):
        init_hstate (chex.ArrayTree):
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int):
        n_steps (int):
        n_minibatch (int):
        n_epochs (int):
        clip_eps (float):
        entropy_coeff (float):
        critic_coeff (float):
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.
        compute_per_step_grads (bool, optional): If True, compute and return per-step gradient norms.

    Returns:
        Tuple of ((rng, train_state), (losses, grad_norms)).
        grad_norms is None if compute_per_step_grads is False.

    Loop tensor trace:
        input batch tensors:
            obs leaves: (T, N, ...)
            actions/dones/log_probs/values/targets/advantages: (T, N)
            last_dones: (T, N)
        optional `per_step_grad_norms`: (T,)
        epoch-level scan (`update_epoch`, scanned `n_epochs` times):
            permutation: (N,)
            reshaped minibatches:
                init_hstate leaves: (M, B, ...)
                obs leaves: (M, T, B, ...)
                actions/last_dones/log_probs/values/targets/advantages: (M, T, B)
        minibatch scan (`update_minibatch`, scanned M times):
            minibatch tensors seen by loss:
                init_hstate leaves: (B, ...)
                obs leaves: (T, B, ...)
                actions/last_dones/log_probs/values/targets/advantages: (T, B)
                values_pred/log_probs_pred/ratio: (T, B)
            scalar outputs per minibatch:
                total_loss, l_vf, l_clip, entropy
        returned `losses` leaves: (n_epochs, M)
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    # Compute per-step gradient norms before any updates (using original params)
    per_step_grad_norms = None
    if compute_per_step_grads:
        per_step_grad_norms = compute_raw_pg_grad_norms(
            apply_fn=train_state.apply_fn,
            params=train_state.params,
            obs=obs,
            last_dones=last_dones,
            actions=actions,
            advantages=advantages,
            init_hstate=init_hstate,
            pg_n_minibatch=config["pg_n_minibatch"],
        )

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            (
                init_hstate,
                obs,
                actions,
                last_dones,
                log_probs,
                values,
                targets,
                advantages,
            ) = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(
                    params, (obs, last_dones), init_hstate
                )
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (
                    -jnp.minimum(
                        ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
                    )
                ).mean()

                values_pred_clipped = values + (values_pred - values).clip(
                    -clip_eps, clip_eps
                )
                l_vf = (
                    0.5
                    * jnp.maximum(
                        (values_pred - targets) ** 2,
                        (values_pred_clipped - targets) ** 2,
                    ).mean()
                )

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(
                    n_minibatch, -1, *x.shape[1:]
                ),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    (rng, train_state), losses = jax.lax.scan(
        update_epoch, (rng, train_state), None, n_epochs
    )
    return (rng, train_state), (losses, per_step_grad_norms)


class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM"""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(
            obs.image
        )
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="scalar_embed",
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


# endregion


# region checkpointing
def setup_checkpointing(
    config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams
) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict):
        train_state (TrainState):
        env (UnderspecifiedEnv):
        env_params (EnvParams):

    Returns:
        ocp.CheckpointManager:
    """
    overall_save_dir = os.path.join(
        os.getcwd(), "checkpoints", f"{config['run_name']}", str(config["seed"])
    )
    os.makedirs(overall_save_dir, exist_ok=True)

    # save the config
    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config.as_dict(), indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        ),
    )
    return checkpoint_manager


# endregion


def train_state_to_log_dict(
    train_state: TrainState, level_sampler: LevelSampler
) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.

    Args:
        train_state (TrainState):
        level_sampler (LevelSampler):

    Returns:
        dict:
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    log = {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": sampler["scores"].max(),
        "level_sampler/weighted_score": (
            sampler["scores"] * level_sampler.level_weights(sampler)
        ).sum(),
        "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
    }
    if "levels_extra" in sampler and "has_transfer_targets" in sampler["levels_extra"]:
        log["level_sampler/transfer_target_bank_count"] = (
            sampler["levels_extra"]["has_transfer_targets"] & idx
        ).sum()

    return {
        "log": log,
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def _agent_log_metrics(losses: chex.ArrayTree) -> dict[str, chex.Array]:
    """Average PPO agent metrics across one W&B logging interval."""
    loss, (value_loss, policy_loss, entropy) = losses
    return {
        "agent/loss": loss.mean(),
        "agent/value_loss": value_loss.mean(),
        "agent/policy_loss": policy_loss.mean(),
        "agent/entropy": entropy.mean(),
    }


def compute_score(
    config: Dict[str, Any],
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    advantages: chex.Array,
    targets: Optional[chex.Array] = None,
    grad_norms: Optional[chex.Array] = None,
) -> chex.Array:
    """Compute level score based on configured score function.

    Args:
        config: Configuration dict with 'score_function' key.
        dones: Episode done flags. Shape: (num_steps, num_envs).
        values: Value estimates. Shape: (num_steps, num_envs).
        max_returns: Max return per env. Shape: (num_envs,).
        advantages: Advantage estimates. Shape: (num_steps, num_envs).
        targets: GAE targets, i.e. advantages + values (for ppo_value_loss).
            Shape: (num_steps, num_envs).
        grad_norms: Per-step, per-env gradient norms (for abs_pg).
            Shape: (num_steps, num_envs).

    Returns:
        Score per environment. Shape: (num_envs,).
    """
    score_fn = config["score_function"]

    if score_fn == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif score_fn == "pvl":
        return positive_value_loss(dones, advantages)
    elif score_fn == "abs_pg":
        assert grad_norms is not None, "abs_pg requires grad_norms"
        return abs_policy_grad(dones, grad_norms)
    elif score_fn == "ppo_value_loss":
        assert targets is not None, "ppo_value_loss requires targets"
        return ppo_value_loss(values, targets)
    else:
        raise ValueError(f"Unknown score function: {score_fn}")


def build_ppo_eval_batch_from_rollout(
    obs: chex.ArrayTree,
    actions: chex.Array,
    dones: chex.Array,
    log_probs: chex.Array,
    rewards: chex.Array,
    values: chex.Array,
    last_value: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[
    chex.ArrayTree,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.ArrayTree,
]:
    """Build PPO-loss evaluation batch from a rollout.

    Args:
        obs: Rollout observations. Shape leaves: (T, *B, ...).
        actions: Rollout actions. Shape: (T, *B).
        dones: Rollout done flags. Shape: (T, *B).
        log_probs: Behavior log probs. Shape: (T, *B).
        rewards: Rollout rewards. Shape: (T, *B).
        values: Rollout value predictions. Shape: (T, *B).
        last_value: Bootstrap value. Shape: (*B).
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple: (eval_obs, eval_actions, eval_last_dones, eval_log_probs, eval_values,
                eval_targets, eval_advantages, eval_init_hstate)
            eval_obs leaves: (T, *B, ...)
            eval_actions: (T, *B)
            eval_last_dones: (T, *B)
            eval_log_probs: (T, *B)
            eval_values: (T, *B)
            eval_targets: (T, *B)
            eval_advantages: (T, *B)
            eval_init_hstate leaves: (*B, ...)
    """
    advantages, targets = compute_gae(
        gamma=gamma,
        lambd=gae_lambda,
        last_value=last_value,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch_shape = values.shape[1:]
    init_hstate = ActorCritic.initialize_carry(batch_shape)
    return (
        obs,
        actions,
        last_dones,
        log_probs,
        values,
        targets,
        advantages,
        init_hstate,
    )


def ppo_loss_fn_for_s_in(
    virtual_train_state: TrainState,
    eval_batch: tuple[
        chex.ArrayTree,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.ArrayTree,
    ],
    config: Dict[str, Any],
) -> chex.Array:
    """Compute per-level PPO loss used by S_in.

    Args:
        virtual_train_state: Virtual train state containing params and optimizer state.
        eval_batch: (obs, actions, last_dones, log_probs, values, targets, advantages,
            init_hstate). After selecting one training environment slot, tensor
            leaves have shape (T, G, ...) where G is the number of independent
            rollout samples for that slot.
        config: Training config containing clip/entropy/critic coefficients.

    Returns:
        Per-level PPO total loss after reducing over T and G. Shape: ().
    """
    (
        obs,
        actions,
        last_dones,
        log_probs,
        values,
        targets,
        advantages,
        init_hstate,
    ) = eval_batch

    _, pi, values_pred = virtual_train_state.apply_fn(
        virtual_train_state.params, (obs, last_dones), init_hstate
    )
    log_probs_pred = pi.log_prob(actions)

    ratio = jnp.exp(log_probs_pred - log_probs)
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    adv_norm = (advantages - adv_mean) / (adv_std + 1e-5)
    l_clip = -jnp.minimum(
        ratio * adv_norm,
        jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * adv_norm,
    ).mean()

    values_pred_clipped = values + (values_pred - values).clip(
        -config["clip_eps"], config["clip_eps"]
    )
    l_vf = (
        0.5
        * jnp.maximum(
            (values_pred - targets) ** 2,
            (values_pred_clipped - targets) ** 2,
        ).mean()
    )
    entropy = pi.entropy().mean()

    return l_clip + config["critic_coeff"] * l_vf - config["entropy_coeff"] * entropy


def virtual_update_fn_for_s_in(
    rng: chex.PRNGKey,
    virtual_train_state: TrainState,
    update_batch: tuple,
    config: Dict[str, Any],
) -> tuple[chex.PRNGKey, TrainState]:
    """One virtual PPO update step on a single-level batch.

    Args:
        rng: PRNG key.
        virtual_train_state: Current virtual train state (params + optimizer state).
        update_batch: (obs, actions, dones, log_probs, values, targets,
            advantages, init_hstate), where non-hidden arrays have shapes
            (T, G, ...) and init_hstate leaves have shape (G, ...).
        config: Training config.

    Returns:
        (rng, updated_virtual_train_state) after one virtual update.
    """
    (
        obs,
        actions,
        dones,
        log_probs,
        values,
        targets,
        advantages,
        init_hstate,
    ) = update_batch

    (rng, virtual_train_state), _ = update_actor_critic_rnn(
        rng=rng,
        train_state=virtual_train_state,
        init_hstate=init_hstate,
        batch=(obs, actions, dones, log_probs, values, targets, advantages),
        num_envs=actions.shape[1],
        n_steps=actions.shape[0],
        n_minibatch=1,
        n_epochs=config["epoch_ppo"],
        clip_eps=config["clip_eps"],
        entropy_coeff=config["entropy_coeff"],
        critic_coeff=config["critic_coeff"],
        update_grad=True,
        compute_per_step_grads=False,
    )
    return rng, virtual_train_state


def collect_s_in_rollout_set(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: Level,
    config: Dict[str, Any],
) -> tuple[
    chex.PRNGKey,
    tuple[
        chex.ArrayTree,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.ArrayTree,
    ],
]:
    """Collect one S_in rollout set with G trajectories per sampled level.

    Shape legend:
        T = config["num_steps"]
        N = config["num_train_envs"]
        G = config["sin_num_rollouts_per_level"]

    Args:
        rng: PRNG key.
        env: Environment.
        env_params: Environment params.
        train_state: Current policy state.
        levels: Sampled training environment slots. Leaves have shape (N, ...).
        config: Training config.

    Returns:
        (rng, rollout_set) where rollout_set is
            (obs, actions, rewards, dones, log_probs, values, last_value, init_hstate)
        and non-hidden rollout tensors have shape (T, N, G, ...), last_value has
        shape (N, G), and init_hstate leaves have shape (N, G, ...).
    """
    num_levels = config["num_train_envs"]
    num_rollouts_per_level = config["sin_num_rollouts_per_level"]
    num_rollout_envs = num_levels * num_rollouts_per_level

    flat_levels = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x, num_rollouts_per_level, axis=0), levels
    )
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_rollout_envs),
        flat_levels,
        env_params,
    )
    init_hstate_flat = ActorCritic.initialize_carry((num_rollout_envs,))
    (
        (rng, _train_state, _, _, _, last_value_flat),
        (
            obs_flat,
            actions_flat,
            rewards_flat,
            dones_flat,
            log_probs_flat,
            values_flat,
            _,
        ),
    ) = sample_trajectories_rnn(
        rng=rng,
        env=env,
        env_params=env_params,
        train_state=train_state,
        init_hstate=init_hstate_flat,
        init_obs=init_obs,
        init_env_state=init_env_state,
        num_envs=num_rollout_envs,
        max_episode_length=config["num_steps"],
    )

    def _reshape_time_batch(x: chex.Array) -> chex.Array:
        return x.reshape(
            x.shape[0],
            num_levels,
            num_rollouts_per_level,
            *x.shape[2:],
        )

    def _reshape_batch(x: chex.Array) -> chex.Array:
        return x.reshape(num_levels, num_rollouts_per_level, *x.shape[1:])

    return rng, (
        jax.tree_util.tree_map(_reshape_time_batch, obs_flat),
        _reshape_time_batch(actions_flat),
        _reshape_time_batch(rewards_flat),
        _reshape_time_batch(dones_flat),
        _reshape_time_batch(log_probs_flat),
        _reshape_time_batch(values_flat),
        _reshape_batch(last_value_flat),
        jax.tree_util.tree_map(_reshape_batch, init_hstate_flat),
    )


def compute_s_in_scores(
    rng: chex.PRNGKey,
    train_state: TrainState,
    rollout_a: tuple[
        chex.ArrayTree,  # obs
        chex.Array,  # actions
        chex.Array,  # rewards
        chex.Array,  # dones
        chex.Array,  # log_probs
        chex.Array,  # values
        chex.Array,  # last_value
        chex.ArrayTree,  # init_hstate
    ],
    rollout_b: tuple[
        chex.ArrayTree,  # obs
        chex.Array,  # actions
        chex.Array,  # rewards
        chex.Array,  # dones
        chex.Array,  # log_probs
        chex.Array,  # values
        chex.Array,  # last_value
        chex.ArrayTree,  # init_hstate
    ],
    config: Dict[str, Any],
) -> tuple[chex.PRNGKey, chex.Array, Dict[str, chex.Array]]:
    """Compute per-level S_in scores using holdout A/B rollouts.

    Args:
        rng: PRNG key.
        train_state: Current train state (base params used for scoring).
        rollout_a: Update rollout set used for virtual updates. Structure:
            (obs_a, actions_a, rewards_a, dones_a, log_probs_a, values_a, last_value_a, init_hstate_a)
            with shapes:
                obs_a leaves: (T, N, G, ...)
                actions_a/rewards_a/dones_a/log_probs_a/values_a: (T, N, G)
                last_value_a: (N, G)
                init_hstate_a leaves: (N, G, ...)
        rollout_b: Holdout rollout set used for evaluation only (same structure/shapes
            as rollout_a), collected from independent trajectories on the same levels:
            (obs_b, actions_b, rewards_b, dones_b, log_probs_b, values_b, last_value_b, init_hstate_b)
        config: Config containing S_in hyperparameters.
            sin_score_batch_size controls how many sampled level slots are scored
            together; the output score is still one scalar per slot.

    Returns:
        (rng, scores, metrics):
            scores: S_in per level. Shape: (N,)
            metrics keys:
                "lp_s_in_mean": ()
                "lp_loss_before_mean": ()
                "lp_loss_after_mean": ()
    """
    (
        obs_a,
        actions_a,
        rewards_a,
        dones_a,
        log_probs_a,
        values_a,
        last_value_a,
        init_hstate_a,
    ) = rollout_a
    (
        obs_b,
        actions_b,
        rewards_b,
        dones_b,
        log_probs_b,
        values_b,
        last_value_b,
        _init_hstate_b,
    ) = rollout_b

    advantages_a, targets_a = compute_gae(
        gamma=config["gamma"],
        lambd=config["gae_lambda"],
        last_value=last_value_a,
        values=values_a,
        rewards=rewards_a,
        dones=dones_a,
    )
    eval_batch_b = build_ppo_eval_batch_from_rollout(
        obs=obs_b,
        actions=actions_b,
        dones=dones_b,
        log_probs=log_probs_b,
        rewards=rewards_b,
        values=values_b,
        last_value=last_value_b,
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
    )

    num_envs = actions_a.shape[1]

    def _slice_level_time(x: chex.Array, idx: chex.Array) -> chex.Array:
        """Slice a single training environment slot from a (T, N, G, ...) array.

        Args:
            x: Array with shape (T, N, G, ...) where N is the number of sampled
                training environment slots.
            idx: Scalar integer index of the environment to extract (may be a traced value).

        Returns:
            Array of shape (T, G, ...) for the selected environment slot.
        """
        return jax.lax.dynamic_index_in_dim(x, idx, axis=1, keepdims=False)

    def _slice_level_batch(x: chex.Array, idx: chex.Array) -> chex.Array:
        """Slice a single training environment slot from an (N, G, ...) array.

        Used for per-environment state tensors (e.g. init_hstate) whose sampled
        level axis is 0 rather than 1.

        Args:
            x: Array with shape (N, G, ...) where N is the number of sampled
                training environment slots.
            idx: Scalar integer index of the environment to extract (may be a traced value).

        Returns:
            Array of shape (G, ...) for the selected environment slot.
        """
        return jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False)

    def _per_level_score(
        rng_level: chex.PRNGKey, level_idx: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Compute the S_in score for a single level.

        Builds the PPO update batch and PPO eval batch for level ``level_idx``,
        then calls ``measure_s_in`` to estimate how much a virtual update on the update
        batch improves PPO loss on the held-out eval batch.

        Args:
            rng_level: PRNG key dedicated to this level.
            level_idx: Scalar integer index identifying which environment (level) to score.

        Returns:
            (s_in_i, loss_before_i, loss_after_i):
                s_in_i: Scalar S_in score for this level.
                loss_before_i: PPO loss on the eval batch before the virtual update.
                loss_after_i: PPO loss on the eval batch after the virtual update.
        """
        update_batch_i = (
            jax.tree_util.tree_map(lambda x: _slice_level_time(x, level_idx), obs_a),
            _slice_level_time(actions_a, level_idx),
            _slice_level_time(dones_a, level_idx),
            _slice_level_time(log_probs_a, level_idx),
            _slice_level_time(values_a, level_idx),
            _slice_level_time(targets_a, level_idx),
            _slice_level_time(advantages_a, level_idx),
            jax.tree_util.tree_map(
                lambda x: _slice_level_batch(x, level_idx), init_hstate_a
            ),
        )
        eval_batch_i = (
            jax.tree_util.tree_map(
                lambda x: _slice_level_time(x, level_idx), eval_batch_b[0]
            ),  # eval obs: (T, G, ...)
            _slice_level_time(eval_batch_b[1], level_idx),  # eval actions: (T, G)
            _slice_level_time(eval_batch_b[2], level_idx),  # eval last_dones: (T, G)
            _slice_level_time(eval_batch_b[3], level_idx),  # eval log_probs: (T, G)
            _slice_level_time(eval_batch_b[4], level_idx),  # eval values: (T, G)
            _slice_level_time(eval_batch_b[5], level_idx),  # eval targets: (T, G)
            _slice_level_time(eval_batch_b[6], level_idx),  # eval advantages: (T, G)
            jax.tree_util.tree_map(
                lambda x: _slice_level_batch(x, level_idx), eval_batch_b[7]
            ),  # eval init_hstate: (G, ...)
        )

        rng_level, s_in_i, diagnostics_i = measure_s_in(
            rng=rng_level,
            virtual_state=train_state,
            update_batch=update_batch_i,
            eval_batch=eval_batch_i,
            loss_fn=lambda state, batch: ppo_loss_fn_for_s_in(
                state, batch, config=config
            ),
            virtual_update_fn=lambda key, state, batch: virtual_update_fn_for_s_in(
                key, state, batch, config=config
            ),
            n_virtual_updates=config["sin_n_virtual_updates"],
            eps=config["sin_eps"],
        )
        return (
            jnp.ravel(s_in_i)[0],
            jnp.ravel(diagnostics_i["loss_before"])[0],
            jnp.ravel(diagnostics_i["loss_after"])[0],
        )

    def _next_level_key(
        rng_carry: chex.PRNGKey, _: None
    ) -> tuple[chex.PRNGKey, chex.PRNGKey]:
        rng_carry, rng_level = jax.random.split(rng_carry)
        return rng_carry, rng_level

    rng, rng_levels = jax.lax.scan(_next_level_key, rng, None, length=num_envs)
    level_indices = jnp.arange(num_envs, dtype=jnp.int32)
    score_batch_size = min(config["sin_score_batch_size"], num_envs)

    if score_batch_size == 1:
        # Fully sequential scoring is the safest TPU/XLA path because it avoids
        # batching virtual TrainState updates across sampled levels.
        def _score_one(
            _: None, batch: tuple[chex.PRNGKey, chex.Array]
        ) -> tuple[None, tuple[chex.Array, chex.Array, chex.Array]]:
            rng_level, level_idx = batch
            return None, _per_level_score(rng_level, level_idx)

        _, (scores, loss_before, loss_after) = jax.lax.scan(
            _score_one, None, (rng_levels, level_indices)
        )
    else:
        # Bound cross-level vectorization by chunking levels. This gives a
        # tunable throughput/compile-size tradeoff without vmapping across all N.
        num_score_batches = (num_envs + score_batch_size - 1) // score_batch_size
        num_padded_levels = num_score_batches * score_batch_size
        pad_count = num_padded_levels - num_envs
        rng_pad = jnp.repeat(rng_levels[:1], pad_count, axis=0)
        idx_pad = jnp.zeros((pad_count,), dtype=level_indices.dtype)
        rng_level_batches = jnp.concatenate([rng_levels, rng_pad], axis=0).reshape(
            num_score_batches, score_batch_size, *rng_levels.shape[1:]
        )
        level_idx_batches = jnp.concatenate([level_indices, idx_pad], axis=0).reshape(
            num_score_batches, score_batch_size
        )

        def _score_batch(
            _: None, batch: tuple[chex.PRNGKey, chex.Array]
        ) -> tuple[None, tuple[chex.Array, chex.Array, chex.Array]]:
            rng_batch, level_idx_batch = batch
            return None, jax.vmap(_per_level_score, in_axes=(0, 0))(
                rng_batch, level_idx_batch
            )

        _, (scores, loss_before, loss_after) = jax.lax.scan(
            _score_batch, None, (rng_level_batches, level_idx_batches)
        )
        scores = scores.reshape(num_padded_levels)[:num_envs]
        loss_before = loss_before.reshape(num_padded_levels)[:num_envs]
        loss_after = loss_after.reshape(num_padded_levels)[:num_envs]
    return (
        rng,
        scores,
        {
            "lp_s_in_mean": scores.mean(),
            "lp_loss_before_mean": loss_before.mean(),
            "lp_loss_after_mean": loss_after.mean(),
        },
    )


def normalize_run_name(name: str) -> str:
    """Normalize an experiment/run name so it is safe as a checkpoint directory."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("._-")
    return normalized or "run"


def validate_editor_transfer_config(config: Dict[str, Any]) -> None:
    """Validate the static shape and ownership constraints of editor transfer."""
    if config["score_function"] != "editor_transfer":
        return
    if config["transfer_target_count"] < 1:
        raise ValueError("--transfer_target_count must be >= 1.")
    if config["transfer_num_edits"] < 1:
        raise ValueError("--transfer_num_edits must be >= 1.")
    if config["use_accel"]:
        raise ValueError(
            "--score_function editor_transfer does not support --use_accel."
        )


def main(config=None, project="JAXUED_TEST"):
    validate_editor_transfer_config(config)
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    wandb.init(
        config=config,
        project=project,
        name=config["wandb_experiment_name"],
        group=config["run_name"],
        tags=tags,
    )
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")
    wandb.define_metric("transfer/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")

        # generic stats
        source_env_steps = (
            stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        )
        transfer_env_steps = 0
        if config["score_function"] == "editor_transfer":
            transfer_env_steps = (
                stats["update_count"]
                * 2
                * config["num_train_envs"]
                * config["transfer_target_count"]
                * env_params.max_steps_in_episode
            )
        env_steps = source_env_steps + transfer_env_steps
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
            **_agent_log_metrics(stats["losses"]),
        }

        # evaluation performance
        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update(
            {
                f"solve_rate/{name}": solve_rate
                for name, solve_rate in zip(config["eval_levels"], solve_rates)
            }
        )
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update(
            {f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)}
        )
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats["eval_ep_lengths"].mean()})

        # level sampler
        log_dict.update(train_state_info["log"])

        # optional S_in diagnostics
        if "lp_s_in_mean" in stats:
            lp_s_in_mean = stats["lp_s_in_mean"].mean()
            lp_loss_before_mean = stats["lp_loss_before_mean"].mean()
            lp_loss_after_mean = stats["lp_loss_after_mean"].mean()
            if jnp.isfinite(lp_s_in_mean):
                log_dict.update(
                    {
                        "lp/s_in_mean": lp_s_in_mean,
                        "lp/loss_before_mean": lp_loss_before_mean,
                        "lp/loss_after_mean": lp_loss_after_mean,
                    }
                )

        if config["score_function"] == "editor_transfer":
            transfer_keys = {
                "transfer/pre_return_mean": "transfer_pre_return_mean",
                "transfer/post_return_mean": "transfer_post_return_mean",
                "transfer/gain_mean": "transfer_gain_mean",
                "transfer/gain_std": "transfer_gain_std",
                "transfer/standard_error_mean": "transfer_standard_error_mean",
                "transfer/positive_gain_fraction": ("transfer_positive_gain_fraction"),
                "transfer/attached_bank_count_per_update": (
                    "transfer_attached_bank_count"
                ),
                "transfer/update_is_virtual_fraction": ("transfer_update_is_virtual"),
            }
            log_dict.update(
                {
                    log_name: stats[metric_name].mean()
                    for log_name, metric_name in transfer_keys.items()
                }
            )
            log_dict.update(
                {
                    "transfer/target_count": config["transfer_target_count"],
                    "transfer/chain_length": config["transfer_num_edits"],
                    "transfer/eval_env_steps_interval": stats[
                        "transfer_eval_env_steps"
                    ].sum(),
                    "transfer/eval_env_steps_total": transfer_env_steps,
                    "transfer/virtual_optimizer_steps_interval": stats[
                        "transfer_virtual_optimizer_steps"
                    ].sum(),
                    "transfer/virtual_optimizer_steps_total": (
                        train_state_info["info"]["num_dr_updates"]
                        * config["num_minibatches"]
                        * config["epoch_ppo"]
                        * (not config["exploratory_grad_updates"])
                    ),
                }
            )

        # images
        log_dict.update(
            {
                "images/highest_scoring_level": wandb.Image(
                    np.array(stats["highest_scoring_level"]),
                    caption="Highest scoring level",
                )
            }
        )
        log_dict.update(
            {
                "images/highest_weighted_level": wandb.Image(
                    np.array(stats["highest_weighted_level"]),
                    caption="Highest weighted level",
                )
            }
        )

        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict.update(
                    {
                        f"images/{s}_levels": [
                            wandb.Image(np.array(image))
                            for image in stats[f"{s}_levels"]
                        ]
                    }
                )

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = (
                stats["eval_animation"][0][:, i],
                stats["eval_animation"][1][i],
            )
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})

        wandb.log(log_dict)

    # Setup the environment
    env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    eval_env = env
    sample_random_level = make_level_generator(
        env.max_height, env.max_width, config["n_walls"]
    )
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)
    mutate_transfer_target = make_level_mutator_minimax(
        config["transfer_num_edits"], allow_no_op=False
    )

    # And the level sampler
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

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                256,
                axis=0,
            ),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(
            rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],))
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=config["lr"], eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        level_extras = {"max_return": -jnp.inf}
        if config["score_function"] == "editor_transfer":
            placeholder_target_bank = jax.tree_util.tree_map(
                lambda leaf: jnp.repeat(
                    jnp.asarray(leaf)[None, ...],
                    config["transfer_target_count"],
                    axis=0,
                ),
                pholder_level,
            )
            level_extras.update(
                {
                    "transfer_targets": placeholder_target_bank,
                    "has_transfer_targets": jnp.array(False),
                }
            )
        sampler = level_sampler.initialize(pholder_level, level_extras)
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
        This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.

        Shape legend:
            T = config["num_steps"]
            N = config["num_train_envs"]
            E = config["epoch_ppo"]
            M = config["num_minibatches"]

        Tensor trace shared by all branches:
            rollout outputs:
                obs leaves: (T, N, ...)
                actions/rewards/dones/log_probs/values: (T, N)
            derived tensors:
                advantages/targets: (T, N)
                max_returns/scores: (N,)
                grad_norms (abs_pg variants): (T,)
            PPO loss tree from `update_actor_critic_rnn`:
                leaves shaped (E, M)
                branch metric stores mean over (E, M) -> scalar leaves
        """

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
            The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.

            Tensor trace:
                new_levels leaves: (N, ...)
                init_obs/init_env_state leaves: (N, ...)
                rollout obs leaves: (T, N, ...)
                rollout actions/rewards/dones/log_probs/values: (T, N)
                advantages/targets: (T, N)
                max_returns/scores: (N,)
                losses leaves: (E, M)
            """
            sampler = train_state.sampler

            transfer_metrics = empty_editor_transfer_metrics()

            # Generate source levels, then resolve their fixed target banks before
            # either target evaluation or the source rollout.
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(
                jax.random.split(rng_levels, config["num_train_envs"])
            )
            if config["score_function"] == "editor_transfer":
                rng, target_banks = resolve_transfer_target_banks(
                    rng=rng,
                    sampler=sampler,
                    source_levels=new_levels,
                    mutate_level=mutate_transfer_target,
                    target_count=config["transfer_target_count"],
                    num_edits=config["transfer_num_edits"],
                )
                rng, transfer_evaluation = prepare_editor_transfer_evaluation(
                    rng=rng,
                    env=eval_env,
                    env_params=env_params,
                    target_banks=target_banks,
                )
                returns_before = evaluate_editor_transfer_returns(
                    env=eval_env,
                    env_params=env_params,
                    policy=train_state,
                    evaluation_batch=transfer_evaluation,
                    max_episode_length=env_params.max_steps_in_episode,
                )

            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(
                jax.random.split(rng_reset, config["num_train_envs"]),
                new_levels,
                env_params,
            )
            # Rollout
            init_hstate_a = ActorCritic.initialize_carry((config["num_train_envs"],))
            (
                (rng, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, _),
            ) = sample_trajectories_rnn(
                rng=rng,
                env=env,
                env_params=env_params,
                train_state=train_state,
                init_hstate=init_hstate_a,
                init_obs=init_obs,
                init_env_state=init_env_state,
                num_envs=config["num_train_envs"],
                max_episode_length=config["num_steps"],
            )
            advantages, targets = compute_gae(
                gamma=config["gamma"],
                lambd=config["gae_lambda"],
                last_value=last_value,
                values=values,
                rewards=rewards,
                dones=dones,
            )
            max_returns = compute_max_returns(dones=dones, rewards=rewards)

            lp_metrics = {
                "lp_s_in_mean": jnp.nan,
                "lp_loss_before_mean": jnp.nan,
                "lp_loss_after_mean": jnp.nan,
            }
            if config["score_function"] == "s_in":
                rng, rollout_a_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=new_levels,
                    config=config,
                )
                rng, rollout_b_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=new_levels,
                    config=config,
                )
                rng, scores, lp_metrics = compute_s_in_scores(
                    rng=rng,
                    train_state=train_state,
                    rollout_a=rollout_a_sin,
                    rollout_b=rollout_b_sin,
                    config=config,
                )
                compute_grads = False
            else:
                compute_grads = config["score_function"] == "abs_pg"

            # Editor transfer always materializes the one full-batch PPO update.
            # Robust mode keeps it only as the post-update scoring state.
            policy_before_update = train_state
            apply_update = (
                True
                if config["score_function"] == "editor_transfer"
                else config["exploratory_grad_updates"]
            )
            (rng, updated_train_state), (losses, grad_norms) = update_actor_critic_rnn(
                rng=rng,
                train_state=train_state,
                init_hstate=init_hstate_a,
                batch=(obs, actions, dones, log_probs, values, targets, advantages),
                num_envs=config["num_train_envs"],
                n_steps=config["num_steps"],
                n_minibatch=config["num_minibatches"],
                n_epochs=config["epoch_ppo"],
                clip_eps=config["clip_eps"],
                entropy_coeff=config["entropy_coeff"],
                critic_coeff=config["critic_coeff"],
                update_grad=apply_update,
                compute_per_step_grads=compute_grads,
            )

            if config["score_function"] == "editor_transfer":
                train_state, scoring_state = select_editor_transfer_states(
                    original_state=policy_before_update,
                    updated_state=updated_train_state,
                    persist_update=config["exploratory_grad_updates"],
                )
                returns_after = evaluate_editor_transfer_returns(
                    env=eval_env,
                    env_params=env_params,
                    policy=scoring_state,
                    evaluation_batch=transfer_evaluation,
                    max_episode_length=env_params.max_steps_in_episode,
                )
                scores, transfer_diagnostics = compute_editor_transfer_scores(
                    returns_before=returns_before,
                    returns_after=returns_after,
                )
                transfer_metrics = {
                    **transfer_diagnostics,
                    "transfer_attached_bank_count": jnp.array(
                        config["num_train_envs"], dtype=jnp.int32
                    ),
                    "transfer_eval_env_steps": jnp.array(
                        2
                        * config["num_train_envs"]
                        * config["transfer_target_count"]
                        * env_params.max_steps_in_episode,
                        dtype=jnp.int32,
                    ),
                    "transfer_virtual_optimizer_steps": jnp.array(
                        config["num_minibatches"] * config["epoch_ppo"]
                        if not config["exploratory_grad_updates"]
                        else 0,
                        dtype=jnp.int32,
                    ),
                    "transfer_update_is_virtual": jnp.array(
                        not config["exploratory_grad_updates"], dtype=jnp.float32
                    ),
                }
            else:
                train_state = updated_train_state

            if config["score_function"] not in ("s_in", "editor_transfer"):
                # Existing score functions are computed after the PPO update path.
                scores = compute_score(
                    config,
                    dones=dones,
                    values=values,
                    max_returns=max_returns,
                    advantages=advantages,
                    targets=targets,
                    grad_norms=grad_norms,
                )
            level_extras = {"max_return": max_returns}
            if config["score_function"] == "editor_transfer":
                level_extras.update(
                    {
                        "transfer_targets": target_banks,
                        "has_transfer_targets": jnp.ones(
                            config["num_train_envs"], dtype=jnp.bool_
                        ),
                    }
                )
            sampler, _ = level_sampler.insert_batch(
                sampler=sampler,
                levels=new_levels,
                scores=scores,
                level_extras=level_extras,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                **lp_metrics,
                **transfer_metrics,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            This samples levels from the level buffer, and updates the policy on them.

            Tensor trace:
                level_inds: (N,)
                replay levels leaves: (N, ...)
                init_obs/init_env_state leaves: (N, ...)
                rollout obs leaves: (T, N, ...)
                rollout actions/rewards/dones/log_probs/values: (T, N)
                advantages/targets: (T, N)
                max_returns/scores: (N,)
                losses leaves: (E, M)
            """
            sampler = train_state.sampler

            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_train_envs"]
            )
            stored_level_extras = level_sampler.get_levels_extra(sampler, level_inds)
            transfer_metrics = empty_editor_transfer_metrics()
            if config["score_function"] == "editor_transfer":
                target_banks = stored_level_extras["transfer_targets"]
                rng, transfer_evaluation = prepare_editor_transfer_evaluation(
                    rng=rng,
                    env=eval_env,
                    env_params=env_params,
                    target_banks=target_banks,
                )
                returns_before = evaluate_editor_transfer_returns(
                    env=eval_env,
                    env_params=env_params,
                    policy=train_state,
                    evaluation_batch=transfer_evaluation,
                    max_episode_length=env_params.max_steps_in_episode,
                )

            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            init_hstate_a = ActorCritic.initialize_carry((config["num_train_envs"],))
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                init_hstate_a,
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                values,
                rewards,
                dones,
            )
            max_returns = jnp.maximum(
                stored_level_extras["max_return"],
                compute_max_returns(dones, rewards),
            )

            lp_metrics = {
                "lp_s_in_mean": jnp.nan,
                "lp_loss_before_mean": jnp.nan,
                "lp_loss_after_mean": jnp.nan,
            }
            if config["score_function"] == "s_in":
                rng, rollout_a_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=levels,
                    config=config,
                )
                rng, rollout_b_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=levels,
                    config=config,
                )
                rng, scores, lp_metrics = compute_s_in_scores(
                    rng=rng,
                    train_state=train_state,
                    rollout_a=rollout_a_sin,
                    rollout_b=rollout_b_sin,
                    config=config,
                )
                compute_grads = False
            else:
                compute_grads = config["score_function"] == "abs_pg"

            # Update (real PPO update on rollout A)
            (rng, train_state), (losses, grad_norms) = update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate_a,
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
                compute_per_step_grads=compute_grads,
            )

            if config["score_function"] == "editor_transfer":
                returns_after = evaluate_editor_transfer_returns(
                    env=eval_env,
                    env_params=env_params,
                    policy=train_state,
                    evaluation_batch=transfer_evaluation,
                    max_episode_length=env_params.max_steps_in_episode,
                )
                scores, transfer_diagnostics = compute_editor_transfer_scores(
                    returns_before=returns_before,
                    returns_after=returns_after,
                )
                scores, max_returns = aggregate_replay_transfer_updates(
                    level_indices=level_inds,
                    scores=scores,
                    max_returns=max_returns,
                )
                transfer_metrics = {
                    **transfer_diagnostics,
                    "transfer_attached_bank_count": jnp.array(
                        config["num_train_envs"], dtype=jnp.int32
                    ),
                    "transfer_eval_env_steps": jnp.array(
                        2
                        * config["num_train_envs"]
                        * config["transfer_target_count"]
                        * env_params.max_steps_in_episode,
                        dtype=jnp.int32,
                    ),
                    "transfer_virtual_optimizer_steps": jnp.array(0, dtype=jnp.int32),
                    "transfer_update_is_virtual": jnp.array(0.0, dtype=jnp.float32),
                }
            elif config["score_function"] != "s_in":
                # Existing score functions are computed after the PPO update path.
                scores = compute_score(
                    config=config,
                    dones=dones,
                    values=values,
                    max_returns=max_returns,
                    advantages=advantages,
                    targets=targets,
                    grad_norms=grad_norms,
                )
            level_extras = {"max_return": max_returns}
            if config["score_function"] == "editor_transfer":
                level_extras.update(
                    {
                        "transfer_targets": target_banks,
                        "has_transfer_targets": jnp.ones(
                            config["num_train_envs"], dtype=jnp.bool_
                        ),
                    }
                )
            sampler = level_sampler.update_batch(
                sampler=sampler,
                level_inds=level_inds,
                scores=scores,
                level_extras=level_extras,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                **lp_metrics,
                **transfer_metrics,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            This mutates the previous batch of replay levels and potentially adds them to the level buffer.
            This also updates the policy iff `config["exploratory_grad_updates"]` is True.

            Tensor trace:
                parent_levels leaves: (N, ...)
                child_levels leaves: (N, ...)
                init_obs/init_env_state leaves: (N, ...)
                rollout obs leaves: (T, N, ...)
                rollout actions/rewards/dones/log_probs/values: (T, N)
                advantages/targets: (T, N)
                max_returns/scores: (N,)
                losses leaves: (E, M)
            """
            sampler = train_state.sampler
            transfer_metrics = empty_editor_transfer_metrics()
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(
                jax.random.split(rng_mutate, config["num_train_envs"]),
                parent_levels,
                config["num_edits"],
            )
            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(
                jax.random.split(rng_reset, config["num_train_envs"]),
                child_levels,
                env_params,
            )

            # rollout
            init_hstate_a = ActorCritic.initialize_carry((config["num_train_envs"],))
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                init_hstate_a,
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                values,
                rewards,
                dones,
            )
            max_returns = compute_max_returns(dones, rewards)

            lp_metrics = {
                "lp_s_in_mean": jnp.nan,
                "lp_loss_before_mean": jnp.nan,
                "lp_loss_after_mean": jnp.nan,
            }
            if config["score_function"] == "s_in":
                rng, rollout_a_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=child_levels,
                    config=config,
                )
                rng, rollout_b_sin = collect_s_in_rollout_set(
                    rng=rng,
                    env=env,
                    env_params=env_params,
                    train_state=train_state,
                    levels=child_levels,
                    config=config,
                )
                rng, scores, lp_metrics = compute_s_in_scores(
                    rng=rng,
                    train_state=train_state,
                    rollout_a=rollout_a_sin,
                    rollout_b=rollout_b_sin,
                    config=config,
                )
                compute_grads = False
            else:
                compute_grads = config["score_function"] == "abs_pg"

            # Update (real PPO update on rollout A)
            (rng, train_state), (losses, grad_norms) = update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate_a,
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
                compute_per_step_grads=compute_grads,
            )

            if config["score_function"] != "s_in":
                # Existing score functions are computed after the PPO update path.
                scores = compute_score(
                    config,
                    dones=dones,
                    values=values,
                    max_returns=max_returns,
                    advantages=advantages,
                    targets=targets,
                    grad_norms=grad_norms,
                )
            sampler, _ = level_sampler.insert_batch(
                sampler=sampler,
                levels=child_levels,
                scores=scores,
                level_extras={"max_return": max_returns},
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum()
                / config["num_train_envs"],
                **lp_metrics,
                **transfer_metrics,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(
                train_state.sampler, rng_replay
            ) + 2 * s
            return jax.lax.switch(
                branch,
                [
                    on_new_levels,
                    on_replay_levels,
                    on_mutate_levels,
                ],
                rng,
                train_state,
            )
        else:
            branch = level_sampler.sample_replay_decision(
                train_state.sampler, rng_replay
            ).astype(int)
            return jax.lax.switch(
                branch,
                [on_new_levels, on_replay_levels],
                rng,
                train_state,
            )

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)

        Shape legend:
            T_eval = env_params.max_steps_in_episode
            L = len(config["eval_levels"])

        Tensor trace:
            levels leaves: (L, ...)
            init_obs/init_env_state leaves: (L, ...)
            evaluate_rnn outputs:
                states leaves: (T_eval, L, ...)
                rewards: (T_eval, L)
                episode_lengths: (L,)
            mask: (T_eval, L)
            cum_rewards: (L,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng=rng,
            env=eval_env,
            env_params=env_params,
            train_state=train_state,
            init_hstate=ActorCritic.initialize_carry((num_levels,)),
            init_obs=init_obs,
            init_env_state=init_env_state,
            max_episode_length=env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return (
            states,
            cum_rewards,
            episode_lengths,
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
        This function runs the train_step for a certain number of iterations, and then evaluates the policy.
        It returns the updated train state, and a dictionary of metrics.

        Shape legend:
            F = config["eval_freq"]
            A = config["eval_num_attempts"]
            L = len(config["eval_levels"])
            T_eval = env_params.max_steps_in_episode

        Loop tensor trace:
            training scan over F updates:
                metrics from `train_step` gain a leading axis F
            evaluation vmap over A attempts:
                states: (A, T_eval, L, ...)
                cum_rewards: (A, L)
                episode_lengths: (A, L)
            aggregated eval stats:
                eval_returns/eval_solve_rates: (L,)
            visualization tensors:
                first-attempt states: (T_eval, L, ...)
                first-attempt episode_lengths: (L,)
                frames: (T_eval, L, C, H, W)
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )

        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(
            axis=0
        )  # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)

        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(
            states, env_params
        )  # (num_steps, num_eval_levels, ...)
        frames = images.transpose(
            0, 1, 4, 2, 3
        )  # WandB expects color channel before image dimensions when dealing with animations for some reason

        metrics["update_count"] = (
            train_state.num_dr_updates
            + train_state.num_replay_updates
            + train_state.num_mutation_updates
        )
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.dr_last_level_batch, env_params
        )
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.replay_last_level_batch, env_params
        )
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.mutation_last_level_batch, env_params
        )

        highest_scoring_level = level_sampler.get_levels(
            train_state.sampler, train_state.sampler["scores"].argmax()
        )
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler,
            level_sampler.level_weights(train_state.sampler).argmax(),
        )

        metrics["highest_scoring_level"] = env_renderer.render_level(
            highest_scoring_level, env_params
        )
        metrics["highest_weighted_level"] = env_renderer.render_level(
            highest_weighted_level, env_params
        )

        return (rng, train_state), metrics

    def eval_checkpoint(og_config):
        """
        This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
        It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(
                os.path.join(os.getcwd(), checkpoint_directory, "models"),
                item_handlers=ocp.StandardCheckpointHandler(),
            )

            train_state_og: TrainState = create_train_state(rng_init)
            step = (
                checkpoint_manager.latest_step()
                if og_config["checkpoint_to_eval"] == -1
                else og_config["checkpoint_to_eval"]
            )

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint["params"]
            train_state = train_state_og.replace(params=params)
            return train_state, config

        train_state, config = load(rng_init, og_config["checkpoint_directory"])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state
        )
        save_loc = og_config["checkpoint_directory"].replace("checkpoints", "results")
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_loc, "results.npz"),
            states=np.asarray(states),
            cum_rewards=np.asarray(cum_rewards),
            episode_lengths=np.asarray(episode_lengths),
            levels=config["eval_levels"],
        )
        return states, cum_rewards, episode_lengths

    if config["mode"] == "eval":
        return eval_checkpoint(config)  # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(
                eval_step, args=ocp.args.StandardSave(runner_state[1])
            )
            checkpoint_manager.wait_until_finished()
    return runner_state[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_EVAL_FREQ)
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
    group = parser.add_argument_group("Training params")
    # === PPO ===
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === PLR ===
    group.add_argument(
        "--score_function",
        type=str,
        default="MaxMC",
        choices=[
            "MaxMC",
            "pvl",
            "abs_pg",
            "ppo_value_loss",
            "s_in",
            "editor_transfer",
        ],
        help="Score function for level prioritization. "
        "abs_pg uses policy gradient magnitudes. "
        "ppo_value_loss uses PPO clipped value loss magnitude. "
        "s_in uses holdout PPO total-loss reduction after virtual updates. "
        "editor_transfer uses paired before/after returns on fixed edited "
        "target banks.",
    )
    group.add_argument(
        "--transfer_target_count",
        type=int,
        default=128,
        help="Number of fixed editor-chain targets attached to each PLR source.",
    )
    group.add_argument(
        "--transfer_num_edits",
        type=int,
        default=16,
        help="Number of sequential editor applications in every target chain.",
    )
    group.add_argument(
        "--sin_n_virtual_updates",
        type=int,
        default=3,
        help="Number of virtual PPO updates used when computing s_in.",
    )
    group.add_argument(
        "--sin_eps",
        type=float,
        default=1e-8,
        help="Numerical stabilizer for normalized s_in score.",
    )
    group.add_argument(
        "--sin_num_rollouts_per_level",
        type=int,
        default=None,
        help="Number of independent Set A and Set B rollouts per sampled "
        "training environment slot when computing s_in. Required for s_in.",
    )
    group.add_argument(
        "--sin_score_batch_size",
        type=int,
        default=1,
        help="Number of sampled training environment slots to score together "
        "inside s_in. Use 1 for the safest TPU compile path; increase for "
        "more throughput if compilation remains stable.",
    )
    group.add_argument(
        "--exploratory_grad_updates",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXPLORATORY_GRAD_UPDATES,
    )
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument(
        "--prioritization", type=str, default="rank", choices=["rank", "topk"]
    )
    group.add_argument(
        "--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True
    )
    group.add_argument(
        "--pg_n_minibatch",
        type=int,
        default=1,
        help="Number of env minibatches for policy gradient norm estimation. "
        "Higher values reduce memory at the cost of sequential processing.",
    )
    # === ACCEL ===
    group.add_argument(
        "--use_accel",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ACCEL,
    )
    group.add_argument("--num_edits", type=int, default=5)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)

    config = vars(parser.parse_args())
    if (
        config["sin_num_rollouts_per_level"] is not None
        and config["sin_num_rollouts_per_level"] < 1
    ):
        parser.error("--sin_num_rollouts_per_level must be >= 1 when provided.")
    if config["sin_score_batch_size"] < 1:
        parser.error("--sin_score_batch_size must be >= 1.")
    if (
        config["score_function"] == "s_in"
        and config["sin_num_rollouts_per_level"] is None
    ):
        parser.error("--sin_num_rollouts_per_level is required for --score_function s_in.")  # fmt: skip
    try:
        validate_editor_transfer_config(config)
    except ValueError as error:
        parser.error(str(error))
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (
            config["num_train_envs"] * config["num_steps"]
        )
    config["group_name"] = "".join(
        [
            str(config[key])
            for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])
        ]
    )
    if config["wandb_experiment_name"] is None:
        config["wandb_experiment_name"] = (
            f"{config['score_function']}-seed{config['seed']}-walls{config['n_walls']}"
        )
    if config["run_name"] is None and config["mode"] == "train":
        config["run_name"] = normalize_run_name(config["wandb_experiment_name"])
    elif config["run_name"] is not None:
        config["run_name"] = normalize_run_name(config["run_name"])

    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    # wandb.login()
    main(config, project=config["project"])
