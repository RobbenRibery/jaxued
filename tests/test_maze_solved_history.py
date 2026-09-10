"""Lifetime success evidence must survive PLR eviction, rejection and checkpoints."""

from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from examples.maze_plr import (
    compute_solved_informed_transfer_update,
    resolve_new_solved_informed_history,
)
from jaxued.environments.maze import Level, make_level_generator
from jaxued.environments.maze import solved_history as history_module
from jaxued.environments.maze.solved_history import (
    SolvedLevelHistory,
    check_solved_history_capacity,
    create_solved_level_history,
    encode_level_keys,
    lookup_solved_levels,
    observe_level_successes,
)
from jaxued.level_sampler import LevelSampler


def _levels():
    return tuple(
        make_level_generator(5, 5, 3)(jax.random.PRNGKey(seed))
        for seed in (401, 402, 403)
    )


def _batch(*levels):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *levels)


def _extras(update):
    return {
        "max_return": update.max_returns,
        "transfer_unsolved_failure_count": update.failure_counts,
        "transfer_ever_solved": update.ever_solved,
        "transfer_solved_confidence": update.confidences,
        "transfer_base_log_relative_score": update.base_scores,
    }


def _sampler(level, *, capacity=1, duplicate_check=True):
    api = LevelSampler(capacity=capacity, duplicate_check=duplicate_check)
    state = api.initialize(
        level,
        {
            "max_return": jnp.array(-jnp.inf),
            "transfer_unsolved_failure_count": jnp.array(0, dtype=jnp.int32),
            "transfer_ever_solved": jnp.array(False),
            "transfer_solved_confidence": jnp.array(0.8),
            "transfer_base_log_relative_score": jnp.array(0.0),
        },
    )
    return api, state


@partial(jax.jit, static_argnames=("duplicate_check",))
def _score_new_sources(history, sampler, levels, current_solved, duplicate_check=True):
    groups, failures, prior_solved, prior_returns = resolve_new_solved_informed_history(
        sampler, levels, duplicate_check
    )
    observed = observe_level_successes(history, levels, current_solved)
    update = compute_solved_informed_transfer_update(
        group_ids=groups,
        base_scores=jnp.zeros(current_solved.shape),
        current_solved=current_solved,
        prior_failure_counts=failures,
        prior_ever_solved=prior_solved | observed.previously_solved,
        known_ever_solved=observed.ever_solved,
        current_max_returns=current_solved.astype(jnp.float32),
        prior_max_returns=prior_returns,
    )
    return observed.history, update


def test_exact_identity_includes_every_level_field():
    level = Level.from_str(".....\n.>...\n...G.\n.....\n.....")
    variants = (
        level,
        level.replace(wall_map=level.wall_map.at[4, 4].set(True)),
        level.replace(agent_pos=jnp.array([2, 1], jnp.uint32)),
        level.replace(goal_pos=jnp.array([2, 2], jnp.uint32)),
        level.replace(agent_dir=jnp.array(1, jnp.uint8)),
        level.replace(width=4),
        level.replace(height=4),
    )
    keys = encode_level_keys(_batch(*variants))
    assert jnp.unique(keys, axis=0).shape[0] == len(variants)
    history = create_solved_level_history(level, max_observations=10)
    observation = jax.jit(observe_level_successes)(
        history, _batch(level), jnp.array([True])
    )
    assert jnp.array_equal(
        lookup_solved_levels(observation.history, _batch(*variants)),
        jnp.array([True, False, False, False, False, False, False]),
    )


def test_same_walls_with_different_backing_shapes_have_different_keys():
    a = Level.from_str(">.G\n...")
    b = Level.from_str(">.\nG.\n..")
    # Six identical wall bits fit in one word, but array dimensions differ.
    assert not jnp.array_equal(
        encode_level_keys(_batch(a)), encode_level_keys(_batch(b))
    )


def test_wall_key_preserves_bits_across_word_boundaries():
    level = Level.from_str(">G...........\n" + ".............\n" * 12)
    variants = [level]
    for index in (2, 31, 32, 63, 64, 127, 128, 168):
        variants.append(
            level.replace(wall_map=level.wall_map.at[index // 13, index % 13].set(True))
        )
    keys = encode_level_keys(_batch(*variants))
    assert jnp.unique(keys, axis=0).shape[0] == len(variants)


def test_jitted_observation_scan_keeps_all_successes_for_full_reserved_horizon():
    levels = jax.vmap(make_level_generator(5, 5, 3))(
        jax.random.split(jax.random.PRNGKey(7), 64)
    )
    history = create_solved_level_history(
        jax.tree_util.tree_map(lambda value: value[0], levels), max_observations=64
    )
    batches = jax.tree_util.tree_map(
        lambda value: value.reshape((16, 4) + value.shape[1:]), levels
    )
    flags = jnp.arange(64).reshape(16, 4) % 3 != 0

    @jax.jit
    def run(history):
        def step(history, item):
            batch, successes = item
            return observe_level_successes(history, batch, successes).history, None

        return jax.lax.scan(step, history, (batches, flags))[0]

    history = run(history)
    assert jnp.array_equal(lookup_solved_levels(history, levels), flags.reshape(-1))
    assert int(history.size) == int(flags.sum())
    check_solved_history_capacity(history)


def test_collision_checks_full_key_and_full_table_never_claims_false_success(
    monkeypatch,
):
    a, b, c = _levels()
    monkeypatch.setattr(history_module, "_key_hash", lambda key: jnp.uint32(0))
    history = create_solved_level_history(a, max_observations=1)  # Two slots.
    observed = jax.jit(observe_level_successes)(
        history, _batch(a, b, a), jnp.array([True, True, True])
    )
    assert int(observed.history.size) == 2
    assert not bool(observed.history.overflow)
    assert jnp.array_equal(
        lookup_solved_levels(observed.history, _batch(a, b, c)),
        jnp.array([True, True, False]),
    )
    exhausted = observe_level_successes(observed.history, _batch(c), jnp.array([True]))
    assert int(exhausted.history.size) == 2
    assert not bool(exhausted.ever_solved[0])
    with pytest.raises(RuntimeError, match="refusing to forget"):
        check_solved_history_capacity(exhausted.history)
    assert jnp.all(lookup_solved_levels(exhausted.history, _batch(a, b)))


def test_success_survives_eviction_and_regeneration():
    a, b, _ = _levels()
    api, sampler = _sampler(a)
    history = create_solved_level_history(a, max_observations=10)
    history, solved = _score_new_sources(history, sampler, _batch(a), jnp.array([True]))
    sampler, _ = api.insert_batch(sampler, _batch(a), solved.scores, _extras(solved))
    sampler, replaced = api.insert_batch(
        sampler, _batch(b), jnp.array([10.0]), _extras(solved)
    )
    assert int(replaced[0]) == 0
    assert int(api.find(sampler, a)) == -1

    history, regenerated = _score_new_sources(
        history, sampler, _batch(a), jnp.array([False])
    )
    assert bool(regenerated.ever_solved[0])
    assert not bool(regenerated.newly_solved[0])
    assert float(regenerated.confidences[0]) == 1.0
    assert float(regenerated.penalties[0]) == 0.0
    assert float(regenerated.scores[0]) == 0.0
    assert int(regenerated.failure_increments[0]) == 0
    assert int(history.size) == 1


def test_success_is_recorded_even_when_plr_rejects_candidate():
    a, b, _ = _levels()
    api, sampler = _sampler(a)
    history = create_solved_level_history(a, max_observations=10)
    history, keeper = _score_new_sources(
        history, sampler, _batch(b), jnp.array([False])
    )
    sampler, _ = api.insert_batch(
        sampler, _batch(b), jnp.array([10.0]), _extras(keeper)
    )
    history, solved = _score_new_sources(history, sampler, _batch(a), jnp.array([True]))
    sampler, rejected = api.insert_batch(
        sampler, _batch(a), solved.scores, _extras(solved)
    )
    assert int(rejected[0]) == -1
    _, regenerated = _score_new_sources(history, sampler, _batch(a), jnp.array([False]))
    assert bool(regenerated.ever_solved[0])
    assert float(regenerated.penalties[0]) == 0.0


@pytest.mark.parametrize("duplicate_check", [False, True])
def test_batch_success_shared_by_identity_without_changing_raw_rollout_metrics(
    duplicate_check,
):
    a, b, _ = _levels()
    _, sampler = _sampler(a, duplicate_check=duplicate_check)
    history = create_solved_level_history(a, max_observations=10)
    flags = jnp.array([False, True, False, False])
    history, update = _score_new_sources(
        history, sampler, _batch(a, a, a, b), flags, duplicate_check
    )
    assert jnp.array_equal(update.current_solved, flags)
    assert jnp.array_equal(update.ever_solved, jnp.array([True, True, True, False]))
    assert jnp.array_equal(update.newly_solved, jnp.array([True, True, True, False]))
    assert jnp.all(update.confidences[:3] == 1)
    assert float(update.confidences[3]) < 1
    assert int(history.size) == 1


def test_replay_uses_lifetime_success_even_if_resident_metadata_is_stale():
    a, _, _ = _levels()
    history = create_solved_level_history(a, max_observations=10)
    history = observe_level_successes(history, _batch(a), jnp.array([True])).history
    observed = observe_level_successes(history, _batch(a), jnp.array([False]))
    update = jax.jit(compute_solved_informed_transfer_update)(
        group_ids=jnp.array([0]),
        base_scores=jnp.array([-0.25]),
        current_solved=jnp.array([False]),
        prior_failure_counts=jnp.array([5]),
        prior_ever_solved=jnp.array([False]) | observed.previously_solved,
        known_ever_solved=observed.ever_solved,
        current_max_returns=jnp.array([0.0]),
        prior_max_returns=jnp.array([0.0]),
    )
    assert bool(update.ever_solved[0])
    assert float(update.scores[0]) == -0.25
    assert int(update.failure_counts[0]) == 5
    assert not bool(update.newly_solved[0])


def test_archive_survives_checkpoint_and_keeps_accepting_new_successes(tmp_path):
    a, b, _ = _levels()
    history = create_solved_level_history(a, max_observations=10)
    history = observe_level_successes(history, _batch(a), jnp.array([True])).history
    path = tmp_path / "lifetime-history"
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(path, {"solved_level_history": history})
        restored = checkpointer.restore(
            path, args=ocp.args.StandardRestore({"solved_level_history": history})
        )
    finally:
        checkpointer.close()
    history = restored["solved_level_history"]
    assert isinstance(history, SolvedLevelHistory)
    assert jnp.array_equal(
        lookup_solved_levels(history, _batch(a, b)), jnp.array([True, False])
    )
    observed = jax.jit(observe_level_successes)(history, _batch(b), jnp.array([True]))
    assert jnp.all(lookup_solved_levels(observed.history, _batch(a, b)))
    assert int(observed.history.size) == 2


def test_independent_training_histories_do_not_share_success():
    a, _, _ = _levels()
    history = create_solved_level_history(a, max_observations=10)
    observed = observe_level_successes(history, _batch(a), jnp.array([True]))
    assert bool(observed.ever_solved[0])
    assert not bool(lookup_solved_levels(history, _batch(a))[0])


@pytest.mark.parametrize("horizon", [-1, 2**30])
def test_invalid_history_capacity_rejected(horizon):
    with pytest.raises(ValueError):
        create_solved_level_history(_levels()[0], max_observations=horizon)
