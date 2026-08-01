"""Functional level buffer used by PLR and ACCEL training loops."""

from typing import Literal, Optional, Tuple, TypedDict

import chex
import jax
import jax.numpy as jnp

from jaxued.environments.underspecified_env import Level


Prioritization = Literal["rank", "topk"]


class Sampler(TypedDict):
    """Array-backed state passed to :class:`LevelSampler` operations.

    Attributes:
        levels: Batched level PyTree with leading dimension ``capacity``.
        scores: Utility score for each buffer slot.
        timestamps: Episode counter at which each slot was last sampled.
        size: Number of populated slots.
        episode_count: Total number of insertions and replay samples processed.
        levels_extra: Optional PyTree of metadata stored alongside levels.
    """

    levels: chex.Array  # shape (capacity, ...)
    scores: chex.Array  # shape (capacity)
    timestamps: chex.Array  # shape (capacity)
    size: int
    episode_count: int
    levels_extra: Optional[dict]


class LevelSampler:
    """Manage an immutable-style level buffer for PLR and ACCEL.

    ``LevelSampler`` stores configuration only. All evolving arrays live in a
    separate :class:`Sampler` PyTree supplied to each operation, which makes the
    operations suitable for JAX transformations.

    Example:
        pholder_level       = ...
        pholder_level_extra = ...
        level_sampler       = LevelSampler(4000)
        sampler             = level_sampler.initialize(pholder_level, pholder_level_extra)
        should_replay       = level_sampler.sample_replay_decision(sampler, rng)
        replay_levels       = level_sampler.sample_replay_levels(sampler, rng, 32) # 32 replay levels
        scores              = ... # eval agent
        sampler             = level_sampler.insert_batch(sampler, level, scores)

    Args:
        capacity: Maximum number of levels stored in the buffer.
        replay_prob: Probability of selecting replay once the minimum fill
            ratio is met.
        staleness_coeff: Mixture coefficient applied to staleness weights;
            ``1 - staleness_coeff`` is applied to score weights.
        minimum_fill_ratio: Required populated fraction before replay is
            allowed.
        prioritization: Score weighting strategy, either ``"rank"`` or
            ``"topk"``.
        prioritization_params: Parameters for the selected strategy. Rank uses
            ``{"temperature": float}``; top-k uses ``{"k": int}``.
        duplicate_check: Whether insertion searches for an identical stored
            level before allocating a new slot.
    """

    def __init__(
        self,
        capacity: int,
        replay_prob: float = 0.95,
        staleness_coeff: float = 0.5,
        minimum_fill_ratio: float = 1.0,  # minimum fill required before replay can occur
        prioritization: Prioritization = "rank",
        prioritization_params: dict = None,
        duplicate_check: bool = False,
    ):
        """Initialize level-buffer configuration.

        Args:
            capacity: Maximum number of stored levels.
            replay_prob: Probability of replay after minimum fill is reached.
            staleness_coeff: Weight assigned to staleness during sampling.
            minimum_fill_ratio: Populated fraction required before replay.
            prioritization: Score weighting strategy.
            prioritization_params: Optional strategy-specific parameters.
            duplicate_check: Whether to update identical levels in place.

        Raises:
            Exception: If ``prioritization`` is unsupported and default
                prioritization parameters must be constructed.
        """
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coeff = staleness_coeff
        self.minimum_fill_ratio = minimum_fill_ratio
        self.prioritization = prioritization
        self.prioritization_params = prioritization_params
        self.duplicate_check = duplicate_check

        if prioritization_params is None:
            if prioritization == "rank":
                self.prioritization_params = {"temperature": 1.0}
            elif prioritization == "topk":
                self.prioritization_params = {"k": 1}
            else:
                raise Exception(f'"{prioritization}" not a valid prioritization.')

    def initialize(self, pholder_level: Level, pholder_level_extra=None) -> Sampler:
        """Initialize array storage from placeholder PyTrees.

        Args:
            pholder_level: Level whose leaf shapes and dtypes determine buffer
                storage.
            pholder_level_extra: Optional metadata PyTree whose leaves are
                repeated alongside every level slot.

        Returns:
            An empty sampler state with ``capacity`` allocated slots, scores
            initialized to negative infinity, and counters initialized to zero.
        """
        sampler = {
            "levels": jax.tree_util.tree_map(
                lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level
            ),
            "scores": jnp.full(self.capacity, -jnp.inf, dtype=jnp.float32),
            "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
            "size": 0,
            "episode_count": 0,
        }
        if pholder_level_extra is not None:
            sampler["levels_extra"] = jax.tree_util.tree_map(
                lambda x: jnp.array([x]).repeat(self.capacity, axis=0),
                pholder_level_extra,
            )
        return sampler

    def sample_replay_decision(self, sampler: Sampler, rng: chex.PRNGKey) -> bool:
        """Choose whether the next update should replay a stored level.

        Args:
            sampler: Current level-buffer state.
            rng: JAX random key used for the Bernoulli decision.

        Returns:
            True when the buffer is sufficiently full and the sampled uniform
            value is below ``replay_prob``.
        """
        proportion_filled = self._proportion_filled(sampler)
        return (proportion_filled >= self.minimum_fill_ratio) & (
            jax.random.uniform(rng) < self.replay_prob
        )

    def sample_replay_level(
        self, sampler: Sampler, rng: chex.PRNGKey
    ) -> Tuple[Sampler, Tuple[int, Level]]:
        """Sample one stored level according to current level weights.

        Args:
            sampler: Current level-buffer state.
            rng: JAX random key used for categorical sampling.

        Returns:
            A tuple containing updated sampler state and ``(index, level)``.
            Sampling increments the episode counter and timestamps the selected
            slot.
        """
        weights = self.level_weights(sampler)
        idx = jax.random.choice(rng, self.capacity, p=weights)
        new_episode_count = sampler["episode_count"] + 1
        sampler = {
            **sampler,
            "timestamps": sampler["timestamps"].at[idx].set(new_episode_count),
            "episode_count": new_episode_count,
        }
        return sampler, (
            idx,
            jax.tree_util.tree_map(lambda x: x[idx], sampler["levels"]),
        )

    def sample_replay_levels(
        self, sampler: Sampler, rng: chex.PRNGKey, num: int
    ) -> Tuple[Sampler, Tuple[chex.Array, Level]]:
        """Sample a batch of replay levels sequentially.

        Args:
            sampler: Current level-buffer state.
            rng: JAX random key split into one key per sample.
            num: Number of levels to sample.

        Returns:
            Updated sampler state together with sampled indices and a batched
            level PyTree.
        """
        return jax.lax.scan(
            self.sample_replay_level, sampler, jax.random.split(rng, num), length=num
        )

    def insert(
        self, sampler: Sampler, level: Level, score: float, level_extra: dict = None
    ) -> Tuple[Sampler, int]:
        """Attempt to insert or update one level.

        An unfilled buffer accepts the level at its next free slot. A full
        buffer replaces its lowest-weighted slot only when ``score`` exceeds
        that slot's current score. With duplicate checking enabled, an identical
        level is updated in place.

        Args:
            sampler: Current level-buffer state.
            level: Level PyTree to insert.
            score: Utility score associated with ``level``.
            level_extra: Optional metadata matching the structure supplied to
                :meth:`initialize`.

        Returns:
            Updated sampler state and the affected slot index. The index is
            ``-1`` when a full buffer rejects the candidate.
        """
        if self.duplicate_check:
            idx = self.find(sampler, level)
            return jax.lax.cond(
                idx == -1,
                lambda: self._insert_new(sampler, level, score, level_extra),
                lambda: (
                    {
                        **self.update(
                            sampler, idx, score, level_extra
                        ),  # what happens to mutation rate here?
                        "timestamps": sampler["timestamps"]
                        .at[idx]
                        .set(sampler["episode_count"] + 1),
                        "episode_count": sampler["episode_count"] + 1,
                    },
                    idx,
                ),
            )
        return self._insert_new(sampler, level, score, level_extra)

    def insert_batch(
        self,
        sampler: Sampler,
        levels: Level,
        scores: chex.Array,
        level_extras: dict = None,
    ) -> Tuple[Sampler, chex.Array]:
        """Insert a batch of levels sequentially.

        Args:
            sampler: Current level-buffer state.
            levels: Batched level PyTree with the batch dimension first.
            scores: Utility scores aligned with the leading level dimension.
            level_extras: Optional batched metadata PyTree.

        Returns:
            Updated sampler state and an array of affected indices, with ``-1``
            for rejected candidates.
        """

        def _insert(sampler, step):
            level, score, level_extra = step
            return self.insert(sampler, level, score, level_extra)

        return jax.lax.scan(_insert, sampler, (levels, scores, level_extras))

    def find(self, sampler: Sampler, level: Level) -> int:
        """Find an identical level in the populated buffer.

        Args:
            sampler: Current level-buffer state.
            level: Level PyTree to compare against stored entries.

        Returns:
            Index of the first identical populated entry, or ``-1`` if absent.
        """
        eq_tree = jax.tree_util.tree_map(
            lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1),
            sampler["levels"],
            level,
        )
        eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
        eq_mask = jnp.array(eq_tree_flat).all(axis=0) & (
            jnp.arange(self.capacity) < sampler["size"]
        )
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)

    def get_levels(self, sampler: Sampler, level_idx: int) -> Level:
        """Read one or more levels by index.

        Args:
            sampler: Current level-buffer state.
            level_idx: Scalar or array index applied to every level leaf.

        Returns:
            Level PyTree selected from the buffer.
        """
        return jax.tree_util.tree_map(lambda x: x[level_idx], sampler["levels"])

    def get_levels_extra(self, sampler: Sampler, level_idx: int) -> dict:
        """Read metadata associated with one or more levels.

        Args:
            sampler: Current level-buffer state.
            level_idx: Scalar or array index applied to every metadata leaf.

        Returns:
            Metadata PyTree selected from ``sampler["levels_extra"]``.
        """
        return jax.tree_util.tree_map(lambda x: x[level_idx], sampler["levels_extra"])

    def update(
        self, sampler: Sampler, idx: int, score: float, level_extra: dict = None
    ) -> Sampler:
        """Update the score and optional metadata for one stored level.

        Args:
            sampler: Current level-buffer state.
            idx: Slot to update.
            score: Replacement utility score.
            level_extra: Optional replacement metadata PyTree.

        Returns:
            New sampler mapping containing the updated arrays.
        """
        new_sampler = {
            **sampler,
            "scores": sampler["scores"].at[idx].set(score),
        }
        if level_extra is not None:
            new_sampler["levels_extra"] = jax.tree_util.tree_map(
                lambda x, y: x.at[idx].set(y), new_sampler["levels_extra"], level_extra
            )
        return new_sampler

    def update_batch(
        self,
        sampler: Sampler,
        level_inds: chex.Array,
        scores: chex.Array,
        level_extras: dict = None,
    ) -> Sampler:
        """Update scores and optional metadata for a batch of slots.

        Args:
            sampler: Current level-buffer state.
            level_inds: One-dimensional array of slot indices.
            scores: Replacement score for each index.
            level_extras: Optional batched metadata aligned with ``level_inds``.

        Returns:
            Sampler state after applying updates sequentially.
        """

        def _update(sampler, step):
            level_idx, score, level_extra = step
            return self.update(sampler, level_idx, score, level_extra), None

        return jax.lax.scan(_update, sampler, (level_inds, scores, level_extras))[0]

    def level_weights(
        self,
        sampler: Sampler,
        prioritization: Prioritization = None,
        prioritization_params: dict = None,
    ) -> chex.Array:
        """Combine score and staleness weights for replay sampling.

        Args:
            sampler: Current level-buffer state.
            prioritization: Optional score-weighting strategy override.
            prioritization_params: Optional parameters for the override.

        Returns:
            Sampling weights with shape ``(capacity,)``.
        """
        w_s = self.score_weights(sampler, prioritization, prioritization_params)
        w_c = self.staleness_weights(sampler)
        return (1 - self.staleness_coeff) * w_s + self.staleness_coeff * w_c

    def score_weights(
        self,
        sampler: Sampler,
        prioritization: Prioritization = None,
        prioritization_params: dict = None,
    ) -> chex.Array:
        """Convert stored utility scores into replay probabilities.

        Rank prioritization applies an inverse-rank power law controlled by
        ``temperature``. Top-k prioritization applies softmax only over the
        highest-ranked ``k`` populated slots.

        Args:
            sampler: Current level-buffer state.
            prioritization: Optional strategy override.
            prioritization_params: Optional parameters for the selected
                strategy.

        Returns:
            Normalized score weights with shape ``(capacity,)``.

        Raises:
            Exception: If the selected prioritization strategy is unsupported.
        """
        mask = jnp.arange(self.capacity) < sampler["size"]

        if prioritization is None:
            prioritization = self.prioritization
        if prioritization_params is None:
            prioritization_params = self.prioritization_params

        if prioritization == "rank":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            ranks = jnp.empty_like(ord).at[ord].set(jnp.arange(len(ord)) + 1)
            temperature = prioritization_params["temperature"]
            w_s = jnp.where(mask, 1 / ranks, 0) ** (1 / temperature)
            w_s = w_s / w_s.sum()
        elif prioritization == "topk":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            k = prioritization_params["k"]
            topk_mask = (
                jnp.empty_like(ord)
                .at[ord]
                .set(jnp.arange(self.capacity) < jnp.minimum(sampler["size"], k))
            )
            w_s = jax.nn.softmax(sampler["scores"], where=topk_mask, initial=0)
        else:
            raise Exception(f'"{self.prioritization}" not a valid prioritization.')

        return w_s

    def staleness_weights(self, sampler: Sampler) -> chex.Array:
        """Compute replay weights proportional to time since last sampling.

        Args:
            sampler: Current level-buffer state.

        Returns:
            Normalized staleness weights with shape ``(capacity,)``. When all
            populated entries have zero staleness, weight is uniform over those
            entries.
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        staleness = sampler["episode_count"] - sampler["timestamps"]
        w_c = jnp.where(mask, staleness, 0)
        w_c = jax.lax.select(w_c.sum() > 0, w_c / w_c.sum(), mask / sampler["size"])
        return w_c

    def freshness_weights(self, sampler: Sampler) -> chex.Array:
        """Compute weights proportional to recency relative to the oldest slot.

        Args:
            sampler: Current level-buffer state.

        Returns:
            Normalized freshness weights with shape ``(capacity,)``. When all
            populated entries have equal timestamps, weight is uniform over
            those entries.
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        earliest_timestamp = jnp.where(
            mask, sampler["timestamps"], jnp.iinfo(jnp.int32).max
        ).min()
        freshness = sampler["timestamps"] - earliest_timestamp
        w_f = jnp.where(mask, freshness, 0)
        w_f = jax.lax.select(w_f.sum() > 0, w_f / w_f.sum(), mask / sampler["size"])
        return w_f

    def flush(self, sampler: Sampler) -> Sampler:
        """Mark all slots empty and reset their scores in place.

        Args:
            sampler: Current mutable sampler mapping.

        Returns:
            The same mapping with size zero and all scores set to negative
            infinity. Level arrays, timestamps, metadata, and episode count are
            retained.
        """
        sampler["size"] = 0
        sampler["scores"] = jnp.full(self.capacity, -jnp.inf, dtype=jnp.float32)
        return sampler

    def _insert_new(
        self, sampler: Sampler, level: Level, score: float, level_extra: dict
    ) -> Tuple[Sampler, int]:
        """Insert a candidate into the next free or lowest-weighted slot.

        Args:
            sampler: Current level-buffer state.
            level: Candidate level PyTree.
            score: Candidate utility score.
            level_extra: Optional candidate metadata.

        Returns:
            Updated state and affected index, or ``-1`` when rejected.
        """
        idx = self._get_next_idx(sampler)
        replace_cond = sampler["scores"][idx] < score

        def _replace():
            new_sampler = {
                **sampler,
                "levels": jax.tree_util.tree_map(
                    lambda x, y: x.at[idx].set(y), sampler["levels"], level
                ),
                "scores": sampler["scores"].at[idx].set(score),
                "timestamps": sampler["timestamps"]
                .at[idx]
                .set(sampler["episode_count"] + 1),
                "size": jnp.minimum(sampler["size"] + 1, self.capacity),
            }
            if level_extra is not None:
                new_sampler["levels_extra"] = jax.tree_util.tree_map(
                    lambda x, y: x.at[idx].set(y),
                    new_sampler["levels_extra"],
                    level_extra,
                )
            return new_sampler

        new_sampler = jax.lax.cond(replace_cond, _replace, lambda: sampler)
        new_sampler["episode_count"] += 1

        return new_sampler, jax.lax.select(replace_cond, idx, -1)

    def _proportion_filled(self, sampler: Sampler) -> float:
        """Return the fraction of allocated slots that are populated.

        Args:
            sampler: Current level-buffer state.

        Returns:
            Scalar fill ratio in the interval ``[0, 1]``.
        """
        return sampler["size"] / self.capacity

    def _get_next_idx(self, sampler: Sampler) -> int:
        """Choose the slot considered by the next insertion.

        Args:
            sampler: Current level-buffer state.

        Returns:
            The next unused slot, or the index with minimum replay weight when
            the buffer is full.
        """
        return jax.lax.select(
            sampler["size"] < self.capacity,
            sampler["size"],
            self.level_weights(sampler).argmin(),
        )
