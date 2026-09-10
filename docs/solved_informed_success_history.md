# Lifetime source-level success history

The `editor_solved_informed_log_relative_transfer` scorer uses an append-only
success archive separate from the PLR buffer. Once a **source rollout** produces
any reward greater than zero, the same level definition receives `q = 1` on all
future scoring visits in that training run, including after eviction, rejection,
or regeneration. Target-evaluation returns do not set this flag.

Identity includes every Maze level field: the full wall map and its backing
dimensions, width, height, initial agent position/direction, and goal position.
Wall bits are packed losslessly. A hash selects a starting slot, but full key
equality resolves collisions; a hash collision cannot mark another level solved.

Both new-source and replay branches record successes before PLR admission/update.
Identical definitions share success even with PLR duplicate checking disabled.
The archive is part of `TrainState` and its standard checkpoints. Independent
training runs/seeds start with independent archives. Old checkpoints without an
archive cannot recover successes that were already evicted.

Only the success flag has lifetime scope. Unsolved failure counts retain the
existing buffer-local semantics. The formula `S = B + log(q)` and PLR insertion,
rank/staleness weighting, and replay mechanics are unchanged. Stored per-entry
flags are caches refreshed when scored; the archive is authoritative.

## Capacity and verification

The archive reserves two slots per possible source observation over the complete
configured horizon (`num_updates * num_train_envs`), so even every observation
being a new solved level fills at most half the table. It never evicts entries.
An explicit overflow guard stops before logging/checkpointing a state that could
have lost a success. Extending training beyond the reserved horizon requires a
larger archive, not clearing the old one.

For the default 13-by-13 Maze and 30,000 updates with 32 sources, the archive uses
approximately 112 MiB for keys and occupancy, separate from PLR storage.
`lifetime_solved_count` reports its count without rescanning all keys.

Regression tests cover full-key identity, hash collisions, bit packing, JIT
scans, eviction, admission rejection, duplicate-check modes, replay with stale
resident metadata, and checkpoint restoration followed by further observations.
