# Maze Ensemble PLR

`examples/maze_ensemble_plr.py` scores a level by asking whether one isolated
PPO update makes independently initialized policies agree more on the physical
states they actually encountered.

## One scoring pass

For each candidate level, the implementation performs one trial:

1. Instantiate the level once and give every persistent ensemble member the
   same initial observation, environment state, and zero recurrent carry.
2. Give every member distinct rollout randomness. Each member samples its own
   actions and collects its own on-policy trajectory.
3. At every pre-action decision point, freeze the observation, previous-done
   reset flag, action distribution, and physical Maze state.
4. For every member-level pair, clone both policy and optimizer state, run PPO
   only on that member's trajectory for that level, and discard the clone after
   scoring.
5. Starting from a fresh recurrent carry, run the updated policy over the exact
   stored observation/reset sequence. No environment step, new state, or new
   action is generated during this replay.
6. Compute the signed disagreement reduction and use it as the level score.
7. Separately update each persistent member on its own full rollout batch.
   Replay updates always train; new and mutated levels train only with
   `--exploratory_grad_updates`.

The physical state is the pose

\[
z = (x, y, d),
\]

encoded in code as

\[
\operatorname{id}(z) = 4 (y W + x) + d.
\]

Elapsed episode time is not part of the state identity.

For level \(e\), the policies that visited state \(z\) are

\[
\mathcal{V}_{e,z}
=
\{ i \in \{1,\ldots,N\} : \exists t,\ z_{i,e,t}=z \}.
\]

Only states with at least two distinct visitors are eligible:

\[
\mathcal{S}_e
=
\{ z : |\mathcal{V}_{e,z}| \ge 2 \}.
\]

If one policy visits the same state repeatedly, its action distributions are
averaged before it contributes one vote. Let \(\bar{\pi}^{-}_{i,e,z}\) and
\(\bar{\pi}^{+}_{i,e,z}\) denote those pre-update and post-update votes. For
either sign \(q \in \{-,+\}\), state-level epistemic uncertainty is

\[
U^{q}_{e,z}
=
H\!\left(
\frac{1}{|\mathcal{V}_{e,z}|}
\sum_{i \in \mathcal{V}_{e,z}}
\bar{\pi}^{q}_{i,e,z}
\right)
-
\frac{1}{|\mathcal{V}_{e,z}|}
\sum_{i \in \mathcal{V}_{e,z}}
H\!\left(\bar{\pi}^{q}_{i,e,z}\right).
\]

The environment-level score is the simple signed mean reduction

\[
\operatorname{score}(e)
=
\frac{1}{|\mathcal{S}_e|}
\sum_{z \in \mathcal{S}_e}
\left(U^{-}_{e,z} - U^{+}_{e,z}\right).
\]

If \(\mathcal{S}_e\) is empty, the implementation returns zero. Scores are not
clipped: a negative value records increased disagreement after virtual
learning.

## Why recurrent replay is valid here

The Maze actor's action distribution is determined by its parameters, current
observation, and LSTM carry. Its recurrent input does not include the previous
action or reward. The carry at every stored decision point can therefore be
reconstructed by starting at the standard zero carry and scanning the stored
observations with the stored previous-done flags. Post-update carries are never
copied from the pre-update rollout.

## Run

```bash
python examples/maze_ensemble_plr.py --num_agents 8
```

`--num_train_envs` is the number of candidate levels scored together. A virtual
update always uses one level and one PPO minibatch; `--num_minibatches` controls
the later persistent update over the full level batch. All other PLR and ACCEL
arguments mirror `maze_plr.py` except that `--score_function` is intentionally
absent: this runner always uses ensemble disagreement reduction.

Environment-step accounting includes all member rollouts:

\[
\text{environment steps per update}
=
N \times \text{num train envs} \times \text{num steps}.
\]

## Evaluation and checkpoints

Checkpoints use the dedicated `jaxued-maze-ensemble-v1` format and contain all
member parameters, all optimizer states, and the shared sampler. They are not
interchangeable with single-policy Maze checkpoints.

Evaluation reports the mean and standard deviation across ensemble members and
stochastic attempts. Videos use member 0, attempt 0 only.

```bash
python examples/maze_ensemble_plr.py \
  --mode eval \
  --checkpoint_directory checkpoints/<run_name>/<seed> \
  --checkpoint_to_eval -1
```
