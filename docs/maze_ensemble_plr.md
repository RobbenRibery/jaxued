# Maze Ensemble PLR

`examples/maze_ensemble_plr.py` scores a level by asking whether isolated,
multi-phase virtual learning makes independently initialized policies agree
more on the union of physical states they encounter.

## One scoring pass

For each candidate level, the implementation performs one trial:

1. Instantiate the level once and give every persistent member the same
   initial observation, environment state, and zero recurrent carry.
2. Give every member distinct rollout randomness and collect the phase-zero
   on-policy trajectory.
3. For every member-level pair, clone both policy and optimizer state and run
   PPO only on that member's trajectory for that level.
4. For each additional virtual phase, reset the clone to the exact same
   instantiated environment state and zero recurrent carry, collect a fresh
   on-policy trajectory with the updated clone, and train on that trajectory.
5. At every pre-action decision point in every phase, retain the observation,
   previous-done reset flag, and physical Maze state.
6. Replay the original persistent policy and final virtual policy over those
   same sequences, reconstructing their separate recurrent carries.
7. Compute signed disagreement reduction on the union of phase visits and use
   it as the level score. All virtual state is then discarded.
8. Separately update each persistent member on only its phase-zero full
   rollout. Replay updates always train; new and mutated levels train only with
   `--exploratory_grad_updates`.

For member (i), level (e), and virtual phase (k), the rollout is

\[
\tau_{i,e}^{k}
\sim
p(\tau \mid \theta_{i,e}^{k},x_e^0,\xi_{i,e}^{k}),
\]

followed by

\[
(\theta_{i,e}^{k+1},m_{i,e}^{k+1})
=
\operatorname{PPO}
(\theta_{i,e}^{k},m_{i,e}^{k},\tau_{i,e}^{k}).
\]

The default is three rollout-PPO phases and five PPO epochs per phase. Every
trajectory is on-policy for its current virtual clone. Only the first
trajectory is eligible for persistent training.

## Visited-state support

The physical state is the pose

\[
z=(x,y,d),
\]

encoded as

\[
\operatorname{id}(z)=4(yW+x)+d.
\]

Elapsed episode time is not part of the state identity.

For member (i), define the occurrences of state (z) across every virtual
phase as

\[
O_{i,e}(z)
=
\{(k,t):z_{i,e,k,t}=z\}.
\]

The member-level visitor indicator is

\[
V_{i,e}(z)
=
1\{|O_{i,e}(z)|>0\}.
\]

Only states visited by at least two distinct members are eligible:

\[
S_e
=
\{z:\sum_i V_{i,e}(z)\ge2\}.
\]

If one policy visits the same state repeatedly, its action distributions are
averaged across phase-timestep occurrences before it contributes one vote. Let
\(\bar{\pi}^{-}_{i,e,z}\) and \(\bar{\pi}^{+}_{i,e,z}\) denote the original and
final virtual policy votes on exactly the same occurrences. For either sign
\(q\in\{-,+\}\), state-level epistemic uncertainty is

\[
U^{q}_{e,z}
=
H\!\left(
\frac{1}{\sum_i V_{i,e}(z)}
\sum_i V_{i,e}(z)\bar{\pi}^{q}_{i,e,z}
\right)
-
\frac{1}{\sum_i V_{i,e}(z)}
\sum_i V_{i,e}(z)
H\!\left(\bar{\pi}^{q}_{i,e,z}\right).
\]

The environment-level score is

\[
\operatorname{score}(e)
=
\frac{1}{|S_e|}
\sum_{z\in S_e}
\left(U^{-}_{e,z}-U^{+}_{e,z}\right).
\]

If (S_e) is empty, the implementation returns zero. Scores are not clipped:
a negative value records increased disagreement after virtual learning.

## Why recurrent replay is valid here

The Maze actor's action distribution is determined by its parameters, current
observation, and LSTM carry. Its recurrent input does not include the previous
action or reward. For every phase, the original and final policies start from
separate zero carries and scan the same stored observations with the same
previous-done flags. Hidden states are never copied between policies or
phases.

Later-phase trajectories are off-policy inputs for the original policy replay,
but they are used only to evaluate its action distribution. PPO training in
every phase remains on-policy for the virtual clone that generated that phase.

## Run

```bash
python examples/maze_ensemble_plr.py \
  --num_agents 8 \
  --virtual_rollout_phases 3 \
  --virtual_epoch_ppo 5 \
  --virtual_level_batch_size 32
```

`--num_train_envs` is the number of candidate levels scored together. A
virtual phase uses one level and one PPO minibatch; `--num_minibatches`
controls the persistent update over the full level batch. Candidate levels are
processed concurrently in memory-bounded groups selected by
`--virtual_level_batch_size`.

All other PLR and ACCEL arguments mirror `maze_plr.py` except that
`--score_function` is intentionally absent: this runner always uses ensemble
disagreement reduction.

Actual environment-step accounting remains

\[
\text{actual steps per update}
=
N\times\text{num train envs}\times\text{num steps}.
\]

With (K) virtual phases, total environment simulation is

\[
\text{total steps per update}
=
K\times N\times\text{num train envs}\times\text{num steps}.
\]

## Evaluation and checkpoints

Checkpoints use the dedicated `jaxued-maze-ensemble-v1` format and contain all
member parameters, optimizer states, and shared sampler state. New
virtual-learning controls live in the stored configuration; old ensemble
checkpoints receive compatibility defaults during evaluation.

Evaluation reports the mean and standard deviation across ensemble members and
stochastic attempts. Videos use member 0, attempt 0 only.

```bash
python examples/maze_ensemble_plr.py \
  --mode eval \
  --checkpoint_directory checkpoints/<run_name>/<seed> \
  --checkpoint_to_eval -1
```
