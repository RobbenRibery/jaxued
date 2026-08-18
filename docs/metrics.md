# Environment utility metrics

JaxUED separates environment utility measurement from level storage and
sampling. A metric is a pure callable that accepts a typed input record and
returns one score per environment. A registry selects that callable once during
configuration, before the training step is transformed by JAX.

The built-in rollout pipeline provides four metrics:

- `MaxMC`, which scores the gap between the maximum observed return and the
  critic value;
- `pvl`, which scores positive value loss;
- `mean_positive_delight`, which averages positive GAE multiplied by the
  sampled action's collection-time surprisal over completed episodes;
- `mean_absolute_advantage`, which averages the magnitude of the GAE residual
  over rollout timesteps independently for each environment.

All four consume `RolloutMetricInputs`. The Maze, Gymnax, and Craftax PLR
examples resolve the configured metric from `create_rollout_metric_registry()`
and call it through `compute_rollout_utility()`. Metric selection is therefore
outside the rollout and level-sampler implementations.

Select mean absolute advantage from any registry-backed PLR command line with:

```console
--score_function mean_absolute_advantage
```

Select mean positive delight with:

```console
--score_function mean_positive_delight
```

For each completed episode, `mean_positive_delight` averages
`max(advantage * -log_prob, 0)` across timesteps, then averages those episode
scores. For a discrete policy this is surprisal-weighted positive value loss.
The log-probability is the value recorded when the rollout action was sampled.
For continuous policies, `-log_prob` is a differential surprisal and can be
negative, so the same formula does not reduce to surprisal-weighted PVL.

## Adding a compatible rollout metric

A project-specific metric using the existing rollout signals only needs a pure
function and one registration call:

```python
import jax.numpy as jnp

from jaxued.metrics import RolloutMetricInputs, create_rollout_metric_registry


def mean_squared_advantage(inputs: RolloutMetricInputs):
    return jnp.square(inputs.advantages).mean(axis=0)


registry = create_rollout_metric_registry()
registry.register("mean_squared_advantage", mean_squared_advantage)
metric = registry.resolve("mean_squared_advantage")
```

The PLR example `main` functions accept this registry through their
`metric_registry` argument, so programmatic callers do not need to modify the
training loop:

```python
from examples.maze_plr import main

config["score_function"] = "mean_squared_advantage"
main(config, metric_registry=registry)
```

Metrics requiring a different evidence source define a separate typed input
record rather than being forced into the rollout-only contract. The visited-
state ensemble scorer uses `EnsembleDisagreementInputs`: visit-averaged action
probabilities before and after virtual learning, plus one frozen visitor mask.
The multi-phase Maze runner builds that mask from the union of all phase
visits; single-phase callers retain their original behavior.

`aggregate_state_action_probabilities()` gives every policy one vote per
physical state by averaging that policy's repeat visits first.
`compute_ensemble_disagreement_reduction()` then:

1. excludes policies that did not visit a state;
2. excludes states visited by fewer than two policies;
3. measures Jensen-Shannon-style action disagreement before and after the
   virtual update on exactly the same support; and
4. returns the raw signed reduction for each level.

The complete rollout, virtual-PPO, recurrent-replay, and persistent-training
pipeline is implemented in `examples/maze_ensemble_plr.py` and documented in
[Maze Ensemble PLR](maze_ensemble_plr.md).
