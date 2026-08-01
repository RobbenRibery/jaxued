# Environment utility metrics

JaxUED separates environment utility measurement from level storage and
sampling. A metric is a pure callable that accepts a typed input record and
returns one score per environment. A registry selects that callable once during
configuration, before the training step is transformed by JAX.

The built-in rollout pipeline provides two metrics with unchanged behavior:

- `MaxMC`, which scores the gap between the maximum observed return and the
  critic value;
- `pvl`, which scores positive value loss.

Both consume `RolloutMetricInputs`. The Maze, Gymnax, and Craftax PLR examples
resolve the configured metric from `create_rollout_metric_registry()` and call
it through `compute_rollout_utility()`. Metric selection is therefore outside
the rollout and level-sampler implementations.

## Adding a compatible rollout metric

A project-specific metric using the existing rollout signals only needs a pure
function and one registration call:

```python
import jax.numpy as jnp

from jaxued.metrics import RolloutMetricInputs, create_rollout_metric_registry


def mean_absolute_advantage(inputs: RolloutMetricInputs):
    return jnp.abs(inputs.advantages).mean(axis=0)


registry = create_rollout_metric_registry()
registry.register("mean_absolute_advantage", mean_absolute_advantage)
metric = registry.resolve("mean_absolute_advantage")
```

The PLR example `main` functions accept this registry through their
`metric_registry` argument, so programmatic callers do not need to modify the
training loop:

```python
from examples.maze_plr import main

config["score_function"] = "mean_absolute_advantage"
main(config, metric_registry=registry)
```

Metrics requiring a different evidence source define a separate typed input
record rather than being forced into the rollout-only contract. The visited-
state ensemble scorer uses `EnsembleDisagreementInputs`: visit-averaged action
probabilities before and after a virtual update, plus a frozen pre-update
visitor mask.

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
