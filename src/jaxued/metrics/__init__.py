"""Public interfaces for environment utility metrics."""

from jaxued.metrics.base import EnvironmentMetric, MetricRegistry
from jaxued.metrics.ensemble import (
    EnsembleDisagreementInputs,
    EnsembleDisagreementResult,
    StateVisitSummary,
    aggregate_state_action_probabilities,
    categorical_entropy,
    compute_ensemble_disagreement_reduction,
)
from jaxued.metrics.rollout import (
    RolloutMetricInputs,
    accumulate_rollout_stats,
    compute_max_mean_returns_epcount,
    compute_max_returns,
    compute_rollout_utility,
    create_rollout_metric_registry,
    max_mc,
    positive_value_loss,
)

__all__ = [
    "EnvironmentMetric",
    "EnsembleDisagreementInputs",
    "EnsembleDisagreementResult",
    "MetricRegistry",
    "RolloutMetricInputs",
    "StateVisitSummary",
    "accumulate_rollout_stats",
    "aggregate_state_action_probabilities",
    "categorical_entropy",
    "compute_ensemble_disagreement_reduction",
    "compute_max_mean_returns_epcount",
    "compute_max_returns",
    "compute_rollout_utility",
    "create_rollout_metric_registry",
    "max_mc",
    "positive_value_loss",
]
