"""Backward-compatible imports for rollout utilities.

The rollout statistics and environment utility metrics now live in
``jaxued.metrics.rollout``. They are re-exported here so existing user code that
imports from ``jaxued.utils`` continues to work unchanged.
"""

from jaxued.metrics.rollout import (
    accumulate_rollout_stats,
    compute_max_mean_returns_epcount,
    compute_max_returns,
    max_mc,
    mean_absolute_advantage,
    mean_positive_delight,
    positive_value_loss,
)

__all__ = [
    "accumulate_rollout_stats",
    "compute_max_mean_returns_epcount",
    "compute_max_returns",
    "max_mc",
    "mean_absolute_advantage",
    "mean_positive_delight",
    "positive_value_loss",
]
