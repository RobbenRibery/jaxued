"""Core interfaces for environment utility metrics.

This module contains only configuration-time abstractions. Metric computation
remains in small, pure callables so that a selected metric can be closed over by
JAX-transformed training functions without introducing mutable runtime state.
"""

from typing import Dict, Generic, Iterable, Protocol, Tuple, TypeVar, runtime_checkable

import chex


MetricInputT = TypeVar("MetricInputT")


@runtime_checkable
class EnvironmentMetric(Protocol[MetricInputT]):
    """Callable contract implemented by an environment utility metric.

    The input type is deliberately generic. Training pipelines can define a
    semantic input record containing exactly the signals they produce, while
    retaining the same selection and registration mechanism. Implementations
    should be pure functions of their input so they remain compatible with JAX
    transformations.
    """

    def __call__(self, inputs: MetricInputT) -> chex.Array:
        """Compute one utility score per environment.

        Args:
            inputs: Typed signals required by the metric.

        Returns:
            An array containing one utility score for each environment in the
            input batch.
        """


class MetricRegistry(Generic[MetricInputT]):
    """Configuration-time lookup table for compatible utility metrics.

    A registry groups metrics that consume the same input contract. Selection
    happens before a training function is transformed by JAX; the resolved
    callable can then be used directly inside the traced function.

    Example:
        >>> registry = MetricRegistry[MyMetricInputs]()
        >>> registry.register("my_metric", compute_my_metric)
        >>> metric = registry.resolve("my_metric")
        >>> scores = metric(inputs)
    """

    def __init__(self) -> None:
        """Initialize an empty metric registry."""
        self._metrics: Dict[str, EnvironmentMetric[MetricInputT]] = {}
        self._canonical_names = []

    @property
    def names(self) -> Tuple[str, ...]:
        """Return canonical metric names in registration order.

        Returns:
            Canonical names accepted by the registry. Aliases are intentionally
            omitted so configuration UIs can expose one stable name per metric.
        """
        return tuple(self._canonical_names)

    def register(
        self,
        name: str,
        metric: EnvironmentMetric[MetricInputT],
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register a metric and optional alternative names.

        Args:
            name: Canonical, non-empty name used in configuration.
            metric: Callable that consumes the registry's input type.
            aliases: Additional non-empty names that resolve to ``metric``.

        Raises:
            ValueError: If a name is empty or any requested name is already
                registered.
            TypeError: If ``metric`` is not callable.
        """
        if not callable(metric):
            raise TypeError("metric must be callable")

        names = (name, *tuple(aliases))
        if any(not candidate for candidate in names):
            raise ValueError("metric names and aliases must be non-empty")

        duplicates = [candidate for candidate in names if candidate in self._metrics]
        if duplicates:
            duplicate_list = ", ".join(repr(candidate) for candidate in duplicates)
            raise ValueError(f"metric name already registered: {duplicate_list}")

        for candidate in names:
            self._metrics[candidate] = metric
        self._canonical_names.append(name)

    def resolve(self, name: str) -> EnvironmentMetric[MetricInputT]:
        """Resolve a configured name to its metric callable.

        Args:
            name: Canonical metric name or registered alias.

        Returns:
            The metric callable associated with ``name``.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        try:
            return self._metrics[name]
        except KeyError as error:
            available = ", ".join(self.names) or "<none>"
            raise ValueError(
                f"Unknown environment metric {name!r}. Available metrics: {available}"
            ) from error
