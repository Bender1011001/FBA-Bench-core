"""Registry for metrics."""

from collections.abc import Callable

_metrics: dict[str, Callable] = {}


def register_metric(name: str, metric_class: Callable) -> None:
    """Register a metric class."""
    _metrics[name] = metric_class


def get_metric(name: str) -> Callable | None:
    """Get a metric class by name."""
    return _metrics.get(name)


def list_metrics() -> list[str]:
    """List all registered metric names."""
    return list(_metrics.keys())
