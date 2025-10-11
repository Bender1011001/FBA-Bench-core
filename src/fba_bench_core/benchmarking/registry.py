"""Global registry for benchmarking components."""

from collections.abc import Callable
from typing import Any


class MetricRegistry:
    """Registry for metrics."""

    _metrics: dict[str, Callable] = {}

    @classmethod
    def register_metric(cls, name: str, metric_fn: Callable) -> None:
        """Register a metric function."""
        cls._metrics[name] = metric_fn

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        """Get a metric function by name."""
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found")
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())

    @classmethod
    def create_metric(cls, name: str, config: dict[str, Any] | None = None) -> Any:
        """Create a metric instance (for compatibility)."""
        return cls.get_metric(name)


# Convenience functions
def register_metric(name: str, metric_fn: Callable) -> None:
    """Register a metric function."""
    MetricRegistry.register_metric(name, metric_fn)


def get_metric(name: str) -> Callable:
    """Get a metric function by name."""
    return MetricRegistry.get_metric(name)


def list_metrics() -> list[str]:
    """List all registered metric names."""
    return MetricRegistry.list_metrics()
