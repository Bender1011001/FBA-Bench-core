"""Registry for metrics."""

from collections.abc import Callable


class MetricRegistry:
    _metrics: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, metric_class: Callable) -> None:
        """Register a metric class."""
        cls._metrics[name] = metric_class

    @classmethod
    def create_metric(cls, name: str, config=None) -> Callable | None:
        """Create a metric instance."""
        fn = cls._metrics.get(name)
        if fn:
            return fn
        return None

    @classmethod
    def get_metric(cls, name: str) -> Callable | None:
        """Get a metric class by name."""
        return cls._metrics.get(name)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())


# Global instance for function-based API
registry = MetricRegistry()


def get_metric(name: str) -> Callable:
    """Get a metric by name, raising KeyError if not found."""
    metric = registry.get_metric(name)
    if metric is None:
        raise KeyError(f"Metric '{name}' not found")
    return metric


def list_metrics() -> list[str]:
    """List all registered metric names."""
    return registry.list_metrics()


def register_metric(name: str):
    """Decorator to register a metric function with the given name."""

    def decorator(func: Callable) -> Callable:
        registry.register(name, func)
        return func

    return decorator
