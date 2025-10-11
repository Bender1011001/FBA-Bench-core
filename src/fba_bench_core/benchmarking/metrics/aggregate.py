"""Aggregation utilities for metrics."""

from typing import Any

from .registry import register_metric


@register_metric("aggregate")
def aggregate_all(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all metrics from a list of data points."""
    if not data:
        return {}
    # Simple implementation: return the first item
    return data[0] if data else {}


def aggregate_metric_values(data: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Aggregate metric values for a specific field."""
    values = [item.get(field) for item in data if field in item]
    if not values:
        return {}
    # Filter out None values for type safety
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return {}
    return {
        "mean": sum(clean_values) / len(clean_values),
        "min": min(clean_values),
        "max": max(clean_values),
        "count": len(clean_values),
    }
