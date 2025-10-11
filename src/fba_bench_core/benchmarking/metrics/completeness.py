"""Completeness metric."""

from typing import Any

from .registry import register_metric


@register_metric("completeness")
def completeness(run: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate completeness."""
    output = run.get("output", {})
    required_fields = config.get("required_fields", [])
    if not required_fields:
        return 1.0
    present = sum(1 for field in required_fields if field in output)
    return present / len(required_fields)
