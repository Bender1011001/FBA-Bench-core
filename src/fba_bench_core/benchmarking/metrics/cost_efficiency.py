"""Cost efficiency metric."""

from typing import Any

from .registry import register_metric


@register_metric("cost_efficiency")
def cost_efficiency(run: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate cost efficiency."""
    output = run.get("output", {})
    cost = output.get("cost", 0)
    score_value = config.get("score_value", 1.0)
    if cost == 0:
        return 0.0
    return score_value / cost
