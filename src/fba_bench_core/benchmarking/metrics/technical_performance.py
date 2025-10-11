"""Technical performance metric."""

from typing import Any

from ..registry import register_metric


def technical_performance(
    run: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Calculate technical performance metrics."""
    duration_ms = run.get("duration_ms", 0)
    latency_threshold_ms = config.get("latency_threshold_ms", 1000)
    fast_enough = duration_ms <= latency_threshold_ms
    return {"fast_enough": fast_enough}


register_metric("technical_performance", technical_performance)
