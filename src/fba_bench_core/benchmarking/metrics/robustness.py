"""Robustness metric."""

from typing import Any

from ..registry import register_metric


def robustness(run: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate robustness."""
    output = run.get("output", "")
    expected_signal = config.get("expected_signal", "")
    mode = config.get("mode", "exact")

    if mode == "exact_casefold":
        return 1.0 if output.lower() == expected_signal.lower() else 0.0
    return 0.0


register_metric("robustness", robustness)
