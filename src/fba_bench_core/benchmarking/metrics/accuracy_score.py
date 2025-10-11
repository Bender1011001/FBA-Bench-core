"""Accuracy score metric."""

from typing import Any

from ..registry import register_metric


def accuracy_score(run: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate accuracy score."""
    output = run.get("output", "")
    expected = config.get("expected_output", "")
    mode = config.get("mode", "exact")
    case_insensitive = config.get("case_insensitive", False)

    if case_insensitive:
        output = output.lower()
        expected = expected.lower()

    if mode == "exact":
        return 1.0 if output == expected else 0.0
    elif mode == "contains":
        return 1.0 if expected in output else 0.0
    else:
        return 0.0  # default


register_metric("accuracy_score", accuracy_score)
