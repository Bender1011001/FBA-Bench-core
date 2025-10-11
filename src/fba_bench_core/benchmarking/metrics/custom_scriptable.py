"""Custom scriptable metric."""

from typing import Any

from ..registry import register_metric


def custom_scriptable(run: dict[str, Any], config: dict[str, Any]) -> Any:
    """Calculate custom scriptable metric."""
    expression = config.get("expression", "0")
    try:
        # Simple eval with run and config in scope
        return eval(expression, {"__builtins__": {}}, {"run": run, "config": config})
    except Exception:
        return 0


register_metric("custom_scriptable", custom_scriptable)
