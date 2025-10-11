"""Keyword coverage metric."""

from typing import Any

from .registry import register_metric


@register_metric("keyword_coverage")
def keyword_coverage(run: dict[str, Any], config: dict[str, Any]) -> float:
    """Calculate keyword coverage."""
    field_path = config.get("field_path", "")
    keywords = config.get("keywords", [])

    text = ""
    data = run.get("output", {})
    if field_path:
        keys = field_path.split(".")
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, "")
            else:
                break
        text = str(data) if data else ""
    else:
        text = str(data)

    if not keywords:
        return 0.0

    found = sum(1 for kw in keywords if kw in text)
    return found / len(keywords)
