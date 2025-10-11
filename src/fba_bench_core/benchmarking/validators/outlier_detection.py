"""Outlier detection validator."""

from typing import Any

from .registry import register_validator


@register_validator("outlier_detection")
def outlier_detection(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Detect outliers in runs."""
    issues = []
    k = config.get("k", 1.5)
    field = config.get("field", "duration_ms")

    durations = [run.get(field, 0) for run in runs if run.get("status") == "success"]
    if len(durations) < 3:
        return {"issues": [], "summary": {"checked": len(durations), "outliers": []}}

    median = sorted(durations)[len(durations) // 2]
    deviations = [abs(d - median) for d in durations]
    mad = sorted(deviations)[len(deviations) // 2]

    outlier_indices = []
    for i, dev in enumerate(deviations):
        if dev > k * mad:
            outlier_indices.append(i)

    for idx in outlier_indices:
        issues.append(
            {
                "id": "duration_outlier",
                "severity": "warning",
                "message": f"Outlier duration at index {idx}: {durations[idx]} (median: {median}, MAD: {mad})",
                "context": {
                    "index": idx,
                    "value": durations[idx],
                    "median": median,
                    "mad": mad,
                },
            }
        )

    return {
        "issues": issues,
        "summary": {
            "total_runs": len(durations),
            "outliers": outlier_indices,
            "median_duration": median,
            "mad_duration": mad,
        },
    }
