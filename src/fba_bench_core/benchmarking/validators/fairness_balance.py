"""Fairness balance validator."""

from typing import Any

from .registry import register_validator


@register_validator("fairness_balance")
def fairness_balance(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate fairness balance across runs."""
    issues = []
    group = config.get("group", "runner_key")
    metric_path = config.get("metric_path", "metrics.accuracy")
    threshold = config.get("threshold", 0.1)
    min_group_size = config.get("min_group_size", 2)

    from collections import defaultdict

    groups = defaultdict(list)
    for run in runs:
        group_key = run.get(group)
        if group_key:
            groups[group_key].append(run)

    for group_key, group_runs in groups.items():
        if len(group_runs) < min_group_size:
            continue

        # Extract metric values for successful runs
        values = []
        for run in group_runs:
            if run.get("status") == "success":
                # Handle nested metric path
                current = run
                for key in metric_path.split("."):
                    current = current.get(key, {})
                value = current if isinstance(current, (int, float)) else 0.0
                values.append(value)

        if values:
            min_val = min(values)
            max_val = max(values)
            if (max_val - min_val) / ((min_val + max_val) / 2) > threshold:
                issues.append(
                    {
                        "id": "fairness_imbalance",
                        "severity": "warning",
                        "message": f"Fairness imbalance in group '{group_key}': range {min_val}-{max_val} exceeds threshold {threshold}",
                        "context": {
                            "group": group_key,
                            "metric": metric_path,
                            "min": min_val,
                            "max": max_val,
                            "threshold": threshold,
                        },
                    }
                )
            else:
                issues.append(
                    {
                        "id": "fairness_within_threshold",
                        "severity": "info",
                        "message": f"Fairness within threshold for group '{group_key}'",
                    }
                )

    return {
        "issues": issues,
        "summary": {
            "groups_checked": len(groups),
            "groups_with_imbalance": len(issues),
        },
    }
