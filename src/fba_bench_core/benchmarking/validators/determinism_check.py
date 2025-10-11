"""Determinism check validator."""

from typing import Any

from .registry import register_validator


@register_validator("determinism_check")
def determinism_check(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Check determinism across multiple runs."""
    issues = []
    tolerance = config.get("tolerance", 0.0)
    fields = config.get("fields", ["value"])

    # Group runs by runner_key and seed
    from collections import defaultdict

    groups = defaultdict(list)
    for run in runs:
        key = (run.get("runner_key"), run.get("seed"))
        groups[key].append(run)

    for (runner, seed), group_runs in groups.items():
        if len(group_runs) < 2:
            continue

        for field in fields:
            values = [
                run.get("output", {}).get(field)
                for run in group_runs
                if run.get("status") == "success"
            ]
            if len(values) < 2:
                continue

            # Check if all values are within tolerance
            if isinstance(values[0], (int, float)):
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val > tolerance:
                    issues.append(
                        {
                            "id": "determinism_mismatch",
                            "severity": "error",
                            "message": f"Determinism mismatch for runner '{runner}', seed {seed}, field '{field}': values {values} exceed tolerance {tolerance}",
                            "context": {
                                "runner": runner,
                                "seed": seed,
                                "field": field,
                                "values": values,
                            },
                        }
                    )
            else:
                # For non-numeric, check exact match
                if not all(v == values[0] for v in values):
                    issues.append(
                        {
                            "id": "determinism_mismatch",
                            "severity": "error",
                            "message": f"Determinism mismatch for runner '{runner}', seed {seed}, field '{field}': values {values} not identical",
                            "context": {
                                "runner": runner,
                                "seed": seed,
                                "field": field,
                                "values": values,
                            },
                        }
                    )

    if not issues:
        issues.append(
            {
                "id": "determinism_ok",
                "severity": "info",
                "message": "All deterministic checks passed",
            }
        )

    return {
        "issues": issues,
        "summary": {
            "total_groups_checked": len(groups),
            "groups_with_issues": len(
                [
                    g
                    for g in groups.values()
                    if len(g) >= 2
                    and any(i["id"] == "determinism_mismatch" for i in issues)
                ]
            ),
        },
    }
