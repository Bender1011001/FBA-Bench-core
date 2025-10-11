"""Structural consistency validator."""

from typing import Any

from .registry import register_validator


@register_validator("structural_consistency")
def structural_consistency(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate structural consistency across runs."""
    issues = []

    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            issues.append(
                {
                    "id": "invalid_run_type",
                    "severity": "error",
                    "message": f"Run {i} is not a dict: {type(run)}",
                    "context": {"index": i, "type": type(run)},
                }
            )
            continue

        # Check required fields
        required_fields = [
            "scenario_key",
            "runner_key",
            "status",
            "duration_ms",
            "metrics",
            "output",
        ]
        for field in required_fields:
            if field not in run:
                issues.append(
                    {
                        "id": "missing_field",
                        "severity": "error",
                        "message": f"Missing required field '{field}' in run {i}",
                        "context": {"index": i, "field": field},
                    }
                )

        # Check duration_ms non-negative
        duration = run.get("duration_ms")
        if isinstance(duration, (int, float)) and duration < 0:
            issues.append(
                {
                    "id": "negative_duration",
                    "severity": "warning",
                    "message": f"Negative duration_ms {duration} in run {i}",
                    "context": {"index": i, "duration": duration},
                }
            )

        # Check output only on success
        status = run.get("status")
        if status != "success" and run.get("output") is not None:
            issues.append(
                {
                    "id": "unexpected_output_on_failure",
                    "severity": "info",
                    "message": f"Output present on non-success status '{status}' in run {i}",
                    "context": {"index": i, "status": status},
                }
            )

    return {
        "issues": issues,
        "summary": {"total_runs": len(runs), "structural_issues": len(issues)},
    }
