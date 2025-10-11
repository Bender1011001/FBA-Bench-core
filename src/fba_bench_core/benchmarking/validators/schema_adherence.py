"""Schema adherence validator."""

from typing import Any

from .registry import register_validator


@register_validator("schema_adherence")
def schema_adherence(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate schema adherence of runs."""
    contract = config.get("contract", {})
    required_fields = contract.get("required", {})
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

        for field_name, field_type in required_fields.items():
            if field_name not in run:
                issues.append(
                    {
                        "id": "missing_field",
                        "severity": "error",
                        "message": f"Missing required field '{field_name}' in run {i}",
                        "context": {"index": i, "field": field_name},
                    }
                )
            else:
                value = run[field_name]
                if field_type == "int" and not isinstance(value, int):
                    issues.append(
                        {
                            "id": "schema_type_mismatch",
                            "severity": "warning",
                            "message": f"Field '{field_name}' in run {i} has type {type(value)} but expected {field_type}",
                            "context": {
                                "index": i,
                                "field": field_name,
                                "expected": field_type,
                                "actual": type(value),
                            },
                        }
                    )

    return {
        "issues": issues,
        "summary": {"total_runs": len(runs), "validation_errors": len(issues)},
    }
