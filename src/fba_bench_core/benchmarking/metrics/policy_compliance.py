"""Policy compliance metric."""

from typing import Any

from .registry import register_metric


@register_metric("policy_compliance")
def policy_compliance(run: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Calculate policy compliance."""
    output = run.get("output", {})
    violations = output.get("policy_violations", 0)
    return {"policy_violations": violations, "compliant": violations == 0}
