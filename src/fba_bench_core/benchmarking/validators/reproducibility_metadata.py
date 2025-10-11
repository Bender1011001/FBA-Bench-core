"""Reproducibility metadata validator."""

from typing import Any

from .registry import register_validator


@register_validator("reproducibility_metadata")
def reproducibility_metadata(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate presence of reproducibility metadata."""
    expected_seeds = config.get("expected_seeds", [])
    config_digest = config.get("config_digest", "")
    issues = []

    for i, run in enumerate(runs):
        seed = run.get("seed")
        if seed is None:
            issues.append(
                {
                    "id": "missing_seed",
                    "severity": "warning",
                    "message": f"Run {i} missing seed for reproducibility",
                    "context": {"index": i},
                }
            )
        elif seed not in expected_seeds:
            issues.append(
                {
                    "id": "unexpected_seed",
                    "severity": "warning",
                    "message": f"Run {i} has unexpected seed {seed}, expected {expected_seeds}",
                    "context": {"index": i, "seed": seed, "expected": expected_seeds},
                }
            )

        # Check for per-run config digest
        run_digest = run.get("config_digest")
        if run_digest is None:
            issues.append(
                {
                    "id": "per_run_digest_missing",
                    "severity": "info",
                    "message": f"Run {i} missing per-run config digest",
                }
            )
        elif run_digest != config_digest:
            issues.append(
                {
                    "id": "config_digest_mismatch",
                    "severity": "error",
                    "message": f"Run {i} config digest {run_digest} does not match expected {config_digest}",
                    "context": {
                        "index": i,
                        "expected": config_digest,
                        "actual": run_digest,
                    },
                }
            )

    return {
        "issues": issues,
        "summary": {
            "total_runs": len(runs),
            "missing_seeds": len([r for r in runs if r.get("seed") is None]),
            "unexpected_seeds": len(
                [r for r in runs if r.get("seed") not in expected_seeds]
            ),
            "digest_mismatches": len(
                [r for r in runs if r.get("config_digest") != config_digest]
            ),
        },
    }
