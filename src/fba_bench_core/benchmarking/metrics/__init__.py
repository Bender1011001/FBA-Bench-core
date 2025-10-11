"""Metrics module for benchmarking."""

# Import metric modules to register them
from . import (
    accuracy_score,
    completeness,
    cost_efficiency,
    custom_scriptable,
    keyword_coverage,
    policy_compliance,
    robustness,
    technical_performance,
)
from .registry import register_metric

__all__ = [
    "accuracy_score",
    "completeness",
    "cost_efficiency",
    "custom_scriptable",
    "keyword_coverage",
    "policy_compliance",
    "robustness",
    "technical_performance",
    "register_metric",
]
