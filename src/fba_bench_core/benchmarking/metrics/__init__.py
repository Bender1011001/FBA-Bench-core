"""Metrics module for benchmarking."""

# Import metric modules to register them
from . import (
    accuracy_score,
    aggregate,
    completeness,
    cost_efficiency,
    custom_scriptable,
    keyword_coverage,
    policy_compliance,
    robustness,
    technical_performance,
)
from .registry import get_metric, list_metrics, register_metric

__all__ = [
    "accuracy_score",
    "aggregate",
    "completeness",
    "cost_efficiency",
    "custom_scriptable",
    "keyword_coverage",
    "policy_compliance",
    "robustness",
    "technical_performance",
    "get_metric",
    "list_metrics",
    "register_metric",
]
