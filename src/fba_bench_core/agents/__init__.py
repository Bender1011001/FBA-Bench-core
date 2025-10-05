"""Package exports for fba_bench_core.agents.

Exports the BaseAgent abstract class and exposes the registry module to allow
external code to discover and register agent implementations.
"""
from __future__ import annotations

from . import registry
from .base import BaseAgent

__all__ = ["BaseAgent", "registry"]
