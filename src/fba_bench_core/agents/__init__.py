"""Package exports for fba_bench_core.agents.

Exports the BaseAgent abstract class and exposes the registry module to allow
external code to discover and register agent implementations. Also export the
typed base configuration model to make it easy for downstream users to extend.
"""

from __future__ import annotations

from fba_bench_core.config import BaseAgentConfig

from . import registry
from .base import BaseAgent

__all__ = ["BaseAgent", "registry", "BaseAgentConfig"]
