"""Exports for fba_bench_core.services package.

Expose the BaseService class and its typed configuration model.
"""
from __future__ import annotations

from .base import BaseService
from fba_bench_core.config import BaseServiceConfig

__all__ = ["BaseService", "BaseServiceConfig"]