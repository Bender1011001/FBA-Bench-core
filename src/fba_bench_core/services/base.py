"""Typed BaseService for fba_bench_core (Phase D).

This module introduces a typed configuration contract for services and a
minimal abstract base class that requires a validated configuration object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fba_bench_core.config import BaseServiceConfig


class BaseService(ABC):
    """Abstract base class for services that receive typed configuration.

    Rationale:
        As with agents, services should accept explicit, validated configuration
        objects rather than arbitrary `**kwargs`. This improves discoverability,
        maintainability, and prevents accidental acceptance of untyped values.

    Initialization:
        Construct with a `BaseServiceConfig` (or subclass) instance. Access the
        service id via the `service_id` property and the full configuration via
        the `config` property.
    """

    def __init__(self, config: BaseServiceConfig) -> None:
        """Initialize the service with a typed configuration model."""
        self._config = config
        self._service_id = config.service_id

    @property
    def service_id(self) -> str:
        """Return the service's unique identifier (from config.service_id)."""
        return self._service_id

    @property
    def config(self) -> BaseServiceConfig:
        """Return the typed configuration object for this service."""
        return self._config

    def get_config(self) -> dict:
        """Return a serializable shallow mapping of the configuration."""
        return self._config.model_dump()

    @abstractmethod
    def start(self) -> None:
        """Start the service (synchronous API). Implementations should provide
        the concrete startup behavior. Use async variants in concrete classes
        if necessary."""
        raise NotImplementedError
