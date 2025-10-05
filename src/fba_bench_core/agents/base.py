"""Typed BaseAgent for fba_bench_core.

Phase D change:
- Replace legacy **kwargs configuration with a typed Pydantic configuration
  object. Downstream implementations should subclass the provided config model
  for specialized parameters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from fba_bench_core.config import BaseAgentConfig
from fba_bench_core.domain.events import BaseEvent, Command


class BaseAgent(ABC):
    """Abstract base class for agents that receive a validated configuration.

    Rationale:
        Using a typed configuration object prevents downstream implementations
        from hiding untyped parameters behind `Any` and enables validation at
        construction time. Implementations that require additional fields may
        subclass `BaseAgentConfig` (see examples in the config module).

    Initialization:
        The agent receives a single `config: BaseAgentConfig` argument. The
        `agent_id` is expected to be present on that config and becomes the
        agent's immutable identifier.

    Immutability:
        The provided config model is frozen (Pydantic frozen model). The agent
        stores the config object directly and exposes it via a read-only
        property to avoid accidental mutation.
    """

    def __init__(self, config: BaseAgentConfig) -> None:
        """Initialize the base agent with a validated, typed configuration.

        Parameters:
            config: An instance of BaseAgentConfig or a subclass thereof. The
                    model is validated by Pydantic prior to construction.

        Notes:
            - Do not accept `**kwargs` here: typed configs are required.
            - The agent keeps a reference to the provided config (which is
              immutable/frozen). Use `agent.config.model_copy()` to obtain a
              mutable copy if necessary.
        """
        self._config = config
        self._agent_id = config.agent_id

    @property
    def agent_id(self) -> str:
        """Return the agent's unique identifier (from config.agent_id)."""
        return self._agent_id

    @property
    def config(self) -> BaseAgentConfig:
        """Return the typed configuration object for this agent.

        The returned object is immutable (Pydantic frozen model). Downstream
        code that needs to modify configuration should create a new instance
        (e.g., via `model_copy(update={...})`).
        """
        return self._config

    def get_config(self) -> Dict:
        """Return a serializable shallow mapping of the configuration.

        Returns:
            A dict produced by Pydantic's model_dump() representing the config.
        """
        return self._config.model_dump()

    @abstractmethod
    async def decide(self, events: List[BaseEvent]) -> List[Command]:
        """Decide on a list of Commands given observed domain events.

        Implementations must be async coroutines and must not mutate the
        provided `events` list.
        """
        raise NotImplementedError