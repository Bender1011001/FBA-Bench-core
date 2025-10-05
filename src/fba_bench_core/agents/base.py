"""Base agent abstractions for fba_bench_core (Phase 5).

This module provides a small, well-documented abstract base class that
concrete agent implementations should inherit from. The BaseAgent class
encapsulates basic identity and configuration storage, and declares the
asynchronous decision contract used by the simulation / scenario runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from fba_bench_core.domain.events import BaseEvent, Command


class BaseAgent(ABC):
    """Abstract base class for agents.

    Purpose:
        Provide a stable, minimal interface that all agent implementations must
        satisfy. The core responsibility is to convert observed domain events
        into a sequence of domain Commands representing intended actions.

    Responsibilities:
        - Maintain an immutable agent identifier.
        - Store agent configuration as a shallow copy to prevent external mutation.
        - Expose an async `decide` method which must be implemented by subclasses.

    Usage:
        Subclasses should implement the async `decide` coroutine. The simulation
        harness will call `await agent.decide(events)` where `events` is a list
        of BaseEvent instances ordered from oldest to newest.
    """

    def __init__(self, agent_id: str, **config: Any) -> None:
        """Initialize the base agent.

        Parameters:
            agent_id: A unique identifier for the agent instance.
            **config: Arbitrary keyword configuration parameters which are stored
                      as a shallow copy on the agent.

        Notes:
            The configuration is shallow-copied (dict(config)) to avoid accidental
            external mutation of the agent's internal configuration mapping.
        """
        self._agent_id = agent_id
        self._config: Dict[str, Any] = dict(config)

    @property
    def agent_id(self) -> str:
        """Return the agent's unique identifier.

        Returns:
            A string identifier assigned at construction time.
        """
        return self._agent_id

    def get_config(self) -> Dict[str, Any]:
        """Return a shallow copy of the agent configuration.

        Returns:
            A shallow copy of the internal configuration mapping (Dict[str, Any]).
        """
        return dict(self._config)

    @abstractmethod
    async def decide(self, events: List[BaseEvent]) -> List[Command]:
        """Decide on a list of Commands given observed domain events.

        Implementations must be an async coroutine. The caller will await the
        result. The decision method receives a list of BaseEvent instances that
        represent recent or historical events ordered from oldest to newest.

        Parameters:
            events: A list of BaseEvent objects used as the information basis
                    for forming decisions. Implementations should treat the
                    provided list as read-only.

        Returns:
            A list of Command instances representing the agent's intended
            actions. An empty list indicates no action.

        Contract and responsibilities:
            - Implementations should avoid blocking I/O; use await for any
              asynchronous operations.
            - Returned Commands should be valid domain Command instances as
              defined in fba_bench_core.domain.events.
            - The method must not mutate the `events` input.
        """
        raise NotImplementedError