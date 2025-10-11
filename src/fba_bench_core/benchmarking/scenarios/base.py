"""Base classes for all scenario types in the FBA-Bench benchmarking framework."""

import abc
from typing import Any


class BaseScenario(abc.ABC):
    """
    Abstract base class for all benchmark scenarios.

    This class defines the minimal interface required for a scenario to be
    executable by the modern benchmarking engine.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initializes the scenario with its parameters.

        Args:
            params: A dictionary of parameters that configure this scenario.
        """
        self.params = params or {}

    @abc.abstractmethod
    async def run(self, runner: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Asynchronously executes the scenario with a given agent runner.

        Args:
            runner: The agent runner instance.
            payload: A dictionary containing runtime parameters, including the seed.

        Returns:
            A dictionary containing the results of the scenario run.
        """
        pass
