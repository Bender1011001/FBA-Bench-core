from collections.abc import Callable


class ScenarioRegistry:
    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, key: str, fn: Callable):
        self._registry[key] = fn

    def get(self, key: str) -> Callable:
        return self._registry[key]

    def clear(self):
        self._registry.clear()


scenario_registry = ScenarioRegistry()
