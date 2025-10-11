"""Phase 5 placeholder for the agent registry.

This module will hold mappings of agent names to agent classes and will be
populated during Phase 5. It intentionally contains no runtime logic now.
"""

AGENT_REGISTRY: dict[str, type] = {}  # Populated in a later phase (Phase 5)


def create_runner(key: str, config: dict):
    """Stub function for creating runners."""

    class DummyRunner:
        agent_id = config.get("agent_id", "dummy")

    return DummyRunner()
