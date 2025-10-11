def create_runner(key: str, config: dict):
    """Stub function for creating runners."""

    class DummyRunner:
        agent_id = config.get("agent_id", "dummy")

    return DummyRunner()


class Registry:
    @staticmethod
    def create_runner(key: str, config: dict):
        return create_runner(key, config)


registry = Registry()
