from .models import EngineConfig, EngineReport


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config

    async def run(self) -> EngineReport:
        # Stub implementation
        return EngineReport(scenario_reports=[])
