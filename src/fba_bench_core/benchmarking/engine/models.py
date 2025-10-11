from typing import Any

from pydantic import BaseModel, field_validator


class RunnerSpec(BaseModel):
    key: str
    config: dict[str, Any]


class ScenarioSpec(BaseModel):
    key: str
    timeout_seconds: float | None = None
    repetitions: int | None = None
    seeds: list[int] | None = None


class EngineConfig(BaseModel):
    scenarios: list[ScenarioSpec]
    runners: list[RunnerSpec]
    metrics: list[str] | None = None
    validators: list[str] | None = None
    parallelism: int = 1
    retries: int = 0

    @field_validator("parallelism")
    @classmethod
    def validate_parallelism(cls, v):
        if v <= 0:
            raise ValueError("parallelism must be greater than 0")
        return v

    @field_validator("scenarios")
    @classmethod
    def validate_scenarios(cls, v):
        if not v:
            raise ValueError("scenarios cannot be empty")
        return v

    @field_validator("runners")
    @classmethod
    def validate_runners(cls, v):
        if not v:
            raise ValueError("runners cannot be empty")
        return v


class RunReport(BaseModel):
    status: str
    output: Any = None
    seed: int | None = None
    metrics: dict[str, Any] = {}


class ScenarioReport(BaseModel):
    key: str
    runs: list[RunReport] = []
    aggregates: dict[str, Any] = {}


class EngineReport(BaseModel):
    scenario_reports: list[ScenarioReport]
