"""Multi-turn tool use scenario."""

from typing import Any

from pydantic import BaseModel, Field


class MultiTurnToolUseConfig(BaseModel):
    """Configuration for multi-turn tool use scenario."""

    steps: int = Field(gt=0, description="Number of steps")
    include_math: bool = Field(description="Include math operations")
    include_extraction: bool = Field(description="Include data extraction")
    include_transform: bool = Field(description="Include data transformation")


def generate_input(
    seed: int = 42, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate input for multi-turn tool use scenario."""
    if params is None:
        params = {}
    config = MultiTurnToolUseConfig(**params)
    return {
        "seed": seed,
        "scenario": "multiturn_tool_use",
        "config": config.model_dump(),
        "steps": config.steps,
        "capabilities": {
            "math": config.include_math,
            "extraction": config.include_extraction,
            "transform": config.include_transform,
        },
    }
