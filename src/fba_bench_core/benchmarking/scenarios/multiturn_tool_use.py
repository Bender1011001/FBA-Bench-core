"""Multi-turn tool use scenario."""

import random
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseScenario


class MultiTurnToolUseConfig(BaseModel):
    """Configuration for multi-turn tool use scenario."""

    steps: int = Field(default=5, gt=0, description="Number of steps")
    include_math: bool = Field(default=True, description="Include math operations")
    include_extraction: bool = Field(
        default=True, description="Include data extraction"
    )
    include_transform: bool = Field(
        default=True, description="Include data transformation"
    )


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


class MultiturnToolUseScenario(BaseScenario):
    """
    Scenario for evaluating agents on multi-turn tool usage across different capabilities.

    Agents must demonstrate effective tool selection and usage over multiple sequential turns,
    handling tasks like mathematical computations, data extraction, and transformations.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize the multiturn tool use scenario.

        Args:
            params: A dictionary of parameters that configure this scenario.
        """
        super().__init__(params)
        self.config = MultiTurnToolUseConfig(**self.params)

    async def run(self, runner: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Asynchronously executes the multiturn tool use scenario.

        This method orchestrates a sequence of turns where the agent must use appropriate tools
        for tasks involving math, extraction, or transformation based on enabled capabilities.
        State is tracked across turns, and metrics are computed based on success rates.

        Args:
            runner: The agent runner instance, expected to have an async `process` method
                    that takes input data and returns a response dict with 'success' bool
                    and optional 'result' for verification.
            payload: Runtime parameters, including 'seed' for reproducible randomness.

        Returns:
            Dictionary with scenario results, including metrics (success rates per capability),
            final state (success counts, attempts), and interaction history.
        """
        seed = payload.get("seed", 0)
        random.seed(seed)  # Ensure determinism

        # Generate initial input using existing helper for consistency
        initial_input = generate_input(seed, self.params)
        capabilities = [
            cap for cap, enabled in initial_input["capabilities"].items() if enabled
        ]
        if not capabilities:
            capabilities = ["basic"]  # Fallback for no capabilities enabled

        # Initialize state for tracking interactions and outcomes
        interactions: list[Any] = []
        successes: dict[str, int] = {cap: 0 for cap in capabilities}
        total_attempts: dict[str, int] = {cap: 0 for cap in capabilities}
        state: dict[str, Any] = {
            "interactions": interactions,
            "successes": successes,
            "total_attempts": total_attempts,
        }

        # Execute multi-turn interactions
        for step in range(self.config.steps):
            # Select task type randomly from enabled capabilities for variety
            task_type = random.choice(capabilities)

            # Generate task-specific input based on capability
            if task_type == "math":
                a, b = random.randint(1, 100), random.randint(1, 100)
                input_data = {
                    "task": "math",
                    "problem": f"Calculate the sum of {a} and {b}.",
                    "expected_result": a + b,  # For potential verification
                }
            elif task_type == "extraction":
                value = random.randint(100, 999)
                input_data = {
                    "task": "extraction",
                    "text": f"The key value in this document is {value}. Extract it.",
                    "expected_result": value,
                }
            elif task_type == "transform":
                data = [random.randint(1, 10) for _ in range(5)]
                input_data = {
                    "task": "transform",
                    "data": data,
                    "operation": "sort ascending",
                    "expected_result": sorted(data),
                }
            else:  # 'basic'
                input_data = {
                    "task": "basic",
                    "query": "Perform a simple tool call to confirm functionality.",
                }

            # Process the turn asynchronously via the runner (assumes tool use handling)
            response = await runner.process(input_data)

            # Track interaction
            state["interactions"].append(
                {
                    "step": step + 1,
                    "task_type": task_type,
                    "input": input_data,
                    "response": response,
                }
            )

            # Update success metrics (assumes runner response indicates success)
            success = response.get("success", False)
            if success:
                state["successes"][task_type] += 1
            state["total_attempts"][task_type] += 1

        # Compute final metrics
        metrics = {}
        overall_attempts = sum(state["total_attempts"].values())
        total_successes = sum(state["successes"].values())
        metrics["overall_success_rate"] = (
            total_successes / overall_attempts if overall_attempts > 0 else 0.0
        )
        metrics["steps_completed"] = self.config.steps

        for cap in capabilities:
            attempts = state["total_attempts"][cap]
            metrics[f"{cap}_success_rate"] = (
                state["successes"][cap] / attempts if attempts > 0 else 0.0
            )

        return {
            "metrics": metrics,
            "final_state": {
                "successes": state["successes"],
                "total_attempts": state["total_attempts"],
            },
            "interactions": state["interactions"],
        }
