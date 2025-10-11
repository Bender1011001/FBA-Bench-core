"""Complex marketplace scenario."""

from typing import Any

from pydantic import BaseModel, Field

from .base import BaseScenario


class ComplexMarketplaceConfig(BaseModel):
    """Configuration for complex marketplace scenario."""

    num_products: int = Field(gt=0, description="Number of products")
    num_orders: int = Field(gt=0, description="Number of orders")
    max_quantity: int = Field(gt=0, description="Maximum quantity per order")
    price_variance: float = Field(ge=0.0, le=1.0, description="Price variance")
    allow_backorder: bool = Field(description="Allow backorders")


def generate_input(
    seed: int = 42, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate input for complex marketplace scenario."""
    if params is None:
        params = {}
    config = ComplexMarketplaceConfig(**params)
    return {
        "seed": seed,
        "scenario": "complex_marketplace",
        "config": config.model_dump(),
        "catalog": [{"id": i, "price": 10.0 + i} for i in range(config.num_products)],
        "orders": [
            {
                "product_id": i % config.num_products,
                "quantity": min(i, config.max_quantity),
            }
            for i in range(config.num_orders)
        ],
        "policies": {"allow_backorder": config.allow_backorder},
    }


def postprocess(output: dict[str, Any]) -> dict[str, Any]:
    """Postprocess output from complex marketplace scenario."""
    processed = output.copy()
    if "revenue" in processed:
        processed["revenue"] = round(processed["revenue"], 2)
    if "fulfilled_rate" in processed:
        processed["fulfilled_rate"] = round(processed["fulfilled_rate"], 4)
    return processed


class ComplexMarketplaceScenario(BaseScenario):
    """Complex marketplace scenario."""

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize the complex marketplace scenario.

        Args:
            params: A dictionary of parameters that configure this scenario.
        """
        super().__init__(params)

    async def run(self, runner: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Asynchronously executes the complex marketplace scenario with a given agent runner.

        Args:
            runner: The agent runner instance.
            payload: A dictionary containing runtime parameters, including the seed.

        Returns:
            A dictionary containing the results of the scenario run.
        """
        seed = payload.get("seed", 42)

        # Generate scenario input using configuration parameters
        input_data = generate_input(seed=seed, params=self.params)

        # Execute the scenario using the provided runner
        # Assuming runner has an async 'run' method that takes input_data and returns agent output
        agent_output = await runner.run(input_data)

        # Postprocess the agent output
        result = postprocess(agent_output)

        return result
