"""Research summarization scenario."""

from typing import Any

from pydantic import BaseModel, Field


class ResearchSummarizationConfig(BaseModel):
    """Configuration for research summarization scenario."""

    num_docs: int = Field(gt=0, description="Number of documents")
    max_tokens: int = Field(gt=0, description="Maximum tokens per document")
    focus_keywords: list[str] = Field(description="Keywords to focus on")
    noise_probability: float = Field(ge=0.0, le=0.5, description="Probability of noise")


def generate_input(
    seed: int = 42, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate input for research summarization scenario."""
    if params is None:
        params = {}
    config = ResearchSummarizationConfig(**params)
    return {
        "seed": seed,
        "scenario": "research_summarization",
        "config": config.model_dump(),
        "documents": [
            {"id": i, "content": f"Document {i} content", "tokens": config.max_tokens}
            for i in range(config.num_docs)
        ],
        "keywords": config.focus_keywords,
        "noise_prob": config.noise_probability,
    }
