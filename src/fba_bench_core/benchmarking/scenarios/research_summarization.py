"""Research summarization scenario."""

import random
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseScenario


class ResearchSummarizationConfig(BaseModel):
    """Configuration for research summarization scenario."""

    num_docs: int = Field(gt=0, description="Number of documents")
    max_tokens: int = Field(gt=0, description="Maximum tokens per document")
    focus_keywords: list[str] = Field(
        default_factory=list, description="Keywords to focus on"
    )
    noise_probability: float = Field(ge=0.0, le=0.5, description="Probability of noise")


class ResearchSummarizationScenario(BaseScenario):
    """Research summarization scenario class."""

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize the research summarization scenario.

        Args:
            params: A dictionary of parameters that configure this scenario.
        """
        super().__init__(params)
        default_params = {
            "num_docs": 5,
            "max_tokens": 200,
            "focus_keywords": ["research", "findings", "methodology"],
            "noise_probability": 0.1,
        }
        merged_params = {**default_params, **(params or {})}
        self.config = ResearchSummarizationConfig(**merged_params)

    async def run(self, runner: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Asynchronously executes the research summarization scenario.

        Args:
            runner: The agent runner instance.
            payload: Runtime parameters, including 'seed'.

        Returns:
            Dictionary with scenario results (e.g., {"metrics": ..., "summaries": ...}).
        """
        seed = payload.get("seed", 0)
        random.seed(seed)

        # Generate research documents
        documents = self._generate_research_documents()

        # Collect summaries from agent
        summaries = []
        for doc in documents:
            prompt = self._create_summarization_prompt(doc)
            response = await runner.process(prompt)
            # Assume response is a dict with 'content' or direct text
            summary_text = (
                response.get("content", str(response))
                if isinstance(response, dict)
                else str(response)
            )
            summaries.append(
                {
                    "doc_id": doc["id"],
                    "summary": summary_text,
                    "original_content": doc["content"],
                }
            )

        # Evaluate summaries
        metrics = self._evaluate_summaries(summaries)

        return {
            "metrics": metrics,
            "summaries": [s["summary"] for s in summaries],
            "documents": [d["content"] for d in documents],
        }

    def _generate_research_documents(self) -> list[dict[str, Any]]:
        """Generate synthetic research documents based on config."""
        documents = []
        for i in range(self.config.num_docs):
            # Simulate document content with focus keywords and some noise
            base_content = f"Research paper {i + 1}: This study explores key findings in {', '.join(self.config.focus_keywords)}. "
            base_content += f"The methodology involved analysis with approximately {self.config.max_tokens} tokens of data. "

            # Add noise with probability
            if random.random() < self.config.noise_probability:
                base_content += (
                    "Irrelevant detail: weather conditions during the study. "
                )

            # Expand to approximate token length (rough estimate: words ~ tokens/0.75)
            content = base_content * (
                self.config.max_tokens // len(base_content.split()) + 1
            )
            content = " ".join(
                content.split()[: self.config.max_tokens // 4]
            )  # Rough token simulation

            documents.append({"id": i, "content": content})
        return documents

    def _create_summarization_prompt(self, doc: dict[str, Any]) -> str:
        """Create a prompt for summarizing a research document."""
        return (
            f"Summarize the following research paper, focusing on the key {', '.join(self.config.focus_keywords)}.\n\n"
            f"Paper content:\n{doc['content']}\n\n"
            f"Provide a concise summary of 100-200 words highlighting the main findings and methodology."
        )

    def _evaluate_summaries(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate the quality of generated summaries."""
        if not summaries:
            return {
                "average_quality_score": 0.0,
                "keyword_coverage": 0.0,
                "conciseness_score": 0.0,
            }

        total_coverage = 0.0
        total_conciseness = 0.0
        num_keywords = len(self.config.focus_keywords)
        if num_keywords == 0:
            num_keywords = 1  # Avoid division by zero

        for summary_info in summaries:
            summary_lower = summary_info["summary"].lower()
            keywords_lower = [kw.lower() for kw in self.config.focus_keywords]

            # Keyword coverage
            coverage = (
                sum(1 for kw in keywords_lower if kw in summary_lower) / num_keywords
            )
            total_coverage += coverage

            # Conciseness: ratio of summary length to original (target < 20% for good summary)
            original_words = len(summary_info["original_content"].split())
            summary_words = len(summary_info["summary"].split())
            conciseness = (
                1.0 if 0.1 <= summary_words / original_words <= 0.3 else 0.5
            )  # Simple scoring

            total_conciseness += conciseness

        avg_coverage = total_coverage / len(summaries)
        avg_conciseness = total_conciseness / len(summaries)
        avg_quality = (avg_coverage + avg_conciseness) / 2

        return {
            "average_quality_score": round(avg_quality, 4),
            "keyword_coverage": round(avg_coverage, 4),
            "conciseness_score": round(avg_conciseness, 4),
            "total_documents": len(summaries),
        }


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
