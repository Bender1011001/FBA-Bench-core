from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

# Import scenario modules to ensure registration happens on import
from fba_bench_core.benchmarking.scenarios import registry as sc_reg  # noqa: F401
from fba_bench_core.benchmarking.scenarios.complex_marketplace import (
    ComplexMarketplaceConfig,
)
from fba_bench_core.benchmarking.scenarios.complex_marketplace import (
    generate_input as cm_generate_input,
)
from fba_bench_core.benchmarking.scenarios.complex_marketplace import (
    postprocess as cm_postprocess,
)
from fba_bench_core.benchmarking.scenarios.multiturn_tool_use import (
    MultiTurnToolUseConfig,
    MultiturnToolUseScenario,
)
from fba_bench_core.benchmarking.scenarios.multiturn_tool_use import (
    generate_input as mt_generate_input,
)
from fba_bench_core.benchmarking.scenarios.research_summarization import (
    ResearchSummarizationConfig,
    ResearchSummarizationScenario,
)
from fba_bench_core.benchmarking.scenarios.research_summarization import (
    generate_input as rs_generate_input,
)


def test_complex_marketplace_generate_input_determinism():
    params = {
        "num_products": 10,
        "num_orders": 15,
        "max_quantity": 3,
        "price_variance": 0.12,
        "allow_backorder": False,
    }
    seed = 42
    p1 = cm_generate_input(seed=seed, params=params)
    p2 = cm_generate_input(seed=seed, params=params)
    assert p1 == p2, (
        "ComplexMarketplace generate_input should be deterministic for same seed/params"
    )

    # Different seed should differ
    p3 = cm_generate_input(seed=seed + 1, params=params)
    assert p1 != p3, "ComplexMarketplace generate_input should vary with different seed"


def test_research_summarization_generate_input_determinism():
    params = {
        "num_docs": 5,
        "max_tokens": 120,
        "focus_keywords": ["Q3", "revenue"],
        "noise_probability": 0.1,
    }
    seed = 123
    p1 = rs_generate_input(seed=seed, params=params)
    p2 = rs_generate_input(seed=seed, params=params)
    assert p1 == p2
    p3 = rs_generate_input(seed=seed + 2, params=params)
    assert p1 != p3


def test_multiturn_tool_use_generate_input_determinism():
    params = {
        "steps": 3,
        "include_math": True,
        "include_extraction": True,
        "include_transform": True,
    }
    seed = 7
    p1 = mt_generate_input(seed=seed, params=params)
    p2 = mt_generate_input(seed=seed, params=params)
    assert p1 == p2
    p3 = mt_generate_input(seed=seed + 8, params=params)
    assert p1 != p3


def test_complex_marketplace_config_validation_errors():
    # Negative num_products should raise
    with pytest.raises(ValidationError):
        ComplexMarketplaceConfig(num_products=-1)

    # price_variance out of bounds
    with pytest.raises(ValidationError):
        ComplexMarketplaceConfig(price_variance=1.5)


def test_research_summarization_config_validation_errors():
    with pytest.raises(ValidationError):
        ResearchSummarizationConfig(num_docs=0)

    with pytest.raises(ValidationError):
        ResearchSummarizationConfig(noise_probability=0.75)


def test_multiturn_tool_use_config_validation_errors():
    with pytest.raises(ValidationError):
        MultiTurnToolUseConfig(steps=0)


def test_complex_marketplace_postprocess_rounding_normalization():
    raw = {"revenue": 123.456789, "fulfilled_rate": 0.987654321}
    out = cm_postprocess(raw)
    # Revenue rounded to 2 decimals, fulfilled_rate to 4 decimals as per implementation
    assert out["revenue"] == 123.46
    assert abs(out["fulfilled_rate"] - 0.9877) < 1e-9


@pytest.mark.parametrize(
    "params",
    [
        {
            "num_products": 5,
            "num_orders": 10,
            "max_quantity": 2,
            "price_variance": 0.05,
            "allow_backorder": False,
        },
        {
            "num_products": 8,
            "num_orders": 12,
            "max_quantity": 4,
            "price_variance": 0.12,
            "allow_backorder": True,
        },
    ],
)
def test_complex_marketplace_generate_input_schema(params: dict[str, Any]):
    seed = 99
    payload = cm_generate_input(seed=seed, params=params)
    assert "catalog" in payload and isinstance(payload["catalog"], list)
    assert "orders" in payload and isinstance(payload["orders"], list)
    assert "policies" in payload and isinstance(payload["policies"], dict)
    assert payload["config"]["num_products"] == params["num_products"]
    assert payload["config"]["allow_backorder"] == params["allow_backorder"]


def test_multiturn_tool_use_scenario_init():
    """Test initialization of MultiturnToolUseScenario with various parameters."""
    # Valid params
    params = {
        "steps": 3,
        "include_math": True,
        "include_extraction": False,
        "include_transform": True,
    }
    scenario = MultiturnToolUseScenario(params=params)
    assert scenario.params == params
    assert scenario.config.steps == 3
    assert scenario.config.include_math is True
    assert scenario.config.include_extraction is False
    assert scenario.config.include_transform is True

    # Default params (None)
    scenario_default = MultiturnToolUseScenario()
    assert scenario_default.params == {}
    assert scenario_default.config.steps == 5

    # Minimal valid params
    minimal_params = {"steps": 1}
    scenario_min = MultiturnToolUseScenario(params=minimal_params)
    assert scenario_min.config.steps == 1
    assert scenario_min.config.include_math is True  # default
    assert scenario_min.config.include_extraction is True
    assert scenario_min.config.include_transform is True


def test_multiturn_tool_use_scenario_init_validation_errors():
    """Test that invalid parameters raise ValidationError during init."""
    invalid_params = {"steps": 0}
    with pytest.raises(
        ValueError
    ):  # Since Pydantic ValidationError wrapped in ValueError? Actually, direct ValidationError
        MultiturnToolUseScenario(params=invalid_params)

    # Test that default params work (no raise)
    scenario_default = MultiturnToolUseScenario(params={})
    assert scenario_default.config.steps == 5


@pytest.mark.asyncio
async def test_multiturn_tool_use_scenario_run_success():
    """Test successful run with all capabilities enabled, mocking runner responses."""
    params = {
        "steps": 2,
        "include_math": True,
        "include_extraction": True,
        "include_transform": True,
    }
    scenario = MultiturnToolUseScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    # Mock two successful responses
    mock_runner.process.side_effect = [
        {"success": True, "type": "math", "result": 42},  # First turn
        {"success": True, "type": "extraction", "result": 100},  # Second turn
    ]

    with (
        patch("random.seed"),
        patch("random.choice", side_effect=["math", "extraction"]),
    ):  # Control randomness for determinism
        result = await scenario.run(mock_runner, payload)

    assert "metrics" in result
    assert result["metrics"]["steps_completed"] == 2
    assert result["metrics"]["overall_success_rate"] == 1.0
    assert result["metrics"]["math_success_rate"] == 1.0
    assert result["metrics"]["extraction_success_rate"] == 1.0
    assert (
        result["metrics"]["transform_success_rate"] == 0.0
    )  # Not attempted in this mock
    assert "final_state" in result
    assert len(result["final_state"]["successes"]) == 3  # All caps
    assert "interactions" in result
    assert len(result["interactions"]) == 2
    assert mock_runner.process.call_count == 2


@pytest.mark.asyncio
async def test_multiturn_tool_use_scenario_run_partial_success():
    """Test run with mixed success responses."""
    params = {"steps": 3}
    scenario = MultiturnToolUseScenario(params=params)
    payload = {"seed": 123}

    mock_runner = AsyncMock()
    mock_runner.process.side_effect = [
        {"success": True},  # Success
        {"success": False},  # Failure
        {"success": True},  # Success
    ]

    with (
        patch("random.seed"),
        patch("random.choice", return_value="math"),
    ):  # All math for simplicity
        result = await scenario.run(mock_runner, payload)

    assert result["metrics"]["steps_completed"] == 3
    assert result["metrics"]["overall_success_rate"] == (2 / 3)
    assert result["metrics"]["math_success_rate"] == (2 / 3)
    assert mock_runner.process.call_count == 3


@pytest.mark.parametrize(
    "params, expected_steps, expected_caps",
    [
        (
            {
                "steps": 1,
                "include_math": False,
                "include_extraction": False,
                "include_transform": False,
            },
            1,
            ["basic"],
        ),
        (
            {
                "steps": 4,
                "include_math": True,
                "include_extraction": False,
                "include_transform": False,
            },
            4,
            ["math"],
        ),
        ({"steps": 2}, 2, ["math", "extraction", "transform"]),  # Defaults
    ],
)
@pytest.mark.asyncio
async def test_multiturn_tool_use_scenario_run_parametrized(
    params, expected_steps, expected_caps
):
    """Parametrized test for run with different configurations."""
    scenario = MultiturnToolUseScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    mock_runner.process.return_value = {
        "success": True
    }  # Always success for param test

    with (
        patch("random.seed"),
        patch("random.choice", side_effect=expected_caps[:1] * expected_steps),
    ):  # Cycle caps
        result = await scenario.run(mock_runner, payload)

    assert result["metrics"]["steps_completed"] == expected_steps
    assert result["metrics"]["overall_success_rate"] == 1.0
    if "basic" in expected_caps:
        assert "math_success_rate" not in result["metrics"]  # Only basic
    else:
        for cap in expected_caps:
            assert f"{cap}_success_rate" in result["metrics"]
    assert mock_runner.process.call_count == expected_steps


@pytest.mark.asyncio
async def test_multiturn_tool_use_scenario_run_determinism():
    """Test that same seed produces same interactions structure."""
    params = {"steps": 2}
    scenario = MultiturnToolUseScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    mock_runner.process.return_value = {"success": True}

    with patch("random.seed"), patch("random.choice", return_value="math"):
        result1 = await scenario.run(mock_runner, payload)

    # Reset mock
    mock_runner.process.return_value = {"success": True}

    with patch("random.seed"), patch("random.choice", return_value="math"):
        result2 = await scenario.run(mock_runner, payload)

    # Same structure and metrics
    assert result1["metrics"] == result2["metrics"]
    assert len(result1["interactions"]) == len(result2["interactions"])
    # Inputs should be same due to seed (but since random in input gen, with patch it's controlled)


def test_research_summarization_scenario_init():
    """Test initialization of ResearchSummarizationScenario with various parameters."""
    # Valid params
    params = {
        "num_docs": 3,
        "max_tokens": 150,
        "focus_keywords": ["AI", "ethics"],
        "noise_probability": 0.05,
    }
    scenario = ResearchSummarizationScenario(params=params)
    assert scenario.params == params
    assert scenario.config.num_docs == 3
    assert scenario.config.max_tokens == 150
    assert scenario.config.focus_keywords == ["AI", "ethics"]
    assert scenario.config.noise_probability == 0.05

    # Default params (None)
    scenario_default = ResearchSummarizationScenario()
    assert scenario_default.params == {}
    assert scenario_default.config.num_docs == 5  # Default
    assert scenario_default.config.max_tokens == 200
    assert scenario_default.config.focus_keywords == [
        "research",
        "findings",
        "methodology",
    ]
    assert scenario_default.config.noise_probability == 0.1

    # Minimal valid params (only required)
    minimal_params = {"num_docs": 1, "max_tokens": 50}
    scenario_min = ResearchSummarizationScenario(params=minimal_params)
    assert scenario_min.config.num_docs == 1
    assert scenario_min.config.max_tokens == 50
    assert scenario_min.config.focus_keywords == [
        "research",
        "findings",
        "methodology",
    ]  # Default
    assert scenario_min.config.noise_probability == 0.1  # Default


def test_research_summarization_scenario_init_validation_errors():
    """Test that invalid parameters raise ValidationError during init."""
    invalid_params = {"num_docs": 0}
    with pytest.raises(ValueError):
        ResearchSummarizationScenario(params=invalid_params)

    invalid_params2 = {"noise_probability": 0.6}
    with pytest.raises(ValueError):
        ResearchSummarizationScenario(params=invalid_params2)

    # Test that defaults work for partial params
    partial_params = {"max_tokens": 100}
    scenario_partial = ResearchSummarizationScenario(params=partial_params)
    assert scenario_partial.config.num_docs == 5  # default
    assert scenario_partial.config.max_tokens == 100
    assert scenario_partial.config.noise_probability == 0.1  # default


@pytest.mark.asyncio
async def test_research_summarization_scenario_run_success():
    """Test successful run with default parameters, mocking runner responses."""
    params = {
        "num_docs": 2,
        "max_tokens": 100,
        "focus_keywords": ["AI"],
        "noise_probability": 0.0,
    }
    scenario = ResearchSummarizationScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    # Mock two successful summary responses
    mock_runner.process.side_effect = [
        {"content": "Summary of AI research: Key findings on ethics."},
        {"content": "Summary of second paper: Methodology overview."},
    ]

    result = await scenario.run(mock_runner, payload)

    assert "metrics" in result
    assert result["metrics"]["average_quality_score"] > 0.0
    assert result["metrics"]["keyword_coverage"] > 0.0
    assert result["metrics"]["conciseness_score"] > 0.0
    assert result["metrics"]["total_documents"] == 2
    assert "summaries" in result
    assert len(result["summaries"]) == 2
    assert "documents" in result
    assert len(result["documents"]) == 2
    assert mock_runner.process.call_count == 2


@pytest.mark.asyncio
async def test_research_summarization_scenario_run_determinism():
    """Test that same seed produces consistent metrics structure."""
    params = {"num_docs": 1, "max_tokens": 50}
    scenario = ResearchSummarizationScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    mock_runner.process.return_value = {"content": "Consistent summary."}

    result1 = await scenario.run(mock_runner, payload)
    mock_runner.process.return_value = {"content": "Consistent summary."}  # Reset
    result2 = await scenario.run(mock_runner, payload)

    # Same structure and metrics (since seed controls document generation)
    assert result1["metrics"] == result2["metrics"]
    assert len(result1["summaries"]) == len(result2["summaries"])


@pytest.mark.parametrize(
    "params, expected_docs",
    [
        ({"num_docs": 1, "max_tokens": 100}, 1),
        ({"num_docs": 3, "max_tokens": 200}, 3),
        ({}, 5),  # Defaults
    ],
)
@pytest.mark.asyncio
async def test_research_summarization_scenario_run_parametrized(params, expected_docs):
    """Parametrized test for run with different configurations."""
    scenario = ResearchSummarizationScenario(params=params)
    payload = {"seed": 42}

    mock_runner = AsyncMock()
    mock_runner.process.return_value = {"content": "Summary"}

    result = await scenario.run(mock_runner, payload)

    assert result["metrics"]["total_documents"] == expected_docs
    assert len(result["summaries"]) == expected_docs
    assert mock_runner.process.call_count == expected_docs
