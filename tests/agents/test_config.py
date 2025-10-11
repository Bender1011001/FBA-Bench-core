"""Tests for configuration contracts: BaseAgentConfig and BaseServiceConfig."""

import pytest
from pydantic import ValidationError

from fba_bench_core.agents.base import BaseAgent
from fba_bench_core.config import BaseAgentConfig, BaseServiceConfig
from fba_bench_core.domain.events import BaseEvent, Command
from fba_bench_core.services.base import BaseService


class TestAgent(BaseAgent):
    """Concrete test implementation of BaseAgent."""

    async def decide(self, events: list[BaseEvent]) -> list[Command]:
        return []


class TestService(BaseService):
    """Concrete test implementation of BaseService."""

    def start(self) -> None:
        pass


def test_base_agent_config_happy_path():
    cfg = BaseAgentConfig(
        agent_id="agent_1",
        poll_interval_seconds=30,
        metadata={"env": "test", "retries": 3},
    )
    assert cfg.agent_id == "agent_1"
    assert cfg.poll_interval_seconds == 30
    assert cfg.metadata["env"] == "test"


def test_base_agent_config_invalid_agent_id_and_negative_poll_and_nested_metadata():
    # invalid agent_id containing space
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="invalid id")

    # negative poll_interval rejected
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="agent2", poll_interval_seconds=-1)

    # nested metadata (non-primitive) should be rejected
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="agent3", metadata={"nested": {"a": 1}})


def test_base_agent_config_immutability_and_model_copy():
    cfg = BaseAgentConfig(agent_id="agent_ok", poll_interval_seconds=10)
    # frozen model should raise ValidationError on assignment
    with pytest.raises(ValidationError):
        cfg.agent_id = "new_id"
    # model_copy(update=...) returns a modified copy; original unchanged
    cfg2 = cfg.model_copy(update={"poll_interval_seconds": 60})
    assert cfg2.poll_interval_seconds == 60
    assert cfg.poll_interval_seconds == 10


def test_base_service_config_happy_and_invalid_and_immutability():
    svc = BaseServiceConfig(service_id="svc1", metadata={"enabled": True})
    assert svc.service_id == "svc1"
    with pytest.raises(ValidationError):
        BaseServiceConfig(service_id="bad id")
    with pytest.raises(ValidationError):
        svc.service_id = "svc2"
    svc2 = svc.model_copy(update={"default_region": "us-west-2"})
    assert svc2.default_region == "us-west-2"


def test_subclass_agent_config_extendability():
    # Downstream users should be able to subclass BaseAgentConfig for extra fields
    class PricingAgentConfig(BaseAgentConfig):
        pricing_tier: str = "basic"

    pac = PricingAgentConfig(agent_id="pricing1", pricing_tier="pro")
    assert pac.pricing_tier == "pro"
    assert pac.agent_id == "pricing1"
    # frozen: assignment raises
    with pytest.raises(ValidationError):
        pac.pricing_tier = "basic"
    pac2 = pac.model_copy(update={"pricing_tier": "enterprise"})
    assert pac2.pricing_tier == "enterprise"


def test_metadata_key_and_value_types_enforced():
    # non-string metadata key should be rejected
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="a1", metadata={1: "one"})
    # non-primitive metadata value should be rejected
    with pytest.raises(ValidationError):
        BaseServiceConfig(service_id="s1", metadata={"k": ["list"]})


def test_base_agent_instantiation_with_typed_config():
    """Test that BaseAgent can be instantiated with a typed config."""
    config = BaseAgentConfig(agent_id="test_agent", poll_interval_seconds=10)
    agent = TestAgent(config)
    assert agent.agent_id == "test_agent"
    assert agent.config.poll_interval_seconds == 10


def test_base_service_instantiation_with_typed_config():
    """Test that BaseService can be instantiated with a typed config."""
    config = BaseServiceConfig(service_id="test_service", max_concurrent_tasks=5)
    service = TestService(config)
    assert service.service_id == "test_service"
    assert service.config.max_concurrent_tasks == 5
