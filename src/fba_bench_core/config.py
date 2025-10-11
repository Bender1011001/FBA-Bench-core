"""Typed configuration contracts for fba_bench_core.

Phase D:
- Introduces BaseAgentConfig and BaseServiceConfig as Pydantic models.
- Enforces typed metadata, forbids extra fields, and validates identifiers.
- Models are frozen (immutable) to prevent accidental mutation by consumers.

Downstream guidance:
- Subclass BaseAgentConfig / BaseServiceConfig to add domain-specific fields.
- Use model_copy(update={...}) to create modified copies rather than mutating.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Allowed primitive metadata value types. Reject nested dicts/lists to avoid
# arbitrary deep structures hiding Any.
Primitive = str | int | float | bool


def _validate_slug(value: str, field_name: str) -> str:
    """Ensure identifier uses a simple slug format (alphanum, hyphen, underscore)."""
    if not isinstance(value, str) or not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise ValueError(
            f"{field_name!r} must be a slug (letters, digits, hyphen, underscore)."
        )
    return value


class BaseConfigModel(BaseModel):
    """Shared base for configuration models.

    Provides strict Pydantic model settings:
    - extra="forbid": disallow unknown fields (prevents accidental additions).
    - validate_assignment=True: validate when creating copies or assigning (still
      compatible with frozen models).
    - frozen=True: make instances immutable to avoid accidental runtime mutation.
    - allow_population_by_field_name=True: helpful if downstream code prefers
      field-name population.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
    )


class BaseAgentConfig(BaseConfigModel):
    """Base configuration contract for agents.

    Fields:
        agent_id: Unique identifier (slug) for the agent instance.
        poll_interval_seconds: Optional polling interval (seconds) for
            agents that poll external systems. Keep None if not used.
        max_concurrent_tasks: Optional concurrency hint for schedulers.
        default_region: Optional region/locale hint (e.g., "us-west-2").
        metadata: Shallow mapping of simple metadata values (no nested dicts/lists).
                  Keys are strings and values are limited to primitive types.

    Example:
        class PricingAgentConfig(BaseAgentConfig):
            pricing_tier: Literal["basic", "pro"] = "basic"
    """

    agent_id: str
    poll_interval_seconds: int | None = None
    max_concurrent_tasks: int | None = None
    default_region: str | None = None
    metadata: dict[str, Primitive] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def _check_agent_id(cls, v: str) -> str:
        return _validate_slug(v, "agent_id")

    @field_validator("poll_interval_seconds", "max_concurrent_tasks")
    @classmethod
    def _non_negative_ints(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: dict[str, Primitive]) -> dict[str, Primitive]:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a mapping of str -> primitive values")
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValueError("metadata keys must be strings")
            if not isinstance(val, (str, int, float, bool)):
                raise ValueError(
                    "metadata values must be primitive types (str, int, float, bool)"
                )
        return v


class BaseServiceConfig(BaseConfigModel):
    """Base configuration contract for services.

    Fields:
        service_id: Unique identifier (slug) for the service instance.
        poll_interval_seconds, max_concurrent_tasks, default_region, metadata:
            same semantics as in BaseAgentConfig.

    Example:
        class CacheServiceConfig(BaseServiceConfig):
            ttl_seconds: int = 300
    """

    service_id: str
    poll_interval_seconds: int | None = None
    max_concurrent_tasks: int | None = None
    default_region: str | None = None
    metadata: dict[str, Primitive] = Field(default_factory=dict)

    @field_validator("service_id")
    @classmethod
    def _check_service_id(cls, v: str) -> str:
        return _validate_slug(v, "service_id")

    @field_validator("poll_interval_seconds", "max_concurrent_tasks")
    @classmethod
    def _non_negative_ints(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: dict[str, Primitive]) -> dict[str, Primitive]:
        # Reuse same validation semantics as agent metadata.
        if not isinstance(v, dict):
            raise ValueError("metadata must be a mapping of str -> primitive values")
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValueError("metadata keys must be strings")
            if not isinstance(val, (str, int, float, bool)):
                raise ValueError(
                    "metadata values must be primitive types (str, int, float, bool)"
                )
        return v


# End of file. Subclass these models in downstream packages to add domain-specific
# configuration while preserving validation and immutability guarantees.
