"""Base classes for domain events and commands."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseEvent(BaseModel):
    """Base contract for all domain events.

    Attributes:
    - event_type: Literal discriminator provided by subclasses.
    - timestamp: UTC timestamp when the event was recorded (defaults to now).
    - tick: non-negative simulation or system tick.
    - correlation_id: optional id to trace this event across services and workflows.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tick: int = Field(0, ge=0)
    correlation_id: str | None = None


class Command(BaseModel):
    """Base contract for commands (intent to change system state).

    Commands are issued by agents or systems. Include optional metadata to
    enable observability and intent tracing:
    - issued_by: human or system identifier that created the command.
    - reason: free-text explanation for auditing.
    - correlation_id: align with events for traceability.
    - metadata: structured map for small typed values (avoid open-ended blobs).
      We deliberately use extra="forbid" at the model level to prevent accidental
      arbitrary attributes; metadata is the supported extensibility point.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    command_type: str
    issued_by: str | None = None
    reason: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: dict[str, Any]):
        # Keep metadata shallow and keyed by str to encourage typed schemas.
        if not isinstance(v, dict):
            raise TypeError("metadata must be a dict[str, Any]")
        for k in v.keys():
            if not isinstance(k, str):
                raise TypeError("metadata keys must be strings")
        return v
